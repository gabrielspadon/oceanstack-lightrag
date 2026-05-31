#!/usr/bin/env bash
# Reconcile inputs/ on disk vs lightrag_doc_status in PG.
#
# Modes:
#   report       (default) - count missing/drift/stale, no writes
#   reprocess    - re-ingest missing + drifted, delete stale
#   missing      - re-ingest only the never-seen-by-KB files
#   batch=N      - cap reprocess to first N files (rate-limit safety)
#
# Drift detection: content_sha256 column (preferred). Falls back to file size
# when sha column is NULL (pre-migration rows).

set -Euo pipefail
trap 'echo "reconcile.sh: error at line $LINENO" | tee -a "$LOG" >&2' ERR

# Single-instance lock so concurrent invocations (watcher + manual run) don't
# stack DELETE+POST work on the lightrag pipeline.
LOCK=/var/lock/lightrag-reconcile.lock
if [[ "${RECONCILE_LOCKED:-0}" != "1" ]]; then
    exec env RECONCILE_LOCKED=1 flock -n -E 75 "$LOCK" "$0" "$@"
    # flock returns 75 if another instance holds the lock; treat as no-op
fi

WS=oceanstack_code_schema
INPUTS=/fast-array/lightrag/inputs
LR=${LIGHTRAG_URL:-http://127.0.0.1:9621}
LOG=/fast-array/lightrag/logs/reconcile.log
MODE=${1:-report}
BATCH=${BATCH:-0}   # 0 = no cap
DELETE_WAIT_SECS=${DELETE_WAIT_SECS:-45}

mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

# Strip optional batch=N suffix
if [[ "$MODE" == batch=* ]]; then
    BATCH=${MODE#batch=}; MODE=reprocess
fi

# Decrypt API key
if [[ -z "${LIGHTRAG_API_KEY:-}" ]]; then
    KEY=$(SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}" \
        sops -d --input-type dotenv --output-type dotenv /fast-array/lightrag/.env.enc \
        2>/dev/null | grep '^LIGHTRAG_API_KEY=' | cut -d= -f2)
else
    KEY=$LIGHTRAG_API_KEY
fi
[[ -z "$KEY" ]] && { log "no API key"; exit 1; }

# Server health
if ! curl -fsS --max-time 5 "$LR/health" >/dev/null 2>&1; then
    log "server unreachable at $LR; abort"
    exit 1
fi

cd "$INPUTS"

# --- Build disk side: file_path \t size \t sha256 ---
DISK=$(mktemp); DB=$(mktemp); DIFF=$(mktemp)
trap 'rm -f "$DISK" "$DB" "$DIFF"' EXIT

log "scanning disk under $INPUTS ..."
# Server stores the raw POST body with `text.strip()` applied (after reversing
# the transport-only `& → &amp;` escape via `html.unescape`). The disk file is
# NOT html-encoded — applying `html.unescape` here would misparse legacy HTML5
# entities like `&reg` (no semicolon required) and corrupt Rust references
# such as `&registry`. Hash `raw.strip()` only.
# Pre-filter unindexable files:
#  - 0-byte after strip()  → server returns 422 (empty content)
#  - > MAX_FILE_BYTES      → server skips (request too large)
# Filtering at scan time keeps them out of MISSING/DRIFT and avoids
# permanent reconcile churn on placeholder files.
find code schema -type f \
    \( -name '*.py' -o -name '*.rs' -o -name '*.sql' -o -name '*.wgsl' \) \
    -not -path '*/papers/*' \
    -not -path '*/oceanstack_code_schema/*' \
    -not -path '*/.serena/*' \
    -not -path '*/.claude/*' \
    -not -path '*/scripts/*/logs/*' \
    -print0 \
  | MAX_FILE_BYTES="${MAX_FILE_BYTES:-200000}" xargs -0 python3 -c '
import hashlib, os, pathlib, sys
cap = int(os.environ.get("MAX_FILE_BYTES", "200000"))
for path in sys.argv[1:]:
    try:
        raw = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"SCAN_FAIL\t{path}\t{e}", file=sys.stderr)
        continue
    text = raw.strip()
    if not text:
        continue
    data = text.encode("utf-8")
    if len(data) > cap:
        continue
    sha = hashlib.sha256(data).hexdigest()
    print(f"{path}\t{len(text)}\t{sha}")
' | sort > "$DISK"

# --- Build DB side: file_path \t content_length \t content_sha256 \t id ---
psql --no-psqlrc -d lightrag -tAF $'\t' -c \
    "SELECT file_path, COALESCE(content_length::text,''), COALESCE(content_sha256,''), id
     FROM lightrag_doc_status WHERE workspace='$WS'" \
  | sort > "$DB"

# --- Diff via join: produce rows of file_path, disk_size, disk_sha, db_size, db_sha, db_id ---
join -t$'\t' -a 1 -a 2 -e '-' -o '0,1.2,1.3,2.2,2.3,2.4' "$DISK" "$DB" > "$DIFF"

MISSING=()
DRIFT=()
STALE=()
while IFS=$'\t' read -r fp dsz dsh dbs dbh did; do
    if [[ "$dbs" == "-" ]]; then
        MISSING+=("$fp")
    elif [[ "$dsz" == "-" ]]; then
        STALE+=("$fp|$did")
    elif [[ -n "$dsh" && -n "$dbh" && "$dsh" != "$dbh" ]]; then
        DRIFT+=("$fp|$did")
    elif [[ -z "$dbh" && "$dsz" != "$dbs" ]]; then
        # No hash recorded yet, fallback to size
        DRIFT+=("$fp|$did")
    fi
done < "$DIFF"

log "disk=$(wc -l <"$DISK") db=$(wc -l <"$DB") missing=${#MISSING[@]} drift=${#DRIFT[@]} stale=${#STALE[@]} mode=$MODE batch=$BATCH"

if [[ "$MODE" == "report" ]]; then
    if (( ${#MISSING[@]} > 0 )); then
        log "MISSING (showing up to 10):"
        printf '  %s\n' "${MISSING[@]:0:10}" | tee -a "$LOG"
    fi
    if (( ${#DRIFT[@]} > 0 )); then
        log "DRIFT (showing up to 10):"
        printf '  %s\n' "${DRIFT[@]:0:10}" | sed 's/|.*//' | tee -a "$LOG"
    fi
    if (( ${#STALE[@]} > 0 )); then
        log "STALE (showing up to 10):"
        printf '  %s\n' "${STALE[@]:0:10}" | sed 's/|.*//' | tee -a "$LOG"
    fi
    exit 0
fi

# --- Action helpers ---
# SQL-level purge: bypasses the lightrag async DELETE queue, which under
# parallel load hits a serialized graph-rebuild bottleneck causing 60%+ of
# DELETE attempts to time out. Removes doc_status + doc_full + chunks +
# vdb_chunks + full_entities + full_relations rows for this doc_id, in a
# single transaction. Entity/relation chunk_ids referencing the purged
# chunks become stale; cleanup query in retry-watcher's sweep handles them.
purge_doc_sql() {
    local id="$1"
    [[ "$id" =~ ^doc-[a-f0-9]+$ ]] || return 0
    # W1-A3 fix: psql -v binding + dynamic vdb_chunks table discovery
    # (was hard-coded to `..._gemini_embedding_2_1536d` which became a silent
    # no-op after the Ollama migration).
    psql --no-psqlrc -d lightrag -v ON_ERROR_STOP=1 \
        -v ws="$WS" -v did="$id" <<'SQL' >/dev/null 2>&1
SET lock_timeout = '5s';
SELECT set_config('os.ws', :'ws', false);
SELECT set_config('os.did', :'did', false);
BEGIN;
DO $$
DECLARE
  vt text;
  ws_v text := current_setting('os.ws');
  did_v text := current_setting('os.did');
BEGIN
  FOR vt IN SELECT tablename FROM pg_tables
            WHERE tablename LIKE 'lightrag_vdb_chunks_%'
              AND tablename NOT LIKE '%_legacy'
  LOOP
    EXECUTE format(
      'DELETE FROM %I WHERE workspace=$1 AND full_doc_id=$2',
      vt
    ) USING ws_v, did_v;
  END LOOP;
END
$$;
DELETE FROM lightrag_doc_chunks  WHERE workspace=:'ws' AND full_doc_id=:'did';
DELETE FROM lightrag_full_entities  WHERE workspace=:'ws' AND id=:'did';
DELETE FROM lightrag_full_relations WHERE workspace=:'ws' AND id=:'did';
DELETE FROM lightrag_doc_full       WHERE workspace=:'ws' AND id=:'did';
DELETE FROM lightrag_doc_status     WHERE workspace=:'ws' AND id=:'did';
COMMIT;
SQL
}

# Legacy API-based DELETE (kept for stale-removal path where async is fine)
delete_doc_api() {
    local id="$1"
    [[ "$id" =~ ^doc-[a-f0-9]+$ ]] || return 0
    curl -fsS --max-time 10 -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
        -X DELETE "$LR/documents/delete_document" \
        -d "{\"doc_ids\":[\"$id\"],\"delete_file\":false,\"delete_llm_cache\":false}" \
        >/dev/null 2>&1 || true
    local i still
    for i in $(seq 1 "$((DELETE_WAIT_SECS/2))"); do
        sleep 2
        # W1-A3 fix: psql -v binding (doc_id is bot-shaped + already validated,
        # but switch is defense in depth).
        still=$(psql --no-psqlrc -d lightrag -v did="$id" -tAc \
            "SELECT count(*) FROM lightrag_doc_status WHERE id=:'did'" 2>/dev/null | tr -d ' ')
        [[ "$still" == "0" ]] && return 0
    done
    return 1
}

# Default delete = SQL purge (instant, reliable). Override with USE_API_DELETE=1.
delete_doc() {
    if [[ "${USE_API_DELETE:-0}" == "1" ]]; then
        delete_doc_api "$@"
    else
        purge_doc_sql "$@"
    fi
}

post_file() {
    local fp="$1"
    local abs="$INPUTS/$fp"
    # W1-A3 fix: enforce MAX_FILE_BYTES cap (was sync-from-git only).
    local size
    size=$(stat -c%s "$abs" 2>/dev/null || echo 0)
    if (( size > ${MAX_FILE_BYTES:-200000} )); then
        log "skip too_big ($size > ${MAX_FILE_BYTES:-200000}): $fp"
        echo 000
        return
    fi
    # W1-A3 fix: pass $abs / $fp via argv to avoid Python injection.
    # Server HTML-decodes ingested text (& timestamps -> ×tamps). Pre-encode
    # every & as &amp; so the round-trip preserves the original content.
    HTTP=$(curl -fsS --max-time 30 -o /dev/null -w '%{http_code}' \
        -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
        -X POST "$LR/documents/text" \
        --data @<(python3 - "$abs" "$fp" <<'PY'
import json, pathlib, sys
abs_path, fp = sys.argv[1], sys.argv[2]
text = pathlib.Path(abs_path).read_text(encoding='utf-8', errors='replace').replace('&', '&amp;')
print(json.dumps({'text': text, 'file_source': fp}))
PY
) 2>/dev/null)
    echo "$HTTP"
}

# --- Stale removal ---
removed=0
for entry in "${STALE[@]:-}"; do
    [[ -z "$entry" ]] && continue
    fp=${entry%%|*}; did=${entry##*|}
    if delete_doc "$did"; then
        removed=$((removed+1))
        log "stale-removed $fp ($did)"
    else
        log "stale-delete-timeout $fp ($did)"
    fi
done

# --- Build worklist: missing first (cheap POST), then drift (delete+POST) ---
WORK=()
for fp in "${MISSING[@]:-}"; do [[ -n "$fp" ]] && WORK+=("M|$fp|-"); done
if [[ "$MODE" != "missing" ]]; then
    for entry in "${DRIFT[@]:-}"; do
        [[ -z "$entry" ]] && continue
        fp=${entry%%|*}; did=${entry##*|}
        WORK+=("D|$fp|$did")
    done
fi

total=${#WORK[@]}
if (( BATCH > 0 && BATCH < total )); then
    log "capping work: $BATCH of $total"
    WORK=("${WORK[@]:0:BATCH}")
    total=$BATCH
fi

log "starting reprocess: total=$total parallel=${RECONCILE_PARALLEL:-6}"

if (( total == 0 )); then
    log "reconcile done. stale_removed=$removed posted=0 failed=0 skipped=0"
    exit 0
fi

# Process one work item (kind|fp|did). Emits TAB-separated status line:
#   OK <file_path>       — POST succeeded
#   SKIP <file_path>     — delete didn't commit in window (server still working)
#   FAIL <code> <fp>     — POST returned non-200
process_one() {
    local kind="$1" fp="$2" did="$3"
    if [[ "$kind" == "D" && "$did" == doc-* ]]; then
        if ! delete_doc "$did"; then
            echo "SKIP $fp"
            return
        fi
    fi
    local code
    code=$(post_file "$fp")
    if [[ "$code" == "200" ]]; then
        echo "OK $fp"
    else
        echo "FAIL $code $fp"
    fi
}

# Run worklist with bounded parallelism. xargs -P gives free fan-out; export
# helpers + key so child shells can call them.
export -f process_one delete_doc delete_doc_api purge_doc_sql post_file log
export LR KEY INPUTS LOG DELETE_WAIT_SECS WS USE_API_DELETE

PARALLEL=${RECONCILE_PARALLEL:-6}
posted=0; failed=0; skipped=0; done_ct=0

# Pipe worklist as NUL-delimited tuples into xargs.
results=$(
    for w in "${WORK[@]}"; do printf '%s\0' "$w"; done \
      | xargs -0 -n1 -P "$PARALLEL" bash -c '
            IFS="|" read -r kind fp did <<<"$1"
            process_one "$kind" "$fp" "$did"
        ' _
)

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    done_ct=$((done_ct+1))
    case "$line" in
        OK\ *)    posted=$((posted+1)) ;;
        SKIP\ *)  skipped=$((skipped+1))
                  log "delete-timeout, skipping POST: ${line#SKIP }" ;;
        FAIL\ *)  failed=$((failed+1))
                  log "POST FAIL ${line#FAIL }" ;;
    esac
    if (( done_ct % 25 == 0 )); then
        log "progress $done_ct/$total posted=$posted failed=$failed skipped=$skipped"
    fi
done <<<"$results"

log "reconcile done. stale_removed=$removed posted=$posted failed=$failed skipped=$skipped"

# Post-reconcile quality gates: structural health (density, orphan share, largest
# component, entity-type validity) plus architecture smoke queries against the live
# server. A regression here is a hard error so the auditor surfaces it.
log "running knowledge-graph quality gates ..."
if LR_KEY="$KEY" /fast-array/lightrag/.venv/bin/python /fast-array/lightrag/kg-quality-gates.py >>"$LOG" 2>&1; then
    log "quality gates PASS"
else
    log "quality gates FAIL — structural regression or architecture-query miss (see $LOG)"
    exit 1
fi
