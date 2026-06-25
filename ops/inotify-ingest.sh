#!/usr/bin/env bash
# Sub-second LightRAG ingest. Watches /fast-array/lightrag/inputs recursively
# and for every close_write / moved_to / delete event on .py/.rs/.sql files,
# does SQL-purge + immediate POST of the new content. Bypasses lightrag's
# slow async DELETE queue.
#
# Coalescing: file events arriving within COALESCE_SECS of the same path are
# debounced — only the last write triggers ingest, so a save-burst (editor
# autoformat + save) results in one ingest per path.
#
# Throttling: at most MAX_INFLIGHT concurrent ingests so we don't saturate
# the lightrag pipeline (server has max_async=12, max_parallel_insert=4).
#
# retry-watcher.sh continues to run as a 2-minute safety net for events
# we miss (eg. while this service is down).

set -uo pipefail

WS=oceanstack_code_schema
INPUTS=/fast-array/lightrag/inputs
LR=${LIGHTRAG_URL:-http://127.0.0.1:9621}
LOG=/fast-array/lightrag/logs/inotify-ingest.log
COALESCE_SECS=${COALESCE_SECS:-2}
MAX_INFLIGHT=${MAX_INFLIGHT:-4}
PEND_DIR=/dev/shm/lightrag-inotify-pending

mkdir -p "$(dirname "$LOG")" "$PEND_DIR"
log() { echo "$(date -Iseconds) $*" >>"$LOG"; }

KEY=$(SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}" \
    sops -d --input-type dotenv --output-type dotenv /fast-array/lightrag/.env.enc \
    2>/dev/null | grep '^LIGHTRAG_API_KEY=' | cut -d= -f2)
[[ -z "$KEY" ]] && { log "FATAL: no API key"; exit 1; }

# In-process backoff: wait for server up to ~10 minutes with exponential
# backoff capped at 30 s. Stops the systemd restart storm that would otherwise
# burn 6 cycles per minute (Restart=on-failure RestartSec=10) while PG is
# performing post-crash recovery — the LightRAG server cannot accept HTTP
# until PG reaches consistent recovery state, and the inotify watcher's
# previous one-shot health check exited with status 1 inside that window.
backoff=2
attempts=0
max_attempts=60
while ! curl -fsS --max-time 5 "$LR/health" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if (( attempts >= max_attempts )); then
        log "FATAL: server unreachable at $LR after ${attempts} attempts"
        exit 1
    fi
    if (( attempts % 5 == 1 )); then
        log "waiting for $LR/health (attempt ${attempts}/${max_attempts}, sleep ${backoff}s)"
    fi
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    (( backoff > 30 )) && backoff=30
done
log "server reachable at $LR after ${attempts} retr$([[ $attempts == 1 ]] && echo y || echo ies)"

# --- helpers ---

# Map fp to a collision-free marker filename in PEND_DIR. Hash the path rather
# than tr'ing / and . to _, which collided distinct paths (dir/file.py vs
# dir_file.py); the marker's *contents* carry the real path, the name only
# needs to be unique.
pend_file() {
    local fp="$1"
    echo "$PEND_DIR/$(printf '%s' "$fp" | sha1sum | cut -d' ' -f1)"
}

purge_doc_sql() {
    local fp="$1"
    # R5-A4 fix: psql -v binding; DO block resolves live vdb_chunks table.
    psql --no-psqlrc -d lightrag -v ON_ERROR_STOP=1 \
        -v ws="$WS" -v fp="$fp" <<'SQL' >/dev/null 2>&1
SET lock_timeout='5s';
SELECT set_config('os.ws', :'ws', false);
SELECT set_config('os.fp', :'fp', false);
BEGIN;
DO $$
DECLARE
  vt text;
  ws_v text := current_setting('os.ws');
  fp_v text := current_setting('os.fp');
BEGIN
  FOR vt IN SELECT tablename FROM pg_tables
            WHERE tablename LIKE 'lightrag_vdb_chunks_%'
              AND tablename NOT LIKE '%_legacy'
  LOOP
    EXECUTE format(
      'DELETE FROM %I WHERE workspace = $1 AND full_doc_id IN ('
      || ' SELECT id FROM lightrag_doc_status WHERE workspace = $1 AND file_path = $2)',
      vt
    ) USING ws_v, fp_v;
  END LOOP;
END
$$;
DELETE FROM lightrag_doc_chunks
WHERE workspace = :'ws' AND full_doc_id IN (
  SELECT id FROM lightrag_doc_status WHERE workspace = :'ws' AND file_path = :'fp');
DELETE FROM lightrag_full_entities
WHERE workspace = :'ws' AND id IN (
  SELECT id FROM lightrag_doc_status WHERE workspace = :'ws' AND file_path = :'fp');
DELETE FROM lightrag_full_relations
WHERE workspace = :'ws' AND id IN (
  SELECT id FROM lightrag_doc_status WHERE workspace = :'ws' AND file_path = :'fp');
DELETE FROM lightrag_doc_full
WHERE workspace = :'ws' AND id IN (
  SELECT id FROM lightrag_doc_status WHERE workspace = :'ws' AND file_path = :'fp');
DELETE FROM lightrag_doc_status
WHERE workspace = :'ws' AND file_path = :'fp';
COMMIT;
SQL
}

post_file() {
    local fp="$1"
    local abs="$INPUTS/$fp"
    [[ -f "$abs" ]] || return 1
    # R5-A4 fix: pass $abs / $fp via argv so quotes/specials in paths can't
    # inject Python. R5-A4 I4 fix: enforce MAX_FILE_BYTES cap.
    local size
    size=$(stat -c%s "$abs" 2>/dev/null || echo 0)
    if (( size > ${MAX_FILE_BYTES:-200000} )); then
        log "skip too_big ($size > ${MAX_FILE_BYTES:-200000}): $fp"
        return 1
    fi
    curl -fsS --max-time 30 -o /dev/null -w '%{http_code}' \
        -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
        -X POST "$LR/documents/text" \
        --data @<(python3 - "$abs" "$fp" <<'PY'
import json, pathlib, sys
abs_path, fp = sys.argv[1], sys.argv[2]
text = pathlib.Path(abs_path).read_text(encoding='utf-8', errors='replace').replace('&', '&amp;')
print(json.dumps({'text': text, 'file_source': fp}))
PY
) 2>/dev/null
}

handle_change() {
    local fp="$1"
    local abs="$INPUTS/$fp"

    if [[ ! -f "$abs" ]]; then
        # File deleted: purge from KB
        if psql --no-psqlrc -d lightrag -v ws="$WS" -v fp="$fp" -tAc \
            "SELECT 1 FROM lightrag_doc_status WHERE workspace = :'ws' AND file_path = :'fp' LIMIT 1" \
            2>/dev/null | grep -q 1; then
            log "DELETE: $fp"
            purge_doc_sql "$fp"
        fi
        return
    fi

    # SHA check: skip if no content change vs DB
    local disk_sha db_sha
    # Canonical SHA = sha256(stripped raw bytes). Server stores text.strip()
    # (both ends; document_routes.py) and the trigger hashes that, matching
    # reconcile.sh — so strip both ends here too, or leading-whitespace files
    # never match and re-ingest on every save. & encoding is transport-only.
    # R5-A4 fix: pass $abs via argv so a path with `'` cannot inject Python.
    disk_sha=$(python3 - "$abs" <<'PY' 2>/dev/null
import hashlib, pathlib, sys
try:
    t = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace').strip()
    print(hashlib.sha256(t.encode()).hexdigest())
except Exception:
    sys.exit(1)
PY
)
    db_sha=$(psql --no-psqlrc -d lightrag -v ws="$WS" -v fp="$fp" -tAc \
        "SELECT COALESCE(content_sha256,'') FROM lightrag_doc_status
         WHERE workspace = :'ws' AND file_path = :'fp' LIMIT 1" 2>/dev/null | tr -d ' ')
    if [[ -n "$disk_sha" && -n "$db_sha" && "$disk_sha" == "$db_sha" ]]; then
        log "skip nochange: $fp"
        return
    fi

    log "INGEST: $fp"
    purge_doc_sql "$fp"
    local code
    code=$(post_file "$fp")
    if [[ "$code" == "200" ]]; then
        log "POST ok: $fp"
    else
        log "POST FAIL [$code]: $fp"
    fi
}

# Wait until under inflight cap (counts background jobs in this shell)
wait_inflight() {
    while (( $(jobs -rp | wc -l) >= MAX_INFLIGHT )); do
        sleep 0.5
        wait -n 2>/dev/null
    done
}

# Process one pending file path: if its pend_file mtime is older than
# coalesce window, dispatch handle_change and remove the marker.
maybe_dispatch() {
    local pf="$1"
    local fp_marker
    [[ -f "$pf" ]] || return
    fp_marker=$(<"$pf")
    local mt now
    mt=$(stat -c%Y "$pf" 2>/dev/null || echo 0)
    now=$(date +%s)
    if (( now - mt >= COALESCE_SECS )); then
        rm -f "$pf"
        wait_inflight
        handle_change "$fp_marker" &
    fi
}

# Sweep all pending markers periodically
sweep_loop() {
    while true; do
        for pf in "$PEND_DIR"/*; do
            [[ -e "$pf" ]] || continue
            maybe_dispatch "$pf"
        done
        sleep 1
    done
}

log "inotify-ingest start (watch=$INPUTS, coalesce=${COALESCE_SECS}s, max_inflight=$MAX_INFLIGHT)"

# Background sweeper for delayed dispatch
sweep_loop &
SWEEP_PID=$!
trap "kill $SWEEP_PID 2>/dev/null; rm -rf $PEND_DIR; exit" INT TERM

# Reader: each inotify event refreshes the pend marker's mtime, postponing
# dispatch by another COALESCE_SECS. Bursts collapse into one ingest.
inotifywait -mqr -e close_write -e moved_to -e moved_from -e delete \
    --format '%w%f|%e' "$INPUTS" 2>>"$LOG" \
  | while IFS='|' read -r fpath events; do
        case "$fpath" in
            *.py|*.rs|*.sql|*.wgsl) ;;
            *) continue ;;
        esac
        rel=${fpath#$INPUTS/}
        # Exclude workspace storage + papers + auto-generated docs + tooling.
        # W2-A1 + W2-A4: .serena/* and .claude/* are project tooling, not
        # OceanStack source — exclude entirely.
        case "$rel" in
            oceanstack_code_schema/*|papers/*) continue ;;
            tests/*|*/tests/*) continue ;;
            */docs/api/*) continue ;;
            */CHANGELOG.md) continue ;;
            *.lock|*/uv.lock|*/Cargo.lock) continue ;;
            *.log|*/coverage.xml|*/htmlcov/*|*/site/*) continue ;;
            scripts/*/logs/*|*/scripts/*/logs/*) continue ;;
            .serena/*|*/.serena/*) continue ;;
            .claude/*|*/.claude/*) continue ;;
            # Legal / governance / community-health (user feedback: not code)
            LICENSE*|*/LICENSE*) continue ;;
            COPYRIGHT*|*/COPYRIGHT*) continue ;;
            NOTICE*|*/NOTICE*) continue ;;
            CONTRIBUTING.md|*/CONTRIBUTING.md) continue ;;
            CODE_OF_CONDUCT.md|*/CODE_OF_CONDUCT.md) continue ;;
            SECURITY.md|*/SECURITY.md) continue ;;
            AUTHORS|*/AUTHORS|MAINTAINERS|*/MAINTAINERS) continue ;;
            CITATION.cff|*/CITATION.cff) continue ;;
            *.github/ISSUE_TEMPLATE/*) continue ;;
            *.github/PULL_REQUEST_TEMPLATE*|*.github/PR_TEMPLATE*) continue ;;
            *.github/FUNDING.yml) continue ;;
            # Linter / audit configs: pattern allowlists, not OceanStack code
            _typos.toml|*/_typos.toml) continue ;;
            .gitleaks.toml|*/.gitleaks.toml) continue ;;
            .pip-audit.toml|*/.pip-audit.toml) continue ;;
            .cargo/audit.toml|*/.cargo/audit.toml) continue ;;
            .audit-baseline.json|*/.audit-baseline.json) continue ;;
            deny.toml|*/deny.toml) continue ;;
            # Test-fixture inventories (list of files, not actual content)
            tests/data/README.md|*/tests/data/README.md) continue ;;
            tests/quarantine/README.md|*/tests/quarantine/README.md) continue ;;
            tests/fixtures/README.md|*/tests/fixtures/README.md) continue ;;
            # Dependency manifests (LLM extracts every listed lib as entity)
            pyproject.toml|*/pyproject.toml) continue ;;
            Cargo.toml|*/Cargo.toml) continue ;;
            requirements.txt|*/requirements.txt) continue ;;
            requirements-*.txt|*/requirements-*.txt) continue ;;
            Pipfile|*/Pipfile|Pipfile.lock|*/Pipfile.lock) continue ;;
            licenses.md|*/licenses.md) continue ;;
        esac
        pf=$(pend_file "$rel")
        echo "$rel" > "$pf"
        log "event: $rel ($events) coalesce=${COALESCE_SECS}s"
    done
