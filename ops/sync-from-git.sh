#!/usr/bin/env bash
# Sync changed files from a git ref-range to LightRAG inputs + ingest.
# Usage: sync-from-git.sh <repo-path> <old-sha> <new-sha>
# Called by git pre-push hook in OceanStack.
#
# Drift-proof behaviour (vs. original buggy version):
#  - Handles Added | Copied | Modified | Renamed | Deleted (was: ADM only).
#  - Deletions of source files now propagate: doc removed via SQL purge and
#    inputs/code/ stub deleted.
#  - Renames handled as delete-old + add-new (was: silent breakage — old
#    inputs path retained).
#  - Removed the 200-byte size MINIMUM that excluded most __init__.py /
#    short stubs from the KG. Cap kept at 200 KB.
#  - Cost guard: pre-POST SHA check. If the disk file SHA matches the
#    stored content_sha256 for the existing doc_id, SKIP — no LLM calls
#    issued. Saves Gemini extraction + embedding fees on no-op pushes.
#  - SQL purge bypass replaces the slow lightrag DELETE API (saw 60%
#    timeout rate under parallel load). Atomic, instant, server picks up
#    the next POST cleanly.

set -euo pipefail

REPO="${1:-$HOME/Codebases/OceanStack}"
OLD="${2:-HEAD~1}"
NEW="${3:-HEAD}"

LIGHTRAG_DIR=/fast-array/lightrag
INPUTS_CODE="$LIGHTRAG_DIR/inputs/code"
LIGHTRAG_URL="${LIGHTRAG_URL:-http://127.0.0.1:9621}"
LOG="$LIGHTRAG_DIR/logs/git-sync.log"
WS=oceanstack_code_schema
MAX_FILE_BYTES=${MAX_FILE_BYTES:-200000}

mkdir -p "$INPUTS_CODE" "$(dirname "$LOG")"

log() { echo "$(date -Iseconds) $*" | tee -a "$LOG" >&2; }

if ! curl -fsS --max-time 3 "$LIGHTRAG_URL/health" >/dev/null 2>&1; then
    log "lightrag-server not reachable at $LIGHTRAG_URL — skipping sync"
    exit 0
fi

KEY=$(SOPS_AGE_KEY_FILE="$HOME/.config/sops/age/keys.txt" \
    sops -d --input-type dotenv --output-type dotenv "$LIGHTRAG_DIR/.env.enc" \
    | grep '^LIGHTRAG_API_KEY=' | cut -d= -f2)

# Resolve sentinel SHA (new branch case)
ZERO_SHA="0000000000000000000000000000000000000000"
if [[ "$NEW" == "$ZERO_SHA" ]]; then
    log "NEW=zero (branch deletion) — skip"
    exit 0
fi

ext_match() {
    # KG corpus is code + GPU shaders (.py, .rs, .sql, .wgsl). Architecture
    # docs (.md, etc.), config (.toml), and lockfiles are rejected here — they
    # drift, list third-party packages, and dilute the entity graph.
    [[ "$1" =~ \.(py|rs|sql|wgsl)$ ]]
}
path_excluded() {
    case "$1" in
        target/*|*/target/*) return 0 ;;
        .venv/*|*/.venv/*) return 0 ;;
        tests/*|*/tests/*) return 0 ;;
        node_modules/*|*/node_modules/*) return 0 ;;
        __pycache__/*|*/__pycache__/*) return 0 ;;
        build/*|*/build/*) return 0 ;;
        dist/*|*/dist/*) return 0 ;;
        .pytest_cache/*|.mypy_cache/*|.ruff_cache/*) return 0 ;;
        external/lightrag/*) return 0 ;;
        CHANGELOG.md|*/CHANGELOG.md) return 0 ;;
        scripts/*/logs/*) return 0 ;;
        uv.lock|*/uv.lock|Cargo.lock|*/Cargo.lock) return 0 ;;
        site/*|*/site/*|htmlcov/*|*/htmlcov/*) return 0 ;;
        coverage.xml|*/coverage.xml|*.log|errors.log) return 0 ;;
        papers/*|*/papers/*) return 0 ;;
        oceanstack_code_schema/*|*/oceanstack_code_schema/*) return 0 ;;
        */docs/api/*) return 0 ;;
        # Agent memory and Claude harness directories — not OceanStack source.
        .serena/*|*/.serena/*) return 0 ;;
        .claude/*|*/.claude/*) return 0 ;;
        # Legal / governance / citation files. Defense-in-depth — extensions
        # already reject most of these, but the explicit list documents intent.
        LICENSE*|*/LICENSE*) return 0 ;;
        COPYRIGHT*|*/COPYRIGHT*) return 0 ;;
        NOTICE*|*/NOTICE*) return 0 ;;
        CONTRIBUTING.md|*/CONTRIBUTING.md) return 0 ;;
        CODE_OF_CONDUCT.md|*/CODE_OF_CONDUCT.md) return 0 ;;
        SECURITY.md|*/SECURITY.md) return 0 ;;
        AUTHORS|*/AUTHORS|MAINTAINERS|*/MAINTAINERS) return 0 ;;
        CITATION.cff|*/CITATION.cff) return 0 ;;
        # GitHub templates / community health files — governance, not code
        *.github/ISSUE_TEMPLATE/*) return 0 ;;
        *.github/PULL_REQUEST_TEMPLATE*) return 0 ;;
        *.github/PR_TEMPLATE*) return 0 ;;
        *.github/FUNDING.yml) return 0 ;;
        # Linter / audit configs containing pattern allowlists (typos, secrets,
        # vulnerabilities). The LLM extracts the typo fragments as entities.
        _typos.toml|*/_typos.toml) return 0 ;;
        .gitleaks.toml|*/.gitleaks.toml) return 0 ;;
        .pip-audit.toml|*/.pip-audit.toml) return 0 ;;
        .cargo/audit.toml|*/.cargo/audit.toml) return 0 ;;
        .audit-baseline.json|*/.audit-baseline.json) return 0 ;;
        deny.toml|*/deny.toml) return 0 ;;
        # Test-fixture inventories: README files that list every test data file.
        # The LLM emits each filename (sample.nm4, dateline.csv, parquet/) as
        # an entity. Skip these inventory files; real tests live elsewhere.
        tests/data/README.md|*/tests/data/README.md) return 0 ;;
        tests/quarantine/README.md|*/tests/quarantine/README.md) return 0 ;;
        tests/fixtures/README.md|*/tests/fixtures/README.md) return 0 ;;
        # Dependency manifests: pyproject.toml and Cargo.toml list 200+ third-
        # party packages. The LLM extracts every package as a LIBRARY entity
        # (uvicorn, vulture, py7zr, etc.) — pure noise that overwhelms the KG.
        pyproject.toml|*/pyproject.toml) return 0 ;;
        Cargo.toml|*/Cargo.toml) return 0 ;;
        requirements.txt|*/requirements.txt) return 0 ;;
        requirements-*.txt|*/requirements-*.txt) return 0 ;;
        Pipfile|*/Pipfile|Pipfile.lock|*/Pipfile.lock) return 0 ;;
        # licenses.md — inventory of package licenses (extends LICENSE.md)
        licenses.md|*/licenses.md) return 0 ;;
    esac
    return 1
}

# Read git's name-status output: <STATUS><TAB><path>[<TAB><new_path>]
# Status codes: A added, C copied, M modified, R renamed, D deleted, T type-change
# For renames git emits R<score>\t<old_path>\t<new_path>
ADDED=(); MODIFIED=(); DELETED=()
while IFS=$'\t' read -r status old_p new_p; do
    [[ -z "$status" ]] && continue
    case "${status:0:1}" in
        A|C) ext_match "$old_p" && ! path_excluded "$old_p" && ADDED+=("$old_p") ;;
        M|T) ext_match "$old_p" && ! path_excluded "$old_p" && MODIFIED+=("$old_p") ;;
        D)
            ext_match "$old_p" && ! path_excluded "$old_p" && DELETED+=("$old_p") ;;
        R)
            # Rename: delete old path, add new path
            ext_match "$old_p" && ! path_excluded "$old_p" && DELETED+=("$old_p")
            ext_match "$new_p" && ! path_excluded "$new_p" && ADDED+=("$new_p")
            ;;
    esac
done < <(
    cd "$REPO"
    git diff --name-status --diff-filter=ACMRD "$OLD".."$NEW" 2>/dev/null || true
)

total=$((${#ADDED[@]} + ${#MODIFIED[@]} + ${#DELETED[@]}))
if (( total == 0 )); then
    log "no relevant file changes in $OLD..$NEW"
    exit 0
fi

log "syncing $OLD..$NEW: added=${#ADDED[@]} modified=${#MODIFIED[@]} deleted=${#DELETED[@]}"

# SQL-level purge: bypasses the slow API DELETE queue. Removes doc rows +
# chunks + vdb_chunks atomically; entity/relation chunk_id refs cleaned up
# by retry-watcher's hourly vacuum_graph.
purge_doc_for_path() {
    local fp="$1"
    # R5-A4: psql -v binding so a path containing `'` (e.g. `notes/it's.md`)
    # cannot break out of the string literal. The live vdb_chunks table is
    # resolved dynamically via a DO block — original code hard-coded
    # `..._gemini_embedding_2_1536d` which became stale after the Ollama
    # migration. Inside the DO block we read the `os.fp` / `os.ws` session
    # settings via current_setting() because psql `:'var'` substitution does
    # not reach into server-side plpgsql. Caller is expected to redirect stderr
    # to the log if it wants to surface lock_timeout / constraint failures.
    psql --no-psqlrc -d lightrag -v ON_ERROR_STOP=1 \
        -v ws="$WS" -v fp="$fp" <<'SQL' >/dev/null
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

# Cost guard: skip POST if disk SHA matches stored DB SHA for the existing
# doc. Cheapest possible no-op — no LLM, no embedding, no chunking.
content_changed() {
    local fp="$1"      # path relative to repo root
    local src_path="code/$fp"
    local abs="$REPO/$fp"
    [[ -f "$abs" ]] || return 0   # missing file = treat as changed (we'll handle later)
    local disk_sha
    # Server stores `text.strip()` of the POST body after reversing the
    # `& → &amp;` transport escape. The disk file is NOT html-encoded —
    # applying `html.unescape` would misparse legacy HTML5 entities
    # (e.g. `&reg` without `;`) and corrupt Rust references like `&registry`.
    # Hash `raw.strip()` only to match what the DB trigger computes.
    # R5-A4: pass abs via argv so a path with `'` can't inject Python.
    disk_sha=$(python3 - "$abs" <<'PY' 2>/dev/null
import hashlib, pathlib, sys
try:
    raw = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
    t = raw.strip()
    print(hashlib.sha256(t.encode()).hexdigest())
except Exception:
    sys.exit(1)
PY
)
    [[ -z "$disk_sha" ]] && return 0
    local db_sha
    # R5-A4: psql -v binding for ws/src_path.
    db_sha=$(psql --no-psqlrc -d lightrag -v ws="$WS" -v sp="$src_path" -tAc \
        "SELECT COALESCE(content_sha256,'') FROM lightrag_doc_status
         WHERE workspace = :'ws' AND file_path = :'sp' LIMIT 1" 2>/dev/null | tr -d ' ')
    [[ "$disk_sha" == "$db_sha" ]] && return 1  # unchanged
    return 0
}

# Pre-POST sanity: skip files the server would reject anyway.
#  - 0-byte after strip() → HTTP 422
#  - > MAX_FILE_BYTES     → request too large
ingestable() {
    local abs="$1"
    [[ -f "$abs" ]] || return 1
    local size
    size=$(stat -c%s "$abs" 2>/dev/null || echo 0)
    (( size > MAX_FILE_BYTES )) && return 1
    python3 - "$abs" <<'PY' 2>/dev/null
import pathlib, sys
raw = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
sys.exit(0 if raw.strip() else 1)
PY
}

post_file() {
    local fp="$1" src_path="code/$1"
    # R5-A4 fix: pass abs / src_path via argv to avoid Python injection.
    curl -fsS -o /dev/null -w '%{http_code}' \
        -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
        -X POST "$LIGHTRAG_URL/documents/text" \
        --data @<(python3 - "$REPO/$fp" "$src_path" <<'PY'
import json, pathlib, sys
abs_path, src_path = sys.argv[1], sys.argv[2]
text = pathlib.Path(abs_path).read_text(encoding='utf-8', errors='replace').replace('&', '&amp;')
print(json.dumps({'text': text, 'file_source': src_path}))
PY
) 2>>"$LOG"
}

# --- Deletions first (cheapest, frees server queue for adds) ---
# Per-iteration failures must not abort the batch: surface them via DEL FAIL,
# log, and move on. Errors from `purge_doc_for_path` (lock timeout, etc.) are
# captured in $LOG instead of being silenced to /dev/null.
del_ok=0; del_fail=0
for f in "${DELETED[@]:-}"; do
    [[ -z "$f" ]] && continue
    src_path="code/$f"
    if purge_doc_for_path "$src_path" 2>>"$LOG"; then
        rm -f "$INPUTS_CODE/$f"
        rmdir -p "$(dirname "$INPUTS_CODE/$f")" 2>/dev/null || true
        del_ok=$((del_ok+1))
    else
        del_fail=$((del_fail+1))
        log "DEL FAIL $src_path"
    fi
done

# --- Adds + Modifications ---
# A single curl 422/500 must NOT short-circuit the batch — earlier versions
# used `set -e` + `curl -fsS` which aborted on first non-2xx, silently
# dropping the rest of the batch. Per-file isolation: capture HTTP code,
# log the failure, continue the loop.
ok=0; fail=0; skipped=0; too_big=0; empty=0
for f in "${ADDED[@]:-}" "${MODIFIED[@]:-}"; do
    [[ -z "$f" ]] && continue
    [[ -f "$REPO/$f" ]] || { log "missing source $f"; continue; }

    SIZE=$(stat -c%s "$REPO/$f")
    if (( SIZE > MAX_FILE_BYTES )); then
        log "skip $f (size=$SIZE > $MAX_FILE_BYTES)"
        too_big=$((too_big+1))
        continue
    fi

    # Pre-POST: skip files the server would reject anyway (empty after
    # canonicalization → HTTP 422). Saves a curl round trip per push.
    if ! ingestable "$REPO/$f"; then
        log "skip $f (empty after canonicalization)"
        empty=$((empty+1))
        continue
    fi

    # Mirror to inputs/code/
    src_path="code/$f"
    dst="$INPUTS_CODE/$f"
    mkdir -p "$(dirname "$dst")"
    cp "$REPO/$f" "$dst"

    # Cost guard: if disk SHA matches DB SHA, skip POST entirely
    if ! content_changed "$f"; then
        skipped=$((skipped+1))
        continue
    fi

    # Purge any existing doc with this file_path (SQL bypass)
    purge_doc_for_path "$src_path" 2>>"$LOG" || true

    HTTP=$(post_file "$f" || echo "000")
    if [[ "$HTTP" == "200" ]]; then
        ok=$((ok+1))
    else
        fail=$((fail+1))
        log "POST FAIL [$HTTP] $f"
    fi
done

log "sync done: del_ok=$del_ok del_fail=$del_fail ingest_ok=$ok skipped_nochange=$skipped too_big=$too_big empty=$empty fail=$fail"
exit 0
