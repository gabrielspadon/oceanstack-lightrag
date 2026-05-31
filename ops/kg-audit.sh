#!/usr/bin/env bash
# Hourly KG quality audit + safe auto-fix + structured log.
#
# Checks 13 dimensions, auto-fixes safe ones, logs everything to
# /fast-array/lightrag/logs/audit-<date>.log + emits a JSON summary line.
# Wired via lightrag-audit.timer (systemd user).

set -uo pipefail

WS=oceanstack_code_schema
LR=${LIGHTRAG_URL:-http://127.0.0.1:9621}
LR_DIR=/fast-array/lightrag
INPUTS=$LR_DIR/inputs
GRAPHML=$LR_DIR/rag-storage/$WS/graph_chunk_entity_relation.graphml
LOG_DIR=$LR_DIR/logs
LOG="$LOG_DIR/audit-$(date +%Y-%m-%d).log"
SUMMARY="$LOG_DIR/audit-summary.jsonl"
mkdir -p "$LOG_DIR"

ts() { date -Iseconds; }
log() { echo "$(ts) [audit] $*" | tee -a "$LOG"; }
sql() { psql --no-psqlrc -d lightrag -tAc "$1" 2>/dev/null; }
api() {
    local method="${2:-GET}" path="$1"
    curl -fsS --max-time 15 -X "$method" -H "X-API-Key: $KEY" "$LR$path" 2>/dev/null
}

KEY=$(SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}" \
    sops -d --input-type dotenv --output-type dotenv "$LR_DIR/.env.enc" \
    2>/dev/null | grep '^LIGHTRAG_API_KEY=' | cut -d= -f2)

log "=== KG audit start ==="

# Track findings + fixes
declare -A FOUND
declare -A FIXED

# --- 1. Service health ---
SVC_STATE=$(systemctl --user is-active lightrag.service 2>&1)
WATCH_STATE=$(systemctl --user is-active lightrag-retry.service 2>&1)
INOTIFY_STATE=$(systemctl --user is-active lightrag-inotify.service 2>&1)
FOUND[svc]="lightrag=$SVC_STATE watcher=$WATCH_STATE inotify=$INOTIFY_STATE"
log "[1] services: ${FOUND[svc]}"
if ! curl -fsS --max-time 5 "$LR/health" >/dev/null 2>&1; then
    log "[1] WARN server unreachable; aborting downstream checks"
    echo "{\"ts\":\"$(ts)\",\"status\":\"server_down\"}" >> "$SUMMARY"
    exit 0
fi

# --- 2. doc_status by status ---
mapfile -t STATUS_ROWS < <(sql "SELECT status||':'||COUNT(*) FROM lightrag_doc_status WHERE workspace='$WS' GROUP BY status")
DSTATUS="${STATUS_ROWS[*]}"
FOUND[doc_status]="$DSTATUS"
log "[2] doc_status: $DSTATUS"

# --- 3. Failed docs ---
FAILED_N=$(sql "SELECT COUNT(*) FROM lightrag_doc_status WHERE workspace='$WS' AND status='failed'" | tr -d ' ')
FOUND[failed]=$FAILED_N
if (( FAILED_N > 0 )); then
    log "[3] failed=$FAILED_N — listing top 3:"
    sql "SELECT id||'|'||file_path||'|'||LEFT(COALESCE(error_msg,''),60) FROM lightrag_doc_status WHERE workspace='$WS' AND status='failed' ORDER BY updated_at DESC LIMIT 3" | sed 's/^/    /' | tee -a "$LOG" >/dev/null
fi

# --- 4. dup-* server-emitted markers — reap ---
DUP_N=$(sql "SELECT COUNT(*) FROM lightrag_doc_status WHERE workspace='$WS' AND id LIKE 'dup-%'" | tr -d ' ')
FOUND[dup_markers]=$DUP_N
if (( DUP_N > 0 )); then
    sql "DELETE FROM lightrag_doc_status WHERE workspace='$WS' AND id LIKE 'dup-%'" >/dev/null
    FIXED[dup_markers]=$DUP_N
    log "[4] reaped $DUP_N dup-* markers"
fi

# --- 5. NULL content_sha256 — backfill from doc_full content ---
SHA_NULL_BEFORE=$(sql "SELECT COUNT(*) FROM lightrag_doc_status WHERE workspace='$WS' AND content_sha256 IS NULL" | tr -d ' ')
FOUND[null_sha]=$SHA_NULL_BEFORE
if (( SHA_NULL_BEFORE > 0 )); then
    UPDATED=$(sql "
        UPDATE lightrag_doc_status ds
        SET content_sha256=ENCODE(SHA256(CONVERT_TO(df.content,'UTF8')),'hex')
        FROM lightrag_doc_full df
        WHERE ds.workspace=df.workspace AND ds.id=df.id
          AND ds.workspace='$WS' AND ds.content_sha256 IS NULL
          AND df.content IS NOT NULL
        RETURNING ds.id" | wc -l)
    FIXED[null_sha]=$UPDATED
    log "[5] backfilled $UPDATED SHAs"
fi

# --- 6. Stuck processing (>1 hour) ---
STUCK_N=$(sql "SELECT COUNT(*) FROM lightrag_doc_status WHERE workspace='$WS' AND status='processing' AND updated_at < NOW() - INTERVAL '1 hour'" | tr -d ' ')
FOUND[stuck]=$STUCK_N
if (( STUCK_N > 0 )); then
    log "[6] stuck processing >1h: $STUCK_N (listing top 3)"
    sql "SELECT id||'|'||file_path||'|'||updated_at FROM lightrag_doc_status WHERE workspace='$WS' AND status='processing' AND updated_at < NOW() - INTERVAL '1 hour' LIMIT 3" | sed 's/^/    /' | tee -a "$LOG" >/dev/null
fi

# --- 7. Orphan rows (cross-table integrity) ---
ORPH_DF=$(sql "SELECT COUNT(*) FROM lightrag_doc_full df WHERE df.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM lightrag_doc_status ds WHERE ds.workspace=df.workspace AND ds.id=df.id)" | tr -d ' ')
ORPH_DS=$(sql "SELECT COUNT(*) FROM lightrag_doc_status ds WHERE ds.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM lightrag_doc_full df WHERE df.workspace=ds.workspace AND df.id=ds.id)" | tr -d ' ')
ORPH_C=$(sql "SELECT COUNT(*) FROM lightrag_doc_chunks dc WHERE dc.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM lightrag_doc_full df WHERE df.workspace=dc.workspace AND df.id=dc.full_doc_id)" | tr -d ' ')
FOUND[orphan_doc_full]=$ORPH_DF
FOUND[orphan_doc_status]=$ORPH_DS
FOUND[orphan_chunks]=$ORPH_C
log "[7] cross-table orphans: doc_full=$ORPH_DF doc_status=$ORPH_DS chunks=$ORPH_C"

# --- 8. Live vdb tables: zero-valid + partial-orphan + arrays ---
VDB_ENT=$(sql "SELECT tablename FROM pg_tables WHERE tablename LIKE 'lightrag_vdb_entity_%' AND tablename NOT LIKE '%_gemini_legacy' LIMIT 1" | tr -d ' ')
VDB_REL=$(sql "SELECT tablename FROM pg_tables WHERE tablename LIKE 'lightrag_vdb_relation_%' AND tablename NOT LIKE '%_gemini_legacy' LIMIT 1" | tr -d ' ')
VDB_CHK=$(sql "SELECT tablename FROM pg_tables WHERE tablename LIKE 'lightrag_vdb_chunks_%' AND tablename NOT LIKE '%_gemini_legacy' LIMIT 1" | tr -d ' ')

if [[ -n "$VDB_ENT" && -n "$VDB_REL" ]]; then
    DEAD_ENT=$(sql "SELECT COUNT(*) FROM $VDB_ENT v WHERE v.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM unnest(v.chunk_ids) cid JOIN lightrag_doc_chunks c ON c.workspace=v.workspace AND c.id=cid)" | tr -d ' ')
    DEAD_REL=$(sql "SELECT COUNT(*) FROM $VDB_REL v WHERE v.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM unnest(v.chunk_ids) cid JOIN lightrag_doc_chunks c ON c.workspace=v.workspace AND c.id=cid)" | tr -d ' ')
    FOUND[dead_entities]=$DEAD_ENT
    FOUND[dead_relations]=$DEAD_REL
    log "[8] dead vdb: entities=$DEAD_ENT relations=$DEAD_REL"

    if (( DEAD_ENT + DEAD_REL > 0 )); then
        log "[8] running vacuum_graph..."
        psql --no-psqlrc -d lightrag <<SQL >> "$LOG" 2>&1
SET lock_timeout='15s';
SET statement_timeout='180s';
BEGIN;
DELETE FROM $VDB_ENT v WHERE v.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM unnest(v.chunk_ids) cid JOIN lightrag_doc_chunks c ON c.workspace=v.workspace AND c.id=cid);
DELETE FROM lightrag_entity_chunks ec WHERE ec.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(ec.chunk_ids) cid JOIN lightrag_doc_chunks c ON c.workspace=ec.workspace AND c.id=cid);
WITH cleaned AS (
  SELECT v.id, (SELECT ARRAY_AGG(c) FROM unnest(v.chunk_ids) c
                WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks dc WHERE dc.workspace='$WS' AND dc.id=c)) AS nc
  FROM $VDB_ENT v WHERE v.workspace='$WS')
UPDATE $VDB_ENT v SET chunk_ids=COALESCE(cleaned.nc, ARRAY[]::varchar[])
FROM cleaned WHERE v.workspace='$WS' AND v.id=cleaned.id AND v.chunk_ids IS DISTINCT FROM cleaned.nc;
UPDATE lightrag_entity_chunks ec SET chunk_ids=(SELECT COALESCE(jsonb_agg(cid),'[]'::jsonb) FROM jsonb_array_elements_text(ec.chunk_ids) cid WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks c WHERE c.workspace='$WS' AND c.id=cid)) WHERE ec.workspace='$WS';
DELETE FROM $VDB_REL v WHERE v.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM unnest(v.chunk_ids) cid JOIN lightrag_doc_chunks c ON c.workspace=v.workspace AND c.id=cid);
DELETE FROM lightrag_relation_chunks rc WHERE rc.workspace='$WS' AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(rc.chunk_ids) cid JOIN lightrag_doc_chunks c ON c.workspace=rc.workspace AND c.id=cid);
WITH cleaned AS (
  SELECT v.id, (SELECT ARRAY_AGG(c) FROM unnest(v.chunk_ids) c
                WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks dc WHERE dc.workspace='$WS' AND dc.id=c)) AS nc
  FROM $VDB_REL v WHERE v.workspace='$WS')
UPDATE $VDB_REL v SET chunk_ids=COALESCE(cleaned.nc, ARRAY[]::varchar[])
FROM cleaned WHERE v.workspace='$WS' AND v.id=cleaned.id AND v.chunk_ids IS DISTINCT FROM cleaned.nc;
UPDATE lightrag_relation_chunks rc SET chunk_ids=(SELECT COALESCE(jsonb_agg(cid),'[]'::jsonb) FROM jsonb_array_elements_text(rc.chunk_ids) cid WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks c WHERE c.workspace='$WS' AND c.id=cid)) WHERE rc.workspace='$WS';
COMMIT;
SQL
        FIXED[dead_entities]=$DEAD_ENT
        FIXED[dead_relations]=$DEAD_REL
    fi
fi

# --- 9. Disk-vs-DB drift (reconcile report only — don't auto-reprocess) ---
if command -v "$LR_DIR/reconcile.sh" >/dev/null 2>&1; then
    RECON=$("$LR_DIR/reconcile.sh" report 2>/dev/null | tail -1)
    DRIFT_LINE=$(echo "$RECON" | grep -oE 'disk=[0-9]+ db=[0-9]+ missing=[0-9]+ drift=[0-9]+ stale=[0-9]+' || echo "")
    FOUND[reconcile]="$DRIFT_LINE"
    log "[9] reconcile: $DRIFT_LINE"
fi

# --- 10. Case+separator entity duplicates (incremental dedup) ---
ALL_LABELS=$(api /graph/label/list 2>/dev/null || echo '[]')
DUP_GROUPS=$(echo "$ALL_LABELS" | /fast-array/lightrag/.venv/bin/python -c "
import sys, json, importlib.util
sys.path.insert(0, '/fast-array/lightrag/.venv/lib/python3.13/site-packages')
from lightrag.operate import _canonical_entity_name
from collections import defaultdict
labels = json.load(sys.stdin)
g = defaultdict(list)
for l in labels:
    g[_canonical_entity_name(l)].append(l)
print(sum(1 for v in g.values() if len(v) > 1))" 2>/dev/null || echo 0)
FOUND[case_dup_groups]=$DUP_GROUPS
log "[10] case+separator dup groups: $DUP_GROUPS"

# Auto-merge cap: 50 groups per audit run (~3 min) to avoid hogging GPU
if (( DUP_GROUPS > 0 )) && pgrep -f 'dedup-entities.py' >/dev/null 2>&1; then
    log "[10] dedup already running elsewhere; skip"
elif (( DUP_GROUPS > 0 )); then
    log "[10] running dedup-entities (limit=50)..."
    cd "$LR_DIR" && source .venv/bin/activate
    LIGHTRAG_URL=$LR LIGHTRAG_API_KEY=$KEY \
      python3 dedup-entities.py --limit 50 --sleep 1 --max-failures 20 2>&1 | tee -a "$LOG" >/dev/null
    deactivate 2>/dev/null || true
    MERGED=$(grep -c '    ok' "$LOG" | tail -1)
    FIXED[case_dups]=$MERGED
fi

# --- 11. Inputs/ vs tree drift (only as a tripwire — don't auto-rsync) ---
if [[ -d /home/spadon/Codebases/OceanStack ]]; then
    TREE_N=$(cd /home/spadon/Codebases/OceanStack && find . -type f \( -name '*.py' -o -name '*.rs' -o -name '*.sql' \) \
        -not -path '*/external/lightrag/*' -not -path '*/target/*' -not -path '*/.venv/*' \
        -not -path '*/__pycache__/*' -not -path '*/.git/*' -not -path '*/build/*' \
        -not -path '*/.pytest_cache/*' -not -path '*/.ruff_cache/*' -not -path '*/.mypy_cache/*' 2>/dev/null | wc -l)
    INPUTS_N=$(find "$INPUTS/code" -type f \( -name '*.py' -o -name '*.rs' -o -name '*.sql' \) 2>/dev/null | wc -l)
    FOUND[tree_vs_inputs]="tree=$TREE_N inputs=$INPUTS_N delta=$((INPUTS_N - TREE_N))"
    log "[11] ${FOUND[tree_vs_inputs]}"
fi

# --- 12. graphml file freshness ---
if [[ -f "$GRAPHML" ]]; then
    GMTIME=$(stat -c%Y "$GRAPHML")
    NOW=$(date +%s)
    GMAGE=$(( (NOW - GMTIME) / 60 ))
    GSIZE=$(stat -c%s "$GRAPHML")
    FOUND[graphml]="mtime_age_min=$GMAGE size_bytes=$GSIZE"
    log "[12] graphml: ${FOUND[graphml]}"
fi

# --- 13. Pipeline currently busy? ---
PSTAT=$(api /documents/pipeline_status 2>/dev/null)
PBUSY=$(echo "$PSTAT" | python3 -c 'import sys,json
try: d=json.load(sys.stdin); print(d.get("busy",False))
except: print("?")' 2>/dev/null)
FOUND[pipeline_busy]="$PBUSY"
log "[13] pipeline busy=$PBUSY"

# --- 14. Graphml orphan + garble prune ---
# Removes code-class graphml nodes absent from the PG vdb (orphans the SQL-purge
# ingest path leaves behind) whose id has no current source-token referent
# (deleted symbols + gemma transcription garbles). PG-membership + source-token
# gates keep real canonicalized entities. Skipped while the pipeline is busy so
# in-flight extractions are not misjudged. Deletes via the server API.
if [[ "$PBUSY" != "True" && -f "$LR_DIR/kg-prune-orphans.py" ]]; then
    PRUNE_OUT=$("$LR_DIR/.venv/bin/python" "$LR_DIR/kg-prune-orphans.py" --apply 2>&1 | tee -a "$LOG")
    PRUNED=$(echo "$PRUNE_OUT" | grep -oE 'deleted_ok=[0-9]+' | grep -oE '[0-9]+' | tail -1)
    FIXED[orphans_pruned]=${PRUNED:-0}
    log "[14] orphan prune: deleted=${PRUNED:-0}"
else
    log "[14] orphan prune skipped (pipeline busy or script missing)"
fi

# --- Final summary line (JSON for telemetry/grep) ---
SUMMARY_JSON=$(python3 <<PY
import json
f = {
    "ts": "$(ts)",
    "doc_status": "${FOUND[doc_status]:-}",
    "failed": ${FOUND[failed]:-0},
    "dup_markers_reaped": ${FIXED[dup_markers]:-0},
    "null_sha_filled": ${FIXED[null_sha]:-0},
    "stuck_processing": ${FOUND[stuck]:-0},
    "orphan_doc_full": ${FOUND[orphan_doc_full]:-0},
    "orphan_doc_status": ${FOUND[orphan_doc_status]:-0},
    "orphan_chunks": ${FOUND[orphan_chunks]:-0},
    "dead_entities_vacuumed": ${FIXED[dead_entities]:-0},
    "dead_relations_vacuumed": ${FIXED[dead_relations]:-0},
    "reconcile": "${FOUND[reconcile]:-}",
    "case_dup_groups": ${FOUND[case_dup_groups]:-0},
    "case_dups_merged": ${FIXED[case_dups]:-0},
    "orphans_pruned": ${FIXED[orphans_pruned]:-0},
    "tree_vs_inputs": "${FOUND[tree_vs_inputs]:-}",
    "graphml": "${FOUND[graphml]:-}",
    "pipeline_busy": "${FOUND[pipeline_busy]:-}",
    "services": "${FOUND[svc]:-}",
}
print(json.dumps(f, separators=(",",":")))
PY
)
echo "$SUMMARY_JSON" | tee -a "$SUMMARY" >> "$LOG"
log "=== audit done ==="
exit 0
