#!/usr/bin/env bash
# Full workspace reset — use when migrating LLM/embed models, changing
# embedding dim, or doing a clean re-ingest. Stops lightrag, backs up
# everything that's hard to recreate, clears everything that should be
# regenerated, then leaves the server ready for fresh ingest.
#
# Usage: reset-workspace.sh [workspace_name]
#   workspace_name defaults to WORKSPACE env var or 'oceanstack_code_schema'

set -euo pipefail

WS="${1:-${WORKSPACE:-oceanstack_code_schema}}"
LIGHTRAG_DIR=/fast-array/lightrag
GRAPHML="$LIGHTRAG_DIR/rag-storage/$WS/graph_chunk_entity_relation.graphml"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LIGHTRAG_DIR/logs/reset-$TS.log"

mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

log "=== workspace reset: $WS ==="

# 1) Stop the ingest stack (watcher + inotify + any in-flight reconcile)
log "stopping watcher + inotify..."
systemctl --user stop lightrag-retry.service 2>/dev/null || true
systemctl --user stop lightrag-inotify.service 2>/dev/null || true
sleep 2
pkill -TERM -f 'reconcile.sh' 2>/dev/null || true
sleep 1
pkill -KILL -f 'reconcile.sh' 2>/dev/null || true

# 2) Stop the server cleanly so it flushes graphml before we move it
log "stopping lightrag.service..."
systemctl --user stop lightrag.service 2>/dev/null || true
sleep 3
pkill -9 -f lightrag-server 2>/dev/null || true

# 3) Backup graphml file (cheap; reused if user wants to roll back)
if [[ -f "$GRAPHML" ]]; then
    log "backing up graphml → ${GRAPHML}.bak.$TS"
    mv "$GRAPHML" "${GRAPHML}.bak.$TS"
fi

# 4) Clean PG state for this workspace:
#    - keep raw inputs intact (inputs/code/ on disk)
#    - keep doc_status / doc_full as the canonical source-of-truth if we
#      want to preserve "what was indexed" between resets — caller chooses
#      via FULL_PURGE env var
FULL=${FULL_PURGE:-1}
if [[ "$FULL" == "1" ]]; then
    log "FULL_PURGE=1 → clearing all PG state for workspace='$WS'"
    psql --no-psqlrc -d lightrag -v ON_ERROR_STOP=1 <<SQL >> "$LOG" 2>&1
SET lock_timeout='15s';
SET statement_timeout='300s';
BEGIN;
DELETE FROM lightrag_doc_chunks         WHERE workspace='$WS';
DELETE FROM lightrag_full_entities      WHERE workspace='$WS';
DELETE FROM lightrag_full_relations     WHERE workspace='$WS';
DELETE FROM lightrag_entity_chunks      WHERE workspace='$WS';
DELETE FROM lightrag_relation_chunks    WHERE workspace='$WS';
DELETE FROM lightrag_doc_full           WHERE workspace='$WS';
DELETE FROM lightrag_doc_status         WHERE workspace='$WS';
DELETE FROM lightrag_llm_cache          WHERE workspace='$WS';
COMMIT;
SQL
fi

# 5) Clear current (non-legacy) vdb tables for this workspace. The legacy
#    Gemini tables (renamed _gemini_legacy) are preserved as backup.
log "clearing live vdb tables for workspace='$WS'..."
mapfile -t LIVE_VDB < <(psql --no-psqlrc -d lightrag -tAc "
    SELECT tablename FROM pg_tables
    WHERE schemaname='public'
      AND tablename LIKE 'lightrag_vdb%'
      AND tablename NOT LIKE '%_gemini_legacy%'
      AND tablename NOT LIKE '%_legacy'")
for t in "${LIVE_VDB[@]}"; do
    [[ -z "$t" ]] && continue
    psql --no-psqlrc -d lightrag -v ON_ERROR_STOP=1 -tAc \
        "DELETE FROM $t WHERE workspace='$WS'" >> "$LOG" 2>&1
    log "  cleared $t"
done

# 6) Restart lightrag — recreates fresh graphml on first ingest
log "starting lightrag.service..."
systemctl --user start lightrag.service
for i in $(seq 1 30); do
    sleep 2
    if curl -fsS --max-time 3 http://127.0.0.1:9621/health >/dev/null 2>&1; then
        log "lightrag healthy"
        break
    fi
done

# 7) Restart watcher + inotify
systemctl --user start lightrag-retry.service
systemctl --user start lightrag-inotify.service
sleep 3
log "service state:"
systemctl --user is-active lightrag.service lightrag-retry.service lightrag-inotify.service 2>&1 | tee -a "$LOG"

# 8) Final hint
log "RESET COMPLETE. Trigger re-ingest with:"
log "  cd $LIGHTRAG_DIR && env RECONCILE_PARALLEL=4 ./reconcile.sh reprocess"
log ""
log "Or wait for inotify-ingest to repopulate as files change."
log "Backup graphml: ${GRAPHML}.bak.$TS"
