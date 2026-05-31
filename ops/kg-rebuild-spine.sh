#!/usr/bin/env bash
# Purpose: re-apply the OceanStack knowledge-graph augmentation after a from-scratch
#          LightRAG extraction — deterministic structural spine, architecture backbone,
#          abbreviation aliases, vector embedding of injected nodes, and quality gates.
# Usage:   /fast-array/lightrag/kg-rebuild-spine.sh
#
# Run this after the base corpus has been extracted into the oceanstack_code_schema
# workspace (inotify ingest or a manual drop + scan). Each step is idempotent: edges
# that already exist are skipped, nodes already embedded are upserted in place.

set -euo pipefail

DIR=/fast-array/lightrag
PY="$DIR/.venv/bin/python"
GRAPH="$DIR/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml"
LR=http://127.0.0.1:9621
export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"

log() { echo "$(date -Iseconds) [rebuild-spine] $*"; }

KEY=$(sops -d --input-type dotenv --output-type dotenv "$DIR/.env.enc" 2>/dev/null \
    | grep '^LIGHTRAG_API_KEY=' | cut -d= -f2)
[[ -n "$KEY" ]] || { log "could not decrypt LIGHTRAG_API_KEY; abort"; exit 1; }

log "1/6 extracting structural spine from the repository ..."
"$PY" "$DIR/kg-structural-extract.py"

log "2/6 stopping server for offline graph writes ..."
systemctl --user stop lightrag.service
cp -a "$GRAPH" "$GRAPH.prerebuild-$(date +%Y%m%d-%H%M%S)"

log "3/6 injecting structural spine, architecture backbone, aliases; normalizing predicates ..."
"$PY" "$DIR/kg-inject-structural.py"
"$PY" "$DIR/kg-architecture.py"
"$PY" "$DIR/kg-aliases.py"
"$PY" "$DIR/kg-normalize-predicates.py"

log "4/6 restarting server ..."
systemctl --user start lightrag.service
for _ in $(seq 1 30); do
    code=$(curl -sS -m4 -o /dev/null -w '%{http_code}' "$LR/health" 2>/dev/null || true)
    [[ "$code" == "200" ]] && break
    sleep 2
done
[[ "${code:-}" == "200" ]] || { log "server did not return healthy; abort"; exit 1; }

log "5/6 embedding injected nodes into the vector store ..."
"$PY" "$DIR/kg-embed-missing.py"

log "6/6 running quality gates ..."
if LR_KEY="$KEY" "$PY" "$DIR/kg-quality-gates.py"; then
    log "DONE — all gates pass"
else
    log "DONE — gates FAILED, inspect output above"
    exit 1
fi
