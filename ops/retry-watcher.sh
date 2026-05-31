#!/usr/bin/env bash
# Continuously retry failed docs, reconcile disk-vs-DB drift, and detect
# stuck pipeline state. Designed to run forever as a systemd user service.
#
# Per tick:
#   1. Health check; skip if server unreachable.
#   2. Snapshot doc status counts.
#   3. Retry status='failed' docs by deleting + re-POSTing from inputs/.
#   4. Backfill content_sha256 NULLs (covers race where doc_full INSERT
#      fired the trigger before doc_status row existed).
#   5. Run reconcile.sh in capped batches to drain missing + drifted docs.
#   6. Pipeline-jam watchdog: if pipeline_busy is true on the same
#      latest_message for >JAM_THRESHOLD secs AND graphml file mtime
#      has not advanced, restart lightrag.service.
#   7. Case-duplicate merge sweep on idle transition / hourly.

set -uo pipefail

WS="oceanstack_code_schema"
INPUTS=/fast-array/lightrag/inputs
LIGHTRAG_URL=http://127.0.0.1:9621
LR_DIR=/fast-array/lightrag
LOG=$LR_DIR/logs/retry-watcher.log
RECONCILE=$LR_DIR/reconcile.sh
GRAPHML=$LR_DIR/rag-storage/$WS/graph_chunk_entity_relation.graphml

SLEEP_OK=120       # 5 min between idle ticks
RECONCILE_BATCH=${RECONCILE_BATCH:-25}   # max files per reconcile call
JAM_THRESHOLD=1200 # 20 min on same latest_message = jam

mkdir -p "$(dirname "$LOG")"
# Avoid double-logging: systemd already captures stdout via StandardOutput=append.
log() { echo "$(date -Iseconds) $*"; }

KEY=$(SOPS_AGE_KEY_FILE="$HOME/.config/sops/age/keys.txt" \
    sops -d --input-type dotenv --output-type dotenv \
    /fast-array/lightrag/.env.enc | grep '^LIGHTRAG_API_KEY=' | cut -d= -f2)

find_source() {
    # file_path is stored already including its top-level dir ("code/..." or
    # "schema/..."), so concat directly with $INPUTS. Also tolerate legacy
    # rows that stored just the relative-to-code path.
    local fp="$1"
    if [[ -f "$INPUTS/$fp" ]]; then echo "$INPUTS/$fp"; return; fi
    for d in "$INPUTS/code" "$INPUTS/schema"; do
        [[ -f "$d/$fp" ]] && { echo "$d/$fp"; return; }
    done
    return 1
}

retry_failed() {
    local count=0
    while IFS=$'\t' read -r doc_id file_path _err; do
        [[ -z "$doc_id" ]] && continue
        # R5-A4 fix: validate doc_id shape so DELETE URL can't inject.
        [[ "$doc_id" =~ ^doc-[a-f0-9]+$ ]] || { log "skip invalid doc_id: $doc_id"; continue; }
        src=$(find_source "$file_path") || { log "src missing: $file_path"; continue; }

        curl -fsS -H "X-API-Key: $KEY" \
            -X DELETE "$LIGHTRAG_URL/documents/$doc_id" >/dev/null 2>&1 || true

        # R5-A4 fix: pass $src / $file_path via argv to avoid Python injection.
        HTTP=$(curl -fsS -o /dev/null -w '%{http_code}' \
            -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
            -X POST "$LIGHTRAG_URL/documents/text" \
            --data @<(python3 - "$src" "$file_path" <<'PY'
import json, pathlib, sys
src, fp = sys.argv[1], sys.argv[2]
text = pathlib.Path(src).read_text(encoding='utf-8', errors='replace').replace('&', '&amp;')
print(json.dumps({'text': text, 'file_source': fp}))
PY
) 2>/dev/null)
        if [[ "$HTTP" == "200" ]]; then
            count=$((count+1))
        else
            log "re-post FAIL [$HTTP] $file_path"
        fi
    done < <(psql --no-psqlrc -d lightrag -v ws="$WS" -tAF $'\t' -c \
        "SELECT id, file_path, COALESCE(error_msg,'') FROM lightrag_doc_status
         WHERE workspace = :'ws' AND status='failed'
         AND id NOT LIKE 'dup-%';" 2>/dev/null)

    # Server-emitted dup-* markers are not retryable (server already has
    # the content under a different doc_id). Sweep them here so they don't
    # accumulate.
    local dup_n
    dup_n=$(psql --no-psqlrc -d lightrag -v ws="$WS" -tAc \
        "DELETE FROM lightrag_doc_status WHERE workspace = :'ws'
         AND id LIKE 'dup-%' AND status='failed' RETURNING 1" 2>/dev/null | wc -l)
    if (( dup_n > 0 )); then
        log "reaped $dup_n dup-* markers"
    fi

    log "retried $count failed docs"
}

backfill_sha() {
    # Covers race: trigger fired on doc_full INSERT before doc_status row was
    # visible (different transactions). Reapplies the join-update.
    # R5-A4 fix: psql -v binding instead of f-string interpolation.
    local n
    n=$(psql --no-psqlrc -d lightrag -v ws="$WS" -tAc "
        UPDATE lightrag_doc_status ds
        SET content_sha256 = ENCODE(SHA256(CONVERT_TO(df.content,'UTF8')),'hex')
        FROM lightrag_doc_full df
        WHERE ds.workspace=df.workspace AND ds.id=df.id
          AND ds.workspace = :'ws'
          AND ds.content_sha256 IS NULL
          AND df.content IS NOT NULL
        RETURNING ds.id;" 2>/dev/null | wc -l)
    if (( n > 0 )); then
        log "backfilled sha for $n rows"
    fi
}

reconcile_batch() {
    if [[ ! -x "$RECONCILE" ]]; then
        return 0
    fi
    BATCH="$RECONCILE_BATCH" "$RECONCILE" batch="$RECONCILE_BATCH" >>"$LOG" 2>&1 || true
}

# Periodic graph vacuum: deletes entities/relations whose chunk_ids all
# point to deleted chunks (truly orphan nodes), and prunes stale chunk_id
# refs from partial-orphan entities/relations. Necessary because
# purge_doc_sql in reconcile.sh nukes per-doc chunks but leaves the
# entity_chunks/relation_chunks/vdb_entity/vdb_relation rows referencing
# them stale — those clean up here.
vacuum_graph() {
    # R5-A4 fix: psql -v binding + dynamic vdb table discovery (no more
    # hard-coded `_gemini_embedding_2_1536d` which became a silent no-op
    # after the Ollama migration).
    local n
    n=$(psql --no-psqlrc -d lightrag -v ws="$WS" -tAc "
        SET lock_timeout='10s';
        SET statement_timeout='180s';
        SELECT set_config('os.ws', :'ws', false);
        DO \$\$
        DECLARE
          ws_v constant text := current_setting('os.ws');
          vt text;
          n_del int := 0;
          got int;
        BEGIN
          -- vdb_entity: delete rows whose chunk_ids reference no live chunks.
          FOR vt IN SELECT tablename FROM pg_tables
                    WHERE tablename LIKE 'lightrag_vdb_entity_%'
                      AND tablename NOT LIKE '%_legacy'
          LOOP
            EXECUTE format(
              'DELETE FROM %I v WHERE v.workspace = \$1 AND NOT EXISTS ('
              || 'SELECT 1 FROM unnest(v.chunk_ids) AS cid '
              || 'JOIN lightrag_doc_chunks c ON c.workspace = \$1 AND c.id = cid)',
              vt) USING ws_v;
            GET DIAGNOSTICS got = ROW_COUNT;
            n_del := n_del + got;
            -- Prune stale chunk_id refs.
            EXECUTE format(
              'UPDATE %I v SET chunk_ids = ('
              || 'SELECT COALESCE(ARRAY_AGG(c), ARRAY[]::varchar[]) FROM unnest(v.chunk_ids) AS c '
              || 'WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks dc '
              || 'WHERE dc.workspace = \$1 AND dc.id = c)) '
              || 'WHERE v.workspace = \$1',
              vt) USING ws_v;
          END LOOP;
          -- vdb_relation: same cleanup.
          FOR vt IN SELECT tablename FROM pg_tables
                    WHERE tablename LIKE 'lightrag_vdb_relation_%'
                      AND tablename NOT LIKE '%_legacy'
          LOOP
            EXECUTE format(
              'DELETE FROM %I v WHERE v.workspace = \$1 AND NOT EXISTS ('
              || 'SELECT 1 FROM unnest(v.chunk_ids) AS cid '
              || 'JOIN lightrag_doc_chunks c ON c.workspace = \$1 AND c.id = cid)',
              vt) USING ws_v;
            EXECUTE format(
              'UPDATE %I v SET chunk_ids = ('
              || 'SELECT COALESCE(ARRAY_AGG(c), ARRAY[]::varchar[]) FROM unnest(v.chunk_ids) AS c '
              || 'WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks dc '
              || 'WHERE dc.workspace = \$1 AND dc.id = c)) '
              || 'WHERE v.workspace = \$1',
              vt) USING ws_v;
          END LOOP;
          -- entity_chunks: drop rows where all chunk_ids are dead; prune the rest.
          DELETE FROM lightrag_entity_chunks ec
          WHERE ec.workspace = ws_v
            AND NOT EXISTS (
              SELECT 1 FROM jsonb_array_elements_text(ec.chunk_ids) AS cid
              JOIN lightrag_doc_chunks c ON c.workspace = ws_v AND c.id = cid
            );
          UPDATE lightrag_entity_chunks ec
          SET chunk_ids = (
            SELECT COALESCE(jsonb_agg(cid), '[]'::jsonb)
            FROM jsonb_array_elements_text(ec.chunk_ids) AS cid
            WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks c
                          WHERE c.workspace = ws_v AND c.id = cid)
          )
          WHERE ec.workspace = ws_v;
          -- relation_chunks: same.
          DELETE FROM lightrag_relation_chunks rc
          WHERE rc.workspace = ws_v
            AND NOT EXISTS (
              SELECT 1 FROM jsonb_array_elements_text(rc.chunk_ids) AS cid
              JOIN lightrag_doc_chunks c ON c.workspace = ws_v AND c.id = cid
            );
          UPDATE lightrag_relation_chunks rc
          SET chunk_ids = (
            SELECT COALESCE(jsonb_agg(cid), '[]'::jsonb)
            FROM jsonb_array_elements_text(rc.chunk_ids) AS cid
            WHERE EXISTS (SELECT 1 FROM lightrag_doc_chunks c
                          WHERE c.workspace = ws_v AND c.id = cid)
          )
          WHERE rc.workspace = ws_v;
          RAISE NOTICE 'vacuum_graph deleted=%', n_del;
        END
        \$\$;
        SELECT 0;" 2>/dev/null | head -1 | tr -d ' ')
    if [[ "${n:-0}" -gt 0 ]]; then
        log "vacuum_graph: deleted $n orphan entities and refreshed chunk_id refs"
    fi
}

count_dups() {
    # Count labels whose canonicalized form collides with another label's.
    # Uses the SAME canonicalizer as ingest + dedup so the count matches the
    # actual merge target set (including bare→schema-qualified collisions).
    curl -fsS --max-time 8 -H "X-API-Key: $KEY" "$LIGHTRAG_URL/graph/label/list" 2>/dev/null \
      | /fast-array/lightrag/.venv/bin/python -c "
import json, sys
from collections import defaultdict
from lightrag.operate import _canonical_entity_name as canon
g = defaultdict(int)
for l in json.load(sys.stdin):
    c = canon(l)
    if c:
        g[c] += 1
print(sum(1 for v in g.values() if v > 1))" 2>/dev/null || echo 0
}

run_case_merge() {
    # Deep dedup: collapses case+separator+wrapping AND bare→schema-qualified
    # via the live _canonical_entity_name patch (operate.py). Single source
    # of truth — dedup-entities.py grouping is the canonicalizer fixpoint.
    log "running deep dedup..."
    cd /fast-array/lightrag
    source .venv/bin/activate
    LIGHTRAG_URL="$LIGHTRAG_URL" LIGHTRAG_API_KEY="$KEY" \
        python3 dedup-entities.py 2>&1 | tee -a "$LOG"
    log "dedup done."
}

# --- Pipeline jam state (persists across ticks) ---
prev_jam_msg=""
prev_jam_seen=0
prev_graphml_mtime=0

pipeline_jam_check() {
    local resp busy msg now graphml_mt elapsed
    resp=$(curl -fsS --max-time 5 -H "X-API-Key: $KEY" \
        "$LIGHTRAG_URL/documents/pipeline_status" 2>/dev/null) || return 0
    busy=$(echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("busy",False))' 2>/dev/null)
    msg=$(echo "$resp"  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("latest_message","") or "")' 2>/dev/null)
    if [[ "$busy" != "True" ]]; then
        prev_jam_msg=""; prev_jam_seen=0; return 0
    fi

    now=$(date +%s)
    graphml_mt=0
    [[ -f "$GRAPHML" ]] && graphml_mt=$(stat -c%Y "$GRAPHML" 2>/dev/null || echo 0)

    if [[ -z "$prev_jam_msg" || "$prev_jam_msg" != "$msg" || "$graphml_mt" -ne "$prev_graphml_mtime" ]]; then
        prev_jam_msg=$msg
        prev_jam_seen=$now
        prev_graphml_mtime=$graphml_mt
        return 0
    fi

    elapsed=$(( now - prev_jam_seen ))
    if (( elapsed > JAM_THRESHOLD )); then
        log "PIPELINE JAM: busy=$busy msg='$msg' stuck=${elapsed}s graphml_mt unchanged ($graphml_mt). Restarting lightrag.service."
        systemctl --user restart lightrag.service || log "service restart FAILED"
        sleep 30
        prev_jam_msg=""; prev_jam_seen=0; prev_graphml_mtime=0
        # Surface the previously stuck doc_id, if any, so it can be retried
        # next tick once status flips to 'failed'.
    fi
}

log "watcher start (workspace=$WS) — runs continuously"
prev_done=0
prev_idle=0
tick=0

while true; do
    tick=$((tick + 1))

    if ! curl -fsS --max-time 5 "$LIGHTRAG_URL/health" >/dev/null 2>&1; then
        log "server down, sleeping ${SLEEP_OK}s"
        sleep "$SLEEP_OK"; continue
    fi

    eval "$(psql --no-psqlrc -d lightrag -tAc \
        "SELECT 'PEND='||COALESCE(SUM(CASE WHEN status='pending' THEN 1 END),0)||' '||
                'PROC='||COALESCE(SUM(CASE WHEN status='processing' THEN 1 END),0)||' '||
                'DONE='||COALESCE(SUM(CASE WHEN status='processed' THEN 1 END),0)||' '||
                'FAIL='||COALESCE(SUM(CASE WHEN status='failed' THEN 1 END),0)
         FROM lightrag_doc_status WHERE workspace='$WS';" 2>/dev/null \
        | tr ' ' '\n' | sed 's/^/export /')"

    log "PEND=$PEND PROC=$PROC DONE=$DONE FAIL=$FAIL tick=$tick"

    pipeline_jam_check

    if (( FAIL > 0 )); then
        retry_failed
    fi

    backfill_sha

    # Hourly graph vacuum (cheap when no orphans; logs when there are)
    if (( tick % 30 == 0 )); then
        vacuum_graph
    fi

    # Run reconcile only when not actively burning through processing work.
    # Throttles new POSTs so we never pile up beyond RECONCILE_BATCH per tick.
    if (( PEND == 0 && PROC == 0 )); then
        reconcile_batch
    fi

    is_idle=0
    if (( PEND == 0 && PROC == 0 && FAIL == 0 && DONE > 0 )); then
        is_idle=1
    fi
    # Deep dedup runs at every idle transition + hourly. The Gemini→Ollama
    # migration removed the rate-limit failure mode that originally disabled
    # this block; the canonicalizer now also collapses bare→schema-qualified
    # at ingest, so this loop is the legacy-state convergence path.
    if (( is_idle == 1 && prev_idle == 0 && DONE > prev_done )); then
        log "ingest reached idle (DONE=$DONE). triggering deep dedup."
        if (( $(count_dups) > 0 )); then
            run_case_merge
        else
            log "no canonicalizable duplicates, skipping dedup"
        fi
    fi

    if (( is_idle == 1 && tick % 12 == 0 )); then
        D=$(count_dups)
        if (( D > 5 )); then
            log "hourly sweep: canonical_dups=$D, running dedup"
            run_case_merge
        fi
    fi

    prev_done=$DONE
    prev_idle=$is_idle
    sleep "$SLEEP_OK"
done
