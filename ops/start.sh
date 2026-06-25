#!/usr/bin/env bash
# Decrypt sops-encrypted env and exec lightrag-server.
set -euo pipefail

cd "$(dirname "$0")"
export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"

if [[ ! -f .env.enc ]]; then
    echo "missing .env.enc" >&2
    exit 1
fi

# Decrypt secrets to a transient fd-3 stream then load via env
TMP=$(mktemp -p /dev/shm lightrag-env.XXXXXX)
trap 'rm -f "$TMP"' EXIT
chmod 600 "$TMP"
sops --decrypt --input-type dotenv --output-type dotenv .env.enc > "$TMP"

source .venv/bin/activate
set -a
. "$TMP"
set +a
rm -f "$TMP"; trap - EXIT

# Resolve the active project (default: code) and override workspace/port/storage
# for this knowledge graph. PROJECT=code reproduces the original single-KG values
# from .env.enc; PROJECT=maritime selects the bridge-fed vessel KG on its own port.
PROJECT="${PROJECT:-code}"
# shellcheck source=lib/project-env.sh
source lib/project-env.sh
echo "starting lightrag project=${PROJECT} workspace=${WORKSPACE} port=${PORT}" >&2

# Override the merge-summary threshold sourced from .env.enc (set to 2, which
# invokes the LLM on nearly every 2-fragment entity merge and dominates ingest
# time). At this threshold entity descriptions concatenate below the count and
# only genuinely-duplicated entities are LLM-summarized.
export FORCE_LLM_SUMMARY_ON_MERGE=8

# Entity-type taxonomy for extraction — overrides the generic 11-type set in
# .env.enc. Grounded in a structural census of the inserted corpus (no tests,
# +.wgsl shaders): language constructs, database objects, and AIS domain.
# gemma4 classifies each extracted entity into exactly one of these labels.
# A project may override this in projects/<name>.env (e.g. maritime uses a vessel
# taxonomy); otherwise the code-KG taxonomy below is the default.
export ENTITY_TYPES="${ENTITY_TYPES:-[\"MODULE\",\"FUNCTION\",\"METHOD\",\"CLASS\",\"DATACLASS\",\"ENUM\",\"PROTOCOL\",\"MACRO\",\"FFI_BINDING\",\"CONSTANT\",\"EXCEPTION\",\"SCHEMA\",\"TABLE\",\"COLUMN\",\"DOMAIN_TYPE\",\"SQL_FUNCTION\",\"CAGG\",\"INDEX\",\"GPU_KERNEL\",\"AIS_CONCEPT\",\"LIBRARY\",\"CONCEPT\"]}"

# Verify the OceanStack canonicalizer patch is installed before launch.
# Without it, entity extraction fragments the KG into bare + schema-qualified
# duplicates. Re-apply with: just rag-patch (in the OceanStack repo).
OPERATE_PY=".venv/lib/python3.13/site-packages/lightrag/operate.py"
if [[ ! -f "$OPERATE_PY" ]]; then
    echo "FATAL: $OPERATE_PY missing — venv not synced" >&2
    exit 1
fi
if ! grep -qF '# --- BEGIN OceanStack patch: canonical entity-name normalization ---' "$OPERATE_PY"; then
    echo "FATAL: OceanStack canonicalizer patch missing from $OPERATE_PY" >&2
    echo "  fix: cd ~/Codebases/OceanStack && just rag-patch" >&2
    exit 1
fi

# Sanitize workspace (LightRAG requires alphanumeric+underscore only)
export WORKSPACE="${WORKSPACE//-/_}"

# Auto-confirm .env-existence prompt (we use sops, .env is placeholder)
exec lightrag-server "$@" <<<'yes'
