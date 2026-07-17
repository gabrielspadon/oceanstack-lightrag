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
# shellcheck source=/dev/null  # transient decrypted env, no static path to follow
. "$TMP"
set +a
rm -f "$TMP"; trap - EXIT

# Drop the generic ENTITY_TYPES carried in .env.enc so it cannot shadow the
# per-project taxonomy: project-env.sh (sourced next) sets it for projects that
# define one, and the code-KG default at the bottom applies otherwise. Without
# this unset the `:-` default below is inert against the .env.enc value.
unset ENTITY_TYPES

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

# Apply the intended LLM context window. .env.enc carries `OLLAMA_NUM_CTX`, but
# the Ollama LLM binding reads `OLLAMA_LLM_NUM_CTX` (the `ollama_llm` arg prefix),
# so the bare name was dead and Ollama silently fell back to the model's Modelfile
# default. 16384 matches the value intended in .env.enc; a project .env may still
# override it. NOTE: raises KV-cache VRAM on the next restart — verify headroom.
export OLLAMA_LLM_NUM_CTX="${OLLAMA_LLM_NUM_CTX:-16384}"

# Entity-type taxonomy for extraction — overrides the generic 11-type set in
# .env.enc. Grounded in a structural census of the inserted corpus (no tests,
# +.wgsl shaders): language constructs, database objects, and AIS domain.
# gemma4 classifies each extracted entity into exactly one of these labels.
# A project may override this in projects/<name>.env (e.g. maritime uses a vessel
# taxonomy); otherwise the code-KG taxonomy below is the default.
export ENTITY_TYPES="${ENTITY_TYPES:-[\"MODULE\",\"FUNCTION\",\"METHOD\",\"CLASS\",\"DATACLASS\",\"ENUM\",\"PROTOCOL\",\"MACRO\",\"FFI_BINDING\",\"CONSTANT\",\"EXCEPTION\",\"SCHEMA\",\"TABLE\",\"COLUMN\",\"DOMAIN_TYPE\",\"SQL_FUNCTION\",\"CAGG\",\"INDEX\",\"GPU_KERNEL\",\"AIS_CONCEPT\",\"LIBRARY\",\"CONCEPT\"]}"

# Sanitize workspace (LightRAG requires alphanumeric+underscore only)
export WORKSPACE="${WORKSPACE//-/_}"

# Auto-confirm .env-existence prompt (we use sops, .env is placeholder)
exec lightrag-server "$@" <<<'yes'
