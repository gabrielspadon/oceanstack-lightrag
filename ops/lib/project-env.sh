#!/usr/bin/env bash
# Purpose: resolve the active LightRAG project and export its workspace/port/storage
#          settings, so the same ops scripts serve N isolated knowledge graphs.
# Usage:   PROJECT=<name> source ops/lib/project-env.sh   (default project: code)
#
# A "project" is one knowledge graph: a distinct LightRAG workspace + server port,
# sharing the host PostgreSQL `lightrag` database (rows are isolated by the
# workspace column + composite PK, so projects never collide). Per-project values
# live in ops/projects/<name>.env; shared secrets stay in the sops .env.enc.

PROJECT="${PROJECT:-code}"
_pe_ops_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_pe_cfg="${_pe_ops_dir}/projects/${PROJECT}.env"
if [[ ! -f "${_pe_cfg}" ]]; then
    echo "project-env: unknown project '${PROJECT}' (no ${_pe_cfg})" >&2
    _pe_avail=""
    for _pe_f in "${_pe_ops_dir}"/projects/*.env; do
        [[ -e "${_pe_f}" ]] && _pe_avail+="$(basename "${_pe_f}" .env) "
    done
    echo "  available: ${_pe_avail}" >&2
    unset _pe_avail _pe_f
    return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck source=/dev/null
. "${_pe_cfg}"
set +a

: "${WORKSPACE:?project ${PROJECT} must set WORKSPACE}"
: "${PORT:?project ${PROJECT} must set PORT}"
# LightRAG sanitizes the workspace to [A-Za-z0-9_]; normalize dashes up front.
export WORKSPACE="${WORKSPACE//-/_}"
export LIGHTRAG_URL="${LIGHTRAG_URL:-http://127.0.0.1:${PORT}}"
unset _pe_ops_dir _pe_cfg
