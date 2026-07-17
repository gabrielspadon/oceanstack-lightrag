# Coordination notes, oceanstack-lightrag → OceanStack-nightly

Session date: 2026-07-17. Branch: `feat/greenfield-kg-core` (fork `gabrielspadon/oceanstack-lightrag`).

## Current stable surface (unchanged by this session)

- HTTP API is read-only and plane-scoped only: `/planes/{plane}/query`, `/planes/{plane}/query/data`, `/planes/{plane}/query/stream`, `/planes/{plane}/graphs`, `/planes/{plane}/graph/label/{list,popular,search}`, `/planes/{plane}/graph/entity/exists`, plus `/health`, `/login`, `/auth-status`, `/webui`. Planes are `oceanstack_dev`, `oceanstack_product`, `oceanstack_maritime`.
- Library build surface: `LightRAG.ainsert_knowledge_graph(KnowledgeGraphBuild)`, `lightrag.generation` lifecycle (candidate → BUILD fence → READY → publish CAS → cleanup), `lightrag.api.rag_factory.create_postgres_generation_runtime`. These contracts are NOT changing.
- Provenance headers on every plane response: `X-LightRAG-Plane`, `X-LightRAG-Generation-Id`, `X-LightRAG-Build-Id`, `X-LightRAG-Source-Revision`, `X-LightRAG-Manifest-Digest`.
- `pyproject.toml` package name stays `lightrag-hku` (module `lightrag`) so your deptry package→module mapping and the submodule pin keep working. Only metadata comments/URLs change.

## Changes landing in this session (fork-internal, review-fixed, pushed to `feat/greenfield-kg-core`)

1. Test-suite alignment: `DEFAULT_WOKERS` 2→1 (single-worker contract); stale offline tests repaired. No runtime behavior change for you (the server already rejected workers != 1).
2. Deletion of dead, unmounted router modules (`query_routes.py`, `graph_routes.py`, `ollama_api.py`, `map_routes.py`) and their tests. These were unreachable over HTTP already; if your code imports any of these modules directly (it should not), tell us before you re-pin. `document_routes.py` is retained (never mounted) because it hosts the internal document-ingestion machinery (DocumentManager, pipeline_enqueue_file, file-variant cleanup).
3. Classic LLM extraction no longer mints `UNKNOWN` placeholder entities for dangling edges; such edges are dropped with a warning. Typed builds were already fail-closed. Affects only LLM-driven document ingestion, not `ainsert_knowledge_graph`.
4. Document source identity keeps the caller-supplied relative path instead of reducing to basename (`file_path` no longer collapses `pkg/mod.rs` → `mod.rs`). Typed `source_key` semantics unchanged (already repo-relative). If your extractors relied on basename collapse of `file_path` in doc-status rows, adjust; typed-plane builds are unaffected.
5. `retrieve_typed_records` gains an optional `jurisdiction_predicates` parameter (default `{"located_in", "overlaps_zone"}`, identical behavior; members are matched case-insensitively). The hardcoded maritime predicate set moves behind this parameter so generic core carries no OceanStack ontology. Your publication gate sees the same claims output by default.
6. WebUI: adds generation/build provenance display (headers + citations already emitted by the backend are now rendered). No API change.
7. Doc/env scrub: `env.example`, `env.docker-compose-full`, `AGENTS.md`, `WHITELIST_PATHS` default no longer reference removed `/api/*` (Ollama emulation) or `/documents/*` routes. The `WHITELIST_PATHS` default is now `/health` and the security checks flag exposure of `/planes` routes instead of the removed `/api` ones. If your deployment env pins `WHITELIST_PATHS=/health,/api/*`, it keeps working (the `/api/*` entry just matches nothing).
8. Review-pass hardening (post-review, same branch): parser sidecar directories for relative directory-carrying identities gain an 8-char identity digest (`mod.rs.<digest>.parsed`) so same-basename documents cannot share artifacts; the duplicate-content archive step only moves files inside LightRAG-managed input roots (never caller-owned paths); `whitelist_exposes_plane_routes` in `scripts/setup/lib/validation.sh` now mirrors the server's prefix/exact matching exactly (parity-tested against the Python check). None of these touch the typed build/query contracts.

## Action needed from you

- None immediate. These commits are pushed on `feat/greenfield-kg-core`; re-pin the submodule at your convenience. Nothing in the list above changes the typed build/query contracts you consume.
- If you directly import `lightrag.api.routers.{document,query,graph,map,ollama}_routes` anywhere, flag it (item 2 deletes them).
