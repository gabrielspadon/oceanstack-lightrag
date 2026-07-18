# Immutable graph-plane server

This fork exposes a read-only API and WebUI for three authority planes.

- `oceanstack_dev`
- `oceanstack_product`
- `oceanstack_maritime`

Each request names its plane. The server resolves the active immutable generation once, holds a read lease for the complete request or stream, and returns that generation's provenance. A request never falls back to a default workspace.

## Runtime contract

Run one Uvicorn worker.

```bash
lightrag-server --workers 1
```

The server rejects any other worker count. All four storage selections must use their PostgreSQL implementations.

```dotenv
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
```

Provider, model, dimension, database, and authentication settings come from the deployment environment. The bounded graph builder and query server use the same public runtime factory, so they cannot drift onto different storage or provider configuration.

The server bootstraps the generation registry before accepting traffic. Expired building generations become failed and are removed through an exact cleanup fence before startup completes. Shutdown waits for read leases, finalizes their storage handles, and closes the shared registry.

## Public API

The public surface contains only plane-qualified reads.

```text
POST /planes/{plane}/query
POST /planes/{plane}/query/stream
POST /planes/{plane}/query/data
GET  /planes/{plane}/graphs
GET  /planes/{plane}/graph/label/list
GET  /planes/{plane}/graph/label/popular
GET  /planes/{plane}/graph/label/search
GET  /planes/{plane}/graph/entity/exists
```

Request bodies are strict. `/query` and `/query/data` reject unknown fields with HTTP 422, including `stream`, which earlier releases accepted and ignored on these two endpoints. Only `/query/stream` accepts `stream`. Clients that previously sent `stream` on the non-streaming endpoints must drop the field.

`/query/data` returns typed entities, directed assertions, chunks, citations, and claims. Assertion records preserve caller-owned IDs, predicates, source and destination IDs, evidence, source revisions, confidence, extraction method, score, and traversal path. Graph edges use the `ASSERTION` type and preserve parallel and reciprocal assertions.

Every query response identifies the exact generation through headers.

```text
X-LightRAG-Plane
X-LightRAG-Generation-Id
X-LightRAG-Build-Id
X-LightRAG-Source-Revision
X-LightRAG-Manifest-Digest
```

There are no document ingestion, queue, graph mutation, default-workspace, migration, compatibility, or model-emulation routes.

## WebUI

Build the bundled WebUI from its source directory.

```bash
cd lightrag_webui
bun install --frozen-lockfile
bun test
bun run lint
bun run build
```

The WebUI requires an explicit plane selection. It displays a directed multigraph keyed by assertion ID and calls only the plane-qualified read API.

## Generation publication

OceanStack owns the bounded build command. LightRAG owns the immutable generation registry, storage fences, typed insertion, retrieval, and publication compare-and-swap.

The only generation states are `building`, `ready`, and `failed`. A builder writes under a live build lease, runs integrity and retrieval gates against the unpublished candidate, marks it ready, and publishes it only if the expected active generation still matches. Failed candidates and compare-and-swap losers are deleted storage first and registry last. No watcher or retry daemon revives them.
