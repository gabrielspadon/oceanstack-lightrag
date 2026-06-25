# OceanStack-LightRAG

A graph-RAG knowledge engine for the [OceanStack](https://github.com/gabrielspadon) maritime/AIS
platform — a curated fork of [LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS) tuned to build
**code-** and **vessel-knowledge graphs** from locally-hosted models, with the host operational
tooling that runs it in production on a single RTX-4090 workstation.

Upstream LightRAG extracts entities and relationships from documents, assembles them into a
knowledge graph, and answers questions with multi-modal retrieval (local, global, hybrid, naive,
mix). This fork keeps that machinery intact and re-points it at a narrow domain: the OceanStack
source tree and SQL schema, and the vessel/AIS records its pipeline derives. The changes are
small, deliberate, and isolated so that upstream releases can still be tracked and replayed.

> **Status:** private research fork. Based on upstream `v1.4.15`. MIT-licensed, inheriting
> upstream's license. The pristine upstream README is preserved in git history at the baseline
> commit and remains the reference for general LightRAG usage.

---

## Table of contents

- [Origins](#origins)
- [What changed, and why](#what-changed-and-why)
- [Architecture](#architecture)
- [Knowledge graphs (projects)](#knowledge-graphs-projects)
- [Getting started](#getting-started)
- [Operations](#operations)
- [Tracking upstream](#tracking-upstream)
- [Repository layout](#repository-layout)
- [License & acknowledgements](#license--acknowledgements)

---

## Origins

This repository is a fork, not a rewrite. It exists to **version a set of in-place patches** that
previously lived only in the live server's virtual environment, untracked.

The parent OceanStack project vendors upstream LightRAG as a **read-only** submodule
(`external/lightrag`), pinned and installed pristine via `uv`; that submodule is never edited. But
the server that actually runs on the host is patched to teach the extractor the maritime/AIS code
taxonomy and to drive local Ollama models correctly. Pinning those patches against a known upstream
commit makes them diffable, reviewable, replayable on a new release, and recoverable if the host is
rebuilt.

| | |
| -------------- | ------------------------------------------------------------------- |
| Upstream       | <https://github.com/HKUDS/LightRAG>                                 |
| Pinned tag     | `v1.4.15`                                                           |
| Pinned commit  | `64d3326f858db300ec7699b2cb84e4edd5e5869e`                         |
| Baseline tag   | `v1.4.15-baseline` (pristine upstream, build artifacts stripped)    |
| Package        | `lightrag-hku` — `__version__ 1.4.15`, `__api_version__ 0287`       |

Every OceanStack change is layered on the baseline, so the fork delta is always
`git diff v1.4.15-baseline..HEAD`.

---

## What changed, and why

The fork delta is split into two layers with opposite maintenance rules. The Python package
(`lightrag/`) tracks **upstream + a minimal delta** and must survive a rebase; the host layer
(`ops/`) has no upstream counterpart and never participates in a rebase.

### 1. A domain-specific extraction taxonomy — `lightrag/prompt.py`, `lightrag/operate.py`

Upstream ships a generic entity taxonomy (organization, person, location, event…). Run over source
code and a SQL schema it produces a graph of almost no analytical value. It is replaced with a
**22-type code/DB/GPU/AIS taxonomy** (module, function, class, dataclass, enum, protocol, macro,
FFI binding, schema, table, column, SQL function, continuous aggregate, GPU kernel, AIS concept, …),
grounded in a structural census of the actual corpus. Because the LLM still emits off-taxonomy
spellings, a **remap table** (~270 entries) folds variants back onto the canonical labels, and the
set is closure-checked at import. Relations use a **verb-only keyword vocabulary** (`calls`,
`reads_from`, `implements`, `raises`, `inherits_from`, …) instead of free text.

### 2. A deterministic entity canonicalizer — `lightrag/operate.py`

The same logical entity is extracted under many surface forms: bare vs schema-qualified
(`vessel_tracks` vs `derived.vessel_tracks`), `camelCase` vs `snake_case`, smart-quoted,
zero-width-polluted, wrapped in backticks or call parentheses. Left alone, the graph fragments into
duplicate nodes. `_canonical_entity_name` applies a deterministic, idempotent normalization
(Unicode fold, control-character strip, wrapper unwrap, separator collapse) while preserving
meaningful tokens (Rust `!`/`?`, leading `_`, `::`, dotted namespacing). Crucially, the **same
function is imported by the offline `ops/kg-*.py` maintenance tools**, so the live extractor and the
batch reconciler agree on entity identity. A boot-time guard in `ops/start.sh` refuses to start if
the patch is missing from the installed package.

### 3. Local-model tuning — `lightrag/llm/ollama.py`

The deployment runs entirely on locally-hosted models via Ollama. The Ollama binding adds a
**reasoning (`/think`) toggle keyed on the system-prompt signature**: reasoning is disabled during
extraction (faster, deterministic, no token waste) and enabled when answering queries (where
chain-of-thought helps). Models are configured per project (see below).

### 4. A merge-summary cost gate — `ops/start.sh`

Upstream invokes the LLM to summarize merged entity descriptions once a merge fuses **2**
fragments — which fires on nearly every merge and dominates ingest time. The deployment raises
`FORCE_LLM_SUMMARY_ON_MERGE` to **8**: below the threshold descriptions are concatenated for free,
and only genuinely heavily-duplicated entities pay for an LLM summary.

### 5. The host operational layer — `ops/`

Everything required to run, feed, and maintain the server on the host: a server launcher, an MCP
bridge, an ingest/durability pipeline, knowledge-graph maintenance scripts, and encrypted secrets.
None of it is upstream. See [Operations](#operations).

### 6. WebUI additions — `lightrag_webui/`, `lightrag/api/routers/map_routes.py`

Two maritime-oriented views layered onto the upstream React UI: a **deck.gl + MapLibre map** of
ports and vessel tracks, and a GPU-accelerated **`@cosmos.gl` graph viewer** (with Louvain
community detection) alongside the stock sigma.js viewer. The map reads a **separate** PostgreSQL
database (the OceanStack source DB) read-only via its own connection pool.

---

## Architecture

```
        Claude Code ──(MCP)──► ops/lightrag_mcp.py ──┐
                                                      │ REST
   WebUI (React 19 · Zustand · Vite/Bun) ────────────┤
     sigma.js · @cosmos.gl graph · deck.gl map        │
                                                      ▼
                            FastAPI server (lightrag/api)
                  document · query · graph · map · ollama-compat
                                      │
                       LightRAG orchestrator (lightrag.py)
            ainsert → chunk → extract (canonicalize · taxonomy · merge gate)
                       query → vdb select → rerank → context
                          │                         │
                  Ollama (local models)     ┌───────┴────────────────────────┐
                                            ▼                                 ▼
                        PostgreSQL  «lightrag»                 PostgreSQL  «oceanstack»
                  KV · pgvector · Apache AGE graph             external.world_ports
                  doc-status · PK (workspace, id)              derived.vessel_tracks
                                                               (read-only, for the map)
```

LightRAG organizes storage into four pluggable roles — **KV** (LLM cache, chunks, doc info),
**vector** (entity/relation/chunk embeddings), **graph** (entity-relation structure), and
**doc-status** (ingest state). This deployment binds all four to **PostgreSQL**: pgvector for
vectors and **Apache AGE** for the graph, with every row keyed by a composite
`(workspace, id)` primary key so independent knowledge graphs share one database without colliding.

Two databases are in play and should not be confused:

- **`lightrag`** — the RAG store (KV, vectors, AGE graph, doc-status). One AGE graph per workspace.
- **`oceanstack`** — the maritime source database, read **read-only** by `map_routes.py` to draw the
  map (`external.world_ports`, and `derived.vessel_tracks`, a large TimescaleDB hypertable).

---

## Knowledge graphs (projects)

The same codebase serves **N isolated knowledge graphs**. A *project* is one graph: a distinct
LightRAG **workspace + server port**, all sharing the host `lightrag` database via workspace-column
isolation. NetworkX graphml lands under `WORKING_DIR/<workspace>/`, one subdirectory per project.
Per-project settings live in `ops/projects/<name>.env`; shared secrets stay in the sops-encrypted
`ops/.env.enc`. `ops/lib/project-env.sh` resolves the active project (default `code`).

| Project | Workspace               | Port  | Fed by                                   | Extraction taxonomy           |
| ------- | ----------------------- | ----- | ---------------------------------------- | ----------------------------- |
| `code`  | `oceanstack_code_schema`| 9621  | `inotify-ingest.sh` (source/schema files)| 22 code/DB/GPU/AIS types       |
| `ships` | `oceanstack_ships`      | 9622  | OceanStack derive bridge (HTTP inserts)  | 13 vessel types               |

The `ships` project overrides `ENTITY_TYPES` with a vessel-domain set (vessel, port, port-call,
voyage, AIS event, spoofing alert, dark activity, interaction, zone, flag state, ship type, nav
status). Both projects currently run the same local models, configured in their `*.env` files:

- **LLM:** `gemma4:26b-a4b-it-q4_K_M`
- **Embeddings:** `qwen3-embedding:0.6b` (1024-dimensional)

> Changing the embedding model or dimension requires clearing vector storage and re-indexing —
> embeddings must be consistent across ingest and query.

Each project is a separate server process (LightRAG is one-workspace-per-process). Add a project by
dropping a new `ops/projects/<name>.env` with a `WORKSPACE` and a free `PORT` — no script edits.

---

## Getting started

### Prerequisites

- **PostgreSQL** with the `pgvector` and Apache **AGE** extensions (plus PostGIS/TimescaleDB on the
  separate `oceanstack` DB if you want the map view).
- **[Ollama](https://ollama.com/)** with the configured models pulled.
- **[uv](https://docs.astral.sh/uv/)** (Python) and **[Bun](https://bun.sh/)** (WebUI).
- **[sops](https://github.com/getsops/sops) + [age](https://github.com/FiloSottile/age)** to decrypt
  the host config (only needed to run the patched server via `ops/`).

### Install

```bash
make dev                 # uv sync (test + offline extras) + bun install + bun run build
source .venv/bin/activate
```

`make dev` requires `uv` and `bun` to be installed. For finer control, the upstream extras still
work: `uv sync --extra api`, `--extra offline-storage`, `--extra offline-llm`, `--extra docling`.

### Configure & run — OceanStack deployment

```bash
# install the templated systemd unit once
cp ops/systemd/lightrag@.service ~/.config/systemd/user/
systemctl --user daemon-reload

systemctl --user start lightrag@code     # code KG on :9621
systemctl --user start lightrag@ships    # ships KG on :9622
```

Or run a project directly in the foreground:

```bash
PROJECT=ships ./ops/start.sh
```

`ops/start.sh` decrypts `ops/.env.enc`, activates the host venv, applies the project overrides, and
verifies the canonicalizer patch before launching.

### Configure & run — vanilla LightRAG

This is still LightRAG. For a non-OceanStack setup, follow upstream's interactive wizard and run the
stock entry points (see the preserved upstream README and `env.example`):

```bash
make env-base            # interactive LLM / embedding / reranker setup -> writes .env
lightrag-server          # or: uvicorn lightrag.api.lightrag_server:app --reload
```

### Develop & test

```bash
ruff check .                                   # Python lint
python -m pytest tests                         # offline tests (default)
python -m pytest tests --run-integration       # opt into integration (needs live backends)
python -m pytest tests/test_postgres_upsert.py # a single file
./scripts/test.sh                              # venv-resolving wrapper for fresh shells

cd lightrag_webui && bun run lint && bun run build && bun test   # WebUI
```

---

## Operations

Everything under `ops/` is host glue with no upstream counterpart.

- **`start.sh`** — boots a project: sops-decrypts secrets, applies project overrides
  (`ENTITY_TYPES`, `FORCE_LLM_SUMMARY_ON_MERGE`, …), and guards the canonicalizer patch.
- **`lightrag_mcp.py`** — a FastMCP bridge over the LightRAG REST API, exposing documents, query,
  graph, and entity/relation tools to Claude Code. Reads `LIGHTRAG_URL` / `LIGHTRAG_API_KEY`.
- **Ingest / durability pipeline** (run as systemd services):
  - `inotify-ingest.sh` — sub-second: watches the input tree, SHA-checks, ingests on save.
  - `retry-watcher.sh` — periodic: replays failed documents, backfills content hashes, prunes
    orphaned graph rows, and restarts a jammed pipeline.
  - `reconcile.sh` — on demand: diffs the source tree against the database (missing / drifted /
    stale) and repairs it.
  - Content dedup keys on `sha256(text.rstrip())`; `schema/content_sha256.sql` installs the trigger.
- **Knowledge-graph maintenance** (`kg-*.py`, `dedup-entities.py`, `batch_ingest.py`) — structural
  extraction/injection, type/predicate normalization, orphan pruning, alias management, embedding
  backfill, and read-only quality gates. These share the live extractor's canonicalizer.
- **`sync-from-git.sh`** — pulls the patched sources into the live host venv after a rebase.

### Secrets

`ops/.env.enc` is sops-encrypted at rest (age recipient in `ops/.age-recipient`). The age **private
key is not in this repository** — it lives at `~/.config/sops/age/keys.txt` on the host. No
plaintext secrets, API keys, or private keys are committed.

```bash
SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt sops -d ops/.env.enc
```

---

## Tracking upstream

```bash
# one-time: wire the upstream remote
git remote add upstream https://github.com/HKUDS/LightRAG.git
git fetch upstream --tags

# what upstream changed since our fork point, and our delta in isolation
git diff v1.4.15-baseline..<new-tag> -- lightrag/
git diff v1.4.15-baseline..HEAD      -- lightrag/
```

Rebase onto a new release:

```bash
git checkout -b bump/<new-tag> <new-tag>
git rebase --onto <new-tag> v1.4.15-baseline main
# conflicts land ONLY in lightrag/operate.py, prompt.py, llm/ollama.py
git tag <new-tag>-baseline <new-tag>
bash ops/sync-from-git.sh            # on the host, sync patched sources into the live venv
```

`ops/` never conflicts. The parent OceanStack submodule pin is bumped separately
(`just rag-update <tag>`), because the parent tracks pristine upstream, not this fork.

---

## Repository layout

```
oceanstack-lightrag/
├── lightrag/                 # upstream package (baseline = pristine v1.4.15, delta on top)
│   ├── operate.py            # MODIFIED — canonicalizer, 22-type taxonomy + remap,
│   │                         #   ghost typing, merge-summary gate
│   ├── prompt.py             # MODIFIED — domain extraction taxonomy + verb-only relations
│   ├── llm/ollama.py         # MODIFIED — reasoning (/think) toggle
│   ├── api/                  # FastAPI server, routers (incl. map_routes.py), auth, WebUI host
│   └── kg/                   # storage backends (PostgreSQL/AGE used here)
├── lightrag_webui/           # React 19 + TypeScript UI (adds map + cosmos graph views)
├── ops/                      # host layer, NOT upstream (launcher, MCP, ingest, KG tools, secrets)
│   ├── projects/<name>.env   # per-project workspace / port / models / taxonomy
│   └── systemd/lightrag@.service
├── tests/                    # pytest suite
└── CLAUDE.md                 # contributor guide for AI coding agents
```

The split is the whole point: `lightrag/` is rebasable upstream-plus-delta; `ops/` is host glue
that never rebases.

---

## License & acknowledgements

Licensed under the **MIT License** (see [`LICENSE`](LICENSE)), inheriting the license of upstream
LightRAG.

This project is built on **[LightRAG](https://github.com/HKUDS/LightRAG)** by HKUDS — all of the
core retrieval-augmented generation machinery is their work. If you use this in research, please
cite the upstream LightRAG paper as directed in their repository. The OceanStack fork contributes
only the domain taxonomy, canonicalization, local-model tuning, and host operational tooling
described above.
