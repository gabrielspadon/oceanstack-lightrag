# oceanstack-lightrag

Private fork of [LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS) carrying the
OceanStack maritime/AIS code-knowledge-graph modifications and the host operational
scripts that run the LightRAG server on the RTX-4090. Upstream's original README is
preserved in git history at the baseline commit.

## Fork point

|                | |
| -------------- | --------------------------------------------------------------- |
| Upstream       | https://github.com/HKUDS/LightRAG                               |
| Pinned tag     | `v1.4.15`                                                       |
| Pinned commit  | `64d3326f858db300ec7699b2cb84e4edd5e5869e`                     |
| Package        | `lightrag-hku` (`__version__ = "1.4.15"`, `__api_version__ = "0287"`) |

The baseline commit (tag `v1.4.15-baseline`) is a copy of pristine upstream `v1.4.15`
with local build artifacts stripped. Every OceanStack change is layered on top, so the
fork delta is always `git diff v1.4.15-baseline..HEAD`.

## Why this repo exists

The OceanStack parent repo (`~/Codebases/OceanStack`) vendors upstream LightRAG as a
read-only submodule at `external/lightrag`, pinned to `v1.4.15` and installed pristine
via `uv`. That submodule stays untouched — four gates enforce it (`.claude/rules/external.md`).

The LightRAG server that actually runs on the host (`/fast-array/lightrag/.venv`) is
patched in place to teach the extractor the maritime/AIS code taxonomy and to drive the
local Ollama models correctly. Those patches previously lived only in the live venv,
untracked. This repo versions them against a known upstream commit, so upstream
improvements can be tracked and replayed.

## Repository layout

```
oceanstack-lightrag/
├── lightrag/              # upstream package; baseline = pristine v1.4.15, mods on top
│   ├── operate.py         # MODIFIED — entity/relation extraction: canonicalizer,
│   │                      #   22-type taxonomy remap, <|#|> field recovery, ghost typing
│   ├── prompt.py          # MODIFIED — 22-type code/DB/GPU/AIS extraction taxonomy,
│   │                      #   verb-only relation keywords, strict output format
│   └── llm/ollama.py      # MODIFIED — qwen3 /think toggle keyed on system-prompt
│                          #   signature (think=False extraction, True for RAG answers)
└── ops/                   # host scripts from /fast-array/lightrag (NOT upstream)
    ├── start.sh                    # boot the LightRAG server (sops-decrypt env)
    ├── inotify-ingest.sh           # watch inputs/, ingest code/schema on save
    ├── retry-watcher.sh            # replay failed ingests
    ├── reconcile.sh                # KG/SQL drift reconciliation
    ├── reset-workspace.sh          # wipe + rebuild a workspace
    ├── sync-from-git.sh            # pull patched sources into the live venv
    ├── kg-audit.sh / kg-rebuild-spine.sh
    ├── lightrag_mcp.py             # MCP server bridge
    ├── batch_ingest.py / dedup-entities.py
    ├── kg-*.py                     # KG maintenance: structural extract/inject, embed
    │                               #   missing, normalize types/predicates, prune
    │                               #   orphans, quality gates, aliases, architecture
    ├── schema/content_sha256.sql   # content-hash dedup helper
    ├── .env.enc                    # sops-encrypted host config (ciphertext only)
    └── .age-recipient              # public age recipient for the encrypted env
```

The split is deliberate: `lightrag/` tracks upstream-plus-delta so it can be diffed and
rebased against new releases; `ops/` holds host glue with no upstream counterpart and
never participates in a rebase.

## Multi-project knowledge graphs

The same structure serves N isolated knowledge graphs. A **project** is one KG:
a distinct LightRAG workspace + server port, all sharing the host PostgreSQL
`lightrag` database — LightRAG tags every row with a `workspace` column and a
composite primary key `(workspace, id)`, so projects never collide. NetworkX
graphml lands under `WORKING_DIR/<workspace>/`, one subdirectory per project.

Per-project settings live in `ops/projects/<name>.env`; shared secrets stay in
the sops `ops/.env.enc`. `ops/lib/project-env.sh` resolves the active project
(default `code`) and exports its workspace/port/storage.

| Project | Workspace                | Port   | Fed by                                  |
| ------- | ------------------------ | ------ | --------------------------------------- |
| `code`  | `oceanstack_code_schema` | `9621` | `inotify-ingest.sh` (source/schema files) |
| `ships` | `oceanstack_ships`       | `9622` | OceanStack derive bridge (HTTP inserts) |

The LightRAG server is one-workspace-per-process, so each project is a separate
server instance. Run one with the templated unit:

```bash
# install the template once
cp ops/systemd/lightrag@.service ~/.config/systemd/user/
systemctl --user daemon-reload

# start the ships KG (reads projects/ships.env -> workspace oceanstack_ships, :9622)
systemctl --user start lightrag@ships
# the original code KG keeps running as lightrag.service / lightrag@code on :9621
```

Or directly: `PROJECT=ships ./ops/start.sh`. Add a project by dropping a new
`ops/projects/<name>.env` (set `WORKSPACE` and a free `PORT`) — no script edits.

## Secrets

`ops/.env.enc` is sops-encrypted at rest (age recipient in `ops/.age-recipient`). The age
private key is NOT in this repo — it lives at `~/.config/sops/age/keys.txt` on the host.
No decrypted secrets, API keys, or private keys are committed. Decrypt with:

```bash
SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt sops -d ops/.env.enc
```

## Pulling upstream improvements

```bash
# one-time: wire the upstream remote
git remote add upstream https://github.com/HKUDS/LightRAG.git
git fetch upstream --tags

# what upstream changed since our fork point
git diff v1.4.15-baseline..<new-tag> -- lightrag/

# our delta in isolation (what must replay)
git diff v1.4.15-baseline..HEAD -- lightrag/
```

### Rebase onto a new release

```bash
git checkout -b bump/<new-tag> <new-tag>
git rebase --onto <new-tag> v1.4.15-baseline main
# conflicts land ONLY in lightrag/operate.py, prompt.py, llm/ollama.py
git tag <new-tag>-baseline <new-tag>
```

`ops/` never conflicts (no upstream counterpart). After the bump, sync the patched
sources into the live host venv:

```bash
# on the RTX-4090 host
bash ops/sync-from-git.sh
```

and bump the parent OceanStack submodule pin separately via `just rag-update <tag>` (the
parent repo tracks pristine upstream, not this fork).
