#!/usr/bin/env python3
# Purpose: durable graphml-orphan + garble prune for the oceanstack_code_schema KG.
# Removes code-class entities whose id has no referent in the current OceanStack
# source tree — i.e. symbols deleted from the codebase (stale orphans the
# SQL-purge ingest path leaves in the NetworkX graphml) and gemma transcription
# garbles (e.g. `_prepare_query_traragories` for `_prepare_query_trajectories`).
#
# Acts as the post-extraction existence gate (run from sync-from-git.sh after a
# push) and the maintenance prune (run hourly from kg-audit.sh). Deletes go
# through the LightRAG server API so the graphml and PG vector store stay
# consistent — never raw SQL.
#
# Narrative entities (concept / ais_concept / library) are exempt: they
# legitimately have no code-token referent.
#
# Usage: kg-prune-orphans.py [--apply] [--repo PATH] [--min-tokens N]
#                            [--max-delete-frac F] [--quiet]
# Default is dry-run (reports, deletes nothing). --apply performs deletions.
# Schedule: invoked by kg-audit.sh (hourly) and sync-from-git.sh (per push).

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

WS = "oceanstack_code_schema"
LR = "http://127.0.0.1:9621"
LR_DIR = Path("/fast-array/lightrag")
GRAPHML = LR_DIR / "rag-storage" / WS / "graph_chunk_entity_relation.graphml"
LOG = LR_DIR / "logs" / "prune.log"
LOCK = Path("/var/lock/lightrag-prune.lock")

# Entity types that are natural-language narrative nodes, not code symbols, so
# they carry no source-token referent by design and must never be pruned.
EXEMPT_TYPES = {"concept", "ais_concept", "library"}

# Source corpus mirrors the ingest filter in sync-from-git.sh exactly: every
# .py/.rs/.sql/.wgsl in the repo (examples/ and config/ ARE ingested), minus the
# paths sync-from-git's path_excluded() drops (tests are not ingested).
SOURCE_GLOBS = ("*.py", "*.rs", "*.sql", "*.wgsl")
SOURCE_EXCLUDES = ("tests", "target", ".venv", "node_modules", "__pycache__", "build", "dist")

# Safety floors. Abort rather than risk a mass-delete if the token set looks
# empty (wrong repo path) or the flagged fraction is implausibly large.
MIN_TOKENS = 8000
MAX_DELETE_FRAC = 0.15


def emit(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        sys.stderr.write(msg + "\n")


def log(msg: str, *, quiet: bool = False) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} [prune] {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")
    emit(line, quiet=quiet)


def read_api_key() -> str:
    out = subprocess.run(
        [
            "sops",
            "-d",
            "--input-type",
            "dotenv",
            "--output-type",
            "dotenv",
            str(LR_DIR / ".env.enc"),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={
            "SOPS_AGE_KEY_FILE": str(Path.home() / ".config/sops/age/keys.txt"),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    ).stdout
    for raw in out.splitlines():
        if raw.startswith("LIGHTRAG_API_KEY="):
            return raw.split("=", 1)[1].strip()
    raise RuntimeError("LIGHTRAG_API_KEY not found in .env.enc")


def build_source_tokens(repo: Path) -> set[str]:
    """Collect every identifier-like token in the current source tree (lowercased).

    A symbol that exists in the codebase appears here at least once; a deleted
    symbol or a transcription garble does not. Lenient by construction — false
    keeps are acceptable, false deletes are not.
    """
    if not repo.is_dir():
        return set()
    args: list[str] = []
    for g in SOURCE_GLOBS:
        args += ["-g", g]
    for ex in SOURCE_EXCLUDES:
        args += ["-g", f"!{ex}/**", "-g", f"!**/{ex}/**"]
    proc = subprocess.run(
        ["rg", "-oN", "--no-filename", r"\b[A-Za-z_][A-Za-z0-9_]*\b", *args, str(repo)],
        capture_output=True,
        text=True,
    )
    return {t.strip().lower() for t in proc.stdout.splitlines() if t.strip()}


def load_pg_entities() -> set[str]:
    """Lowercased entity_name set from the PG vector store (the live-entity set).

    Every entity LightRAG keeps has a vdb embedding row; a graphml node absent
    here is an orphan the SQL-purge ingest path left behind. This membership
    gate is what makes the prune safe: real entities whose canonical name
    differs from raw source spelling (`idx_*` -> `index_*`, PascalCase ->
    snake_case, `Result<T>`) stay because they are present in PG, even when
    their literal id has no source-token match.
    """
    table = subprocess.run(
        [
            "psql",
            "--no-psqlrc",
            "-d",
            "lightrag",
            "-tAc",
            "SELECT tablename FROM pg_tables WHERE tablename LIKE 'lightrag_vdb_entity_%' "
            "AND tablename NOT LIKE '%gemini_legacy' LIMIT 1",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not table:
        return set()
    rows = subprocess.run(
        ["psql", "--no-psqlrc", "-d", "lightrag", "-tAc", f"SELECT entity_name FROM {table} WHERE workspace='{WS}'"],
        capture_output=True,
        text=True,
    ).stdout
    return {r.strip().lower() for r in rows.splitlines() if r.strip()}


def id_segments(entity_id: str) -> list[str]:
    """Lowercased id plus each dotted / Rust-path segment."""
    parts = [entity_id.lower()]
    parts += [p.lower() for p in entity_id.replace("::", ".").split(".") if p]
    return parts


def load_graph_entities() -> list[tuple[str, str]]:
    """Return (entity_id, entity_type) for every node in the persisted graphml.

    Parsed via networkx (the graph store's own library) rather than a raw XML
    parser — the graphml is trusted local server output, and networkx avoids a
    direct stdlib-xml dependency.
    """
    import networkx as nx

    g = nx.read_graphml(GRAPHML)
    return [(str(nid), str(data.get("entity_type", "")).strip().lower()) for nid, data in g.nodes(data=True)]


def delete_entity(name: str, key: str) -> str:
    """Delete one entity through the LightRAG server API via curl.

    Uses curl (as every sibling maintenance script does) against the local-only
    server so the server updates both the graphml and the PG vector store.
    """
    body = json.dumps({"entity_name": name})
    try:
        proc = subprocess.run(
            [
                "curl",
                "-s",
                "-m",
                "20",
                "-X",
                "DELETE",
                "-H",
                f"X-API-Key: {key}",
                "-H",
                "Content-Type: application/json",
                "-d",
                body,
                f"{LR}/documents/delete_entity",
            ],
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout or "{}")
        return "ok" if payload.get("status") == "success" else "noop"
    except Exception as exc:  # noqa: BLE001 — log and continue; one failure must not abort the sweep
        return f"err:{type(exc).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform deletions (default: dry-run)")
    ap.add_argument("--repo", default=str(Path.home() / "Codebases/OceanStack"))
    ap.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    ap.add_argument("--max-delete-frac", type=float, default=MAX_DELETE_FRAC)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not GRAPHML.exists():
        log(f"graphml missing at {GRAPHML}; nothing to prune", quiet=args.quiet)
        return 0

    tokens = build_source_tokens(Path(args.repo))
    if len(tokens) < args.min_tokens:
        log(
            f"ABORT: source token set too small ({len(tokens)} < {args.min_tokens}); "
            f"repo path '{args.repo}' likely wrong — refusing to prune",
            quiet=args.quiet,
        )
        return 1

    pg = load_pg_entities()
    if not pg:
        log("ABORT: PG vdb entity set empty/unreadable — refusing to prune", quiet=args.quiet)
        return 1

    entities = load_graph_entities()
    code_entities = [(nid, et) for nid, et in entities if et not in EXEMPT_TYPES]
    # Safe prune target: graphml-only (absent from PG vdb) AND no source-token
    # referent. The PG-membership gate keeps real canonicalized entities whose
    # literal id does not match raw source spelling; the referent gate keeps
    # genuine orphans whose symbol still exists under another node.
    orphans = [(nid, et) for nid, et in code_entities if nid.lower() not in pg]
    flagged = [nid for nid, _ in orphans if all(seg not in tokens for seg in id_segments(nid))]

    n_code = len(code_entities)
    frac = (len(flagged) / n_code) if n_code else 0.0
    log(
        f"scan: graph_nodes={len(entities)} code_entities={n_code} "
        f"exempt={len(entities) - n_code} pg_entities={len(pg)} graphml_only={len(orphans)} "
        f"tokens={len(tokens)} flagged(orphan+no_referent)={len(flagged)} ({frac:.1%} of code)",
        quiet=args.quiet,
    )

    if frac > args.max_delete_frac:
        log(
            f"ABORT: flagged fraction {frac:.1%} exceeds cap {args.max_delete_frac:.0%}; "
            f"token set likely stale/incomplete — refusing to prune",
            quiet=args.quiet,
        )
        return 1

    if not flagged:
        log("clean: no no-referent code entities", quiet=args.quiet)
        return 0

    if not args.apply:
        sample = ", ".join(sorted(flagged)[:15])
        log(f"DRY-RUN: would delete {len(flagged)} entities. sample: {sample}", quiet=args.quiet)
        return 0

    key = read_api_key()
    results: Counter[str] = Counter()
    for name in flagged:
        results[delete_entity(name, key)] += 1
        time.sleep(0.1)
    errors = sum(v for k, v in results.items() if k.startswith(("http_", "err:")))
    log(
        f"APPLIED: deleted_ok={results['ok']} noop={results['noop']} errors={errors}",
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    import fcntl

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as _lock_fh:
        try:
            fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.stderr.write("kg-prune-orphans: another instance holds the lock; exiting\n")
            sys.exit(0)
        sys.exit(main())
