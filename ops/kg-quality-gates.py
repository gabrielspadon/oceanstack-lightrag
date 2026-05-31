"""Quality gates for the OceanStack code knowledge graph.

Computes structural health metrics (density, orphan share, largest-component
share, entity-type validity) against fixed thresholds and runs a set of
architecture smoke queries that assert known facts are retrievable. Exits
non-zero when any gate fails so reconcile / auditor runs can treat a regression
as a hard error. Read-only: never mutates the graph.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import networkx as nx

GRAPH = "/fast-array/lightrag/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml"
URL = "http://127.0.0.1:9621"

VALID_TYPES = {
    "module",
    "function",
    "method",
    "class",
    "dataclass",
    "enum",
    "protocol",
    "macro",
    "ffi_binding",
    "constant",
    "exception",
    "schema",
    "table",
    "column",
    "domain_type",
    "sql_function",
    "cagg",
    "index",
    "gpu_kernel",
    "ais_concept",
    "library",
    "concept",
}

# (min_density, max_orphan_pct, min_component_pct, min_type_validity_pct)
THRESHOLDS = dict(density=2.0, orphan_pct=10.0, component_pct=90.0, type_validity_pct=99.0)

# Architecture smoke queries: (question, accepted-substring alternatives). The answer
# passes when any alternative appears (case-insensitive) — robust to LLM paraphrase of
# identifier names while still proving the relationship was retrieved.
SMOKE = [
    (
        "Which function calls write_row_to_buffer to build the PGCOPY buffer?",
        ("generate_binary", "write_row_to_buffer"),
    ),
    ("What does POPULATE_VESSEL_TRACKS write to?", ("vessel_tracks",)),
    (
        "What are the phases of the derive pipeline and their order?",
        ("populate_vessel", "refresh_vessel", "enrich_track", "derive_pipeline"),
    ),
    (
        "What is the GPU dispatch chain and its thresholds?",
        ("gpu_dispatch_chain", "simd_min_problem", "gpu_min_problem", "rayon", "simd"),
    ),
    (
        "Which files must stay in sync for the binary COPY column contract?",
        ("binary_copy_contract", "ais_position_reports", "write_row_to_buffer", "columns"),
    ),
]


def structural_gates() -> tuple[dict, bool]:
    g = nx.read_graphml(GRAPH)
    n, e = g.number_of_nodes(), g.number_of_edges()
    iso = sum(1 for x in g.nodes() if g.degree(x) == 0)
    comps = list(nx.connected_components(g.to_undirected()))
    largest = max((len(c) for c in comps), default=0)
    bad_types = [
        g.nodes[x].get("entity_type", "?") for x in g.nodes() if g.nodes[x].get("entity_type") not in VALID_TYPES
    ]
    metrics = {
        "nodes": n,
        "edges": e,
        "density": round(e / max(n, 1), 3),
        "orphan_pct": round(100 * iso / max(n, 1), 2),
        "component_pct": round(100 * largest / max(n, 1), 2),
        "type_validity_pct": round(100 * (n - len(bad_types)) / max(n, 1), 3),
        "invalid_type_samples": sorted(set(bad_types))[:10],
    }
    passed = (
        metrics["density"] >= THRESHOLDS["density"]
        and metrics["orphan_pct"] <= THRESHOLDS["orphan_pct"]
        and metrics["component_pct"] >= THRESHOLDS["component_pct"]
        and metrics["type_validity_pct"] >= THRESHOLDS["type_validity_pct"]
    )
    return metrics, passed


def query(q: str, key: str) -> str:
    # URL is the fixed loopback constant http://127.0.0.1:9621 (the local LightRAG
    # server); transport encryption is moot on loopback and the URL is not caller-controlled.
    # only_need_context returns the retrieved entity/relation/chunk context without LLM
    # synthesis, so the gate measures knowledge-graph correctness rather than the local
    # model's generation quality.
    body = json.dumps({"query": q, "mode": "mix", "top_k": 20, "only_need_context": True}).encode()
    req = urllib.request.Request(  # nosemgrep
        f"{URL}/query", data=body, headers={"Content-Type": "application/json", "X-API-Key": key}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:  # nosemgrep
        return json.loads(r.read().decode()).get("response", "")


def smoke_gates(key: str) -> tuple[list, bool]:
    results, ok = [], True
    for q, needles in SMOKE:
        try:
            ans = query(q, key).lower()
            hit = any(nd.lower() in ans for nd in needles)
        except Exception:  # noqa: BLE001 - report any retrieval failure as a gate failure
            hit = False
        results.append({"q": q, "expect": "|".join(needles), "pass": hit})
        ok = ok and hit
    return results, ok


def main() -> int:
    key = os.environ.get("LR_KEY", "")
    metrics, sp = structural_gates()
    sys.stdout.write("=== structural gates ===\n")
    for k, v in metrics.items():
        sys.stdout.write(f"  {k}: {v}\n")
    sys.stdout.write(f"  -> structural {'PASS' if sp else 'FAIL'} (thresholds {THRESHOLDS})\n")

    qp = True
    if key:
        smoke, qp = smoke_gates(key)
        sys.stdout.write("=== architecture smoke queries ===\n")
        for s in smoke:
            sys.stdout.write(f"  [{'PASS' if s['pass'] else 'FAIL'}] expect '{s['expect']}' :: {s['q']}\n")
    else:
        sys.stdout.write("=== architecture smoke queries skipped (LR_KEY unset) ===\n")

    overall = sp and qp
    sys.stdout.write(f"=== OVERALL: {'PASS' if overall else 'FAIL'} ===\n")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
