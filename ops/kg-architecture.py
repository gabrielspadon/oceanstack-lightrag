"""Inject the curated OceanStack architecture backbone into the code knowledge graph.

The deterministic structural spine captures symbol-level edges (calls, defines,
has_column). This layer adds the cross-cutting architectural relations that no
single-file parser can see: derive-phase ordering, the binary-COPY wiring
contract, the GPU dispatch chain, the four-layer stack, the cross-layer truth
hierarchy, and the ingestion data flow. Edges connect existing canonicalised
nodes; a small set of anchor concept nodes is created where the relation has no
code symbol to attach to.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import networkx as nx
from lightrag.operate import _canonical_entity_name as canon

GRAPH = "/fast-array/lightrag/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
SRC = "architecture-backbone"

g = nx.read_graphml(GRAPH)
names = set(g.nodes())


def resolve(name: str) -> str | None:
    """Return the actual graph node id for a logical name, trying canon + variants."""
    cands = [name, canon(name), name.lower(), name.replace("-", "_"), name.replace(".", "_")]
    for c in cands:
        if c in names:
            return c
    return None


anchors_added = 0
edges_added = 0
edges_skipped = 0


def add_anchor(nid: str, etype: str, desc: str) -> str:
    global anchors_added
    if nid not in names:
        g.add_node(
            nid,
            entity_id=nid,
            entity_type=etype,
            description=desc,
            source_id=SRC,
            file_path="architecture",
            created_at=NOW,
        )
        names.add(nid)
        anchors_added += 1
    return nid


def add_edge(a: str, b: str, predicate: str, desc: str) -> None:
    """Add an architectural edge between two logical node names; resolve both first."""
    global edges_added, edges_skipped
    ra, rb = resolve(a), resolve(b)
    if ra is None or rb is None:
        edges_skipped += 1
        return
    if g.has_edge(ra, rb):
        e = g[ra][rb]
        kw = {k.strip() for k in str(e.get("keywords", "")).split(",") if k.strip()}
        kw.add(predicate)
        e["keywords"] = ",".join(sorted(kw))
    else:
        g.add_edge(
            ra,
            rb,
            weight=9.0,
            keywords=predicate,
            description=desc,
            source_id=SRC,
            file_path="architecture",
            created_at=NOW,
        )
        edges_added += 1


# ---- anchor concepts ----
L1 = add_anchor(
    "layer_l1_postgresql",
    "concept",
    "L1 storage layer: PostgreSQL 18 with TimescaleDB hypertables, PostGIS geography, pg_h3.",
)
L2 = add_anchor("layer_l2_rust", "concept", "L2 compute layer: oceanstack-core SIMD f64x4 + Rayon, PyO3 FFI bindings.")
L3 = add_anchor(
    "layer_l3_python", "concept", "L3 orchestration layer: Polars + Arrow FFI, ingestion and derive pipelines."
)
L4 = add_anchor("layer_l4_gpu", "concept", "L4 acceleration layer: optional wgpu WGSL shaders with CuPy/CUDA fallback.")
DERIVE = add_anchor(
    "derive_pipeline",
    "concept",
    "Nine-phase derived-data pipeline driven by scripts/data-sync/derive.py over signals.ais_position_reports.",
)
DISPATCH = add_anchor(
    "gpu_dispatch_chain",
    "concept",
    "Compute dispatch chain: GPU (>=GPU_MIN_PROBLEM) falls back to SIMD (>=SIMD_MIN_PROBLEM) falls back to Rayon scalar.",
)
WIRING = add_anchor(
    "binary_copy_contract",
    "concept",
    "Binary COPY wiring invariant: column set must match across the position-report DDL, columns.py, and write_row_to_buffer.",
)
TRUTH = add_anchor(
    "cross_layer_truth_hierarchy",
    "concept",
    "Data-contract precedence: SQL schemas > constants.rs > FFI bindings > Python > type stubs.",
)
FLOW = add_anchor(
    "ingestion_data_flow",
    "concept",
    "Raw files flow through FormatDetector, format adapters, AISRecord, BulkLoader, the Rust PGCOPY buffer, staging, into the hypertable.",
)

# ---- layer stack ----
add_edge(L4, L2, "depends_on", "GPU acceleration layer dispatches from the Rust compute layer.")
add_edge(L3, L2, "depends_on", "Python orchestration calls into Rust compute via PyO3.")
add_edge(L3, L1, "depends_on", "Python orchestration reads and writes PostgreSQL storage.")
add_edge(L2, L1, "depends_on", "Rust compute reads from and writes binary COPY buffers to PostgreSQL.")
add_edge("oceanstack-core", L2, "bound_to", "oceanstack-core is the L2 compute crate.")
add_edge("bulk_loader", L3, "bound_to", "BulkLoader is the L3 ingestion orchestrator.")

# ---- derive phase order ----
phases = [
    "refresh_vessel_registry",
    "enrich_vessel_registry_metadata",
    "populate_vessel_tracks",
    "enrich_track_h3_distance",
    "enrich_track_behaviors",
    "enrich_track_ports",
]
for a, b in zip(phases, phases[1:]):
    add_edge(a, b, "precedes", f"{a} runs before {b} in the derive pipeline.")
for p in phases:
    add_edge(DERIVE, p, "orchestrates", f"The derive pipeline invokes {p}.")
add_edge(
    "populate_vessel_tracks",
    "derived.vessel_tracks",
    "writes_to",
    "POPULATE_VESSEL_TRACKS materialises derived.vessel_tracks.",
)
add_edge(
    "refresh_vessel_registry",
    "derived.vessel_state",
    "writes_to",
    "REFRESH_VESSEL_REGISTRY materialises derived.vessel_state.",
)
add_edge(
    "populate_vessel_tracks",
    "signals.ais_position_reports",
    "reads_from",
    "Track population reads raw position reports.",
)

# ---- binary COPY contract ----
trio = ["signals.ais_position_reports", "binary_copy_columns", "write_row_to_buffer"]
for n in trio + ["generate_binary_copy_buffer"]:
    add_edge(WIRING, n, "binds", f"{n} participates in the binary COPY wiring contract.")
add_edge(
    "binary_copy_columns",
    "signals.ais_position_reports",
    "must_sync_with",
    "Column order must match the hypertable DDL.",
)
add_edge("binary_copy_columns", "write_row_to_buffer", "must_sync_with", "Column order must match the Rust row writer.")
add_edge(
    "generate_binary_copy_buffer", "write_row_to_buffer", "calls", "The FFI entry point invokes the per-row writer."
)

# ---- dispatch chain ----
for thr in ["gpu_min_problem", "simd_min_problem", "gil_release_threshold"]:
    add_edge(DISPATCH, thr, "gated_by", f"Dispatch tier selection is gated by {thr}.")

# ---- truth hierarchy ----
add_edge(TRUTH, "signals.ais_position_reports", "ranks_first", "SQL schemas are the top data-contract authority.")
add_edge(TRUTH, "binary_copy_columns", "ranks", "Python column wiring conforms to the schema.")
add_edge(TRUTH, "write_row_to_buffer", "ranks", "Rust FFI conforms to schema and constants.")

# ---- ingestion data flow ----
flow = [
    "format_detector",
    "ais_data_adapter",
    "ais_record",
    "bulk_loader",
    "generate_binary_copy_buffer",
    "signals.ais_position_reports",
]
for a, b in zip(flow, flow[1:]):
    add_edge(a, b, "flows_to", f"Ingestion data flows from {a} to {b}.")
add_edge(FLOW, "format_detector", "begins_at", "The ingestion flow starts at format detection.")

nx.write_graphml(g, GRAPH)
iso = sum(1 for n in g.nodes() if g.degree(n) == 0)
comps = list(nx.connected_components(g.to_undirected()))
largest = max((len(c) for c in comps), default=0)
sys.stdout.write(
    f"anchors_added={anchors_added} edges_added={edges_added} edges_skipped={edges_skipped}\n"
    f"nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
    f"density={g.number_of_edges() / max(g.number_of_nodes(), 1):.2f}\n"
    f"orphans={iso} ({100 * iso / max(g.number_of_nodes(), 1):.1f}%) "
    f"components={len(comps)} largest={100 * largest / max(g.number_of_nodes(), 1):.1f}%\n"
)
