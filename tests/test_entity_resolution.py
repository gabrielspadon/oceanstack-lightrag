"""Unit tests for the deterministic entity-resolution core (no reasoner/LLM).

Exercises ``lightrag.entity_resolution.resolve_batch``, ``apply_name_map``,
and the ``_extract_similarity`` helper against fake graph/vdb stubs - no
real storage backend, no network, no LLM calls. ``asyncio_mode = "auto"``
is set in pyproject.toml, so ``async def test_...`` works without a
decorator.
"""

import pytest

from lightrag.entity_resolution import (
    Decision,
    _extract_similarity,
    apply_name_map,
    resolve_batch,
)

GLOBAL_CONFIG = {
    "working_dir": "/tmp/test_entity_resolution",
    "workspace": "test_ws",
    "enable_entity_resolution": True,
    "entity_resolution_auto_merge_similarity": 0.98,
    "entity_resolution_candidate_similarity": 0.85,
    "entity_resolution_use_reasoner": False,
    "entity_resolution_min_confidence": 0.80,
    "entity_resolution_allow_promote": False,
    "entity_resolution_top_k": 5,
    "entity_resolution_dry_run": True,
    "entity_resolution_max_llm_calls_per_batch": 20,
}


class FakeGraph:
    """Minimal BaseGraphStorage stub: only the methods resolve_batch calls."""

    def __init__(self, nodes):
        self.nodes = set(nodes)

    async def get_node(self, node_id):
        return {"entity_id": node_id} if node_id in self.nodes else None

    async def get_node_edges(self, source_node_id):
        return [
            (source_node_id, other) for other in self.nodes if other != source_node_id
        ]


class FakeVDB:
    """Minimal BaseVectorStorage stub: query() returns a fixed hit list."""

    def __init__(self, hits=None):
        self._hits = hits or []

    async def query(self, query, top_k, query_embedding=None):
        return list(self._hits[:top_k])

    async def delete_entity(self, entity_name):
        self._hits = [
            hit for hit in self._hits if hit.get("entity_name") != entity_name
        ]


def _node_items(name, description="a test entity"):
    return [{"entity_name": name, "entity_id": name, "description": description}]


async def test_exact_match_short_circuits_to_create_new():
    graph = FakeGraph(nodes={"widget"})
    vdb = FakeVDB(hits=[])

    result = await resolve_batch(
        {"widget": _node_items("widget")}, {}, graph, vdb, GLOBAL_CONFIG
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record.decision == Decision.CREATE_NEW
    assert record.method == "exact"
    assert result.name_map["widget"] == "widget"


async def test_namespace_guard_blocks_cross_namespace_merge():
    graph = FakeGraph(nodes={"signals.ais_position_reports"})
    vdb = FakeVDB(
        hits=[{"entity_name": "signals.ais_position_reports", "similarity": 0.99}]
    )

    result = await resolve_batch(
        {"derived.ais_position_reports": _node_items("derived.ais_position_reports")},
        {},
        graph,
        vdb,
        GLOBAL_CONFIG,
    )

    record = result.records[0]
    assert record.decision == Decision.CREATE_NEW
    assert record.method == "namespace_guard"
    assert (
        result.name_map["derived.ais_position_reports"]
        == "derived.ais_position_reports"
    )


async def test_auto_merge_discards_and_reuses_live_target():
    graph = FakeGraph(nodes={"Coffee-Breaks"})
    vdb = FakeVDB(hits=[{"entity_name": "Coffee-Breaks", "similarity": 0.99}])

    result = await resolve_batch(
        {"Coffee Breaks": _node_items("Coffee Breaks")}, {}, graph, vdb, GLOBAL_CONFIG
    )

    record = result.records[0]
    assert record.decision == Decision.DISCARD_AND_REUSE
    assert record.method == "auto_threshold"
    assert record.target_name == "Coffee-Breaks"
    assert result.name_map["Coffee Breaks"] == "Coffee-Breaks"


async def test_residue_mismatch_defers_to_reasoner_band_despite_high_similarity():
    graph = FakeGraph(nodes={"OceanStack-core"})
    vdb = FakeVDB(hits=[{"entity_name": "OceanStack-core", "similarity": 0.99}])

    result = await resolve_batch(
        {"OceanStack": _node_items("OceanStack")}, {}, graph, vdb, GLOBAL_CONFIG
    )

    record = result.records[0]
    assert record.decision == Decision.CREATE_NEW
    assert record.method == "reasoner_band_deferred"
    assert result.name_map["OceanStack"] == "OceanStack"


async def test_stale_vdb_hit_falls_back_to_create_new():
    # "ghost-entity" scores high enough (and is residue-equal) to auto-merge,
    # but it is not actually a live node in the graph - the vdb is stale.
    graph = FakeGraph(nodes=set())
    vdb = FakeVDB(hits=[{"entity_name": "ghost-entity", "similarity": 0.99}])

    result = await resolve_batch(
        {"ghost entity": _node_items("ghost entity")}, {}, graph, vdb, GLOBAL_CONFIG
    )

    record = result.records[0]
    assert record.decision == Decision.CREATE_NEW
    assert record.method == "stale_vdb"


async def test_below_threshold_creates_new():
    graph = FakeGraph(nodes={"unrelated"})
    vdb = FakeVDB(hits=[{"entity_name": "unrelated", "similarity": 0.10}])

    result = await resolve_batch(
        {"widget": _node_items("widget")}, {}, graph, vdb, GLOBAL_CONFIG
    )

    record = result.records[0]
    assert record.decision == Decision.CREATE_NEW
    assert record.method == "below_threshold"


def test_extract_similarity_reads_every_backend_key_shape():
    assert _extract_similarity({"distance": 0.1}) == pytest.approx(0.9)
    assert _extract_similarity({"similarity": 0.8}) == pytest.approx(0.8)
    assert _extract_similarity({"__metrics__": 0.7}) == pytest.approx(0.7)
    assert _extract_similarity({"entity_name": "x", "created_at": 123}) is None


def test_apply_name_map_merges_collisions_and_drops_self_loops():
    name_map = {"alpha": "canonical", "beta": "canonical"}
    nodes_by_name = {
        "alpha": [{"entity_name": "alpha", "entity_id": "alpha", "description": "a"}],
        "beta": [{"entity_name": "beta", "entity_id": "beta", "description": "b"}],
    }
    edges_by_pair = {
        ("alpha", "gamma"): [
            {"src_id": "alpha", "tgt_id": "gamma", "description": "edge1"}
        ],
        ("alpha", "beta"): [
            {"src_id": "alpha", "tgt_id": "beta", "description": "self-loop-after-map"}
        ],
    }

    new_nodes, new_edges = apply_name_map(name_map, nodes_by_name, edges_by_pair)

    assert set(new_nodes) == {"canonical"}
    assert len(new_nodes["canonical"]) == 2
    assert all(item["entity_name"] == "canonical" for item in new_nodes["canonical"])
    assert all(item["entity_id"] == "canonical" for item in new_nodes["canonical"])

    assert ("alpha", "beta") not in new_edges
    remapped_key = tuple(sorted(("canonical", "gamma")))
    assert remapped_key in new_edges
    assert len(new_edges[remapped_key]) == 1
    assert new_edges[remapped_key][0]["src_id"] == "canonical"
    assert new_edges[remapped_key][0]["tgt_id"] == "gamma"


async def test_none_similarity_pg_case_lands_in_reasoner_band():
    # PG entities vdb returns no similarity score today; _extract_similarity -> None.
    # Must land in reasoner_band_deferred (CREATE_NEW), never error_fallback/crash.
    graph = FakeGraph(nodes={"Existing Widget"})
    vdb = FakeVDB(
        hits=[{"entity_name": "Existing Widget"}]
    )  # no distance/similarity/__metrics__
    res = await resolve_batch(
        {"New Widget": _node_items("New Widget")}, {}, graph, vdb, GLOBAL_CONFIG
    )
    rec = res.records[0]
    assert rec.decision is Decision.CREATE_NEW
    assert rec.method == "reasoner_band_deferred"
    assert res.name_map["New Widget"] == "New Widget"
