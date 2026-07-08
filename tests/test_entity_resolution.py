"""Unit tests for the entity-resolution layer (deterministic gates + reasoner).

Exercises ``lightrag.entity_resolution`` against fake graph/vdb/LLM stubs - no
real storage backend, no network, no GPU. ``asyncio_mode = "auto"`` is set in
pyproject.toml, so ``async def test_...`` works without a decorator.
"""

from collections import defaultdict

import pytest

from lightrag.entity_resolution import (
    Decision,
    _capture_promote_plan,
    _extract_similarity,
    _is_suffix_variant,
    apply_name_map,
    apply_promotions,
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


# --- deterministic gates -------------------------------------------------


async def test_exact_match_short_circuits_to_create_new():
    graph = FakeGraph(nodes={"widget"})
    vdb = FakeVDB(hits=[])
    result = await resolve_batch(
        {"widget": _node_items("widget")}, {}, graph, vdb, GLOBAL_CONFIG
    )
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


async def test_suffix_variant_never_merges_even_at_high_similarity():
    # HARD rule: OceanStack must never merge onto OceanStack-core. Same (empty)
    # namespace + similarity 0.99, but the deterministic variant guard keeps
    # them distinct without ever consulting the reasoner.
    graph = FakeGraph(nodes={"OceanStack-core"})
    vdb = FakeVDB(hits=[{"entity_name": "OceanStack-core", "similarity": 0.99}])
    result = await resolve_batch(
        {"OceanStack": _node_items("OceanStack")}, {}, graph, vdb, GLOBAL_CONFIG
    )
    record = result.records[0]
    assert record.decision == Decision.CREATE_NEW
    assert record.method == "variant_guard"
    assert result.name_map["OceanStack"] == "OceanStack"


def test_is_suffix_variant_rules():
    assert _is_suffix_variant("OceanStack", "OceanStack-core") is True
    assert _is_suffix_variant("Model", "Model-v2") is True
    assert _is_suffix_variant("New York City", "NYC") is False  # neither a prefix
    assert (
        _is_suffix_variant("Coffee Breaks", "Coffee-Breaks") is False
    )  # equal residue
    assert _is_suffix_variant("widget", "unrelated") is False


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


async def test_stale_vdb_hit_falls_back_to_create_new():
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


async def test_scoreless_hit_routes_to_reasoner_band():
    # A backend hit with no similarity key -> None -> reasoner band (deferred
    # here since GLOBAL_CONFIG has the reasoner off). Never error_fallback.
    graph = FakeGraph(nodes={"Existing Widget"})
    vdb = FakeVDB(hits=[{"entity_name": "Existing Widget"}])
    res = await resolve_batch(
        {"New Widget": _node_items("New Widget")}, {}, graph, vdb, GLOBAL_CONFIG
    )
    rec = res.records[0]
    assert rec.decision is Decision.CREATE_NEW
    assert rec.method == "reasoner_band_deferred"
    assert res.name_map["New Widget"] == "New Widget"


def test_apply_name_map_merges_collisions_and_drops_self_loops():
    name_map = {"alpha": "canonical", "beta": "canonical"}
    nodes_by_name = {
        "alpha": [{"entity_name": "alpha", "entity_id": "alpha", "description": "a"}],
        "beta": [{"entity_name": "beta", "entity_id": "beta", "description": "b"}],
    }
    edges_by_pair = {
        ("alpha", "gamma"): [
            {"src_id": "alpha", "tgt_id": "gamma", "description": "e1"}
        ],
        ("alpha", "beta"): [
            {"src_id": "alpha", "tgt_id": "beta", "description": "self"}
        ],
    }
    new_nodes, new_edges = apply_name_map(name_map, nodes_by_name, edges_by_pair)
    assert set(new_nodes) == {"canonical"}
    assert len(new_nodes["canonical"]) == 2
    assert all(item["entity_name"] == "canonical" for item in new_nodes["canonical"])
    assert ("alpha", "beta") not in new_edges
    remapped_key = tuple(sorted(("canonical", "gamma")))
    assert remapped_key in new_edges
    assert new_edges[remapped_key][0]["src_id"] == "canonical"


def test_apply_name_map_accepts_defaultdict_input():
    # The operate.py hook passes the batch's defaultdict(list) maps; apply_name_map
    # must return plain dicts usable by the downstream .items()/len()-only path.
    name_map = {"Coffee Breaks": "Coffee-Breaks"}
    nodes = defaultdict(list)
    nodes["Coffee Breaks"] = _node_items("Coffee Breaks")
    edges = defaultdict(list)
    new_nodes, new_edges = apply_name_map(name_map, nodes, edges)
    assert isinstance(new_nodes, dict)
    assert list(new_nodes.keys()) == ["Coffee-Breaks"]
    assert len(new_nodes) == 1


# --- reasoner ------------------------------------------------------------


class FakeLLM:
    """Records call count; returns a fixed reply for the reasoner prompt."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def __call__(self, prompt, system_prompt=None, **kwargs):
        self.calls += 1
        return self.reply


def _reasoner_config(llm, **overrides):
    cfg = dict(GLOBAL_CONFIG)
    cfg["entity_resolution_use_reasoner"] = True
    cfg["llm_model_func"] = llm
    cfg.update(overrides)
    return cfg


# Non-variant near-duplicate that legitimately enters the reasoner band:
# "NYC" vs live "New York City" (residue-unequal, neither a prefix of the
# other, similarity 0.99). This is NOT a suffix variant, so the guard admits it.
def _reasoner_band_fixtures():
    graph = FakeGraph(nodes={"New York City"})
    vdb = FakeVDB(hits=[{"entity_name": "New York City", "similarity": 0.99}])
    nodes = {"NYC": _node_items("NYC")}
    return graph, vdb, nodes


async def test_reasoner_discard_and_reuse():
    graph, vdb, nodes = _reasoner_band_fixtures()
    llm = FakeLLM(
        '{"decision":"discard_and_reuse","target":"New York City",'
        '"confidence":0.95,"rationale":"same city"}'
    )
    res = await resolve_batch(nodes, {}, graph, vdb, _reasoner_config(llm))
    rec = res.records[0]
    assert rec.decision is Decision.DISCARD_AND_REUSE
    assert rec.method == "reasoner_discard"
    assert rec.target_name == "New York City"
    assert res.name_map["NYC"] == "New York City"
    assert res.llm_calls == 1
    assert llm.calls == 1


async def test_reasoner_create_new():
    graph, vdb, nodes = _reasoner_band_fixtures()
    llm = FakeLLM(
        '{"decision":"create_new","target":null,"confidence":0.9,'
        '"rationale":"distinct thing"}'
    )
    res = await resolve_batch(nodes, {}, graph, vdb, _reasoner_config(llm))
    rec = res.records[0]
    assert rec.decision is Decision.CREATE_NEW
    assert rec.method == "reasoner_create"
    assert res.name_map["NYC"] == "NYC"


async def test_reasoner_malformed_reply_fails_safe():
    graph, vdb, nodes = _reasoner_band_fixtures()
    llm = FakeLLM("the model forgot to emit json")
    res = await resolve_batch(nodes, {}, graph, vdb, _reasoner_config(llm))
    rec = res.records[0]
    assert rec.decision is Decision.CREATE_NEW
    assert rec.method == "malformed_llm"


async def test_reasoner_low_confidence_fails_safe():
    graph, vdb, nodes = _reasoner_band_fixtures()
    llm = FakeLLM(
        '{"decision":"discard_and_reuse","target":"New York City",'
        '"confidence":0.5,"rationale":"unsure"}'
    )
    res = await resolve_batch(nodes, {}, graph, vdb, _reasoner_config(llm))
    rec = res.records[0]
    assert rec.decision is Decision.CREATE_NEW
    assert rec.method == "low_confidence"
    assert res.name_map["NYC"] == "NYC"


async def test_reasoner_off_list_target_fails_safe():
    graph, vdb, nodes = _reasoner_band_fixtures()
    llm = FakeLLM(
        '{"decision":"discard_and_reuse","target":"Unlisted Thing",'
        '"confidence":0.95,"rationale":"hallucinated target"}'
    )
    res = await resolve_batch(nodes, {}, graph, vdb, _reasoner_config(llm))
    rec = res.records[0]
    assert rec.decision is Decision.CREATE_NEW
    assert rec.method == "malformed_llm"


async def test_reasoner_variant_target_is_rejected():
    # Defence-in-depth: even if a candidate slipped through and the reasoner
    # returned a suffix variant, the reasoner-side guard keeps them distinct.
    # Here "Model" vs live "Model" won't enter the band, so drive _is_suffix_variant
    # indirectly via a scoreless hit that the reasoner tries to merge onto a variant.
    graph = FakeGraph(nodes={"Model-v2"})
    vdb = FakeVDB(hits=[{"entity_name": "Model-v2", "similarity": 0.99}])
    llm = FakeLLM(
        '{"decision":"discard_and_reuse","target":"Model-v2",'
        '"confidence":0.99,"rationale":"same model"}'
    )
    res = await resolve_batch(
        {"Model": _node_items("Model")}, {}, graph, vdb, _reasoner_config(llm)
    )
    rec = res.records[0]
    # "Model" vs "Model-v2" is a suffix variant -> filtered before the reasoner,
    # so it never merges (variant_guard), and the reasoner is not even consulted.
    assert rec.decision is Decision.CREATE_NEW
    assert rec.method == "variant_guard"
    assert llm.calls == 0


async def test_reasoner_call_cap_leaves_decision_deferred():
    graph, vdb, nodes = _reasoner_band_fixtures()
    llm = FakeLLM(
        '{"decision":"discard_and_reuse","target":"New York City",'
        '"confidence":0.95,"rationale":"same"}'
    )
    cfg = _reasoner_config(llm, entity_resolution_max_llm_calls_per_batch=0)
    res = await resolve_batch(nodes, {}, graph, vdb, cfg)
    rec = res.records[0]
    assert rec.method == "reasoner_band_deferred"
    assert llm.calls == 0
    assert res.llm_calls == 0


# --- PROMOTE executor ----------------------------------------------------


class PromoteGraph:
    """Mutable duck-typed BaseGraphStorage for promote tests."""

    def __init__(self, nodes, edges=None):
        self.nodes = {k: dict(v) for k, v in nodes.items()}
        self.edges = {}
        for src, tgt, data in edges or []:
            self.edges[(src, tgt)] = dict(data)

    async def get_node(self, node_id):
        return dict(self.nodes[node_id]) if node_id in self.nodes else None

    async def get_node_edges(self, source_node_id):
        return [(a, b) for (a, b) in self.edges if source_node_id in (a, b)]

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, data):
        self.edges[(src, tgt)] = dict(data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {k: v for k, v in self.edges.items() if node_id not in k}


class PromoteVDB:
    def __init__(self, hits):
        self.hits = hits
        self.deleted = []

    async def query(self, query, top_k, query_embedding=None):
        return list(self.hits[:top_k])

    async def delete_entity(self, entity_name):
        self.deleted.append(entity_name)


class PromoteKV:
    def __init__(self, data=None):
        self.data = dict(data or {})

    async def get_by_id(self, id):
        return self.data.get(id)

    async def upsert(self, data):
        self.data.update(data)

    async def delete(self, ids):
        for i in ids:
            self.data.pop(i, None)


def _promote_fixtures():
    # Existing node "NYC" is promoted onto the better canonical name "New York
    # City" (non-variant, so it reaches the reasoner). Carries one edge.
    graph = PromoteGraph(
        nodes={"NYC": {"description": "the city", "entity_type": "place"}},
        edges=[("NYC", "Gabriel Spadon", {"description": "lived in"})],
    )
    vdb = PromoteVDB(hits=[{"entity_name": "NYC", "similarity": 0.99}])
    nodes = {"New York City": _node_items("New York City")}
    return graph, vdb, nodes


async def test_reasoner_promote_builds_plan(tmp_path):
    graph, vdb, nodes = _promote_fixtures()
    llm = FakeLLM(
        '{"decision":"promote","target":"NYC",'
        '"confidence":0.95,"rationale":"full name is canonical"}'
    )
    cfg = _reasoner_config(
        llm, entity_resolution_allow_promote=True, working_dir=str(tmp_path)
    )
    res = await resolve_batch(nodes, {}, graph, vdb, cfg)
    rec = res.records[0]
    assert rec.decision is Decision.PROMOTE
    assert rec.method == "reasoner_promote"
    assert rec.target_name == "NYC"
    assert len(res.promote_plans) == 1
    plan = res.promote_plans[0]
    assert plan.old_name == "NYC"
    assert plan.new_name == "New York City"
    assert plan.old_edges
    assert res.name_map["New York City"] == "New York City"


async def test_apply_promotions_rehomes_and_is_idempotent(tmp_path):
    graph, vdb, nodes = _promote_fixtures()
    llm = FakeLLM(
        '{"decision":"promote","target":"NYC","confidence":0.95,"rationale":"canonical"}'
    )
    cfg = _reasoner_config(
        llm, entity_resolution_allow_promote=True, working_dir=str(tmp_path)
    )
    res = await resolve_batch(nodes, {}, graph, vdb, cfg)
    kv = PromoteKV(data={"NYC": {"chunk_ids": ["c1", "c2"], "count": 2}})

    applied = await apply_promotions(res.promote_plans, nodes, graph, vdb, cfg, kv)
    assert applied == 1
    assert await graph.get_node("NYC") is None
    assert "NYC" in vdb.deleted
    assert await graph.get_edge("New York City", "Gabriel Spadon") is not None
    assert any(item.get("description") == "the city" for item in nodes["New York City"])
    assert "NYC" not in kv.data
    assert kv.data["New York City"]["chunk_ids"] == ["c1", "c2"]
    undo = tmp_path / "entity_resolution_promote_undo.jsonl"
    assert undo.exists() and "NYC" in undo.read_text()

    applied_again = await apply_promotions(
        res.promote_plans, nodes, graph, vdb, cfg, kv
    )
    assert applied_again == 0


async def test_capture_promote_plan_none_for_stale_target():
    graph = PromoteGraph(nodes={}, edges=[])
    plan = await _capture_promote_plan("ghost", "new-name", graph)
    assert plan is None


async def test_promote_disabled_downgrades_to_discard(tmp_path):
    graph, vdb, nodes = _promote_fixtures()
    llm = FakeLLM(
        '{"decision":"promote","target":"NYC","confidence":0.95,"rationale":"canonical"}'
    )
    cfg = _reasoner_config(
        llm, working_dir=str(tmp_path)
    )  # allow_promote defaults False
    res = await resolve_batch(nodes, {}, graph, vdb, cfg)
    rec = res.records[0]
    assert rec.decision is Decision.DISCARD_AND_REUSE
    assert rec.method == "promote_downgraded"
    assert res.promote_plans == []
    assert res.name_map["New York City"] == "NYC"
