from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest

from lightrag.base import BaseGraphStorage
from lightrag.kg.graph_contract import EvidenceRef, GraphAssertion, GraphEntity
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc


pytestmark = pytest.mark.offline

CONTRACT_DIGEST = "a" * 64


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8))


def _make_storage(tmp_path) -> NetworkXStorage:
    return NetworkXStorage(
        namespace="typed_graph",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=_embed,
        ),
    )


def _evidence(chunk_id: str = "chunk-1") -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id,
        source_key="oceanstack/src/schema.py",
        source_revision="7801c2a7",
        metadata={
            "span": {"start": 10, "end": 42},
            "tags": ["schema", None, True],
        },
    )


def _entity(entity_id: str, *, build_id: str = "build-1") -> GraphEntity:
    return GraphEntity(
        build_id=build_id,
        entity_id=entity_id,
        entity_type="table",
        evidence=(_evidence(),),
        metadata={
            "qualified_name": f"ais.{entity_id}",
            "shape": {"columns": ["mmsi", "time"], "partitioned": True},
            "xml_hostile_json": "control:\u0000 noncharacter:\ufffe",
        },
        observed_from=datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc),
        observed_to=datetime(2026, 7, 2, 12, 30, tzinfo=timezone.utc),
        valid_from=datetime(2026, 7, 3, 12, 30, tzinfo=timezone.utc),
        valid_to=datetime(2026, 7, 4, 12, 30, tzinfo=timezone.utc),
    )


def _assertion(
    assertion_id: str,
    predicate: str,
    src_id: str = "source",
    dst_id: str = "target",
    *,
    confidence: float = 0.875,
) -> GraphAssertion:
    return GraphAssertion(
        build_id="build-1",
        assertion_id=assertion_id,
        predicate=predicate,
        src_id=src_id,
        dst_id=dst_id,
        evidence=(_evidence(f"chunk-{assertion_id}"),),
        metadata={
            "join": {"left": ["mmsi"], "right": ["mmsi"]},
            "cardinality": [1, "many"],
        },
        confidence=confidence,
        method="static-analysis",
        observed_from=datetime(2026, 7, 5, 8, 0, tzinfo=timezone.utc),
        observed_to=datetime(2026, 7, 5, 9, 0, tzinfo=timezone.utc),
        valid_from=datetime(2026, 7, 6, 8, 0, tzinfo=timezone.utc),
        valid_to=datetime(2026, 7, 6, 9, 0, tzinfo=timezone.utc),
    )


def _assert_record_data(
    stored: dict[str, Any], expected: GraphEntity | GraphAssertion
) -> None:
    for item in fields(expected):
        expected_value = getattr(expected, item.name)
        if item.name == "evidence":
            expected_value = [
                {
                    "chunk_id": evidence.chunk_id,
                    "source_key": evidence.source_key,
                    "source_revision": evidence.source_revision,
                    "metadata": _native_json(evidence.metadata),
                }
                for evidence in expected_value
            ]
        elif item.name == "metadata":
            expected_value = _native_json(expected_value)
        assert stored[item.name] == expected_value
    assert stored["contract_digest"] == CONTRACT_DIGEST


def _native_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _native_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_json(item) for item in value]
    return value


@pytest.mark.asyncio
async def test_base_typed_graph_operations_fail_clearly_for_unsupported_backends():
    entity = _entity("source")
    assertion = _assertion("assert-1", "depends_on")
    unsupported = cast(BaseGraphStorage, object())

    operations = (
        BaseGraphStorage.upsert_graph_entity(
            unsupported, entity, contract_digest=CONTRACT_DIGEST
        ),
        BaseGraphStorage.upsert_graph_entities(
            unsupported, [entity], contract_digest=CONTRACT_DIGEST
        ),
        BaseGraphStorage.get_graph_entity(unsupported, entity.entity_id),
        BaseGraphStorage.upsert_graph_assertion(
            unsupported, assertion, contract_digest=CONTRACT_DIGEST
        ),
        BaseGraphStorage.upsert_graph_assertions(
            unsupported, [assertion], contract_digest=CONTRACT_DIGEST
        ),
        BaseGraphStorage.get_graph_assertion(unsupported, assertion.assertion_id),
    )

    for operation in operations:
        with pytest.raises(NotImplementedError, match="typed directed multigraph"):
            await operation


@pytest.mark.asyncio
async def test_typed_entities_round_trip_all_record_properties(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        entity = _entity("source")

        await storage.upsert_graph_entity(entity, contract_digest=CONTRACT_DIGEST)

        stored = await storage.get_graph_entity(entity.entity_id)
        assert stored is not None
        _assert_record_data(stored, entity)
        assert isinstance(stored["evidence"], list)
        assert isinstance(stored["evidence"][0], dict)
        assert isinstance(stored["evidence"][0]["metadata"], dict)
        assert isinstance(stored["evidence"][0]["metadata"]["tags"], list)
        assert isinstance(stored["metadata"], dict)
        assert isinstance(stored["metadata"]["shape"], dict)
        assert isinstance(stored["metadata"]["shape"]["columns"], list)
        assert isinstance(stored["observed_from"], datetime)
        assert isinstance(storage._graph, nx.MultiDiGraph)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_assertions_preserve_parallel_predicates_and_reciprocal_direction(
    tmp_path,
):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_graph_entities(
            [_entity("source"), _entity("target")],
            contract_digest=CONTRACT_DIGEST,
        )
        assertions = [
            _assertion("assert-depends", "depends_on"),
            _assertion("assert-reads", "reads_from"),
            _assertion("assert-reciprocal", "feeds", "target", "source"),
        ]

        await storage.upsert_graph_assertions(
            assertions,
            contract_digest=CONTRACT_DIGEST,
        )

        graph = storage._graph
        assert set(graph.edges(keys=True)) == {
            ("source", "target", "assert-depends"),
            ("source", "target", "assert-reads"),
            ("target", "source", "assert-reciprocal"),
        }
        for assertion in assertions:
            stored = await storage.get_graph_assertion(assertion.assertion_id)
            assert stored is not None
            _assert_record_data(stored, assertion)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_duplicate_assertion_id_replaces_only_that_assertion(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_graph_entities(
            [_entity("source"), _entity("target")],
            contract_digest=CONTRACT_DIGEST,
        )
        original = _assertion("assert-reused", "depends_on", confidence=0.25)
        independent = _assertion("assert-independent", "reads_from")
        replacement = _assertion("assert-reused", "feeds", confidence=0.95)
        await storage.upsert_graph_assertions(
            [original, independent], contract_digest=CONTRACT_DIGEST
        )

        await storage.upsert_graph_assertion(
            replacement, contract_digest=CONTRACT_DIGEST
        )

        graph = storage._graph
        assert set(graph["source"]["target"]) == {
            "assert-reused",
            "assert-independent",
        }
        stored_replacement = await storage.get_graph_assertion("assert-reused")
        stored_independent = await storage.get_graph_assertion("assert-independent")
        assert stored_replacement is not None
        assert stored_independent is not None
        _assert_record_data(stored_replacement, replacement)
        _assert_record_data(stored_independent, independent)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_duplicate_assertion_id_can_move_without_leaving_stale_edge(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_graph_entities(
            [_entity("source"), _entity("target"), _entity("third")],
            contract_digest=CONTRACT_DIGEST,
        )
        await storage.upsert_graph_assertion(
            _assertion("assert-moved", "depends_on"),
            contract_digest=CONTRACT_DIGEST,
        )
        moved = _assertion("assert-moved", "depends_on", "third", "source")

        await storage.upsert_graph_assertion(moved, contract_digest=CONTRACT_DIGEST)

        graph = storage._graph
        assert not graph.has_edge("source", "target", key="assert-moved")
        assert graph.has_edge("third", "source", key="assert-moved")
        stored = await storage.get_graph_assertion("assert-moved")
        assert stored is not None
        _assert_record_data(stored, moved)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_assertion_index_avoids_global_edge_scans_after_initial_build(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_graph_entities(
            [_entity("source"), _entity("target"), _entity("third")],
            contract_digest=CONTRACT_DIGEST,
        )
        original = _assertion("assert-indexed", "depends_on")
        await storage.upsert_graph_assertion(original, contract_digest=CONTRACT_DIGEST)
        assert storage._assertion_index == {"assert-indexed": ("source", "target")}

        class _NoEdgeIteration:
            def __call__(self, *args, **kwargs):
                raise AssertionError("typed assertion operation scanned every edge")

        storage._graph.__dict__["edges"] = _NoEdgeIteration()
        replacement = _assertion(
            "assert-indexed", "feeds", src_id="third", dst_id="source"
        )
        await storage.upsert_graph_assertions(
            [replacement], contract_digest=CONTRACT_DIGEST
        )

        stored = await storage.get_graph_assertion("assert-indexed")
        assert stored is not None
        _assert_record_data(stored, replacement)
        assert storage._assertion_index == {"assert-indexed": ("third", "source")}
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_assertion_index_tracks_node_removal_and_drop(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_graph_entities(
            [_entity("source"), _entity("target"), _entity("third")],
            contract_digest=CONTRACT_DIGEST,
        )
        await storage.upsert_graph_assertions(
            [
                _assertion("assert-removed", "depends_on"),
                _assertion("assert-kept", "feeds", src_id="target", dst_id="third"),
            ],
            contract_digest=CONTRACT_DIGEST,
        )

        await storage.remove_nodes(["source"])

        assert storage._assertion_index == {"assert-kept": ("target", "third")}
        assert await storage.get_graph_assertion("assert-removed") is None
        assert await storage.get_graph_assertion("assert-kept") is not None

        assert (await storage.drop())["status"] == "success"
        assert storage._assertion_index == {}
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_legacy_reverse_and_sink_retrieval_remain_undirected(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_nodes_batch(
            [
                ("source", {"entity_id": "source"}),
                ("sink", {"entity_id": "sink"}),
                ("typed-source", {"entity_id": "typed-source"}),
            ]
        )
        edge_data = {"description": "legacy edge", "weight": "1.0"}
        await storage.upsert_edge("source", "sink", edge_data)
        typed_assertion = _assertion(
            "assert-incoming", "feeds", src_id="typed-source", dst_id="sink"
        )
        await storage.upsert_graph_assertion(
            typed_assertion, contract_digest=CONTRACT_DIGEST
        )

        assert await storage.has_edge("sink", "source")
        assert await storage.get_edge("sink", "source") == edge_data
        assert await storage.get_node_edges("sink") == [("source", "sink")]

        result = await storage.get_knowledge_graph("sink", max_depth=1, max_nodes=10)
        assert {node.id for node in result.nodes} == {
            "source",
            "sink",
            "typed-source",
        }
        assert {(edge.id, edge.source, edge.target) for edge in result.edges} == {
            ("source-sink", "source", "sink"),
            ("assert-incoming", "typed-source", "sink"),
        }
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_missing_assertion_endpoint_rejects_whole_batch_without_placeholders(
    tmp_path,
):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_graph_entities(
            [_entity("source"), _entity("target")],
            contract_digest=CONTRACT_DIGEST,
        )
        valid = _assertion("assert-valid", "depends_on")
        invalid = _assertion("assert-invalid", "depends_on", "source", "missing")

        with pytest.raises(ValueError, match="missing endpoint"):
            await storage.upsert_graph_assertions(
                [valid, invalid], contract_digest=CONTRACT_DIGEST
            )

        assert await storage.get_graph_assertion("assert-valid") is None
        assert await storage.get_graph_assertion("assert-invalid") is None
        assert not await storage.has_node("missing")
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_graphml_reload_retains_keys_direction_and_typed_metadata(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        entities = [_entity("source"), _entity("target")]
        assertions = [
            _assertion("assert-forward-1", "depends_on"),
            _assertion("assert-forward-2", "reads_from"),
            _assertion("assert-reverse", "feeds", "target", "source"),
        ]
        await storage.upsert_graph_entities(entities, contract_digest=CONTRACT_DIGEST)
        await storage.upsert_graph_assertions(
            assertions, contract_digest=CONTRACT_DIGEST
        )
        assert await storage.index_done_callback() is True
    finally:
        await storage.finalize()

    reloaded = _make_storage(tmp_path)
    await reloaded.initialize()
    try:
        assert isinstance(reloaded._graph, nx.MultiDiGraph)
        assert set(reloaded._graph.edges(keys=True)) == {
            ("source", "target", "assert-forward-1"),
            ("source", "target", "assert-forward-2"),
            ("target", "source", "assert-reverse"),
        }
        assert reloaded._assertion_index == {
            "assert-forward-1": ("source", "target"),
            "assert-forward-2": ("source", "target"),
            "assert-reverse": ("target", "source"),
        }
        for entity in entities:
            stored = await reloaded.get_graph_entity(entity.entity_id)
            assert stored is not None
            _assert_record_data(stored, entity)
            assert isinstance(stored["evidence"], list)
            assert isinstance(stored["evidence"][0], dict)
            assert isinstance(stored["metadata"], dict)
            assert isinstance(stored["metadata"]["shape"]["columns"], list)
            assert isinstance(stored["observed_from"], datetime)
        for assertion in assertions:
            stored = await reloaded.get_graph_assertion(assertion.assertion_id)
            assert stored is not None
            _assert_record_data(stored, assertion)
    finally:
        await reloaded.finalize()


def test_legacy_undirected_graphml_requires_clean_start(tmp_path):
    graph_path = tmp_path / "ws" / "graph_typed_graph.graphml"
    graph_path.parent.mkdir(parents=True)
    legacy = nx.Graph()
    legacy.add_edge("source", "target")
    nx.write_graphml(legacy, graph_path)

    with pytest.raises(ValueError, match="clean startup"):
        _make_storage(tmp_path)
