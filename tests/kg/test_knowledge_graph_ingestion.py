"""Strict ingestion tests for validated knowledge-graph builds."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lightrag import LightRAG
from lightrag.base import BaseGraphStorage
from lightrag.kg.graph_contract import (
    EvidenceRef,
    GraphAssertion,
    GraphChunk,
    GraphEntity,
    KnowledgeGraphBuild,
)
from lightrag.kg.shared_storage import initialize_share_data


def _chunk(chunk_id: str = "chunk:1") -> GraphChunk:
    return GraphChunk(
        build_id="build:1",
        chunk_id=chunk_id,
        source_key="oceanstack/src/schema.py",
        source_revision="abc123",
        content=f"content for {chunk_id}",
        metadata={"line_start": 10, "line_end": 12},
    )


def _evidence(chunk_id: str = "chunk:1") -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            chunk_id=chunk_id,
            source_key="oceanstack/src/schema.py",
            source_revision="abc123",
            metadata={"quote": "CREATE TABLE vessels"},
        ),
    )


def _entity(entity_id: str, *, metadata: dict | None = None) -> GraphEntity:
    return GraphEntity(
        build_id="build:1",
        entity_id=entity_id,
        entity_type="Table",
        evidence=_evidence(),
        metadata={"schema": "ais"} if metadata is None else metadata,
    )


def _assertion(
    assertion_id: str,
    src_id: str,
    dst_id: str,
    *,
    predicate: str = "references",
) -> GraphAssertion:
    return GraphAssertion(
        build_id="build:1",
        assertion_id=assertion_id,
        predicate=predicate,
        src_id=src_id,
        dst_id=dst_id,
        evidence=_evidence(),
        metadata={"constraint": "fk_vessel"},
        confidence=0.98,
        method="ddl",
    )


def _build(
    *,
    chunks: tuple[GraphChunk, ...] | None = None,
    entities: tuple[GraphEntity, ...] | None = None,
    assertions: tuple[GraphAssertion, ...] | None = None,
) -> KnowledgeGraphBuild:
    return KnowledgeGraphBuild.create(
        build_id="build:1",
        chunks=(_chunk(),) if chunks is None else chunks,
        entities=(_entity("entity:A"), _entity("entity:B"))
        if entities is None
        else entities,
        assertions=(_assertion("assertion:1", "entity:A", "entity:B"),)
        if assertions is None
        else assertions,
        metadata={"manifest_digest": "b" * 64, "plane": "dev"},
    )


def _storage() -> SimpleNamespace:
    return SimpleNamespace(upsert=AsyncMock())


def _bare_rag() -> LightRAG:
    initialize_share_data()
    rag = LightRAG.__new__(LightRAG)
    rag._owning_loop = None
    rag.workspace = "typed-test"
    rag.tokenizer = SimpleNamespace(encode=Mock(side_effect=lambda text: text.split()))
    rag.full_docs = None
    rag.doc_status = None
    rag.full_entities = None
    rag.full_relations = None
    rag.entity_chunks = None
    rag.relation_chunks = None
    rag.llm_response_cache = None
    rag.chunks_vdb = _storage()
    rag.text_chunks = _storage()
    rag.entities_vdb = _storage()
    rag.relationships_vdb = _storage()
    rag.chunk_entity_relation_graph = SimpleNamespace(
        upsert_graph_entities=AsyncMock(),
        upsert_graph_assertions=AsyncMock(),
        get_typed_graph_census=AsyncMock(),
    )
    rag._insert_done_with_cleanup = AsyncMock()
    rag._discard_pending_index_ops = AsyncMock()
    return rag


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_preserves_caller_ids_direction_and_parallel_assertions():
    rag = _bare_rag()
    build = _build(
        assertions=(
            _assertion("assertion:forward:1", "entity:A", "entity:B"),
            _assertion("assertion:forward:2", "entity:A", "entity:B"),
            _assertion("assertion:reverse", "entity:B", "entity:A"),
        )
    )

    await rag.ainsert_knowledge_graph(build)

    chunk_payload = rag.text_chunks.upsert.await_args.args[0]
    assert list(chunk_payload) == ["chunk:1"]
    assert chunk_payload["chunk:1"] == {
        "content": "content for chunk:1",
        "source_id": "chunk:1",
        "tokens": 3,
        "chunk_order_index": 0,
        "full_doc_id": "build:1",
        "file_path": "oceanstack/src/schema.py",
        "status": "processed",
        "sidecar": {
            "build_id": "build:1",
            "contract_digest": build.contract_digest,
            "manifest_digest": "b" * 64,
            "source_key": "oceanstack/src/schema.py",
            "source_revision": "abc123",
            "metadata": {"line_end": 12, "line_start": 10},
        },
    }
    chunk_vectors = rag.chunks_vdb.upsert.await_args.args[0]
    assert list(chunk_vectors) == ["chunk:1"]
    assert chunk_vectors["chunk:1"]["build_id"] == "build:1"
    assert chunk_vectors["chunk:1"]["contract_digest"] == build.contract_digest
    assert chunk_vectors["chunk:1"]["source_key"] == ("oceanstack/src/schema.py")
    assert chunk_vectors["chunk:1"]["source_revision"] == "abc123"
    assert chunk_vectors["chunk:1"]["metadata"] == {
        "line_end": 12,
        "line_start": 10,
    }
    graph_entities = (
        rag.chunk_entity_relation_graph.upsert_graph_entities.await_args.args[0]
    )
    graph_assertions = (
        rag.chunk_entity_relation_graph.upsert_graph_assertions.await_args.args[0]
    )
    assert [entity.entity_id for entity in graph_entities] == [
        "entity:A",
        "entity:B",
    ]
    assert [assertion.assertion_id for assertion in graph_assertions] == [
        "assertion:forward:1",
        "assertion:forward:2",
        "assertion:reverse",
    ]
    assert [(item.src_id, item.dst_id) for item in graph_assertions] == [
        ("entity:A", "entity:B"),
        ("entity:A", "entity:B"),
        ("entity:B", "entity:A"),
    ]
    assert (
        rag.chunk_entity_relation_graph.upsert_graph_entities.await_args.kwargs[
            "contract_digest"
        ]
        == build.contract_digest
    )

    entity_vectors = rag.entities_vdb.upsert.await_args.args[0]
    assert list(entity_vectors) == ["entity:A", "entity:B"]
    assert entity_vectors["entity:A"]["entity_id"] == "entity:A"
    assert entity_vectors["entity:A"]["entity_type"] == "Table"
    assert entity_vectors["entity:A"]["metadata"] == {"schema": "ais"}
    assert entity_vectors["entity:A"]["evidence_chunk_ids"] == ["chunk:1"]

    relation_vectors = rag.relationships_vdb.upsert.await_args.args[0]
    assert list(relation_vectors) == [
        "assertion:forward:1",
        "assertion:forward:2",
        "assertion:reverse",
    ]
    forward = relation_vectors["assertion:forward:1"]
    assert forward["assertion_id"] == "assertion:forward:1"
    assert forward["predicate"] == "references"
    assert forward["src_entity_id"] == "entity:A"
    assert forward["dst_entity_id"] == "entity:B"
    assert forward["evidence_chunk_ids"] == ["chunk:1"]
    assert forward["evidence"] == [
        {
            "chunk_id": "chunk:1",
            "metadata": {"quote": "CREATE TABLE vessels"},
            "source_key": "oceanstack/src/schema.py",
            "source_revision": "abc123",
        }
    ]
    assert forward["metadata"] == {"constraint": "fk_vessel"}
    canonical_assertion = next(
        record
        for record in build.to_canonical_dict()["assertions"]
        if record["assertion_id"] == "assertion:forward:1"
    )
    assert forward["content"] == json.dumps(
        {"contract_digest": build.contract_digest, **canonical_assertion},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rag._insert_done_with_cleanup.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_persisted_generation_evidence_comes_from_storage_census() -> None:
    rag = _bare_rag()
    build = _build()
    rag.chunk_entity_relation_graph.get_typed_graph_census = AsyncMock(
        return_value={
            "assertions": 1,
            "contract_digests": [build.contract_digest],
            "entities": 2,
            "missing_contract_digests": 0,
        }
    )
    rag.text_chunks.get_typed_chunk_census = AsyncMock(
        return_value={
            "chunks": 1,
            "contract_digests": [build.contract_digest],
            "manifest_digests": ["b" * 64],
            "missing_contract_digests": 0,
            "missing_manifest_digests": 0,
            "sources": 1,
        }
    )

    evidence = await rag.avalidate_persisted_knowledge_graph(
        expected_counts={
            "assertions": 1,
            "chunks": 1,
            "entities": 2,
            "sources": 1,
        },
        expected_contract_digest=build.contract_digest,
        expected_manifest_digest="b" * 64,
    )

    assert evidence.counts == {
        "assertions": 1,
        "chunks": 1,
        "entities": 2,
        "sources": 1,
    }
    assert evidence.contract_digest == build.contract_digest
    assert evidence.manifest_digest == "b" * 64


@pytest.mark.offline
@pytest.mark.asyncio
async def test_persisted_generation_evidence_rejects_partial_storage() -> None:
    rag = _bare_rag()
    build = _build()
    rag.chunk_entity_relation_graph.get_typed_graph_census = AsyncMock(
        return_value={
            "assertions": 0,
            "contract_digests": [build.contract_digest],
            "entities": 2,
            "missing_contract_digests": 0,
        }
    )
    rag.text_chunks.get_typed_chunk_census = AsyncMock(
        return_value={
            "chunks": 1,
            "contract_digests": [build.contract_digest],
            "manifest_digests": ["b" * 64],
            "missing_contract_digests": 0,
            "missing_manifest_digests": 0,
            "sources": 1,
        }
    )

    with pytest.raises(ValueError, match="persisted graph census"):
        await rag.avalidate_persisted_knowledge_graph(
            expected_counts={
                "assertions": 1,
                "chunks": 1,
                "entities": 2,
                "sources": 1,
            },
            expected_contract_digest=build.contract_digest,
            expected_manifest_digest="b" * 64,
        )


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("census_name", "field", "value", "message"),
    [
        ("graph", "contract_digests", ["d" * 64], "contract digest"),
        ("chunks", "manifest_digests", ["e" * 64], "manifest digest"),
        ("chunks", "missing_manifest_digests", 1, "manifest digest is missing"),
    ],
)
async def test_persisted_generation_evidence_rejects_provenance_drift(
    census_name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    rag = _bare_rag()
    build = _build()
    graph_census = {
        "assertions": 1,
        "contract_digests": [build.contract_digest],
        "entities": 2,
        "missing_contract_digests": 0,
    }
    chunk_census = {
        "chunks": 1,
        "contract_digests": [build.contract_digest],
        "manifest_digests": ["b" * 64],
        "missing_contract_digests": 0,
        "missing_manifest_digests": 0,
        "sources": 1,
    }
    target = graph_census if census_name == "graph" else chunk_census
    target[field] = value  # type: ignore[assignment]
    rag.chunk_entity_relation_graph.get_typed_graph_census = AsyncMock(
        return_value=graph_census
    )
    rag.text_chunks.get_typed_chunk_census = AsyncMock(return_value=chunk_census)

    with pytest.raises(ValueError, match=message):
        await rag.avalidate_persisted_knowledge_graph(
            expected_counts={
                "assertions": 1,
                "chunks": 1,
                "entities": 2,
                "sources": 1,
            },
            expected_contract_digest=build.contract_digest,
            expected_manifest_digest="b" * 64,
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_rejects_wrong_type_before_storage_mutation():
    rag = _bare_rag()

    with pytest.raises(TypeError, match="KnowledgeGraphBuild"):
        await rag.ainsert_knowledge_graph({"build_id": "build:1"})  # type: ignore[arg-type]

    rag.text_chunks.upsert.assert_not_awaited()
    rag.chunk_entity_relation_graph.upsert_graph_entities.assert_not_awaited()
    rag._discard_pending_index_ops.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_revalidates_digest_before_storage_mutation():
    rag = _bare_rag()
    build = _build()
    object.__setattr__(build, "contract_digest", "0" * 64)

    with pytest.raises(ValueError, match="contract_digest"):
        await rag.ainsert_knowledge_graph(build)

    rag.chunks_vdb.upsert.assert_not_awaited()
    rag.chunk_entity_relation_graph.upsert_graph_entities.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_rejects_unsupported_graph_backend_before_any_mutation():
    rag = _bare_rag()
    unsupported_graph = SimpleNamespace()
    unsupported_graph.upsert_graph_entities = (
        BaseGraphStorage.upsert_graph_entities.__get__(
            unsupported_graph, BaseGraphStorage
        )
    )
    unsupported_graph.upsert_graph_assertions = (
        BaseGraphStorage.upsert_graph_assertions.__get__(
            unsupported_graph, BaseGraphStorage
        )
    )
    rag.chunk_entity_relation_graph = unsupported_graph

    with pytest.raises(NotImplementedError, match="typed directed multigraph"):
        await rag.ainsert_knowledge_graph(_build())

    rag.text_chunks.upsert.assert_not_awaited()
    rag.chunks_vdb.upsert.assert_not_awaited()
    rag.entities_vdb.upsert.assert_not_awaited()
    rag.relationships_vdb.upsert.assert_not_awaited()
    rag._insert_done_with_cleanup.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_rejects_graph_backend_missing_typed_census():
    """A backend overriding only the two upserts (e.g. NetworkXStorage) must
    still be rejected before mutation, since ``avalidate_persisted_knowledge_graph``
    unconditionally calls ``get_typed_graph_census`` after the flush and would
    otherwise blow up on a backend that passed this earlier gate."""
    rag = _bare_rag()
    # Real overrides for the two upserts (like NetworkXStorage), but
    # get_typed_graph_census is simply absent (not inherited from
    # BaseGraphStorage) -- the gate must still reject on the missing method.
    partial_graph = SimpleNamespace(
        upsert_graph_entities=AsyncMock(),
        upsert_graph_assertions=AsyncMock(),
    )
    rag.chunk_entity_relation_graph = partial_graph

    with pytest.raises(NotImplementedError, match="typed directed multigraph"):
        await rag.ainsert_knowledge_graph(_build())

    rag.text_chunks.upsert.assert_not_awaited()
    rag.chunks_vdb.upsert.assert_not_awaited()
    rag.entities_vdb.upsert.assert_not_awaited()
    rag.relationships_vdb.upsert.assert_not_awaited()
    rag._insert_done_with_cleanup.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_batches_graph_and_vector_records_at_one_hundred():
    rag = _bare_rag()
    build = _build(
        entities=tuple(_entity(f"entity:{index:03d}") for index in range(205)),
        assertions=(),
    )

    await rag.ainsert_knowledge_graph(build)

    assert [
        len(call.args[0])
        for call in rag.chunk_entity_relation_graph.upsert_graph_entities.await_args_list
    ] == [100, 100, 5]
    assert [len(call.args[0]) for call in rag.entities_vdb.upsert.await_args_list] == [
        100,
        100,
        5,
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_sorts_every_storage_call_by_caller_id():
    rag = _bare_rag()
    build = _build(
        chunks=(_chunk("chunk:2"), _chunk("chunk:1")),
        entities=(_entity("entity:B"), _entity("entity:A")),
        assertions=(
            _assertion("assertion:z", "entity:B", "entity:A"),
            _assertion("assertion:a", "entity:A", "entity:B"),
        ),
    )

    await rag.ainsert_knowledge_graph(build)

    assert list(rag.text_chunks.upsert.await_args.args[0]) == ["chunk:1", "chunk:2"]
    assert list(rag.chunks_vdb.upsert.await_args.args[0]) == ["chunk:1", "chunk:2"]
    assert {
        chunk_id: payload["chunk_order_index"]
        for chunk_id, payload in rag.text_chunks.upsert.await_args.args[0].items()
    } == {"chunk:1": 0, "chunk:2": 1}
    assert [
        item.entity_id
        for item in rag.chunk_entity_relation_graph.upsert_graph_entities.await_args.args[
            0
        ]
    ] == ["entity:A", "entity:B"]
    assert list(rag.entities_vdb.upsert.await_args.args[0]) == [
        "entity:A",
        "entity:B",
    ]
    assert [
        item.assertion_id
        for item in rag.chunk_entity_relation_graph.upsert_graph_assertions.await_args.args[
            0
        ]
    ] == ["assertion:a", "assertion:z"]
    assert list(rag.relationships_vdb.upsert.await_args.args[0]) == [
        "assertion:a",
        "assertion:z",
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_concurrent_builds_serialize_all_mutations_per_workspace():
    first = _bare_rag()
    second = _bare_rag()
    active_mutations = 0
    maximum_active_mutations = 0

    async def _observe_mutation(_payload):
        nonlocal active_mutations, maximum_active_mutations
        active_mutations += 1
        maximum_active_mutations = max(maximum_active_mutations, active_mutations)
        await asyncio.sleep(0.02)
        active_mutations -= 1

    first.text_chunks.upsert.side_effect = _observe_mutation
    second.text_chunks.upsert.side_effect = _observe_mutation

    await asyncio.gather(
        first.ainsert_knowledge_graph(_build()),
        second.ainsert_knowledge_graph(_build()),
    )

    assert maximum_active_mutations == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_splits_batches_at_cumulative_byte_budget(monkeypatch):
    rag = _bare_rag()
    monkeypatch.setattr("lightrag.lightrag._GRAPH_CALL_MAX_BYTES", 3_000)
    build = _build(
        entities=(
            _entity("entity:A", metadata={"value": "a" * 600}),
            _entity("entity:B", metadata={"value": "b" * 600}),
        ),
        assertions=(),
    )

    await rag.ainsert_knowledge_graph(build)

    assert [
        len(call.args[0])
        for call in rag.chunk_entity_relation_graph.upsert_graph_entities.await_args_list
    ] == [2]
    assert [len(call.args[0]) for call in rag.entities_vdb.upsert.await_args_list] == [
        1,
        1,
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_batches_chunk_storage_calls_at_one_hundred():
    rag = _bare_rag()
    build = _build(
        chunks=tuple(_chunk(f"chunk:{index:03d}") for index in range(205)),
        entities=(),
        assertions=(),
    )

    await rag.ainsert_knowledge_graph(build)

    assert [len(call.args[0]) for call in rag.text_chunks.upsert.await_args_list] == [
        100,
        100,
        5,
    ]
    assert [len(call.args[0]) for call in rag.chunks_vdb.upsert.await_args_list] == [
        100,
        100,
        5,
    ]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_rejects_oversized_chunk_before_mutation(monkeypatch):
    rag = _bare_rag()
    monkeypatch.setattr("lightrag.lightrag._GRAPH_CALL_MAX_BYTES", 256)
    build = _build(
        chunks=(
            GraphChunk(
                build_id="build:1",
                chunk_id="chunk:large",
                source_key="oceanstack/src/schema.py",
                source_revision="abc123",
                content="x" * 512,
                metadata={},
            ),
        ),
        entities=(),
        assertions=(),
    )

    with pytest.raises(ValueError, match="single storage record"):
        await rag.ainsert_knowledge_graph(build)

    rag.text_chunks.upsert.assert_not_awaited()
    rag.chunks_vdb.upsert.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_rejects_single_oversized_record_before_mutation(monkeypatch):
    rag = _bare_rag()
    monkeypatch.setattr("lightrag.lightrag._GRAPH_CALL_MAX_BYTES", 256)
    build = _build(
        entities=(_entity("entity:large", metadata={"value": "x" * 512}),),
        assertions=(),
    )

    with pytest.raises(ValueError, match="single graph record"):
        await rag.ainsert_knowledge_graph(build)

    rag.text_chunks.upsert.assert_not_awaited()
    rag.chunk_entity_relation_graph.upsert_graph_entities.assert_not_awaited()
    rag.entities_vdb.upsert.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_discards_pending_buffers_after_failure():
    rag = _bare_rag()
    rag.relationships_vdb.upsert.side_effect = RuntimeError("vector write failed")

    with pytest.raises(RuntimeError, match="vector write failed"):
        await rag.ainsert_knowledge_graph(_build())

    rag._discard_pending_index_ops.assert_awaited_once_with(skip_enqueue_owned=False)
    rag._insert_done_with_cleanup.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ingestion_shields_pending_buffer_cleanup_after_cancellation():
    rag = _bare_rag()
    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def _cancel_write(_payload):
        raise asyncio.CancelledError

    async def _cleanup(*, skip_enqueue_owned):
        assert skip_enqueue_owned is False
        cleanup_started.set()
        await asyncio.sleep(0)
        cleanup_finished.set()

    rag.chunks_vdb.upsert.side_effect = _cancel_write
    rag._discard_pending_index_ops.side_effect = _cleanup

    with pytest.raises(asyncio.CancelledError):
        await rag.ainsert_knowledge_graph(_build())

    assert cleanup_started.is_set()
    assert cleanup_finished.is_set()


@pytest.mark.offline
def test_sync_ingestion_wrapper_runs_async_method():
    rag = _bare_rag()
    build = _build()
    rag.ainsert_knowledge_graph = AsyncMock()

    rag.insert_knowledge_graph(build)

    rag.ainsert_knowledge_graph.assert_awaited_once_with(build)


@pytest.mark.offline
def test_vector_storages_accept_typed_metadata_fields(tmp_path):
    import numpy as np

    from lightrag.utils import EmbeddingFunc

    async def _fake_cpu_embedding(texts):
        return np.zeros((len(texts), 8), dtype=np.float32)

    rag = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=_fake_cpu_embedding,
        ),
    )

    assert {
        "build_id",
        "contract_digest",
        "entity_id",
        "entity_type",
        "evidence_chunk_ids",
        "metadata",
    } <= rag.entities_vdb.meta_fields
    assert {
        "build_id",
        "contract_digest",
        "assertion_id",
        "predicate",
        "src_entity_id",
        "dst_entity_id",
        "evidence_chunk_ids",
        "metadata",
    } <= rag.relationships_vdb.meta_fields
    assert {
        "build_id",
        "contract_digest",
        "source_key",
        "source_revision",
        "metadata",
    } <= rag.chunks_vdb.meta_fields
