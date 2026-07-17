from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lightrag.kg.graph_contract import EvidenceRef, GraphAssertion, GraphEntity
from lightrag.kg.postgres_impl import PGGraphStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data


pytestmark = pytest.mark.offline

CONTRACT_DIGEST = "a" * 64


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _Capture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fetch_results: list[list[dict[str, Any]]] = []
        self.transactions = 0
        self.rollbacks = 0


class _FakeTransaction:
    def __init__(self, capture: _Capture) -> None:
        self._capture = capture

    async def __aenter__(self) -> _FakeTransaction:
        self._capture.transactions += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._capture.rollbacks += 1
        return False


class _FakeConnection:
    def __init__(self, capture: _Capture) -> None:
        self._capture = capture

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self._capture)

    async def execute(self, sql: str, *args: object) -> str:
        self._capture.calls.append({"method": "execute", "sql": sql, "args": args})
        return ""

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        self._capture.calls.append({"method": "fetch", "sql": sql, "args": args})
        if self._capture.fetch_results:
            return self._capture.fetch_results.pop(0)
        return []


def _make_storage(
    *, max_upsert_records: int | None = None
) -> tuple[PGGraphStorage, _Capture]:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.__post_init__()
    if max_upsert_records is not None:
        storage._max_upsert_records_per_batch = max_upsert_records

    capture = _Capture()
    connection = _FakeConnection(capture)

    async def _run_with_retry(operation, **_kwargs):
        return await operation(connection)

    storage.db = AsyncMock()
    storage.db._run_with_retry = AsyncMock(side_effect=_run_with_retry)
    return storage, capture


def _evidence(chunk_id: str = "chunk-1") -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id,
        source_key="oceanstack/src/schema.py",
        source_revision="7801c2a7",
        metadata={
            "span": {"start": 10, "end": 42},
            "hostile": "nul:\u0000 noncharacter:\ufffe",
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
            "hostile": "nul:\u0000 noncharacter:\ufffe",
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
            "hostile": "nul:\u0000 noncharacter:\ufffe",
        },
        confidence=0.875,
        method="static-analysis",
        observed_from=datetime(2026, 7, 5, 8, 0, tzinfo=timezone.utc),
        observed_to=datetime(2026, 7, 5, 9, 0, tzinfo=timezone.utc),
        valid_from=datetime(2026, 7, 6, 8, 0, tzinfo=timezone.utc),
        valid_to=datetime(2026, 7, 6, 9, 0, tzinfo=timezone.utc),
    )


def _native_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _native_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_json(item) for item in value]
    return value


def _expected_record(record: GraphEntity | GraphAssertion) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(record):
        value = getattr(record, item.name)
        if item.name == "evidence":
            value = [
                {
                    "chunk_id": evidence.chunk_id,
                    "source_key": evidence.source_key,
                    "source_revision": evidence.source_revision,
                    "metadata": _native_json(evidence.metadata),
                }
                for evidence in value
            ]
        elif item.name == "metadata":
            value = _native_json(value)
        result[item.name] = value
    result["contract_digest"] = CONTRACT_DIGEST
    return result


def _typed_write_calls(capture: _Capture, label: str) -> list[dict[str, Any]]:
    return [
        call
        for call in capture.calls
        if call["method"] == "execute" and f":{label}" in call["sql"]
    ]


@pytest.mark.asyncio
async def test_entity_write_parameterizes_values_and_round_trip_payload() -> None:
    storage, capture = _make_storage()
    entity = _entity("source")

    await storage.upsert_graph_entity(entity, contract_digest=CONTRACT_DIGEST)

    writes = _typed_write_calls(capture, "base")
    assert len(writes) == 1
    sql = writes[0]["sql"]
    params = json.loads(writes[0]["args"][0])
    assert entity.entity_id not in sql
    assert entity.metadata["hostile"] not in sql
    assert "$entity_id" in sql
    assert params["entity_id"] == entity.entity_id
    assert "\u0000" not in params["metadata"]
    assert "\ufffe" not in params["metadata"]


@pytest.mark.asyncio
async def test_entity_batch_uses_one_transaction_across_bounded_chunks() -> None:
    storage, capture = _make_storage(max_upsert_records=2)

    await storage.upsert_graph_entities(
        [_entity(f"entity-{index}") for index in range(5)],
        contract_digest=CONTRACT_DIGEST,
    )

    assert capture.transactions == 1
    assert len(_typed_write_calls(capture, "base")) == 5
    exclusive_locks = [
        call
        for call in capture.calls
        if "pg_advisory_xact_lock(" in call["sql"] and "_shared" not in call["sql"]
    ]
    assert len(exclusive_locks) == 1
    assert exclusive_locks[0]["args"] == ("test_graph",)


@pytest.mark.asyncio
async def test_single_entity_uses_shared_graph_and_exact_record_locks() -> None:
    storage, capture = _make_storage()

    await storage.upsert_graph_entity(_entity("source"))

    assert "pg_advisory_xact_lock_shared" in capture.calls[0]["sql"]
    assert capture.calls[0]["args"] == ("test_graph",)
    assert "pg_advisory_xact_lock(" in capture.calls[1]["sql"]
    assert capture.calls[1]["args"] == ("test_graph", "GraphEntity", "source")


@pytest.mark.asyncio
async def test_assertions_preserve_direction_parallel_edges_and_parameters() -> None:
    storage, capture = _make_storage()
    assertions = [
        _assertion("assert-depends", "depends_on"),
        _assertion("assert-reads", "reads_from"),
        _assertion("assert-reverse", "feeds", "target", "source"),
    ]

    await storage.upsert_graph_assertions(assertions, contract_digest=CONTRACT_DIGEST)

    writes = _typed_write_calls(capture, "ASSERTION")
    assert len(writes) == 3
    payloads = [json.loads(call["args"][0]) for call in writes]
    assert {(item["src_id"], item["dst_id"]) for item in payloads} == {
        ("source", "target"),
        ("target", "source"),
    }
    assert {item["assertion_id"] for item in payloads} == {
        "assert-depends",
        "assert-reads",
        "assert-reverse",
    }
    assert all("OPTIONAL MATCH ()-[old:ASSERTION" in call["sql"] for call in writes)
    assert all("DELETE old" in call["sql"] for call in writes)
    assert all("DIRECTED" not in call["sql"] for call in writes)
    assert all(
        assertion.assertion_id not in call["sql"]
        for assertion, call in zip(assertions, writes, strict=True)
    )


@pytest.mark.asyncio
async def test_missing_endpoint_rejects_whole_assertion_batch_before_mutation() -> None:
    storage, capture = _make_storage()
    capture.fetch_results = [[{"entity_id": "missing"}]]

    with pytest.raises(ValueError, match="missing endpoint"):
        await storage.upsert_graph_assertions(
            [
                _assertion("assert-valid", "depends_on"),
                _assertion("assert-invalid", "depends_on", "source", "missing"),
            ]
        )

    assert capture.transactions == 1
    assert capture.rollbacks == 1
    assert _typed_write_calls(capture, "ASSERTION") == []
    assert not any("CREATE (missing" in call["sql"] for call in capture.calls)


@pytest.mark.asyncio
async def test_assertion_batch_is_atomic_and_uses_one_graph_lock() -> None:
    storage, capture = _make_storage(max_upsert_records=2)

    await storage.upsert_graph_assertions(
        [_assertion(f"assert-{index}", "depends_on") for index in range(5)]
    )

    assert capture.transactions == 1
    assert len(_typed_write_calls(capture, "ASSERTION")) == 5
    exclusive_locks = [
        call
        for call in capture.calls
        if "pg_advisory_xact_lock(" in call["sql"] and "_shared" not in call["sql"]
    ]
    assert len(exclusive_locks) == 1


@pytest.mark.asyncio
async def test_single_assertion_uses_exact_assertion_id_lock() -> None:
    storage, capture = _make_storage()

    await storage.upsert_graph_assertion(_assertion("assert-1", "depends_on"))

    assert "pg_advisory_xact_lock_shared" in capture.calls[0]["sql"]
    assert capture.calls[0]["args"] == ("test_graph",)
    assert "pg_advisory_xact_lock(" in capture.calls[1]["sql"]
    assert capture.calls[1]["args"] == (
        "test_graph",
        "GraphAssertion",
        "assert-1",
    )


@pytest.mark.asyncio
async def test_get_entity_decodes_native_shape_and_hostile_json() -> None:
    storage, _capture = _make_storage()
    entity = _entity("source")
    storage._query = AsyncMock(return_value=[{"n": {"properties": {}}}])
    storage._query.return_value[0]["n"]["properties"] = (
        storage._typed_record_properties(entity, CONTRACT_DIGEST)
    )

    stored = await storage.get_graph_entity(entity.entity_id)

    assert stored == _expected_record(entity)
    assert isinstance(stored["evidence"], list)
    assert isinstance(stored["evidence"][0]["metadata"], dict)
    assert (
        stored["evidence"][0]["metadata"]["hostile"] == "nul:\u0000 noncharacter:\ufffe"
    )
    assert isinstance(stored["metadata"], dict)
    assert stored["metadata"]["hostile"] == "nul:\u0000 noncharacter:\ufffe"
    assert isinstance(stored["observed_from"], datetime)
    query = storage._query.await_args.args[0]
    params = storage._query.await_args.kwargs["params"]
    assert entity.entity_id not in query
    assert json.loads(params["params"])["entity_id"] == entity.entity_id


@pytest.mark.asyncio
async def test_get_assertion_decodes_native_shape_and_rejects_duplicates() -> None:
    storage, _capture = _make_storage()
    assertion = _assertion("assert-1", "depends_on")
    properties = storage._typed_record_properties(assertion, CONTRACT_DIGEST)
    storage._query = AsyncMock(return_value=[{"r": {"properties": properties}}])

    stored = await storage.get_graph_assertion(assertion.assertion_id)

    assert stored == _expected_record(assertion)
    assert isinstance(stored["metadata"], dict)
    assert isinstance(stored["valid_to"], datetime)

    storage._query.return_value = [
        {"r": {"properties": properties}},
        {"r": {"properties": properties}},
    ]
    with pytest.raises(ValueError, match="duplicate assertion_id"):
        await storage.get_graph_assertion(assertion.assertion_id)


@pytest.mark.asyncio
async def test_initialize_creates_typed_label_and_exact_property_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _capture = _make_storage()
    storage.graph_name = "unused_until_initialize"
    storage.db.workspace = "test_ws"
    storage.db.execute = AsyncMock()
    monkeypatch.setattr(
        "lightrag.kg.postgres_impl.PostgreSQLDB.configure_age_extension",
        AsyncMock(),
    )

    await storage.initialize()

    statements = [call.args[0] for call in storage.db.execute.await_args_list]
    assert any("create_elabel" in sql and "ASSERTION" in sql for sql in statements)
    assert any('"ASSERTION"' in sql and '"assertion_id"' in sql for sql in statements)
    assert any('"base"' in sql and '"entity_id"' in sql for sql in statements)
    assert any(
        "UNIQUE INDEX CONCURRENTLY" in sql and '"assertion_id"' in sql
        for sql in statements
    )
    unique_call = next(
        call
        for call in storage.db.execute.await_args_list
        if "UNIQUE INDEX CONCURRENTLY" in call.args[0]
        and '"assertion_id"' in call.args[0]
    )
    assert unique_call.kwargs["ignore_if_exists"] is False


@pytest.mark.asyncio
async def test_typed_writes_reject_invalid_digest_before_transaction() -> None:
    storage, capture = _make_storage()

    with pytest.raises(ValueError, match="contract_digest"):
        await storage.upsert_graph_entity(_entity("source"), contract_digest="bad")
    with pytest.raises(ValueError, match="contract_digest"):
        await storage.upsert_graph_assertion(
            _assertion("assert-1", "depends_on"), contract_digest="bad"
        )

    assert capture.transactions == 0


@pytest.mark.asyncio
async def test_legacy_single_node_upsert_takes_exclusive_graph_lock() -> None:
    storage, capture = _make_storage()

    await storage.upsert_node("source", {"entity_id": "source", "name": "Source"})

    assert capture.transactions == 1
    assert "pg_advisory_xact_lock(" in capture.calls[0]["sql"]
    assert capture.calls[0]["args"] == ("test_graph",)
    assert "MERGE (n:base" in capture.calls[1]["sql"]


@pytest.mark.asyncio
async def test_legacy_node_batch_locks_each_bounded_chunk() -> None:
    storage, capture = _make_storage(max_upsert_records=2)
    nodes = [(f"node-{index}", {"entity_id": f"node-{index}"}) for index in range(5)]

    await storage.upsert_nodes_batch(nodes)

    locks = [call for call in capture.calls if "pg_advisory_xact_lock(" in call["sql"]]
    assert capture.transactions == 3
    assert len(locks) == 3
    assert all(call["args"] == ("test_graph",) for call in locks)
    assert "pg_advisory_xact_lock(" in capture.calls[0]["sql"]
    assert "pg_advisory_xact_lock(" in capture.calls[3]["sql"]
    assert "pg_advisory_xact_lock(" in capture.calls[6]["sql"]


@pytest.mark.asyncio
async def test_legacy_single_node_delete_takes_exclusive_graph_lock() -> None:
    storage, capture = _make_storage()

    await storage.delete_node("source")

    assert capture.transactions == 1
    assert "pg_advisory_xact_lock(" in capture.calls[0]["sql"]
    assert capture.calls[0]["args"] == ("test_graph",)
    assert "DETACH DELETE n" in capture.calls[1]["sql"]
    assert json.loads(capture.calls[1]["args"][0]) == {"entity_id": "source"}


@pytest.mark.asyncio
async def test_legacy_node_batch_delete_takes_graph_lock_before_mutation() -> None:
    storage, capture = _make_storage(max_upsert_records=2)

    await storage.remove_nodes(["source", "target", "third"])

    assert capture.transactions == 1
    assert "pg_advisory_xact_lock(" in capture.calls[0]["sql"]
    assert capture.calls[0]["args"] == ("test_graph",)
    assert all("DETACH DELETE n" in call["sql"] for call in capture.calls[1:])
