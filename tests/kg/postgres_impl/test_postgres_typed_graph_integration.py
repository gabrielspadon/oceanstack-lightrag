from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from lightrag.kg.graph_contract import EvidenceRef, GraphAssertion, GraphEntity
from lightrag.kg.postgres_impl import PGGraphStorage, PostgreSQLDB
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data


pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

CONTRACT_DIGEST = "b" * 64


class _LiveGraphDB:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool
        self.workspace = ""
        self.pause_after_shared_lock = False
        self.shared_lock_acquired = asyncio.Event()
        self.release_shared_lock = asyncio.Event()

    async def _run_with_retry(
        self,
        operation,
        *,
        with_age: bool = False,
        graph_name: str | None = None,
        **_kwargs: object,
    ):
        async with self.pool.acquire() as connection:
            if with_age:
                assert graph_name is not None
                await PostgreSQLDB.configure_age(connection, graph_name)
            operation_connection = connection
            if self.pause_after_shared_lock:
                operation_connection = _PausingConnection(connection, self)
            return await operation(operation_connection)

    async def execute(
        self,
        sql: str,
        data: dict[str, Any] | None = None,
        *,
        ignore_if_exists: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
        **_kwargs: object,
    ) -> None:
        async with self.pool.acquire() as connection:
            if with_age:
                assert graph_name is not None
                await PostgreSQLDB.configure_age(connection, graph_name)
            try:
                await connection.execute(sql, *(data.values() if data else ()))
            except (
                asyncpg.DuplicateObjectError,
                asyncpg.DuplicateTableError,
                asyncpg.InvalidSchemaNameError,
                asyncpg.UniqueViolationError,
            ):
                if not ignore_if_exists:
                    raise

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        multirows: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
        **_kwargs: object,
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        async with self.pool.acquire() as connection:
            if with_age:
                assert graph_name is not None
                await PostgreSQLDB.configure_age(connection, graph_name)
            rows = await connection.fetch(sql, *(params or ()))
        converted = [dict(row) for row in rows]
        if multirows:
            return converted
        return converted[0] if converted else None


class _PausingConnection:
    def __init__(
        self,
        connection: asyncpg.Connection,
        graph_db: _LiveGraphDB,
    ) -> None:
        self._connection = connection
        self._graph_db = graph_db

    def transaction(self):
        return self._connection.transaction()

    async def execute(self, sql: str, *args: object):
        result = await self._connection.execute(sql, *args)
        if (
            self._graph_db.pause_after_shared_lock
            and "pg_advisory_xact_lock_shared" in sql
        ):
            self._graph_db.pause_after_shared_lock = False
            self._graph_db.shared_lock_acquired.set()
            await self._graph_db.release_shared_lock.wait()
        return result

    async def fetch(self, sql: str, *args: object):
        return await self._connection.fetch(sql, *args)


def _connection_kwargs(database: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "database": database,
        "user": os.getenv("POSTGRES_USER", os.getenv("USER", "postgres")),
    }
    for env_name, key, converter in (
        ("POSTGRES_HOST", "host", str),
        ("POSTGRES_PORT", "port", int),
        ("POSTGRES_PASSWORD", "password", str),
    ):
        value = os.getenv(env_name)
        if value:
            kwargs[key] = converter(value)
    return kwargs


@pytest_asyncio.fixture
async def live_storage() -> PGGraphStorage:
    database = os.getenv("LIGHTRAG_PG_TEST_DATABASE")
    if not database:
        pytest.skip("set LIGHTRAG_PG_TEST_DATABASE to an isolated test database")
    if database == "oceanstack" or "test" not in database.casefold():
        pytest.fail("LIGHTRAG_PG_TEST_DATABASE must name an isolated test database")

    pool = await asyncpg.create_pool(
        **_connection_kwargs(database), min_size=1, max_size=2
    )
    graph_db = _LiveGraphDB(pool)
    namespace = f"typed_it_{uuid.uuid4().hex[:12]}"
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = namespace
    storage.graph_name = namespace
    storage.__post_init__()
    storage.db = graph_db

    finalize_share_data()
    initialize_share_data()
    await storage.initialize()
    try:
        yield storage
    finally:
        async with pool.acquire() as connection:
            await PostgreSQLDB.configure_age(connection, storage.graph_name)
            await connection.execute(f"SELECT drop_graph('{storage.graph_name}', true)")
        await pool.close()
        finalize_share_data()


def _evidence(chunk_id: str) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id,
        source_key="oceanstack/src/schema.py",
        source_revision="integration",
        metadata={"hostile": "nul:\u0000 noncharacter:\ufffe"},
    )


def _entity(entity_id: str) -> GraphEntity:
    return GraphEntity(
        build_id="build-integration",
        entity_id=entity_id,
        entity_type="table",
        evidence=(_evidence(f"chunk-{entity_id}"),),
        metadata={"nested": {"ids": [entity_id]}, "hostile": "nul:\u0000\ufffe"},
        observed_from=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )


def _assertion(
    assertion_id: str,
    predicate: str,
    src_id: str,
    dst_id: str,
) -> GraphAssertion:
    return GraphAssertion(
        build_id="build-integration",
        assertion_id=assertion_id,
        predicate=predicate,
        src_id=src_id,
        dst_id=dst_id,
        evidence=(_evidence(f"chunk-{assertion_id}"),),
        metadata={"nested": {"predicate": predicate}, "hostile": "nul:\u0000\ufffe"},
        confidence=0.9,
        method="integration-test",
        valid_from=datetime(2026, 7, 2, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_live_age_typed_assertion_contract(live_storage: PGGraphStorage) -> None:
    storage = live_storage
    entities = [_entity("source"), _entity("target"), _entity("third")]
    await storage.upsert_graph_entities(entities, contract_digest=CONTRACT_DIGEST)

    stored_entity = await storage.get_graph_entity("source")
    assert stored_entity is not None
    assert stored_entity["entity_id"] == "source"
    assert stored_entity["metadata"]["hostile"] == "nul:\u0000\ufffe"
    assert stored_entity["evidence"][0]["metadata"]["hostile"] == (
        "nul:\u0000 noncharacter:\ufffe"
    )
    assert isinstance(stored_entity["observed_from"], datetime)

    assertions = [
        _assertion("parallel-1", "depends_on", "source", "target"),
        _assertion("parallel-2", "reads_from", "source", "target"),
        _assertion("reciprocal", "feeds", "target", "source"),
    ]
    await storage.upsert_graph_assertions(assertions, contract_digest=CONTRACT_DIGEST)
    for assertion in assertions:
        stored = await storage.get_graph_assertion(assertion.assertion_id)
        assert stored is not None
        assert stored["src_id"] == assertion.src_id
        assert stored["dst_id"] == assertion.dst_id
        assert stored["predicate"] == assertion.predicate
        assert stored["metadata"]["hostile"] == "nul:\u0000\ufffe"

    moved = _assertion("parallel-1", "arrives_at", "third", "source")
    await storage.upsert_graph_assertion(moved, contract_digest=CONTRACT_DIGEST)
    stored_moved = await storage.get_graph_assertion("parallel-1")
    assert stored_moved is not None
    assert (stored_moved["src_id"], stored_moved["dst_id"]) == (
        "third",
        "source",
    )
    assert await storage.get_graph_assertion("parallel-2") is not None

    with pytest.raises(ValueError, match="missing endpoint"):
        await storage.upsert_graph_assertions(
            [
                _assertion("would-be-valid", "depends_on", "source", "target"),
                _assertion("invalid", "depends_on", "source", "missing"),
            ]
        )
    assert await storage.get_graph_assertion("would-be-valid") is None
    assert await storage.get_graph_assertion("invalid") is None

    indexes = await storage.db.query(
        "SELECT indexname FROM pg_indexes WHERE schemaname = $1",
        [storage.graph_name],
        multirows=True,
    )
    assert any(row["indexname"] == "assertion_idx_assertion_id" for row in indexes)


@pytest.mark.asyncio
async def test_live_age_concurrent_single_and_batch_writers(
    live_storage: PGGraphStorage,
) -> None:
    storage = live_storage
    await storage.upsert_graph_entities([_entity("source"), _entity("target")])

    same_id_writes = [
        storage.upsert_graph_assertion(
            _assertion("same-id", f"predicate-{index}", "source", "target")
        )
        for index in range(8)
    ]
    await asyncio.wait_for(asyncio.gather(*same_id_writes), timeout=10)

    stored = await storage.get_graph_assertion("same-id")
    assert stored is not None
    assert stored["predicate"] in {f"predicate-{index}" for index in range(8)}

    batch = [
        _assertion(f"batch-{index}", "depends_on", "source", "target")
        for index in range(20)
    ]
    batch.append(_assertion("batch-race", "batch", "source", "target"))
    await asyncio.wait_for(
        asyncio.gather(
            storage.upsert_graph_assertions(batch),
            storage.upsert_graph_assertion(
                _assertion("batch-race", "single", "target", "source")
            ),
        ),
        timeout=10,
    )

    raced = await storage.get_graph_assertion("batch-race")
    assert raced is not None
    assert raced["predicate"] in {"batch", "single"}
    for index in range(20):
        assert await storage.get_graph_assertion(f"batch-{index}") is not None

    count_sql = f"""
        SELECT count(*)::bigint AS total
        FROM {storage.graph_name}."ASSERTION"
        WHERE ag_catalog.agtype_access_operator(
                VARIADIC ARRAY[properties, '"assertion_id"'::ag_catalog.agtype]
              ) = (to_json($1::text)::text)::ag_catalog.agtype
    """
    for assertion_id in ("same-id", "batch-race"):
        count = await storage.db.query(
            count_sql,
            [assertion_id],
            with_age=True,
            graph_name=storage.graph_name,
        )
        assert count == {"total": 1}


@pytest.mark.asyncio
async def test_live_legacy_delete_waits_for_typed_assertion_write(
    live_storage: PGGraphStorage,
) -> None:
    storage = live_storage
    graph_db = storage.db
    assert isinstance(graph_db, _LiveGraphDB)
    await storage.upsert_graph_entities([_entity("source"), _entity("target")])

    graph_db.pause_after_shared_lock = True
    typed_write = asyncio.create_task(
        storage.upsert_graph_assertion(
            _assertion("delete-race", "depends_on", "source", "target")
        )
    )
    await asyncio.wait_for(graph_db.shared_lock_acquired.wait(), timeout=5)

    legacy_delete = asyncio.create_task(storage.delete_node("target"))
    await asyncio.sleep(0.05)
    assert not legacy_delete.done()

    graph_db.release_shared_lock.set()
    await asyncio.wait_for(asyncio.gather(typed_write, legacy_delete), timeout=5)

    assert await storage.get_graph_entity("target") is None
    assert await storage.get_graph_assertion("delete-race") is None
