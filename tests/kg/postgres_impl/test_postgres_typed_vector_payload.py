"""Typed knowledge-graph vector payload persistence for fresh PostgreSQL tables."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import asyncpg
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from lightrag.kg.postgres_impl import PGVectorStorage, TABLES
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


async def _fake_cpu_embedding(texts, **kwargs):
    return np.zeros((len(texts), 3), dtype=np.float32)


def _storage(namespace: str) -> PGVectorStorage:
    storage = PGVectorStorage(
        namespace=namespace,
        workspace="typed_ws",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.5,
            },
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=3,
            func=_fake_cpu_embedding,
            model_name="fake-cpu",
        ),
    )
    storage._flush_lock = asyncio.Lock()
    return storage


def _payload_value(values: tuple) -> dict:
    json_values = [
        json.loads(value)
        for value in values
        if isinstance(value, str) and value.startswith("{")
    ]
    assert len(json_values) == 1
    return json_values[0]


@pytest.mark.parametrize(
    "table_name",
    ["LIGHTRAG_VDB_CHUNKS", "LIGHTRAG_VDB_ENTITY", "LIGHTRAG_VDB_RELATION"],
)
def test_fresh_vector_tables_have_jsonb_payload(table_name):
    assert "payload JSONB NOT NULL" in TABLES[table_name]["ddl"]


def test_fresh_graph_vector_tables_use_unbounded_caller_identifiers():
    assert "id TEXT" in TABLES["LIGHTRAG_DOC_CHUNKS"]["ddl"]
    assert "full_doc_id TEXT" in TABLES["LIGHTRAG_DOC_CHUNKS"]["ddl"]
    for table_name in (
        "LIGHTRAG_VDB_CHUNKS",
        "LIGHTRAG_VDB_ENTITY",
        "LIGHTRAG_VDB_RELATION",
    ):
        ddl = TABLES[table_name]["ddl"]
        assert "id TEXT" in ddl
        assert "id VARCHAR" not in ddl
    assert "entity_name TEXT" in TABLES["LIGHTRAG_VDB_ENTITY"]["ddl"]
    relation_ddl = TABLES["LIGHTRAG_VDB_RELATION"]["ddl"]
    assert "source_id TEXT" in relation_ddl
    assert "target_id TEXT" in relation_ddl
    assert "chunk_ids TEXT[]" in TABLES["LIGHTRAG_VDB_ENTITY"]["ddl"]
    assert "chunk_ids TEXT[]" in relation_ddl


def test_entity_upsert_persists_typed_payload():
    storage = _storage(NameSpace.VECTOR_STORE_ENTITIES)
    sql, values = storage._upsert_entities(
        {
            "__id__": "entity:A",
            "__vector__": np.zeros(3, dtype=np.float32),
            "entity_name": "entity:A",
            "content": "canonical entity",
            "source_id": "chunk:1",
            "file_path": "oceanstack/schema.sql",
            "build_id": "build:1",
            "contract_digest": "a" * 64,
            "entity_id": "entity:A",
            "entity_type": "Table",
            "evidence_chunk_ids": ["chunk:1"],
            "metadata": {"schema": "ais"},
        },
        datetime(2026, 1, 1),
    )

    assert "payload" in sql
    assert _payload_value(values) == {
        "build_id": "build:1",
        "contract_digest": "a" * 64,
        "entity_id": "entity:A",
        "entity_type": "Table",
        "evidence_chunk_ids": ["chunk:1"],
        "metadata": {"schema": "ais"},
    }


def test_relation_upsert_persists_assertion_identity_and_evidence():
    storage = _storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    sql, values = storage._upsert_relationships(
        {
            "__id__": "assertion:1",
            "__vector__": np.zeros(3, dtype=np.float32),
            "src_id": "entity:A",
            "tgt_id": "entity:B",
            "source_id": "chunk:1",
            "content": "canonical assertion",
            "file_path": "oceanstack/schema.sql",
            "build_id": "build:1",
            "contract_digest": "b" * 64,
            "assertion_id": "assertion:1",
            "predicate": "references",
            "src_entity_id": "entity:A",
            "dst_entity_id": "entity:B",
            "evidence": [{"chunk_id": "chunk:1"}],
            "evidence_chunk_ids": ["chunk:1"],
            "metadata": {"constraint": "fk_vessel"},
        },
        datetime(2026, 1, 1),
    )

    assert "payload" in sql
    assert _payload_value(values) == {
        "assertion_id": "assertion:1",
        "build_id": "build:1",
        "contract_digest": "b" * 64,
        "dst_entity_id": "entity:B",
        "evidence": [{"chunk_id": "chunk:1"}],
        "evidence_chunk_ids": ["chunk:1"],
        "metadata": {"constraint": "fk_vessel"},
        "predicate": "references",
        "src_entity_id": "entity:A",
    }


def test_chunk_upsert_persists_caller_source_identity():
    storage = _storage(NameSpace.VECTOR_STORE_CHUNKS)
    sql, values = storage._upsert_chunks(
        {
            "__id__": "chunk:1",
            "__vector__": np.zeros(3, dtype=np.float32),
            "tokens": 2,
            "chunk_order_index": 0,
            "full_doc_id": "build:1",
            "content": "source text",
            "file_path": "oceanstack/schema.sql",
            "build_id": "build:1",
            "contract_digest": "c" * 64,
            "source_key": "oceanstack/schema.sql",
            "source_revision": "abc123",
            "metadata": {"line": 10},
        },
        datetime(2026, 1, 1),
    )

    assert "payload" in sql
    assert _payload_value(values) == {
        "build_id": "build:1",
        "contract_digest": "c" * 64,
        "metadata": {"line": 10},
        "source_key": "oceanstack/schema.sql",
        "source_revision": "abc123",
    }


@pytest.mark.asyncio
async def test_committed_vector_read_restores_typed_payload():
    storage = _storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    storage.db = AsyncMock()
    storage.db.query.return_value = {
        "id": "assertion:1",
        "source_id": "entity:A",
        "target_id": "entity:B",
        "content": "canonical assertion",
        "content_vector": np.zeros(3, dtype=np.float32),
        "payload": json.dumps(
            {
                "assertion_id": "assertion:1",
                "predicate": "references",
                "evidence_chunk_ids": ["chunk:1"],
            }
        ),
        "created_at": 1,
    }

    result = await storage.get_by_id("assertion:1")

    assert result is not None
    assert result["assertion_id"] == "assertion:1"
    assert result["predicate"] == "references"
    assert result["evidence_chunk_ids"] == ["chunk:1"]
    assert "payload" not in result
    assert "content_vector" not in result


@pytest.mark.parametrize("corrupt_payload", ["{not-json", ["not", "an", "object"]])
@pytest.mark.asyncio
async def test_committed_vector_read_rejects_corrupt_typed_payload(corrupt_payload):
    storage = _storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    storage.db = AsyncMock()
    storage.db.query.return_value = {
        "id": "assertion:1",
        "source_id": "entity:A",
        "target_id": "entity:B",
        "content": "canonical assertion",
        "content_vector": np.zeros(3, dtype=np.float32),
        "payload": corrupt_payload,
        "created_at": 1,
    }

    with pytest.raises(ValueError, match="vector payload"):
        await storage.get_by_id("assertion:1")


@pytest.mark.asyncio
async def test_vector_query_restores_typed_payload():
    storage = _storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    storage.db = AsyncMock()
    storage.db.vector_index_type = "HNSW"
    storage.db.query.return_value = [
        {
            "id": "assertion:1",
            "src_id": "entity:A",
            "tgt_id": "entity:B",
            "payload": {
                "assertion_id": "assertion:1",
                "predicate": "references",
                "evidence_chunk_ids": ["chunk:1"],
            },
            "created_at": 1,
        }
    ]

    result = await storage.query("unused", 5, query_embedding=[0.0, 0.0, 0.0])

    assert result == [
        {
            "id": "assertion:1",
            "src_id": "entity:A",
            "tgt_id": "entity:B",
            "assertion_id": "assertion:1",
            "predicate": "references",
            "evidence_chunk_ids": ["chunk:1"],
            "created_at": 1,
        }
    ]
    sql = storage.db.query.await_args.args[0]
    assert "payload" in sql


class _LiveVectorDB:
    vector_index_type = "HNSW"

    def __init__(self, connection: asyncpg.Connection) -> None:
        self.connection = connection

    async def _run_with_retry(self, operation, **_kwargs):
        return await operation(self.connection)

    async def query(self, sql, params=None, *, multirows=False, **_kwargs):
        rows = await self.connection.fetch(sql, *(params or ()))
        converted = [dict(row) for row in rows]
        return converted if multirows else (converted[0] if converted else None)


def _connection_kwargs(database: str) -> dict:
    kwargs = {
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
async def live_relation_table():
    database = os.getenv("LIGHTRAG_PG_TEST_DATABASE")
    if not database:
        pytest.skip("set LIGHTRAG_PG_TEST_DATABASE to an isolated test database")
    if database == "oceanstack" or "test" not in database.casefold():
        pytest.fail("LIGHTRAG_PG_TEST_DATABASE must name an isolated test database")

    connection = await asyncpg.connect(**_connection_kwargs(database))
    await register_vector(connection)
    await connection.execute(
        """
        CREATE TEMP TABLE lightrag_vdb_relation_typed_it (
            id TEXT,
            workspace VARCHAR(255),
            source_id TEXT,
            target_id TEXT,
            content TEXT,
            content_vector VECTOR(3),
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            chunk_ids TEXT[] NULL,
            file_path TEXT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            PRIMARY KEY (workspace, id)
        )
        """
    )
    try:
        yield _LiveVectorDB(connection), "lightrag_vdb_relation_typed_it"
    finally:
        await connection.close()


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.asyncio
async def test_live_flush_and_new_instance_preserve_long_typed_identity(
    live_relation_table,
):
    db, table_name = live_relation_table
    long_assertion_id = "assertion:" + "a" * 300
    long_chunk_id = "chunk:" + "c" * 300
    payload = {
        "content": "canonical assertion",
        "src_id": "entity:A",
        "tgt_id": "entity:B",
        "source_id": long_chunk_id,
        "file_path": "oceanstack/schema.sql",
        "build_id": "build:1",
        "contract_digest": "d" * 64,
        "assertion_id": long_assertion_id,
        "predicate": "references",
        "src_entity_id": "entity:A",
        "dst_entity_id": "entity:B",
        "evidence": [{"chunk_id": long_chunk_id}],
        "evidence_chunk_ids": [long_chunk_id],
        "metadata": {"constraint": "fk_vessel"},
    }

    writer = _storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    writer.table_name = table_name
    writer.db = db
    await writer.upsert({long_assertion_id: payload})
    await writer.index_done_callback()

    reader = _storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    reader.table_name = table_name
    reader.db = db
    stored = await reader.get_by_id(long_assertion_id)

    assert stored is not None
    assert stored["id"] == long_assertion_id
    assert stored["assertion_id"] == long_assertion_id
    assert stored["predicate"] == "references"
    assert stored["evidence_chunk_ids"] == [long_chunk_id]
    assert stored["metadata"] == {"constraint": "fk_vessel"}
