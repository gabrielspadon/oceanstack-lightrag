"""
Unit tests for batch graph operations (PR #2910 follow-up).

Verifies:
1. BaseGraphStorage default batch methods fall back to serial single-item calls.
2. NetworkXStorage overrides batch methods with optimized in-memory operations.
4. has_nodes_batch returns only existing nodes, including newly inserted ones.
5. upsert_edges_batch and upsert_nodes_batch are idempotent (safe to call twice).
"""

import json
import time
import tempfile
import pytest
import numpy as np
from unittest.mock import AsyncMock

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLOBAL_CONFIG = {
    "embedding_batch_num": 10,
    "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
    "working_dir": "/tmp/test_batch_graph",
}


async def _raw_embedding_func(texts):
    return np.random.rand(len(texts), 10)


mock_embedding_func = EmbeddingFunc(
    embedding_dim=10,
    max_token_size=512,
    func=_raw_embedding_func,
)


def make_networkx_storage(tmp_dir: str) -> NetworkXStorage:
    config = dict(GLOBAL_CONFIG, working_dir=tmp_dir)
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="test_graph",
        workspace="test_ws",
        global_config=config,
        embedding_func=_raw_embedding_func,
    )
    return storage


def _make_node(entity_id: str, entity_type: str = "TEST") -> dict:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "description": f"Description of {entity_id}",
        "source_id": "chunk-1",
        "file_path": "test.txt",
        "created_at": int(time.time()),
    }


def _make_edge(weight: float = 1.0) -> dict:
    return {
        "weight": weight,
        "description": "test edge",
        "keywords": "test",
        "source_id": "chunk-1",
        "file_path": "test.txt",
        "created_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# 1. BaseGraphStorage default implementations delegate to single-item methods
# ---------------------------------------------------------------------------


class TestBaseGraphStorageDefaults:
    """
    Use NetworkXStorage as a concrete instance but spy on the single-item
    methods to verify the default batch implementations delegate correctly.
    """

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_calls_upsert_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            nodes = [
                ("NodeA", _make_node("NodeA")),
                ("NodeB", _make_node("NodeB")),
            ]

            call_log: list[str] = []
            original = storage.upsert_node

            async def spy(node_id, *, node_data):
                call_log.append(node_id)
                return await original(node_id, node_data=node_data)

            # Temporarily replace the optimised override with the base default

            async def base_upsert_nodes_batch(self, nodes):
                for node_id, node_data in nodes:
                    await self.upsert_node(node_id, node_data=node_data)

            storage.upsert_node = spy  # type: ignore[assignment]
            await base_upsert_nodes_batch(storage, nodes)

            assert call_log == ["NodeA", "NodeB"]

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_has_nodes_batch_calls_has_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()
            await storage.upsert_node("NodeA", node_data=_make_node("NodeA"))

            call_log: list[str] = []
            original = storage.has_node

            async def spy(node_id):
                call_log.append(node_id)
                return await original(node_id)

            async def base_has_nodes_batch(self, node_ids):
                existing = set()
                for node_id in node_ids:
                    if await self.has_node(node_id):
                        existing.add(node_id)
                return existing

            storage.has_node = spy  # type: ignore[assignment]
            result = await base_has_nodes_batch(storage, ["NodeA", "NodeB"])

            assert call_log == ["NodeA", "NodeB"]
            assert result == {"NodeA"}

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_calls_upsert_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()
            await storage.upsert_node("NodeA", node_data=_make_node("NodeA"))
            await storage.upsert_node("NodeB", node_data=_make_node("NodeB"))
            await storage.upsert_node("NodeC", node_data=_make_node("NodeC"))

            call_log: list[tuple] = []
            original = storage.upsert_edge

            async def spy(src, tgt, *, edge_data):
                call_log.append((src, tgt))
                return await original(src, tgt, edge_data=edge_data)

            async def base_upsert_edges_batch(self, edges):
                for src, tgt, edge_data in edges:
                    await self.upsert_edge(src, tgt, edge_data=edge_data)

            edges = [
                ("NodeA", "NodeB", _make_edge()),
                ("NodeB", "NodeC", _make_edge()),
            ]
            storage.upsert_edge = spy  # type: ignore[assignment]
            await base_upsert_edges_batch(storage, edges)

            assert call_log == [("NodeA", "NodeB"), ("NodeB", "NodeC")]


# ---------------------------------------------------------------------------
# 2. NetworkXStorage optimised batch implementations
# ---------------------------------------------------------------------------


class TestNetworkXBatchOperations:
    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_inserts_all_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            nodes = [(f"Entity{i}", _make_node(f"Entity{i}")) for i in range(5)]
            await storage.upsert_nodes_batch(nodes)

            for entity_id, _ in nodes:
                assert await storage.has_node(entity_id), f"{entity_id} should exist"

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            node_data = _make_node("Alpha")
            await storage.upsert_nodes_batch([("Alpha", node_data)])
            await storage.upsert_nodes_batch([("Alpha", node_data)])  # second call

            assert await storage.has_node("Alpha")
            node = await storage.get_node("Alpha")
            assert node["entity_id"] == "Alpha"

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_has_nodes_batch_returns_existing_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            await storage.upsert_nodes_batch(
                [
                    ("Present1", _make_node("Present1")),
                    ("Present2", _make_node("Present2")),
                ]
            )

            result = await storage.has_nodes_batch(["Present1", "Present2", "Missing"])
            assert result == {"Present1", "Present2"}

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_has_nodes_batch_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            result = await storage.has_nodes_batch([])
            assert result == set()

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_creates_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            await storage.upsert_nodes_batch(
                [
                    ("A", _make_node("A")),
                    ("B", _make_node("B")),
                    ("C", _make_node("C")),
                ]
            )

            edges = [
                ("A", "B", _make_edge(1.5)),
                ("B", "C", _make_edge(2.0)),
            ]
            await storage.upsert_edges_batch(edges)

            edge_ab = await storage.get_edge("A", "B")
            assert edge_ab is not None
            assert float(edge_ab["weight"]) == pytest.approx(1.5)

            edge_bc = await storage.get_edge("B", "C")
            assert edge_bc is not None
            assert float(edge_bc["weight"]) == pytest.approx(2.0)

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            await storage.upsert_nodes_batch(
                [
                    ("X", _make_node("X")),
                    ("Y", _make_node("Y")),
                ]
            )
            edge_data = _make_edge(3.0)
            await storage.upsert_edges_batch([("X", "Y", edge_data)])
            await storage.upsert_edges_batch([("X", "Y", edge_data)])  # second call

            edge = await storage.get_edge("X", "Y")
            assert edge is not None

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_updates_existing_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            original = _make_node("Node1")
            await storage.upsert_nodes_batch([("Node1", original)])

            updated = dict(original, description="Updated description")
            await storage.upsert_nodes_batch([("Node1", updated)])

            node = await storage.get_node("Node1")
            assert node["description"] == "Updated description"


# ---------------------------------------------------------------------------
class TestPostgresBatchOrdering:
    @staticmethod
    def _make_pg_storage():
        """PGGraphStorage with a fake connection capturing executed Cypher.

        The chunk-level batch paths build SQL and run it via
        ``db._run_with_retry`` instead of calling ``upsert_node`` / ``upsert_edge``
        per row, so the captured statements are how we observe dedup + ordering.
        """
        from lightrag.kg.postgres_impl import PGGraphStorage

        storage = PGGraphStorage.__new__(PGGraphStorage)
        storage.workspace = "test_ws"
        storage.namespace = "test_graph"
        storage.graph_name = "test_graph"
        storage.__post_init__()  # resolves chunk-level batch limits

        calls: list[dict] = []

        class _Tx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Conn:
            def transaction(self):
                return _Tx()

            async def execute(self, sql, *args):
                calls.append({"sql": sql, "args": args})
                return ""

        conn = _Conn()

        async def fake_run_with_retry(operation, **_kwargs):
            return await operation(conn)

        storage.db = AsyncMock()
        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
        return storage, calls

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_preserves_last_write_wins(self):
        from lightrag.kg.postgres_impl import PGGraphStorage

        storage, calls = self._make_pg_storage()

        await PGGraphStorage.upsert_nodes_batch(
            storage,
            [
                ("EntityA", _make_node("EntityA")),
                ("EntityA", dict(_make_node("EntityA"), description="latest")),
                ("EntityB", _make_node("EntityB")),
            ],
        )

        merge_calls = [c for c in calls if "MERGE (n:base" in c["sql"]]
        entity_ids = [json.loads(c["args"][0])["entity_id"] for c in merge_calls]
        # Deduped to one EntityA (moved to its last position), then EntityB.
        assert entity_ids == ["EntityA", "EntityB"]
        # EntityA carries the latest payload, not the first.
        assert '"latest"' in merge_calls[0]["sql"]
        assert "Description of EntityA" not in merge_calls[0]["sql"]

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_preserves_last_write_wins(self):
        from lightrag.kg.postgres_impl import PGGraphStorage

        storage, calls = self._make_pg_storage()

        await PGGraphStorage.upsert_edges_batch(
            storage,
            [
                ("EntityA", "EntityB", _make_edge(1.0)),
                ("EntityB", "EntityA", _make_edge(2.0)),
                ("EntityB", "EntityC", _make_edge(3.0)),
            ],
        )

        cypher_calls = [c for c in calls if "CREATE (source)-[r:DIRECTED" in c["sql"]]
        log = [
            (json.loads(c["args"][0])["src_id"], json.loads(c["args"][0])["tgt_id"])
            for c in cypher_calls
        ]
        # Canonical (LEAST, GREATEST) key order: (A,B) then (B,C); each pair keeps
        # its last-write orientation/payload.
        assert log == [("EntityB", "EntityA"), ("EntityB", "EntityC")]
        assert "2.0" in cypher_calls[0]["sql"]  # weight 2.0 won the (A,B) pair


class TestMongoBatchOrdering:
    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_uses_ordered_bulk_write(self):
        pytest.importorskip("pymongo")
        from lightrag.kg.mongo_impl import (
            MongoGraphStorage,
            DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES,
            DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH,
        )

        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.collection = AsyncMock()
        storage.workspace = "test_ws"
        storage.namespace = "test_graph"
        storage._max_upsert_payload_bytes = DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES
        storage._max_upsert_records_per_batch = (
            DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH
        )

        await MongoGraphStorage.upsert_nodes_batch(
            storage,
            [
                ("EntityA", _make_node("EntityA")),
                ("EntityA", dict(_make_node("EntityA"), description="latest")),
            ],
        )

        assert storage.collection.bulk_write.await_args.kwargs["ordered"] is True

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_uses_ordered_bulk_write(self):
        pytest.importorskip("pymongo")
        from lightrag.kg.mongo_impl import (
            MongoGraphStorage,
            DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES,
            DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH,
        )

        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.collection = AsyncMock()
        storage.edge_collection = AsyncMock()
        storage.workspace = "test_ws"
        storage.namespace = "test_graph"
        storage._max_upsert_payload_bytes = DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES
        storage._max_upsert_records_per_batch = (
            DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH
        )

        await MongoGraphStorage.upsert_edges_batch(
            storage,
            [
                ("EntityA", "EntityB", _make_edge(1.0)),
                ("EntityB", "EntityA", _make_edge(2.0)),
            ],
        )

        assert storage.edge_collection.bulk_write.await_args.kwargs["ordered"] is True

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_deduplicates_source_node_upserts(self):
        pytest.importorskip("pymongo")
        from lightrag.kg.mongo_impl import (
            MongoGraphStorage,
            DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES,
            DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH,
        )

        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.collection = AsyncMock()
        storage.edge_collection = AsyncMock()
        storage.workspace = "test_ws"
        storage.namespace = "test_graph"
        storage._max_upsert_payload_bytes = DEFAULT_MONGO_UPSERT_MAX_PAYLOAD_BYTES
        storage._max_upsert_records_per_batch = (
            DEFAULT_MONGO_UPSERT_MAX_RECORDS_PER_BATCH
        )

        await MongoGraphStorage.upsert_edges_batch(
            storage,
            [
                ("EntityA", "EntityB", _make_edge(1.0)),
                ("EntityA", "EntityC", _make_edge(2.0)),
            ],
        )

        node_ops = storage.collection.bulk_write.await_args.args[0]
        assert len(node_ops) == 1
        assert node_ops[0]._filter == {"_id": "EntityA"}
