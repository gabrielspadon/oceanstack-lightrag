import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from pymongo.errors import PyMongoError, BulkWriteError, DuplicateKeyError

from lightrag.kg.mongo_impl import (
    MongoDocStatusStorage,
    MongoGraphStorage,
    _canonical_edge_endpoints,
)

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def limit(self, n: int):
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class TestMongoGraphStorage:
    def _make_storage(self):
        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.workspace = "test"
        storage.global_config = {"max_graph_nodes": 1000}
        storage._edge_collection_name = "test_edges"
        storage.collection = SimpleNamespace()
        storage.edge_collection = SimpleNamespace()
        return storage

    @pytest.mark.asyncio
    async def test_get_knowledge_graph_all_backfills_isolated_nodes_when_truncated(
        self,
    ):
        storage = self._make_storage()
        storage.collection.count_documents = AsyncMock(return_value=5)
        storage.edge_collection.aggregate = AsyncMock(
            return_value=_AsyncCursor(
                [{"_id": "A", "degree": 1}, {"_id": "B", "degree": 1}]
            )
        )

        def collection_find_side_effect(query, projection=None):
            if query == {"_id": {"$nin": ["A", "B"]}}:
                return _AsyncCursor(
                    [
                        {"_id": "C", "entity_type": "person"},
                        {"_id": "D", "entity_type": "person"},
                        {"_id": "E", "entity_type": "person"},
                    ]
                )
            if query == {"_id": {"$in": ["A", "B", "C", "D"]}}:
                return _AsyncCursor(
                    [
                        {"_id": "B", "entity_type": "person"},
                        {"_id": "D", "entity_type": "person"},
                        {"_id": "A", "entity_type": "person"},
                        {"_id": "C", "entity_type": "person"},
                    ]
                )
            raise AssertionError(f"Unexpected node query: {query}")

        storage.collection.find = Mock(side_effect=collection_find_side_effect)
        storage.edge_collection.find = Mock(
            return_value=_AsyncCursor(
                [
                    {
                        "source_node_id": "A",
                        "target_node_id": "B",
                        "relationship": "knows",
                    }
                ]
            )
        )

        result = await storage.get_knowledge_graph_all_by_degree(
            max_depth=2, max_nodes=4
        )

        assert result.is_truncated is True
        assert [node.id for node in result.nodes] == ["A", "B", "C", "D"]
        assert len(result.edges) == 1
        assert result.edges[0].source == "A"
        assert result.edges[0].target == "B"


class TestMongoEdgeKey:
    """Canonical edge-endpoint writes and duplicate-key retries."""

    def _make_storage(self):
        s = MongoGraphStorage.__new__(MongoGraphStorage)
        s.workspace = "test"
        s.namespace = "chunk_entity_relation"
        s.global_config = {}
        s._edge_collection_name = "test_edges"
        s._max_upsert_payload_bytes = 16 * 1024 * 1024
        s._max_upsert_records_per_batch = 128
        s.collection = SimpleNamespace(update_one=AsyncMock())
        s.edge_collection = SimpleNamespace()
        return s

    def test_canonical_edge_endpoints_are_direction_independent(self):
        assert _canonical_edge_endpoints("B", "A") == _canonical_edge_endpoints(
            "A", "B"
        )
        assert _canonical_edge_endpoints("B", "A") == ("A", "B")

    def test_canonical_endpoints_never_collide_across_delimiter(self):
        # Distinct pairs that a delimiter-joined key could conflate must stay
        # distinct as separate (lo, hi) fields.
        assert _canonical_edge_endpoints("A\x1fB", "C") != _canonical_edge_endpoints(
            "A", "B\x1fC"
        )

    @pytest.mark.asyncio
    async def test_upsert_edge_filters_and_sets_canonical_endpoints(self):
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock()
        await s.upsert_edge("B", "A", {"weight": 1.0, "source_id": "c1<SEP>c2"})

        args, kwargs = s.edge_collection.update_one.call_args
        filt, update = args[0], args[1]
        lo, hi = _canonical_edge_endpoints("B", "A")
        assert filt == {"edge_lo": lo, "edge_hi": hi}
        assert update["$set"]["edge_lo"] == lo
        assert update["$set"]["edge_hi"] == hi
        assert update["$set"]["source_node_id"] == "B"
        assert update["$set"]["target_node_id"] == "A"
        assert update["$set"]["source_ids"] == ["c1", "c2"]
        assert kwargs.get("upsert") is True

    @pytest.mark.asyncio
    async def test_upsert_edge_retries_once_on_duplicate_key(self):
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock(
            side_effect=[DuplicateKeyError("E11000 dup"), None]
        )
        await s.upsert_edge("A", "B", {"weight": 1.0})
        assert s.edge_collection.update_one.await_count == 2

    @pytest.mark.asyncio
    async def test_upsert_edge_reraises_on_persistent_duplicate(self):
        s = self._make_storage()
        s.edge_collection.update_one = AsyncMock(
            side_effect=DuplicateKeyError("E11000 dup")
        )
        with pytest.raises(DuplicateKeyError):
            await s.upsert_edge("A", "B", {"weight": 1.0})
        assert s.edge_collection.update_one.await_count == 2

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_dedupes_reciprocal_and_sets_endpoints(self):
        s = self._make_storage()
        calls = []

        async def fake_bulk(collection, ops, **kwargs):
            calls.append((collection, ops))

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            await s.upsert_edges_batch(
                [("A", "B", {"weight": 1.0}), ("B", "A", {"weight": 2.0})]
            )

        # Last call is the edge bulk (first is the node-placeholder bulk).
        edge_collection, edge_ops = calls[-1]
        assert edge_collection is s.edge_collection
        assert len(edge_ops) == 1  # reciprocal pair collapsed to one op
        op, _bytes, _logid = edge_ops[0]
        lo, hi = _canonical_edge_endpoints("A", "B")
        assert op._filter == {"edge_lo": lo, "edge_hi": hi}
        assert op._doc["$set"]["edge_lo"] == lo
        assert op._doc["$set"]["edge_hi"] == hi
        assert op._doc["$set"]["weight"] == 2.0  # last-write-wins

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_retries_on_duplicate_bulk_error(self):
        s = self._make_storage()
        seq = []

        async def fake_bulk(collection, ops, **kwargs):
            seq.append(collection)
            # Fail the first edge bulk with an all-11000 BulkWriteError.
            if collection is s.edge_collection and seq.count(s.edge_collection) == 1:
                raise BulkWriteError({"writeErrors": [{"code": 11000}]})

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])

        # node bulk + edge bulk (raises) + edge bulk (retry succeeds)
        assert seq.count(s.edge_collection) == 2

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_reraises_non_duplicate_bulk_error(self):
        s = self._make_storage()

        async def fake_bulk(collection, ops, **kwargs):
            if collection is s.edge_collection:
                raise BulkWriteError({"writeErrors": [{"code": 121}]})

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            with pytest.raises(BulkWriteError):
                await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_surfaces_non_dup_error_hidden_behind_dup(self):
        """Under ordered=True the bulk aborts at the first failing op, so a
        non-11000 error sitting *behind* a leading 11000 race is not reported on
        the first pass. It must still surface — on the retry, where the leading
        dup has resolved to an update and the real error now leads."""
        s = self._make_storage()
        edge_calls = []

        async def fake_bulk(collection, ops, **kwargs):
            if collection is s.edge_collection:
                edge_calls.append(collection)
                if len(edge_calls) == 1:
                    # First pass: ordered abort reports only the leading dup race.
                    raise BulkWriteError({"writeErrors": [{"code": 11000}]})
                # Retry: dup resolved to an update, the hidden error now leads.
                raise BulkWriteError({"writeErrors": [{"code": 121}]})

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            with pytest.raises(BulkWriteError) as exc:
                await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])
        # Retried once (dup-only first pass), then the non-dup error re-raised.
        assert len(edge_calls) == 2
        assert exc.value.details["writeErrors"][0]["code"] == 121

    @pytest.mark.asyncio
    async def test_upsert_edges_batch_reraises_write_concern_error(self):
        """A writeConcern-only BulkWriteError (empty writeErrors) is a durability
        failure — it must surface, not be masked by the duplicate-key retry."""
        s = self._make_storage()
        edge_calls = []

        async def fake_bulk(collection, ops, **kwargs):
            if collection is s.edge_collection:
                edge_calls.append(collection)
                raise BulkWriteError(
                    {"writeErrors": [], "writeConcernErrors": [{"code": 64}]}
                )

        with patch(
            "lightrag.kg.mongo_impl._run_batched_bulk_write",
            new=AsyncMock(side_effect=fake_bulk),
        ):
            with pytest.raises(BulkWriteError):
                await s.upsert_edges_batch([("A", "B", {"weight": 1.0})])
        assert len(edge_calls) == 1  # not retried

    @pytest.mark.asyncio
    async def test_edge_index_creation_rejects_missing_canonical_endpoints(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value={"_id": 1})
        s.edge_collection.list_indexes = AsyncMock()

        with pytest.raises(ValueError, match="canonical endpoint"):
            await s.create_edge_indexes_if_not_exists()

        s.edge_collection.list_indexes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edge_index_creation_builds_strict_unique_index(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value=None)
        s.edge_collection.list_indexes = AsyncMock(
            return_value=SimpleNamespace(
                to_list=AsyncMock(return_value=[{"name": "_id_"}])
            )
        )
        s.edge_collection.create_index = AsyncMock()

        await s.create_edge_indexes_if_not_exists()

        s.edge_collection.create_index.assert_awaited_once_with(
            [("edge_lo", 1), ("edge_hi", 1)],
            name="test_edge_endpoints_unique",
            unique=True,
        )


class TestMongoEdgeReadCanonicalFilter:
    """Exact single-edge reads/deletes use the canonical (edge_lo, edge_hi)
    filter so they hit the compound unique index instead of the bidirectional
    ``$or``. Per-node scans (node_degree/get_node_edges/delete_node) intentionally
    keep ``$or`` — edge_hi is not a compound-index prefix, so they gain nothing."""

    def _make_storage(self, max_delete_records_per_batch=1000):
        s = MongoGraphStorage.__new__(MongoGraphStorage)
        s.workspace = "test"
        s.namespace = "chunk_entity_relation"
        s._edge_collection_name = "test_edges"
        s._max_delete_records_per_batch = max_delete_records_per_batch
        s.edge_collection = SimpleNamespace()
        return s

    @pytest.mark.asyncio
    async def test_has_edge_uses_canonical_filter(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value={"_id": "x"})
        assert await s.has_edge("B", "A") is True

        filt, projection = s.edge_collection.find_one.call_args[0]
        lo, hi = _canonical_edge_endpoints("B", "A")
        assert filt == {"edge_lo": lo, "edge_hi": hi}
        assert projection == {"_id": 1}

    @pytest.mark.asyncio
    async def test_has_edge_direction_independent(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value=None)
        await s.has_edge("A", "B")
        forward = s.edge_collection.find_one.call_args[0][0]
        await s.has_edge("B", "A")
        backward = s.edge_collection.find_one.call_args[0][0]
        assert forward == backward

    @pytest.mark.asyncio
    async def test_get_edge_uses_canonical_filter(self):
        s = self._make_storage()
        s.edge_collection.find_one = AsyncMock(return_value={"weight": 1.0})
        await s.get_edge("B", "A")
        filt = s.edge_collection.find_one.call_args[0][0]
        lo, hi = _canonical_edge_endpoints("B", "A")
        assert filt == {"edge_lo": lo, "edge_hi": hi}

    @pytest.mark.asyncio
    async def test_remove_edges_uses_canonical_pairs_and_collapses_reciprocals(self):
        s = self._make_storage()
        s.edge_collection.delete_many = AsyncMock()
        # (A,B) and its reciprocal (B,A) must collapse to a single clause.
        await s.remove_edges([("A", "B"), ("B", "A"), ("C", "D")])

        query = s.edge_collection.delete_many.call_args[0][0]
        lo_ab, hi_ab = _canonical_edge_endpoints("A", "B")
        lo_cd, hi_cd = _canonical_edge_endpoints("C", "D")
        assert query == {
            "$or": [
                {"edge_lo": lo_ab, "edge_hi": hi_ab},
                {"edge_lo": lo_cd, "edge_hi": hi_cd},
            ]
        }

    @pytest.mark.asyncio
    async def test_remove_edges_empty_is_noop(self):
        s = self._make_storage()
        s.edge_collection.delete_many = AsyncMock()
        await s.remove_edges([])
        s.edge_collection.delete_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_remove_edges_chunks_large_or_by_record_cap(self):
        # 5 distinct edges with a cap of 2 → 3 delete_many calls (2 + 2 + 1),
        # and every input edge is covered exactly once across the chunks.
        s = self._make_storage(max_delete_records_per_batch=2)
        s.edge_collection.delete_many = AsyncMock()
        edges = [(f"n{i}", f"n{i + 1}") for i in range(5)]
        await s.remove_edges(edges)

        calls = s.edge_collection.delete_many.call_args_list
        assert [len(c[0][0]["$or"]) for c in calls] == [2, 2, 1]
        clauses = [clause for c in calls for clause in c[0][0]["$or"]]
        expected = [
            {"edge_lo": lo, "edge_hi": hi}
            for lo, hi in (_canonical_edge_endpoints(src, tgt) for src, tgt in edges)
        ]
        assert clauses == expected

    @pytest.mark.asyncio
    async def test_remove_edges_non_positive_cap_disables_chunking(self):
        # A non-positive cap means "no record-count splitting": one delete_many.
        s = self._make_storage(max_delete_records_per_batch=0)
        s.edge_collection.delete_many = AsyncMock()
        edges = [(f"n{i}", f"n{i + 1}") for i in range(5)]
        await s.remove_edges(edges)

        assert s.edge_collection.delete_many.await_count == 1
        assert len(s.edge_collection.delete_many.call_args[0][0]["$or"]) == 5


class TestMongoDocStatusLookup:
    @pytest.mark.asyncio
    async def test_doc_status_index_setup_creates_current_indexes_only(self):
        storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
        storage.workspace = "test"
        storage._collection_name = "test_doc_status"
        storage._data = SimpleNamespace(
            list_indexes=AsyncMock(
                return_value=SimpleNamespace(
                    to_list=AsyncMock(return_value=[{"name": "_id_"}])
                )
            ),
            create_index=AsyncMock(),
        )

        await storage.create_indexes_if_not_exists()

        assert storage._data.create_index.await_count == 9

    """Cover the Mongo-native overrides for basename / content_hash lookups."""

    def _make_storage(self):
        storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
        storage.workspace = "test"
        storage.global_config = {}
        storage._collection_name = "test_doc_status"
        storage._data = SimpleNamespace()
        return storage

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_returns_tuple_on_hit(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(
            return_value={
                "_id": "doc-1",
                "file_path": "report.pdf",
                "status": "processed",
            }
        )

        result = await storage.get_doc_by_file_basename("report.pdf")

        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-1"
        assert doc["file_path"] == "report.pdf"
        storage._data.find_one.assert_awaited_once_with({"file_path": "report.pdf"})

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_empty_returns_none_without_query(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock()

        assert await storage.get_doc_by_file_basename("") is None
        storage._data.find_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_unknown_source_sentinel(self):
        # Lookup for the sentinel must not match real rows that happen to have
        # file_path == "unknown_source".
        storage = self._make_storage()
        storage._data.find_one = AsyncMock()

        assert await storage.get_doc_by_file_basename("unknown_source") is None
        storage._data.find_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_file_basename_miss_returns_none(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(return_value=None)

        assert await storage.get_doc_by_file_basename("missing.pdf") is None

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_returns_tuple_on_hit(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(
            return_value={
                "_id": "doc-1",
                "file_path": "report.pdf",
                "content_hash": "abc123",
                "status": "processed",
            }
        )

        result = await storage.get_doc_by_content_hash("abc123")

        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-1"
        assert doc["content_hash"] == "abc123"
        storage._data.find_one.assert_awaited_once_with({"content_hash": "abc123"})

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_empty_returns_none_without_query(self):
        # Empty hash must short-circuit before querying storage.
        storage = self._make_storage()
        storage._data.find_one = AsyncMock()

        assert await storage.get_doc_by_content_hash("") is None
        storage._data.find_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_doc_by_content_hash_miss_returns_none(self):
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(return_value=None)

        assert await storage.get_doc_by_content_hash("zzz999") is None

    @pytest.mark.asyncio
    async def test_lookup_swallows_pymongo_error_and_returns_none(self):
        # PyMongoError must not propagate to the caller; the dedup path treats
        # a storage failure as "no match" and the error is logged instead.
        storage = self._make_storage()
        storage._data.find_one = AsyncMock(side_effect=PyMongoError("boom"))

        assert await storage.get_doc_by_file_basename("report.pdf") is None
        assert await storage.get_doc_by_content_hash("abc123") is None
