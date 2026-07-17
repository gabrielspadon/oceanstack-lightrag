"""Canonical relation vector identity on document-pipeline update paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.operate import _merge_edges_then_upsert, _rebuild_single_relationship
from lightrag.lightrag import LightRAG
from lightrag.utils import compute_mdhash_id

pytestmark = pytest.mark.offline


def _graph() -> AsyncMock:
    graph = AsyncMock()
    graph.has_edge.return_value = False
    graph.get_edge.return_value = {
        "description": "references",
        "keywords": "schema",
        "weight": 1.0,
        "source_id": "chunk:1",
        "file_path": "schema.sql",
    }
    graph.has_node.return_value = True
    graph.get_node.return_value = {
        "description": "table",
        "entity_type": "Table",
        "source_id": "chunk:1",
        "file_path": "schema.sql",
    }
    return graph


def _edge_fragment() -> dict:
    return {
        "description": "references",
        "keywords": "schema",
        "weight": 1.0,
        "source_id": "chunk:1",
        "file_path": "schema.sql",
        "timestamp": 1,
    }


class _PipelineLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


@pytest.mark.asyncio
async def test_merge_edge_update_deletes_only_canonical_relation_id():
    graph = _graph()
    relationships_vdb = AsyncMock()
    config = {
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_relation": 10,
        "max_source_ids_per_entity": 10,
        "max_file_paths": 10,
    }

    with patch(
        "lightrag.operate._handle_entity_relation_summary",
        new=AsyncMock(return_value=("references", False)),
    ):
        await _merge_edges_then_upsert(
            "entity:B",
            "entity:A",
            [_edge_fragment()],
            graph,
            relationships_vdb,
            None,
            config,
        )

    canonical_id = compute_mdhash_id("entity:Aentity:B", prefix="rel-")
    relationships_vdb.delete.assert_awaited_once_with([canonical_id])


@pytest.mark.asyncio
async def test_cached_rebuild_deletes_only_canonical_relation_id():
    graph = _graph()
    relationships_vdb = AsyncMock()
    config = {
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_relation": 10,
        "max_file_paths": 10,
    }
    chunk_relationships = {"chunk:1": {("entity:B", "entity:A"): [_edge_fragment()]}}

    with patch(
        "lightrag.operate._handle_entity_relation_summary",
        new=AsyncMock(return_value=("references", False)),
    ):
        await _rebuild_single_relationship(
            graph,
            relationships_vdb,
            "entity:B",
            "entity:A",
            ["chunk:1"],
            chunk_relationships,
            None,
            config,
        )

    canonical_id = compute_mdhash_id("entity:Aentity:B", prefix="rel-")
    relationships_vdb.delete.assert_awaited_once_with([canonical_id])


@pytest.mark.asyncio
async def test_document_purge_deletes_only_canonical_relation_id():
    full_entities = AsyncMock()
    full_entities.get_by_id.return_value = None
    full_relations = AsyncMock()
    full_relations.get_by_id.return_value = {
        "relation_pairs": [("entity:B", "entity:A")]
    }
    graph = AsyncMock()
    graph.get_edges_batch.return_value = {
        ("entity:B", "entity:A"): {
            "source_id": "chunk:1",
            "source": "entity:B",
            "target": "entity:A",
        }
    }
    relationships_vdb = AsyncMock()
    rag = SimpleNamespace(
        full_entities=full_entities,
        full_relations=full_relations,
        chunk_entity_relation_graph=graph,
        entity_chunks=None,
        relation_chunks=None,
        chunks_vdb=AsyncMock(),
        text_chunks=AsyncMock(),
        relationships_vdb=relationships_vdb,
        entities_vdb=AsyncMock(),
        _insert_done=AsyncMock(),
    )

    await LightRAG._purge_doc_chunks_and_kg(
        rag,
        "doc:1",
        ["chunk:1"],
        pipeline_status={"latest_message": "", "history_messages": []},
        pipeline_status_lock=_PipelineLock(),
    )

    canonical_id = compute_mdhash_id("entity:Aentity:B", prefix="rel-")
    relationships_vdb.delete.assert_awaited_once_with([canonical_id])
