"""Canonical relation vector identity on document-pipeline update paths."""

from unittest.mock import AsyncMock, patch

import pytest

from lightrag.operate import _merge_edges_then_upsert, _rebuild_single_relationship
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
            None,
            "entity:B",
            "entity:A",
            ["chunk:1"],
            chunk_relationships,
            None,
            config,
        )

    canonical_id = compute_mdhash_id("entity:Aentity:B", prefix="rel-")
    relationships_vdb.delete.assert_awaited_once_with([canonical_id])
