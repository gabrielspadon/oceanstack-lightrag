"""Dangling relations are dropped instead of minting UNKNOWN placeholder entities."""

from unittest.mock import AsyncMock, patch

import pytest

from lightrag.operate import _merge_edges_then_upsert, _rebuild_single_relationship

pytestmark = pytest.mark.offline


_CONFIG = {
    "source_ids_limit_method": "KEEP",
    "max_source_ids_per_relation": 10,
    "max_source_ids_per_entity": 10,
    "max_file_paths": 10,
}


def _edge_fragment() -> dict:
    return {
        "description": "references",
        "keywords": "schema",
        "weight": 1.0,
        "source_id": "chunk:1",
        "file_path": "oceanstack/schema.sql",
        "timestamp": 1,
    }


@pytest.mark.asyncio
async def test_merge_edges_drops_relation_with_missing_endpoint():
    graph = AsyncMock()
    graph.has_edge.return_value = False
    # src exists, tgt was never extracted
    graph.get_node.side_effect = lambda node_id: (
        {"entity_type": "Table", "source_id": "chunk:1"}
        if node_id == "entity:A"
        else None
    )

    with patch(
        "lightrag.operate._handle_entity_relation_summary",
        new=AsyncMock(return_value=("references", False)),
    ):
        result = await _merge_edges_then_upsert(
            "entity:A",
            "entity:MISSING",
            [_edge_fragment()],
            graph,
            AsyncMock(),
            None,
            _CONFIG,
        )

    assert result is None
    graph.upsert_node.assert_not_awaited()
    graph.upsert_edge.assert_not_awaited()


@pytest.mark.asyncio
async def test_rebuild_relationship_drops_edge_with_missing_endpoint():
    graph = AsyncMock()
    graph.get_edge.return_value = {
        "description": "references",
        "keywords": "schema",
        "weight": 1.0,
        "source_id": "chunk:1",
        "file_path": "oceanstack/schema.sql",
    }
    graph.has_node.return_value = False

    chunk_relationships = {"chunk:1": {("entity:A", "entity:B"): [_edge_fragment()]}}

    with patch(
        "lightrag.operate._handle_entity_relation_summary",
        new=AsyncMock(return_value=("references", False)),
    ):
        await _rebuild_single_relationship(
            graph,
            AsyncMock(),
            "entity:A",
            "entity:B",
            ["chunk:1"],
            chunk_relationships,
            None,
            _CONFIG,
        )

    graph.upsert_node.assert_not_awaited()
    graph.upsert_edge.assert_not_awaited()
