"""
Unit tests for PGGraphStorage.get_knowledge_graph("*") — the full-graph view.

The wildcard ('*') branch reads the Apache AGE backing tables with native SQL
instead of cypher(): a cypher() full-graph expansion over a large selected node
set can crash a PostgreSQL backend into cluster recovery. The flow issues four
queries — a COUNT(*) over ``{graph}.base``, an undirected degree-ranked node
selection, a node read, and a directed edge read over ``{graph}._ag_label_edge``.

The degree selection counts both ``start_id`` and ``end_id`` (undirected degree)
and LEFT JOINs the base vertex table so isolated (degree-0) nodes are preserved.
The edge read stays directed (both endpoints in the selected set).

All tests mock ``PGGraphStorage._query`` and inspect the SQL it receives.
"""

import pytest
from unittest.mock import MagicMock, patch

from lightrag.kg.postgres_impl import PGGraphStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked _query method."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.global_config = {"max_graph_nodes": 1000}
    storage.db = MagicMock()
    return storage


class _QueryCapture:
    """Dispatch the four _query calls of the native '*' branch by SQL content."""

    def __init__(self, *, total_nodes, degree_rows, node_rows, edge_rows):
        self._total_nodes = total_nodes
        self._degree_rows = degree_rows
        self._node_rows = node_rows
        self._edge_rows = edge_rows
        self.count_sql = None
        self.degree_sql = None
        self.degree_params = None
        self.node_sql = None
        self.node_params = None
        self.edge_sql = None
        self.edge_params = None

    def as_side_effect(self):
        """Return an ``async def`` so AsyncMock awaits it."""

        async def fake_query(query, **kwargs):
            if "node_degrees" in query:
                self.degree_sql = query
                self.degree_params = kwargs.get("params")
                return self._degree_rows
            if "total_nodes" in query:
                self.count_sql = query
                return [{"total_nodes": self._total_nodes}]
            if "array_position" in query:
                self.node_sql = query
                self.node_params = kwargs.get("params")
                return self._node_rows
            self.edge_sql = query
            self.edge_params = kwargs.get("params")
            return self._edge_rows

        return fake_query


def _node_row(node_id, entity_id):
    return {"node_id": str(node_id), "properties": {"entity_id": entity_id}}


def _edge_row(edge_id, start_id, end_id, edge_type="DIRECTED", weight="1"):
    return {
        "edge_id": str(edge_id),
        "start_id": str(start_id),
        "end_id": str(end_id),
        "edge_type": edge_type,
        "properties": {"weight": weight},
    }


# ---------------------------------------------------------------------------
# degree node selection SQL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degree_selection_sql_is_undirected_and_preserves_isolated():
    """The degree query must count start_id + end_id and keep degree-0 nodes."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 3}, {"node_id": 2, "degree": 0}],
        node_rows=[_node_row(1, "Alice"), _node_row(2, "Bob")],
        edge_rows=[],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        await storage.get_knowledge_graph("*", max_nodes=50)

    sql = capture.degree_sql
    assert sql is not None, "degree selection query was never issued"
    assert "start_id" in sql
    assert "end_id" in sql
    assert "UNION ALL" in sql
    assert "LEFT JOIN" in sql
    assert "COALESCE" in sql
    assert "ORDER BY degree DESC" in sql
    assert "v.id ASC" in sql
    assert "-[r]->()" not in sql
    assert "OPTIONAL MATCH (n)-[r]->()" not in sql


@pytest.mark.asyncio
async def test_degree_selection_limit_is_parameterized():
    """max_nodes must be passed via params, not interpolated into the SQL."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 3}],
        node_rows=[_node_row(1, "Alice")],
        edge_rows=[],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        await storage.get_knowledge_graph("*", max_nodes=37)

    assert "LIMIT $1" in capture.degree_sql
    assert "37" not in capture.degree_sql
    assert capture.degree_params == {"limit": 37}


# ---------------------------------------------------------------------------
# native wildcard read — no cypher(), reads the AGE backing tables
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wildcard_read_uses_native_sql_not_cypher():
    """The '*' branch reads {graph}.base / _ag_label_edge, never cypher()."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 1}, {"node_id": 2, "degree": 1}],
        node_rows=[_node_row(1, "Alice"), _node_row(2, "Bob")],
        edge_rows=[_edge_row(10, 1, 2)],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        await storage.get_knowledge_graph("*", max_nodes=50)

    assert "cypher(" not in (capture.count_sql or "")
    assert "cypher(" not in (capture.node_sql or "")
    assert "cypher(" not in (capture.edge_sql or "")
    assert ".base" in capture.count_sql
    assert "_ag_label_edge" in capture.edge_sql


@pytest.mark.asyncio
async def test_isolated_node_survives_end_to_end():
    """A degree-0 node selected by the degree query reaches the final KnowledgeGraph."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 2}, {"node_id": 2, "degree": 0}],
        node_rows=[_node_row(1, "Alice"), _node_row(2, "Bob")],
        edge_rows=[],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        kg = await storage.get_knowledge_graph("*", max_nodes=50)

    assert "2" in capture.node_params["node_ids"]
    assert {node.id for node in kg.nodes} == {"1", "2"}
    assert kg.is_truncated is False


@pytest.mark.asyncio
async def test_subgraph_read_stays_directed_and_maps_endpoints():
    """The native edge read keeps direction (start_id -> end_id) and maps endpoints."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 1}, {"node_id": 2, "degree": 1}],
        node_rows=[_node_row(1, "Alice"), _node_row(2, "Bob")],
        edge_rows=[_edge_row(10, 1, 2)],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        kg = await storage.get_knowledge_graph("*", max_nodes=50)

    assert "start_id" in capture.edge_sql
    assert "end_id" in capture.edge_sql
    assert "_ag_label_edge" in capture.edge_sql
    assert len(kg.edges) == 1
    edge = kg.edges[0]
    assert edge.source == "1"
    assert edge.target == "2"


@pytest.mark.asyncio
async def test_typed_subgraph_edge_uses_assertion_id_and_keeps_direction():
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 2}, {"node_id": 2, "degree": 2}],
        node_rows=[_node_row(1, "Alice"), _node_row(2, "Bob")],
        edge_rows=[
            {
                **_edge_row(10, 2, 1, edge_type="ASSERTION"),
                "properties": {"assertion_id": "assert-1", "predicate": "feeds"},
            }
        ],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        kg = await storage.get_knowledge_graph("*", max_nodes=50)

    assert len(kg.edges) == 1
    edge = kg.edges[0]
    assert edge.id == "assert-1"
    assert edge.type == "ASSERTION"
    assert (edge.source, edge.target) == ("2", "1")
