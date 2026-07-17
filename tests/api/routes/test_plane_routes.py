from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from fastapi.testclient import TestClient

from tests.api.routes._plane_helpers import (
    CONTRACT_DIGEST,
    MANIFEST_DIGEST,
    PlaneLease as _Lease,
    make_plane_client,
)


GENERATION_ID = UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")


def _client(rag: object) -> tuple[TestClient, _Lease]:
    return make_plane_client(rag, GENERATION_ID)


def _assert_provenance_headers(response) -> None:
    assert response.headers["X-LightRAG-Plane"] == "oceanstack_product"
    assert response.headers["X-LightRAG-Generation-Id"] == str(GENERATION_ID)
    assert response.headers["X-LightRAG-Build-Id"] == "build-product-001"
    assert response.headers["X-LightRAG-Source-Revision"] == "source-abc123"
    assert response.headers["X-LightRAG-Manifest-Digest"] == MANIFEST_DIGEST


def test_plane_query_returns_answer_references_and_provenance() -> None:
    rag = SimpleNamespace(
        aquery_llm=AsyncMock(
            return_value={
                "llm_response": {"content": "answer", "is_streaming": False},
                "data": {
                    "entities": [],
                    "assertions": [],
                    "citations": [
                        {
                            "citation_id": "citation:abc123",
                            "chunk_id": "chunk-1",
                            "source_key": "src.py",
                            "source_revision": "source-abc123",
                        }
                    ],
                    "chunks": [],
                    "claims": [],
                },
            }
        )
    )
    client, lease = _client(rag)

    response = client.post(
        "/planes/oceanstack_product/query", json={"query": "What changed?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "answer",
        "citations": [
            {
                "citation_id": "citation:abc123",
                "chunk_id": "chunk-1",
                "source_key": "src.py",
                "source_revision": "source-abc123",
                "content": None,
            }
        ],
    }
    _assert_provenance_headers(response)
    lease.close.assert_awaited_once()


def test_plane_data_query_uses_generation_scoped_rag() -> None:
    payload = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "assertions": [],
            "chunks": [],
            "citations": [],
            "claims": [],
        },
        "metadata": {"query_mode": "mix"},
    }
    rag = SimpleNamespace(aquery_data=AsyncMock(return_value=payload))
    client, lease = _client(rag)

    response = client.post(
        "/planes/oceanstack_product/query/data", json={"query": "What changed?"}
    )

    assert response.status_code == 200
    assert response.json() == payload
    _assert_provenance_headers(response)
    lease.close.assert_awaited_once()


def test_plane_stream_starts_with_generation_and_closes_lease_after_output() -> None:
    async def chunks():
        yield "first"
        yield "second"

    rag = SimpleNamespace(
        aquery_llm=AsyncMock(
            return_value={
                "llm_response": {
                    "is_streaming": True,
                    "response_iterator": chunks(),
                },
                "data": {
                    "entities": [],
                    "assertions": [],
                    "chunks": [],
                    "citations": [],
                    "claims": [],
                },
            }
        )
    )
    client, lease = _client(rag)

    response = client.post(
        "/planes/oceanstack_product/query/stream",
        json={"query": "What changed?", "stream": True},
    )

    records = [json.loads(line) for line in response.text.splitlines()]
    assert records[0] == {
        "generation": {
            "plane": "oceanstack_product",
            "generation_id": str(GENERATION_ID),
            "build_id": "build-product-001",
            "source_revision": "source-abc123",
            "manifest_digest": MANIFEST_DIGEST,
        }
    }
    assert records[1:] == [
        {"citations": []},
        {"response": "first"},
        {"response": "second"},
    ]
    _assert_provenance_headers(response)
    lease.close.assert_awaited_once()


def test_graph_read_routes_preserve_typed_assertion_and_traversal_fields() -> None:
    graph = {
        "nodes": [
            {
                "id": "service",
                "labels": ["service"],
                "properties": {
                    "_lightrag_record_kind": "GraphEntity",
                    "build_id": "build-product-001",
                    "entity_id": "service",
                    "entity_type": "service",
                    "evidence": [
                        {
                            "chunk_id": "chunk-1",
                            "source_key": "src.py",
                            "source_revision": "source-abc123",
                            "metadata": {},
                        }
                    ],
                    "metadata": {},
                    "observed_from": None,
                    "observed_to": None,
                    "valid_from": None,
                    "valid_to": None,
                    "contract_digest": CONTRACT_DIGEST,
                },
            }
        ],
        "edges": [
            {
                "id": "assertion-1",
                "type": "ASSERTION",
                "source": "service",
                "target": "table",
                "properties": {
                    "_lightrag_record_kind": "GraphAssertion",
                    "build_id": "build-product-001",
                    "assertion_id": "assertion-1",
                    "predicate": "depends_on",
                    "src_id": "service",
                    "dst_id": "table",
                    "evidence": [
                        {
                            "chunk_id": "chunk-1",
                            "source_key": "src.py",
                            "source_revision": "source-abc123",
                            "metadata": {},
                        }
                    ],
                    "metadata": {},
                    "confidence": 0.9,
                    "method": "static-analysis",
                    "observed_from": None,
                    "observed_to": None,
                    "valid_from": None,
                    "valid_to": None,
                    "contract_digest": CONTRACT_DIGEST,
                },
            }
        ],
        "is_truncated": False,
    }
    rag = SimpleNamespace(get_knowledge_graph=AsyncMock(return_value=graph))
    client, lease = _client(rag)

    response = client.get(
        "/planes/oceanstack_product/graphs", params={"label": "service", "max_depth": 2}
    )

    assert response.status_code == 200
    assert response.json() == graph
    _assert_provenance_headers(response)
    lease.close.assert_awaited_once()


def test_plane_data_query_rejects_legacy_relationship_shape() -> None:
    rag = SimpleNamespace(
        aquery_data=AsyncMock(
            return_value={
                "status": "success",
                "message": "legacy",
                "data": {
                    "entities": [],
                    "relationships": [
                        {"src_id": "a", "tgt_id": "b", "description": "legacy"}
                    ],
                    "chunks": [],
                    "references": [],
                },
                "metadata": {"query_mode": "mix"},
            }
        )
    )
    client, lease = _client(rag)

    response = client.post(
        "/planes/oceanstack_product/query/data", json={"query": "What changed?"}
    )

    assert response.status_code == 503
    lease.close.assert_awaited_once()


def test_plane_data_query_rejects_records_from_another_generation() -> None:
    rag = SimpleNamespace(
        aquery_data=AsyncMock(
            return_value={
                "status": "success",
                "message": "wrong generation",
                "data": {
                    "entities": [
                        {
                            "build_id": "other-build",
                            "entity_id": "service",
                            "entity_type": "service",
                            "evidence": [
                                {
                                    "chunk_id": "chunk-1",
                                    "source_key": "src.py",
                                    "source_revision": "other-source",
                                    "metadata": {},
                                }
                            ],
                            "metadata": {},
                            "contract_digest": "f" * 64,
                            "score": 0.9,
                            "traversal_path": ["service"],
                        }
                    ],
                    "assertions": [],
                    "chunks": [],
                    "citations": [],
                    "claims": [],
                },
                "metadata": {"query_mode": "mix"},
            }
        )
    )
    client, lease = _client(rag)

    response = client.post(
        "/planes/oceanstack_product/query/data", json={"query": "What changed?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "generation query returned records outside the active generation"
    )
    lease.close.assert_awaited_once()


def test_plane_query_rejects_inconsistent_active_manifest_identity() -> None:
    rag = SimpleNamespace(aquery_data=AsyncMock())
    client, lease = _client(rag)
    lease.generation.manifest = {
        **lease.generation.manifest,
        "build_id": "different-build",
    }

    response = client.post(
        "/planes/oceanstack_product/query/data", json={"query": "What changed?"}
    )

    assert response.status_code == 503
    assert "manifest build_id" in response.json()["detail"]
    rag.aquery_data.assert_not_awaited()
    lease.close.assert_awaited_once()


def test_plane_query_rejects_naive_and_bypass_modes() -> None:
    client, _lease = _client(SimpleNamespace())

    for mode in ("naive", "bypass"):
        response = client.post(
            "/planes/oceanstack_product/query",
            json={"query": "What changed?", "mode": mode},
        )
        assert response.status_code == 422


def test_plane_router_exposes_only_fixed_planes_and_read_only_graph_operations() -> (
    None
):
    client, _lease = _client(SimpleNamespace())

    assert (
        client.post(
            "/planes/private/query", json={"query": "What changed?"}
        ).status_code
        == 422
    )
    paths = client.get("/openapi.json").json()["paths"]
    assert "/planes/{plane}/query" in paths
    assert "/planes/{plane}/query/stream" in paths
    assert "/planes/{plane}/query/data" in paths
    assert "/planes/{plane}/graph/label/list" in paths
    assert "/planes/{plane}/graph/label/popular" in paths
    assert "/planes/{plane}/graph/label/search" in paths
    assert "/planes/{plane}/graphs" in paths
    assert "/planes/{plane}/graph/entity/exists" in paths
    assert all(
        operation not in methods
        for path, methods in paths.items()
        if "/graph" in path
        for operation in ("post", "put", "patch", "delete")
    )
