"""
Verify the plane-scoped query endpoint response types.

Ensures:
  - /planes/{plane}/query        → application/json (no streaming)
  - /planes/{plane}/query/stream → application/x-ndjson
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.routers.plane_routes import create_plane_routes


GENERATION_ID = UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa02")
CONTRACT_DIGEST = "b" * 64
MANIFEST_DIGEST = "a" * 64


class _Lease:
    def __init__(self, rag: object) -> None:
        self.rag = rag
        self.generation = SimpleNamespace(
            plane="oceanstack_product",
            generation_id=GENERATION_ID,
            build_id="build-product-001",
            contract_digest=CONTRACT_DIGEST,
            manifest_digest=MANIFEST_DIGEST,
            manifest={
                "build_id": "build-product-001",
                "digest": MANIFEST_DIGEST,
                "plane": "oceanstack_product",
                "source_revision": "source-abc123",
            },
            metadata={"source_revision": "source-abc123"},
        )
        self.close = AsyncMock()

    async def __aenter__(self) -> "_Lease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.close()
        return False

    async def run(self, operation):
        return await operation()


def _client(rag: object) -> TestClient:
    lease = _Lease(rag)
    pool = SimpleNamespace(acquire=AsyncMock(return_value=lease))
    app = FastAPI()
    app.include_router(create_plane_routes(pool, auth_dependency=lambda: None))
    return TestClient(app)


def _openapi_content(client: TestClient, path: str) -> dict:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    op = paths.get(path, {})
    assert op, f"{path} must be in OpenAPI paths"
    return op.get("post", {}).get("responses", {}).get("200", {}).get("content", {})


class TestPlaneQueryJsonOnly:
    """The plane query endpoint must stay JSON-only."""

    def test_openapi_spec_declares_json_response(self):
        client = _client(SimpleNamespace())
        content = _openapi_content(client, "/planes/{plane}/query")
        assert "application/json" in content, (
            "/planes/{plane}/query must declare application/json in OpenAPI spec"
        )
        assert "application/x-ndjson" not in content, (
            "/planes/{plane}/query must NOT declare application/x-ndjson — "
            "streaming belongs to /planes/{plane}/query/stream"
        )

    def test_query_route_exists_and_accepts_post(self):
        rag = SimpleNamespace(
            aquery_llm=AsyncMock(
                return_value={
                    "llm_response": {"content": "answer", "is_streaming": False},
                    "data": {
                        "entities": [],
                        "assertions": [],
                        "citations": [],
                        "chunks": [],
                        "claims": [],
                    },
                }
            )
        )
        client = _client(rag)
        response = client.post(
            "/planes/oceanstack_product/query", json={"query": "test"}
        )
        assert response.status_code not in (404, 405), (
            "/planes/{plane}/query route must exist and accept POST"
        )


class TestPlaneQueryStreamRoute:
    """The plane stream endpoint must serve application/x-ndjson."""

    def test_openapi_spec_declares_ndjson_response(self):
        client = _client(SimpleNamespace())
        content = _openapi_content(client, "/planes/{plane}/query/stream")
        assert "application/x-ndjson" in content, (
            "/planes/{plane}/query/stream must declare application/x-ndjson "
            "in OpenAPI spec"
        )

    def test_stream_response_has_ndjson_content_type(self):
        """Even without a real LLM, the streaming response must carry the
        correct media type header."""
        rag = SimpleNamespace(
            aquery_llm=AsyncMock(
                return_value={
                    "llm_response": {"content": "answer", "is_streaming": False},
                    "data": {
                        "entities": [],
                        "assertions": [],
                        "citations": [],
                        "chunks": [],
                        "claims": [],
                    },
                }
            )
        )
        client = _client(rag)
        response = client.post(
            "/planes/oceanstack_product/query/stream", json={"query": "test"}
        )
        content_type = response.headers.get("content-type", "")
        assert "application/x-ndjson" in content_type, (
            "/planes/{plane}/query/stream must return application/x-ndjson, "
            f"got: {content_type}"
        )
