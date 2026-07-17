"""Shared plane-route test doubles: generation lease + pool + client factory.

Used by the plane-route API tests so the lease/pool stubs live in one place
instead of being copy-pasted per test module.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.routers.plane_routes import create_plane_routes

DEFAULT_GENERATION_ID = UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
CONTRACT_DIGEST = "b" * 64
MANIFEST_DIGEST = "a" * 64


class PlaneLease:
    """Async-context lease double matching the generation pool contract."""

    def __init__(
        self, rag: object, generation_id: UUID = DEFAULT_GENERATION_ID
    ) -> None:
        self.rag = rag
        self.generation = SimpleNamespace(
            plane="oceanstack_product",
            generation_id=generation_id,
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

    async def __aenter__(self) -> PlaneLease:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.close()
        return False

    async def run(self, operation):
        return await operation()


def make_plane_client(
    rag: object, generation_id: UUID = DEFAULT_GENERATION_ID
) -> tuple[TestClient, PlaneLease]:
    """Build a TestClient over the plane routes backed by a stub lease pool."""
    lease = PlaneLease(rag, generation_id)
    pool = SimpleNamespace(acquire=AsyncMock(return_value=lease))
    app = FastAPI()
    app.include_router(create_plane_routes(pool, auth_dependency=lambda: None))
    return TestClient(app), lease
