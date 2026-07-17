from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from lightrag.api.generation_pool import (
    GenerationPool,
    GenerationUnavailableError,
    UnsupportedPlaneError,
)
from lightrag.generation import (
    GenerationFenceKind,
    current_generation_operation_fence,
)


def _generation(plane: str, generation_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        plane=plane,
        generation_id=UUID(generation_id),
        workspace=f"kg_{plane}_{UUID(generation_id).hex}",
    )


@pytest.mark.asyncio
async def test_pool_rejects_planes_outside_the_public_contract() -> None:
    registry = SimpleNamespace(resolve_active=AsyncMock())
    pool = GenerationPool(registry, AsyncMock())

    with pytest.raises(UnsupportedPlaneError, match="unsupported plane"):
        await pool.acquire("private")

    registry.resolve_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_pool_reports_a_plane_without_an_active_generation() -> None:
    registry = SimpleNamespace(resolve_active=AsyncMock(return_value=None))
    pool = GenerationPool(registry, AsyncMock())

    with pytest.raises(GenerationUnavailableError, match="no active generation"):
        await pool.acquire("oceanstack_dev")


@pytest.mark.asyncio
async def test_pool_singleflights_concurrent_generation_construction() -> None:
    generation = _generation(
        "oceanstack_product", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"
    )
    registry = SimpleNamespace(resolve_active=AsyncMock(return_value=generation))
    release_factory = asyncio.Event()
    rag = SimpleNamespace(finalize_storages=AsyncMock())
    calls = 0

    async def factory(_generation: object) -> object:
        nonlocal calls
        calls += 1
        await release_factory.wait()
        return rag

    pool = GenerationPool(registry, factory)
    first = asyncio.create_task(pool.acquire("oceanstack_product"))
    second = asyncio.create_task(pool.acquire("oceanstack_product"))
    await asyncio.sleep(0)
    release_factory.set()

    first_lease, second_lease = await asyncio.gather(first, second)

    assert calls == 1
    assert first_lease.rag is rag
    assert second_lease.rag is rag
    await first_lease.close()
    await second_lease.close()
    rag.finalize_storages.assert_not_awaited()
    await pool.close()
    rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_pool_keeps_superseded_generation_alive_until_last_lease_closes() -> None:
    old_generation = _generation(
        "oceanstack_maritime", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"
    )
    new_generation = _generation(
        "oceanstack_maritime", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa02"
    )
    registry = SimpleNamespace(
        resolve_active=AsyncMock(side_effect=[old_generation, new_generation])
    )
    old_rag = SimpleNamespace(finalize_storages=AsyncMock())
    new_rag = SimpleNamespace(finalize_storages=AsyncMock())
    rags = iter((old_rag, new_rag))

    async def factory(_generation: object) -> object:
        return next(rags)

    cleanup = AsyncMock()
    pool = GenerationPool(registry, factory, cleanup_retired=cleanup)
    old_lease = await pool.acquire("oceanstack_maritime")
    new_lease = await pool.acquire("oceanstack_maritime")

    old_rag.finalize_storages.assert_not_awaited()
    cleanup.assert_not_awaited()
    await old_lease.close()
    old_rag.finalize_storages.assert_awaited_once()
    cleanup.assert_awaited_once_with(old_generation, old_rag)
    new_rag.finalize_storages.assert_not_awaited()

    await new_lease.close()
    new_rag.finalize_storages.assert_not_awaited()
    await pool.close()
    new_rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_retired_cleanup_releases_replacement_acquisition() -> None:
    old_generation = _generation(
        "oceanstack_maritime", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"
    )
    new_generation = _generation(
        "oceanstack_maritime", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa02"
    )
    registry = SimpleNamespace(
        resolve_active=AsyncMock(
            side_effect=[old_generation, new_generation, new_generation]
        )
    )
    old_rag = SimpleNamespace(finalize_storages=AsyncMock())
    new_rag = SimpleNamespace(finalize_storages=AsyncMock())
    rags = iter((old_rag, new_rag))

    async def factory(_generation: object) -> object:
        return next(rags)

    cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    pool = GenerationPool(registry, factory, cleanup_retired=cleanup)
    old_lease = await pool.acquire("oceanstack_maritime")
    await old_lease.close()

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await pool.acquire("oceanstack_maritime")

    replacement_key = (
        new_generation.plane,
        new_generation.generation_id,
    )
    assert pool._entries[replacement_key].references == 0
    replacement_lease = await pool.acquire("oceanstack_maritime")
    await replacement_lease.close()
    await pool.close()

    old_rag.finalize_storages.assert_awaited_once()
    new_rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_pool_releases_failed_singleflight_reservations() -> None:
    generation = _generation("oceanstack_dev", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    registry = SimpleNamespace(resolve_active=AsyncMock(return_value=generation))
    factory = AsyncMock(side_effect=RuntimeError("factory failed"))
    pool = GenerationPool(registry, factory)

    with pytest.raises(RuntimeError, match="factory failed"):
        await pool.acquire("oceanstack_dev")

    assert pool.entry_count == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_retire_shared_singleflight() -> None:
    generation = _generation(
        "oceanstack_product", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"
    )
    registry = SimpleNamespace(resolve_active=AsyncMock(return_value=generation))
    release_factory = asyncio.Event()
    rag = SimpleNamespace(finalize_storages=AsyncMock())
    calls = 0

    async def factory(_generation: object) -> object:
        nonlocal calls
        calls += 1
        await release_factory.wait()
        return rag

    pool = GenerationPool(registry, factory)
    cancelled = asyncio.create_task(pool.acquire("oceanstack_product"))
    survivor = asyncio.create_task(pool.acquire("oceanstack_product"))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release_factory.set()

    survivor_lease = await survivor
    await survivor_lease.close()
    later_lease = await pool.acquire("oceanstack_product")

    assert calls == 1
    assert later_lease.rag is rag
    await later_lease.close()
    await pool.close()


@pytest.mark.asyncio
async def test_lease_runs_database_work_under_exact_read_fence() -> None:
    generation = _generation("oceanstack_dev", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    registry = SimpleNamespace(resolve_active=AsyncMock(return_value=generation))
    rag = SimpleNamespace(finalize_storages=AsyncMock())
    pool = GenerationPool(registry, AsyncMock(return_value=rag))
    lease = await pool.acquire("oceanstack_dev")

    observed = None

    async def operation() -> str:
        nonlocal observed
        observed = current_generation_operation_fence()
        return "ok"

    assert await lease.run(operation) == "ok"
    assert observed.kind is GenerationFenceKind.READ
    assert observed.plane == generation.plane
    assert observed.generation_id == generation.generation_id
    assert observed.workspace == generation.workspace
    assert current_generation_operation_fence() is None
    await lease.close()
    await pool.close()


@pytest.mark.asyncio
async def test_cancelled_read_operation_resets_context_fence() -> None:
    generation = _generation("oceanstack_dev", "018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    registry = SimpleNamespace(resolve_active=AsyncMock(return_value=generation))
    rag = SimpleNamespace(finalize_storages=AsyncMock())
    pool = GenerationPool(registry, AsyncMock(return_value=rag))
    lease = await pool.acquire("oceanstack_dev")
    entered = asyncio.Event()

    async def operation() -> None:
        assert current_generation_operation_fence() is not None
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(lease.run(operation))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert current_generation_operation_fence() is None
    await lease.close()
    await pool.close()
