from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from lightrag.generation import (
    InactiveGenerationCleanup,
    GenerationCleanupError,
    GenerationFenceKind,
    GenerationOperationFence,
    StorageDropResult,
    WorkspaceDropReport,
    bind_generation_operation_fence,
    reset_generation_operation_fence,
)
from lightrag.lightrag import LightRAG


class _Storage:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        workspace: str = "kg_oceanstack_dev_018f0f7dc68b7a2f8f7d724a24f9aa01",
        status: str = "success",
    ) -> None:
        self.namespace = name
        self.workspace = workspace
        self._events = events
        self._status = status

    async def drop_pending_index_ops(self) -> None:
        self._events.append(f"pending:{self.namespace}")

    async def drop(self) -> dict[str, str]:
        self._events.append(f"drop:{self.namespace}")
        return {"status": self._status, "message": f"{self.namespace} {self._status}"}


_Storage.__module__ = "lightrag.kg.postgres_impl"


def _bind_cleanup_fence():
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    return bind_generation_operation_fence(
        GenerationOperationFence(
            kind=GenerationFenceKind.CLEANUP,
            plane="oceanstack_dev",
            generation_id=generation_id,
            workspace="kg_oceanstack_dev_018f0f7dc68b7a2f8f7d724a24f9aa01",
            token=uuid.uuid4(),
        )
    )


class _CleanupClaim:
    def __init__(
        self,
        enter_error: Exception | None = None,
        workspace: str = "kg_oceanstack_dev_018f0f7dc68b7a2f8f7d724a24f9aa01",
    ) -> None:
        self.enter_error = enter_error
        self.workspace = workspace
        self.record_failure = AsyncMock()
        self.delete = AsyncMock(return_value=True)

    async def __aenter__(self) -> "_CleanupClaim":
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _registry(claim: _CleanupClaim) -> Mock:
    registry = Mock()
    registry.acquire_inactive_cleanup.return_value = claim
    return registry


def _bare_rag(storages: list[_Storage], *, duplicate_first: bool = False) -> LightRAG:
    rag = object.__new__(LightRAG)
    rag.workspace = storages[0].workspace
    values = storages.copy()
    if duplicate_first:
        values[1] = values[0]
    (
        rag.full_docs,
        rag.doc_status,
        rag.text_chunks,
        rag.full_entities,
        rag.full_relations,
        rag.entity_chunks,
        rag.relation_chunks,
        rag.llm_response_cache,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunks_vdb,
        rag.chunk_entity_relation_graph,
    ) = values
    return rag


@pytest.mark.asyncio
async def test_drop_workspace_covers_all_storages_deduplicates_and_discards_first() -> (
    None
):
    events: list[str] = []
    storages = [_Storage(f"s{index}", events) for index in range(12)]
    rag = _bare_rag(storages, duplicate_first=True)

    token = _bind_cleanup_fence()
    try:
        report = await rag.adrop_workspace_storages(rag.workspace)
    finally:
        reset_generation_operation_fence(token)

    assert report.success
    assert len(report.results) == 11
    assert [result.name for result in report.results] == [
        "full_docs",
        "text_chunks",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
    ]
    first_drop = next(
        index for index, event in enumerate(events) if event.startswith("drop:")
    )
    assert all(event.startswith("pending:") for event in events[:first_drop])
    assert events.count("pending:s0") == 1
    assert events.count("drop:s0") == 1


@pytest.mark.asyncio
async def test_drop_workspace_refuses_any_mismatched_storage_before_mutation() -> None:
    events: list[str] = []
    storages = [_Storage(f"s{index}", events) for index in range(12)]
    storages[-1].workspace = "kg_other_018f0f7dc68b7a2f8f7d724a24f9aa01"
    rag = _bare_rag(storages)

    token = _bind_cleanup_fence()
    try:
        with pytest.raises(ValueError, match="workspace mismatch"):
            await rag.adrop_workspace_storages(rag.workspace)
    finally:
        reset_generation_operation_fence(token)

    assert events == []


@pytest.mark.asyncio
async def test_drop_workspace_inspects_every_status_and_reports_partial_failure() -> (
    None
):
    events: list[str] = []
    storages = [_Storage(f"s{index}", events) for index in range(12)]
    storages[4]._status = "error"
    rag = _bare_rag(storages)

    token = _bind_cleanup_fence()
    try:
        report = await rag.adrop_workspace_storages(rag.workspace)
    finally:
        reset_generation_operation_fence(token)

    assert not report.success
    assert len(report.results) == 12
    assert report.results[4].status == "error"
    assert len([event for event in events if event.startswith("drop:")]) == 12


@pytest.mark.asyncio
async def test_inactive_cleanup_deletes_registry_last_only_after_all_drops_succeed() -> (
    None
):
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    workspace = "kg_oceanstack_dev_018f0f7dc68b7a2f8f7d724a24f9aa01"
    claim = _CleanupClaim()
    registry = _registry(claim)
    attempts = 0

    async def drop(candidate_workspace: str) -> WorkspaceDropReport:
        nonlocal attempts
        assert candidate_workspace == workspace
        attempts += 1
        if attempts == 1:
            return WorkspaceDropReport(
                workspace,
                (
                    StorageDropResult("graph", "success", "dropped"),
                    StorageDropResult("vectors", "error", "backend unavailable"),
                ),
            )
        return WorkspaceDropReport(
            workspace,
            (
                StorageDropResult("graph", "success", "already absent"),
                StorageDropResult("vectors", "success", "dropped"),
            ),
        )

    cleanup = InactiveGenerationCleanup(registry, drop)

    with pytest.raises(GenerationCleanupError, match="vectors"):
        await cleanup.cleanup("oceanstack_dev", generation_id)
    claim.delete.assert_not_awaited()
    claim.record_failure.assert_awaited_once()

    await cleanup.cleanup("oceanstack_dev", generation_id)

    claim.delete.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_inactive_cleanup_records_callback_exception_and_keeps_registry_row() -> (
    None
):
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    claim = _CleanupClaim()
    registry = _registry(claim)

    async def drop(_workspace: str) -> WorkspaceDropReport:
        raise RuntimeError("storage coordinator unavailable")

    cleanup = InactiveGenerationCleanup(registry, drop)

    with pytest.raises(GenerationCleanupError, match="callback failed"):
        await cleanup.cleanup("oceanstack_dev", generation_id)

    claim.delete.assert_not_awaited()
    claim.record_failure.assert_awaited_once_with(
        {
            "code": "storage_cleanup_callback_failed",
            "error_type": "RuntimeError",
            "message": "storage coordinator unavailable",
        },
    )


@pytest.mark.asyncio
async def test_inactive_cleanup_rejects_mismatched_drop_report_workspace() -> None:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    claim = _CleanupClaim()
    registry = _registry(claim)

    async def drop(_workspace: str) -> WorkspaceDropReport:
        return WorkspaceDropReport(
            "kg_wrong_018f0f7dc68b7a2f8f7d724a24f9aa01",
            (StorageDropResult("graph", "success", "dropped"),),
        )

    cleanup = InactiveGenerationCleanup(registry, drop)

    with pytest.raises(GenerationCleanupError, match="report workspace mismatch"):
        await cleanup.cleanup("oceanstack_dev", generation_id)

    claim.delete.assert_not_awaited()
    claim.record_failure.assert_awaited_once()


@pytest.mark.asyncio
async def test_inactive_cleanup_requires_claim_before_storage_drop() -> None:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    claim = _CleanupClaim(RuntimeError("generation is ready or active"))
    registry = _registry(claim)
    drop = AsyncMock()
    cleanup = InactiveGenerationCleanup(registry, drop)

    with pytest.raises(RuntimeError, match="ready or active"):
        await cleanup.cleanup("oceanstack_dev", generation_id)

    registry.acquire_inactive_cleanup.assert_called_once_with(
        "oceanstack_dev", generation_id
    )
    drop.assert_not_awaited()
