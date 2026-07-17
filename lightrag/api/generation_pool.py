"""Request-scoped leases for immutable graph-generation RAG instances."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Final, Protocol
from uuid import UUID

from lightrag.generation import (
    GenerationFenceKind,
    GenerationOperationFence,
    GraphGeneration,
    bind_generation_operation_fence,
    complete_cleanup_preserving_cancellation,
    reset_generation_operation_fence,
)


PUBLIC_PLANES: Final[frozenset[str]] = frozenset(
    {"oceanstack_dev", "oceanstack_product", "oceanstack_maritime"}
)


class UnsupportedPlaneError(ValueError):
    """A request named a plane outside the fixed public API contract."""


class GenerationUnavailableError(LookupError):
    """A public plane has no active immutable generation."""


class GenerationPoolClosedError(RuntimeError):
    """The server is shutting down and cannot grant another read lease."""


class _GraphStorage(Protocol):
    async def get_popular_labels(self, limit: int) -> Any: ...

    async def search_labels(self, query: str, limit: int) -> Any: ...

    async def has_node(self, name: str) -> bool: ...

    async def get_graph_entity(self, entity_id: str) -> dict[str, Any] | None: ...

    async def get_graph_assertions(
        self, assertion_ids: list[str]
    ) -> dict[str, dict[str, Any]]: ...


class GenerationRAG(Protocol):
    chunk_entity_relation_graph: _GraphStorage

    async def finalize_storages(self) -> None: ...

    async def adrop_workspace_storages(self, workspace: str) -> Any: ...

    async def aquery_llm(self, query: str, *, param: Any) -> dict[str, Any]: ...

    async def aquery_data(self, query: str, *, param: Any) -> dict[str, Any]: ...

    async def get_graph_labels(self) -> list[str]: ...

    async def get_knowledge_graph(
        self, *, node_label: str, max_depth: int, max_nodes: int
    ) -> Any: ...


class ActiveGenerationRegistry(Protocol):
    async def resolve_active(self, plane: str) -> GraphGeneration | None: ...


RAGFactory = Callable[[GraphGeneration], Coroutine[Any, Any, GenerationRAG]]
RetiredCleanup = Callable[[GraphGeneration, GenerationRAG], Awaitable[None]]
GenerationKey = tuple[str, UUID]


@dataclass(slots=True)
class _PoolEntry:
    generation: GraphGeneration
    task: asyncio.Task[GenerationRAG]
    references: int = 0
    retired: bool = False


class GenerationReadLease:
    """Idempotent read lease pinning one exact generation until closed."""

    def __init__(
        self,
        pool: GenerationPool,
        key: GenerationKey,
        generation: GraphGeneration,
        rag: GenerationRAG,
    ) -> None:
        self._pool = pool
        self._key = key
        self.generation = generation
        self.rag = rag
        self._closed = False
        self._fence = GenerationOperationFence(
            kind=GenerationFenceKind.READ,
            plane=generation.plane,
            generation_id=generation.generation_id,
            workspace=generation.workspace,
            token=generation.generation_id,
        )

    async def __aenter__(self) -> GenerationReadLease:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._pool._release(self._key)

    async def run(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Run one database-owning operation under this generation's read fence."""
        token = bind_generation_operation_fence(self._fence)
        try:
            return await operation()
        finally:
            reset_generation_operation_fence(token)


class GenerationPool:
    """Refcounted, singleflight pool keyed by ``(plane, generation_id)``."""

    def __init__(
        self,
        registry: ActiveGenerationRegistry,
        factory: RAGFactory,
        *,
        public_planes: frozenset[str] = PUBLIC_PLANES,
        cleanup_retired: RetiredCleanup | None = None,
    ) -> None:
        self._registry = registry
        self._factory = factory
        self._public_planes = public_planes
        self._cleanup_retired = cleanup_retired
        self._entries: dict[GenerationKey, _PoolEntry] = {}
        self._active: dict[str, GenerationKey] = {}
        self._lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    async def acquire(self, plane: str) -> GenerationReadLease:
        if plane not in self._public_planes:
            raise UnsupportedPlaneError(f"unsupported plane: {plane}")

        cleanup: asyncio.Task[None] | None = None
        async with self._lock:
            if self._closed:
                raise GenerationPoolClosedError("generation pool is closed")
            generation = await self._registry.resolve_active(plane)
            if generation is None:
                raise GenerationUnavailableError(
                    f"no active generation for plane: {plane}"
                )
            key = (plane, generation.generation_id)

            prior_key = self._active.get(plane)
            if prior_key is not None and prior_key != key:
                prior = self._entries.get(prior_key)
                if prior is not None:
                    prior.retired = True
                    cleanup = self._schedule_cleanup_if_unused(prior_key, prior)

            entry = self._entries.get(key)
            if entry is None:
                entry = _PoolEntry(
                    generation=generation,
                    task=asyncio.create_task(self._factory(generation)),
                )
                self._entries[key] = entry
            entry.references += 1
            self._active[plane] = key

        if cleanup is not None:
            try:
                await cleanup
            except asyncio.CancelledError as exc:
                await complete_cleanup_preserving_cancellation(
                    self._release(key), initial_cancellation=exc
                )
                raise exc
            except BaseException:
                await self._release(key)
                raise

        try:
            rag = await asyncio.shield(entry.task)
        except asyncio.CancelledError as exc:
            await complete_cleanup_preserving_cancellation(
                self._release(key, discard=entry.task.cancelled()),
                initial_cancellation=exc,
            )
            raise exc
        except BaseException:
            await self._release(key, discard=True)
            raise

        return GenerationReadLease(self, key, generation, rag)

    async def _release(self, key: GenerationKey, *, discard: bool = False) -> None:
        cleanup: asyncio.Task[None] | None = None
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if discard:
                entry.retired = True
            if entry.references <= 0:
                raise RuntimeError("generation pool reference count underflow")
            entry.references -= 1
            cleanup = self._schedule_cleanup_if_unused(key, entry)

        if cleanup is not None:
            await cleanup

    def _schedule_cleanup_if_unused(
        self, key: GenerationKey, entry: _PoolEntry
    ) -> asyncio.Task[None] | None:
        if entry.references != 0 or not (entry.retired or self._closed):
            return None
        self._entries.pop(key, None)
        if self._active.get(key[0]) == key:
            self._active.pop(key[0], None)
        task = asyncio.create_task(self._finalize_entry(entry))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        return task

    async def _finalize_entry(self, entry: _PoolEntry) -> None:
        try:
            rag = await asyncio.shield(entry.task)
        except BaseException:
            return
        try:
            if self._cleanup_retired is not None and entry.retired and not self._closed:
                await self._cleanup_retired(entry.generation, rag)
        finally:
            await rag.finalize_storages()

    async def close(self) -> None:
        cleanups: list[asyncio.Task[None]] = []
        async with self._lock:
            if self._closed:
                cleanups.extend(self._cleanup_tasks)
            else:
                self._closed = True
                self._active.clear()
                for key, entry in tuple(self._entries.items()):
                    entry.retired = True
                    cleanup = self._schedule_cleanup_if_unused(key, entry)
                    if cleanup is not None:
                        cleanups.append(cleanup)
                cleanups.extend(self._cleanup_tasks)

        if cleanups:
            await asyncio.gather(*set(cleanups))
