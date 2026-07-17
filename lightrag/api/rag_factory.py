"""Storage-neutral construction boundary for generation-scoped RAG instances."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast
from uuid import UUID

from lightrag.api.generation_pool import (
    PUBLIC_PLANES,
    GenerationPool,
    GenerationRAG as PooledGenerationRAG,
)
from lightrag.generation import (
    InactiveGenerationCleanup,
    GenerationFenceKind,
    GenerationOperationFence,
    GenerationState,
    GraphGeneration,
    PersistedGenerationEvidence,
    WorkspaceDropCallback,
    WorkspaceDropReport,
    bind_generation_operation_fence,
    current_generation_operation_fence,
    generation_workspace,
    reset_generation_operation_fence,
)
from lightrag.kg.postgres_impl import (
    ClientManager,
    PostgresGenerationRegistry,
    PostgreSQLDB,
)


class GenerationRAG(PooledGenerationRAG, Protocol):
    chunk_entity_relation_graph: Any

    async def initialize_storages(self) -> None: ...

    async def finalize_storages(self) -> None: ...

    async def adrop_workspace_storages(self, workspace: str) -> WorkspaceDropReport: ...

    async def avalidate_persisted_knowledge_graph(
        self,
        *,
        expected_counts: Mapping[str, int],
        expected_contract_digest: str,
        expected_manifest_digest: str,
    ) -> PersistedGenerationEvidence: ...


class GenerationRuntimeConfig(Protocol):
    kv_storage: str
    graph_storage: str
    vector_storage: str
    doc_status_storage: str


_RAG = TypeVar("_RAG", bound=GenerationRAG)


@dataclass(frozen=True, slots=True)
class ConfiguredGenerationRAGBuilder(Generic[_RAG]):
    """Exact env/config-bound constructor shared by build and query runtimes."""

    rag_class: Callable[..., _RAG]
    constructor_kwargs: Mapping[str, Any]
    role_llm_builder: Callable[..., Any] | None = None
    on_created: Callable[[_RAG], None] | None = None

    def __call__(self, workspace: str) -> _RAG:
        rag = self.rag_class(**self.constructor_kwargs, workspace=workspace)
        if self.role_llm_builder is not None:
            register = getattr(rag, "register_role_llm_builder")
            register(self.role_llm_builder)
        if self.on_created is not None:
            self.on_created(rag)
        return rag


def create_generation_rag_builder(
    rag_class: Callable[..., _RAG],
    *,
    constructor_kwargs: Mapping[str, Any],
    role_llm_builder: Callable[..., Any] | None = None,
    on_created: Callable[[_RAG], None] | None = None,
) -> ConfiguredGenerationRAGBuilder[_RAG]:
    """Create one reusable builder with no implicit or caller-selected workspace."""
    if "workspace" in constructor_kwargs:
        raise ValueError("constructor_kwargs must not contain a workspace")
    return ConfiguredGenerationRAGBuilder(
        rag_class=rag_class,
        constructor_kwargs=dict(constructor_kwargs),
        role_llm_builder=role_llm_builder,
        on_created=on_created,
    )


class GenerationRAGFactory(Generic[_RAG]):
    """Build and initialize a RAG bound to one immutable workspace."""

    def __init__(
        self,
        builder: Callable[[str], _RAG],
        *,
        fence_kind: GenerationFenceKind = GenerationFenceKind.READ,
    ) -> None:
        self._builder = builder
        if fence_kind not in (
            GenerationFenceKind.READ,
            GenerationFenceKind.BUILD,
            GenerationFenceKind.CLEANUP,
        ):
            raise ValueError("RAG factory fence kind is invalid")
        self._fence_kind = fence_kind

    @property
    def builder(self) -> Callable[[str], _RAG]:
        return self._builder

    async def __call__(self, generation: GraphGeneration) -> _RAG:
        expected_workspace = generation_workspace(
            generation.plane, generation.generation_id
        )
        if generation.workspace != expected_workspace:
            raise ValueError("generation workspace does not match its identity")

        rag = self._builder(generation.workspace)
        token = None
        if self._fence_kind is GenerationFenceKind.READ:
            fence = GenerationOperationFence(
                kind=GenerationFenceKind.READ,
                plane=generation.plane,
                generation_id=generation.generation_id,
                workspace=generation.workspace,
                token=generation.generation_id,
            )
            token = bind_generation_operation_fence(fence)
        else:
            fence = current_generation_operation_fence()
            if (
                fence is None
                or fence.kind is not self._fence_kind
                or fence.plane != generation.plane
                or fence.generation_id != generation.generation_id
                or fence.workspace != generation.workspace
            ):
                raise RuntimeError(
                    f"{self._fence_kind.value} RAG initialization requires its active fence"
                )
        try:
            await rag.initialize_storages()
        except BaseException:
            await rag.finalize_storages()
            raise
        finally:
            if token is not None:
                reset_generation_operation_fence(token)
        return rag


@dataclass(slots=True)
class PostgresGenerationRuntime(Generic[_RAG]):
    """Shared PostgreSQL generation registry and exact RAG construction surface."""

    registry: PostgresGenerationRegistry
    read_factory: GenerationRAGFactory[_RAG]
    build_factory: GenerationRAGFactory[_RAG]
    cleanup_factory: GenerationRAGFactory[_RAG]
    pool: GenerationPool

    @classmethod
    def create(
        cls,
        builder: Callable[[str], _RAG],
        *,
        vector_storage: str | None,
    ) -> PostgresGenerationRuntime[_RAG]:
        postgres_config = ClientManager.get_config(vector_storage=vector_storage)
        if postgres_config.get("workspace"):
            raise ValueError(
                "POSTGRES_WORKSPACE and postgres.workspace are forbidden; "
                "generation identity owns the workspace"
            )
        db = PostgreSQLDB(postgres_config)
        registry = PostgresGenerationRegistry(db)
        read_factory = GenerationRAGFactory(builder)
        build_factory = GenerationRAGFactory(
            builder, fence_kind=GenerationFenceKind.BUILD
        )
        cleanup_factory = GenerationRAGFactory(
            builder, fence_kind=GenerationFenceKind.CLEANUP
        )

        async def cleanup_retired(
            generation: GraphGeneration, rag: PooledGenerationRAG
        ) -> None:
            claim = registry.acquire_inactive_cleanup(
                generation.plane, generation.generation_id
            )
            async with claim:
                report = cast(
                    WorkspaceDropReport,
                    await rag.adrop_workspace_storages(claim.workspace),
                )
                if not report.success:
                    await claim.record_failure(
                        {
                            "code": "retired_generation_storage_cleanup_failed",
                            "workspace": claim.workspace,
                        }
                    )
                    raise RuntimeError("retired generation storage cleanup failed")
                if not await claim.delete():
                    raise RuntimeError("retired generation registry cleanup was fenced")

        pool = GenerationPool(registry, read_factory, cleanup_retired=cleanup_retired)
        return cls(
            registry=registry,
            read_factory=read_factory,
            build_factory=build_factory,
            cleanup_factory=cleanup_factory,
            pool=pool,
        )

    async def bootstrap(self) -> list[GraphGeneration]:
        await self.registry.bootstrap()
        await self.registry.fail_stale()
        cleanup_candidates: list[GraphGeneration] = []
        for plane in sorted(PUBLIC_PLANES):
            active = await self.registry.resolve_active(plane)
            active_generation_id = active.generation_id if active is not None else None
            for generation in await self.registry.list_generations(plane):
                if generation.state not in (
                    GenerationState.READY,
                    GenerationState.FAILED,
                ):
                    continue
                if generation.generation_id == active_generation_id:
                    continue
                cleanup_candidates.append(generation)
        for generation in cleanup_candidates:
            await self.cleanup_uninitialized(generation)
        return cleanup_candidates

    async def close(self) -> None:
        await self.pool.close()
        await self.registry.close()

    async def validate_persisted_candidate(
        self,
        rag: _RAG,
        *,
        expected_counts: Mapping[str, int],
        expected_contract_digest: str,
        expected_manifest_digest: str,
    ) -> PersistedGenerationEvidence:
        """Return measured storage evidence only when it matches the candidate."""
        return await rag.avalidate_persisted_knowledge_graph(
            expected_counts=expected_counts,
            expected_contract_digest=expected_contract_digest,
            expected_manifest_digest=expected_manifest_digest,
        )

    async def cleanup_inactive(
        self,
        plane: str,
        generation_id: UUID,
        drop_workspace: WorkspaceDropCallback,
    ) -> WorkspaceDropReport:
        """Drop all inactive-generation storage, then delete its registry row."""
        cleanup = InactiveGenerationCleanup(self.registry, drop_workspace)
        return await cleanup.cleanup(plane, generation_id)

    async def cleanup_uninitialized(
        self, generation: GraphGeneration
    ) -> WorkspaceDropReport:
        """Construct a cleanup-only RAG, drop partial storage, then delete registry."""
        claim = self.registry.acquire_inactive_cleanup(
            generation.plane, generation.generation_id
        )
        async with claim:
            rag = await self.cleanup_factory(generation)
            try:
                report = await rag.adrop_workspace_storages(claim.workspace)
                if not report.success:
                    await claim.record_failure(
                        {
                            "code": "inactive_generation_storage_cleanup_failed",
                            "workspace": claim.workspace,
                        }
                    )
                    raise RuntimeError("inactive generation storage cleanup failed")
                if not await claim.delete():
                    raise RuntimeError(
                        "inactive generation registry cleanup was fenced"
                    )
                return report
            finally:
                await rag.finalize_storages()
        raise RuntimeError("inactive generation cleanup ended without evidence")


def create_postgres_generation_runtime(
    args: GenerationRuntimeConfig | None = None,
    *,
    builder: Callable[[str], GenerationRAG] | None = None,
) -> PostgresGenerationRuntime[GenerationRAG]:
    """Create the deployment-configured PostgreSQL generation runtime.

    With no arguments this resolves the same deployment arguments and exact
    RAG builder used by the API server. Supplying ``builder`` lets the server
    pass its already composed builder without repeating runtime construction.
    """
    if args is None:
        from lightrag.api.config import get_deployment_config

        resolved_args = cast(GenerationRuntimeConfig, get_deployment_config())
    else:
        resolved_args = args
    resolved_builder = builder
    if resolved_builder is None:
        resolved_builder = _create_deployment_generation_rag_builder(resolved_args)

    configured_storages = {
        resolved_args.kv_storage,
        resolved_args.graph_storage,
        resolved_args.vector_storage,
        resolved_args.doc_status_storage,
    }
    required_storages = {
        "PGKVStorage",
        "PGGraphStorage",
        "PGVectorStorage",
        "PGDocStatusStorage",
    }
    if configured_storages != required_storages:
        raise ValueError(
            "generation runtime requires exact PostgreSQL storage backends"
        )
    return PostgresGenerationRuntime.create(
        resolved_builder, vector_storage=resolved_args.vector_storage
    )


def _create_deployment_generation_rag_builder(
    args: Any,
) -> Callable[[str], GenerationRAG]:
    from lightrag.api.lightrag_server import (
        create_deployment_generation_rag_builder,
    )

    return create_deployment_generation_rag_builder(args)
