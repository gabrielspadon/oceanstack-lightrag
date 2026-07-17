from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch
from uuid import UUID

import pytest

from lightrag.api.rag_factory import (
    GenerationRAGFactory,
    PostgresGenerationRuntime,
    create_generation_rag_builder,
    create_postgres_generation_runtime,
)
from lightrag.generation import (
    GenerationFenceKind,
    GenerationOperationFence,
    GenerationState,
    bind_generation_operation_fence,
    current_generation_operation_fence,
    reset_generation_operation_fence,
)


@pytest.mark.asyncio
async def test_factory_builds_and_initializes_exact_generation_workspace() -> None:
    observed_fences = []

    async def initialize() -> None:
        observed_fences.append(current_generation_operation_fence())

    rag = SimpleNamespace(initialize_storages=AsyncMock(side_effect=initialize))
    builder = Mock(return_value=rag)
    factory = GenerationRAGFactory(builder)
    generation = SimpleNamespace(
        plane="oceanstack_product",
        generation_id=UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"),
        workspace="kg_oceanstack_product_018f0f7dc68b7a2f8f7d724a24f9aa01",
    )

    result = await factory(generation)

    assert result is rag
    builder.assert_called_once_with(generation.workspace)
    rag.initialize_storages.assert_awaited_once_with()
    assert observed_fences[0].kind is GenerationFenceKind.READ
    assert observed_fences[0].workspace == generation.workspace
    assert current_generation_operation_fence() is None


@pytest.mark.asyncio
async def test_factory_finalizes_partially_initialized_rag_on_failure() -> None:
    rag = SimpleNamespace(
        initialize_storages=AsyncMock(side_effect=RuntimeError("init failed")),
        finalize_storages=AsyncMock(),
    )
    factory = GenerationRAGFactory(Mock(return_value=rag))
    generation = SimpleNamespace(
        plane="oceanstack_product",
        generation_id=UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"),
        workspace="kg_oceanstack_product_018f0f7dc68b7a2f8f7d724a24f9aa01",
    )

    with pytest.raises(RuntimeError, match="init failed"):
        await factory(generation)

    rag.finalize_storages.assert_awaited_once_with()


def test_runtime_uses_one_builder_for_read_and_build_factories(monkeypatch) -> None:
    builder = Mock()
    db = SimpleNamespace()
    registry = SimpleNamespace()
    monkeypatch.setattr("lightrag.api.rag_factory.PostgreSQLDB", Mock(return_value=db))
    monkeypatch.setattr(
        "lightrag.api.rag_factory.PostgresGenerationRegistry",
        Mock(return_value=registry),
    )
    monkeypatch.setattr(
        "lightrag.api.rag_factory.ClientManager.get_config",
        Mock(return_value={"enable_vector": True}),
    )

    runtime = PostgresGenerationRuntime.create(
        builder, vector_storage="PGVectorStorage"
    )

    assert runtime.registry is registry
    assert runtime.read_factory.builder is builder
    assert runtime.build_factory.builder is builder
    assert runtime.cleanup_factory.builder is builder
    assert runtime.pool._registry is registry


def test_public_builder_binds_exact_workspace_and_shared_configuration() -> None:
    rag = SimpleNamespace(register_role_llm_builder=Mock())
    rag_class = Mock(return_value=rag)
    role_builder = Mock()
    on_created = Mock()
    builder = create_generation_rag_builder(
        rag_class,
        constructor_kwargs={
            "working_dir": "/srv/lightrag",
            "graph_storage": "PGGraphStorage",
            "vector_storage": "PGVectorStorage",
        },
        role_llm_builder=role_builder,
        on_created=on_created,
    )

    result = builder("kg_oceanstack_dev_018f0f7dc68b7a2f8f7d724a24f9aa01")

    assert result is rag
    rag_class.assert_called_once_with(
        working_dir="/srv/lightrag",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        workspace="kg_oceanstack_dev_018f0f7dc68b7a2f8f7d724a24f9aa01",
    )
    rag.register_role_llm_builder.assert_called_once_with(role_builder)
    on_created.assert_called_once_with(rag)


def test_public_builder_rejects_hidden_default_workspace() -> None:
    with pytest.raises(ValueError, match="workspace"):
        create_generation_rag_builder(
            Mock(), constructor_kwargs={"workspace": "default"}
        )


def test_public_postgres_runtime_factory_uses_deployment_builder_by_default(
    monkeypatch,
) -> None:
    args = SimpleNamespace(
        kv_storage="PGKVStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
    )
    builder = Mock()
    runtime = Mock()
    build_deployment = Mock(return_value=builder)
    create_runtime = Mock(return_value=runtime)
    get_deployment_config = Mock(return_value=args)
    monkeypatch.setattr(
        "lightrag.api.config.get_deployment_config", get_deployment_config
    )
    monkeypatch.setattr(
        "lightrag.api.rag_factory._create_deployment_generation_rag_builder",
        build_deployment,
    )
    monkeypatch.setattr(PostgresGenerationRuntime, "create", create_runtime)

    assert create_postgres_generation_runtime() is runtime

    get_deployment_config.assert_called_once_with()
    build_deployment.assert_called_once_with(args)
    create_runtime.assert_called_once_with(builder, vector_storage="PGVectorStorage")


def test_public_postgres_runtime_factory_rejects_non_postgres_storage() -> None:
    args = SimpleNamespace(
        kv_storage="JsonKVStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
    )

    with pytest.raises(ValueError, match="exact PostgreSQL"):
        create_postgres_generation_runtime(args, builder=Mock())


def test_public_postgres_runtime_factory_rejects_forced_workspace(
    monkeypatch,
) -> None:
    args = SimpleNamespace(
        kv_storage="PGKVStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
    )
    monkeypatch.setenv("POSTGRES_WORKSPACE", "forced_workspace")

    with pytest.raises(ValueError, match="POSTGRES_WORKSPACE"):
        create_postgres_generation_runtime(args, builder=Mock())


def test_explicit_environment_config_ignores_embedding_process_argv(
    monkeypatch,
) -> None:
    from lightrag.api.config import parse_args

    monkeypatch.setattr(sys, "argv", ["oceanstack", "--foreign-option"])
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")

    args = parse_args([])

    assert args.llm_binding == "openai"
    assert args.embedding_binding == "openai"


@pytest.mark.asyncio
async def test_cleanup_factory_initializes_exact_workspace_without_build_access() -> (
    None
):
    generation = SimpleNamespace(
        plane="oceanstack_product",
        generation_id=UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"),
        workspace="kg_oceanstack_product_018f0f7dc68b7a2f8f7d724a24f9aa01",
    )
    observed = []

    async def initialize() -> None:
        observed.append(current_generation_operation_fence())

    rag = SimpleNamespace(
        initialize_storages=AsyncMock(side_effect=initialize),
        finalize_storages=AsyncMock(),
    )
    factory = GenerationRAGFactory(
        Mock(return_value=rag), fence_kind=GenerationFenceKind.CLEANUP
    )
    fence = GenerationOperationFence(
        kind=GenerationFenceKind.CLEANUP,
        plane=generation.plane,
        generation_id=generation.generation_id,
        workspace=generation.workspace,
        token=UUID("118f0f7d-c68b-7a2f-8f7d-724a24f9aa01"),
    )
    token = bind_generation_operation_fence(fence)
    try:
        assert await factory(generation) is rag
    finally:
        reset_generation_operation_fence(token)

    assert observed == [fence]


@pytest.mark.asyncio
async def test_cleanup_uninitialized_drops_before_registry_delete_and_finalizes() -> (
    None
):
    events: list[str] = []
    report = SimpleNamespace(success=True, workspace="workspace")
    rag = SimpleNamespace(
        adrop_workspace_storages=AsyncMock(
            side_effect=lambda _workspace: events.append("drop") or report
        ),
        finalize_storages=AsyncMock(side_effect=lambda: events.append("finalize")),
    )

    class Claim:
        workspace = "workspace"
        record_failure = AsyncMock()
        delete = AsyncMock(side_effect=lambda: events.append("delete") or True)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    claim = Claim()
    registry = SimpleNamespace(acquire_inactive_cleanup=Mock(return_value=claim))
    runtime = PostgresGenerationRuntime(
        registry=registry,
        read_factory=Mock(),
        build_factory=Mock(),
        cleanup_factory=AsyncMock(return_value=rag),
        pool=Mock(),
    )
    generation = SimpleNamespace(
        plane="oceanstack_dev",
        generation_id=UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"),
    )

    assert await runtime.cleanup_uninitialized(generation) is report
    assert events == ["drop", "delete", "finalize"]


@pytest.mark.asyncio
async def test_runtime_validates_persisted_candidate_through_rag_storage() -> None:
    evidence = SimpleNamespace(
        counts={"assertions": 4, "chunks": 5, "entities": 3, "sources": 2},
        contract_digest="a" * 64,
        manifest_digest="b" * 64,
    )
    rag = SimpleNamespace(
        avalidate_persisted_knowledge_graph=AsyncMock(return_value=evidence)
    )
    runtime = PostgresGenerationRuntime(
        registry=Mock(),
        read_factory=Mock(),
        build_factory=Mock(),
        cleanup_factory=Mock(),
        pool=Mock(),
    )

    result = await runtime.validate_persisted_candidate(
        rag,
        expected_counts={
            "assertions": 4,
            "chunks": 5,
            "entities": 3,
            "sources": 2,
        },
        expected_contract_digest="a" * 64,
        expected_manifest_digest="b" * 64,
    )

    assert result is evidence
    rag.avalidate_persisted_knowledge_graph.assert_awaited_once_with(
        expected_counts={
            "assertions": 4,
            "chunks": 5,
            "entities": 3,
            "sources": 2,
        },
        expected_contract_digest="a" * 64,
        expected_manifest_digest="b" * 64,
    )


@pytest.mark.asyncio
async def test_bootstrap_cleans_every_stale_generation_before_returning() -> None:
    active = SimpleNamespace(
        plane="oceanstack_dev",
        generation_id=UUID(int=1),
        state=GenerationState.READY,
    )
    stale = [
        SimpleNamespace(
            plane="oceanstack_dev",
            generation_id=UUID(int=2),
            state=GenerationState.FAILED,
        )
    ]
    inactive_ready = SimpleNamespace(
        plane="oceanstack_product",
        generation_id=UUID(int=3),
        state=GenerationState.READY,
    )
    building = SimpleNamespace(
        plane="oceanstack_maritime",
        generation_id=UUID(int=4),
        state=GenerationState.BUILDING,
    )
    cleanup_candidates = [stale[0], inactive_ready]
    registry = SimpleNamespace(
        bootstrap=AsyncMock(),
        fail_stale=AsyncMock(return_value=stale),
        resolve_active=AsyncMock(
            side_effect=lambda plane: active if plane == "oceanstack_dev" else None
        ),
        list_generations=AsyncMock(
            side_effect=lambda plane: {
                "oceanstack_dev": [active, stale[0]],
                "oceanstack_product": [inactive_ready],
                "oceanstack_maritime": [building],
            }[plane]
        ),
    )
    runtime = PostgresGenerationRuntime(
        registry=registry,
        read_factory=Mock(),
        build_factory=Mock(),
        cleanup_factory=Mock(),
        pool=Mock(),
    )
    cleanup = AsyncMock()
    with patch.object(PostgresGenerationRuntime, "cleanup_uninitialized", cleanup):
        assert await runtime.bootstrap() == cleanup_candidates

    registry.bootstrap.assert_awaited_once_with()
    registry.fail_stale.assert_awaited_once_with()
    assert registry.resolve_active.await_count == 3
    assert registry.list_generations.await_count == 3
    assert cleanup.await_args_list == [
        call(cleanup_candidates[0]),
        call(cleanup_candidates[1]),
    ]


@pytest.mark.asyncio
async def test_bootstrap_fails_closed_when_stale_cleanup_fails() -> None:
    stale = [
        SimpleNamespace(
            plane="oceanstack_dev",
            generation_id=UUID(int=1),
            state=GenerationState.FAILED,
        )
    ]
    registry = SimpleNamespace(
        bootstrap=AsyncMock(),
        fail_stale=AsyncMock(return_value=stale),
        resolve_active=AsyncMock(return_value=None),
        list_generations=AsyncMock(
            side_effect=lambda plane: stale if plane == "oceanstack_dev" else []
        ),
    )
    runtime = PostgresGenerationRuntime(
        registry=registry,
        read_factory=Mock(),
        build_factory=Mock(),
        cleanup_factory=Mock(),
        pool=Mock(),
    )
    cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    with (
        patch.object(PostgresGenerationRuntime, "cleanup_uninitialized", cleanup),
        pytest.raises(RuntimeError, match="cleanup failed"),
    ):
        await runtime.bootstrap()
