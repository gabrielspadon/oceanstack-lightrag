from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import lightrag.kg.postgres_impl as postgres_impl
from lightrag.generation import (
    GenerationFenceError,
    GenerationFenceKind,
    GenerationOperationFence,
    GenerationStorageAccess,
    bind_generation_operation_fence,
    generation_workspace,
    reset_generation_operation_fence,
)
from lightrag.kg.postgres_impl import (
    ClientManager,
    PGDocStatusStorage,
    PGGraphStorage,
    PostgreSQLDB,
)
from lightrag.namespace import NameSpace


def _generation_storage(
    storage_type: type[PGDocStatusStorage] | type[PGGraphStorage],
    workspace: str,
) -> PGDocStatusStorage | PGGraphStorage:
    storage = storage_type.__new__(storage_type)
    storage.workspace = workspace
    storage.namespace = (
        NameSpace.DOC_STATUS
        if storage_type is PGDocStatusStorage
        else NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
    )
    storage.global_config = {"vector_storage": "PGVectorStorage"}
    storage.db = None
    if storage_type is PGGraphStorage:
        storage.__post_init__()
    return storage


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_type", [PGDocStatusStorage, PGGraphStorage])
async def test_generation_storage_requires_build_fence_before_client_acquire(
    storage_type: type[PGDocStatusStorage] | type[PGGraphStorage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id = uuid.uuid4()
    workspace = generation_workspace("oceanstack_dev", generation_id)
    storage = _generation_storage(storage_type, workspace)
    get_client = AsyncMock()
    monkeypatch.setattr(ClientManager, "get_client", get_client)

    @asynccontextmanager
    async def init_lock():
        yield

    monkeypatch.setattr(postgres_impl, "get_data_init_lock", init_lock)

    with pytest.raises(GenerationFenceError, match="active fence"):
        await storage.initialize()

    get_client.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_type", [PGDocStatusStorage, PGGraphStorage])
async def test_generation_storage_rejects_postgres_workspace_override(
    storage_type: type[PGDocStatusStorage] | type[PGGraphStorage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id = uuid.uuid4()
    workspace = generation_workspace("oceanstack_dev", generation_id)
    storage = _generation_storage(storage_type, workspace)
    storage.db = SimpleNamespace(workspace="wrong_workspace")

    @asynccontextmanager
    async def init_lock():
        yield

    monkeypatch.setattr(postgres_impl, "get_data_init_lock", init_lock)
    fence = GenerationOperationFence(
        kind=GenerationFenceKind.BUILD,
        plane="oceanstack_dev",
        generation_id=generation_id,
        workspace=workspace,
        token=uuid.uuid4(),
    )
    token = bind_generation_operation_fence(fence)
    try:
        with pytest.raises(GenerationFenceError, match="PG_WORKSPACE"):
            await storage.initialize()
    finally:
        reset_generation_operation_fence(token)


@pytest.mark.asyncio
async def test_client_manager_threads_exact_build_context_through_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id = uuid.uuid4()
    workspace = generation_workspace("oceanstack_dev", generation_id)
    db = MagicMock()
    db.initdb = AsyncMock()
    db.check_tables = AsyncMock()
    db.pool = None
    db.workspace = None
    original_instances = ClientManager._instances
    ClientManager._instances = {
        "db": None,
        "ref_count": 0,
        "vector_signature": None,
    }
    ClientManager._lock = asyncio.Lock()
    monkeypatch.setattr(postgres_impl, "PostgreSQLDB", MagicMock(return_value=db))
    monkeypatch.setattr(
        ClientManager,
        "get_config",
        staticmethod(lambda vector_storage=None: {"enable_vector": False}),
    )
    fence = GenerationOperationFence(
        kind=GenerationFenceKind.BUILD,
        plane="oceanstack_dev",
        generation_id=generation_id,
        workspace=workspace,
        token=uuid.uuid4(),
    )
    token = bind_generation_operation_fence(fence)
    try:
        result = await ClientManager.get_client(
            vector_storage=None,
            operation_workspace=workspace,
            generation_access=GenerationStorageAccess.WRITE,
        )
    finally:
        reset_generation_operation_fence(token)
        ClientManager._instances = original_instances
        ClientManager._lock = asyncio.Lock()

    assert result is db
    db.initdb.assert_awaited_once_with(
        operation_workspace=workspace,
        generation_access=GenerationStorageAccess.WRITE,
    )
    db.check_tables.assert_awaited_once_with(
        operation_workspace=workspace,
        generation_access=GenerationStorageAccess.WRITE,
    )


@pytest.mark.asyncio
async def test_strict_bootstrap_propagates_index_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.query = AsyncMock(side_effect=[None, []])
    db.execute = AsyncMock(
        side_effect=[None, RuntimeError("generic index creation failed")]
    )
    monkeypatch.setattr(
        postgres_impl,
        "TABLES",
        {
            "FRESH": {
                "qualified_name": "public.FRESH",
                "ddl": "CREATE TABLE public.FRESH (id text, workspace text)",
            }
        },
    )
    context = postgres_impl._BootstrapConnectionContext(
        connection=MagicMock(),
        workspace="kg_oceanstack_dev_00000000000000000000000000000000",
        generation_access=GenerationStorageAccess.WRITE,
    )
    token = postgres_impl._BOOTSTRAP_CONNECTION.set(context)
    try:
        with pytest.raises(RuntimeError, match="generic index creation failed"):
            await db._check_tables_impl()
    finally:
        postgres_impl._BOOTSTRAP_CONNECTION.reset(token)


@pytest.mark.asyncio
async def test_read_fence_authorizes_ready_generation_without_active_pointer() -> None:
    generation_id = uuid.uuid4()
    workspace = generation_workspace("oceanstack_dev", generation_id)
    fence = GenerationOperationFence(
        kind=GenerationFenceKind.READ,
        plane="oceanstack_dev",
        generation_id=generation_id,
        workspace=workspace,
        token=uuid.uuid4(),
    )
    connection = AsyncMock()
    connection.fetchval = AsyncMock(side_effect=[None, True])

    await PostgreSQLDB._authorize_generation_connection(connection, fence)

    authorization_sql = connection.fetchval.await_args_list[1].args[0]
    assert "state='ready'" in authorization_sql
    assert "lightrag_graph_plane" not in authorization_sql
    assert connection.fetchval.await_args_list[1].args[1:4] == (
        fence.plane,
        fence.generation_id,
        fence.workspace,
    )


@pytest.mark.asyncio
async def test_client_manager_read_initialization_validates_without_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id = uuid.uuid4()
    workspace = generation_workspace("oceanstack_dev", generation_id)
    db = MagicMock()
    db.initdb = AsyncMock()
    db.validate_tables = AsyncMock()
    db.check_tables = AsyncMock()
    db.pool = None
    db.workspace = None
    original_instances = ClientManager._instances
    ClientManager._instances = {
        "db": None,
        "ref_count": 0,
        "vector_signature": None,
    }
    ClientManager._lock = asyncio.Lock()
    monkeypatch.setattr(postgres_impl, "PostgreSQLDB", MagicMock(return_value=db))
    monkeypatch.setattr(
        ClientManager,
        "get_config",
        staticmethod(lambda vector_storage=None: {"enable_vector": False}),
    )
    fence = GenerationOperationFence(
        kind=GenerationFenceKind.READ,
        plane="oceanstack_dev",
        generation_id=generation_id,
        workspace=workspace,
        token=uuid.uuid4(),
    )
    token = bind_generation_operation_fence(fence)
    try:
        await ClientManager.get_client(
            vector_storage=None,
            operation_workspace=workspace,
            generation_access=GenerationStorageAccess.READ,
        )
    finally:
        reset_generation_operation_fence(token)
        ClientManager._instances = original_instances
        ClientManager._lock = asyncio.Lock()

    db.initdb.assert_awaited_once_with(
        operation_workspace=workspace,
        generation_access=GenerationStorageAccess.READ,
    )
    db.validate_tables.assert_awaited_once_with(
        operation_workspace=workspace,
        generation_access=GenerationStorageAccess.READ,
    )
    db.check_tables.assert_not_awaited()
