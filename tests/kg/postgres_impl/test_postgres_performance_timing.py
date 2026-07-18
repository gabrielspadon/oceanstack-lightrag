import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import lightrag.utils as utils_module
from lightrag.kg.postgres_impl import PGGraphStorage, PostgreSQLDB
from lightrag.namespace import NameSpace


def make_db() -> PostgreSQLDB:
    return PostgreSQLDB(
        {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "postgres",
            "workspace": "test_ws",
            "max_connections": 10,
            "connection_retry_attempts": 3,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
            "pool_close_timeout": 5.0,
        }
    )


@pytest.mark.asyncio
async def test_execute_timing_logs_success():
    db = make_db()

    async def fake_run_with_retry(operation, **kwargs):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        await operation(conn)

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    with patch("lightrag.kg.postgres_impl.performance_timing_log") as timing_log:
        await db.execute("SELECT 1", timing_label="test label")

    assert any(
        "connection.execute completed" in call.args[0]
        for call in timing_log.call_args_list
    )


@pytest.mark.asyncio
async def test_execute_timing_logs_failure():
    db = make_db()

    async def fake_run_with_retry(operation, **kwargs):
        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=RuntimeError("boom"))
        await operation(conn)

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    with patch("lightrag.kg.postgres_impl.performance_timing_log") as timing_log:
        with pytest.raises(RuntimeError, match="boom"):
            await db.execute("SELECT 1", timing_label="test label")

    assert any(
        "connection.execute failed" in call.args[0]
        for call in timing_log.call_args_list
    )


@pytest.mark.asyncio
async def test_graph_upsert_node_passes_timing_label():
    storage = PGGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace="test_ws",
        global_config={},
        embedding_func=AsyncMock(),
    )
    storage.graph_name = "test_graph"
    storage.db = MagicMock()
    storage.db._run_with_retry = AsyncMock(return_value=None)

    await storage.upsert_node(
        "node-1",
        {
            "entity_id": "node-1",
            "description": "desc",
        },
    )

    assert storage.db._run_with_retry.await_args.kwargs["timing_label"] == (
        "test_ws PGGraphStorage.upsert_node"
    )


@pytest.mark.asyncio
async def test_graph_upsert_edge_passes_timing_label():
    storage = PGGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace="test_ws",
        global_config={},
        embedding_func=AsyncMock(),
    )
    storage.graph_name = "test_graph"
    # upsert_edge drives the lock + cypher via db._run_with_retry, not _query.
    storage.db = MagicMock()
    storage.db._run_with_retry = AsyncMock(return_value=None)

    await storage.upsert_edge(
        "node-1",
        "node-2",
        {
            "weight": 1.0,
            "description": "desc",
        },
    )

    assert storage.db._run_with_retry.await_args.kwargs["timing_label"] == (
        "test_ws PGGraphStorage.upsert_edge"
    )


def _reload_utils_preserving_identities(monkeypatch, env, expected):
    """Reload lightrag.utils under `env` and restore original attribute identities.

    ``importlib.reload`` re-executes the module in place, rebinding every
    top-level name (classes, functions, constants) to fresh objects while the
    module's ``__dict__`` object stays the same. Names already imported by other
    test modules (e.g. ``TruncatedStr``/``mark_truncated``) keep pointing at the
    pre-reload objects, so a bare reload leaks new class identities process-wide
    and breaks unrelated ``isinstance`` checks. Snapshot the module dict and
    restore it afterwards so identities survive the reload.
    """
    original = dict(utils_module.__dict__)
    try:
        with monkeypatch.context() as m:
            for key, value in env.items():
                m.setenv(key, value)
            reloaded = importlib.reload(utils_module)
            assert reloaded.PERFORMANCE_TIMING_LOGS is expected
    finally:
        utils_module.__dict__.clear()
        utils_module.__dict__.update(original)


def test_performance_timing_logs_reads_new_env_only(monkeypatch):
    _reload_utils_preserving_identities(
        monkeypatch,
        {
            "LIGHTRAG_DOC_QUERY_TIMING_LOGS": "false",
            "LIGHTRAG_PERFORMANCE_TIMING_LOGS": "true",
        },
        expected=True,
    )


def test_performance_timing_logs_ignores_old_env(monkeypatch):
    _reload_utils_preserving_identities(
        monkeypatch,
        {
            "LIGHTRAG_DOC_QUERY_TIMING_LOGS": "true",
            "LIGHTRAG_PERFORMANCE_TIMING_LOGS": "false",
        },
        expected=False,
    )
