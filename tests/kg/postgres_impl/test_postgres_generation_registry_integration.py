from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
import lightrag.kg.postgres_impl as postgres_impl

from lightrag.generation import (
    FailedGenerationCleanup,
    GenerationCandidate,
    GenerationFenceError,
    GenerationState,
    GenerationStorageAccess,
    GenerationValidationError,
)
from lightrag.kg.postgres_impl import (
    ClientManager,
    GenerationLeaseError,
    GenerationTransitionError,
    PGDocStatusStorage,
    PGGraphStorage,
    PGVectorStorage,
    PostgreSQLDB,
    PostgresGenerationRegistry,
)
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc


pytestmark = [pytest.mark.integration, pytest.mark.requires_db]
_REGISTRY_SCHEMA_TEST_LOCK = 7_103_248_197_462_155_842


def _connection_kwargs(database: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "database": database,
        "user": os.getenv("POSTGRES_USER", os.getenv("USER", "postgres")),
    }
    for env_name, key, converter in (
        ("POSTGRES_HOST", "host", str),
        ("POSTGRES_PORT", "port", int),
        ("POSTGRES_PASSWORD", "password", str),
    ):
        value = os.getenv(env_name)
        if value:
            kwargs[key] = converter(value)
    return kwargs


class _RegistryDB:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def _ensure_pool(self) -> None:
        return None

    async def execute(
        self, sql: str, data: dict[str, Any] | None = None, **_kwargs: object
    ) -> None:
        async with self.pool.acquire() as connection:
            await connection.execute(sql, *(data.values() if data else ()))


async def _assert_live_state_shape(connection: asyncpg.Connection) -> None:
    definition = await connection.fetchval(
        """
        SELECT pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'public.lightrag_graph_generation'::regclass
          AND conname = 'lightrag_graph_generation_state_shape'
        """
    )
    assert definition is not None
    normalized = " ".join(str(definition).casefold().split())
    building_start = normalized.index("state = 'building'")
    ready_start = normalized.index("state = 'ready'")
    failed_start = normalized.index("state = 'failed'")
    building_clause = normalized[building_start:ready_start]
    ready_clause = normalized[ready_start:failed_start]

    assert "published_at is null" in building_clause
    assert "lease_token is null" in ready_clause
    assert "worker_id is null" in ready_clause
    assert "lease_heartbeat is null" in ready_clause
    assert "lease_expires is null" in ready_clause


@pytest_asyncio.fixture
async def registry() -> PostgresGenerationRegistry:
    database = os.getenv("LIGHTRAG_PG_TEST_DATABASE")
    if not database:
        pytest.skip("set LIGHTRAG_PG_TEST_DATABASE to an isolated test database")
    if database.casefold() == "oceanstack" or "test" not in database.casefold():
        pytest.fail("LIGHTRAG_PG_TEST_DATABASE must name an isolated test database")
    pool = await asyncpg.create_pool(
        **_connection_kwargs(database), min_size=1, max_size=8
    )
    value = PostgresGenerationRegistry(_RegistryDB(pool))  # type: ignore[arg-type]
    lock_connection = await pool.acquire()
    locked = False
    try:
        actual_database = await lock_connection.fetchval("SELECT current_database()")
        if actual_database != database:
            pytest.fail("connected database does not match LIGHTRAG_PG_TEST_DATABASE")
        while not locked:
            locked = bool(
                await lock_connection.fetchval(
                    "SELECT pg_try_advisory_lock($1)", _REGISTRY_SCHEMA_TEST_LOCK
                )
            )
            if not locked:
                await asyncio.sleep(0.05)
        await lock_connection.execute(
            "DROP TABLE IF EXISTS public.lightrag_generation_fence_probe, "
            "public.lightrag_graph_generation, "
            "public.lightrag_graph_plane"
        )
        await value.bootstrap()
        await lock_connection.execute(
            "CREATE TABLE public.lightrag_generation_fence_probe ("
            "generation_id UUID NOT NULL, value TEXT NOT NULL, "
            "PRIMARY KEY (generation_id, value))"
        )
        await _assert_live_state_shape(lock_connection)
        yield value
    finally:
        try:
            if locked:
                try:
                    await lock_connection.execute(
                        "DROP TABLE IF EXISTS "
                        "public.lightrag_generation_fence_probe, "
                        "public.lightrag_graph_generation, "
                        "public.lightrag_graph_plane"
                    )
                finally:
                    await lock_connection.fetchval(
                        "SELECT pg_advisory_unlock($1)", _REGISTRY_SCHEMA_TEST_LOCK
                    )
        finally:
            await pool.release(lock_connection)
            await pool.close()


def _plane() -> str:
    return f"t_{uuid.uuid4().hex[:12]}"


def _acquired_pool_connections(pool: asyncpg.Pool) -> int:
    return sum(bool(holder._in_use) for holder in pool._holders)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_concurrent_fresh_bootstrap_serializes_without_repair(
    registry: PostgresGenerationRegistry,
) -> None:
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "DROP TABLE public.lightrag_graph_generation, public.lightrag_graph_plane"
        )

    await asyncio.wait_for(
        asyncio.gather(registry.bootstrap(), registry.bootstrap()), timeout=10
    )

    async with registry.db.pool.acquire() as connection:
        await _assert_live_state_shape(connection)


@pytest.mark.asyncio
async def test_bootstrap_rejects_column_drift_without_repair(
    registry: PostgresGenerationRegistry,
) -> None:
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "ALTER TABLE public.lightrag_graph_generation ADD COLUMN drift TEXT"
        )

    with pytest.raises(RuntimeError, match="columns drift"):
        await registry.bootstrap()


@pytest.mark.asyncio
async def test_bootstrap_rejects_same_name_index_drift_without_repair(
    registry: PostgresGenerationRegistry,
) -> None:
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "DROP INDEX public.idx_lightrag_graph_generation_stale_unleased"
        )
        await connection.execute(
            "CREATE INDEX idx_lightrag_graph_generation_stale_unleased "
            "ON public.lightrag_graph_generation (started_at) "
            "WHERE state='building'"
        )

    with pytest.raises(RuntimeError, match="index definitions drift"):
        await registry.bootstrap()


@pytest.mark.asyncio
async def test_bootstrap_rejects_same_name_constraint_drift(
    registry: PostgresGenerationRegistry,
) -> None:
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "ALTER TABLE public.lightrag_graph_generation "
            "DROP CONSTRAINT lightrag_graph_generation_workspace_length"
        )
        await connection.execute(
            "ALTER TABLE public.lightrag_graph_generation "
            "ADD CONSTRAINT lightrag_graph_generation_workspace_length "
            "CHECK (octet_length(workspace) <= 62)"
        )

    with pytest.raises(RuntimeError, match="constraint definitions drift"):
        await registry.bootstrap()


@pytest.mark.asyncio
async def test_bootstrap_rejects_default_drift(
    registry: PostgresGenerationRegistry,
) -> None:
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "ALTER TABLE public.lightrag_graph_generation "
            "ALTER COLUMN counts SET DEFAULT '{\"drift\": 1}'::jsonb"
        )

    with pytest.raises(RuntimeError, match="columns drift"):
        await registry.bootstrap()


@pytest.mark.asyncio
async def test_bootstrap_rejects_unlogged_registry_tables(
    registry: PostgresGenerationRegistry,
) -> None:
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "ALTER TABLE public.lightrag_graph_plane "
            "DROP CONSTRAINT lightrag_graph_plane_active_generation_fk"
        )
        await connection.execute(
            "ALTER TABLE public.lightrag_graph_generation "
            "DROP CONSTRAINT lightrag_graph_generation_plane_fk"
        )
        await connection.execute("ALTER TABLE public.lightrag_graph_plane SET UNLOGGED")
        await connection.execute(
            "ALTER TABLE public.lightrag_graph_generation SET UNLOGGED"
        )

    with pytest.raises(RuntimeError, match="table properties drift"):
        await registry.bootstrap()


def _candidate(
    plane: str, generation_id: uuid.UUID | None = None
) -> GenerationCandidate:
    generation_id = generation_id or uuid.uuid4()
    manifest = {"sources": ["src/oceanstack/core.py"]}
    return GenerationCandidate(
        plane=plane,
        generation_id=generation_id,
        build_id=f"build-{generation_id.hex}",
        contract_digest="a" * 64,
        manifest_digest=hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        manifest=manifest,
        metadata={"source_revision": "abc123"},
    )


def _storage_db(registry: PostgresGenerationRegistry) -> PostgreSQLDB:
    db = PostgreSQLDB(
        {
            "host": "unused",
            "port": 5432,
            "user": "unused",
            "password": "unused",
            "database": "unused",
            "workspace": None,
            "max_connections": 8,
            "enable_vector": False,
            "connection_retry_attempts": 1,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
            "pool_close_timeout": 1,
        }
    )
    db.pool = registry.db.pool
    return db


async def _make_ready(
    registry: PostgresGenerationRegistry, candidate: GenerationCandidate
) -> None:
    await registry.create_candidate(candidate)
    async with registry.acquire_build_lease(
        candidate.plane,
        candidate.generation_id,
        worker_id=f"worker-{candidate.generation_id.hex}",
        ttl=timedelta(seconds=30),
    ) as lease:
        await lease.mark_ready(
            counts={"chunks": 1, "entities": 2, "assertions": 1},
            storage_flushed=True,
            gates_passed=True,
        )


@pytest.mark.asyncio
async def test_build_and_cleanup_contexts_hold_no_pooled_connection(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    building = _candidate(plane)
    failed = _candidate(plane)
    await registry.create_candidate(building)
    await registry.create_candidate(failed)
    async with registry.acquire_build_lease(
        plane,
        failed.generation_id,
        worker_id="worker-fail-for-cleanup",
        ttl=timedelta(seconds=30),
    ) as lease:
        assert await lease.mark_failed({"code": "expected_failure"})

    baseline = _acquired_pool_connections(registry.db.pool)
    async with registry.acquire_build_lease(
        plane,
        building.generation_id,
        worker_id="worker-no-held-connection",
        ttl=timedelta(seconds=30),
    ):
        assert _acquired_pool_connections(registry.db.pool) == baseline

    async with registry.acquire_failed_cleanup(plane, failed.generation_id):
        assert _acquired_pool_connections(registry.db.pool) == baseline


@pytest.mark.asyncio
async def test_candidate_lease_ready_publish_and_active_guards(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    created = await registry.create_candidate(candidate)
    assert created.state is GenerationState.BUILDING
    assert await registry.resolve_active(plane) is None

    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-a",
        ttl=timedelta(seconds=30),
    ) as lease:
        heartbeat = await lease.heartbeat(ttl=timedelta(seconds=30))
        assert heartbeat.lease_token == lease.token
        with pytest.raises(GenerationTransitionError, match="storage_flushed"):
            await lease.mark_ready(
                counts={"chunks": 1},
                storage_flushed=False,
                gates_passed=True,
            )
        ready = await lease.mark_ready(
            counts={"chunks": 1}, storage_flushed=True, gates_passed=True
        )
        assert ready.state is GenerationState.READY

    with pytest.raises(TypeError, match="expected_active_generation_id"):
        await registry.publish(plane, candidate.generation_id)  # type: ignore[call-arg]

    result = await registry.publish(
        plane, candidate.generation_id, expected_active_generation_id=None
    )
    assert result.active.generation_id == candidate.generation_id
    assert result.superseded is None
    assert (await registry.resolve_active(plane)).state is GenerationState.READY  # type: ignore[union-attr]
    drop = AsyncMock()
    cleanup = FailedGenerationCleanup(registry, drop)
    with pytest.raises(GenerationTransitionError, match="inactive failed"):
        await cleanup.cleanup(plane, candidate.generation_id)
    drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_cas_two_publishers_and_superseded_cleanup_target(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    old = _candidate(plane)
    first = _candidate(plane)
    second = _candidate(plane)
    await _make_ready(registry, old)
    await registry.publish(plane, old.generation_id, expected_active_generation_id=None)
    await _make_ready(registry, first)
    await _make_ready(registry, second)

    results = await asyncio.gather(
        registry.publish(
            plane, first.generation_id, expected_active_generation_id=old.generation_id
        ),
        registry.publish(
            plane, second.generation_id, expected_active_generation_id=old.generation_id
        ),
        return_exceptions=True,
    )

    successes = [item for item in results if not isinstance(item, BaseException)]
    failures = [item for item in results if isinstance(item, BaseException)]
    assert len(successes) == 1, repr(results)
    assert len(failures) == 1
    assert isinstance(failures[0], GenerationTransitionError)
    published = successes[0]
    assert published.superseded is not None
    assert published.superseded.generation_id == old.generation_id
    assert published.superseded.state is GenerationState.FAILED
    assert published.superseded.failure["code"] == "superseded"
    active = await registry.resolve_active(plane)
    assert active is not None
    assert active.generation_id == published.active.generation_id
    assert (
        await registry.get_generation(plane, old.generation_id)
    ).state is GenerationState.FAILED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_cross_plane_active_pointer_is_rejected_by_composite_fk(
    registry: PostgresGenerationRegistry,
) -> None:
    source_plane = _plane()
    target_plane = _plane()
    candidate = _candidate(source_plane)
    await _make_ready(registry, candidate)
    await registry.create_candidate(_candidate(target_plane))

    with pytest.raises(asyncpg.ForeignKeyViolationError):
        async with registry.db.pool.acquire() as connection:
            await connection.execute(
                "UPDATE public.lightrag_graph_plane SET active_generation_id=$1 "
                "WHERE plane=$2",
                candidate.generation_id,
                target_plane,
            )


@pytest.mark.asyncio
async def test_stale_failure_fences_resumed_worker_without_a_session_lock(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-stale",
        ttl=timedelta(seconds=30),
    ) as lease:
        async with registry.db.pool.acquire() as connection:
            await connection.execute(
                "UPDATE public.lightrag_graph_generation "
                "SET lease_heartbeat=clock_timestamp()-interval '2 seconds', "
                "lease_expires=clock_timestamp()-interval '1 second' "
                "WHERE plane=$1 AND generation_id=$2",
                plane,
                candidate.generation_id,
            )
        failed = await registry.fail_stale()
        assert [item.generation_id for item in failed] == [candidate.generation_id]
        with pytest.raises(GenerationLeaseError, match="stale"):
            await lease.heartbeat()
        async with registry.acquire_failed_cleanup(
            plane, candidate.generation_id
        ) as claim:
            assert claim.workspace == candidate.workspace


@pytest.mark.asyncio
async def test_inflight_storage_transaction_precedes_stale_failure_and_cleanup(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    storage_db = _storage_db(registry)
    entered = asyncio.Event()
    release = asyncio.Event()

    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-inflight",
        ttl=timedelta(seconds=30),
    ) as lease:

        async def storage_operation(connection: asyncpg.Connection) -> int:
            await connection.execute(
                "INSERT INTO public.lightrag_generation_fence_probe "
                "(generation_id, value) VALUES ($1, $2)",
                candidate.generation_id,
                "committed-before-stale",
            )
            entered.set()
            await release.wait()
            return 1

        operation_task = asyncio.create_task(
            storage_db._run_with_retry(
                storage_operation,
                operation_workspace=candidate.workspace,
            )
        )
        await entered.wait()

        heartbeat = await lease.heartbeat(ttl=timedelta(seconds=30))
        assert heartbeat.lease_token == lease.token
        async with registry.db.pool.acquire() as connection:
            await connection.execute(
                "UPDATE public.lightrag_graph_generation "
                "SET lease_heartbeat=clock_timestamp()-interval '2 seconds', "
                "lease_expires=clock_timestamp()-interval '1 second' "
                "WHERE plane=$1 AND generation_id=$2",
                plane,
                candidate.generation_id,
            )

        stale_task = asyncio.create_task(registry.fail_stale())
        await asyncio.sleep(0.05)
        assert not stale_task.done()
        release.set()
        assert await operation_task == 1
        failed = await stale_task
        assert [item.generation_id for item in failed] == [candidate.generation_id]

        with pytest.raises(GenerationFenceError, match="stale or expired"):
            await storage_db._run_with_retry(
                storage_operation,
                operation_workspace=candidate.workspace,
            )

        async with registry.acquire_failed_cleanup(
            plane, candidate.generation_id
        ) as claim:
            assert claim.workspace == candidate.workspace
            await storage_db._run_with_retry(
                lambda connection: connection.execute(
                    "DELETE FROM public.lightrag_generation_fence_probe "
                    "WHERE generation_id=$1",
                    candidate.generation_id,
                ),
                operation_workspace=candidate.workspace,
            )

        async def recreate(connection: asyncpg.Connection) -> None:
            await connection.execute(
                "INSERT INTO public.lightrag_generation_fence_probe "
                "(generation_id, value) VALUES ($1, $2)",
                candidate.generation_id,
                "recreated-by-stale-build",
            )

        with pytest.raises(GenerationFenceError, match="stale or expired"):
            await storage_db._run_with_retry(
                recreate,
                operation_workspace=candidate.workspace,
            )

    async with registry.db.pool.acquire() as connection:
        assert (
            await connection.fetchval(
                "SELECT count(*) FROM public.lightrag_generation_fence_probe "
                "WHERE generation_id=$1",
                candidate.generation_id,
            )
            == 0
        )


@pytest.mark.asyncio
async def test_mark_ready_waits_for_storage_commit_and_fences_inherited_writer(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    storage_db = _storage_db(registry)
    entered = asyncio.Event()
    release = asyncio.Event()
    child_go = asyncio.Event()

    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-ready-barrier",
        ttl=timedelta(seconds=30),
    ) as lease:

        async def mutate(connection: asyncpg.Connection) -> None:
            await connection.execute(
                "INSERT INTO public.lightrag_generation_fence_probe "
                "(generation_id, value) VALUES ($1, $2)",
                candidate.generation_id,
                "committed-before-ready",
            )
            entered.set()
            await release.wait()

        async def inherited_late_writer() -> None:
            await child_go.wait()
            await storage_db._run_with_retry(
                lambda connection: connection.execute(
                    "INSERT INTO public.lightrag_generation_fence_probe "
                    "(generation_id, value) VALUES ($1, $2)",
                    candidate.generation_id,
                    "late-child-write",
                ),
                operation_workspace=candidate.workspace,
            )

        mutation_task = asyncio.create_task(
            storage_db._run_with_retry(
                mutate,
                operation_workspace=candidate.workspace,
            )
        )
        inherited_task = asyncio.create_task(inherited_late_writer())
        await entered.wait()
        ready_task = asyncio.create_task(
            lease.mark_ready(
                counts={"chunks": 1},
                storage_flushed=True,
                gates_passed=True,
            )
        )
        await asyncio.sleep(0.05)
        assert not ready_task.done()
        release.set()
        await mutation_task
        ready = await ready_task
        assert ready.state is GenerationState.READY

        child_go.set()
        with pytest.raises(GenerationFenceError, match="stale or expired"):
            await inherited_task
        with pytest.raises(GenerationFenceError, match="stale or expired"):
            await storage_db._run_with_retry(
                lambda connection: connection.fetchval("SELECT 1"),
                operation_workspace=candidate.workspace,
            )

    async with registry.db.pool.acquire() as connection:
        values = await connection.fetch(
            "SELECT value FROM public.lightrag_generation_fence_probe "
            "WHERE generation_id=$1 ORDER BY value",
            candidate.generation_id,
        )
    assert [row["value"] for row in values] == ["committed-before-ready"]


@pytest.mark.asyncio
async def test_cleanup_claim_expiry_takeover_fences_old_token(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-cleanup-takeover",
        ttl=timedelta(seconds=30),
    ) as lease:
        assert await lease.mark_failed({"code": "expected_failure"})

    old_claim = registry.acquire_failed_cleanup(
        plane, candidate.generation_id, ttl=timedelta(seconds=30)
    )
    await old_claim.__aenter__()
    try:
        with pytest.raises(GenerationTransitionError, match="live claim"):
            async with registry.acquire_failed_cleanup(plane, candidate.generation_id):
                pytest.fail("a live cleanup token must not be overwritten")

        async with registry.db.pool.acquire() as connection:
            await connection.execute(
                "UPDATE public.lightrag_graph_generation "
                "SET cleanup_started_at=clock_timestamp()-interval '2 seconds', "
                "cleanup_expires=clock_timestamp()-interval '1 second' "
                "WHERE plane=$1 AND generation_id=$2",
                plane,
                candidate.generation_id,
            )

        async with registry.acquire_failed_cleanup(
            plane, candidate.generation_id
        ) as replacement:
            assert replacement.token != old_claim.token

        storage_db = _storage_db(registry)
        with pytest.raises(GenerationFenceError, match="stale or expired"):
            await storage_db._run_with_retry(
                lambda connection: connection.fetchval("SELECT 1"),
                operation_workspace=candidate.workspace,
            )
        with pytest.raises(GenerationTransitionError, match="stale or fenced"):
            await old_claim.record_failure({"code": "old_token"})
        assert not await old_claim.delete()
    finally:
        await old_claim.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_generation_statement_timeout_returns_pool_connection(
    registry: PostgresGenerationRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(postgres_impl, "_GENERATION_STATEMENT_TIMEOUT_MS", 50)
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    storage_db = _storage_db(registry)
    baseline = _acquired_pool_connections(registry.db.pool)

    with pytest.raises(asyncpg.QueryCanceledError):
        async with registry.acquire_build_lease(
            plane,
            candidate.generation_id,
            worker_id="worker-timeout",
            ttl=timedelta(seconds=30),
        ):
            await storage_db._run_with_retry(
                lambda connection: connection.fetchval("SELECT pg_sleep(0.2)"),
                operation_workspace=candidate.workspace,
            )

    assert _acquired_pool_connections(registry.db.pool) == baseline
    async with registry.db.pool.acquire() as connection:
        assert await connection.fetchval("SELECT 1") == 1


@pytest.mark.asyncio
async def test_cancelled_lease_releases_connection_and_advisory_lock(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    entered = asyncio.Event()

    async def hold() -> None:
        async with registry.acquire_build_lease(
            plane,
            candidate.generation_id,
            worker_id="worker-cancel",
            ttl=timedelta(seconds=30),
        ):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    failed = await registry.get_generation(plane, candidate.generation_id)
    assert failed is not None
    assert failed.state is GenerationState.FAILED
    assert failed.failure == {"code": "lease_cancelled"}

    async with registry.db.pool.acquire() as connection:
        acquired = await connection.fetchval(
            "SELECT pg_try_advisory_lock($1)",
            registry.advisory_key(plane, candidate.generation_id),
        )
        assert acquired
        assert await connection.fetchval(
            "SELECT pg_advisory_unlock($1)",
            registry.advisory_key(plane, candidate.generation_id),
        )


@pytest.mark.asyncio
async def test_cancellation_during_normal_exit_finishes_cleanup_then_reraises(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    original = registry._abandon_build_lease

    async def delayed_abandon(lease: object, failure: object) -> None:
        cleanup_started.set()
        await allow_cleanup.wait()
        await original(lease, failure)  # type: ignore[arg-type]

    registry._abandon_build_lease = delayed_abandon  # type: ignore[method-assign]
    try:

        async def normal_body() -> None:
            async with registry.acquire_build_lease(
                plane,
                candidate.generation_id,
                worker_id="worker-cancel-exit",
                ttl=timedelta(seconds=30),
            ):
                pass

        task = asyncio.create_task(normal_body())
        await cleanup_started.wait()
        task.cancel()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        registry._abandon_build_lease = original  # type: ignore[method-assign]

    failed = await registry.get_generation(plane, candidate.generation_id)
    assert failed is not None
    assert failed.state is GenerationState.FAILED
    assert failed.failure == {"code": "lease_released_without_terminal_state"}


@pytest.mark.asyncio
async def test_cleanup_claim_cancellation_during_exit_releases_then_reraises(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-cleanup-cancel",
        ttl=timedelta(seconds=30),
    ) as lease:
        assert await lease.mark_failed({"code": "expected_failure"})

    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    original = registry._release_cleanup_claim

    async def delayed_release(claim: object) -> None:
        cleanup_started.set()
        await allow_cleanup.wait()
        await original(claim)  # type: ignore[arg-type]

    registry._release_cleanup_claim = delayed_release  # type: ignore[method-assign]
    try:

        async def normal_body() -> None:
            async with registry.acquire_failed_cleanup(plane, candidate.generation_id):
                pass

        task = asyncio.create_task(normal_body())
        await cleanup_started.wait()
        task.cancel()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        registry._release_cleanup_claim = original  # type: ignore[method-assign]

    async with registry.db.pool.acquire() as connection:
        row = await connection.fetchrow(
            "SELECT cleanup_token, cleanup_started_at, cleanup_expires "
            "FROM public.lightrag_graph_generation "
            "WHERE plane=$1 AND generation_id=$2",
            plane,
            candidate.generation_id,
        )
    assert row is not None
    assert tuple(row) == (None, None, None)


@pytest.mark.asyncio
async def test_cleanup_claim_release_waits_for_shared_storage_transaction(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-cleanup-release-lock",
        ttl=timedelta(seconds=30),
    ) as lease:
        assert await lease.mark_failed({"code": "expected_failure"})

    storage_entered = asyncio.Event()
    allow_storage_exit = asyncio.Event()
    cleanup_body_exiting = asyncio.Event()
    cleanup_context_exited = asyncio.Event()
    storage_db = _storage_db(registry)

    async def hold_storage_transaction(connection: asyncpg.Connection) -> None:
        storage_entered.set()
        await allow_storage_exit.wait()
        await connection.fetchval("SELECT 1")

    async def cleanup_owner() -> None:
        storage_task: asyncio.Task[None]
        async with registry.acquire_failed_cleanup(plane, candidate.generation_id):
            storage_task = asyncio.create_task(
                storage_db._run_with_retry(
                    hold_storage_transaction,
                    operation_workspace=candidate.workspace,
                    generation_access=GenerationStorageAccess.DROP,
                )
            )
            await storage_entered.wait()
            cleanup_body_exiting.set()
        cleanup_context_exited.set()
        await storage_task

    owner_task = asyncio.create_task(cleanup_owner())
    await cleanup_body_exiting.wait()
    await asyncio.sleep(0.05)
    assert not cleanup_context_exited.is_set()
    allow_storage_exit.set()
    await owner_task
    assert cleanup_context_exited.is_set()

    async with registry.db.pool.acquire() as connection:
        token = await connection.fetchval(
            "SELECT cleanup_token FROM public.lightrag_graph_generation "
            "WHERE plane=$1 AND generation_id=$2",
            plane,
            candidate.generation_id,
        )
    assert token is None


@pytest.mark.asyncio
async def test_fresh_vector_bootstrap_reuses_exact_schema_and_rejects_drift(
    registry: PostgresGenerationRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    db = _storage_db(registry)
    db.vector_index_type = "HNSW"
    db.hnsw_m = 16
    db.hnsw_ef = 64

    async def embed(texts: list[str], **_kwargs: object) -> np.ndarray:
        return np.zeros((len(texts), 3), dtype=np.float32)

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        workspace=candidate.workspace,
        global_config={
            "embedding_batch_num": 8,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=3,
            func=embed,
            model_name=f"generation_{candidate.generation_id.hex[:12]}",
        ),
        db=db,
    )

    @asynccontextmanager
    async def init_lock():
        yield

    monkeypatch.setattr(postgres_impl, "get_data_init_lock", init_lock)
    monkeypatch.setattr(
        postgres_impl,
        "get_namespace_lock",
        lambda *_args, **_kwargs: asyncio.Lock(),
    )
    try:
        async with registry.acquire_build_lease(
            plane,
            candidate.generation_id,
            worker_id="worker-vector-bootstrap",
            ttl=timedelta(seconds=30),
        ):
            await storage.initialize()
            assert storage.workspace == candidate.workspace
            await storage._initialize_current_table()

            async def assert_catalog_drift(mutate_sql: str, restore_sql: str) -> None:
                async with registry.db.pool.acquire() as connection:
                    await connection.execute(mutate_sql)
                with pytest.raises(
                    GenerationValidationError, match="exact current schema"
                ):
                    await storage._initialize_current_table()
                async with registry.db.pool.acquire() as connection:
                    await connection.execute(restore_sql)
                await storage._initialize_current_table()

            await assert_catalog_drift(
                f"ALTER TABLE {storage.table_name} SET UNLOGGED",
                f"ALTER TABLE {storage.table_name} SET LOGGED",
            )
            await assert_catalog_drift(
                f"ALTER TABLE {storage.table_name} ENABLE ROW LEVEL SECURITY",
                f"ALTER TABLE {storage.table_name} DISABLE ROW LEVEL SECURITY",
            )
            await assert_catalog_drift(
                f"ALTER TABLE {storage.table_name} FORCE ROW LEVEL SECURITY",
                f"ALTER TABLE {storage.table_name} NO FORCE ROW LEVEL SECURITY",
            )
            await assert_catalog_drift(
                f"ALTER TABLE {storage.table_name} "
                "ADD CONSTRAINT vector_drift_check CHECK (id <> '')",
                f"ALTER TABLE {storage.table_name} DROP CONSTRAINT vector_drift_check",
            )

            function_name = f"vector_drift_{candidate.generation_id.hex[:12]}"
            async with registry.db.pool.acquire() as connection:
                await connection.execute(
                    f"CREATE FUNCTION public.{function_name}() RETURNS trigger "
                    "LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$"
                )
            try:
                await assert_catalog_drift(
                    f"CREATE TRIGGER vector_drift_trigger BEFORE INSERT ON "
                    f"{storage.table_name} FOR EACH ROW EXECUTE FUNCTION "
                    f"public.{function_name}()",
                    f"DROP TRIGGER vector_drift_trigger ON {storage.table_name}",
                )
            finally:
                async with registry.db.pool.acquire() as connection:
                    await connection.execute(
                        f"DROP FUNCTION IF EXISTS public.{function_name}()"
                    )

            vector_index = storage._fresh_vector_index_sql()
            assert vector_index is not None
            await assert_catalog_drift(
                f"ALTER INDEX {vector_index[0]} SET (m=32)",
                f"ALTER INDEX {vector_index[0]} SET (m=16)",
            )

            async with registry.db.pool.acquire() as connection:
                await connection.execute(f"DROP TABLE {storage.table_name}")
                await connection.execute(
                    f"CREATE VIEW {storage.table_name} AS SELECT NULL::text AS id"
                )
            try:
                with pytest.raises(
                    GenerationValidationError, match="exact current schema"
                ):
                    await storage._initialize_current_table()
            finally:
                async with registry.db.pool.acquire() as connection:
                    await connection.execute(f"DROP VIEW {storage.table_name}")
            await storage._initialize_current_table()

            async with registry.db.pool.acquire() as connection:
                await connection.execute(
                    f"ALTER TABLE {storage.table_name} ADD COLUMN drift TEXT"
                )
            with pytest.raises(GenerationValidationError, match="exact current schema"):
                await storage._initialize_current_table()
    finally:
        async with registry.db.pool.acquire() as connection:
            await connection.execute(f"DROP TABLE IF EXISTS {storage.table_name}")


def _client_manager_config(database: str) -> dict[str, Any]:
    connection = _connection_kwargs(database)
    return {
        "host": connection.get("host"),
        "port": connection.get("port", 5432),
        "user": connection["user"],
        "password": connection.get("password", ""),
        "database": database,
        "workspace": None,
        "max_connections": 4,
        "enable_vector": True,
        "vector_index_type": "HNSW",
        "hnsw_m": 16,
        "hnsw_ef": 64,
        "ivfflat_lists": 100,
        "vchordrq_build_options": "",
        "vchordrq_probes": "",
        "vchordrq_epsilon": 1.9,
        "connection_retry_attempts": 1,
        "connection_retry_backoff": 0,
        "connection_retry_backoff_max": 0,
        "pool_close_timeout": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_kind", ["vector", "doc_status", "graph"])
async def test_fresh_storage_bootstrap_runs_under_exact_build_fence(
    registry: PostgresGenerationRegistry,
    monkeypatch: pytest.MonkeyPatch,
    storage_kind: str,
) -> None:
    database = os.environ["LIGHTRAG_PG_TEST_DATABASE"]
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    async with registry.db.pool.acquire() as connection:
        await connection.execute("DROP TABLE IF EXISTS public.lightrag_doc_status")

    original_instances = ClientManager._instances
    original_lock = ClientManager._lock
    ClientManager._instances = {
        "db": None,
        "ref_count": 0,
        "vector_signature": None,
    }
    ClientManager._lock = asyncio.Lock()
    monkeypatch.setattr(
        ClientManager,
        "get_config",
        staticmethod(lambda vector_storage=None: _client_manager_config(database)),
    )

    @asynccontextmanager
    async def init_lock():
        yield

    monkeypatch.setattr(postgres_impl, "get_data_init_lock", init_lock)
    monkeypatch.setattr(
        postgres_impl,
        "get_namespace_lock",
        lambda *_args, **_kwargs: asyncio.Lock(),
    )

    if storage_kind == "vector":

        async def embed(texts: list[str], **_kwargs: object) -> np.ndarray:
            return np.zeros((len(texts), 3), dtype=np.float32)

        storage: PGVectorStorage | PGDocStatusStorage | PGGraphStorage = (
            PGVectorStorage(
                namespace=NameSpace.VECTOR_STORE_CHUNKS,
                workspace=candidate.workspace,
                global_config={
                    "vector_storage": "PGVectorStorage",
                    "embedding_batch_num": 8,
                    "vector_db_storage_cls_kwargs": {
                        "cosine_better_than_threshold": 0.5
                    },
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=3,
                    func=embed,
                    model_name=f"bootstrap_{candidate.generation_id.hex[:12]}",
                ),
            )
        )
    elif storage_kind == "doc_status":
        storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
        storage.workspace = candidate.workspace
        storage.namespace = NameSpace.DOC_STATUS
        storage.global_config = {"vector_storage": "PGVectorStorage"}
        storage.db = None
        storage.__post_init__()
    else:
        storage = PGGraphStorage.__new__(PGGraphStorage)
        storage.workspace = candidate.workspace
        storage.namespace = NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
        storage.global_config = {
            "vector_storage": "PGVectorStorage",
            "max_graph_nodes": 1000,
        }
        storage.__post_init__()

    try:
        async with registry.acquire_build_lease(
            plane,
            candidate.generation_id,
            worker_id=f"worker-bootstrap-{storage_kind}",
            ttl=timedelta(seconds=30),
        ) as lease:
            await storage.initialize()
            assert storage.workspace == candidate.workspace
            async with registry.db.pool.acquire() as connection:
                assert await connection.fetchval(
                    "SELECT to_regclass('public.lightrag_doc_status') IS NOT NULL"
                )
                if storage_kind == "vector":
                    assert await connection.fetchval(
                        "SELECT to_regclass($1) IS NOT NULL", storage.table_name
                    )
                elif storage_kind == "graph":
                    assert await connection.fetchval(
                        "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=$1)",
                        storage.graph_name,
                    )
            assert await lease.mark_failed({"code": "test_cleanup"})

        async with registry.acquire_failed_cleanup(plane, candidate.generation_id):
            result = await storage.drop()
            assert result["status"] == "success"
        await storage.finalize()
    finally:
        active_db = ClientManager._instances["db"]
        if active_db is not None and active_db.pool is not None:
            await active_db.pool.close()
        ClientManager._instances = original_instances
        ClientManager._lock = original_lock


@pytest.mark.asyncio
async def test_unleased_timeout_and_abandoned_context_become_failed(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    unleased = _candidate(plane)
    abandoned = _candidate(plane)
    await registry.create_candidate(unleased)
    await registry.create_candidate(abandoned)
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "UPDATE public.lightrag_graph_generation "
            "SET started_at=clock_timestamp()-interval '1 day' "
            "WHERE plane=$1 AND generation_id=$2",
            plane,
            unleased.generation_id,
        )

    failed = await registry.fail_stale()
    assert [generation.generation_id for generation in failed] == [
        unleased.generation_id
    ]
    assert failed[0].failure == {"code": "stale_unleased_candidate"}

    async with registry.acquire_build_lease(
        plane,
        abandoned.generation_id,
        worker_id="worker-abandoned",
        ttl=timedelta(seconds=30),
    ):
        pass

    abandoned_record = await registry.get_generation(plane, abandoned.generation_id)
    assert abandoned_record is not None
    assert abandoned_record.state is GenerationState.FAILED
    assert abandoned_record.failure == {"code": "lease_released_without_terminal_state"}


@pytest.mark.asyncio
async def test_lease_body_exception_fails_owned_generation(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)

    with pytest.raises(RuntimeError, match="build failed"):
        async with registry.acquire_build_lease(
            plane,
            candidate.generation_id,
            worker_id="worker-exception",
            ttl=timedelta(seconds=30),
        ):
            raise RuntimeError("build failed")

    failed = await registry.get_generation(plane, candidate.generation_id)
    assert failed is not None
    assert failed.state is GenerationState.FAILED
    assert failed.failure == {
        "code": "lease_exception",
        "error_type": "RuntimeError",
    }


@pytest.mark.asyncio
async def test_ready_generation_cannot_be_failed_by_public_transition(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await _make_ready(registry, candidate)

    assert not hasattr(registry, "heartbeat")
    assert not hasattr(registry, "mark_ready")
    assert not hasattr(registry, "mark_failed")
    assert (
        await registry.get_generation(plane, candidate.generation_id)
    ).state is GenerationState.READY  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_build_failure_cleanup_metadata_and_registry_last_delete(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-fail",
        ttl=timedelta(seconds=30),
    ) as lease:
        assert await lease.mark_failed({"code": "build_gate_failed"})

    async with registry.acquire_failed_cleanup(plane, candidate.generation_id) as claim:
        await claim.record_failure(
            {"code": "storage_cleanup_failed", "storage": "vectors"}
        )
        failed = await registry.get_generation(plane, candidate.generation_id)
        assert failed is not None
        assert failed.state is GenerationState.FAILED
        assert failed.cleanup_failure == {
            "code": "storage_cleanup_failed",
            "storage": "vectors",
        }
        assert await claim.delete()
    assert await registry.get_generation(plane, candidate.generation_id) is None


@pytest.mark.asyncio
async def test_cleanup_uses_only_registry_derived_workspace(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await registry.create_candidate(candidate)
    async with registry.acquire_build_lease(
        plane,
        candidate.generation_id,
        worker_id="worker-corrupt-workspace",
        ttl=timedelta(seconds=30),
    ) as lease:
        assert await lease.mark_failed({"code": "build_failed"})
    async with registry.db.pool.acquire() as connection:
        await connection.execute(
            "UPDATE public.lightrag_graph_generation "
            "SET workspace=$3 WHERE plane=$1 AND generation_id=$2",
            plane,
            candidate.generation_id,
            f"kg_{plane}_{uuid.uuid4().hex}",
        )
    drop = AsyncMock()
    cleanup = FailedGenerationCleanup(registry, drop)

    with pytest.raises(GenerationTransitionError, match="stored cleanup workspace"):
        await cleanup.cleanup(plane, candidate.generation_id)

    drop.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolvers_observe_old_or_new_only_and_lists_are_plane_scoped(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    old = _candidate(plane)
    new = _candidate(plane)
    await _make_ready(registry, old)
    first_publish = await registry.publish(
        plane, old.generation_id, expected_active_generation_id=None
    )
    await _make_ready(registry, new)

    publish_task = asyncio.create_task(
        registry.publish(
            plane, new.generation_id, expected_active_generation_id=old.generation_id
        )
    )
    observed: set[uuid.UUID] = set()
    while not publish_task.done():
        resolved = await registry.resolve_active(plane)
        assert resolved is not None
        observed.add(resolved.generation_id)
    published = await publish_task
    observed.add((await registry.resolve_active(plane)).generation_id)  # type: ignore[union-attr]

    assert observed <= {old.generation_id, new.generation_id}
    assert published.plane.revision == first_publish.plane.revision + 1
    assert [item.generation_id for item in await registry.list_generations(plane)] == [
        old.generation_id,
        new.generation_id,
    ]
    assert plane in {item.plane for item in await registry.list_planes()}


@pytest.mark.asyncio
async def test_republishing_current_active_generation_is_idempotent(
    registry: PostgresGenerationRegistry,
) -> None:
    plane = _plane()
    candidate = _candidate(plane)
    await _make_ready(registry, candidate)
    first = await registry.publish(
        plane, candidate.generation_id, expected_active_generation_id=None
    )

    second = await registry.publish(
        plane,
        candidate.generation_id,
        expected_active_generation_id=candidate.generation_id,
    )

    assert second.plane.revision == first.plane.revision
    assert second.plane.updated_at == first.plane.updated_at
    assert second.active.published_at == first.active.published_at
    assert second.superseded is None
