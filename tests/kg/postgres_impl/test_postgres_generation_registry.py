from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lightrag.generation import (
    GenerationRegistry,
    GenerationValidationError,
    generation_advisory_key,
)
from lightrag.kg.postgres_impl import (
    GENERATION_INDEX_DDL,
    GENERATION_SCHEMA_DDL,
    PostgresGenerationRegistry,
    _generation_from_row,
    _plane_from_row,
)


def _building_row() -> dict[str, object]:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    digest = "b" * 64
    manifest = {"digest": digest, "sources": ["src/oceanstack/core.py"]}
    return {
        "plane": "oceanstack_dev",
        "generation_id": generation_id,
        "workspace": f"kg_oceanstack_dev_{generation_id.hex}",
        "state": "building",
        "build_id": "build-dev-001",
        "contract_digest": "a" * 64,
        "manifest_digest": digest,
        "manifest": json.dumps(manifest),
        "metadata": "{}",
        "counts": "{}",
        "worker_id": None,
        "lease_token": None,
        "lease_heartbeat": None,
        "lease_expires": None,
        "started_at": datetime.now(timezone.utc),
        "ready_at": None,
        "published_at": None,
        "failed_at": None,
        "storage_flushed": False,
        "gates_passed": False,
        "failure": None,
        "cleanup_failure": None,
    }


def test_greenfield_registry_schema_has_cross_plane_fk_and_state_constraints() -> None:
    schema = "\n".join(GENERATION_SCHEMA_DDL).lower()

    assert "lightrag_graph_plane" in schema
    assert "lightrag_graph_generation" in schema
    assert "unique (plane, generation_id)" in schema
    assert "foreign key (plane, active_generation_id)" in schema
    assert "references public.lightrag_graph_generation(plane, generation_id)" in schema
    assert "state in ('building', 'ready', 'failed')" in schema
    assert "jsonb_typeof(manifest) = 'object'" in schema
    assert "manifest <> '{}'::jsonb" in schema
    assert "state = 'building'" in schema and "published_at is null" in schema
    assert "state = 'ready'" in schema and "lease_token is null" in schema
    assert "migration" not in schema


def test_registry_indexes_use_concurrent_safe_bootstrap() -> None:
    assert GENERATION_INDEX_DDL
    assert all("CREATE INDEX CONCURRENTLY" in sql for sql in GENERATION_INDEX_DDL)
    assert all("IF NOT EXISTS" not in sql for sql in GENERATION_INDEX_DDL)


def test_greenfield_schema_contains_no_implicit_repair_paths() -> None:
    schema = "\n".join(GENERATION_SCHEMA_DDL).upper()

    assert "IF NOT EXISTS" not in schema
    assert "DO $" not in schema


def test_postgres_registry_implements_storage_neutral_protocol() -> None:
    registry = PostgresGenerationRegistry(SimpleNamespace())  # type: ignore[arg-type]

    assert isinstance(registry, GenerationRegistry)


def test_registry_and_operation_fences_share_one_advisory_key_derivation() -> None:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")

    assert PostgresGenerationRegistry.advisory_key(
        "oceanstack_dev", generation_id
    ) == generation_advisory_key("oceanstack_dev", generation_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plane", "BadPlane"),
        ("workspace", "kg_wrong_018f0f7dc68b7a2f8f7d724a24f9aa01"),
        ("state", "readyish"),
        ("build_id", ""),
        ("started_at", None),
        ("started_at", datetime.now()),
        ("published_at", datetime.now(timezone.utc)),
        ("storage_flushed", 1),
    ],
)
def test_generation_decoder_rejects_corrupt_building_rows(
    field: str, value: object
) -> None:
    row = _building_row()
    row[field] = value

    with pytest.raises(GenerationValidationError):
        _generation_from_row(row)


def test_generation_decoder_rejects_partial_or_backwards_lease() -> None:
    row = _building_row()
    row["worker_id"] = "worker-a"
    row["lease_token"] = uuid.uuid4()
    row["lease_heartbeat"] = datetime.now(timezone.utc)

    with pytest.raises(GenerationValidationError, match="lease fields"):
        _generation_from_row(row)

    row["lease_expires"] = row["lease_heartbeat"] - timedelta(seconds=1)  # type: ignore[operator]
    with pytest.raises(GenerationValidationError, match="lease_expires"):
        _generation_from_row(row)


def test_generation_decoder_rejects_backwards_state_timestamp() -> None:
    row = _building_row()
    row.update(
        state="ready",
        counts='{"chunks": 1}',
        ready_at=row["started_at"] - timedelta(seconds=1),  # type: ignore[operator]
        storage_flushed=True,
        gates_passed=True,
    )

    with pytest.raises(GenerationValidationError, match="ready_at"):
        _generation_from_row(row)


@pytest.mark.parametrize("revision", [True, -1])
def test_plane_decoder_rejects_noncanonical_revision(revision: object) -> None:
    now = datetime.now(timezone.utc)

    with pytest.raises(GenerationValidationError, match="revision"):
        _plane_from_row(
            {
                "plane": "oceanstack_dev",
                "active_generation_id": None,
                "revision": revision,
                "created_at": now,
                "updated_at": now,
            }
        )


@pytest.mark.parametrize("counts", [{"chunks": True}, {"chunks": -1}])
def test_generation_decoder_rejects_invalid_ready_counts(
    counts: dict[str, object],
) -> None:
    row = _building_row()
    row.update(
        state="ready",
        counts=json.dumps(counts),
        ready_at=datetime.now(timezone.utc),
        storage_flushed=True,
        gates_passed=True,
    )

    with pytest.raises(GenerationValidationError, match="ready generation"):
        _generation_from_row(row)
