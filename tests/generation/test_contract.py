from __future__ import annotations

import hashlib
import uuid

import pytest

from lightrag.generation import (
    GenerationCandidate,
    GenerationFenceKind,
    GenerationOperationFence,
    GenerationState,
    GenerationValidationError,
    canonical_json_object,
    canonical_json_text,
    generation_advisory_key,
    generation_workspace,
)


def _candidate(**overrides: object) -> GenerationCandidate:
    manifest = {"sources": ["src/oceanstack/core.py"]}
    values: dict[str, object] = {
        "plane": "oceanstack_dev",
        "generation_id": uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01"),
        "build_id": "build-dev-001",
        "contract_digest": "a" * 64,
        "manifest_digest": hashlib.sha256(
            canonical_json_text(
                canonical_json_object(manifest, name="manifest")
            ).encode()
        ).hexdigest(),
        "manifest": manifest,
        "metadata": {"source_revision": "abc123"},
    }
    values.update(overrides)
    return GenerationCandidate(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "plane",
    ["", "OceanStack", "1dev", "dev-plane", "a" * 21, "dev/plane"],
)
def test_plane_identifier_rejects_noncanonical_values(plane: str) -> None:
    with pytest.raises(GenerationValidationError, match="plane"):
        _candidate(plane=plane)


def test_generation_workspace_is_deterministic_and_postgres_safe() -> None:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")

    first = generation_workspace("oceanstack_dev", generation_id)
    second = generation_workspace("oceanstack_dev", generation_id)
    other = generation_workspace(
        "oceanstack_dev", uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa02")
    )

    assert first == "kg_oceanstack_dev_018f0f7dc68b7a2f8f7d724a24f9aa01"
    assert first == second
    assert first != other
    assert len(first.encode()) <= 63


def test_candidate_is_immutable_and_computes_physical_workspace() -> None:
    candidate = _candidate()

    assert candidate.state is GenerationState.BUILDING
    assert candidate.workspace == generation_workspace(
        candidate.plane, candidate.generation_id
    )
    with pytest.raises((AttributeError, TypeError)):
        candidate.plane = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_id", "not-a-uuid"),
        ("contract_digest", "short"),
        ("manifest_digest", "z" * 64),
        ("manifest", {}),
        ("manifest", {"value": "UNKNOWN"}),
        ("metadata", {"value": "placeholder"}),
        ("metadata", {"bad": float("nan")}),
        ("manifest", {"valid": 1, 1: "value"}),
    ],
)
def test_candidate_rejects_invalid_or_placeholder_contract_data(
    field: str, value: object
) -> None:
    with pytest.raises(GenerationValidationError):
        _candidate(**{field: value})


def test_candidate_rejects_manifest_digest_that_does_not_match_canonical_json() -> None:
    with pytest.raises(
        GenerationValidationError, match="manifest_digest does not match"
    ):
        _candidate(manifest_digest="c" * 64)


def test_generation_states_are_exact() -> None:
    assert {state.value for state in GenerationState} == {
        "building",
        "ready",
        "failed",
    }


def test_generation_fence_derives_the_shared_advisory_key() -> None:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    fence = GenerationOperationFence(
        kind=GenerationFenceKind.BUILD,
        plane="oceanstack_dev",
        generation_id=generation_id,
        workspace=generation_workspace("oceanstack_dev", generation_id),
        token=uuid.UUID("118f0f7d-c68b-7a2f-8f7d-724a24f9aa01"),
    )

    assert fence.advisory_key == generation_advisory_key(
        "oceanstack_dev", generation_id
    )


def test_generation_fence_rejects_caller_selected_advisory_key() -> None:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    with pytest.raises(TypeError, match="advisory_key"):
        GenerationOperationFence(
            kind=GenerationFenceKind.BUILD,
            plane="oceanstack_dev",
            generation_id=generation_id,
            workspace=generation_workspace("oceanstack_dev", generation_id),
            token=uuid.uuid4(),
            advisory_key=1,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "build"),
        ("plane", "OceanStack"),
        ("generation_id", "not-a-uuid"),
        ("workspace", "kg_oceanstack_dev_wrong"),
        ("token", "not-a-uuid"),
    ],
)
def test_generation_fence_rejects_invalid_identity(field: str, value: object) -> None:
    generation_id = uuid.UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    values: dict[str, object] = {
        "kind": GenerationFenceKind.CLEANUP,
        "plane": "oceanstack_dev",
        "generation_id": generation_id,
        "workspace": generation_workspace("oceanstack_dev", generation_id),
        "token": uuid.uuid4(),
    }
    values[field] = value

    with pytest.raises(GenerationValidationError):
        GenerationOperationFence(**values)  # type: ignore[arg-type]
