"""Greenfield immutable graph-generation contracts and lifecycle storage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import uuid
from contextvars import ContextVar, Token
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, TypeAlias, runtime_checkable


_PLANE_RE = re.compile(r"^[a-z][a-z0-9_]{0,19}$")
_WORKSPACE_RE = re.compile(r"^kg_[a-z][a-z0-9_]{0,19}_[0-9a-f]{32}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_VALUE_RE = re.compile(r"(?:^|[^a-z])(unknown|placeholder)(?:$|[^a-z])", re.I)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]


class GenerationValidationError(ValueError):
    """Generation contract data is not canonical or safe to publish."""


class GenerationCleanupError(RuntimeError):
    """A failed generation still owns one or more physical storage records."""


class GenerationFenceError(RuntimeError):
    """A generation storage operation lacks current durable authorization."""


class GenerationState(str, Enum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class GenerationFenceKind(str, Enum):
    BUILD = "build"
    CLEANUP = "cleanup"


class GenerationStorageAccess(str, Enum):
    READ = "read"
    WRITE = "write"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class GenerationOperationFence:
    """Durable token identity bound to one generation storage operation context."""

    kind: GenerationFenceKind
    plane: str
    generation_id: uuid.UUID
    workspace: str
    token: uuid.UUID
    advisory_key: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GenerationFenceKind):
            raise GenerationValidationError("fence kind must be GenerationFenceKind")
        validate_plane(self.plane)
        if not isinstance(self.generation_id, uuid.UUID):
            raise GenerationValidationError("generation_id must be a UUID")
        expected_workspace = generation_workspace(self.plane, self.generation_id)
        if self.workspace != expected_workspace:
            raise GenerationValidationError(
                "fence workspace does not match generation identity"
            )
        if not isinstance(self.token, uuid.UUID):
            raise GenerationValidationError("fence token must be a UUID")
        object.__setattr__(
            self,
            "advisory_key",
            generation_advisory_key(self.plane, self.generation_id),
        )


_GENERATION_OPERATION_FENCE: ContextVar[GenerationOperationFence | None] = ContextVar(
    "generation_operation_fence", default=None
)


def current_generation_operation_fence() -> GenerationOperationFence | None:
    return _GENERATION_OPERATION_FENCE.get()


def bind_generation_operation_fence(
    fence: GenerationOperationFence,
) -> Token[GenerationOperationFence | None]:
    return _GENERATION_OPERATION_FENCE.set(fence)


def reset_generation_operation_fence(
    token: Token[GenerationOperationFence | None],
) -> None:
    _GENERATION_OPERATION_FENCE.reset(token)


async def complete_cleanup_preserving_cancellation(
    cleanup: Awaitable[None],
    *,
    initial_cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Finish cleanup, then re-raise the first cancellation without losing failures."""
    task = asyncio.ensure_future(cleanup)
    cancellation = initial_cancellation
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    try:
        task.result()
    except BaseException as cleanup_error:
        if cancellation is not None:
            raise cancellation from cleanup_error
        raise
    if cancellation is not None:
        raise cancellation


def validate_plane(plane: str) -> str:
    if not isinstance(plane, str) or _PLANE_RE.fullmatch(plane) is None:
        raise GenerationValidationError("plane must match [a-z][a-z0-9_]{0,19}")
    return plane


def generation_workspace(plane: str, generation_id: uuid.UUID) -> str:
    """Return the deterministic PostgreSQL-safe physical workspace name."""
    validate_plane(plane)
    if not isinstance(generation_id, uuid.UUID):
        raise GenerationValidationError("generation_id must be a UUID")
    workspace = f"kg_{plane}_{generation_id.hex}"
    if len(workspace.encode("utf-8")) > 63:
        raise GenerationValidationError("physical workspace exceeds 63 bytes")
    return workspace


def generation_advisory_key(plane: str, generation_id: uuid.UUID) -> int:
    """Return the deterministic signed bigint lock key for one generation."""
    validate_plane(plane)
    if not isinstance(generation_id, uuid.UUID):
        raise GenerationValidationError("generation_id must be a UUID")
    digest = hashlib.blake2b(
        f"lightrag-generation:{plane}:{generation_id.hex}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big", signed=True)


def is_generation_workspace(workspace: str) -> bool:
    """Return whether a workspace has the canonical immutable-generation shape."""
    return isinstance(workspace, str) and _WORKSPACE_RE.fullmatch(workspace) is not None


def _freeze_json(value: Any, *, path: str) -> JsonValue:
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) or not key for key in keys):
            raise GenerationValidationError(f"{path} keys must be nonempty strings")
        frozen: dict[str, JsonValue] = {}
        for key in sorted(keys):
            if _FORBIDDEN_VALUE_RE.search(key):
                raise GenerationValidationError(
                    f"{path} contains forbidden placeholder data"
                )
            frozen[key] = _freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GenerationValidationError(f"{path} must contain finite numbers")
        return value
    if isinstance(value, str):
        if _FORBIDDEN_VALUE_RE.search(value):
            raise GenerationValidationError(
                f"{path} contains forbidden placeholder data"
            )
        return value
    raise GenerationValidationError(f"{path} is not canonical JSON")


def canonical_json_object(value: Mapping[str, Any], *, name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise GenerationValidationError(f"{name} must be a JSON object")
    frozen = _freeze_json(value, path=name)
    assert isinstance(frozen, Mapping)
    return frozen


def mutable_json(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: mutable_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [mutable_json(item) for item in value]
    return value


def canonical_json_text(value: JsonObject) -> str:
    return json.dumps(
        mutable_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_digest(value: JsonObject) -> str:
    """Return the SHA-256 digest of a canonical JSON object."""
    return hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise GenerationValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class GenerationCandidate:
    """Validated immutable input for one physical graph generation."""

    plane: str
    generation_id: uuid.UUID
    build_id: str
    contract_digest: str
    manifest_digest: str
    manifest: JsonObject
    metadata: JsonObject = field(default_factory=dict)
    workspace: str = field(init=False)
    state: GenerationState = field(init=False, default=GenerationState.BUILDING)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane", validate_plane(self.plane))
        if not isinstance(self.generation_id, uuid.UUID):
            raise GenerationValidationError("generation_id must be a UUID")
        if not isinstance(self.build_id, str) or not self.build_id.strip():
            raise GenerationValidationError("build_id must be nonempty")
        if _FORBIDDEN_VALUE_RE.search(self.build_id):
            raise GenerationValidationError(
                "build_id contains forbidden placeholder data"
            )
        object.__setattr__(
            self,
            "contract_digest",
            _validate_digest(self.contract_digest, name="contract_digest"),
        )
        object.__setattr__(
            self,
            "manifest_digest",
            _validate_digest(self.manifest_digest, name="manifest_digest"),
        )
        manifest = canonical_json_object(self.manifest, name="manifest")
        if not manifest:
            raise GenerationValidationError("manifest must be nonempty")
        if self.manifest_digest != canonical_json_digest(manifest):
            raise GenerationValidationError(
                "manifest_digest does not match canonical manifest JSON"
            )
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(
            self,
            "metadata",
            canonical_json_object(self.metadata, name="metadata"),
        )
        object.__setattr__(
            self,
            "workspace",
            generation_workspace(self.plane, self.generation_id),
        )


@dataclass(frozen=True, slots=True)
class StorageDropResult:
    """One inspected workspace-storage drop result."""

    name: str
    status: Literal["success", "error"]
    message: str


@dataclass(frozen=True, slots=True)
class WorkspaceDropReport:
    """Complete result of dropping every distinct storage for one workspace."""

    workspace: str
    results: tuple[StorageDropResult, ...]

    @property
    def success(self) -> bool:
        return bool(self.results) and all(
            result.status == "success" for result in self.results
        )


@dataclass(frozen=True, slots=True)
class GraphGeneration:
    """Immutable registry view of one physical graph generation."""

    plane: str
    generation_id: uuid.UUID
    workspace: str
    state: GenerationState
    build_id: str
    contract_digest: str
    manifest_digest: str
    manifest: JsonObject
    metadata: JsonObject
    counts: JsonObject
    worker_id: str | None
    lease_token: uuid.UUID | None
    lease_heartbeat: datetime | None
    lease_expires: datetime | None
    started_at: datetime
    ready_at: datetime | None
    published_at: datetime | None
    failed_at: datetime | None
    storage_flushed: bool
    gates_passed: bool
    failure: JsonObject | None
    cleanup_failure: JsonObject | None

    def __post_init__(self) -> None:
        validate_plane(self.plane)
        if not isinstance(self.generation_id, uuid.UUID):
            raise GenerationValidationError("generation_id must be a UUID")
        if self.workspace != generation_workspace(self.plane, self.generation_id):
            raise GenerationValidationError(
                "workspace does not match plane and generation_id"
            )
        if not isinstance(self.build_id, str) or not self.build_id.strip():
            raise GenerationValidationError("build_id must be nonempty")
        if _FORBIDDEN_VALUE_RE.search(self.build_id):
            raise GenerationValidationError(
                "build_id contains forbidden placeholder data"
            )
        _validate_digest(self.contract_digest, name="contract_digest")
        _validate_digest(self.manifest_digest, name="manifest_digest")
        manifest = canonical_json_object(self.manifest, name="manifest")
        if not manifest:
            raise GenerationValidationError("manifest must be nonempty")
        if self.manifest_digest != canonical_json_digest(manifest):
            raise GenerationValidationError(
                "manifest_digest does not match canonical manifest JSON"
            )
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(
            self, "metadata", canonical_json_object(self.metadata, name="metadata")
        )
        object.__setattr__(
            self, "counts", canonical_json_object(self.counts, name="counts")
        )
        if not isinstance(self.storage_flushed, bool) or not isinstance(
            self.gates_passed, bool
        ):
            raise GenerationValidationError(
                "storage_flushed and gates_passed must be booleans"
            )
        if self.failure is not None:
            object.__setattr__(
                self,
                "failure",
                canonical_json_object(self.failure, name="failure"),
            )
        if self.cleanup_failure is not None:
            object.__setattr__(
                self,
                "cleanup_failure",
                canonical_json_object(self.cleanup_failure, name="cleanup_failure"),
            )
        if (
            not isinstance(self.started_at, datetime)
            or self.started_at.utcoffset() is None
        ):
            raise GenerationValidationError(
                "started_at must be a timezone-aware datetime"
            )
        for name in (
            "ready_at",
            "published_at",
            "failed_at",
            "lease_heartbeat",
            "lease_expires",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, datetime) or value.utcoffset() is None
            ):
                raise GenerationValidationError(
                    f"{name} must be a timezone-aware datetime"
                )
        if self.ready_at is not None and self.ready_at < self.started_at:
            raise GenerationValidationError("ready_at cannot precede started_at")
        if self.published_at is not None:
            if self.ready_at is None:
                raise GenerationValidationError("published_at requires ready_at")
            if self.published_at < self.ready_at:
                raise GenerationValidationError("published_at cannot precede ready_at")
        if self.failed_at is not None and self.failed_at < self.started_at:
            raise GenerationValidationError("failed_at cannot precede started_at")
        lease_values = (
            self.worker_id,
            self.lease_token,
            self.lease_heartbeat,
            self.lease_expires,
        )
        lease_present = [value is not None for value in lease_values]
        if any(lease_present) and not all(lease_present):
            raise GenerationValidationError("lease fields must be all set or all null")
        if all(lease_present):
            if not isinstance(self.lease_token, uuid.UUID):
                raise GenerationValidationError("lease_token must be a UUID")
            if not isinstance(self.worker_id, str) or not self.worker_id.strip():
                raise GenerationValidationError("worker_id must be nonempty")
            assert self.lease_heartbeat is not None
            assert self.lease_expires is not None
            if self.lease_heartbeat < self.started_at:
                raise GenerationValidationError(
                    "lease_heartbeat cannot precede started_at"
                )
            if self.lease_expires <= self.lease_heartbeat:
                raise GenerationValidationError(
                    "lease_expires must follow lease_heartbeat"
                )
        if self.state is GenerationState.BUILDING:
            if (
                self.ready_at is not None
                or self.published_at is not None
                or self.failed_at is not None
                or self.failure is not None
                or self.storage_flushed
                or self.gates_passed
            ):
                raise GenerationValidationError(
                    "building generation state is inconsistent"
                )
        elif self.state is GenerationState.READY:
            if (
                self.ready_at is None
                or self.failed_at is not None
                or self.failure is not None
                or not self.storage_flushed
                or not self.gates_passed
                or not self.counts
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in self.counts.values()
                )
                or any(lease_present)
            ):
                raise GenerationValidationError(
                    "ready generation state is inconsistent"
                )
        elif self.state is GenerationState.FAILED:
            if self.failed_at is None or not self.failure or any(lease_present):
                raise GenerationValidationError(
                    "failed generation state is inconsistent"
                )
        else:
            raise GenerationValidationError("generation state is invalid")


@dataclass(frozen=True, slots=True)
class GraphPlane:
    """Immutable active-pointer view for a logical graph plane."""

    plane: str
    active_generation_id: uuid.UUID | None
    revision: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        validate_plane(self.plane)
        if self.active_generation_id is not None and not isinstance(
            self.active_generation_id, uuid.UUID
        ):
            raise GenerationValidationError("active_generation_id must be a UUID")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise GenerationValidationError("plane revision must be nonnegative")
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise GenerationValidationError(
                    f"{name} must be a timezone-aware datetime"
                )
        if self.updated_at < self.created_at:
            raise GenerationValidationError("updated_at cannot precede created_at")


@dataclass(frozen=True, slots=True)
class GenerationPublishResult:
    """Atomic publication result plus the explicit cleanup target, if any."""

    active: GraphGeneration
    plane: GraphPlane
    superseded: GraphGeneration | None


@runtime_checkable
class GenerationBuildLease(Protocol):
    """Storage-neutral held lease used by graph build workers."""

    token: uuid.UUID

    async def __aenter__(self) -> "GenerationBuildLease": ...

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...

    async def heartbeat(self, *, ttl: timedelta | None = None) -> GraphGeneration: ...

    async def mark_ready(
        self,
        *,
        counts: Mapping[str, Any],
        storage_flushed: bool,
        gates_passed: bool,
    ) -> GraphGeneration: ...

    async def mark_failed(self, failure: Mapping[str, Any]) -> bool: ...


@runtime_checkable
class GenerationRegistry(Protocol):
    """Storage-neutral lifecycle contract consumed by builders and API routing."""

    async def create_candidate(
        self, candidate: GenerationCandidate
    ) -> GraphGeneration: ...

    def acquire_build_lease(
        self,
        plane: str,
        generation_id: uuid.UUID,
        *,
        worker_id: str,
        ttl: timedelta,
    ) -> GenerationBuildLease: ...

    async def fail_stale(self) -> list[GraphGeneration]: ...

    async def publish(
        self,
        plane: str,
        generation_id: uuid.UUID,
        *,
        expected_active_generation_id: uuid.UUID | None,
    ) -> GenerationPublishResult: ...

    async def resolve_active(self, plane: str) -> GraphGeneration | None: ...

    async def get_plane(self, plane: str) -> GraphPlane | None: ...

    async def list_planes(self) -> list[GraphPlane]: ...

    async def get_generation(
        self, plane: str, generation_id: uuid.UUID
    ) -> GraphGeneration | None: ...

    async def list_generations(self, plane: str) -> list[GraphGeneration]: ...

    def acquire_failed_cleanup(
        self,
        plane: str,
        generation_id: uuid.UUID,
        *,
        ttl: timedelta | None = None,
    ) -> "FailedGenerationCleanupClaim": ...


@runtime_checkable
class FailedGenerationCleanupClaim(Protocol):
    """Fenced authorization for cleaning one inactive failed generation."""

    @property
    def workspace(self) -> str: ...

    async def __aenter__(self) -> "FailedGenerationCleanupClaim": ...

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...

    async def record_failure(self, failure: Mapping[str, Any]) -> None: ...

    async def delete(self) -> bool: ...


class FailedGenerationRegistry(Protocol):
    """Storage-neutral registry boundary used by explicit failed cleanup."""

    def acquire_failed_cleanup(
        self,
        plane: str,
        generation_id: uuid.UUID,
        *,
        ttl: timedelta | None = None,
    ) -> FailedGenerationCleanupClaim: ...


WorkspaceDropCallback: TypeAlias = Callable[[str], Awaitable[WorkspaceDropReport]]


class FailedGenerationCleanup:
    """Retryable registry-last cleanup for one explicitly failed generation."""

    def __init__(
        self,
        registry: FailedGenerationRegistry,
        drop_workspace: WorkspaceDropCallback,
    ) -> None:
        self._registry = registry
        self._drop_workspace = drop_workspace

    async def cleanup(
        self,
        plane: str,
        generation_id: uuid.UUID,
    ) -> WorkspaceDropReport:
        validate_plane(plane)
        claim = self._registry.acquire_failed_cleanup(plane, generation_id)
        async with claim:
            workspace = claim.workspace
            try:
                report = await self._drop_workspace(workspace)
            except Exception as exc:
                failure = {
                    "code": "storage_cleanup_callback_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                await claim.record_failure(failure)
                raise GenerationCleanupError(
                    f"workspace cleanup callback failed: {exc}"
                ) from exc
            if report.workspace != workspace:
                await claim.record_failure(
                    {
                        "code": "storage_cleanup_workspace_mismatch",
                        "expected_workspace": workspace,
                        "reported_workspace": report.workspace,
                    }
                )
                raise GenerationCleanupError(
                    f"workspace cleanup report workspace mismatch: "
                    f"{report.workspace!r} != {workspace!r}"
                )
            failures = [
                result for result in report.results if result.status != "success"
            ]
            if failures or not report.success:
                await claim.record_failure(
                    {
                        "code": "storage_cleanup_failed",
                        "storages": [
                            {
                                "name": result.name,
                                "status": result.status,
                                "message": result.message,
                            }
                            for result in report.results
                        ],
                    }
                )
                failed_names = ", ".join(result.name for result in failures)
                raise GenerationCleanupError(
                    f"workspace cleanup failed for storage(s): {failed_names}"
                )
            deleted = await claim.delete()
            if not deleted:
                raise GenerationCleanupError(
                    "failed generation was not deleted after storage cleanup"
                )
            return report
        raise GenerationCleanupError("cleanup claim exited without a result")
