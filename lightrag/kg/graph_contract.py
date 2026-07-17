"""Validated, storage-independent input contract for knowledge-graph builds.

``source_key`` values are canonical POSIX paths beneath an explicit repository
namespace. A repository-root file therefore uses ``oceanstack/README.md`` rather
than the ambiguous basename ``README.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = (
    JSONScalar | list["JSONValue"] | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]
)
JSONMetadata: TypeAlias = Mapping[str, JSONValue]


def _validate_token(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL characters")
    if value.casefold() == "unknown":
        raise ValueError(f"{field_name} must not be UNKNOWN")
    return value


def _validate_source_key(value: object) -> str:
    source_key = _validate_token(value, "source_key")
    if "\\" in source_key:
        raise ValueError("source_key must use repository-relative POSIX syntax")
    path = PurePosixPath(source_key)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or ".." in path.parts
        or str(path) != source_key
    ):
        raise ValueError(
            "source_key must include a repository namespace and normalized "
            "relative path, for example 'oceanstack/README.md'"
        )
    return source_key


def _freeze_json(value: object, path: str) -> JSONValue:
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(f"{path} contains NUL characters")
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            if "\x00" in key:
                raise ValueError(f"{path} contains a key with NUL characters")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _freeze_metadata(value: object) -> JSONMetadata:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON object")
    frozen = _freeze_json(value, "metadata")
    if not isinstance(frozen, Mapping):
        raise ValueError("metadata must be a JSON object")
    return frozen


def _validate_intervals(record: object) -> None:
    for field_name in ("observed_from", "observed_to", "valid_from", "valid_to"):
        value = getattr(record, field_name)
        if value is None:
            continue
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(f"{field_name} must be a timezone-aware datetime")
    for prefix in ("observed", "valid"):
        start = getattr(record, f"{prefix}_from")
        end = getattr(record, f"{prefix}_to")
        if start is not None and end is not None and start > end:
            raise ValueError(f"{prefix}_from must not be after {prefix}_to")


def _prepare_evidence(record: object) -> None:
    evidence = tuple(getattr(record, "evidence"))
    if not evidence:
        raise ValueError("evidence must contain at least one chunk reference")
    if not all(isinstance(item, EvidenceRef) for item in evidence):
        raise ValueError("evidence must contain only EvidenceRef records")
    object.__setattr__(record, "evidence", evidence)
    object.__setattr__(
        record, "metadata", _freeze_metadata(getattr(record, "metadata"))
    )


@dataclass(frozen=True)
class EvidenceRef:
    """Link a graph record to a chunk using namespaced repository identity."""

    chunk_id: str
    source_key: str
    source_revision: str
    metadata: JSONMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.chunk_id, "chunk_id")
        _validate_source_key(self.source_key)
        _validate_token(self.source_revision, "source_revision")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class GraphChunk:
    """Source text identified by a namespaced repository-relative key."""

    build_id: str
    chunk_id: str
    source_key: str
    source_revision: str
    content: str
    metadata: JSONMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.build_id, "build_id")
        _validate_token(self.chunk_id, "chunk_id")
        _validate_source_key(self.source_key)
        _validate_token(self.source_revision, "source_revision")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-blank string")
        if "\x00" in self.content:
            raise ValueError("content must not contain NUL characters")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class GraphEntity:
    """Typed graph entity supported by source evidence."""

    build_id: str
    entity_id: str
    entity_type: str
    evidence: tuple[EvidenceRef, ...]
    metadata: JSONMetadata = field(default_factory=dict)
    observed_from: datetime | None = None
    observed_to: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        _validate_token(self.build_id, "build_id")
        _validate_token(self.entity_id, "entity_id")
        _validate_token(self.entity_type, "entity_type")
        _prepare_evidence(self)
        _validate_intervals(self)


@dataclass(frozen=True)
class GraphAssertion:
    """Directed, typed assertion between two graph entities."""

    build_id: str
    assertion_id: str
    predicate: str
    src_id: str
    dst_id: str
    evidence: tuple[EvidenceRef, ...]
    metadata: JSONMetadata = field(default_factory=dict)
    confidence: float | None = None
    method: str | None = None
    observed_from: datetime | None = None
    observed_to: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    def __post_init__(self) -> None:
        _validate_token(self.build_id, "build_id")
        _validate_token(self.assertion_id, "assertion_id")
        _validate_token(self.predicate, "predicate")
        _validate_token(self.src_id, "src_id")
        _validate_token(self.dst_id, "dst_id")
        _prepare_evidence(self)
        if self.confidence is not None:
            if (
                isinstance(self.confidence, bool)
                or not isinstance(self.confidence, (int, float))
                or not math.isfinite(self.confidence)
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise ValueError("confidence must be a finite number between 0 and 1")
            object.__setattr__(self, "confidence", float(self.confidence))
        if self.method is not None:
            _validate_token(self.method, "method")
        _validate_intervals(self)


GraphRecord: TypeAlias = EvidenceRef | GraphChunk | GraphEntity | GraphAssertion


def _canonicalize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Mapping):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _record_dict(record: GraphRecord) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(record):
        value = getattr(record, item.name)
        if item.name == "evidence":
            result[item.name] = [_record_dict(evidence) for evidence in value]
        else:
            result[item.name] = _canonicalize(value)
    if "evidence" in result:
        result["evidence"] = sorted(
            result["evidence"],
            key=lambda item: (
                item["chunk_id"],
                item["source_key"],
                item["source_revision"],
                _canonical_json(item["metadata"]),
            ),
        )
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class KnowledgeGraphBuild:
    """Complete validated graph build with a deterministic content digest."""

    build_id: str
    contract_digest: str
    chunks: tuple[GraphChunk, ...]
    entities: tuple[GraphEntity, ...]
    assertions: tuple[GraphAssertion, ...]
    metadata: JSONMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.build_id, "build_id")
        chunks, entities, assertions, metadata = self._normalize_build_inputs(
            self.chunks,
            self.entities,
            self.assertions,
            self.metadata,
        )
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "assertions", assertions)
        object.__setattr__(self, "metadata", metadata)

        self._validate_build_records(
            build_id=self.build_id,
            chunks=chunks,
            entities=entities,
            assertions=assertions,
        )

        if (
            not isinstance(self.contract_digest, str)
            or len(self.contract_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.contract_digest
            )
        ):
            raise ValueError("contract_digest must be a lowercase SHA-256 digest")
        expected_digest = self._digest_payload(
            self._canonical_payload(
                build_id=self.build_id,
                chunks=self.chunks,
                entities=self.entities,
                assertions=self.assertions,
                metadata=self.metadata,
            )
        )
        if self.contract_digest != expected_digest:
            raise ValueError("contract_digest does not match the canonical build")

    @classmethod
    def create(
        cls,
        *,
        build_id: str,
        chunks: Iterable[GraphChunk],
        entities: Iterable[GraphEntity],
        assertions: Iterable[GraphAssertion],
        metadata: JSONMetadata | None = None,
    ) -> "KnowledgeGraphBuild":
        """Create a build and calculate its canonical contract digest."""
        _validate_token(build_id, "build_id")
        chunk_records, entity_records, assertion_records, build_metadata = (
            cls._normalize_build_inputs(
                chunks,
                entities,
                assertions,
                {} if metadata is None else metadata,
            )
        )
        cls._validate_build_records(
            build_id=build_id,
            chunks=chunk_records,
            entities=entity_records,
            assertions=assertion_records,
        )
        payload = cls._canonical_payload(
            build_id=build_id,
            chunks=chunk_records,
            entities=entity_records,
            assertions=assertion_records,
            metadata=build_metadata,
        )
        digest = cls._digest_payload(payload)
        return cls._from_validated(
            build_id,
            digest,
            chunk_records,
            entity_records,
            assertion_records,
            build_metadata,
        )

    @staticmethod
    def _normalize_build_inputs(
        chunks: Iterable[GraphChunk],
        entities: Iterable[GraphEntity],
        assertions: Iterable[GraphAssertion],
        metadata: object,
    ) -> tuple[
        tuple[GraphChunk, ...],
        tuple[GraphEntity, ...],
        tuple[GraphAssertion, ...],
        JSONMetadata,
    ]:
        return (
            tuple(chunks),
            tuple(entities),
            tuple(assertions),
            _freeze_metadata(metadata),
        )

    @classmethod
    def _from_validated(
        cls,
        build_id: str,
        contract_digest: str,
        chunks: tuple[GraphChunk, ...],
        entities: tuple[GraphEntity, ...],
        assertions: tuple[GraphAssertion, ...],
        metadata: JSONMetadata,
    ) -> "KnowledgeGraphBuild":
        """Construct a frozen instance after shared validation and digesting."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "build_id", build_id)
        object.__setattr__(instance, "contract_digest", contract_digest)
        object.__setattr__(instance, "chunks", chunks)
        object.__setattr__(instance, "entities", entities)
        object.__setattr__(instance, "assertions", assertions)
        object.__setattr__(instance, "metadata", metadata)
        return instance

    @staticmethod
    def _digest_payload(payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def _validate_build_records(
        cls,
        *,
        build_id: str,
        chunks: tuple[GraphChunk, ...],
        entities: tuple[GraphEntity, ...],
        assertions: tuple[GraphAssertion, ...],
    ) -> None:
        cls._validate_record_types(chunks, entities, assertions)
        cls._validate_unique_ids(chunks, entities, assertions)
        cls._validate_build_membership(build_id, chunks, entities, assertions)
        cls._validate_graph_links(chunks, entities, assertions)

    @staticmethod
    def _validate_record_types(
        chunks: tuple[GraphChunk, ...],
        entities: tuple[GraphEntity, ...],
        assertions: tuple[GraphAssertion, ...],
    ) -> None:
        expected = (
            (chunks, GraphChunk, "chunks"),
            (entities, GraphEntity, "entities"),
            (assertions, GraphAssertion, "assertions"),
        )
        for records, record_type, field_name in expected:
            if not all(isinstance(record, record_type) for record in records):
                raise ValueError(f"{field_name} contains an invalid record type")

    @staticmethod
    def _validate_unique_ids(
        chunks: tuple[GraphChunk, ...],
        entities: tuple[GraphEntity, ...],
        assertions: tuple[GraphAssertion, ...],
    ) -> None:
        identifiers = (
            ("chunk_id", (record.chunk_id for record in chunks)),
            ("entity_id", (record.entity_id for record in entities)),
            ("assertion_id", (record.assertion_id for record in assertions)),
        )
        for field_name, values in identifiers:
            duplicates = sorted(
                value for value, count in Counter(values).items() if count > 1
            )
            if duplicates:
                raise ValueError(f"duplicate {field_name} values: {duplicates!r}")

    @staticmethod
    def _validate_build_membership(
        build_id: str,
        chunks: tuple[GraphChunk, ...],
        entities: tuple[GraphEntity, ...],
        assertions: tuple[GraphAssertion, ...],
    ) -> None:
        for record in (*chunks, *entities, *assertions):
            if record.build_id != build_id:
                raise ValueError(
                    f"record build_id {record.build_id!r} does not match {build_id!r}"
                )

    @staticmethod
    def _validate_graph_links(
        chunks: tuple[GraphChunk, ...],
        entities: tuple[GraphEntity, ...],
        assertions: tuple[GraphAssertion, ...],
    ) -> None:
        chunks_by_id = {record.chunk_id: record for record in chunks}
        entity_ids = {record.entity_id for record in entities}
        for assertion in assertions:
            if assertion.src_id not in entity_ids or assertion.dst_id not in entity_ids:
                raise ValueError(
                    f"assertion {assertion.assertion_id!r} has a missing endpoint"
                )
        for record in (*entities, *assertions):
            for evidence in record.evidence:
                chunk = chunks_by_id.get(evidence.chunk_id)
                if chunk is None:
                    raise ValueError(
                        f"record evidence references missing chunk {evidence.chunk_id!r}"
                    )
                if (
                    evidence.source_key != chunk.source_key
                    or evidence.source_revision != chunk.source_revision
                ):
                    raise ValueError(
                        f"record evidence identity does not match chunk {chunk.chunk_id!r}"
                    )

    @staticmethod
    def _canonical_payload(
        *,
        build_id: str,
        chunks: tuple[GraphChunk, ...],
        entities: tuple[GraphEntity, ...],
        assertions: tuple[GraphAssertion, ...],
        metadata: JSONMetadata,
    ) -> dict[str, Any]:
        return {
            "assertions": [
                _record_dict(record)
                for record in sorted(assertions, key=lambda item: item.assertion_id)
            ],
            "build_id": build_id,
            "chunks": [
                _record_dict(record)
                for record in sorted(chunks, key=lambda item: item.chunk_id)
            ],
            "entities": [
                _record_dict(record)
                for record in sorted(entities, key=lambda item: item.entity_id)
            ],
            "metadata": _canonicalize(dict(metadata)),
        }

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible build representation."""
        payload = self._canonical_payload(
            build_id=self.build_id,
            chunks=self.chunks,
            entities=self.entities,
            assertions=self.assertions,
            metadata=self.metadata,
        )
        return {"contract_digest": self.contract_digest, **payload}

    def to_canonical_json(self) -> str:
        """Serialize the build to deterministic canonical JSON."""
        return _canonical_json(self.to_canonical_dict())
