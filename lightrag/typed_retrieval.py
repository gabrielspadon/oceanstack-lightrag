"""Fail-closed structured retrieval for immutable typed graph generations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal, Protocol


QueryMode = Literal["local", "global", "hybrid", "mix"]

# Default deployment policy for which predicates carry a ``jurisdiction``
# claim. Callers with a domain ontology pass their own set; the default keeps
# the OceanStack service behavior without hardcoding it at the claim site.
DEFAULT_JURISDICTION_PREDICATES = frozenset({"located_in", "overlaps_zone"})


class TypedRetrievalContractError(ValueError):
    """Stored or candidate data violated the typed retrieval contract."""


@dataclass(frozen=True, slots=True)
class TypedRetrievalIdentity:
    """Exact active-generation identity expected from every retrieved sidecar."""

    build_id: str
    contract_digest: str
    manifest_digest: str
    source_revision: str

    def __post_init__(self) -> None:
        _required_token(self.build_id, "build_id")
        _validate_digest(self.contract_digest, "contract_digest")
        _validate_digest(self.manifest_digest, "manifest_digest")
        _required_token(self.source_revision, "source_revision")

    @classmethod
    def from_generation(cls, generation: Any) -> TypedRetrievalIdentity:
        """Validate one registry generation and return its retrieval identity."""
        manifest = getattr(generation, "manifest", None)
        metadata = getattr(generation, "metadata", None)
        if not isinstance(manifest, Mapping) or not isinstance(metadata, Mapping):
            raise TypedRetrievalContractError(
                "active generation lacks manifest provenance"
            )
        build_id = _required_token(getattr(generation, "build_id", None), "build_id")
        manifest_digest = _validate_digest(
            getattr(generation, "manifest_digest", None), "manifest_digest"
        )
        source_revision = _required_token(
            metadata.get("source_revision"), "source_revision"
        )
        expected_manifest = {
            "build_id": build_id,
            "digest": manifest_digest,
            "plane": _required_token(getattr(generation, "plane", None), "plane"),
            "source_revision": source_revision,
        }
        for field, expected in expected_manifest.items():
            if manifest.get(field) != expected:
                raise TypedRetrievalContractError(
                    f"active generation manifest {field} does not match its registry identity"
                )
        return cls(
            build_id=build_id,
            contract_digest=_validate_digest(
                getattr(generation, "contract_digest", None), "contract_digest"
            ),
            manifest_digest=manifest_digest,
            source_revision=source_revision,
        )


_TYPED_RETRIEVAL_IDENTITY: ContextVar[TypedRetrievalIdentity | None] = ContextVar(
    "typed_retrieval_identity", default=None
)


def bind_typed_retrieval_identity(
    identity: TypedRetrievalIdentity,
) -> Token[TypedRetrievalIdentity | None]:
    """Bind exact generation identity to the current retrieval operation."""
    if not isinstance(identity, TypedRetrievalIdentity):
        raise TypeError("identity must be TypedRetrievalIdentity")
    return _TYPED_RETRIEVAL_IDENTITY.set(identity)


def reset_typed_retrieval_identity(
    token: Token[TypedRetrievalIdentity | None],
) -> None:
    """Restore the prior retrieval identity context."""
    _TYPED_RETRIEVAL_IDENTITY.reset(token)


class _Graph(Protocol):
    async def get_graph_entity(self, entity_id: str) -> dict[str, Any] | None: ...

    async def get_graph_assertions(
        self, assertion_ids: list[str]
    ) -> dict[str, dict[str, Any]]: ...

    async def get_graph_assertions_for_entities(
        self, entity_ids: list[str]
    ) -> list[dict[str, Any]]: ...


class _VectorStorage(Protocol):
    async def query(
        self, query: str, top_k: int, query_embedding: Any = None
    ) -> list[dict[str, Any]]: ...


class _ChunkStorage(Protocol):
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class TypedRetrievalResult:
    entities: list[dict[str, Any]]
    assertions: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    claims: list[dict[str, Any]]

    def data(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "entities": self.entities,
            "assertions": self.assertions,
            "chunks": self.chunks,
            "citations": self.citations,
            "claims": self.claims,
        }


def _required_token(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypedRetrievalContractError(f"{field} must be a non-blank string")
    if value in {"UNKNOWN", "unknown_source"}:
        raise TypedRetrievalContractError(f"{field} contains a placeholder value")
    return value


def _validate_digest(value: Any, field: str) -> str:
    digest = _required_token(value, field)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise TypedRetrievalContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _contract_digest(record: dict[str, Any]) -> str:
    return _validate_digest(record.get("contract_digest"), "contract_digest")


def _evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise TypedRetrievalContractError(
            "evidence must contain typed chunk provenance"
        )
    normalized: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            raise TypedRetrievalContractError("evidence entries must be objects")
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise TypedRetrievalContractError("evidence metadata must be an object")
        normalized.append(
            {
                "chunk_id": _required_token(item.get("chunk_id"), "chunk_id"),
                "source_key": _required_token(item.get("source_key"), "source_key"),
                "source_revision": _required_token(
                    item.get("source_revision"), "source_revision"
                ),
                "metadata": metadata,
            }
        )
    return normalized


def _validate_assertion(record: dict[str, Any]) -> dict[str, Any]:
    assertion_id = _required_token(record.get("assertion_id"), "assertion_id")
    predicate = _required_token(record.get("predicate"), "predicate")
    src_id = _required_token(record.get("src_id"), "src_id")
    dst_id = _required_token(record.get("dst_id"), "dst_id")
    build_id = _required_token(record.get("build_id"), "build_id")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise TypedRetrievalContractError("assertion metadata must be an object")
    confidence = record.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise TypedRetrievalContractError("confidence must be between zero and one")
    method = record.get("method")
    if method is not None:
        method = _required_token(method, "method")
    result = {
        "build_id": build_id,
        "assertion_id": assertion_id,
        "predicate": predicate,
        "src_id": src_id,
        "dst_id": dst_id,
        "evidence": _evidence(record),
        "metadata": metadata,
        "confidence": float(confidence) if confidence is not None else None,
        "method": method,
        "observed_from": record.get("observed_from"),
        "observed_to": record.get("observed_to"),
        "valid_from": record.get("valid_from"),
        "valid_to": record.get("valid_to"),
        "contract_digest": _contract_digest(record),
    }
    return result


def _validate_entity(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise TypedRetrievalContractError("entity metadata must be an object")
    return {
        "build_id": _required_token(record.get("build_id"), "build_id"),
        "entity_id": _required_token(record.get("entity_id"), "entity_id"),
        "entity_type": _required_token(record.get("entity_type"), "entity_type"),
        "evidence": _evidence(record),
        "metadata": metadata,
        "observed_from": record.get("observed_from"),
        "observed_to": record.get("observed_to"),
        "valid_from": record.get("valid_from"),
        "valid_to": record.get("valid_to"),
        "contract_digest": _contract_digest(record),
    }


def _candidate_id(candidate: dict[str, Any], field: str) -> str:
    record_id = _required_token(candidate.get(field), field)
    candidate_id = candidate.get("id")
    if candidate_id is not None and candidate_id != record_id:
        raise TypedRetrievalContractError(f"candidate id does not match {field}")
    return record_id


def _candidate_score(candidate: dict[str, Any]) -> float:
    for field in ("similarity", "distance", "score"):
        value = candidate.get(field)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            break
        return float(value)
    raise TypedRetrievalContractError("typed candidate lacks a finite retrieval score")


def _citation_id(evidence: dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "chunk_id": evidence["chunk_id"],
            "source_key": evidence["source_key"],
            "source_revision": evidence["source_revision"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"citation:{digest[:24]}"


def _validate_record_identity(
    record: Mapping[str, Any],
    identity: TypedRetrievalIdentity,
    *,
    record_name: str,
    require_evidence: bool = False,
    require_manifest_digest: bool = False,
) -> None:
    if record.get("build_id") != identity.build_id:
        raise TypedRetrievalContractError(
            f"{record_name} build_id does not match the active generation"
        )
    if record.get("contract_digest") != identity.contract_digest:
        raise TypedRetrievalContractError(
            f"{record_name} contract_digest does not match the active generation"
        )
    has_manifest_digest = "manifest_digest" in record
    manifest_digest = record.get("manifest_digest")
    if require_manifest_digest and not has_manifest_digest:
        raise TypedRetrievalContractError(
            f"{record_name} lacks active generation manifest identity"
        )
    if has_manifest_digest and manifest_digest != identity.manifest_digest:
        raise TypedRetrievalContractError(
            f"{record_name} manifest_digest does not match the active generation"
        )
    evidence = record.get("evidence")
    if require_evidence and (not isinstance(evidence, list) or not evidence):
        raise TypedRetrievalContractError(
            f"{record_name} lacks active generation chunk provenance"
        )
    if evidence is not None:
        if not isinstance(evidence, list):
            raise TypedRetrievalContractError(f"{record_name} evidence must be a list")
        for item in evidence:
            if not isinstance(item, Mapping) or (
                item.get("source_revision") != identity.source_revision
            ):
                raise TypedRetrievalContractError(
                    f"{record_name} evidence does not match the active generation source revision"
                )
    source_revision = record.get("source_revision")
    if source_revision is not None and source_revision != identity.source_revision:
        raise TypedRetrievalContractError(
            f"{record_name} source revision does not match the active generation"
        )


def validate_typed_retrieval_data_identity(
    data: Any,
    identity: TypedRetrievalIdentity | None = None,
) -> None:
    """Fail closed when structured response rows escape one active generation."""
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, Mapping):
        raise TypedRetrievalContractError("typed retrieval data must be an object")
    expected = identity or _TYPED_RETRIEVAL_IDENTITY.get()
    if expected is None:
        return
    for field in ("entities", "assertions", "chunks", "citations"):
        rows = data.get(field)
        if not isinstance(rows, list):
            raise TypedRetrievalContractError(f"typed retrieval {field} must be a list")
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypedRetrievalContractError(
                    f"typed retrieval {field} entries must be objects"
                )
            if field == "citations":
                if row.get("source_revision") != expected.source_revision:
                    raise TypedRetrievalContractError(
                        "citation source revision does not match the active generation"
                    )
            else:
                _validate_record_identity(
                    row, expected, record_name=field.removesuffix("s")
                )


async def retrieve_typed_records(
    *,
    query: str,
    mode: QueryMode,
    top_k: int,
    graph: _Graph,
    entities_vdb: _VectorStorage | None,
    relationships_vdb: _VectorStorage | None,
    text_chunks_db: _ChunkStorage,
    jurisdiction_predicates: frozenset[str] = DEFAULT_JURISDICTION_PREDICATES,
) -> TypedRetrievalResult:
    """Retrieve immutable graph records without endpoint or provenance fallback.

    ``jurisdiction_predicates`` selects which assertion predicates additionally
    emit a ``jurisdiction`` claim; the set is deployment policy (OceanStack
    passes its maritime predicates), not generic core ontology. Members are
    matched case-insensitively against assertion predicates.
    """
    if mode not in {"local", "global", "hybrid", "mix"}:
        raise TypedRetrievalContractError(f"unsupported typed query mode {mode!r}")

    # Assertion predicates are compared casefolded (see the claim emission
    # below); casefold the policy set on entry so an uppercase member such as
    # "LOCATED_IN" matches instead of silently never firing.
    jurisdiction_predicates = frozenset(
        predicate.casefold() for predicate in jurisdiction_predicates
    )

    entity_scores: dict[str, float] = {}
    if mode in {"local", "hybrid", "mix"}:
        if entities_vdb is None:
            raise TypedRetrievalContractError(
                "typed local retrieval requires entities_vdb"
            )
        for candidate in await entities_vdb.query(query, top_k=top_k):
            identity = _TYPED_RETRIEVAL_IDENTITY.get()
            if identity is not None:
                _validate_record_identity(
                    candidate,
                    identity,
                    record_name="candidate",
                    require_evidence=True,
                )
            entity_id = _candidate_id(candidate, "entity_id")
            entity_scores[entity_id] = max(
                entity_scores.get(entity_id, -math.inf), _candidate_score(candidate)
            )

    selected: dict[str, tuple[dict[str, Any], float, str, list[str]]] = {}
    if entity_scores:
        incident = await graph.get_graph_assertions_for_entities(sorted(entity_scores))
        for stored in incident:
            assertion = _validate_assertion(stored)
            roots = [
                entity_id
                for entity_id in entity_scores
                if entity_id in {assertion["src_id"], assertion["dst_id"]}
            ]
            if not roots:
                raise TypedRetrievalContractError(
                    "incident assertion does not contain a requested entity"
                )
            root = min(roots, key=lambda item: (-entity_scores[item], item))
            if assertion["src_id"] == root:
                direction = "outbound"
                path = [root, assertion["dst_id"]]
            else:
                direction = "inbound"
                path = [root, assertion["src_id"]]
            selected[assertion["assertion_id"]] = (
                assertion,
                entity_scores[root],
                direction,
                path,
            )

    if mode in {"global", "hybrid", "mix"}:
        if relationships_vdb is None:
            raise TypedRetrievalContractError(
                "typed global retrieval requires relationships_vdb"
            )
        relation_candidates = await relationships_vdb.query(query, top_k=top_k)
        scored_ids: dict[str, float] = {}
        for candidate in relation_candidates:
            identity = _TYPED_RETRIEVAL_IDENTITY.get()
            if identity is not None:
                _validate_record_identity(
                    candidate,
                    identity,
                    record_name="candidate",
                    require_evidence=True,
                )
            assertion_id = _candidate_id(candidate, "assertion_id")
            scored_ids[assertion_id] = max(
                scored_ids.get(assertion_id, -math.inf), _candidate_score(candidate)
            )
        assertions = await graph.get_graph_assertions(list(scored_ids))
        missing = sorted(set(scored_ids) - set(assertions))
        if missing:
            raise TypedRetrievalContractError(
                f"typed assertion candidates are absent from graph storage: {missing}"
            )
        for assertion_id, score in scored_ids.items():
            assertion = _validate_assertion(assertions[assertion_id])
            candidate = (
                assertion,
                score,
                "outbound",
                [assertion["src_id"], assertion["dst_id"]],
            )
            current = selected.get(assertion_id)
            if (
                current is None
                or score > current[1]
                or (score == current[1] and candidate[3] < current[3])
            ):
                selected[assertion_id] = candidate

    assertion_rows: list[dict[str, Any]] = []
    for assertion, score, direction, path in sorted(
        selected.values(), key=lambda item: (-item[1], item[0]["assertion_id"])
    ):
        assertion_rows.append(
            {
                **assertion,
                "direction": direction,
                "score": score,
                "traversal_path": path,
            }
        )

    entity_ids = sorted(
        set(entity_scores)
        | {
            entity_id
            for assertion in assertion_rows
            for entity_id in (assertion["src_id"], assertion["dst_id"])
        }
    )
    entity_rows: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        stored = await graph.get_graph_entity(entity_id)
        if stored is None:
            raise TypedRetrievalContractError(
                f"typed entity {entity_id!r} is absent from graph storage"
            )
        entity = _validate_entity(stored)
        incident_scores = [
            assertion["score"]
            for assertion in assertion_rows
            if entity_id in {assertion["src_id"], assertion["dst_id"]}
        ]
        score = max([entity_scores.get(entity_id, -math.inf), *incident_scores])
        entity_rows.append({**entity, "score": score, "traversal_path": [entity_id]})

    identity = _TYPED_RETRIEVAL_IDENTITY.get()
    if identity is not None:
        for entity in entity_rows:
            _validate_record_identity(entity, identity, record_name="entity")
        for assertion in assertion_rows:
            _validate_record_identity(assertion, identity, record_name="assertion")

    digests = {row["contract_digest"] for row in [*entity_rows, *assertion_rows]}
    if len(digests) > 1:
        raise TypedRetrievalContractError(
            "retrieved records span multiple contract digests"
        )

    evidence_by_chunk: dict[str, dict[str, Any]] = {}
    for record in [*entity_rows, *assertion_rows]:
        for item in record["evidence"]:
            prior = evidence_by_chunk.setdefault(item["chunk_id"], item)
            if (
                prior["source_key"],
                prior["source_revision"],
            ) != (item["source_key"], item["source_revision"]):
                raise TypedRetrievalContractError(
                    f"chunk {item['chunk_id']!r} has conflicting provenance"
                )

    chunk_ids = sorted(evidence_by_chunk)
    chunk_payloads = await text_chunks_db.get_by_ids(chunk_ids)
    if len(chunk_payloads) != len(chunk_ids):
        raise TypedRetrievalContractError(
            "chunk storage did not preserve typed batch read cardinality"
        )
    chunks: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    citation_by_chunk: dict[str, str] = {}
    for chunk_id, payload in zip(chunk_ids, chunk_payloads, strict=True):
        if not isinstance(payload, dict):
            raise TypedRetrievalContractError(
                f"evidence chunk {chunk_id!r} is absent from chunk storage"
            )
        sidecar = payload.get("sidecar")
        if not isinstance(sidecar, dict):
            raise TypedRetrievalContractError(
                f"evidence chunk {chunk_id!r} lacks typed sidecar provenance"
            )
        evidence = evidence_by_chunk[chunk_id]
        source_key = _required_token(sidecar.get("source_key"), "source_key")
        source_revision = _required_token(
            sidecar.get("source_revision"), "source_revision"
        )
        if (source_key, source_revision) != (
            evidence["source_key"],
            evidence["source_revision"],
        ):
            raise TypedRetrievalContractError(
                f"evidence chunk {chunk_id!r} provenance does not match its assertion"
            )
        citation_id = _citation_id(evidence)
        citation_by_chunk[chunk_id] = citation_id
        citation = {
            "citation_id": citation_id,
            "chunk_id": chunk_id,
            "source_key": source_key,
            "source_revision": source_revision,
        }
        citations.append(citation)
        metadata = sidecar.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypedRetrievalContractError("chunk metadata must be an object")
        chunk = {
            **citation,
            "build_id": _required_token(sidecar.get("build_id"), "build_id"),
            "contract_digest": _contract_digest(sidecar),
            "manifest_digest": _validate_digest(
                sidecar.get("manifest_digest"), "manifest_digest"
            ),
            "content": _required_token(payload.get("content"), "content"),
            "metadata": metadata,
        }
        if identity is not None:
            _validate_record_identity(
                chunk,
                identity,
                record_name="chunk",
                require_manifest_digest=True,
            )
        chunks.append(chunk)

    citations.sort(key=lambda item: item["citation_id"])
    chunks.sort(key=lambda item: item["citation_id"])
    assertion_claims = [
        {
            "kind": "assertion",
            "record_ids": [assertion["assertion_id"]],
            "citation_ids": sorted(
                {citation_by_chunk[item["chunk_id"]] for item in assertion["evidence"]}
            ),
        }
        for assertion in sorted(assertion_rows, key=lambda item: item["assertion_id"])
    ]
    identity_claims = [
        {
            "kind": "identity",
            "record_ids": [entity["entity_id"]],
            "citation_ids": sorted(
                {citation_by_chunk[item["chunk_id"]] for item in entity["evidence"]}
            ),
        }
        for entity in sorted(entity_rows, key=lambda item: item["entity_id"])
    ]
    jurisdiction_claims = [
        {
            "kind": "jurisdiction",
            "record_ids": [assertion["assertion_id"]],
            "citation_ids": sorted(
                {citation_by_chunk[item["chunk_id"]] for item in assertion["evidence"]}
            ),
        }
        for assertion in sorted(assertion_rows, key=lambda item: item["assertion_id"])
        if assertion["predicate"].casefold() in jurisdiction_predicates
    ]
    provenance_claims = [
        {
            "kind": "provenance",
            "record_ids": [chunk["chunk_id"]],
            "citation_ids": [chunk["citation_id"]],
        }
        for chunk in sorted(chunks, key=lambda item: item["chunk_id"])
    ]
    claims = [
        *assertion_claims,
        *identity_claims,
        *jurisdiction_claims,
        *provenance_claims,
    ]

    return TypedRetrievalResult(
        entities=entity_rows,
        assertions=assertion_rows,
        chunks=chunks,
        citations=citations,
        claims=claims,
    )


def validate_typed_graph_response(graph: Any) -> dict[str, Any]:
    """Validate and return a JSON graph containing only typed ASSERTION edges."""
    if hasattr(graph, "model_dump"):
        graph = graph.model_dump()
    if not isinstance(graph, dict):
        raise TypedRetrievalContractError("typed graph response must be an object")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TypedRetrievalContractError(
            "typed graph response requires nodes and edges"
        )
    identity = _TYPED_RETRIEVAL_IDENTITY.get()
    for node in nodes:
        if not isinstance(node, dict):
            raise TypedRetrievalContractError("typed graph nodes must be objects")
        properties = node.get("properties")
        if not isinstance(properties, dict):
            raise TypedRetrievalContractError(
                "graph entity properties must be an object"
            )
        record_kind = properties.get("_lightrag_record_kind")
        if record_kind != "GraphEntity":
            raise TypedRetrievalContractError(
                "graph node lacks GraphEntity record identity"
            )
        entity = _validate_entity(properties)
        if node.get("id") != entity["entity_id"]:
            raise TypedRetrievalContractError("node id does not match entity_id")
        if identity is not None:
            _validate_record_identity(entity, identity, record_name="entity")
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("type") != "ASSERTION":
            raise TypedRetrievalContractError(
                "typed graph response may contain only ASSERTION records"
            )
        properties = edge.get("properties")
        if not isinstance(properties, dict):
            raise TypedRetrievalContractError("ASSERTION properties must be an object")
        record_kind = properties.get("_lightrag_record_kind")
        if record_kind != "GraphAssertion":
            raise TypedRetrievalContractError(
                "ASSERTION edge lacks GraphAssertion record identity"
            )
        assertion = _validate_assertion(properties)
        if identity is not None:
            _validate_record_identity(assertion, identity, record_name="assertion")
        if edge.get("id") != assertion["assertion_id"]:
            raise TypedRetrievalContractError("edge id does not match assertion_id")
        if (edge.get("source"), edge.get("target")) != (
            assertion["src_id"],
            assertion["dst_id"],
        ):
            raise TypedRetrievalContractError(
                "edge endpoints do not match the directed assertion"
            )
    return graph
