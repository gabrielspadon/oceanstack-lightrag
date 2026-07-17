from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from lightrag.kg.graph_contract import (
    EvidenceRef,
    GraphAssertion,
    GraphChunk,
    GraphEntity,
    KnowledgeGraphBuild,
)


BUILD_ID = "build-2026-07-16"
SOURCE_KEY = "db/schemas/core.sql"
SOURCE_REVISION = "8b135095"


def _evidence(chunk_id: str = "chunk-schema-1") -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id,
        source_key=SOURCE_KEY,
        source_revision=SOURCE_REVISION,
        metadata={"lines": [10, 20]},
    )


def _chunk(chunk_id: str = "chunk-schema-1") -> GraphChunk:
    return GraphChunk(
        build_id=BUILD_ID,
        chunk_id=chunk_id,
        source_key=SOURCE_KEY,
        source_revision=SOURCE_REVISION,
        content="CREATE TABLE core.vessel (...)",
        metadata={"parser": {"name": "sql", "version": 1}},
    )


def _entity(entity_id: str, entity_type: str = "table") -> GraphEntity:
    return GraphEntity(
        build_id=BUILD_ID,
        entity_id=entity_id,
        entity_type=entity_type,
        evidence=(_evidence(),),
        metadata={"qualified": True, "aliases": [entity_id]},
    )


def _assertion(
    assertion_id: str,
    predicate: str,
    src_id: str = "core.vessel",
    dst_id: str = "core.position",
) -> GraphAssertion:
    return GraphAssertion(
        build_id=BUILD_ID,
        assertion_id=assertion_id,
        predicate=predicate,
        src_id=src_id,
        dst_id=dst_id,
        evidence=(_evidence(),),
        confidence=0.95,
        method="sql-ddl",
        observed_from=datetime(2026, 7, 16, tzinfo=timezone.utc),
        observed_to=datetime(2026, 7, 17, tzinfo=timezone.utc),
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=None,
        metadata={"columns": ["vessel_id"]},
    )


def _build() -> KnowledgeGraphBuild:
    return KnowledgeGraphBuild.create(
        build_id=BUILD_ID,
        chunks=(_chunk(),),
        entities=(_entity("core.vessel"), _entity("core.position")),
        assertions=(_assertion("assertion-references", "references"),),
        metadata={"format": "greenfield"},
    )


def test_build_has_deterministic_canonical_serialization() -> None:
    chunks = [_chunk("chunk-schema-2"), _chunk()]
    entities = [_entity("core.position"), _entity("core.vessel")]
    assertions = [
        _assertion("assertion-owns", "owns"),
        _assertion("assertion-references", "references"),
    ]

    first = KnowledgeGraphBuild.create(
        build_id=BUILD_ID,
        chunks=chunks,
        entities=entities,
        assertions=assertions,
        metadata={"nested": {"flags": [True, None, 3.5]}},
    )
    second = KnowledgeGraphBuild.create(
        build_id=BUILD_ID,
        chunks=reversed(chunks),
        entities=reversed(entities),
        assertions=reversed(assertions),
        metadata={"nested": {"flags": [True, None, 3.5]}},
    )

    assert first.contract_digest == second.contract_digest
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.to_canonical_dict()["assertions"][0]["src_id"] == "core.vessel"
    assert first.to_canonical_dict()["assertions"][0]["dst_id"] == "core.position"
    assert {
        assertion["predicate"] for assertion in first.to_canonical_dict()["assertions"]
    } == {"owns", "references"}
    assert first.to_canonical_dict()["chunks"][0]["source_key"] == SOURCE_KEY


@pytest.mark.parametrize("value", ["", "   ", "UNKNOWN", "unknown"])
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda value: EvidenceRef(value, SOURCE_KEY, SOURCE_REVISION),
            id="evidence-chunk-id",
        ),
        pytest.param(lambda value: replace(_chunk(), chunk_id=value), id="chunk-id"),
        pytest.param(
            lambda value: replace(_entity("core.vessel"), entity_id=value),
            id="entity-id",
        ),
        pytest.param(
            lambda value: replace(_entity("core.vessel"), entity_type=value),
            id="entity-type",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"),
                assertion_id=value,
            ),
            id="assertion-id",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), predicate=value
            ),
            id="predicate",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), src_id=value
            ),
            id="source-entity-id",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), dst_id=value
            ),
            id="destination-entity-id",
        ),
        pytest.param(
            lambda value: KnowledgeGraphBuild.create(
                build_id=value,
                chunks=(),
                entities=(),
                assertions=(),
            ),
            id="build-id",
        ),
    ],
)
def test_rejects_blank_or_unknown_contract_tokens(factory, value: str) -> None:
    with pytest.raises(ValueError):
        factory(value)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda value: EvidenceRef(value, SOURCE_KEY, SOURCE_REVISION),
            id="evidence-chunk-id",
        ),
        pytest.param(
            lambda value: EvidenceRef("chunk", SOURCE_KEY, value),
            id="evidence-source-revision",
        ),
        pytest.param(
            lambda value: EvidenceRef(
                "chunk", f"oceanstack/{value}.py", SOURCE_REVISION
            ),
            id="evidence-source-key",
        ),
        pytest.param(lambda value: replace(_chunk(), build_id=value), id="chunk-build"),
        pytest.param(lambda value: replace(_chunk(), chunk_id=value), id="chunk-id"),
        pytest.param(
            lambda value: replace(_chunk(), source_revision=value),
            id="chunk-source-revision",
        ),
        pytest.param(
            lambda value: replace(_chunk(), source_key=f"oceanstack/{value}.py"),
            id="chunk-source-key",
        ),
        pytest.param(
            lambda value: replace(_entity("core.vessel"), build_id=value),
            id="entity-build",
        ),
        pytest.param(
            lambda value: replace(_entity("core.vessel"), entity_id=value),
            id="entity-id",
        ),
        pytest.param(
            lambda value: replace(_entity("core.vessel"), entity_type=value),
            id="entity-type",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), build_id=value
            ),
            id="assertion-build",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"),
                assertion_id=value,
            ),
            id="assertion-id",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), predicate=value
            ),
            id="assertion-predicate",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), src_id=value
            ),
            id="assertion-source",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), dst_id=value
            ),
            id="assertion-destination",
        ),
        pytest.param(
            lambda value: replace(
                _assertion("assertion-references", "references"), method=value
            ),
            id="assertion-method",
        ),
        pytest.param(
            lambda value: KnowledgeGraphBuild.create(
                build_id=value,
                chunks=(),
                entities=(),
                assertions=(),
            ),
            id="build-id",
        ),
    ],
)
def test_rejects_nul_in_contract_tokens(factory) -> None:
    with pytest.raises(ValueError, match="NUL"):
        factory("before\u0000after")


def test_rejects_nul_in_chunk_content() -> None:
    with pytest.raises(ValueError, match="NUL"):
        replace(_chunk(), content="before\u0000after")


@pytest.mark.parametrize(
    "metadata",
    [
        {"before\u0000after": "value"},
        {"nested": {"before\u0000after": "value"}},
        {"value": "before\u0000after"},
        {"nested": ["safe", {"value": "before\u0000after"}]},
    ],
)
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda metadata: replace(_evidence(), metadata=metadata),
            id="evidence",
        ),
        pytest.param(
            lambda metadata: replace(_chunk(), metadata=metadata),
            id="chunk",
        ),
        pytest.param(
            lambda metadata: replace(_entity("core.vessel"), metadata=metadata),
            id="entity",
        ),
        pytest.param(
            lambda metadata: replace(
                _assertion("assertion-references", "references"),
                metadata=metadata,
            ),
            id="assertion",
        ),
        pytest.param(
            lambda metadata: KnowledgeGraphBuild.create(
                build_id=BUILD_ID,
                chunks=(_chunk(),),
                entities=(_entity("core.vessel"), _entity("core.position")),
                assertions=(_assertion("assertion-references", "references"),),
                metadata=metadata,
            ),
            id="build",
        ),
    ],
)
def test_rejects_nul_recursively_in_all_metadata(factory, metadata) -> None:
    with pytest.raises(ValueError, match="NUL"):
        factory(metadata)


def test_preserves_noncharacters_and_nested_json_metadata() -> None:
    metadata = {"nested": {"values": ["noncharacter:\ufffe", 3, True, None]}}

    chunk = replace(_chunk(), metadata=metadata)

    assert chunk.metadata["nested"]["values"] == (
        "noncharacter:\ufffe",
        3,
        True,
        None,
    )


@pytest.mark.parametrize(
    "source_key",
    [
        "schema.sql",
        "/db/schemas/schema.sql",
        "../db/schemas/schema.sql",
        "db/../schemas/schema.sql",
        "db\\schemas\\schema.sql",
    ],
)
def test_rejects_non_repository_relative_source_keys(source_key: str) -> None:
    with pytest.raises(ValueError, match="source_key"):
        EvidenceRef("chunk-schema-1", source_key, SOURCE_REVISION)


def test_source_key_namespaces_repository_root_files() -> None:
    with pytest.raises(ValueError, match="repository namespace"):
        EvidenceRef("chunk-readme", "README.md", SOURCE_REVISION)

    evidence = EvidenceRef(
        "chunk-readme",
        "oceanstack/README.md",
        SOURCE_REVISION,
    )

    assert evidence.source_key == "oceanstack/README.md"


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_rejects_malformed_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        replace(
            _assertion("assertion-references", "references"),
            confidence=confidence,
        )


def test_rejects_blank_method_when_supplied() -> None:
    with pytest.raises(ValueError, match="method"):
        replace(_assertion("assertion-references", "references"), method=" ")


@pytest.mark.parametrize(
    "changes",
    [
        {
            "observed_from": datetime(2026, 7, 17, tzinfo=timezone.utc),
            "observed_to": datetime(2026, 7, 16, tzinfo=timezone.utc),
        },
        {
            "valid_from": datetime(2026, 7, 17, tzinfo=timezone.utc),
            "valid_to": datetime(2026, 7, 16, tzinfo=timezone.utc),
        },
        {"observed_from": datetime(2026, 7, 16)},
        {"valid_to": "2026-07-16T00:00:00Z"},
    ],
)
def test_rejects_invalid_temporal_intervals(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="observed|valid"):
        replace(_assertion("assertion-references", "references"), **changes)


def test_entity_temporal_intervals_are_validated() -> None:
    with pytest.raises(ValueError, match="valid"):
        replace(
            _entity("core.vessel"),
            valid_from=datetime(2026, 7, 17, tzinfo=timezone.utc),
            valid_to=datetime(2026, 7, 16, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"bad": object()},
        {"bad": {1, 2}},
        {1: "non-string key"},
        {"bad": b"bytes"},
        {"bad": float("nan")},
    ],
)
def test_rejects_non_json_metadata(metadata: object) -> None:
    with pytest.raises(ValueError, match="metadata"):
        replace(_chunk(), metadata=metadata)


def test_metadata_is_deeply_immutable_and_detached_from_input() -> None:
    metadata = {"nested": {"values": [1, 2]}}
    chunk = replace(_chunk(), metadata=metadata)

    metadata["nested"]["values"].append(3)

    assert chunk.metadata["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        chunk.metadata["new"] = "value"


@pytest.mark.parametrize("record_type", ["chunk", "entity", "assertion"])
def test_rejects_duplicate_record_ids(record_type: str) -> None:
    chunks = [_chunk()]
    entities = [_entity("core.vessel"), _entity("core.position")]
    assertions = [_assertion("assertion-references", "references")]
    if record_type == "chunk":
        chunks.append(_chunk())
    elif record_type == "entity":
        entities.append(_entity("core.vessel"))
    else:
        assertions.append(_assertion("assertion-references", "owns"))

    with pytest.raises(ValueError, match="duplicate"):
        KnowledgeGraphBuild.create(
            build_id=BUILD_ID,
            chunks=chunks,
            entities=entities,
            assertions=assertions,
        )


def test_rejects_missing_assertion_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        KnowledgeGraphBuild.create(
            build_id=BUILD_ID,
            chunks=(_chunk(),),
            entities=(_entity("core.vessel"),),
            assertions=(
                _assertion(
                    "assertion-references",
                    "references",
                    dst_id="core.missing",
                ),
            ),
        )


@pytest.mark.parametrize("record_type", ["entity", "assertion"])
def test_rejects_missing_evidence_chunks(record_type: str) -> None:
    missing_evidence = _evidence("chunk-missing")
    entities = [_entity("core.vessel"), _entity("core.position")]
    assertions = [_assertion("assertion-references", "references")]
    if record_type == "entity":
        entities[0] = replace(entities[0], evidence=(missing_evidence,))
    else:
        assertions[0] = replace(assertions[0], evidence=(missing_evidence,))

    with pytest.raises(ValueError, match="evidence"):
        KnowledgeGraphBuild.create(
            build_id=BUILD_ID,
            chunks=(_chunk(),),
            entities=entities,
            assertions=assertions,
        )


def test_rejects_evidence_source_mismatch() -> None:
    mismatched = replace(_evidence(), source_revision="different-revision")
    with pytest.raises(ValueError, match="evidence"):
        KnowledgeGraphBuild.create(
            build_id=BUILD_ID,
            chunks=(_chunk(),),
            entities=(
                replace(_entity("core.vessel"), evidence=(mismatched,)),
                _entity("core.position"),
            ),
            assertions=(_assertion("assertion-references", "references"),),
        )


@pytest.mark.parametrize("record_type", ["chunk", "entity", "assertion"])
def test_rejects_build_record_mismatches(record_type: str) -> None:
    chunks = [_chunk()]
    entities = [_entity("core.vessel"), _entity("core.position")]
    assertions = [_assertion("assertion-references", "references")]
    if record_type == "chunk":
        chunks[0] = replace(chunks[0], build_id="different-build")
    elif record_type == "entity":
        entities[0] = replace(entities[0], build_id="different-build")
    else:
        assertions[0] = replace(assertions[0], build_id="different-build")

    with pytest.raises(ValueError, match="build_id"):
        KnowledgeGraphBuild.create(
            build_id=BUILD_ID,
            chunks=chunks,
            entities=entities,
            assertions=assertions,
        )


def test_rejects_contract_digest_mismatch() -> None:
    with pytest.raises(ValueError, match="contract_digest"):
        replace(_build(), contract_digest="0" * 64)


def test_create_canonicalizes_and_hashes_payload_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_calls = 0
    digest_calls = 0
    canonical_payload = KnowledgeGraphBuild._canonical_payload
    digest_payload = KnowledgeGraphBuild._digest_payload

    def counting_canonical_payload(**kwargs):
        nonlocal canonical_calls
        canonical_calls += 1
        return canonical_payload(**kwargs)

    def counting_digest_payload(payload):
        nonlocal digest_calls
        digest_calls += 1
        return digest_payload(payload)

    monkeypatch.setattr(
        KnowledgeGraphBuild,
        "_canonical_payload",
        staticmethod(counting_canonical_payload),
    )
    monkeypatch.setattr(
        KnowledgeGraphBuild,
        "_digest_payload",
        staticmethod(counting_digest_payload),
    )

    _build()

    assert canonical_calls == 1
    assert digest_calls == 1


def test_digest_is_independent_of_evidence_order() -> None:
    second_chunk = replace(
        _chunk("chunk-schema-2"),
        source_key="db/schemas/derived.sql",
    )
    first_evidence = _evidence()
    second_evidence = EvidenceRef(
        chunk_id=second_chunk.chunk_id,
        source_key=second_chunk.source_key,
        source_revision=second_chunk.source_revision,
    )
    entity = replace(_entity("core.vessel"), evidence=(first_evidence, second_evidence))
    assertion = replace(
        _assertion("assertion-references", "references"),
        evidence=(first_evidence, second_evidence),
    )
    first = KnowledgeGraphBuild.create(
        build_id=BUILD_ID,
        chunks=(_chunk(), second_chunk),
        entities=(entity, _entity("core.position")),
        assertions=(assertion,),
    )
    second = KnowledgeGraphBuild.create(
        build_id=BUILD_ID,
        chunks=(second_chunk, _chunk()),
        entities=(
            replace(entity, evidence=tuple(reversed(entity.evidence))),
            _entity("core.position"),
        ),
        assertions=(replace(assertion, evidence=tuple(reversed(assertion.evidence))),),
    )

    assert first.contract_digest == second.contract_digest
    assert first.to_canonical_json() == second.to_canonical_json()


def test_canonical_empty_build_golden_vector() -> None:
    build = KnowledgeGraphBuild.create(
        build_id="build-empty",
        chunks=(),
        entities=(),
        assertions=(),
    )

    assert (
        build.contract_digest
        == "0343d539adc4f7d63be01a929f296cb3b9579842b38c8a9ee9d74140ab0e375d"
    )
    assert build.to_canonical_json() == (
        '{"assertions":[],"build_id":"build-empty","chunks":[],'
        '"contract_digest":"0343d539adc4f7d63be01a929f296cb3b9579842b38c8a9ee9d74140ab0e375d",'
        '"entities":[],"metadata":{}}'
    )


def test_canonical_full_build_golden_vector() -> None:
    build = _build()

    assert (
        build.contract_digest
        == "8e9e6109fec3ff4fd0ca7197fbcbae3e471fab2897c754e6420eb625fa87e83a"
    )
    assert build.to_canonical_json() == (
        '{"assertions":[{"assertion_id":"assertion-references",'
        '"build_id":"build-2026-07-16","confidence":0.95,'
        '"dst_id":"core.position","evidence":[{"chunk_id":"chunk-schema-1",'
        '"metadata":{"lines":[10,20]},"source_key":"db/schemas/core.sql",'
        '"source_revision":"8b135095"}],"metadata":{"columns":["vessel_id"]},'
        '"method":"sql-ddl","observed_from":"2026-07-16T00:00:00Z",'
        '"observed_to":"2026-07-17T00:00:00Z","predicate":"references",'
        '"src_id":"core.vessel","valid_from":"2026-01-01T00:00:00Z",'
        '"valid_to":null}],"build_id":"build-2026-07-16","chunks":'
        '[{"build_id":"build-2026-07-16","chunk_id":"chunk-schema-1",'
        '"content":"CREATE TABLE core.vessel (...)","metadata":{"parser":'
        '{"name":"sql","version":1}},"source_key":"db/schemas/core.sql",'
        '"source_revision":"8b135095"}],"contract_digest":'
        '"8e9e6109fec3ff4fd0ca7197fbcbae3e471fab2897c754e6420eb625fa87e83a",'
        '"entities":[{"build_id":"build-2026-07-16","entity_id":"core.position",'
        '"entity_type":"table","evidence":[{"chunk_id":"chunk-schema-1",'
        '"metadata":{"lines":[10,20]},"source_key":"db/schemas/core.sql",'
        '"source_revision":"8b135095"}],"metadata":{"aliases":["core.position"],'
        '"qualified":true},"observed_from":null,"observed_to":null,'
        '"valid_from":null,"valid_to":null},{"build_id":"build-2026-07-16",'
        '"entity_id":"core.vessel","entity_type":"table","evidence":'
        '[{"chunk_id":"chunk-schema-1","metadata":{"lines":[10,20]},'
        '"source_key":"db/schemas/core.sql","source_revision":"8b135095"}],'
        '"metadata":{"aliases":["core.vessel"],"qualified":true},'
        '"observed_from":null,"observed_to":null,"valid_from":null,'
        '"valid_to":null}],"metadata":{"format":"greenfield"}}'
    )
