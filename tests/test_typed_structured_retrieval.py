from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import numpy as np
import pytest
import lightrag.typed_retrieval as typed_retrieval_module

from lightrag.base import QueryParam
from lightrag.generation import (
    GenerationFenceKind,
    GenerationOperationFence,
    bind_generation_operation_fence,
    generation_workspace,
    reset_generation_operation_fence,
)
from lightrag.kg.graph_contract import (
    EvidenceRef,
    GraphAssertion,
    GraphChunk,
    GraphEntity,
    KnowledgeGraphBuild,
)
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.operate import _build_query_context
from lightrag.typed_retrieval import (
    TypedRetrievalContractError,
    retrieve_typed_records,
    validate_typed_graph_response,
)
from lightrag.utils import EmbeddingFunc


pytestmark = pytest.mark.offline
MANIFEST_DIGEST = "e" * 64


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8))


def _evidence(chunk_id: str, source_key: str) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=chunk_id,
        source_key=source_key,
        source_revision="source-revision-1",
        metadata={"line_start": 10, "line_end": 20},
    )


def _build() -> KnowledgeGraphBuild:
    chunks = (
        GraphChunk(
            build_id="build-1",
            chunk_id="chunk-a",
            source_key="src/a.py",
            source_revision="source-revision-1",
            content="alpha content",
        ),
        GraphChunk(
            build_id="build-1",
            chunk_id="chunk-b",
            source_key="src/b.py",
            source_revision="source-revision-1",
            content="beta content",
        ),
    )
    entities = (
        GraphEntity(
            build_id="build-1",
            entity_id="alpha",
            entity_type="table",
            evidence=(_evidence("chunk-a", "src/a.py"),),
        ),
        GraphEntity(
            build_id="build-1",
            entity_id="beta",
            entity_type="table",
            evidence=(_evidence("chunk-b", "src/b.py"),),
        ),
    )
    assertions = (
        GraphAssertion(
            build_id="build-1",
            assertion_id="parallel-a",
            predicate="references",
            src_id="alpha",
            dst_id="beta",
            evidence=(_evidence("chunk-a", "src/a.py"),),
            confidence=0.91,
            method="static-analysis",
        ),
        GraphAssertion(
            build_id="build-1",
            assertion_id="parallel-b",
            predicate="located_in",
            src_id="alpha",
            dst_id="beta",
            evidence=(_evidence("chunk-b", "src/b.py"),),
            confidence=0.72,
            method="schema-analysis",
        ),
        GraphAssertion(
            build_id="build-1",
            assertion_id="reciprocal",
            predicate="owns",
            src_id="beta",
            dst_id="alpha",
            evidence=(
                _evidence("chunk-a", "src/a.py"),
                _evidence("chunk-b", "src/b.py"),
            ),
            confidence=0.63,
            method="static-analysis",
        ),
    )
    return KnowledgeGraphBuild.create(
        build_id="build-1",
        chunks=chunks,
        entities=entities,
        assertions=assertions,
    )


def _storage(tmp_path) -> NetworkXStorage:
    return NetworkXStorage(
        namespace="typed_retrieval",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=_embed,
        ),
    )


def _chunk_payload(build: KnowledgeGraphBuild) -> dict[str, dict[str, object]]:
    return {
        chunk.chunk_id: {
            "content": chunk.content,
            "file_path": chunk.source_key,
            "sidecar": {
                "build_id": build.build_id,
                "contract_digest": build.contract_digest,
                "manifest_digest": MANIFEST_DIGEST,
                "source_key": chunk.source_key,
                "source_revision": chunk.source_revision,
                "metadata": dict(chunk.metadata),
            },
        }
        for chunk in build.chunks
    }


@pytest.mark.asyncio
async def test_knowledge_graph_build_retrieval_preserves_typed_assertions(
    tmp_path,
) -> None:
    build = _build()
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert_graph_entities(
        list(build.entities), contract_digest=build.contract_digest
    )
    await storage.upsert_graph_assertions(
        list(build.assertions), contract_digest=build.contract_digest
    )
    relation_candidates = [
        {"id": "parallel-b", "assertion_id": "parallel-b", "similarity": 0.8},
        {"id": "reciprocal", "assertion_id": "reciprocal", "similarity": 0.7},
        {"id": "parallel-a", "assertion_id": "parallel-a", "similarity": 0.9},
    ]
    relationships_vdb = SimpleNamespace(
        query=AsyncMock(return_value=relation_candidates)
    )
    text_chunks = _chunk_payload(build)
    text_chunks_db = SimpleNamespace(
        get_by_ids=AsyncMock(
            side_effect=lambda ids: [text_chunks.get(chunk_id) for chunk_id in ids]
        )
    )

    result = await retrieve_typed_records(
        query="dependencies",
        mode="global",
        top_k=10,
        graph=storage,
        entities_vdb=None,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
    )

    assert [row["assertion_id"] for row in result.assertions] == [
        "parallel-a",
        "parallel-b",
        "reciprocal",
    ]
    assert [(row["src_id"], row["dst_id"]) for row in result.assertions] == [
        ("alpha", "beta"),
        ("alpha", "beta"),
        ("beta", "alpha"),
    ]
    assert [row["score"] for row in result.assertions] == [0.9, 0.8, 0.7]
    assert [row["traversal_path"] for row in result.assertions] == [
        ["alpha", "beta"],
        ["alpha", "beta"],
        ["beta", "alpha"],
    ]
    assert all(row["direction"] == "outbound" for row in result.assertions)
    assert result.assertions[0]["predicate"] == "references"
    assert result.assertions[0]["confidence"] == 0.91
    assert result.assertions[0]["method"] == "static-analysis"
    assert result.assertions[0]["contract_digest"] == build.contract_digest
    assert result.assertions[0]["evidence"][0]["source_revision"] == (
        "source-revision-1"
    )
    assert {chunk["chunk_id"] for chunk in result.chunks} == {"chunk-a", "chunk-b"}
    assert all(
        chunk["source_revision"] == "source-revision-1" for chunk in result.chunks
    )
    assert all(
        chunk["source_key"] in {"src/a.py", "src/b.py"} for chunk in result.chunks
    )
    assert all(chunk["manifest_digest"] == MANIFEST_DIGEST for chunk in result.chunks)
    graph_response = validate_typed_graph_response(
        await storage.get_knowledge_graph("*", max_depth=3, max_nodes=10)
    )
    assert [edge["id"] for edge in graph_response["edges"]] == [
        "parallel-a",
        "parallel-b",
        "reciprocal",
    ]
    assert all(edge["type"] == "ASSERTION" for edge in graph_response["edges"])
    await storage.finalize()


@pytest.mark.asyncio
async def test_claims_and_citations_are_stable_when_candidates_are_shuffled(
    tmp_path,
) -> None:
    build = _build()
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert_graph_entities(
        list(build.entities), contract_digest=build.contract_digest
    )
    await storage.upsert_graph_assertions(
        list(build.assertions), contract_digest=build.contract_digest
    )
    candidates = [
        {
            "id": assertion.assertion_id,
            "assertion_id": assertion.assertion_id,
            "similarity": score,
        }
        for assertion, score in zip(build.assertions, (0.9, 0.8, 0.7))
    ]
    payload = _chunk_payload(build)
    text_chunks_db = SimpleNamespace(
        get_by_ids=AsyncMock(
            side_effect=lambda ids: [payload.get(item) for item in ids]
        )
    )

    first = await retrieve_typed_records(
        query="dependencies",
        mode="global",
        top_k=10,
        graph=storage,
        entities_vdb=None,
        relationships_vdb=SimpleNamespace(query=AsyncMock(return_value=candidates)),
        text_chunks_db=text_chunks_db,
    )
    second = await retrieve_typed_records(
        query="dependencies",
        mode="global",
        top_k=10,
        graph=storage,
        entities_vdb=None,
        relationships_vdb=SimpleNamespace(
            query=AsyncMock(return_value=list(reversed(candidates)))
        ),
        text_chunks_db=text_chunks_db,
    )

    assert first.citations == second.citations
    assert first.claims == second.claims
    assert all(len(claim["record_ids"]) == 1 for claim in first.claims)
    assert all(claim["citation_ids"] for claim in first.claims)
    assert all(
        citation["citation_id"].startswith("citation:") for citation in first.citations
    )
    claims_by_kind = {
        kind: [claim for claim in first.claims if claim["kind"] == kind]
        for kind in {claim["kind"] for claim in first.claims}
    }
    assert set(claims_by_kind) == {
        "assertion",
        "identity",
        "jurisdiction",
        "provenance",
    }
    assert len(claims_by_kind["assertion"]) == 3
    assert len(claims_by_kind["identity"]) == 2
    assert claims_by_kind["jurisdiction"] == [
        {
            "kind": "jurisdiction",
            "record_ids": ["parallel-b"],
            "citation_ids": [
                next(
                    citation["citation_id"]
                    for citation in first.citations
                    if citation["chunk_id"] == "chunk-b"
                )
            ],
        }
    ]
    assert len(claims_by_kind["provenance"]) == 2
    assert all(claim["citation_ids"] for claim in first.claims)

    # The policy set is matched case-insensitively: an uppercase spelling of
    # the same predicates emits the identical jurisdiction claims.
    uppercased = await retrieve_typed_records(
        query="dependencies",
        mode="global",
        top_k=10,
        graph=storage,
        entities_vdb=None,
        relationships_vdb=SimpleNamespace(query=AsyncMock(return_value=candidates)),
        text_chunks_db=text_chunks_db,
        jurisdiction_predicates=frozenset({"LOCATED_IN", "OVERLAPS_ZONE"}),
    )
    assert [
        claim for claim in uppercased.claims if claim["kind"] == "jurisdiction"
    ] == claims_by_kind["jurisdiction"]
    await storage.finalize()


@pytest.mark.asyncio
async def test_retrieval_rejects_consistently_wrong_generation_sidecars(
    tmp_path,
) -> None:
    assert hasattr(typed_retrieval_module, "TypedRetrievalIdentity")
    assert hasattr(typed_retrieval_module, "bind_typed_retrieval_identity")
    assert hasattr(typed_retrieval_module, "reset_typed_retrieval_identity")
    TypedRetrievalIdentity = typed_retrieval_module.TypedRetrievalIdentity
    bind_typed_retrieval_identity = typed_retrieval_module.bind_typed_retrieval_identity
    reset_typed_retrieval_identity = (
        typed_retrieval_module.reset_typed_retrieval_identity
    )
    build = _build()
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert_graph_entities(
        list(build.entities), contract_digest=build.contract_digest
    )
    await storage.upsert_graph_assertions(
        list(build.assertions), contract_digest=build.contract_digest
    )
    payload = _chunk_payload(build)
    token = bind_typed_retrieval_identity(
        TypedRetrievalIdentity(
            build_id="active-build",
            contract_digest="f" * 64,
            manifest_digest="e" * 64,
            source_revision="active-source-revision",
        )
    )
    try:
        with pytest.raises(
            TypedRetrievalContractError,
            match="active generation",
        ):
            await retrieve_typed_records(
                query="dependencies",
                mode="global",
                top_k=10,
                graph=storage,
                entities_vdb=None,
                relationships_vdb=SimpleNamespace(
                    query=AsyncMock(
                        return_value=[
                            {
                                "id": assertion.assertion_id,
                                "assertion_id": assertion.assertion_id,
                                "similarity": score,
                            }
                            for assertion, score in zip(
                                build.assertions, (0.9, 0.8, 0.7)
                            )
                        ]
                    )
                ),
                text_chunks_db=SimpleNamespace(
                    get_by_ids=AsyncMock(
                        side_effect=lambda ids: [payload.get(item) for item in ids]
                    )
                ),
            )
    finally:
        reset_typed_retrieval_identity(token)
        await storage.finalize()


@pytest.mark.asyncio
async def test_retrieval_rejects_vector_candidate_from_another_generation(
    tmp_path,
) -> None:
    build = _build()
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert_graph_entities(
        list(build.entities), contract_digest=build.contract_digest
    )
    await storage.upsert_graph_assertions(
        list(build.assertions), contract_digest=build.contract_digest
    )
    payload = _chunk_payload(build)
    token = typed_retrieval_module.bind_typed_retrieval_identity(
        typed_retrieval_module.TypedRetrievalIdentity(
            build_id=build.build_id,
            contract_digest=build.contract_digest,
            manifest_digest="e" * 64,
            source_revision="source-revision-1",
        )
    )
    try:
        with pytest.raises(
            TypedRetrievalContractError,
            match="candidate build_id does not match the active generation",
        ):
            await retrieve_typed_records(
                query="dependencies",
                mode="global",
                top_k=10,
                graph=storage,
                entities_vdb=None,
                relationships_vdb=SimpleNamespace(
                    query=AsyncMock(
                        return_value=[
                            {
                                "id": "parallel-a",
                                "assertion_id": "parallel-a",
                                "build_id": "other-build",
                                "contract_digest": "f" * 64,
                                "evidence": [
                                    {
                                        "chunk_id": "chunk-a",
                                        "source_key": "src/a.py",
                                        "source_revision": "other-revision",
                                        "metadata": {},
                                    }
                                ],
                                "similarity": 0.9,
                            }
                        ]
                    )
                ),
                text_chunks_db=SimpleNamespace(
                    get_by_ids=AsyncMock(
                        side_effect=lambda ids: [payload.get(item) for item in ids]
                    )
                ),
            )
    finally:
        typed_retrieval_module.reset_typed_retrieval_identity(token)
        await storage.finalize()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_location", ["candidate", "chunk"])
async def test_retrieval_rejects_wrong_manifest_sidecar(
    tmp_path,
    wrong_location: str,
) -> None:
    build = _build()
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert_graph_entities(
        list(build.entities), contract_digest=build.contract_digest
    )
    await storage.upsert_graph_assertions(
        list(build.assertions), contract_digest=build.contract_digest
    )
    payload = _chunk_payload(build)
    if wrong_location == "chunk":
        for item in payload.values():
            sidecar = item["sidecar"]
            assert isinstance(sidecar, dict)
            sidecar["manifest_digest"] = "f" * 64
    assertion = build.assertions[0]
    candidate = {
        "id": assertion.assertion_id,
        "assertion_id": assertion.assertion_id,
        "build_id": build.build_id,
        "contract_digest": build.contract_digest,
        "manifest_digest": "f" * 64
        if wrong_location == "candidate"
        else MANIFEST_DIGEST,
        "evidence": [
            {
                "chunk_id": evidence.chunk_id,
                "source_key": evidence.source_key,
                "source_revision": evidence.source_revision,
                "metadata": dict(evidence.metadata),
            }
            for evidence in assertion.evidence
        ],
        "similarity": 0.9,
    }
    token = typed_retrieval_module.bind_typed_retrieval_identity(
        typed_retrieval_module.TypedRetrievalIdentity(
            build_id=build.build_id,
            contract_digest=build.contract_digest,
            manifest_digest=MANIFEST_DIGEST,
            source_revision="source-revision-1",
        )
    )
    try:
        with pytest.raises(
            TypedRetrievalContractError,
            match=rf"{wrong_location} manifest_digest does not match the active generation",
        ):
            await retrieve_typed_records(
                query="dependencies",
                mode="global",
                top_k=10,
                graph=storage,
                entities_vdb=None,
                relationships_vdb=SimpleNamespace(
                    query=AsyncMock(return_value=[candidate])
                ),
                text_chunks_db=SimpleNamespace(
                    get_by_ids=AsyncMock(
                        side_effect=lambda ids: [payload.get(item) for item in ids]
                    )
                ),
            )
    finally:
        typed_retrieval_module.reset_typed_retrieval_identity(token)
        await storage.finalize()


@pytest.mark.asyncio
async def test_local_traversal_keeps_parallel_and_reciprocal_direction(
    tmp_path,
) -> None:
    build = _build()
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert_graph_entities(
        list(build.entities), contract_digest=build.contract_digest
    )
    await storage.upsert_graph_assertions(
        list(build.assertions), contract_digest=build.contract_digest
    )
    payload = _chunk_payload(build)

    result = await retrieve_typed_records(
        query="alpha",
        mode="local",
        top_k=10,
        graph=storage,
        entities_vdb=SimpleNamespace(
            query=AsyncMock(
                return_value=[
                    {
                        "id": "alpha",
                        "entity_id": "alpha",
                        "similarity": 0.85,
                    }
                ]
            )
        ),
        relationships_vdb=None,
        text_chunks_db=SimpleNamespace(
            get_by_ids=AsyncMock(
                side_effect=lambda ids: [payload.get(item) for item in ids]
            )
        ),
    )

    directions = {
        row["assertion_id"]: (row["direction"], row["traversal_path"])
        for row in result.assertions
    }
    assert directions == {
        "parallel-a": ("outbound", ["alpha", "beta"]),
        "parallel-b": ("outbound", ["alpha", "beta"]),
        "reciprocal": ("inbound", ["alpha", "beta"]),
    }
    await storage.finalize()


@pytest.mark.asyncio
async def test_typed_retrieval_rejects_legacy_and_placeholder_candidates(
    tmp_path,
) -> None:
    storage = _storage(tmp_path)
    await storage.initialize()
    legacy_vdb = SimpleNamespace(
        query=AsyncMock(
            return_value=[
                {"src_id": "legacy-a", "tgt_id": "legacy-b", "similarity": 0.9}
            ]
        )
    )

    with pytest.raises(TypedRetrievalContractError, match="assertion_id"):
        await retrieve_typed_records(
            query="legacy",
            mode="global",
            top_k=10,
            graph=storage,
            entities_vdb=None,
            relationships_vdb=legacy_vdb,
            text_chunks_db=SimpleNamespace(get_by_ids=AsyncMock(return_value=[])),
        )

    with pytest.raises(TypedRetrievalContractError, match="ASSERTION"):
        validate_typed_graph_response(
            {
                "nodes": [],
                "edges": [
                    {
                        "id": "legacy",
                        "type": "DIRECTED",
                        "source": "legacy-a",
                        "target": "legacy-b",
                        "properties": {"predicate": "UNKNOWN"},
                    }
                ],
                "is_truncated": False,
            }
        )
    await storage.finalize()


def test_typed_graph_response_rejects_placeholder_properties() -> None:
    with pytest.raises(TypedRetrievalContractError, match="placeholder"):
        validate_typed_graph_response(
            {
                "nodes": [],
                "edges": [
                    {
                        "id": "assertion-a",
                        "type": "ASSERTION",
                        "source": "alpha",
                        "target": "beta",
                        "properties": {
                            "_lightrag_record_kind": "GraphAssertion",
                            "assertion_id": "assertion-a",
                            "build_id": "build-1",
                            "predicate": "UNKNOWN",
                            "src_id": "alpha",
                            "dst_id": "beta",
                            "evidence": [],
                            "metadata": {},
                            "contract_digest": "a" * 64,
                        },
                    }
                ],
                "is_truncated": False,
            }
        )


@pytest.mark.asyncio
async def test_generation_query_context_uses_only_typed_structured_records(
    tmp_path,
) -> None:
    build = _build()
    storage = _storage(tmp_path)
    await storage.initialize()
    await storage.upsert_graph_entities(
        list(build.entities), contract_digest=build.contract_digest
    )
    await storage.upsert_graph_assertions(
        list(build.assertions), contract_digest=build.contract_digest
    )
    payload = _chunk_payload(build)
    fence_id = UUID("018f0f7d-c68b-7a2f-8f7d-724a24f9aa01")
    token = bind_generation_operation_fence(
        GenerationOperationFence(
            kind=GenerationFenceKind.READ,
            plane="oceanstack_product",
            generation_id=fence_id,
            workspace=generation_workspace("oceanstack_product", fence_id),
            token=fence_id,
        )
    )
    try:
        context = await _build_query_context(
            "dependencies",
            "alpha",
            "dependencies",
            storage,
            SimpleNamespace(query=AsyncMock(return_value=[])),
            SimpleNamespace(
                query=AsyncMock(
                    return_value=[
                        {
                            "id": assertion.assertion_id,
                            "assertion_id": assertion.assertion_id,
                            "similarity": score,
                        }
                        for assertion, score in zip(build.assertions, (0.9, 0.8, 0.7))
                    ]
                )
            ),
            SimpleNamespace(
                get_by_ids=AsyncMock(
                    side_effect=lambda ids: [payload.get(item) for item in ids]
                )
            ),
            QueryParam(mode="global", top_k=10),
        )
    finally:
        reset_generation_operation_fence(token)

    assert context is not None
    assert context.raw_data["data"]["claims"]
    assert "relationships" not in context.raw_data["data"]
    assert "references" not in context.raw_data["data"]
    assert [
        assertion["assertion_id"]
        for assertion in context.raw_data["data"]["assertions"]
    ] == ["parallel-a", "parallel-b", "reciprocal"]
    await storage.finalize()
