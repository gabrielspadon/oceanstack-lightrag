"""Read-only query and graph routes for explicit immutable graph planes."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal, Protocol, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lightrag.base import QueryParam
from lightrag.generation import (
    GraphGeneration,
    complete_cleanup_preserving_cancellation,
)
from lightrag.kg.graph_contract import is_placeholder_token
from lightrag.utils import logger
from lightrag.typed_retrieval import (
    TypedRetrievalContractError,
    TypedRetrievalIdentity,
    bind_typed_retrieval_identity,
    reset_typed_retrieval_identity,
    validate_typed_retrieval_data_identity,
    validate_typed_graph_response,
)

from ..generation_pool import (
    GenerationPoolClosedError,
    GenerationRAG,
    GenerationReadLease,
    GenerationUnavailableError,
)


Plane = Literal["oceanstack_dev", "oceanstack_product", "oceanstack_maritime"]


class ReadLeasePool(Protocol):
    async def acquire(self, plane: str) -> GenerationReadLease: ...

    async def close(self) -> None: ...


class PlaneQueryRequest(BaseModel):
    """Request body for the non-streaming ``/query`` and ``/query/data`` routes.

    Has no ``stream`` field: both routes always run with ``stream=False``, so
    accepting one from the client would be silently ignored. Streaming lives
    on ``PlaneStreamQueryRequest`` instead.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3)
    mode: Literal["local", "global", "hybrid", "mix"] = "mix"
    only_need_context: bool | None = None
    only_need_prompt: bool | None = None
    response_type: str | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    chunk_top_k: int | None = Field(default=None, ge=1)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] | None = None
    user_prompt: str | None = None
    enable_rerank: bool | None = None
    include_references: bool = True
    # Response shaping only: gates whether chunk content is stitched into
    # citations in this router; never passed through to QueryParam.
    include_chunk_content: bool = False

    @field_validator("query", mode="after")
    @classmethod
    def strip_query(cls, query: str) -> str:
        return query.strip()

    @field_validator("conversation_history", mode="after")
    @classmethod
    def validate_history(
        cls, history: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if history is None:
            return None
        for message in history:
            role = message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Each message must have a non-empty role")
        return history

    def to_query_param(self, *, stream: bool) -> QueryParam:
        values = self.model_dump(
            exclude_none=True,
            exclude={"query", "include_chunk_content", "stream"},
        )
        values["stream"] = stream
        return QueryParam(**values)


class PlaneStreamQueryRequest(PlaneQueryRequest):
    """Request body for the streaming ``/query/stream`` route.

    Adds the ``stream`` field back on top of ``PlaneQueryRequest``; only this
    route honors a client-supplied streaming preference.
    """

    stream: bool | None = None


class CitationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chunk_id: str
    source_key: str
    source_revision: str
    content: list[str] | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str
    citations: list[CitationItem] | None = None


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_key: str
    source_revision: str
    metadata: dict[str, Any]


class TypedEntityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_id: str
    entity_id: str
    entity_type: str
    evidence: list[EvidenceItem]
    metadata: dict[str, Any]
    observed_from: datetime | None = None
    observed_to: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    contract_digest: str


class RetrievedEntity(TypedEntityRecord):
    score: float
    traversal_path: list[str]


class TypedAssertionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_id: str
    assertion_id: str
    predicate: str
    src_id: str
    dst_id: str
    evidence: list[EvidenceItem]
    metadata: dict[str, Any]
    confidence: float | None = None
    method: str | None = None
    observed_from: datetime | None = None
    observed_to: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    contract_digest: str


class RetrievedAssertion(TypedAssertionRecord):
    direction: Literal["outbound", "inbound"]
    score: float
    traversal_path: list[str]


class TypedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chunk_id: str
    source_key: str
    source_revision: str
    build_id: str
    contract_digest: str
    manifest_digest: str
    content: str
    metadata: dict[str, Any]


class ClaimItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["assertion", "identity", "jurisdiction", "provenance"]
    record_ids: list[str]
    citation_ids: list[str]


class QueryData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[RetrievedEntity]
    assertions: list[RetrievedAssertion]
    chunks: list[TypedChunk]
    citations: list[CitationItem]
    claims: list[ClaimItem]


class QueryDataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    message: str
    data: QueryData
    metadata: dict[str, Any]


class GraphEntityProperties(TypedEntityRecord):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    record_kind: Literal["GraphEntity"] = Field(alias="_lightrag_record_kind")


class GraphAssertionProperties(TypedAssertionRecord):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    record_kind: Literal["GraphAssertion"] = Field(alias="_lightrag_record_kind")


class GraphNodeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    labels: list[str]
    properties: GraphEntityProperties


class GraphEdgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["ASSERTION"]
    source: str
    target: str
    properties: GraphAssertionProperties


class KnowledgeGraphResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    is_truncated: bool


class GenerationProvenance(BaseModel):
    plane: str
    generation_id: str
    build_id: str
    source_revision: str
    manifest_digest: str

    @classmethod
    def from_generation(cls, generation: GraphGeneration) -> GenerationProvenance:
        source_revision = generation.metadata.get("source_revision")
        if not isinstance(source_revision, str) or not source_revision.strip():
            raise ValueError("active generation lacks source_revision provenance")
        return cls(
            plane=generation.plane,
            generation_id=str(generation.generation_id),
            build_id=generation.build_id,
            source_revision=source_revision,
            manifest_digest=generation.manifest_digest,
        )

    def headers(self) -> dict[str, str]:
        return {
            "X-LightRAG-Plane": self.plane,
            "X-LightRAG-Generation-Id": self.generation_id,
            "X-LightRAG-Build-Id": self.build_id,
            "X-LightRAG-Source-Revision": self.source_revision,
            "X-LightRAG-Manifest-Digest": self.manifest_digest,
        }


def _enrich_citations(
    citations: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    contents: dict[str, list[str]] = {}
    for chunk in chunks:
        citation_id = chunk.get("citation_id")
        content = chunk.get("content")
        if isinstance(citation_id, str) and isinstance(content, str):
            contents.setdefault(citation_id, []).append(content)
    return [
        {**citation, "content": contents[citation["citation_id"]]}
        if citation.get("citation_id") in contents
        else citation
        for citation in citations
    ]


_ModelT = TypeVar("_ModelT", bound=BaseModel)

# Shared detail text for every "records escaped the active generation" 503,
# whether raised from the retrieval operation itself or from post-hoc
# response validation against the typed contract.
_GENERATION_IDENTITY_VIOLATION_DETAIL = (
    "generation query returned records outside the active generation"
)


def _provenance(lease: GenerationReadLease) -> GenerationProvenance:
    try:
        return GenerationProvenance.from_generation(lease.generation)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _retrieval_identity(lease: GenerationReadLease) -> TypedRetrievalIdentity:
    try:
        return TypedRetrievalIdentity.from_generation(lease.generation)
    except TypedRetrievalContractError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _stamp_provenance(response: Response, provenance: GenerationProvenance) -> None:
    response.headers.update(provenance.headers())


async def _run_with_retrieval_identity(
    lease: GenerationReadLease,
    identity: TypedRetrievalIdentity,
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    token = bind_typed_retrieval_identity(identity)
    try:
        return await lease.run(operation)
    except TypedRetrievalContractError as exc:
        raise HTTPException(
            status_code=503,
            detail=_GENERATION_IDENTITY_VIOLATION_DETAIL,
        ) from exc
    finally:
        reset_typed_retrieval_identity(token)


def _validate_typed_generation_response(
    model_cls: type[_ModelT],
    raw: Any,
    identity: TypedRetrievalIdentity,
    *,
    typed_data: Callable[[_ModelT], Any] | None = None,
) -> _ModelT:
    """Validate a typed response payload and enforce single-generation identity.

    ``typed_data`` extracts the identity-checked payload from the validated
    model when it is not the model itself (for example ``QueryDataResponse``
    checks its ``.data`` field).
    """
    try:
        validated = model_cls.model_validate(raw)
        validate_typed_retrieval_data_identity(
            typed_data(validated) if typed_data is not None else validated, identity
        )
        return validated
    except (TypedRetrievalContractError, ValidationError) as exc:
        raise HTTPException(
            status_code=503,
            detail=_GENERATION_IDENTITY_VIOLATION_DETAIL,
        ) from exc


async def _acquire(pool: ReadLeasePool, plane: Plane) -> GenerationReadLease:
    try:
        return await pool.acquire(plane)
    except (GenerationUnavailableError, GenerationPoolClosedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def create_plane_routes(
    pool: ReadLeasePool,
    api_key: str | None = None,
    *,
    auth_dependency: Callable[..., Any] | None = None,
) -> APIRouter:
    """Create the complete public read surface for immutable graph planes."""
    if auth_dependency is None:
        from ..utils_api import get_combined_auth_dependency

        auth_dependency = get_combined_auth_dependency(api_key)
    dependencies = [Depends(auth_dependency)]
    router = APIRouter(tags=["planes"])

    @router.post(
        "/planes/{plane}/query",
        response_model=QueryResponse,
        dependencies=dependencies,
    )
    async def query_plane(plane: Plane, request: PlaneQueryRequest, response: Response):
        lease = await _acquire(pool, plane)
        async with lease:
            provenance = _provenance(lease)
            identity = _retrieval_identity(lease)
            result = await _run_with_retrieval_identity(
                lease,
                identity,
                lambda: lease.rag.aquery_llm(
                    request.query, param=request.to_query_param(stream=False)
                ),
            )
            data = result.get("data", {})
            typed_data = _validate_typed_generation_response(QueryData, data, identity)
            citations = [citation.model_dump() for citation in typed_data.citations]
            if request.include_references and request.include_chunk_content:
                citations = _enrich_citations(
                    citations,
                    [chunk.model_dump() for chunk in typed_data.chunks],
                )
            _stamp_provenance(response, provenance)
            content = result.get("llm_response", {}).get("content")
            return QueryResponse(
                response=content or "No relevant context found for the query.",
                citations=citations if request.include_references else None,
            )

    @router.post(
        "/planes/{plane}/query/data",
        response_model=QueryDataResponse,
        dependencies=dependencies,
    )
    async def query_plane_data(
        plane: Plane, request: PlaneQueryRequest, response: Response
    ):
        lease = await _acquire(pool, plane)
        async with lease:
            provenance = _provenance(lease)
            identity = _retrieval_identity(lease)
            result = await _run_with_retrieval_identity(
                lease,
                identity,
                lambda: lease.rag.aquery_data(
                    request.query, param=request.to_query_param(stream=False)
                ),
            )
            _stamp_provenance(response, provenance)
            return _validate_typed_generation_response(
                QueryDataResponse, result, identity, typed_data=lambda item: item.data
            )

    @router.post(
        "/planes/{plane}/query/stream",
        dependencies=dependencies,
        responses={
            200: {
                "content": {
                    "application/x-ndjson": {
                        "schema": {"type": "string", "format": "ndjson"}
                    }
                }
            }
        },
    )
    async def stream_plane_query(plane: Plane, request: PlaneStreamQueryRequest):
        lease = await _acquire(pool, plane)
        try:
            provenance = _provenance(lease)
            identity = _retrieval_identity(lease)
            result = await _run_with_retrieval_identity(
                lease,
                identity,
                lambda: lease.rag.aquery_llm(
                    request.query,
                    param=request.to_query_param(
                        stream=request.stream if request.stream is not None else True
                    ),
                ),
            )
            typed_data = _validate_typed_generation_response(
                QueryData, result.get("data", {}), identity
            )
        except BaseException:
            await lease.close()
            raise

        async def records():
            try:
                yield json.dumps({"generation": provenance.model_dump()}) + "\n"
                citations = [citation.model_dump() for citation in typed_data.citations]
                if request.include_references and request.include_chunk_content:
                    citations = _enrich_citations(
                        citations,
                        [chunk.model_dump() for chunk in typed_data.chunks],
                    )
                llm_response = result.get("llm_response", {})
                if request.include_references:
                    yield json.dumps({"citations": citations}) + "\n"
                if llm_response.get("is_streaming"):
                    iterator = llm_response.get("response_iterator")
                    if iterator is not None:
                        async for chunk in iterator:
                            if chunk:
                                yield json.dumps({"response": chunk}) + "\n"
                else:
                    content = llm_response.get("content")
                    yield (
                        json.dumps(
                            {
                                "response": content
                                or "No relevant context found for the query."
                            }
                        )
                        + "\n"
                    )
            except Exception as exc:
                logger.error("Plane query stream failed: %s", exc, exc_info=True)
                yield json.dumps({"error": str(exc)}) + "\n"
            finally:
                await complete_cleanup_preserving_cancellation(lease.close())

        return StreamingResponse(
            records(),
            media_type="application/x-ndjson",
            headers={
                **provenance.headers(),
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _graph_read(
        plane: Plane, operation: Callable[[GenerationRAG], Awaitable[Any]]
    ) -> tuple[Any, GenerationProvenance]:
        lease = await _acquire(pool, plane)
        try:
            provenance = _provenance(lease)
            identity = _retrieval_identity(lease)
            result = await _run_with_retrieval_identity(
                lease, identity, lambda: operation(lease.rag)
            )
            return result, provenance
        finally:
            await lease.close()

    async def _typed_graph_result(rag: GenerationRAG, raw_graph: Any) -> dict[str, Any]:
        try:
            return validate_typed_graph_response(raw_graph)
        except TypedRetrievalContractError:
            pass

        if hasattr(raw_graph, "model_dump"):
            raw_graph = raw_graph.model_dump()
        if not isinstance(raw_graph, dict):
            raise TypedRetrievalContractError("graph response must be an object")
        raw_nodes = raw_graph.get("nodes")
        raw_edges = raw_graph.get("edges")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise TypedRetrievalContractError("graph response requires nodes and edges")
        if any(
            not isinstance(edge, dict) or edge.get("type") != "ASSERTION"
            for edge in raw_edges
        ):
            raise TypedRetrievalContractError(
                "generation graph contains a non-ASSERTION edge"
            )

        assertion_ids = [
            _require_graph_token(edge.get("id"), "assertion_id") for edge in raw_edges
        ]
        graph_storage = rag.chunk_entity_relation_graph
        assertions = await graph_storage.get_graph_assertions(assertion_ids)
        if set(assertion_ids) != set(assertions):
            raise TypedRetrievalContractError(
                "graph response references assertions absent from typed sidecars"
            )

        entity_ids: set[str] = set()
        for node in raw_nodes:
            if not isinstance(node, dict):
                raise TypedRetrievalContractError("graph node must be an object")
            properties = node.get("properties")
            if not isinstance(properties, dict):
                raise TypedRetrievalContractError(
                    "graph node properties must be an object"
                )
            entity_ids.add(
                _require_graph_token(properties.get("entity_id"), "entity_id")
            )
        for assertion in assertions.values():
            entity_ids.add(_require_graph_token(assertion.get("src_id"), "src_id"))
            entity_ids.add(_require_graph_token(assertion.get("dst_id"), "dst_id"))

        nodes: list[dict[str, Any]] = []
        for entity_id in sorted(entity_ids):
            entity = await graph_storage.get_graph_entity(entity_id)
            if entity is None:
                raise TypedRetrievalContractError(
                    f"graph entity {entity_id!r} is absent from typed sidecars"
                )
            nodes.append(
                {
                    "id": entity_id,
                    "labels": [entity_id],
                    "properties": {
                        "_lightrag_record_kind": "GraphEntity",
                        **entity,
                    },
                }
            )
        edges = [
            {
                "id": assertion_id,
                "type": "ASSERTION",
                "source": assertions[assertion_id]["src_id"],
                "target": assertions[assertion_id]["dst_id"],
                "properties": {
                    "_lightrag_record_kind": "GraphAssertion",
                    **assertions[assertion_id],
                },
            }
            for assertion_id in assertion_ids
        ]
        return validate_typed_graph_response(
            {
                "nodes": nodes,
                "edges": edges,
                "is_truncated": bool(raw_graph.get("is_truncated", False)),
            }
        )

    def _require_graph_token(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise TypedRetrievalContractError(
                f"graph {field} must be a non-blank string"
            )
        if is_placeholder_token(value):
            raise TypedRetrievalContractError(f"graph {field} contains a placeholder")
        return value

    @router.get("/planes/{plane}/graph/label/list", dependencies=dependencies)
    async def graph_labels(
        plane: Plane,
        response: Response,
        limit: int = Query(1000, ge=1, le=10000),
    ):
        result, provenance = await _graph_read(
            plane, lambda rag: rag.get_graph_labels()
        )
        _stamp_provenance(response, provenance)
        return result[:limit]

    @router.get("/planes/{plane}/graph/label/popular", dependencies=dependencies)
    async def popular_labels(
        plane: Plane,
        response: Response,
        limit: int = Query(300, ge=1, le=1000),
    ):
        result, provenance = await _graph_read(
            plane,
            lambda rag: rag.chunk_entity_relation_graph.get_popular_labels(limit),
        )
        _stamp_provenance(response, provenance)
        return result

    @router.get("/planes/{plane}/graph/label/search", dependencies=dependencies)
    async def search_labels(
        plane: Plane,
        response: Response,
        q: str = Query(min_length=1),
        limit: int = Query(50, ge=1, le=100),
    ):
        result, provenance = await _graph_read(
            plane,
            lambda rag: rag.chunk_entity_relation_graph.search_labels(q, limit),
        )
        _stamp_provenance(response, provenance)
        return result

    @router.get(
        "/planes/{plane}/graphs",
        dependencies=dependencies,
        response_model=KnowledgeGraphResponse,
    )
    async def knowledge_graph(
        plane: Plane,
        response: Response,
        label: str = Query(min_length=1),
        max_depth: int = Query(3, ge=1, le=10),
        max_nodes: int = Query(1000, ge=1, le=10000),
    ):
        async def read_typed_graph(rag: GenerationRAG) -> dict[str, Any]:
            raw_graph = await rag.get_knowledge_graph(
                node_label=label, max_depth=max_depth, max_nodes=max_nodes
            )
            return await _typed_graph_result(rag, raw_graph)

        result, provenance = await _graph_read(
            plane,
            read_typed_graph,
        )
        _stamp_provenance(response, provenance)
        try:
            return KnowledgeGraphResponse.model_validate(result)
        except (TypedRetrievalContractError, ValidationError) as exc:
            raise HTTPException(
                status_code=503,
                detail="generation graph returned non-typed records",
            ) from exc

    @router.get("/planes/{plane}/graph/entity/exists", dependencies=dependencies)
    async def entity_exists(
        plane: Plane, response: Response, name: str = Query(min_length=1)
    ):
        result, provenance = await _graph_read(
            plane, lambda rag: rag.chunk_entity_relation_graph.has_node(name)
        )
        _stamp_provenance(response, provenance)
        return {"exists": result}

    return router
