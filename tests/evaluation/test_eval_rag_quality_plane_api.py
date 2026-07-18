"""Regression tests for eval_rag_quality.py's plane-scoped API port (P3-i).

The RAGAS eval script previously POSTed to {rag_api_url}/query, an endpoint
not mounted by the server; the live surface is /planes/{plane}/query
(lightrag/api/routers/plane_routes.py). These tests exercise the pure
URL/payload/response-parsing helpers offline, without any live HTTP call.
"""

import pytest

from lightrag.evaluation.eval_rag_quality import (
    PLANE_CHOICES,
    build_plane_query_url,
    build_query_payload,
    extract_contexts_from_citations,
)


def test_plane_choices_match_router_literal():
    # Mirrors the `Plane` literal in lightrag/api/routers/plane_routes.py.
    assert PLANE_CHOICES == (
        "oceanstack_dev",
        "oceanstack_product",
        "oceanstack_maritime",
    )


@pytest.mark.parametrize("plane", PLANE_CHOICES)
def test_build_plane_query_url_targets_plane_scoped_endpoint(plane):
    url = build_plane_query_url("http://localhost:9621", plane)
    assert url == f"http://localhost:9621/planes/{plane}/query"


def test_build_plane_query_url_strips_trailing_slash():
    url = build_plane_query_url("http://localhost:9621/", "oceanstack_dev")
    assert url == "http://localhost:9621/planes/oceanstack_dev/query"


def test_build_query_payload_matches_plane_query_request_schema():
    payload = build_query_payload("what is x?", 7)
    # PlaneQueryRequest has extra="forbid"; every key here must be a real
    # field on that model (query, mode, include_references,
    # include_chunk_content, response_type, top_k, ...).
    assert payload == {
        "query": "what is x?",
        "mode": "mix",
        "include_references": True,
        "include_chunk_content": True,
        "response_type": "Multiple Paragraphs",
        "top_k": 7,
    }


def test_extract_contexts_from_citations_flattens_list_content():
    citations = [
        {
            "citation_id": "c1",
            "chunk_id": "chunk-1",
            "source_key": "doc1",
            "source_revision": "r1",
            "content": ["chunk a", "chunk b"],
        },
        {
            "citation_id": "c2",
            "chunk_id": "chunk-2",
            "source_key": "doc2",
            "source_revision": "r1",
            "content": ["chunk c"],
        },
    ]
    assert extract_contexts_from_citations(citations) == [
        "chunk a",
        "chunk b",
        "chunk c",
    ]


def test_extract_contexts_from_citations_skips_none_content():
    citations = [
        {"content": None},
        {"content": ["kept"]},
    ]
    assert extract_contexts_from_citations(citations) == ["kept"]


def test_extract_contexts_from_citations_handles_missing_content_key():
    citations = [{"citation_id": "c1"}]
    assert extract_contexts_from_citations(citations) == []


def test_extract_contexts_from_citations_empty_list_returns_empty():
    assert extract_contexts_from_citations([]) == []
