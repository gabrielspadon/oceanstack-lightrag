"""Serialization semantics of the retained ``DocStatusResponse`` model.

The paginated-documents HTTP endpoint was part of the removed document router
and is gone. ``DocStatusResponse`` itself is retained (it is the serialization
model the pipeline/other modules still import), so its internal-metadata
stripping contract is exercised here by constructing the model directly.
"""

import importlib
import sys

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
sys.argv = _original_argv

DocStatusResponse = _document_routes.DocStatusResponse
DocStatus = _base.DocStatus

pytestmark = pytest.mark.offline


def _doc_status_response(metadata):
    return DocStatusResponse(
        id="doc-1",
        content_summary="s",
        content_length=1,
        status=DocStatus.PROCESSED,
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        file_path="x.pdf",
        metadata=metadata,
    )


def test_docstatusresponse_strips_internal_key_keeps_others():
    resp = _doc_status_response(
        {"smartheading_llm_cache_ids": ["cache-1"], "parse_engine": "native"}
    )
    assert resp.metadata == {"parse_engine": "native"}
    assert "smartheading_llm_cache_ids" not in resp.metadata


def test_docstatusresponse_metadata_none_passes_through():
    assert _doc_status_response(None).metadata is None


def test_docstatusresponse_does_not_mutate_source_metadata():
    """The source dict is shared with the deletion path / carry-over, so the
    validator must copy-then-strip, never mutate in place."""
    source = {"smartheading_llm_cache_ids": ["cache-1"], "parse_engine": "native"}
    resp = _doc_status_response(source)
    assert resp.metadata == {"parse_engine": "native"}
    assert source == {
        "smartheading_llm_cache_ids": ["cache-1"],
        "parse_engine": "native",
    }
