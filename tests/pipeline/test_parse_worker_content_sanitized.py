"""The parse worker must align its in-memory body with the sanitized copy
that ``_persist_parsed_full_docs`` wrote to full_docs.

A parser may return ``ParseResult(content=...)`` carrying the pre-clean text
(the legacy engine returns the raw UTF-8 extraction verbatim). The worker uses
that body for ``content_summary`` / ``content_length`` on doc_status and for
the post-parse duplicate-length check, so control chars left on it would reach
doc_status (and NUL would break PostgreSQL text writes). The worker now strips
control chars off ``parsed_data_w["content"]`` right after ``parser.parse()``,
so both full_docs and doc_status see the same cleaned body.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import strip_lightrag_doc_prefix
from tests.pipeline.conftest import build_pipeline_rag, make_deterministic_chunking

pytestmark = pytest.mark.offline


_deterministic_chunking = make_deterministic_chunking()


async def _build_rag(tmp_path) -> LightRAG:
    return await build_pipeline_rag(
        tmp_path,
        workspace=f"parseworker-{uuid4().hex[:8]}",
        chunking_func=_deterministic_chunking,
    )


def test_legacy_parse_worker_sanitizes_doc_status_and_full_docs(tmp_path, monkeypatch):
    """A PENDING_PARSE .txt whose body carries C0 separators: after the legacy
    worker runs, neither doc_status (content_summary/content_length) nor
    full_docs retains \\x1c-\\x1f / NUL, and the two agree on the cleaned body."""

    async def _run():
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setenv("INPUT_DIR", str(input_dir))
        rag = await _build_rag(tmp_path)
        try:
            dirty = "Alpha\x1cbody\x1d中\x1f文\x00 paragraph with real words."
            clean = "Alphabody中文 paragraph with real words."
            source_path = input_dir / "doc.txt"
            source_path.write_text(dirty, encoding="utf-8")

            await rag.apipeline_enqueue_documents(
                "",
                file_paths=str(source_path),
                docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                parse_engine="legacy",
            )
            doc_id = compute_mdhash_id(str(source_path), prefix="doc-")
            await rag.apipeline_process_enqueue_documents()

            status = await rag.doc_status.get_by_id(doc_id)
            assert status["status"] == DocStatus.PROCESSED
            # content_summary (a substring of the body) and the persisted body
            # must both be control-char-free.
            summary = status["content_summary"]
            assert not any(c in summary for c in "\x1c\x1d\x1f\x00")

            full = await rag.full_docs.get_by_id(doc_id)
            body = strip_lightrag_doc_prefix(full["content"], full.get("parse_format"))
            assert body == clean
            # content_length matches the cleaned, persisted body length — not
            # the longer pre-clean extraction.
            assert status["content_length"] == len(clean)
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
