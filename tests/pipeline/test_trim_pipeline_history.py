"""Regression tests for ``_trim_pipeline_history`` (ITEM 6).

The helper bounds ``pipeline_status['history_messages']`` growth. It MUST trim
in place (``del history[:-keep]``) and never reassign the list, because that
list is a ``Manager.list``-backed shared object; a reassignment would sever the
cross-process binding. It is also lock-agnostic (callers already hold the
pipeline lock), so it must not acquire any lock itself.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from lightrag.base import DocStatus
from lightrag.pipeline import _trim_pipeline_history
from tests.pipeline.conftest import build_pipeline_rag

pytestmark = pytest.mark.offline


def test_trim_in_place_preserves_identity_and_newest_messages():
    history = [str(i) for i in range(10001)]
    pipeline_status = {"history_messages": history}

    _trim_pipeline_history(pipeline_status)

    # Same list object (never reassigned) so Manager.list binding survives.
    assert pipeline_status["history_messages"] is history
    assert len(history) == 5000
    # The newest ``keep`` messages are the ones retained.
    assert history[0] == "5001"
    assert history[-1] == "10000"


def test_trim_below_cap_leaves_list_untouched():
    history = [str(i) for i in range(10)]
    pipeline_status = {"history_messages": history}

    _trim_pipeline_history(pipeline_status)

    assert pipeline_status["history_messages"] is history
    assert history == [str(i) for i in range(10)]


def test_trim_at_cap_boundary_is_noop():
    history = [str(i) for i in range(10000)]
    pipeline_status = {"history_messages": history}

    _trim_pipeline_history(pipeline_status)

    # Exactly ``cap`` (not > cap) must not trim.
    assert len(history) == 10000
    assert pipeline_status["history_messages"] is history


def test_trim_missing_key_is_noop():
    # Must not raise when history_messages is absent or None.
    _trim_pipeline_history({})
    _trim_pipeline_history({"history_messages": None})


@pytest.mark.asyncio
async def test_reset_epilogue_trims_oversized_history_in_place(tmp_path):
    """A held-lock append site (the consistency-check reset epilogue) trims the
    shared history in place once it grows past the cap."""
    rag = await build_pipeline_rag(
        tmp_path,
        workspace="trim-reset-epilogue",
        working_dir=str(tmp_path / "wd"),
    )
    try:
        doc_id = "doc-interrupted"
        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {doc_id: {"content": "interrupted", "file_path": "report.pdf"}}
        )
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": "interrupted",
                    "content_length": 11,
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "report.pdf",
                }
            }
        )

        # Seed the shared history above the cap; the reset epilogue appends a
        # "Reset ... to PENDING" line and must trim in place afterwards.
        history = [f"seed-{i}" for i in range(10001)]
        pipeline_status = {"latest_message": "", "history_messages": history}

        to_process = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSING)
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        # Same object, trimmed to the ``keep`` window.
        assert pipeline_status["history_messages"] is history
        assert len(history) == 5000
    finally:
        await rag.finalize_storages()
