"""Regression test for the batched full_docs lookup in the consistency check.

ITEM 7: ``_validate_and_fix_document_consistency`` used to issue an N+1 series
of ``full_docs.get_by_id`` calls (one per document, twice over). It now batches
each sweep into a single ``full_docs.get_by_ids`` call, which every KV backend
implements order-preserving with ``None`` padding for missing ids.

This test spies on both methods to assert the batched path is taken (get_by_ids
used, get_by_id never called on full_docs), verifies the ordered/None-padded
contract, and confirms behaviour is unchanged: a document missing from
full_docs is treated as inconsistent and deleted, while a present one is reset
to PENDING.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from lightrag.base import DocStatus
from tests.pipeline.conftest import build_pipeline_rag

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_consistency_check_batches_full_docs_lookups(tmp_path, monkeypatch):
    rag = await build_pipeline_rag(
        tmp_path,
        workspace="consistency-batched",
        working_dir=str(tmp_path / "wd"),
    )
    try:
        present_id = "doc-present"
        missing_id = "doc-missing"  # no full_docs content -> inconsistent
        now = datetime.now(timezone.utc).isoformat()

        await rag.full_docs.upsert(
            {present_id: {"content": "present body", "file_path": "present.pdf"}}
        )
        for doc_id, fp in ((present_id, "present.pdf"), (missing_id, "missing.pdf")):
            await rag.doc_status.upsert(
                {
                    doc_id: {
                        "status": DocStatus.PROCESSING,
                        "content_summary": doc_id,
                        "content_length": 5,
                        "created_at": now,
                        "updated_at": now,
                        "file_path": fp,
                    }
                }
            )

        counts = {"get_by_ids": 0, "get_by_id": 0}
        real_get_by_ids = rag.full_docs.get_by_ids
        real_get_by_id = rag.full_docs.get_by_id

        async def _spy_get_by_ids(ids):
            counts["get_by_ids"] += 1
            result = await real_get_by_ids(ids)
            # Order-preserving with None padding: one slot per requested id.
            assert len(result) == len(ids)
            return result

        async def _spy_get_by_id(doc_id):
            counts["get_by_id"] += 1
            return await real_get_by_id(doc_id)

        monkeypatch.setattr(rag.full_docs, "get_by_ids", _spy_get_by_ids)
        monkeypatch.setattr(rag.full_docs, "get_by_id", _spy_get_by_id)

        to_process = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSING)
        assert set(to_process.keys()) == {present_id, missing_id}

        result = await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process,
            pipeline_status={"latest_message": "", "history_messages": []},
            pipeline_status_lock=asyncio.Lock(),
        )

        # Batched path taken: get_by_ids used (two sweeps), get_by_id never.
        assert counts["get_by_ids"] >= 1
        assert counts["get_by_id"] == 0

        # Behaviour preserved: missing doc deleted, present doc reset to PENDING.
        assert missing_id not in result
        assert await rag.doc_status.get_by_id(missing_id) is None
        assert present_id in result
        present_stored = await rag.doc_status.get_by_id(present_id)
        assert present_stored is not None
        assert present_stored["status"] in (DocStatus.PENDING, DocStatus.PENDING.value)
    finally:
        await rag.finalize_storages()
