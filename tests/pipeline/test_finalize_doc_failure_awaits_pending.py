"""Regression test for ``_finalize_doc_failure`` awaiting pending tasks.

ITEM 2: the failure epilogue cancels the sibling first-stage tasks but must
also *await* them (via ``asyncio.gather(..., return_exceptions=True)``) before
writing the FAILED ``doc_status`` row. Without the await, two hazards remain:

* a just-cancelled sibling is not yet ``done()`` when the FAILED upsert runs,
  so a detached PROCESSING write could still land *after* the FAILED row; and
* a sibling that already raised has its exception never retrieved, producing an
  "exception was never retrieved" loop warning.

This test drives ``_finalize_doc_failure`` directly, snapshots every pending
task's ``done()`` state at the moment the FAILED upsert fires, and asserts the
sibling exception was consumed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from lightrag.base import DocProcessingStatus, DocStatus
from tests.pipeline.conftest import build_pipeline_rag

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_finalize_doc_failure_awaits_pending_before_failed_upsert(tmp_path):
    rag = await build_pipeline_rag(
        tmp_path,
        workspace="finalize-awaits-pending",
        working_dir=str(tmp_path / "wd"),
    )
    try:
        now = datetime.now(timezone.utc).isoformat()
        status_doc = DocProcessingStatus(
            content_summary="doc",
            content_length=3,
            file_path="doc.pdf",
            status=DocStatus.PROCESSING,
            created_at=now,
            updated_at=now,
        )

        events: list[tuple[str, tuple[bool, ...] | None]] = []

        # Two sibling tasks: one raises immediately (its exception must be
        # retrieved by the gather), one blocks forever (must be cancelled AND
        # awaited to completion before the FAILED upsert). A ``None`` entry
        # exercises the ``if t is not None`` filter.
        async def _raise() -> None:
            raise ValueError("sibling boom")

        async def _block() -> None:
            await asyncio.Event().wait()

        t_raise = asyncio.create_task(_raise())
        t_block = asyncio.create_task(_block())
        # Let the raising task run to completion so it is already finished
        # (with an unretrieved exception) when the epilogue begins.
        await asyncio.sleep(0)
        pending_tasks = [t_raise, t_block, None]

        async def _spy_upsert(*args, **kwargs) -> None:
            events.append(
                ("upsert", tuple(t.done() for t in pending_tasks if t is not None))
            )

        async def _noop_persist(**kwargs) -> None:
            events.append(("persist", None))

        rag._upsert_doc_status_transition = _spy_upsert  # type: ignore[method-assign]
        rag._persist_llm_response_cache_best_effort = (  # type: ignore[method-assign]
            _noop_persist
        )

        # Capture any "exception was never retrieved" style unhandled errors.
        loop = asyncio.get_running_loop()
        unhandled: list[dict] = []
        loop.set_exception_handler(lambda _loop, ctx: unhandled.append(ctx))

        await rag._finalize_doc_failure(
            doc_id="doc-1",
            status_doc=status_doc,
            file_path="doc.pdf",
            error=RuntimeError("extract stage failed"),
            stage_label="extract",
            current_file_number=1,
            total_files=1,
            failed_chunks_snapshot=([], 0),
            pending_tasks=pending_tasks,
            metadata_extra={},
            pipeline_status={"latest_message": "", "history_messages": []},
            pipeline_status_lock=asyncio.Lock(),
        )

        # The FAILED upsert must observe every pending sibling already done.
        upsert_events = [e for e in events if e[0] == "upsert"]
        assert len(upsert_events) == 1
        _, done_snapshot = upsert_events[0]
        assert done_snapshot == (True, True), (
            "pending siblings must be awaited/cancelled before the FAILED upsert"
        )

        # The blocking sibling was cancelled and its cancellation awaited.
        assert t_block.cancelled()
        # The raising sibling's exception was retrieved by the gather.
        assert t_raise.done() and not t_raise.cancelled()
        assert isinstance(t_raise.exception(), ValueError)

        # No unhandled "exception never retrieved" surfaced from the siblings.
        assert not any(
            "never retrieved" in str(ctx.get("message", "")) for ctx in unhandled
        )
    finally:
        await rag.finalize_storages()
