"""Regression tests for the legacy ``ainsert_custom_chunks`` write ordering.

Before the fix, ``chunks_vdb.upsert``, ``_process_extract_entities``,
``full_docs.upsert``, and ``text_chunks.upsert`` all ran inside one
``asyncio.gather(*tasks)`` (no ``return_exceptions``). A failed extraction
left sibling storage writes behind, and BOTH ``filter_keys`` dedup guards
(full_docs and text_chunks) could then short-circuit every retry: the doc
stayed permanently stored with chunks but zero entities.

The fix runs extraction first and alone (it consumes the in-memory
``inserting_chunks`` dict, so no storage write is a prerequisite), then
commits ``chunks_vdb`` / ``text_chunks`` / ``full_docs`` together only after
extraction succeeds. A failed extraction therefore leaves NO storage rows
behind and a retry re-runs the full path.
"""

from __future__ import annotations

import pytest

from lightrag import LightRAG


class _StoreKV:
    """Minimal KV stand-in tracking upserts and honoring filter_keys dedup."""

    def __init__(self):
        self.upserts: list[dict] = []
        self._existing: set[str] = set()

    async def filter_keys(self, keys):
        return {key for key in keys if key not in self._existing}

    async def upsert(self, data):
        self.upserts.append(data)
        self._existing.update(data.keys())


def _bare_rag() -> LightRAG:
    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = _StoreKV()
    rag.text_chunks = _StoreKV()
    rag.chunks_vdb = _StoreKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: text.split()})()

    async def _insert_done_with_cleanup():
        return None

    rag._insert_done_with_cleanup = _insert_done_with_cleanup
    return rag


@pytest.mark.asyncio
async def test_failed_extraction_leaves_no_storage_rows():
    rag = _bare_rag()

    async def _failing_extract(chunks):
        raise RuntimeError("extraction failed")

    rag._process_extract_entities = _failing_extract

    with pytest.raises(RuntimeError, match="extraction failed"):
        await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-1")

    # No storage write may survive a failed extraction. A stray full_docs
    # row would trip the doc-level dedup guard; a stray text_chunks row
    # would trip the chunk-level guard ("All chunks are already in the
    # storage.") -- either one silently blocks every retry forever.
    assert rag.full_docs.upserts == []
    assert rag.text_chunks.upserts == []
    assert rag.chunks_vdb.upserts == []


@pytest.mark.asyncio
async def test_retry_after_failed_extraction_reextracts_and_persists():
    rag = _bare_rag()
    attempts: list[str] = []

    async def _extract(chunks):
        attempts.append("extract")
        if len(attempts) == 1:
            raise RuntimeError("extraction failed")
        return []

    rag._process_extract_entities = _extract

    with pytest.raises(RuntimeError, match="extraction failed"):
        await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-1")

    # Actual retry: the second call must re-run extraction (not be dedup
    # short-circuited by either guard) and persist everything.
    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-1")

    assert attempts == ["extract", "extract"]
    assert rag.full_docs.upserts == [
        {"doc-1": {"content": "full text", "file_path": "unknown_source"}}
    ]
    assert len(rag.text_chunks.upserts) == 1
    assert len(rag.chunks_vdb.upserts) == 1


@pytest.mark.asyncio
async def test_successful_extraction_runs_before_storage_writes():
    rag = _bare_rag()
    call_order: list[str] = []

    async def _extract(chunks):
        call_order.append("extract")
        return []

    rag._process_extract_entities = _extract

    for name, store in (
        ("full_docs", rag.full_docs),
        ("text_chunks", rag.text_chunks),
        ("chunks_vdb", rag.chunks_vdb),
    ):
        original = store.upsert

        async def _tracked(data, _name=name, _original=original):
            call_order.append(_name)
            await _original(data)

        store.upsert = _tracked

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-2")

    # Extraction strictly precedes every storage write.
    assert call_order[0] == "extract"
    assert sorted(call_order[1:]) == ["chunks_vdb", "full_docs", "text_chunks"]


@pytest.mark.asyncio
async def test_extraction_failure_with_no_pipeline_lock_raises_original_error():
    """The legacy path passes pipeline_status_lock=None; the error handler
    must not mask the extraction error with `async with None` TypeError."""
    rag = _bare_rag()

    async def _boom(*args, **kwargs):
        raise ValueError("real extraction error")

    # Patch one level deeper: the real _process_extract_entities catches,
    # logs, and re-raises; with no lock it must re-raise ValueError, not
    # TypeError.
    import lightrag.lightrag as lightrag_mod

    original = lightrag_mod.extract_entities
    lightrag_mod.extract_entities = _boom
    rag.llm_response_cache = None
    rag._build_global_config = lambda: {}
    try:
        with pytest.raises(ValueError, match="real extraction error"):
            await LightRAG._process_extract_entities(rag, {"chunk-1": {}})
    finally:
        lightrag_mod.extract_entities = original
