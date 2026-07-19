import sys

import pytest

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    normalize_file_path,
    pipeline_index_texts,
)
from lightrag.base import DocStatus  # noqa: E402
from lightrag.constants import PROCESS_OPTION_CHUNK_FIXED  # noqa: E402
from lightrag.pipeline import _PipelineMixin  # noqa: E402


class DummyRAG:
    def __init__(self):
        self.enqueued_calls = []
        self.processed = False
        # _resolve_text_chunking reads addon_params; {} -> default chunker config.
        self.addon_params = {}

    async def apipeline_enqueue_documents(
        self,
        input,
        file_paths=None,
        track_id=None,
        process_options=None,
        chunk_options=None,
    ):
        self.enqueued_calls.append(
            {
                "input": input,
                "file_paths": file_paths,
                "track_id": track_id,
                "process_options": process_options,
                "chunk_options": chunk_options,
            }
        )

    async def apipeline_process_enqueue_documents(self):
        self.processed = True


class CaptureDocStatus:
    def __init__(self):
        self.upserts = []

    async def upsert(self, data):
        self.upserts.append(data)


class DummyPipeline(_PipelineMixin):
    def __init__(self):
        self.doc_status = CaptureDocStatus()


@pytest.mark.asyncio
async def test_pipeline_index_texts_rejects_missing_file_sources():
    rag = DummyRAG()

    with pytest.raises(ValueError, match="valid file source"):
        await pipeline_index_texts(
            rag,
            texts=["alpha"],
            file_sources=[None],
            track_id="track-1",
        )

    assert rag.enqueued_calls == []
    assert rag.processed is False


@pytest.mark.asyncio
async def test_pipeline_index_texts_preserves_caller_owned_file_sources():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha"],
        file_sources=["/tmp/source/alpha.[native-iet].txt"],
        track_id="track-1",
    )

    assert len(rag.enqueued_calls) == 1
    call = rag.enqueued_calls[0]
    assert call["input"] == ["alpha"]
    # Hint stripped, caller-owned path structure preserved.
    assert call["file_paths"] == ["/tmp/source/alpha.txt"]
    assert call["track_id"] == "track-1"
    assert call["process_options"] == PROCESS_OPTION_CHUNK_FIXED
    # No chunking config supplied -> default F snapshot from addon_params.
    assert isinstance(call["chunk_options"], dict)
    assert "fixed_token" in call["chunk_options"]
    assert rag.processed is True


def test_doc_status_response_uses_non_null_unknown_source():
    response = DocStatusResponse(
        id="doc-1",
        content_summary="summary",
        content_length=5,
        status=DocStatus.PENDING,
        created_at="2026-03-19T00:00:00+00:00",
        updated_at="2026-03-19T00:00:00+00:00",
        file_path=normalize_file_path(None),
    )

    assert response.file_path == "unknown_source"


@pytest.mark.asyncio
async def test_error_document_enqueue_canonicalizes_file_path_before_upsert():
    rag = DummyPipeline()

    await rag.apipeline_enqueue_error_documents(
        [
            {
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
                "error_description": "bad file",
                "original_error": "parse failed",
            }
        ],
        track_id="track-1",
    )

    saved = next(iter(rag.doc_status.upserts[0].values()))
    # Hint stripped, caller-owned path structure preserved.
    assert saved["file_path"] == "/tmp/uploads/report.pdf"


@pytest.mark.offline
def test_ainsert_canonicalizes_missing_file_source_to_unknown_source(tmp_path):
    """Ported from the removed ``ainsert_custom_chunks`` normalization test.

    Inserting a document with no file source must persist the canonical
    ``"unknown_source"`` sentinel as its ``file_path``. ``ainsert`` delegates
    this canonicalization to ``apipeline_enqueue_documents``; extraction is
    isolated out so only the enqueue-time normalization is exercised.
    """
    import asyncio
    from unittest.mock import AsyncMock

    import numpy as np

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from tests.conftest import make_char_tokenizer

    async def _mock_embedding(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 32), dtype=np.float32)

    async def _mock_llm(prompt, **kwargs) -> str:
        return ""

    async def _run():
        rag = LightRAG(
            working_dir=str(tmp_path),
            workspace=f"ainsert-fp-{tmp_path.name}",
            llm_model_func=_mock_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=32,
                max_token_size=4096,
                func=_mock_embedding,
            ),
            tokenizer=make_char_tokenizer("mock-tokenizer"),
        )
        await rag.initialize_storages()
        try:
            # Isolate the enqueue-time canonicalization from extraction.
            rag.apipeline_process_enqueue_documents = AsyncMock()
            await rag.ainsert("full text", ids="doc-1", file_paths=None)
            return await rag.full_docs.get_by_id("doc-1")
        finally:
            await rag.finalize_storages()

    row = asyncio.run(_run())
    assert row is not None
    assert row["file_path"] == "unknown_source"
