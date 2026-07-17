"""Shared LightRAG test rig for tests/pipeline/*.

Every ``test_pipeline_*`` / ``test_doc_status_*`` / ``test_parse_*`` module in
this package used to hand-roll its own copy of a "fake LLM + fake embedding +
fake tokenizer + LightRAG builder" scaffold. The bodies were byte-identical
across most files; only workspace naming, the emitted chunk shape, and a
handful of LightRAG constructor kwargs (``max_parallel_insert``,
``vlm_process_enable``, ``role_llm_configs``) varied per file. This module
centralizes the identical parts and exposes the drift points as parameters.

Two call shapes are in play, mirroring the two clusters of pre-existing
per-file rigs:

* ``build_pipeline_rag`` (async): the ``_dummy_llm`` / ``_dummy_embedding`` /
  ``_deterministic_chunking`` cluster used by the doc-status / parse-*
  pipeline tests that drive ``apipeline_process_enqueue_documents`` directly.
  It constructs the ``LightRAG`` instance AND awaits
  ``initialize_storages()`` before returning, matching every caller's
  original ``rag = await _build_rag(...)`` usage (no follow-up
  ``initialize_storages()`` call at the call site).
* ``build_role_rag`` (sync): the ``_mock_embedding`` / ``_noop_llm`` cluster
  used by the worker-level cancellation / content-reread / analyze_multimodal
  tests, which construct ``_BatchRunContext`` directly and call
  ``initialize_storages()`` themselves at the call site (sometimes on more
  than one ``rag`` instance per test). This mirrors the original
  ``rag = _build_rag(...)`` (no ``await``) usage exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lightrag.utils import EmbeddingFunc
from tests.conftest import make_char_tokenizer

if TYPE_CHECKING:
    from lightrag import LightRAG


async def dummy_llm(*args, **kwargs) -> str:
    """Always-succeeds LLM stub shared by the doc-status / parse-* rig."""
    return "ok"


async def dummy_embedding(texts: list[str]) -> np.ndarray:
    """Deterministic 8-dim embedding stub shared by the doc-status / parse-* rig."""
    return np.ones((len(texts), 8), dtype=float)


async def mock_embedding(texts: list[str]) -> np.ndarray:
    """Random 8-dim embedding stub shared by the worker-level cancellation /
    content-reread / analyze_multimodal rig (embeddings are never asserted on
    in that cluster, so a random vector is fine)."""
    return np.random.rand(len(texts), 8)


async def noop_llm(prompt, **kwargs) -> str:  # pragma: no cover - never invoked
    """Placeholder base ``llm_model_func`` for the worker-level rig, which
    drives role-specific funcs directly and never calls the base LLM."""
    return ""


def make_deterministic_chunking(*, num_chunks: int = 1, suffix: str = "::chunk1"):
    """Return a ``chunking_func`` matching LightRAG's chunking signature that
    emits ``num_chunks`` fixed, order-indexed chunks derived from ``content``.

    ``suffix`` is appended to ``content`` for single-chunk output (pass
    ``suffix=""`` for a bare-content chunk); multi-chunk output always uses
    the historical ``::chunk{n}`` suffixes.
    """

    def _chunking(
        tokenizer,
        content: str,
        split_by_character,
        split_by_character_only: bool,
        chunk_overlap_token_size: int,
        chunk_token_size: int,
    ) -> list[dict]:
        if num_chunks == 1:
            return [
                {
                    "tokens": 1,
                    "content": f"{content}{suffix}",
                    "chunk_order_index": 0,
                }
            ]
        return [
            {
                "tokens": 1,
                "content": f"{content}::chunk{i + 1}",
                "chunk_order_index": i,
            }
            for i in range(num_chunks)
        ]

    return _chunking


async def build_pipeline_rag(
    tmp_path,
    *,
    workspace: str,
    llm_model_func=dummy_llm,
    embedding_dim: int = 8,
    max_token_size: int = 8192,
    embedding_func=dummy_embedding,
    tokenizer_model: str = "mock-tokenizer",
    chunking_func=None,
    max_parallel_insert: int = 1,
    working_dir: str | None = None,
) -> "LightRAG":
    """Build + initialize a ``LightRAG`` for the doc-status / parse-* rig.

    ``workspace`` and ``chunking_func`` are the two drift points every caller
    supplies explicitly; everything else defaults to the value every
    pre-refactor copy used and is overridable for the rare file that diverged
    (``tokenizer_model="test-tokenizer"``, a non-default
    ``max_parallel_insert``, ...).
    """
    from lightrag import LightRAG

    kwargs: dict = dict(
        working_dir=working_dir or str(tmp_path / "wd"),
        workspace=workspace,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=embedding_func,
        ),
        tokenizer=make_char_tokenizer(tokenizer_model),
        max_parallel_insert=max_parallel_insert,
    )
    if chunking_func is not None:
        kwargs["chunking_func"] = chunking_func
    rag = LightRAG(**kwargs)
    await rag.initialize_storages()
    return rag


def build_role_rag(
    tmp_path,
    *,
    workspace: str,
    llm_model_func,
    role_llm_configs: dict,
    tokenizer_model: str = "mock-tokenizer",
    embedding_dim: int = 8,
    max_token_size: int = 1024,
    embedding_func=mock_embedding,
    vlm_process_enable: bool | None = None,
) -> "LightRAG":
    """Build (but do NOT initialize) a ``LightRAG`` for the worker-level
    cancellation / content-reread / analyze_multimodal rig.

    Callers build their own ``role_llm_configs`` (vlm-only vs vlm+extract
    routing differs per test module) and call ``await
    rag.initialize_storages()`` themselves, exactly as the pre-refactor
    per-file ``_build_rag`` did. ``vlm_process_enable=None`` omits the kwarg
    entirely so LightRAG's own default applies, matching
    ``test_pipeline_content_reread.py``'s original copy (which never passed
    it).
    """
    from lightrag import LightRAG

    kwargs: dict = dict(
        working_dir=str(tmp_path),
        workspace=workspace,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=embedding_func,
        ),
        tokenizer=make_char_tokenizer(tokenizer_model),
        role_llm_configs=role_llm_configs,
    )
    if vlm_process_enable is not None:
        kwargs["vlm_process_enable"] = vlm_process_enable
    return LightRAG(**kwargs)
