"""Regression coverage for the streaming/TruncatedStr cache-contract exemption.

Claim under test: streamed LLM responses are NEVER written to the LLM
response cache, so the inability to carry the `TruncatedStr.truncated` flag
on an async iterator (`TruncatedStr` is a `str` subclass; an async generator
cannot be one) is harmless.

Traced invariant chain:

- ``save_to_cache`` (lightrag/utils.py, ~2893-2915) is the sole gate that
  actually matters for streaming: it skips any ``cache_data.content`` with
  ``hasattr(content, "__aiter__")`` — a TYPE check, not a flag check.
- ``kg_query`` / ``naive_query`` (lightrag/operate.py, ~3920-3953 and
  ~6022-6053) guard their ``save_to_cache`` calls with
  ``not getattr(response, "truncated", False)``. That guard does NOT catch
  streaming responses (an async generator has no ``truncated`` attribute, so
  the guard evaluates True), so a streamed ``response`` DOES reach
  ``save_to_cache`` as ``content`` — but save_to_cache's own ``__aiter__``
  check catches it there.
- ``use_llm_func_with_cache`` (lightrag/utils.py, ~3421-3583) has no
  ``stream`` parameter, and its three call sites (operate.py lines ~487,
  ~3454, ~3519 — summary generation and entity extraction) never thread a
  ``stream`` argument through to the wrapped LLM function, so ``res`` is
  always a plain str/TruncatedStr in practice, never an async iterator.
  As defense-in-depth, this file also proves that if a misbehaving
  ``use_llm_func`` violated that contract anyway, the response is still
  never cached (it fails loudly via ``remove_think_tags`` instead).
- Every LLM provider binding (openai.py, gemini.py, anthropic.py, ollama.py,
  bedrock.py) applies ``mark_truncated`` only on the assembled non-streaming
  string branch; the streaming branch always returns a raw async generator,
  never wrapped, confirming TruncatedStr and streaming are mutually
  exclusive response shapes.
"""

from unittest.mock import AsyncMock

import pytest

from lightrag.utils import CacheData, save_to_cache, use_llm_func_with_cache


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
        }
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


async def _stream():
    yield "chunk-1"
    yield "chunk-2"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_save_to_cache_skips_async_iterator_content():
    """save_to_cache's __aiter__ check is the actual streaming gate."""
    cache = _FakeKVStorage()

    await save_to_cache(
        cache,
        CacheData(
            args_hash="hash",
            content=_stream(),
            prompt="q",
            mode="mix",
            cache_type="query",
        ),
    )

    assert len(cache._store) == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_path_truncated_guard_does_not_catch_streams_but_save_to_cache_does():
    """Mirrors the kg_query/naive_query save-site shape exactly.

    Reproduces the upstream guard `not getattr(response, "truncated", False)`
    used at operate.py ~3923-3925 / ~6024-6026 to prove it evaluates True
    (i.e. does NOT skip) for a streaming response, and that save_to_cache is
    still called with the raw iterator as `content` — and still skips the
    write, because the protection lives in save_to_cache itself.
    """
    cache = _FakeKVStorage()
    response = _stream()

    # This is the exact condition guarding the save_to_cache call in both
    # kg_query and naive_query.
    truncated_guard_permits_save = not getattr(response, "truncated", False)
    assert truncated_guard_permits_save is True  # does NOT gate the stream

    if cache.global_config.get("enable_llm_cache") and truncated_guard_permits_save:
        await save_to_cache(
            cache,
            CacheData(
                args_hash="hash",
                content=response,
                prompt="query text",
                mode="mix",
                cache_type="query",
            ),
        )

    assert len(cache._store) == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_never_caches_an_async_iterator_response():
    """Defense-in-depth for use_llm_func_with_cache's str-only contract.

    use_llm_func_with_cache has no `stream` kwarg, and none of its three
    call sites (operate.py ~487, ~3454, ~3519) thread `stream=True` through
    to the wrapped LLM function, so this path never legitimately receives an
    async iterator. If a misbehaving `use_llm_func` returned one anyway,
    `remove_think_tags`'s `re.sub` raises TypeError on the non-str object
    before any cache write is attempted, so the response is rejected loudly
    rather than silently persisted.
    """
    cache = _FakeKVStorage()
    llm_func = AsyncMock(return_value=_stream())

    with pytest.raises(TypeError):
        await use_llm_func_with_cache(
            "some prompt",
            llm_func,
            llm_response_cache=cache,
        )

    assert llm_func.await_count == 1
    assert len(cache._store) == 0
