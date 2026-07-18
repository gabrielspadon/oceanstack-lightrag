from unittest.mock import AsyncMock

import pytest

from lightrag.utils import (
    TruncatedStr,
    mark_truncated,
    use_llm_func_with_cache,
)


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache_for_entity_extract": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_partitions_cache_by_response_format():
    cache = _FakeKVStorage()
    llm_func = AsyncMock(side_effect=["plain-text", '{"answer":"json"}'])

    plain_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
    )
    json_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        response_format={"type": "json_object"},
    )

    assert plain_result == "plain-text"
    assert json_result == '{"answer":"json"}'
    assert llm_func.await_count == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_partitions_cache_by_llm_identity():
    cache = _FakeKVStorage()
    llm_func = AsyncMock(side_effect=["model-a", "model-b"])

    first_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        llm_cache_identity={
            "role": "query",
            "binding": "openai",
            "model": "model-a",
            "host": "https://api.example.com/v1",
        },
    )
    second_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        llm_cache_identity={
            "role": "query",
            "binding": "openai",
            "model": "model-b",
            "host": "https://api.example.com/v1",
        },
    )

    assert first_result == "model-a"
    assert second_result == "model-b"
    assert llm_func.await_count == 2
    assert len(cache._store) == 2


@pytest.mark.offline
def test_mark_truncated_is_a_str_carrying_flag():
    marked = mark_truncated("partial output")
    assert isinstance(marked, str)
    assert isinstance(marked, TruncatedStr)
    assert marked == "partial output"
    assert marked.truncated is True
    # A plain str carries no flag; getattr default is the contract callers use.
    assert getattr("plain", "truncated", False) is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_skips_truncated_response():
    """A TruncatedStr result must never be written to the LLM cache."""
    cache = _FakeKVStorage()
    llm_func = AsyncMock(return_value=mark_truncated("cut off mid-extract"))

    result, _ = await use_llm_func_with_cache(
        "some prompt",
        llm_func,
        llm_response_cache=cache,
    )

    assert result == "cut off mid-extract"
    assert llm_func.await_count == 1
    # Nothing persisted: the truncated completion was skipped.
    assert len(cache._store) == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_writes_non_truncated_control():
    """Control: a normal (non-truncated) result IS written to the cache."""
    cache = _FakeKVStorage()
    llm_func = AsyncMock(return_value="complete extraction")

    result, _ = await use_llm_func_with_cache(
        "some prompt",
        llm_func,
        llm_response_cache=cache,
    )

    assert result == "complete extraction"
    assert llm_func.await_count == 1
    assert len(cache._store) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_rejects_json_schema_response_format():
    llm_func = AsyncMock()

    with pytest.raises(ValueError, match="json_schema"):
        await use_llm_func_with_cache(
            "same prompt",
            llm_func,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_payload",
                    "schema": {"type": "object"},
                },
            },
        )

    llm_func.assert_not_awaited()
