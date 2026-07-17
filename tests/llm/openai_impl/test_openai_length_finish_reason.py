import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.openai import InvalidResponseError, openai_complete_if_cache


def _make_completion(
    content: str,
    finish_reason: str = "stop",
    reasoning_content: str = "",
    reasoning_tokens: int | None = None,
):
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    if reasoning_tokens is not None:
        usage.completion_tokens_details = SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    parsed=None,
                    reasoning_content=reasoning_content,
                ),
            )
        ],
        usage=usage,
    )


def _make_fake_client(completion):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=completion),
            )
        ),
        close=AsyncMock(),
    )


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        return None


def _make_stream_chunk(content=None, reasoning_content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                )
            )
        ]
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_length_finish_reason_returns_raw_content():
    """Truncated responses (finish_reason='length') still yield raw content.

    After the dispatch simplification, we no longer rely on the typed
    ``LengthFinishReasonError`` path — ``create()`` returns the partial
    content unchanged and upstream tolerant JSON parsing handles it.
    """
    raw_json = (
        '{"entities":[{"name":"Alice","type":"Person",'
        '"description":"Founder"}],"relationships":[]}'
    )
    completion = _make_completion(raw_json, finish_reason="length")
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract entities",
            response_format={"type": "json_object"},
            max_completion_tokens=128,
        )

    assert result == raw_json
    fake_client.chat.completions.create.assert_awaited_once()
    # Client is cached per loop+config and reused; it is not closed per call.
    fake_client.close.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_object_response_format_forwarded_to_create():
    completion = _make_completion(
        '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    )
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        result = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract keywords",
            response_format={"type": "json_object"},
        )

    assert result == '{"high_level_keywords":["AI"],"low_level_keywords":["RAG"]}'
    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }
    # Client is cached per loop+config and reused; it is not closed per call.
    fake_client.close.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_legacy_entity_extraction_emits_deprecation_warning():
    completion = _make_completion('{"entities":[],"relationships":[]}')
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        with pytest.warns(DeprecationWarning):
            await openai_complete_if_cache(
                model="test-model",
                prompt="Extract entities",
                entity_extraction=True,
            )

    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_legacy_keyword_extraction_emits_deprecation_warning():
    completion = _make_completion('{"high_level_keywords":[],"low_level_keywords":[]}')
    fake_client = _make_fake_client(completion)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        with pytest.warns(DeprecationWarning):
            await openai_complete_if_cache(
                model="test-model",
                prompt="Extract keywords",
                keyword_extraction=True,
            )

    fake_client.chat.completions.create.assert_awaited_once()
    assert fake_client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_typed_response_format_is_rejected():
    completion = _make_completion("{}")
    fake_client = _make_fake_client(completion)

    class FakeSchemaModel:
        pass

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        with pytest.raises(TypeError, match="typed/Pydantic"):
            await openai_complete_if_cache(
                model="test-model",
                prompt="Extract entities",
                response_format=FakeSchemaModel,
            )

    fake_client.chat.completions.create.assert_not_awaited()
    fake_client.close.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_streaming_structured_output_disables_cot():
    fake_stream = _FakeAsyncStream(
        [
            _make_stream_chunk(reasoning_content="this should not be included"),
            _make_stream_chunk(content='{"answer":"ok"}'),
        ]
    )
    fake_client = _make_fake_client(fake_stream)

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        return_value=fake_client,
    ):
        stream = await openai_complete_if_cache(
            model="test-model",
            prompt="Extract entities",
            stream=True,
            enable_cot=True,
            response_format={"type": "json_object"},
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

    assert "".join(chunks) == '{"answer":"ok"}'
    # Cached client stays open for reuse; the per-call stream is closed, not it.
    fake_client.close.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_empty_content_reasoning_only_diagnostics(caplog):
    """Reasoning-only responses surface finish_reason/usage/reasoning clues.

    Thinking models served behind OpenAI-compatible APIs (e.g. vLLM with a
    reasoning parser) can return all output in ``reasoning_content`` with an
    empty ``content``. The raised ``InvalidResponseError`` and the ERROR log
    must identify that failure mode instead of a bare "empty content".
    """
    reasoning_text = "thinking about the diagram..."
    completion = _make_completion(
        "",
        finish_reason="stop",
        reasoning_content=reasoning_text,
        reasoning_tokens=800,
    )
    fake_client = _make_fake_client(completion)

    lightrag_logger = logging.getLogger("lightrag")
    caplog.set_level(logging.ERROR, logger="lightrag")
    original_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with patch(
            "lightrag.llm.openai.create_openai_async_client",
            return_value=fake_client,
        ):
            # Call the undecorated coroutine to exercise the handler exactly
            # once (bypasses the tenacity retry loop and its waits).
            with pytest.raises(InvalidResponseError) as excinfo:
                await openai_complete_if_cache.__wrapped__(
                    model="test-model",
                    prompt="Describe the image",
                    response_format={"type": "json_object"},
                )
    finally:
        lightrag_logger.propagate = original_propagate

    message = str(excinfo.value)
    assert "finish_reason=stop" in message
    assert "reasoning_tokens=800" in message
    assert f"reasoning_content_len={len(reasoning_text)}" in message
    assert "reasoning-only" in caplog.text
    # Cached client is reused across calls, so it is not closed on this path.
    fake_client.close.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_empty_content_length_truncation_diagnostics(caplog):
    """Token-limit truncation with no content is identified as such.

    When thinking exhausts the completion budget before any content token is
    emitted, the response has ``finish_reason="length"`` and empty content;
    usage may lack ``completion_tokens_details`` entirely.
    """
    completion = _make_completion("", finish_reason="length")
    fake_client = _make_fake_client(completion)

    lightrag_logger = logging.getLogger("lightrag")
    caplog.set_level(logging.ERROR, logger="lightrag")
    original_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with patch(
            "lightrag.llm.openai.create_openai_async_client",
            return_value=fake_client,
        ):
            with pytest.raises(InvalidResponseError) as excinfo:
                await openai_complete_if_cache.__wrapped__(
                    model="test-model",
                    prompt="Describe the image",
                    response_format={"type": "json_object"},
                )
    finally:
        lightrag_logger.propagate = original_propagate

    message = str(excinfo.value)
    assert "finish_reason=length" in message
    assert "reasoning_tokens=n/a" in message
    assert "reasoning_content_len=0" in message
    assert "hit the token limit" in caplog.text
    # Cached client is reused across calls, so it is not closed on this path.
    fake_client.close.assert_not_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_cached_per_loop_and_config():
    """Two calls with identical connection config on one loop reuse the same
    client instance; a differing config constructs a separate one.

    This pins the per-loop client cache: construction happens on a cache miss
    keyed by the full connection config, and identical-config calls share the
    reused client instead of rebuilding a fresh httpx pool + TLS handshake.
    """
    created: list = []

    def _factory(*_args, **_kwargs):
        client = _make_fake_client(_make_completion('{"ok":true}'))
        created.append(client)
        return client

    with patch(
        "lightrag.llm.openai.create_openai_async_client",
        side_effect=_factory,
    ) as factory:
        await openai_complete_if_cache(
            model="test-model",
            prompt="first",
            api_key="dummy",
            response_format={"type": "json_object"},
        )
        await openai_complete_if_cache(
            model="test-model",
            prompt="second",
            api_key="dummy",
            response_format={"type": "json_object"},
        )

        # Same loop + identical connection config: one client built and reused.
        assert factory.call_count == 1
        assert len(created) == 1
        assert created[0].chat.completions.create.await_count == 2

        await openai_complete_if_cache(
            model="test-model",
            prompt="third",
            api_key="dummy",
            base_url="https://alt.example/v1",
            response_format={"type": "json_object"},
        )

        # A different connection config (base_url) builds a distinct client.
        assert factory.call_count == 2
        assert len(created) == 2
        assert created[0] is not created[1]
        assert created[1].chat.completions.create.await_count == 1

    # Cached clients are reused across calls, never closed per call.
    created[0].close.assert_not_awaited()
    created[1].close.assert_not_awaited()
