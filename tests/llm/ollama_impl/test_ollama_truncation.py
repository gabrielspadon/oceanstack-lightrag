"""Regression tests: Ollama flags length-truncated output so it is not cached.

A ``done_reason == "length"`` means the model hit the token budget before
finishing; the non-empty content is wrapped in ``TruncatedStr`` (isinstance str
stays True) so ``use_llm_func_with_cache`` skips persisting it.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.ollama import _ollama_model_if_cache


def _make_fake_ollama_client(response: dict) -> SimpleNamespace:
    return SimpleNamespace(
        chat=AsyncMock(return_value=response),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ollama_length_done_reason_marks_truncated():
    response = {
        "message": {"content": "partial output"},
        "done_reason": "length",
    }
    fake_client = _make_fake_ollama_client(response)

    with patch(
        "lightrag.llm.ollama.ollama.AsyncClient",
        return_value=fake_client,
    ):
        result = await _ollama_model_if_cache.__wrapped__(
            "ollama-model",
            "hello",
        )

    assert result == "partial output"
    assert isinstance(result, str)
    assert getattr(result, "truncated", False) is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_ollama_stop_done_reason_not_marked_truncated():
    response = {
        "message": {"content": "complete output"},
        "done_reason": "stop",
    }
    fake_client = _make_fake_ollama_client(response)

    with patch(
        "lightrag.llm.ollama.ollama.AsyncClient",
        return_value=fake_client,
    ):
        result = await _ollama_model_if_cache.__wrapped__(
            "ollama-model",
            "hello",
        )

    assert result == "complete output"
    assert getattr(result, "truncated", False) is False
