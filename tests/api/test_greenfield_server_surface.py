from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.api.runtime_validation import validate_single_worker


def test_single_worker_validation_accepts_exactly_one_worker() -> None:
    validate_single_worker(1)

    for workers in (0, 2, 8):
        with pytest.raises(ValueError, match="exactly one worker"):
            validate_single_worker(workers)


@pytest.mark.parametrize(
    "removed_option",
    (
        "--workspace",
        "--input-dir",
        "--simulated-model-name",
        "--simulated-model-tag",
    ),
)
def test_greenfield_server_rejects_removed_legacy_cli_options(
    removed_option: str,
) -> None:
    from lightrag.api.config import parse_args

    with pytest.raises(SystemExit):
        parse_args([removed_option, "legacy-value"])


def test_greenfield_server_has_no_parser_or_gunicorn_startup_path() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "lightrag/api/lightrag_server.py"
    ).read_text()

    assert "load_third_party_parsers" not in source
    assert "validate_parser_routing_config" not in source
    assert "validate_smart_heading_dependencies" not in source
    assert "GUNICORN_CMD_ARGS" not in source


def test_create_app_exposes_only_explicit_plane_read_api(monkeypatch) -> None:
    for name in (
        "LLM_BINDING",
        "EMBEDDING_BINDING",
        "LIGHTRAG_API_PREFIX",
        "LIGHTRAG_KV_STORAGE",
        "LIGHTRAG_VECTOR_STORAGE",
        "LIGHTRAG_GRAPH_STORAGE",
        "LIGHTRAG_DOC_STATUS_STORAGE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", "--workers", "1"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app

        args = parse_args()
        pool = SimpleNamespace(acquire=AsyncMock(), close=AsyncMock())
        rag = MagicMock()
        with patch("lightrag.api.lightrag_server.LightRAG", return_value=rag):
            app = create_app(args, generation_pool=pool)
        paths = TestClient(app).get("/openapi.json").json()["paths"]
        health = TestClient(app).get("/health").json()
    finally:
        sys.argv = original_argv

    assert "/planes/{plane}/query" in paths
    assert "/planes/{plane}/query/stream" in paths
    assert "/planes/{plane}/query/data" in paths
    assert "/planes/{plane}/graphs" in paths
    assert health["status"] == "ready"
    assert health["generation_runtime"] == "ready"
    assert (
        not {
            "pipeline_busy",
            "pipeline_active",
            "workspace",
            "working_directory",
            "llm_queue_status",
        }
        & health.keys()
    )
    for removed in (
        "/query",
        "/query/stream",
        "/query/data",
        "/graph/entity/edit",
        "/graph/relation/edit",
        "/documents/text",
        "/api/generate",
        "/api/chat",
    ):
        assert removed not in paths


def test_get_application_assembles_concrete_postgres_generation_runtime(
    monkeypatch,
) -> None:
    for name in (
        "LLM_BINDING",
        "EMBEDDING_BINDING",
        "LIGHTRAG_KV_STORAGE",
        "LIGHTRAG_VECTOR_STORAGE",
        "LIGHTRAG_GRAPH_STORAGE",
        "LIGHTRAG_DOC_STATUS_STORAGE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", "--workers", "1"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import get_application

        args = parse_args()
        args.kv_storage = "PGKVStorage"
        args.graph_storage = "PGGraphStorage"
        args.vector_storage = "PGVectorStorage"
        args.doc_status_storage = "PGDocStatusStorage"
        pool = SimpleNamespace(acquire=AsyncMock(), close=AsyncMock())
        runtime = SimpleNamespace(pool=pool)
        with (
            patch(
                "lightrag.api.lightrag_server.create_postgres_generation_runtime",
                return_value=runtime,
            ) as create_runtime,
            patch("lightrag.api.lightrag_server.LightRAG"),
        ):
            app = get_application(args)
    finally:
        sys.argv = original_argv

    assert app is not None
    create_runtime.assert_called_once()
    assert create_runtime.call_args.args == (args,)
    builder = create_runtime.call_args.kwargs["builder"]
    assert builder.constructor_kwargs["vector_storage"] == "PGVectorStorage"


def test_deployment_builder_composition_does_not_construct_fastapi(
    monkeypatch,
) -> None:
    for name in (
        "LLM_BINDING",
        "EMBEDDING_BINDING",
        "LIGHTRAG_KV_STORAGE",
        "LIGHTRAG_VECTOR_STORAGE",
        "LIGHTRAG_GRAPH_STORAGE",
        "LIGHTRAG_DOC_STATUS_STORAGE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", "--workers", "1"]
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import (
            create_deployment_generation_rag_builder,
        )

        args = parse_args()
        builder = MagicMock()
        with (
            patch(
                "lightrag.api.lightrag_server.create_generation_rag_builder",
                return_value=builder,
            ) as create_builder,
            patch(
                "lightrag.api.lightrag_server.FastAPI",
                side_effect=AssertionError("FastAPI must not be constructed"),
            ),
        ):
            result = create_deployment_generation_rag_builder(args)
    finally:
        sys.argv = original_argv

    assert result is builder
    create_builder.assert_called_once()
