"""Tests for configuration non-disclosure on ``GET /health``.

Issue #3294 established that ``/health`` must not reveal sensitive runtime
configuration to unauthenticated callers. The greenfield server goes further:
``/health`` returns ONLY liveness fields (status, generation readiness,
versions, webui availability) to EVERY caller, authenticated or not, across
all three authentication modes. These tests pin that stronger contract while
keeping the endpoint HTTP 200 for external liveness probes.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

# Fields that must NEVER appear in a /health response.
_SENSITIVE_TOP_LEVEL = ("working_directory", "input_directory", "configuration")
# Liveness fields that must always be present (pure liveness signals).
_LIVENESS_FIELDS = ("status", "generation_runtime", "core_version", "api_version")


_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep tests hermetic from developer-local .env and global config state."""
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("WHITELIST_PATHS", "/health,/api/*")
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


class _FakeLightRAG:
    """Minimal stand-in implementing the async surface /health touches."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def set_role_llm_metadata(self, _role, **_metadata):
        return None

    def get_llm_role_config(self):
        return {}

    async def get_llm_queue_status(self, include_base=True):
        return {}

    async def get_embedding_queue_status(self):
        return {}

    async def get_rerank_queue_status(self):
        return {}


def _build_client(monkeypatch, *, api_key=None):
    """Build a /health-capable TestClient with all backend I/O mocked out."""
    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv
    if api_key is not None:
        args.key = api_key
    initialize_config(args, force=True)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))

    generation_pool = SimpleNamespace(acquire=AsyncMock(), close=AsyncMock())
    app = lightrag_server.create_app(args, generation_pool=generation_pool)
    return TestClient(app)


def _set_auth_mode(monkeypatch, *, auth_configured):
    """Override the module-level auth flags the /health gate reads at runtime.

    Also pin a whitelist that exempts /health so combined_auth keeps returning
    200 for anonymous callers (the gate, not combined_auth, hides the config).
    """
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(utils_api, "auth_configured", auth_configured)
    monkeypatch.setattr(
        utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
    )


def _assert_liveness_only(body):
    for field in _LIVENESS_FIELDS:
        assert field in body, f"liveness field {field!r} missing"
    for field in _SENSITIVE_TOP_LEVEL:
        assert field not in body, f"sensitive field {field!r} leaked"
    assert body["status"] == "ready"
    assert body["generation_runtime"] == "ready"


# --------------------------------------------------------------------------- #
# Fully open mode: even with everything open, /health stays liveness-only.
# --------------------------------------------------------------------------- #
def test_open_mode_returns_liveness_only(monkeypatch):
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=False)

    resp = client.get("/health")

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


# --------------------------------------------------------------------------- #
# Password auth: anonymous and authenticated callers both get liveness only.
# --------------------------------------------------------------------------- #
def test_password_mode_anonymous_gets_liveness_only(monkeypatch):
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=True)

    resp = client.get("/health")

    assert resp.status_code == 200  # liveness probe must stay green
    _assert_liveness_only(resp.json())


def test_password_mode_valid_token_stays_liveness_only(monkeypatch):
    import lightrag.api.utils_api as utils_api

    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=True)
    monkeypatch.setattr(
        utils_api.auth_handler,
        "validate_token",
        lambda token: (
            {"username": "admin", "role": "user"}
            if token == "valid-user-token"
            else (_ for _ in ()).throw(ValueError("bad token"))
        ),
    )

    token = "valid-user-token"
    resp = client.get("/health", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


def test_password_mode_guest_token_stays_liveness_only(monkeypatch):
    import lightrag.api.utils_api as utils_api

    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=True)
    monkeypatch.setattr(
        utils_api.auth_handler,
        "validate_token",
        lambda token: {"username": "guest", "role": "guest"},
    )

    token = "guest-token"
    resp = client.get("/health", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


# --------------------------------------------------------------------------- #
# API-key-only mode: a valid X-API-Key still gets liveness only.
# --------------------------------------------------------------------------- #
def test_api_key_mode_anonymous_gets_liveness_only(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=False)

    resp = client.get("/health")

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


def test_api_key_mode_valid_key_stays_liveness_only(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=False)

    resp = client.get("/health", headers={"X-API-Key": "secret-key"})

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


# --------------------------------------------------------------------------- #
# Combined mode (AUTH_ACCOUNTS + LIGHTRAG_API_KEY): liveness only either way.
# --------------------------------------------------------------------------- #
def test_combined_mode_anonymous_gets_liveness_only(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=True)

    resp = client.get("/health")

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


def test_combined_mode_valid_api_key_stays_liveness_only(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=True)

    resp = client.get("/health", headers={"X-API-Key": "secret-key"})

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())
