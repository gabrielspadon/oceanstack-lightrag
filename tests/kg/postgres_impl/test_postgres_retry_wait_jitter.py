"""Regression tests for retry-wait jitter (ITEM 5).

Covers 8 sites in postgres_impl.py that build a tenacity ``wait`` strategy:

* Two dynamic ``wait_strategy`` locals (``PostgreSQLDB.initdb`` and
  ``PostgreSQLDB._run_with_retry``) that used plain ``wait_exponential`` with
  a ``wait_fixed(0)`` busy-spin fallback when ``connection_retry_backoff``
  is non-positive.
* Six ``@retry`` decorators on ``PGGraphStorage`` AGE write paths that used
  jitterless ``wait_exponential(multiplier=1, min=4, max=10)``.

All eight must now use ``tenacity.wait_exponential_jitter`` so concurrent
retries don't thundering-herd in lockstep, and the non-positive-backoff
fallback must be a small jittered floor rather than a busy spin.
"""

import pytest
from tenacity import wait_exponential_jitter

from lightrag.kg.postgres_impl import PGGraphStorage, PostgreSQLDB

pytestmark = pytest.mark.offline

# The six AGE write paths decorated with @retry(..., wait=wait_exponential_jitter(...)).
_AGE_WRITE_RETRY_METHODS = [
    "_write_typed_records",
    "upsert_node",
    "upsert_edge",
    "_upsert_node_chunk",
    "_upsert_edge_chunk",
    "delete_node",
]


@pytest.mark.parametrize("method_name", _AGE_WRITE_RETRY_METHODS)
def test_age_write_retry_wait_is_jittered(method_name: str) -> None:
    """Each AGE write @retry decorator must use wait_exponential_jitter, not
    the jitterless wait_exponential(multiplier=1, min=4, max=10)."""
    method = getattr(PGGraphStorage, method_name)
    assert hasattr(method, "retry"), (
        f"{method_name} is expected to be tenacity @retry-wrapped"
    )
    wait = method.retry.wait
    assert isinstance(wait, wait_exponential_jitter), (
        f"{method_name} retry wait strategy regressed off wait_exponential_jitter "
        f"(got {type(wait).__name__}); concurrent AGE writers will thundering-herd"
    )
    # min=4/max=10 from the original wait_exponential(multiplier=1, min=4, max=10)
    # map onto initial/max on wait_exponential_jitter.
    assert wait.initial == 4
    assert wait.max == 10


def _make_db(
    connection_retry_backoff: float, connection_retry_backoff_max: float = 10.0
) -> PostgreSQLDB:
    return PostgreSQLDB(
        {
            "host": "localhost",
            "port": 5432,
            "user": "test_user",
            "password": "test_password",
            "database": "test_db",
            "workspace": "",
            "max_connections": 4,
            "connection_retry_attempts": 3,
            "connection_retry_backoff": connection_retry_backoff,
            "connection_retry_backoff_max": connection_retry_backoff_max,
            "pool_close_timeout": 5.0,
        }
    )


class _CaptureAndAbort:
    """Fake tenacity.AsyncRetrying that records its kwargs, then aborts.

    Raising inside ``__aiter__`` (a plain sync method, not a coroutine)
    propagates before any ``async for`` body runs, so this intercepts the
    ``wait=`` kwarg passed by ``initdb``/``_run_with_retry`` before either
    method reaches real connection I/O.
    """

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def __aiter__(self):
        raise RuntimeError("captured-before-io")


@pytest.mark.parametrize(
    "backoff,backoff_max",
    [
        (2.0, 10.0),  # positive backoff branch
        (0.0, 10.0),  # non-positive backoff -> jittered floor, not busy-spin
    ],
)
async def test_initdb_wait_strategy_is_jittered(
    monkeypatch, backoff: float, backoff_max: float
) -> None:
    monkeypatch.setattr("lightrag.kg.postgres_impl.AsyncRetrying", _CaptureAndAbort)
    db = _make_db(backoff, backoff_max)

    with pytest.raises(RuntimeError, match="captured-before-io"):
        await db.initdb()

    wait = _CaptureAndAbort.last_kwargs["wait"]
    assert isinstance(wait, wait_exponential_jitter), (
        f"initdb() wait strategy regressed off wait_exponential_jitter "
        f"(got {type(wait).__name__})"
    )
    if backoff > 0:
        assert wait.initial == backoff
        assert wait.max == backoff_max
    else:
        # Non-positive backoff must fall back to a small jittered floor,
        # never a wait_fixed(0) busy spin.
        assert wait.initial == pytest.approx(0.05)
        assert wait.max == pytest.approx(1)


@pytest.mark.parametrize(
    "backoff,backoff_max",
    [
        (3.0, 15.0),
        (0.0, 15.0),
    ],
)
async def test_run_with_retry_wait_strategy_is_jittered(
    monkeypatch, backoff: float, backoff_max: float
) -> None:
    monkeypatch.setattr("lightrag.kg.postgres_impl.AsyncRetrying", _CaptureAndAbort)
    db = _make_db(backoff, backoff_max)

    async def _noop_operation(connection):
        return None

    with pytest.raises(RuntimeError, match="captured-before-io"):
        await db._run_with_retry(_noop_operation)

    wait = _CaptureAndAbort.last_kwargs["wait"]
    assert isinstance(wait, wait_exponential_jitter), (
        f"_run_with_retry() wait strategy regressed off wait_exponential_jitter "
        f"(got {type(wait).__name__})"
    )
    if backoff > 0:
        assert wait.initial == backoff
        assert wait.max == backoff_max
    else:
        assert wait.initial == pytest.approx(0.05)
        assert wait.max == pytest.approx(1)
