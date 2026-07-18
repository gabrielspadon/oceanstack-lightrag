"""Regression test for ITEM 8: MongoDocStatusStorage.upsert gather hygiene.

``upsert`` used to fan out one ``update_one`` per record into
``asyncio.gather(*update_tasks)`` without ``return_exceptions=True``. On the
first failing write, ``gather`` immediately raises and cancels/abandons the
remaining sibling tasks as fire-and-forget, leaking partially-applied writes
with no guarantee the rest ever completed.

The fix awaits every task to completion (``return_exceptions=True``), then
raises the first captured error. Writes are idempotent by ``_id``
(``upsert=True``), so a retry safely reconciles any partial batch.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock

from lightrag.kg.mongo_impl import MongoDocStatusStorage

pytestmark = pytest.mark.offline


def _make_storage() -> MongoDocStatusStorage:
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_doc_status"
    storage._data = AsyncMock()
    return storage


async def test_upsert_awaits_all_siblings_before_raising_first_error():
    storage = _make_storage()
    completed: list[str] = []

    async def _update_one(filter_, update, upsert=True):
        key = filter_["_id"]
        if key == "fail-me":
            raise RuntimeError("simulated transient write failure")
        # Simulate a slow sibling write that must still run to completion
        # even though another task in the same gather() fails.
        await asyncio.sleep(0)
        completed.append(key)
        return None

    storage._data.update_one = AsyncMock(side_effect=_update_one)

    data = {
        "fail-me": {"status": "pending"},
        "ok-1": {"status": "pending"},
        "ok-2": {"status": "pending"},
    }

    with pytest.raises(RuntimeError, match="simulated transient write failure"):
        await storage.upsert(data)

    # Every update_one call must be awaited to completion despite the first
    # failure -- return_exceptions=True must not leak sibling writes.
    assert storage._data.update_one.await_count == len(data)
    assert set(completed) == {"ok-1", "ok-2"}


async def test_upsert_no_error_returns_normally():
    storage = _make_storage()
    storage._data.update_one = AsyncMock(return_value=None)

    data = {
        "ok-1": {"status": "pending"},
        "ok-2": {"status": "pending"},
    }

    await storage.upsert(data)

    assert storage._data.update_one.await_count == len(data)
