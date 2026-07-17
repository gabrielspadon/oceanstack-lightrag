from unittest.mock import AsyncMock

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB


@pytest.mark.asyncio
async def test_check_table_exists_uses_search_path_visible_regclass():
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.query = AsyncMock(return_value={"exists": True})

    assert await db.check_table_exists("LIGHTRAG_DOC_FULL") is True

    db.query.assert_awaited_once_with(
        "SELECT to_regclass($1) IS NOT NULL AS exists",
        ["lightrag_doc_full"],
    )
