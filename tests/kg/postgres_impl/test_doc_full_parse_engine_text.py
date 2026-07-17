"""LIGHTRAG_DOC_FULL.parse_engine must be TEXT.

The parse_engine field may carry an encoded engine-parameter directive
(``mineru(page_range=1-3,language=en)``) longer than the original VARCHAR(32),
so the greenfield CREATE DDL uses TEXT.
"""

from lightrag.kg.postgres_impl import TABLES


def test_create_ddl_uses_text_for_parse_engine():
    ddl = TABLES["LIGHTRAG_DOC_FULL"]["ddl"]
    assert "parse_engine TEXT" in ddl
    assert "parse_engine VARCHAR(32)" not in ddl
