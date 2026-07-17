"""Shared PGGraphStorage test-double builder for tests/kg/postgres_impl/*.

Every ``test_postgres_*graph*`` / ``test_postgres_*cypher*`` module in this
package used to hand-roll its own ``make_graph_storage()`` helper that
``PGGraphStorage.__new__``-constructs an instance (bypassing ``__init__`` and
its dataclass machinery) and stamps the same ``workspace`` / ``namespace`` /
``graph_name`` onto it. Two things diverged per copy:

* whether ``__post_init__()`` is called — it resolves the chunk-level batch
  limit attrs read by the ``*_batch`` write paths, so callers that exercise
  those need it while read-path callers skip it;
* what extra attributes end up set (a mocked ``db``, a ``global_config``,
  ...).

``test_postgres_graph_batch.py`` additionally wires a hand-rolled
``AsyncMock`` ``db`` with a ``_run_with_retry`` side effect and returns a
capture object alongside the storage; that shape is unique to its file (not
duplicated elsewhere) so it keeps its own local ``make_graph_storage``
wrapper, using this factory only for the shared identity-attr + optional
``__post_init__`` stamping.
"""

from unittest.mock import MagicMock

from lightrag.kg.postgres_impl import PGGraphStorage


def make_graph_storage(
    *,
    use_post_init: bool = False,
    workspace: str = "test_ws",
    namespace: str = "test_graph",
    graph_name: str = "test_graph",
    **extra_attrs,
) -> PGGraphStorage:
    """Construct a ``PGGraphStorage`` test double, bypassing ``__init__``.

    ``use_post_init=True`` calls ``__post_init__()`` after the identity
    attrs are stamped (needed by callers that exercise the chunk-level batch
    limits). Every keyword in ``extra_attrs`` is set as an attribute on the
    storage afterward; ``db`` defaults to a plain ``MagicMock()`` unless the
    caller supplies its own via ``extra_attrs``.
    """
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = workspace
    storage.namespace = namespace
    storage.graph_name = graph_name
    if use_post_init:
        storage.__post_init__()
    for key, value in extra_attrs.items():
        setattr(storage, key, value)
    if "db" not in extra_attrs:
        storage.db = MagicMock()
    return storage
