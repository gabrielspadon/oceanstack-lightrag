"""Offline unit test for the ghost-entity typing heuristic (OceanStack fork).

`_ghost_entity_type` assigns a type to a relation endpoint that the LLM named
but never emitted as its own entity tuple. It must (a) follow the naming
heuristic and (b) only ever return a canonical type — the remap+validate routing
guarantees no off-taxonomy label (e.g. ``test_suite``) can leak into the graph.
"""

import pytest

from lightrag.operate import _OS_VALID_TYPES, _ghost_entity_type

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("test_foo", "concept"),  # tests are demoted (off-taxonomy -> concept)
        ("test.bar", "concept"),
        ("ValueError", "exception"),  # endswith error/exception
        ("my_custom_exception", "exception"),
        ("signals.ais_position_reports", "function"),  # dotted path -> code ref
        ("module.func", "function"),
        ("SCREAMING_CONST", "function"),  # isupper
        ("_private", "function"),  # leading underscore
        ("snake_case_name", "function"),  # contains underscore
        ("vessel", "concept"),  # bare lowercase word -> default
        (".hidden", "concept"),  # leading dot is NOT a dotted path
        ("", "concept"),  # empty
    ],
)
def test_ghost_entity_type_heuristic(name: str, expected: str) -> None:
    assert _ghost_entity_type(name) == expected


@pytest.mark.parametrize(
    "name",
    ["test_x", "Boom", "a.b.c", "FOO_BAR", "plain", "Ünîcødé", ""],
)
def test_ghost_entity_type_is_always_canonical(name: str) -> None:
    # No matter what the heuristic produces, the result is a valid canonical
    # type — this is the invariant the remap+validate routing exists to enforce.
    assert _ghost_entity_type(name) in _OS_VALID_TYPES
