"""Hypothesis property tests for the pure string helpers in lightrag/entity_resolution.py.

Targets the deterministic core only: residue normalisation, suffix-variant
detection, and the reasoner-reply parser (a total function that must never
raise on arbitrary LLM output).
"""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lightrag.entity_resolution import (
    Decision,
    _is_suffix_variant,
    _parse_reasoner_response,
    _residue_no_dots,
)

pytestmark = pytest.mark.offline

_ALNUM_LOWER = set("abcdefghijklmnopqrstuvwxyz0123456789")


@given(st.text())
def test_residue_no_dots_idempotent(name):
    once = _residue_no_dots(name)
    assert _residue_no_dots(once) == once


@given(st.text())
def test_residue_no_dots_output_is_lowercase_alnum(name):
    assert set(_residue_no_dots(name)) <= _ALNUM_LOWER


@given(st.text(), st.text())
def test_is_suffix_variant_symmetric_and_total(a, b):
    assert _is_suffix_variant(a, b) == _is_suffix_variant(b, a)


@given(st.text())
def test_is_suffix_variant_irreflexive(name):
    # Equal residues are the auto-merge path, never a suffix-variant conflict.
    assert _is_suffix_variant(name, name) is False


@given(st.text(), st.sets(st.text(), max_size=5))
def test_parse_reasoner_response_is_total(text, candidates):
    decision, target, confidence = _parse_reasoner_response(text, candidates)
    assert decision is None or isinstance(decision, Decision)
    assert 0.0 <= confidence <= 1.0
    if target is not None:
        assert target in candidates


@given(
    st.sampled_from([d.value for d in Decision]),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_parse_reasoner_response_accepts_valid_payload(decision_value, confidence):
    payload = json.dumps(
        {"decision": decision_value, "target": "CANDIDATE", "confidence": confidence}
    )
    decision, target, conf = _parse_reasoner_response(payload, {"CANDIDATE"})
    assert decision == Decision(decision_value)
    assert target == "CANDIDATE"
    assert conf == confidence


def test_parse_reasoner_response_rejects_out_of_range_confidence():
    payload = json.dumps(
        {"decision": "discard_and_reuse", "target": "X", "confidence": 1.5}
    )
    decision, target, confidence = _parse_reasoner_response(payload, {"X"})
    assert decision is None
    assert target is None
    assert confidence == 0.0


@given(st.sets(st.text(), min_size=1, max_size=5), st.text())
def test_parse_reasoner_response_rejects_off_list_target(candidates, off_list):
    if off_list in candidates:
        return
    payload = json.dumps(
        {"decision": "discard_and_reuse", "target": off_list, "confidence": 0.9}
    )
    decision, target, confidence = _parse_reasoner_response(payload, candidates)
    if off_list in (None, "", "null"):
        # Coerced to a target-less reply; the parser keeps the decision.
        assert target is None
    else:
        assert decision is None
        assert target is None
        assert confidence == 0.0
