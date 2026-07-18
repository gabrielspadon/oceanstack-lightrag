"""Regression tests for the deprecated ``enable_multimodal_pipeline`` addon param.

The key is accepted for backward compatibility but is fully inert: it is
stripped from ``addon_params`` and never gates multimodal processing (that is
now controlled per-document via filename-hint ``process_options``). See
``lightrag/addon_params.py``.
"""

import pytest

from lightrag.addon_params import normalize_addon_params

pytestmark = pytest.mark.offline


def test_enable_multimodal_pipeline_is_stripped_from_normalized_params():
    normalized = normalize_addon_params({"enable_multimodal_pipeline": True})

    assert "enable_multimodal_pipeline" not in normalized


def test_enable_multimodal_pipeline_warning_states_no_effect(monkeypatch):
    # `lightrag.utils.logger` has propagate=False, so caplog (which attaches
    # to the root logger) never sees it; capture via the logger directly
    # instead. Each key only warns once per process, so also reset the
    # dedup set for test independence.
    import lightrag.addon_params as addon_params_module

    monkeypatch.setattr(addon_params_module, "_warned_deprecated_keys", set())
    warnings = []
    monkeypatch.setattr(
        addon_params_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg)
    )

    normalize_addon_params({"enable_multimodal_pipeline": True})

    matches = [w for w in warnings if "enable_multimodal_pipeline" in w]
    assert len(matches) == 1
    assert "has no effect" in matches[0]
    assert "stripped" in matches[0]


def test_enable_multimodal_pipeline_absent_key_emits_no_warning(monkeypatch):
    import lightrag.addon_params as addon_params_module

    monkeypatch.setattr(addon_params_module, "_warned_deprecated_keys", set())
    warnings = []
    monkeypatch.setattr(
        addon_params_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg)
    )

    normalize_addon_params({"language": "English"})

    assert not any("enable_multimodal_pipeline" in w for w in warnings)
