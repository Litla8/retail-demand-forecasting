"""Tests for forecast.selected."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from forecast.baselines import naive
from forecast.config import ForecastConfig, GbmConfig
from forecast.gbm import GbmForecaster
from forecast.selected import build_selected_forecaster, gbm_config_from, load_selected


def _selected(**overrides):
    base = {"variant": "gbm_tuned_calibrated", "gbm": {**vars(GbmConfig(engine="hist_gb")), "num_leaves": 14,
                                                          "trend_windows": [28, 91]},
            "calibrate": True, "blend": False}
    return {**base, **overrides}


def test_load_selected_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run_tuning"):
        load_selected(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"variant": "x"}))
    with pytest.raises(ValueError, match="missing keys"):
        load_selected(broken)
    good = tmp_path / "ok.json"
    good.write_text(json.dumps(_selected(), default=str))
    assert load_selected(good)["variant"] == "gbm_tuned_calibrated"


def test_gbm_config_restores_tuples_and_ignores_unknown_keys() -> None:
    saved = _selected()
    saved["gbm"]["not_a_field"] = 1
    cfg = gbm_config_from(saved, GbmConfig())
    assert cfg.trend_windows == (28, 91) and isinstance(cfg.trend_windows, tuple)
    assert cfg.num_leaves == 14 and cfg.engine == "hist_gb"


def test_plain_variant_is_a_bare_model_and_others_are_stacked() -> None:
    cfg = ForecastConfig()
    plain = build_selected_forecaster(_selected(calibrate=False, blend=False), cfg, naive)
    assert isinstance(plain, GbmForecaster) and plain.cfg.num_leaves == 14
    for calibrate, blend in ((True, False), (False, True), (True, True)):
        forecaster = build_selected_forecaster(_selected(calibrate=calibrate, blend=blend), cfg, naive)
        assert callable(forecaster) and not isinstance(forecaster, GbmForecaster)
