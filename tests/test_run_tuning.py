"""Tests for forecast.run_tuning (small synthetic data, random search, scikit-learn engine)."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from forecast.config import ForecastConfig, GbmConfig, TuneConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.run_tuning import COMPLEXITY_ORDER, run_tuning, select_variant


@pytest.fixture
def tune_cfg(cfg: ForecastConfig) -> ForecastConfig:
    return replace(
        cfg,
        gbm=replace(GbmConfig(engine="hist_gb"), n_estimators=20, min_child_samples=20),
        tune=TuneConfig(n_trials=2, search="random", tuning_window_ends=("2015-03-31", "2015-05-31")),
    )


def test_select_variant_prefers_the_simpler_model_unless_the_gain_is_clear() -> None:
    means = {"gbm_default": 10.50, "gbm_tuned": 10.48, "gbm_tuned_calibrated": 10.30,
             "gbm_tuned_blend": 10.40, "gbm_tuned_calibrated_blend": 10.28}
    assert select_variant(means, min_gain=0.05) == "gbm_tuned_calibrated"  # +0.02 is noise, +0.18 is not
    assert select_variant(means, min_gain=0.5) == "gbm_default"
    assert select_variant({name: 10.0 for name in COMPLEXITY_ORDER}, 0.05) == "gbm_default"


def test_run_tuning_writes_every_output(raw_frames: RawFrames, tune_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, tune_cfg)[0]
    summary = run_tuning(tune_cfg, prepared)
    assert summary["selected_variant"] in COMPLEXITY_ORDER and summary["tuning"]["n_trials"] == 2
    assert set(summary["variants_mean_wape"]) == set(COMPLEXITY_ORDER)
    assert summary["improvement_vs_default_points"]["gbm_default"] == 0.0
    assert len(summary["stack_diagnostics"]) == 3 and "by_store" in summary["error_analysis"]
    params = json.loads(tune_cfg.paths.gbm_params.read_text())
    assert params["variant"] == summary["selected_variant"] and params["gbm"]["engine"] == "hist_gb"
    for path in (tune_cfg.paths.tuning_trials, tune_cfg.paths.step7_results, tune_cfg.paths.step7_summary):
        assert path.is_file()
    figures = {p.name for p in tune_cfg.paths.figures_dir.glob("*.png")}
    assert {"tuning_trials.png", "step7_wape_by_window.png", "error_store_item_heatmap.png",
            "error_volume_scatter.png", "error_horizon_calendar.png"} <= figures


def test_tuning_windows_must_end_before_validation_starts(raw_frames: RawFrames, tune_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, tune_cfg)[0]
    leaky = replace(tune_cfg, tune=replace(tune_cfg.tune, tuning_window_ends=("2015-03-31", "2015-06-30")))
    with pytest.raises(ValueError, match="must end before"):
        run_tuning(leaky, prepared)  # 2015-06-30 is inside the first validation window


def test_run_tuning_is_repeatable(raw_frames: RawFrames, tune_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, tune_cfg)[0]
    first, second = run_tuning(tune_cfg, prepared), run_tuning(tune_cfg, prepared)
    assert first["variants_mean_wape"] == second["variants_mean_wape"]
    assert first["tuning"]["best_params"] == second["tuning"]["best_params"]
