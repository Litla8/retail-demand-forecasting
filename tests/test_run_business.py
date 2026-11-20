"""Tests for forecast.run_business (small synthetic data and a small boosting model)."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from forecast.config import ForecastConfig, GbmConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.run_business import run_business


@pytest.fixture
def biz_cfg(cfg: ForecastConfig) -> ForecastConfig:
    small = replace(cfg, gbm=replace(GbmConfig(engine="hist_gb"), n_estimators=20, min_child_samples=20))
    small.paths.models_dir.mkdir(parents=True, exist_ok=True)
    selected = {"variant": "gbm_tuned_calibrated", "gbm": vars(small.gbm), "baseline": "month_weekday_profile_multi_year",
                "calibrate": True, "blend": False}
    small.paths.gbm_params.write_text(json.dumps(selected, default=str))
    return small


def test_run_business_writes_every_output(raw_frames: RawFrames, biz_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, biz_cfg)[0]
    summary = run_business(biz_cfg, prepared)
    assert summary["selected_variant"] == "gbm_tuned_calibrated"
    assert set(summary["cost_optimal"]) == {"seasonal_naive_364", "month_weekday_profile_multi_year", "selected_model"}
    assert summary["assumptions"]["windows_simulated"] == [1, 2]
    frontier = pd.read_csv(biz_cfg.paths.business_results)
    assert len(frontier) == 3 * len(biz_cfg.business.service_levels) and frontier["fill_rate"].between(0, 1).all()
    plan = pd.read_csv(biz_cfg.paths.reorder_plan)
    assert len(plan) == 6 * 3 and (plan["order_up_to"] >= plan["forecast_units"]).all()
    intervals = pd.read_csv(biz_cfg.paths.forecast_intervals)
    assert len(intervals) == 6 * 28 and (intervals["lower_95"] <= intervals["upper_95"]).all()
    submission = pd.read_csv(biz_cfg.paths.submission)
    assert len(submission) == 6 * 28 and submission["id"].is_unique and (submission["sales"] >= 0).all()
    backtest = pd.read_csv(biz_cfg.paths.backtest_forecasts)
    assert len(backtest) == 3 * 6 * 28 and {"actual", "benchmark", "baseline", "model"} <= set(backtest.columns)
    names = {p.name for p in biz_cfg.paths.figures_dir.glob("*.png")}
    assert {"business_frontier.png", "business_costs.png", "business_coverage.png", "forecast_fan_chart.png"} <= names
    assert json.loads(biz_cfg.paths.business_summary.read_text())["plan"]["service_level"] == summary["plan"]["service_level"]


def test_run_business_needs_the_step_7_selection(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    with pytest.raises(FileNotFoundError, match="run_tuning"):
        run_business(cfg, prepared)
