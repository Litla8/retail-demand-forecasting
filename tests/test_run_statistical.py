"""Tests for forecast.run_statistical (stand-in models instead of statsmodels/prophet)."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast.config import ForecastConfig, StatisticalConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.run_statistical import run_statistical
from forecast.statistical import SeriesModelForecaster


def _mean_model(y: np.ndarray, dates: pd.DatetimeIndex, future: pd.DatetimeIndex) -> np.ndarray:
    return np.full(len(future), y[-28:].mean())


@pytest.fixture
def small_cfg(cfg: ForecastConfig) -> ForecastConfig:
    return replace(cfg, statistical=StatisticalConfig(n_series=3))


def test_run_statistical_writes_outputs_and_compares_with_baselines(raw_frames: RawFrames, small_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, small_cfg)[0]
    models = {"flat_mean": SeriesModelForecaster("flat_mean", _mean_model, fit_days=100)}
    summary = run_statistical(small_cfg, prepared, models=models)

    assert summary["sample"]["n_series"] == 3 and len(summary["sample"]["series"]) == 3
    assert summary["best_statistical"] == "flat_mean" and summary["best_baseline"] in summary["baselines"]
    assert summary["best_model"] in summary["baselines"] and "best_model_mean_wape" in summary["headroom"]
    assert "flat_mean" in summary["statistical_vs_best_baseline_pct"]
    assert summary["fits"]["flat_mean"]["failures"] == 0
    saved = json.loads(small_cfg.paths.statistical_summary.read_text())
    assert saved["best_statistical"] == "flat_mean"
    results = pd.read_csv(small_cfg.paths.statistical_results)
    assert "flat_mean" in set(results["baseline"]) and set(results["scope"]) == {"all", "1-7", "8-14", "15-28"}
    names = {p.name for p in small_cfg.paths.figures_dir.glob("statistical_*.png")}
    assert names == {"statistical_wape_by_window.png", "statistical_wape_by_horizon.png",
                     "statistical_total_forecast.png"}


def test_run_statistical_requires_at_least_one_model(raw_frames: RawFrames, small_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, small_cfg)[0]
    with pytest.raises(RuntimeError, match="No statistical model"):
        run_statistical(small_cfg, prepared, models={})
