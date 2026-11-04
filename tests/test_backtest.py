"""Tests for forecast.backtest."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import noise_floor, run_backtest, score_forecast, scope_slices, summarize_backtest
from forecast.baselines import build_baselines
from forecast.config import BaselineConfig, ForecastConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.splits import make_validation_windows
from forecast.wide import to_wide


def test_scope_slices() -> None:
    slices = scope_slices((28, 60, 90))
    assert list(slices) == ["all", "1-28", "29-60", "61-90"]
    assert slices["29-60"] == slice(28, 60) and slices["all"] == slice(0, 90)


def test_score_forecast_values_and_validation() -> None:
    actual = np.full((2, 6), 10.0)
    forecast = actual.copy()
    forecast[:, 3:] = 12.0  # second half over-forecast by 20%
    rows = {r["scope"]: r for r in score_forecast(actual, forecast, (3, 6))}
    assert set(rows) == {"all", "1-3", "4-6"}
    assert rows["1-3"]["wape"] == 0.0 and rows["4-6"]["wape"] == pytest.approx(20.0)
    assert rows["all"]["wape"] == pytest.approx(10.0) and rows["4-6"]["bias"] == pytest.approx(20.0)
    with pytest.raises(ValueError, match="Shape mismatch"):
        score_forecast(actual, forecast[:, :5], (3, 6))
    with pytest.raises(ValueError, match="horizon"):
        score_forecast(actual, forecast, (3, 7))


def test_noise_floor_recovers_known_noise() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2016-01-02", periods=90, freq="D")
    actual = 100.0 + rng.normal(0, 10, size=(300, 90))  # flat level, noise sd 10 -> WAPE floor 0.798 * 10 = 7.98%
    result = noise_floor(actual, dates)
    assert 7.0 < result["estimated_floor_wape"] < 9.0
    assert result["loo_wape"] > result["estimated_floor_wape"]  # the raw leave-one-out error is inflated
    with pytest.raises(ValueError, match="No day"):
        noise_floor(actual[:, :1], dates[:1])


def test_backtest_and_summary_on_synthetic_data(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    wide = to_wide(prepared, cfg)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)
    forecasters = build_baselines(cfg.baselines)
    scores = run_backtest(wide, windows, forecasters, cfg)
    assert len(scores) == len(forecasters) * len(windows) * 4  # 'all' plus three horizon buckets
    assert scores["wape"].between(0, 200).all() and scores.notna().all().all()

    floors = {w.valid_end.date().isoformat(): 10.0 for w in windows}
    summary = summarize_backtest(scores, cfg, floors)
    bench = summary["baselines"][cfg.benchmark]
    assert bench["mean_improvement_vs_benchmark_pct"] == pytest.approx(0.0)  # the benchmark vs itself
    assert summary["best_model"] in forecasters and summary["noise_floor"]["mean"] == 10.0
    assert summary["headroom"]["best_model_mean_wape"] == summary["baselines"][summary["best_model"]]["mean_wape"]
    needed = summary["target"]["wape_needed_by_window"][windows[0].valid_end.date().isoformat()]
    assert needed["for_high"] < needed["for_low"]


def test_summary_requires_the_benchmark(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    wide = to_wide(prepared, cfg)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)
    only_naive = build_baselines(replace(BaselineConfig(), names=("naive",)))
    scores = run_backtest(wide, windows, only_naive, cfg)
    with pytest.raises(ValueError, match="Benchmark"):
        summarize_backtest(scores, cfg, {})
