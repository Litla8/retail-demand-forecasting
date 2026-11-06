"""Tests for forecast.error_analysis."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast.baselines import build_baselines
from forecast.config import BaselineConfig, ForecastConfig
from forecast.data import RawFrames
from forecast.error_analysis import (
    RecordingForecaster,
    analyze,
    calendar_tables,
    group_wape,
    horizon_table,
    series_table,
    stack_windows,
)
from forecast.prepare import build_prepared_frame
from forecast.splits import make_validation_windows
from forecast.wide import to_wide, window_arrays

SERIES = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1), (2, 2)], names=["store", "item"])


def test_series_and_group_tables_have_known_values() -> None:
    actual = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    forecast = actual * np.array([[1.1], [0.9], [1.0], [1.2]])
    table = series_table(actual, forecast, SERIES)
    assert table["wape"].round(6).tolist() == [10.0, 10.0, 0.0, 20.0]
    assert table["bias"].round(6).tolist() == [10.0, -10.0, 0.0, 20.0]
    by_store = group_wape(table, "store").set_index("store")
    assert by_store.loc[1, "wape"] == pytest.approx(100 * (2 + 4) / 60)  # sums first, then the ratio
    assert by_store.loc[2, "bias"] == pytest.approx(100 * 16 / 140)
    with pytest.raises(ValueError, match="align"):
        series_table(actual[:3], forecast[:3], SERIES)


def test_horizon_and_calendar_tables() -> None:
    dates = pd.to_datetime(["2017-01-02", "2017-01-03", "2017-02-08", "2017-02-09"])  # Mon, Tue, Wed, Thu
    actual = np.full((2, 4), 100.0)
    forecast = actual.copy()
    forecast[:, 1] = 120.0  # only the second day is wrong (+20%)
    horizon = horizon_table(actual, forecast, np.array([1, 2, 1, 2])).set_index("day")
    assert horizon.loc[1, "wape"] == 0.0 and horizon.loc[2, "wape"] == pytest.approx(10.0)
    calendar = calendar_tables(actual, forecast, dates)
    weekday = calendar["weekday"].set_index("weekday")
    assert weekday.loc[1, "wape"] == pytest.approx(20.0) and weekday.loc[0, "wape"] == 0.0  # Tuesday is the wrong day
    assert set(calendar["month"]["month"]) == {1, 2}
    assert calendar["day_of_month"].set_index("group").loc["1-7", "wape"] == pytest.approx(10.0)  # Jan 2 and Jan 3


def test_analyze_summary_keys_and_baseline_comparison() -> None:
    rng = np.random.default_rng(42)
    actual = rng.poisson(50, size=(4, 12)).astype(float) + 1
    good = actual * (1 + rng.normal(0, 0.05, actual.shape))
    bad = actual * (1 + rng.normal(0, 0.30, actual.shape))
    dates = pd.date_range("2017-03-01", periods=12)
    result = analyze(actual, good, bad, dates, np.tile(np.arange(1, 7), 2), SERIES)
    summary = result["summary"]
    assert summary["share_series_better_than_baseline"] == 1.0  # the accurate forecast wins every series
    assert {"by_store", "worst_series", "best_series", "by_weekday", "by_month", "horizon_day"} <= set(summary)
    assert summary["worst_series"][0]["wape"] >= summary["best_series"][0]["wape"]
    assert len(result["horizon"]) == 6 and len(result["series"]) == 4


def test_recording_and_stacking_windows(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    wide = to_wide(build_prepared_frame(raw_frames, cfg)[0], cfg)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)
    naive = RecordingForecaster(build_baselines(replace(BaselineConfig(), names=("naive",)))["naive"])
    for window in windows:
        arrays = window_arrays(wide, window)
        naive(arrays.history, arrays.history_dates, arrays.future_dates)
    actual, forecast, dates, horizon_days = stack_windows(wide, windows, naive)
    assert actual.shape == forecast.shape == (6, 3 * cfg.horizon_days) and len(dates) == actual.shape[1]
    assert horizon_days[: cfg.horizon_days].tolist() == list(range(1, cfg.horizon_days + 1))
    with pytest.raises(KeyError):
        stack_windows(wide, windows, RecordingForecaster(lambda h, d, f: h[:, :1]))  # never run: nothing recorded
