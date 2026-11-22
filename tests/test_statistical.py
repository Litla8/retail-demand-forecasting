"""Tests for forecast.statistical (library-dependent tests skip when statsmodels/prophet are missing)."""
from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forecast.baselines import seasonal_naive_7
from forecast.config import StatisticalConfig
from forecast.metrics import wape
from forecast.statistical import (
    SeriesModelForecaster,
    build_statistical_models,
    month_dummies,
    select_series_by_volume,
)
from forecast.wide import WideData

DATES = pd.date_range("2014-01-01", "2016-12-31", freq="D")  # 1,096 days
FUTURE = pd.date_range("2017-01-01", periods=60, freq="D")  # January and February 2017
MONTH_LEVEL = np.array([50, 55, 70, 90, 100, 110, 120, 100, 90, 80, 85, 60], dtype=float)
WEEKDAY_FACTOR = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.2, 1.3])


def _staircase(dates: pd.DatetimeIndex, noise: float, seed: int = 42) -> np.ndarray:
    """Sales with a flat level per month, a weekly pattern and small noise."""
    rng = np.random.default_rng(seed)
    base = MONTH_LEVEL[dates.month.to_numpy() - 1] * WEEKDAY_FACTOR[dates.day_of_week.to_numpy()]
    return base + rng.normal(0, noise, len(dates))


def test_select_series_by_volume_spans_the_range() -> None:
    rng = np.random.default_rng(42)
    levels = rng.permutation(np.arange(1.0, 11.0))  # series volumes 1..10 in shuffled order
    wide = WideData(values=np.tile(levels[:, None], (1, 30)), dates=pd.date_range("2016-01-01", periods=30),
                    series=pd.MultiIndex.from_product([[1, 2], [1, 2, 3, 4, 5]], names=["store", "item"]))
    chosen = select_series_by_volume(wide, 3)
    assert chosen.n_series == 3 and chosen.values.mean(axis=1).min() == 1.0 and chosen.values.mean(axis=1).max() == 10.0
    again = select_series_by_volume(wide, 3)
    assert np.array_equal(chosen.values, again.values) and list(chosen.series) == list(again.series)
    assert select_series_by_volume(wide, 10).n_series == 10
    for bad in (0, 11):
        with pytest.raises(ValueError, match="between 1 and 10"):
            select_series_by_volume(wide, bad)


def test_month_dummies() -> None:
    dummies = month_dummies(pd.to_datetime(["2016-01-15", "2016-02-01", "2016-12-31"]))
    assert dummies.shape == (3, 11)
    assert dummies[0].sum() == 0  # January is the baseline month
    assert dummies[1, 0] == 1 and dummies[2, -1] == 1 and dummies.sum(axis=1).tolist() == [0, 1, 1]


def test_forecaster_shapes_clipping_and_fit_window() -> None:
    history = np.tile(np.arange(100, dtype=float), (2, 1))
    dates = pd.date_range("2016-01-01", periods=100)
    future = pd.date_range("2016-04-10", periods=10)
    seen = {}

    def model(y, d, fd):
        seen["n"] = len(y)
        return np.full(len(fd), -5.0)  # negative sales are impossible and must be clipped

    out = SeriesModelForecaster("neg", model, fit_days=40)(history, dates, future)
    assert out.shape == (2, 10) and (out == 0).all() and seen["n"] == 40


def test_failed_fits_fall_back_to_seasonal_naive_and_are_counted() -> None:
    rng = np.random.default_rng(42)
    history = rng.poisson(20, size=(3, 120)).astype(float)
    dates = pd.date_range("2016-01-01", periods=120)
    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=14)
    calls = {"n": 0}

    def flaky(y, d, fd):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("did not converge")
        if calls["n"] == 2:
            return np.full(len(fd), np.nan)  # a non-finite forecast is also a failure
        return np.full(len(fd), 7.0)

    forecaster = SeriesModelForecaster("flaky", flaky, fit_days=100)
    out = forecaster(history, dates, future)
    fallback = seasonal_naive_7(history, dates, future)
    assert forecaster.failures == 2
    assert np.array_equal(out[0], fallback[0]) and np.array_equal(out[1], fallback[1]) and (out[2] == 7.0).all()


def test_missing_library_is_not_swallowed() -> None:
    def needs_library(y, d, fd):
        raise ImportError("no module")

    with pytest.raises(ImportError):
        SeriesModelForecaster("lib", needs_library, 50)(np.ones((1, 60)), pd.date_range("2016-01-01", periods=60),
                                                         pd.date_range("2016-03-01", periods=5))


def test_build_respects_names_and_skips_missing_libraries() -> None:
    with patch.dict(sys.modules, {"prophet": None, "statsmodels": None}):
        assert build_statistical_models(StatisticalConfig()) == {}  # nothing importable, nothing built
    with patch.dict(sys.modules, {"prophet": None}):
        built = build_statistical_models(StatisticalConfig(models=("prophet_default",)))
        assert built == {}


def test_statistical_config_validation() -> None:
    with pytest.raises(ValueError, match="subset"):
        StatisticalConfig(models=("arima_magic",))
    with pytest.raises(ValueError, match="n_series"):
        StatisticalConfig(n_series=0)
    with pytest.raises(ValueError, match="at least 400"):
        StatisticalConfig(sarima_fit_days=200)
    with pytest.raises(ValueError, match="period must be 7"):
        StatisticalConfig(sarima_seasonal_order=(1, 1, 1, 12))


@pytest.mark.parametrize("name", ["holt_winters", "sarima"])
def test_weekly_models_forecast_a_stable_series(name: str) -> None:
    pytest.importorskip("statsmodels")
    dates = pd.date_range("2016-01-01", periods=500, freq="D")
    rng = np.random.default_rng(42)
    y = 100 * WEEKDAY_FACTOR[dates.day_of_week.to_numpy()] + rng.normal(0, 2, 500)
    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=28)
    model = build_statistical_models(StatisticalConfig(models=(name,)))[name]
    forecast = model(y[None, :], dates, future)
    truth = 100 * WEEKDAY_FACTOR[future.day_of_week.to_numpy()]
    assert forecast.shape == (1, 28) and model.failures == 0
    assert wape(truth, forecast[0]) < 12.0  # the weekly pattern is learned


@pytest.mark.parametrize("name", ["sarimax_month", "prophet_month"])
def test_month_aware_models_learn_the_monthly_steps(name: str) -> None:
    library = "statsmodels" if name.startswith("sarimax") else "prophet"
    pytest.importorskip(library)
    y = _staircase(DATES, noise=2.0)
    truth = MONTH_LEVEL[FUTURE.month.to_numpy() - 1] * WEEKDAY_FACTOR[FUTURE.day_of_week.to_numpy()]
    model = build_statistical_models(StatisticalConfig(models=(name,)))[name]
    forecast = model(y[None, :], DATES, FUTURE)
    # December level is 60 but January is 50 and February 55: a model without month steps would miss badly.
    assert forecast.shape == (1, 60) and np.isfinite(forecast).all() and model.failures == 0
    assert wape(truth, forecast[0]) < 25.0
