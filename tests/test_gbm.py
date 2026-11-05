"""Tests for forecast.gbm (the LightGBM test skips when lightgbm is not installed)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast.config import GbmConfig
from forecast.gbm import GbmForecaster, build_regressor
from forecast.gbm_features import feature_names
from forecast.metrics import wape

DATES = pd.date_range("2014-01-01", "2016-12-31", freq="D")  # 1,096 days
FUTURE = pd.date_range("2017-01-01", periods=60, freq="D")
MONTH_LEVEL = np.array([50, 55, 70, 90, 100, 110, 120, 100, 90, 80, 85, 60], dtype=float)
WEEKDAY_FACTOR = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.2, 1.3])
SCALES = np.array([0.5, 1.0, 2.0])  # three series of different size sharing one seasonal shape
SMALL = replace(GbmConfig(engine="hist_gb"), n_estimators=120, min_child_samples=30)


def _series(dates: pd.DatetimeIndex, noise: float = 0.0, seed: int = 42) -> np.ndarray:
    year_growth = 1.05 ** (dates.year.to_numpy() - 2014)
    shape = MONTH_LEVEL[dates.month.to_numpy() - 1] * WEEKDAY_FACTOR[dates.dayofweek.to_numpy()] * year_growth
    rng = np.random.default_rng(seed)
    return SCALES[:, None] * shape[None, :] * (1 + rng.normal(0, noise, (3, len(dates))))


def test_forecast_shape_nonnegative_and_accuracy_on_a_clean_staircase() -> None:
    history = _series(DATES, noise=0.02)
    truth = _series(DATES.append(FUTURE))[:, len(DATES):]
    forecast = GbmForecaster(SMALL, max_horizon=60)(history, DATES, FUTURE)
    assert forecast.shape == (3, 60) and (forecast >= 0).all() and np.isfinite(forecast).all()
    assert wape(truth, forecast) < 12.0  # learns the shared month-by-weekday shape and each series' scale


def test_forecast_is_repeatable() -> None:
    history = _series(DATES, noise=0.05)
    model = GbmForecaster(SMALL, max_horizon=60)
    assert np.array_equal(model(history, DATES, FUTURE), model(history, DATES, FUTURE))


def test_horizon_and_settings_are_validated() -> None:
    history = _series(DATES)
    with pytest.raises(ValueError, match="exceeds max_horizon"):
        GbmForecaster(SMALL, max_horizon=30)(history, DATES, FUTURE)
    with pytest.raises(ValueError, match="max_horizon"):
        GbmForecaster(SMALL, max_horizon=500)
    with pytest.raises(ValueError, match="engine"):
        GbmConfig(engine="xgboost")
    with pytest.raises(ValueError, match="min_origin"):
        GbmConfig(min_origin=100)
    with pytest.raises(ValueError, match="subsample"):
        GbmConfig(subsample=0.0)


def test_importance_is_only_available_for_lightgbm() -> None:
    history = _series(DATES, noise=0.02)
    model = GbmForecaster(SMALL, max_horizon=60)
    model(history, DATES, FUTURE)
    assert model.importance is None and model.fit_seconds > 0  # scikit-learn's engine has no gain importance


def test_lightgbm_engine_trains_and_reports_importance() -> None:
    pytest.importorskip("lightgbm")
    cfg = replace(GbmConfig(engine="lightgbm"), n_estimators=120, min_child_samples=30, n_jobs=1)
    history = _series(DATES, noise=0.02)
    truth = _series(DATES.append(FUTURE))[:, len(DATES):]
    model = GbmForecaster(cfg, max_horizon=60)
    forecast = model(history, DATES, FUTURE)
    assert wape(truth, forecast) < 12.0 and build_regressor(cfg, 42).get_params()["random_state"] == 42
    assert set(model.importance) == set(feature_names(cfg)) and sum(model.importance.values()) > 0
