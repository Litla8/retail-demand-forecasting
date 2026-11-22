"""Tests for forecast.gbm_features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.config import GbmConfig
from forecast.gbm_features import MAX_SAFE_HORIZON, FeatureBuilder, feature_names

DATES = pd.date_range("2013-01-01", periods=1000, freq="D")
CFG = GbmConfig(engine="hist_gb")


def _history(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(40, size=(4, len(DATES))).astype(float) + 1.0


def test_feature_names_match_the_matrix() -> None:
    builder = FeatureBuilder(_history(), DATES, CFG)
    features, y, weight = builder.training_rows(30, np.random.default_rng(42))
    assert list(features.columns) == feature_names(CFG) and len(features) == len(y) == len(weight)
    assert np.isfinite(y).all() and (weight > 0).all()
    assert features["horizon"].between(1, 30).all() and features["month"].between(1, 12).all()
    assert len(features) <= 4 * (len(DATES) - CFG.min_origin - 1)


def test_constant_series_gives_neutral_features() -> None:
    builder = FeatureBuilder(np.full((2, len(DATES)), 50.0), DATES, CFG)
    series = np.array([0, 1])
    origin = np.array([800, 800])
    target = DATES[origin + 10]
    features, scale = builder.features(series, origin, target, np.array([10, 10]))
    assert np.allclose(scale, 50.0) and np.allclose(features["log_scale"], np.log(50.0))
    for column in ("mean28_ratio", "mean91_ratio", "growth28", "growth91", "growth_year", "lag364",
                   "cell1", "month1", "cell2", "month2"):
        assert np.allclose(features[column], 1.0), column  # a flat series is its own average everywhere


def test_features_never_use_data_after_the_origin() -> None:
    history = _history()
    full = FeatureBuilder(history, DATES, CFG)
    rng = np.random.default_rng(0)
    series, horizon = rng.integers(0, 4, 300), rng.integers(1, 91, 300)
    origin = rng.integers(CFG.min_origin, len(DATES) - 91, 300)
    target = DATES[origin + horizon]
    x_full, s_full = full.features(series, origin, target, horizon)
    cut = int(origin.max()) + 1  # drop everything after the latest origin
    x_cut, s_cut = FeatureBuilder(history[:, :cut], DATES[:cut], CFG).features(series, origin, target, horizon)
    assert np.array_equal(x_full.isna().to_numpy(), x_cut.isna().to_numpy())
    assert np.allclose(x_full.to_numpy(), x_cut.to_numpy(), equal_nan=True) and np.allclose(s_full, s_cut)


def test_forecast_rows_layout() -> None:
    builder = FeatureBuilder(_history(), DATES, CFG)
    future = pd.date_range(DATES[-1] + pd.Timedelta(days=1), periods=14)
    features, scale = builder.forecast_rows(future)
    assert features.shape == (4 * 14, len(feature_names(CFG))) and scale.shape == (56,)
    assert features["horizon"].to_numpy()[:14].tolist() == list(range(1, 15))  # series-major, days ascending
    assert features["dayofweek"].to_numpy()[:14].tolist() == future.day_of_week.tolist()


def test_series_ids_are_optional() -> None:
    cfg = GbmConfig(engine="hist_gb", use_series_ids=True)
    features, _, _ = FeatureBuilder(_history(), DATES, cfg).training_rows(10, np.random.default_rng(1))
    assert "series_id" in features.columns and set(features["series_id"]) == {0, 1, 2, 3}
    assert "series_id" not in feature_names(CFG)


def test_input_validation() -> None:
    history = _history()
    builder = FeatureBuilder(history, DATES, CFG)
    with pytest.raises(ValueError, match="origins must be"):
        builder.features(np.array([0]), np.array([100]), DATES[[110]], np.array([10]))
    with pytest.raises(ValueError, match="horizon must be"):
        builder.features(np.array([0]), np.array([600]), DATES[[900]], np.array([MAX_SAFE_HORIZON + 1]))
    with pytest.raises(ValueError, match="match dates"):
        FeatureBuilder(history[:, :-1], DATES, CFG)
    with pytest.raises(ValueError, match="finite"):
        FeatureBuilder(np.where(history > 60, np.nan, history), DATES, CFG)
    with pytest.raises(ValueError, match="too short"):
        FeatureBuilder(history[:, :400], DATES[:400], CFG).training_rows(10, np.random.default_rng(1))
