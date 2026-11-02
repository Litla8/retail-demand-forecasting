"""Tests for forecast.metrics (deterministic synthetic data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.metrics import (
    add_horizon_bucket,
    bias,
    error_by_group,
    evaluate_forecast,
    improvement_over_baseline,
    mae,
    mape,
    rmse,
    smape,
    wape,
)

ACTUAL = np.array([10.0, 20.0, 30.0])
FORECAST = np.array([12.0, 18.0, 30.0])


def test_known_values() -> None:
    assert mae(ACTUAL, FORECAST) == pytest.approx(4 / 3)
    assert rmse(ACTUAL, FORECAST) == pytest.approx(np.sqrt(8 / 3))
    assert wape(ACTUAL, FORECAST) == pytest.approx(100 * 4 / 60)
    assert mape(ACTUAL, FORECAST) == pytest.approx(10.0)
    assert smape(ACTUAL, FORECAST) == pytest.approx(100 * (2 / 11 + 2 / 19 + 0) / 3)


def test_bias_sign_means_over_forecast() -> None:
    assert bias(ACTUAL, ACTUAL + 3) > 0
    assert bias(ACTUAL, ACTUAL - 3) < 0
    assert bias(ACTUAL, FORECAST) == pytest.approx(0.0)  # +2 and -2 cancel


def test_smape_handles_zeros() -> None:
    assert smape([0.0, 10.0], [0.0, 10.0]) == 0.0  # 0 vs 0 is a perfect forecast
    assert smape([0.0], [5.0]) == pytest.approx(200.0)  # the scale maximum


def test_mape_ignores_zero_actuals_and_rejects_all_zero() -> None:
    assert mape([0.0, 10.0], [3.0, 12.0]) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="every actual"):
        mape([0.0, 0.0], [1.0, 1.0])


def test_wape_and_bias_reject_zero_total_demand() -> None:
    with pytest.raises(ValueError, match="undefined"):
        wape([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="undefined"):
        bias([0.0, 0.0], [1.0, 2.0])


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        mae([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        mae([], [])
    with pytest.raises(ValueError, match="finite"):
        mae([1.0, np.nan], [1.0, 2.0])


def test_evaluate_forecast_matches_individual_metrics() -> None:
    rng = np.random.default_rng(42)
    actual = rng.integers(10, 100, size=1000).astype(float)
    forecast = actual + rng.normal(0, 5, size=1000)
    result = evaluate_forecast(actual, forecast)
    assert set(result) == {"wape", "smape", "mae", "rmse", "mape", "bias"}
    assert result["wape"] == pytest.approx(wape(actual, forecast))
    assert all(np.isfinite(v) for v in result.values())


def test_improvement_over_baseline() -> None:
    assert improvement_over_baseline(14.76, 16.4) == pytest.approx(10.0, abs=0.01)
    assert improvement_over_baseline(18.0, 16.0) < 0  # worse than the baseline
    with pytest.raises(ValueError):
        improvement_over_baseline(10.0, 0.0)


def test_horizon_buckets() -> None:
    labels = add_horizon_bucket(pd.Series([1, 28, 29, 60, 61, 90]), (28, 60, 90))
    assert labels.astype(str).tolist() == ["1-28", "1-28", "29-60", "29-60", "61-90", "61-90"]
    with pytest.raises(ValueError, match="within"):
        add_horizon_bucket(pd.Series([0, 5]), (28, 60, 90))
    with pytest.raises(ValueError, match="within"):
        add_horizon_bucket(pd.Series([91]), (28, 60, 90))


def test_error_by_group() -> None:
    df = pd.DataFrame(
        {
            "store": [1, 1, 2, 2],
            "sales": [10.0, 30.0, 20.0, 20.0],
            "forecast": [12.0, 30.0, 20.0, 10.0],
        }
    )
    out = error_by_group(df, ["store"]).set_index("store")
    assert out.loc[1, "rows"] == 2
    assert out.loc[1, "wape"] == pytest.approx(100 * 2 / 40)
    assert out.loc[2, "wape"] == pytest.approx(100 * 10 / 40)
    assert out.loc[2, "bias"] == pytest.approx(-25.0)  # under-forecast
    with pytest.raises(ValueError, match="Missing columns"):
        error_by_group(df, ["item"])
