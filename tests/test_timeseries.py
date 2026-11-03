"""Tests for forecast.timeseries (statsmodels-dependent tests skip when it is missing)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.timeseries import (
    autocorrelation,
    component_strength,
    decomposition_strengths,
    partial_autocorrelation,
    stationarity_verdict,
)


def _ar1(phi: float, n: int = 5000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def test_acf_matches_the_direct_definition() -> None:
    x = np.random.default_rng(42).normal(size=60)
    centred = x - x.mean()
    direct = np.array([np.sum(centred[: len(x) - k] * centred[k:]) for k in range(11)]) / np.sum(centred ** 2)
    assert np.allclose(autocorrelation(x, 10), direct)  # FFT result equals the textbook sum


def test_acf_and_pacf_of_an_ar1_process() -> None:
    x = _ar1(0.7)
    acf, pacf = autocorrelation(x, 5), partial_autocorrelation(x, 5)
    assert acf[0] == pytest.approx(1.0) and abs(acf[1] - 0.7) < 0.05
    assert abs(pacf[1] - 0.7) < 0.05 and abs(pacf[2]) < 0.06  # PACF cuts off after lag 1


def test_acf_finds_a_weekly_cycle() -> None:
    t = np.arange(700)
    acf = autocorrelation(np.sin(2 * np.pi * t / 7), 14)
    assert acf[7] > 0.9 and acf[3] < 0


def test_acf_input_validation() -> None:
    with pytest.raises(ValueError, match="constant"):
        autocorrelation(np.ones(50), 5)
    with pytest.raises(ValueError, match="nlags"):
        autocorrelation(np.arange(10.0), 10)
    with pytest.raises(ValueError, match="finite"):
        autocorrelation(np.array([1.0, np.nan, 2.0, 3.0, 4.0]), 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        autocorrelation(np.ones((4, 4)), 2)


def test_component_strength() -> None:
    rng = np.random.default_rng(42)
    seasonal = np.sin(np.linspace(0, 40 * np.pi, 500)) * 10
    assert component_strength(seasonal, rng.normal(0, 0.5, 500)) > 0.95
    assert component_strength(rng.normal(0, 0.1, 500), rng.normal(0, 5, 500)) < 0.05
    assert component_strength(seasonal, np.zeros(500)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same shape"):
        component_strength([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="zero"):
        component_strength([1.0, 1.0], [0.0, 0.0])


def test_decomposition_strengths_reads_every_seasonal_column() -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"trend": np.linspace(0, 10, 300), "seasonal_7": np.sin(np.arange(300)),
                          "resid": rng.normal(0, 0.1, 300)})
    assert set(decomposition_strengths(frame)) == {"trend", "seasonal_7"}


def test_stationarity_verdict_branches() -> None:
    assert stationarity_verdict(0.001, 0.50) == "stationary"
    assert stationarity_verdict(0.60, 0.01) == "non-stationary"
    assert stationarity_verdict(0.001, 0.01) == "trend-stationary (detrend)"
    assert stationarity_verdict(0.60, 0.50) == "inconclusive"


def test_adf_kpss_separates_noise_from_a_trending_series() -> None:
    pytest.importorskip("statsmodels")
    from forecast.timeseries import adf_kpss

    rng = np.random.default_rng(42)
    noise = adf_kpss(rng.normal(size=1500))
    trending = adf_kpss(np.arange(1500) + rng.normal(size=1500))
    assert noise["adf_p"] < 0.01
    assert trending["adf_p"] > 0.05 and trending["kpss_p"] <= 0.05
    with pytest.raises(ValueError, match="at least 30"):
        adf_kpss(np.arange(10.0))


def test_mstl_components_reconstruct_the_series() -> None:
    pytest.importorskip("statsmodels")
    from forecast.timeseries import mstl_components

    rng = np.random.default_rng(42)
    index = pd.date_range("2016-01-01", periods=400, freq="D")
    values = 100 + 0.05 * np.arange(400) + 10 * np.sin(2 * np.pi * np.arange(400) / 7) + rng.normal(0, 0.5, 400)
    parts = mstl_components(pd.Series(values, index=index), (7,))
    assert list(parts.columns) == ["observed", "trend", "seasonal_7", "resid"]
    assert np.allclose(parts[["trend", "seasonal_7", "resid"]].sum(axis=1), parts["observed"], atol=1e-6)
    assert decomposition_strengths(parts)["seasonal_7"] > 0.9
    with pytest.raises(ValueError, match="at least"):
        mstl_components(pd.Series(values[:100], index=index[:100]), (7, 365))
