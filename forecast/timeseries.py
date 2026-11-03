"""Time series statistics: ACF/PACF (numpy), decomposition strength and stationarity tests.

ACF and PACF are implemented with numpy so they have no heavy dependency. STL/MSTL
decomposition and the ADF/KPSS tests use statsmodels, which is imported lazily.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_series(x: Sequence[float], nlags: int) -> np.ndarray:
    """Validate a 1-D finite series that is long enough for ``nlags`` lags."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1:
        raise ValueError("Series must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("Series must be finite (no NaN or inf)")
    if nlags < 1 or len(values) <= nlags:
        raise ValueError("nlags must be positive and smaller than the series length")
    if np.ptp(values) == 0:
        raise ValueError("Series is constant; autocorrelation is undefined")
    return values


def autocorrelation(x: Sequence[float], nlags: int) -> np.ndarray:
    """Sample autocorrelation for lags 0..nlags, computed with an FFT (O(n log n)).

    Uses the standard biased estimator (divide by n), the same convention as statsmodels.
    """
    values = _check_series(x, nlags)
    n = len(values)
    centred = values - values.mean()
    size = 1 << (2 * n - 1).bit_length()  # zero-pad so the circular product equals the linear one
    spectrum = np.fft.rfft(centred, size)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), size)[: nlags + 1] / n
    return acov / acov[0]


def partial_autocorrelation(x: Sequence[float], nlags: int) -> np.ndarray:
    """Partial autocorrelation for lags 0..nlags via the Levinson-Durbin recursion."""
    r = autocorrelation(x, nlags)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    phi_prev = np.array([])
    for k in range(1, nlags + 1):
        if k == 1:
            phi_kk = r[1]
            phi = np.array([phi_kk])
        else:
            phi_kk = (r[k] - np.dot(phi_prev, r[k - 1:0:-1])) / (1.0 - np.dot(phi_prev, r[1:k]))
            phi = np.concatenate([phi_prev - phi_kk * phi_prev[::-1], [phi_kk]])
        pacf[k] = phi_kk
        phi_prev = phi
    return pacf


def component_strength(component: Sequence[float], residual: Sequence[float]) -> float:
    """Strength of a trend or seasonal component, from 0 (none) to 1 (dominant).

    Defined as max(0, 1 - Var(residual) / Var(component + residual)).

    Raises:
        ValueError: If lengths differ or the combined variance is zero.
    """
    comp, resid = np.asarray(component, dtype=float), np.asarray(residual, dtype=float)
    if comp.shape != resid.shape:
        raise ValueError("component and residual must have the same shape")
    total_var = np.var(comp + resid)
    if total_var == 0:
        raise ValueError("Variance of component + residual is zero")
    return float(max(0.0, 1.0 - np.var(resid) / total_var))


def stationarity_verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    """Combine ADF (null: unit root) and KPSS (null: stationary) into one verdict."""
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p > alpha
    if adf_stationary and kpss_stationary:
        return "stationary"
    if not adf_stationary and not kpss_stationary:
        return "non-stationary"
    if adf_stationary and not kpss_stationary:
        return "trend-stationary (detrend)"
    return "inconclusive"


def mstl_components(series: pd.Series, periods: Tuple[int, ...]) -> pd.DataFrame:
    """Decompose a daily series into trend, one seasonal term per period and residual (MSTL).

    Raises:
        ValueError: If the series is shorter than two cycles of the longest period.
        ImportError: If statsmodels is not installed.
    """
    from statsmodels.tsa.seasonal import MSTL  # lazy: heavy optional dependency

    if len(series) < 2 * max(periods):
        raise ValueError(f"Series needs at least {2 * max(periods)} points for periods {periods}")
    result = MSTL(series.astype(float), periods=list(periods)).fit()
    seasonal = np.asarray(result.seasonal, dtype=float)
    if seasonal.ndim == 1:
        seasonal = seasonal[:, None]
    out = pd.DataFrame({"observed": series.astype(float).to_numpy(),
                        "trend": np.asarray(result.trend, dtype=float)}, index=series.index)
    for column, period in enumerate(periods):
        out[f"seasonal_{period}"] = seasonal[:, column]
    out["resid"] = np.asarray(result.resid, dtype=float)
    return out


def decomposition_strengths(components: pd.DataFrame) -> Dict[str, float]:
    """Trend and seasonal strengths from MSTL components (see :func:`component_strength`)."""
    resid = components["resid"].to_numpy()
    strengths = {"trend": component_strength(components["trend"].to_numpy(), resid)}
    for column in components.columns:
        if column.startswith("seasonal_"):
            strengths[column] = component_strength(components[column].to_numpy(), resid)
    return strengths


def adf_kpss(x: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Run the ADF and KPSS tests on one series and return statistics, p-values and a verdict.

    Raises:
        ImportError: If statsmodels is not installed.
        ValueError: If the series is too short.
    """
    from statsmodels.tsa.stattools import adfuller, kpss  # lazy: heavy optional dependency

    values = np.asarray(x, dtype=float)
    if len(values) < 30 or not np.isfinite(values).all():
        raise ValueError("Need at least 30 finite observations for stationarity tests")
    adf_stat, adf_p = adfuller(values, regression="c", autolag="AIC")[:2]
    with warnings.catch_warnings():
        # KPSS reports p-values clipped to the table range [0.01, 0.10] with a warning.
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p = kpss(values, regression="c", nlags="auto")[:2]
    return {
        "adf_stat": float(adf_stat), "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p),
        "verdict": stationarity_verdict(float(adf_p), float(kpss_p), alpha),
    }


def stationarity_table(series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """ADF/KPSS results for the level, first difference, weekly difference and both."""
    transforms = {
        "level": series,
        "diff(1)": series.diff(1),
        "diff(7)": series.diff(7),
        "diff(7) then diff(1)": series.diff(7).diff(1),
    }
    rows = []
    for name, transformed in transforms.items():
        rows.append({"transformation": name, **adf_kpss(transformed.dropna().to_numpy(), alpha)})
    return pd.DataFrame(rows)
