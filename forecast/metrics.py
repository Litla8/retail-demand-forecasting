"""Forecast accuracy metrics.

WAPE is the primary metric: it weights every unit sold equally, so it is not distorted by
low-volume series the way MAPE is. sMAPE is also reported because it is the metric used by
the Kaggle Store Item Demand Forecasting competition.
"""
from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _as_pair(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert actuals and forecasts to aligned float arrays.

    Raises:
        ValueError: If shapes differ, the input is empty, or any value is not finite.
    """
    actual = np.asarray(y_true, dtype=float)
    forecast = np.asarray(y_pred, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")
    if actual.size == 0:
        raise ValueError("Cannot compute a metric on empty input")
    if not (np.isfinite(actual).all() and np.isfinite(forecast).all()):
        raise ValueError("Actuals and forecasts must be finite (no NaN or inf)")
    return actual, forecast


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error, in units sold."""
    actual, forecast = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(forecast - actual)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error, in units sold (penalises large misses more than MAE)."""
    actual, forecast = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((forecast - actual) ** 2)))


def wape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Weighted absolute percentage error in percent: sum|error| / sum(actual) * 100.

    Raises:
        ValueError: If total actual demand is zero (the ratio is undefined).
    """
    actual, forecast = _as_pair(y_true, y_pred)
    total = np.abs(actual).sum()
    if total == 0:
        raise ValueError("WAPE is undefined when total actual demand is zero")
    return float(100.0 * np.abs(forecast - actual).sum() / total)


def bias(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Signed forecast bias in percent: positive means over-forecasting (excess stock risk).

    Raises:
        ValueError: If total actual demand is zero.
    """
    actual, forecast = _as_pair(y_true, y_pred)
    total = np.abs(actual).sum()
    if total == 0:
        raise ValueError("Bias is undefined when total actual demand is zero")
    return float(100.0 * (forecast - actual).sum() / total)


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute percentage error in percent, ignoring rows where the actual is zero.

    MAPE is undefined at zero demand, so those rows are excluded rather than inflated.

    Raises:
        ValueError: If every actual value is zero.
    """
    actual, forecast = _as_pair(y_true, y_pred)
    nonzero = actual != 0
    if not nonzero.any():
        raise ValueError("MAPE is undefined when every actual value is zero")
    if not nonzero.all():
        logger.debug("MAPE ignored %d rows with zero actuals", int((~nonzero).sum()))
    return float(100.0 * np.mean(np.abs(forecast[nonzero] - actual[nonzero]) / np.abs(actual[nonzero])))


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Symmetric MAPE on the 0-200 scale used by Kaggle: 100 * mean(|F-A| / ((|A|+|F|)/2)).

    A row where both actual and forecast are zero counts as a perfect forecast (0).
    """
    actual, forecast = _as_pair(y_true, y_pred)
    denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
    ratio = np.divide(np.abs(forecast - actual), denominator,
                      out=np.zeros_like(denominator), where=denominator > 0)
    return float(100.0 * ratio.mean())


def improvement_over_baseline(model_error: float, baseline_error: float) -> float:
    """Relative error reduction versus a baseline, in percent (positive = better).

    Raises:
        ValueError: If the baseline error is not positive.
    """
    if baseline_error <= 0:
        raise ValueError("baseline_error must be positive")
    return float(100.0 * (baseline_error - model_error) / baseline_error)


def evaluate_forecast(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Return every headline metric for one set of actuals and forecasts."""
    return {
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def add_horizon_bucket(horizon_day: pd.Series, edges: Sequence[int]) -> pd.Series:
    """Label days-ahead values with buckets such as '1-28', '29-60', '61-90'.

    Args:
        horizon_day: Days ahead of the forecast origin (1 = first day forecast).
        edges: Strictly increasing upper edges, for example (28, 60, 90).

    Raises:
        ValueError: If a value falls outside 1..edges[-1] or edges are not increasing.
    """
    if list(edges) != sorted(set(edges)):
        raise ValueError("edges must be strictly increasing")
    if horizon_day.min() < 1 or horizon_day.max() > edges[-1]:
        raise ValueError(f"horizon_day must be within 1..{edges[-1]}")
    lowers = [1] + [edge + 1 for edge in edges[:-1]]
    labels = [f"{low}-{high}" for low, high in zip(lowers, edges)]
    return pd.cut(horizon_day, bins=[0, *edges], labels=labels)


def error_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    actual_col: str = "sales",
    pred_col: str = "forecast",
) -> pd.DataFrame:
    """Compute WAPE, sMAPE, MAE and bias per group (store, item, horizon bucket, ...).

    Fully vectorised with a single groupby, so it scales to the 913,000-row training set.

    Raises:
        ValueError: If required columns are missing or values are not finite.
    """
    missing = [c for c in (*group_cols, actual_col, pred_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    actual, forecast = _as_pair(df[actual_col], df[pred_col])
    abs_err = np.abs(forecast - actual)
    denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
    work = df[list(group_cols)].copy()
    work["actual"] = actual
    work["abs_err"] = abs_err
    work["signed_err"] = forecast - actual
    work["smape_term"] = 100.0 * np.divide(abs_err, denominator,
                                           out=np.zeros_like(denominator), where=denominator > 0)
    grouped = work.groupby(list(group_cols), observed=True).agg(
        rows=("actual", "size"),
        actual_sum=("actual", "sum"),
        abs_err_sum=("abs_err", "sum"),
        signed_err_sum=("signed_err", "sum"),
        smape=("smape_term", "mean"),
    )
    safe_total = grouped["actual_sum"].where(grouped["actual_sum"] > 0)
    grouped["wape"] = 100.0 * grouped["abs_err_sum"] / safe_total
    grouped["bias"] = 100.0 * grouped["signed_err_sum"] / safe_total
    grouped["mae"] = grouped["abs_err_sum"] / grouped["rows"]
    return grouped[["rows", "mae", "wape", "smape", "bias"]].reset_index()
