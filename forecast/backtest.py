"""Backtesting: score forecasters on every validation window, overall and by horizon bucket."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from forecast.baselines import Forecaster
from forecast.config import ForecastConfig
from forecast.metrics import bias, improvement_over_baseline, mae, rmse, smape, wape
from forecast.splits import ValidationWindow
from forecast.wide import WideData, window_arrays

logger = logging.getLogger(__name__)

SCORE_COLUMNS = ("baseline", "window_end", "scope", "wape", "smape", "mae", "rmse", "bias", "days")


def scope_slices(edges: Sequence[int]) -> Dict[str, slice]:
    """Column slices for the whole horizon ('all') and each bucket such as '1-28', '29-60'."""
    horizon = edges[-1]
    lowers = [0, *edges[:-1]]
    slices: Dict[str, slice] = {"all": slice(0, horizon)}
    for lower, upper in zip(lowers, edges):
        slices[f"{lower + 1}-{upper}"] = slice(lower, upper)
    return slices


def score_forecast(actual: np.ndarray, forecast: np.ndarray, edges: Sequence[int]) -> List[Dict[str, Any]]:
    """Score one forecast matrix overall and per horizon bucket.

    Raises:
        ValueError: If the matrices differ in shape or do not span the full horizon.
    """
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")
    if actual.shape[1] != edges[-1]:
        raise ValueError(f"Forecast spans {actual.shape[1]} days but the horizon is {edges[-1]}")
    rows = []
    for scope, columns in scope_slices(edges).items():
        a, f = actual[:, columns], forecast[:, columns]
        rows.append({"scope": scope, "wape": wape(a, f), "smape": smape(a, f), "mae": mae(a, f),
                     "rmse": rmse(a, f), "bias": bias(a, f), "days": int(a.shape[1])})
    return rows


def run_backtest(wide: WideData, windows: Sequence[ValidationWindow],
                 forecasters: Mapping[str, Forecaster], cfg: ForecastConfig) -> pd.DataFrame:
    """Run every forecaster on every validation window and return one row per score.

    Each forecaster receives only the history up to the window's ``train_end``.
    """
    rows: List[Dict[str, Any]] = []
    for window in windows:
        arrays = window_arrays(wide, window)
        for name, forecaster in forecasters.items():
            forecast = forecaster(arrays.history, arrays.history_dates, arrays.future_dates)
            for row in score_forecast(arrays.actual, forecast, cfg.horizon_buckets):
                rows.append({"baseline": name, "window_end": window.valid_end.date().isoformat(), **row})
        logger.info("Scored %d forecasters on the window ending %s", len(forecasters), window.valid_end.date())
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def noise_floor(actual: np.ndarray, future_dates: pd.DatetimeIndex) -> Dict[str, float]:
    """Estimate the lowest WAPE any forecaster could reach (irreducible daily noise).

    Because the level is flat inside a month, a day's own noise is what is left after predicting
    it from the other days with the same weekday in the same month of the same series. That
    leave-one-out error is inflated by the noise in the reference mean (a factor of
    sqrt(1 + 1/(n - 1)) for n matching days), so the estimate divides it back out. It is an
    estimate, not an exact bound.

    Raises:
        ValueError: If no day has at least one other matching day.
    """
    key = (future_dates.year.to_numpy() * 100 + future_dates.month.to_numpy()) * 10 + future_dates.dayofweek.to_numpy()
    _, codes = np.unique(key, return_inverse=True)
    loo = np.full(actual.shape, np.nan)
    inflation = np.ones(actual.shape[1])
    for code in np.unique(codes):
        columns = np.where(codes == code)[0]
        n = len(columns)
        if n < 2:
            continue
        total = actual[:, columns].sum(axis=1, keepdims=True)
        loo[:, columns] = (total - actual[:, columns]) / (n - 1)
        inflation[columns] = 1.0 + 1.0 / (n - 1)
    usable = ~np.isnan(loo[0])
    if not usable.any():
        raise ValueError("No day has another matching weekday in the same month")
    loo_wape = float(100.0 * np.abs(loo[:, usable] - actual[:, usable]).sum() / actual[:, usable].sum())
    return {"loo_wape": loo_wape, "estimated_floor_wape": float(loo_wape / np.sqrt(inflation[usable].mean())),
            "days_used": int(usable.sum())}


def summarize_backtest(scores: pd.DataFrame, cfg: ForecastConfig, floor_by_window: Mapping[str, float]) -> Dict[str, Any]:
    """Build the JSON summary: per-baseline accuracy, stability, improvement and headroom."""
    benchmark = cfg.benchmark
    overall = scores[scores["scope"] == "all"]
    if benchmark not in set(overall["baseline"]):
        raise ValueError(f"Benchmark {benchmark!r} was not part of the backtest")
    wape_table = overall.pivot(index="baseline", columns="window_end", values="wape")
    bench = wape_table.loc[benchmark]
    low, high = cfg.target_improvement_pct

    per_baseline: Dict[str, Any] = {}
    for name in wape_table.index:
        improvement = [improvement_over_baseline(w, b) for w, b in zip(wape_table.loc[name], bench)]
        rows = overall[overall["baseline"] == name]
        per_baseline[name] = {
            "wape_by_window": {k: round(float(v), 3) for k, v in wape_table.loc[name].items()},
            "mean_wape": round(float(wape_table.loc[name].mean()), 3),
            "std_wape": round(float(wape_table.loc[name].std(ddof=0)), 3),
            "mean_smape": round(float(rows["smape"].mean()), 3),
            "bias_by_window": {r.window_end: round(float(r.bias), 2) for r in rows.itertuples()},
            "mean_improvement_vs_benchmark_pct": round(float(np.mean(improvement)), 2),
        }
    by_horizon = (scores[scores["scope"] != "all"].groupby(["baseline", "scope"])["wape"].mean().round(3)
                  .unstack("scope").to_dict(orient="index"))
    best = str(wape_table.mean(axis=1).idxmin())
    floor_mean = float(np.mean(list(floor_by_window.values())))
    best_wape = per_baseline[best]["mean_wape"]
    return {
        "windows": list(wape_table.columns),
        "benchmark": benchmark,
        "baselines": per_baseline,
        "wape_by_horizon_bucket": by_horizon,
        "best_baseline": best,
        "target": {
            "improvement_pct_range": [low, high],
            "wape_needed_by_window": {w: {"for_low": round(float(v) * (1 - low / 100), 3),
                                          "for_high": round(float(v) * (1 - high / 100), 3)}
                                      for w, v in bench.items()},
        },
        "noise_floor": {"by_window": {k: round(v, 3) for k, v in floor_by_window.items()}, "mean": round(floor_mean, 3)},
        "headroom": {"best_baseline_mean_wape": best_wape, "estimated_floor_mean_wape": round(floor_mean, 3),
                     "max_further_improvement_pct": round(100 * (1 - floor_mean / best_wape), 2)},
    }
