"""Business layer: inventory trade-off curves, savings, reorder plan, intervals and the submission file."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from forecast.config import BusinessConfig
from forecast.inventory import (
    block_starts,
    interval_quantiles,
    protection_sums,
    quantile_table,
    simulate_policy,
    standardized_errors,
)

WindowRecord = Tuple[np.ndarray, np.ndarray]  # (actual, forecast), each [n_series, horizon]


def evaluate_policies(records: Mapping[str, Sequence[WindowRecord]], cfg: BusinessConfig) -> pd.DataFrame:
    """Simulate every forecaster at every service level on the evaluation windows.

    The safety-stock quantiles for window k come only from the forecast errors of windows before k, so
    nothing is calibrated on the window it is tested on or on later ones.

    Returns:
        One row per forecaster and service level: fill rate, average inventory, costs and the achieved
        out-of-sample coverage of the demand over the protection interval.

    Raises:
        ValueError: If there are too few windows to calibrate and evaluate.
    """
    lead, review, protection = cfg.lead_time_days, cfg.review_days, cfg.protection_days
    n_windows = {len(v) for v in records.values()}
    if len(n_windows) != 1 or min(n_windows) <= cfg.first_eval_window:
        raise ValueError("every forecaster needs the same number of windows, more than first_eval_window")
    rows: List[Dict[str, float]] = []
    for name, windows in records.items():
        errors = [standardized_errors(a, f, review, protection) for a, f in windows]
        for level in cfg.service_levels:
            totals = dict(demand=0.0, sold=0.0, lost=0.0, inventory_days=0.0, series_days=0.0, hit=0.0, blocks=0.0)
            for k in range(cfg.first_eval_window, len(windows)):
                q = quantile_table(np.concatenate(errors[:k]), [level])[level]
                actual, forecast = windows[k]
                result = simulate_policy(actual, forecast, q, lead, review)
                totals["demand"] += result.demand
                totals["sold"] += result.sold
                totals["lost"] += result.lost
                totals["inventory_days"] += result.inventory_days
                totals["series_days"] += result.n_series * result.n_days
                hits = errors[k] <= q  # the standardised error is within the safety stock
                totals["hit"] += float(hits.sum())
                totals["blocks"] += float(hits.size)
            holding = cfg.holding_cost_per_unit_day * totals["inventory_days"] / totals["series_days"]
            stockout = cfg.stockout_cost_per_unit * totals["lost"] / totals["series_days"]
            rows.append({
                "forecaster": name, "service_level": level,
                "fill_rate": totals["sold"] / totals["demand"],
                "avg_inventory": totals["inventory_days"] / totals["series_days"],
                "lost_units_per_series_day": totals["lost"] / totals["series_days"],
                "holding_cost": holding, "stockout_cost": stockout, "total_cost": holding + stockout,
                "coverage": totals["hit"] / totals["blocks"],
            })
    return pd.DataFrame(rows)


def inventory_at_fill_rate(frontier: pd.DataFrame, forecaster: str, target: float) -> float:
    """Average inventory a forecaster needs to reach ``target`` fill rate (linear interpolation).

    Returns NaN if the target lies outside the fill rates the simulation reached.
    """
    sub = frontier[frontier["forecaster"] == forecaster].sort_values("fill_rate")
    if not sub["fill_rate"].min() <= target <= sub["fill_rate"].max():
        return float("nan")
    return float(np.interp(target, sub["fill_rate"], sub["avg_inventory"]))


def matched_target(frontier: pd.DataFrame, requested: float) -> float:
    """Fill rate at which to compare forecasters: ``requested`` if every forecaster reaches it, else the
    middle of the range they all reach (NaN if the ranges do not overlap)."""
    spans = frontier.groupby("forecaster")["fill_rate"].agg(["min", "max"])
    low, high = float(spans["min"].max()), float(spans["max"].min())
    if low > high:
        return float("nan")
    return requested if low <= requested <= high else (low + high) / 2


def cost_optimal(frontier: pd.DataFrame) -> pd.DataFrame:
    """The service level with the lowest total cost for each forecaster."""
    return frontier.loc[frontier.groupby("forecaster")["total_cost"].idxmin()].reset_index(drop=True)


def reorder_plan(forecast: np.ndarray, series: pd.Index, future_dates: pd.DatetimeIndex,
                 quantile: float, service_level: float, cfg: BusinessConfig) -> pd.DataFrame:
    """Order-up-to levels for every review day of the forecast period.

    Raises:
        ValueError: If the forecast does not match the series and dates.
    """
    if forecast.shape != (len(series), len(future_dates)):
        raise ValueError("forecast must be [n_series, n_dates]")
    horizon = forecast.shape[1]
    prefix = np.concatenate([np.zeros((forecast.shape[0], 1)), np.cumsum(forecast, axis=1)], axis=1)
    keys = pd.DataFrame(list(series), columns=["store", "item"])
    frames = []
    for day in range(0, horizon, cfg.review_days):
        if day + cfg.lead_time_days >= horizon:
            break  # an order placed now could no longer arrive inside the period
        expected = prefix[:, min(day + cfg.protection_days, horizon)] - prefix[:, day]
        safety = np.maximum(quantile * np.sqrt(np.maximum(expected, 1e-9)), 0.0)
        frames.append(keys.assign(order_date=future_dates[day], forecast_units=expected, safety_stock=safety,
                                  order_up_to=expected + safety, service_level=service_level))
    return pd.concat(frames, ignore_index=True)


def interval_table(forecast: np.ndarray, series: pd.Index, future_dates: pd.DatetimeIndex,
                   daily_errors: np.ndarray, levels: Sequence[float]) -> pd.DataFrame:
    """Daily forecasts with central prediction intervals from the empirical error distribution."""
    quantiles = interval_quantiles(daily_errors, levels)
    store = np.repeat([s for s, _ in series], forecast.shape[1])
    item = np.repeat([i for _, i in series], forecast.shape[1])
    table = pd.DataFrame({"store": store, "item": item,
                          "date": np.tile(future_dates.to_numpy(), forecast.shape[0]),
                          "forecast": forecast.ravel()})
    root = np.sqrt(np.maximum(forecast.ravel(), 1e-9))
    for level, (low, high) in quantiles.items():
        label = int(round(level * 100))
        table[f"lower_{label}"] = np.maximum(forecast.ravel() + low * root, 0.0)
        table[f"upper_{label}"] = forecast.ravel() + high * root
    return table


def build_submission(prepared: pd.DataFrame, forecast: np.ndarray, series: pd.Index,
                     future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Kaggle-format (id, sales) table: the forecast joined to the test ids.

    Raises:
        ValueError: If the forecast does not cover exactly the test rows.
    """
    test = prepared.loc[prepared["is_test"], ["id", "date", "store", "item"]]
    long = pd.DataFrame({"store": np.repeat([s for s, _ in series], forecast.shape[1]),
                         "item": np.repeat([i for _, i in series], forecast.shape[1]),
                         "date": np.tile(future_dates.to_numpy(), forecast.shape[0]),
                         "sales": np.maximum(forecast.ravel(), 0.0)})
    merged = test.merge(long, on=["store", "item", "date"], how="left", validate="one_to_one")
    if merged["sales"].isna().any() or len(merged) != len(long):
        raise ValueError("the forecast does not cover exactly the test rows")
    return merged[["id", "sales"]].astype({"id": "int64"}).sort_values("id").reset_index(drop=True)
