"""Step 8 pipeline: turn the forecasts into inventory decisions and quantify the benefit.

Run:  python -m forecast.run_business
Needs models/gbm_params.json from Step 7. Writes reports/business_results.csv, business_summary.json,
reorder_plan.csv, forecast_intervals.csv, submission.csv and figures in reports/figures/.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend so the pipeline runs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast.baselines import Forecaster, build_baselines  # noqa: E402
from forecast.business import (  # noqa: E402
    build_submission,
    cost_optimal,
    evaluate_policies,
    interval_table,
    inventory_at_fill_rate,
    matched_target,
    reorder_plan,
)
from forecast.config import ForecastConfig  # noqa: E402
from forecast.data import load_prepared  # noqa: E402
from forecast.inventory import quantile_table, standardized_errors  # noqa: E402
from forecast.logging_utils import configure_logging  # noqa: E402
from forecast.selected import build_selected_forecaster, load_selected  # noqa: E402
from forecast.splits import make_validation_windows  # noqa: E402
from forecast.wide import WideData, to_wide, window_arrays  # noqa: E402

logger = logging.getLogger(__name__)
BEST_BASELINE = "month_weekday_profile_multi_year"
MODEL_NAME = "selected_model"


def plot_frontier(frontier: pd.DataFrame, path: Path, dpi: int) -> None:
    """Average inventory against fill rate for each forecaster (lower-right is better)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, group in frontier.groupby("forecaster"):
        group = group.sort_values("fill_rate")
        ax.plot(100 * group["fill_rate"], group["avg_inventory"], marker="o", label=name)
    ax.set_xlabel("fill rate (% of demand met from stock)")
    ax.set_ylabel("average units in stock per series per day")
    ax.set_title("Inventory needed for a given service level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_costs(frontier: pd.DataFrame, path: Path, dpi: int) -> None:
    """Total cost per series per day against the service level, per forecaster."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in frontier.groupby("forecaster"):
        ax.plot(100 * group["service_level"], group["total_cost"], marker="o", label=name)
    ax.set_xlabel("service level used for safety stock (%)")
    ax.set_ylabel("holding + stock-out cost per series per day")
    ax.set_title("Cost against service level (placeholder costs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_coverage(frontier: pd.DataFrame, path: Path, dpi: int) -> None:
    """Achieved out-of-sample coverage of the safety stock against its target level."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([75, 100], [75, 100], "k--", linewidth=1, label="perfect")
    for name, group in frontier.groupby("forecaster"):
        ax.plot(100 * group["service_level"], 100 * group["coverage"], marker="o", label=name)
    ax.set_xlabel("target level (%)")
    ax.set_ylabel("achieved coverage on later windows (%)")
    ax.set_title("Are the intervals honest?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_fan_chart(wide: WideData, forecast: np.ndarray, intervals: pd.DataFrame, future_dates: pd.DatetimeIndex,
                   path: Path, dpi: int, history_days: int = 120) -> None:
    """Recent history and the forecast with 80% and 95% intervals for four series of different size."""
    order = np.argsort(wide.values.mean(axis=1))
    picks = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    for ax, row in zip(axes.ravel(), picks):
        store, item = wide.series[row]
        part = intervals[(intervals["store"] == store) & (intervals["item"] == item)]
        ax.plot(wide.dates[-history_days:], wide.values[row, -history_days:], color="black", linewidth=0.9, label="actual")
        ax.plot(future_dates, forecast[row], color="tab:blue", label="forecast")
        for level, alpha in ((95, 0.15), (80, 0.3)):
            if f"lower_{level}" in part:
                ax.fill_between(part["date"], part[f"lower_{level}"], part[f"upper_{level}"], color="tab:blue",
                                alpha=alpha, label=f"{level}% interval")
        ax.set_title(f"store {store}, item {item} (about {wide.values[row].mean():.0f} units a day)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _rounded(table: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Round only the numeric columns (dates are left alone)."""
    out = table.copy()
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].round(digits)
    return out


def _pct_saving(reference: float, value: float) -> Optional[float]:
    """Percent reduction of ``value`` versus ``reference`` (None when either is missing)."""
    if reference is None or value is None or np.isnan(reference) or np.isnan(value) or reference <= 0:
        return None
    return round(100 * (1 - value / reference), 2)


def run_business(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None, model: Optional[Forecaster] = None) -> Dict[str, Any]:
    """Run Step 8 end to end and return the summary that is also written to JSON.

    Args:
        cfg: Project configuration.
        df: Prepared table from Step 2; read from parquet when omitted.
        model: Forecaster to use instead of rebuilding the Step 7 selection (used by tests).
    """
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    biz = cfg.business
    frame = load_prepared(cfg.paths.prepared_parquet) if df is None else df
    wide = to_wide(frame, cfg)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)
    baselines = build_baselines(replace(cfg.baselines, names=(cfg.benchmark, BEST_BASELINE)))
    selected = None
    if model is None:
        selected = load_selected(cfg.paths.gbm_params)
        model = build_selected_forecaster(selected, cfg, baselines[BEST_BASELINE])
    forecasters: Dict[str, Forecaster] = {cfg.benchmark: baselines[cfg.benchmark], BEST_BASELINE: baselines[BEST_BASELINE],
                                          MODEL_NAME: model}

    records: Dict[str, List] = {name: [] for name in forecasters}
    for window in windows:
        arrays = window_arrays(wide, window)
        for name, forecaster in forecasters.items():
            records[name].append((arrays.actual, forecaster(arrays.history, arrays.history_dates, arrays.future_dates)))
        logger.info("Forecasts ready for the window ending %s", window.valid_end.date())

    frontier = evaluate_policies(records, biz)
    frontier.round(5).to_csv(cfg.paths.business_results, index=False)
    optimal = cost_optimal(frontier).set_index("forecaster")
    target = matched_target(frontier, biz.target_fill_rate)
    matched = {name: inventory_at_fill_rate(frontier, name, target) for name in forecasters}
    benchmark, baseline = cfg.benchmark, BEST_BASELINE

    # final forecast for the test period, trained on all labelled history
    test_dates = pd.DatetimeIndex(sorted(frame.loc[frame["is_test"], "date"].unique()))
    final = np.maximum(model(wide.values, wide.dates, test_dates), 0.0)
    level = float(optimal.loc[MODEL_NAME, "service_level"])
    review, protection = biz.review_days, biz.protection_days
    errors = np.concatenate([standardized_errors(a, f, review, protection) for a, f in records[MODEL_NAME]])
    daily = np.concatenate([standardized_errors(a, f, 1, 1) for a, f in records[MODEL_NAME]])
    plan = reorder_plan(final, wide.series, test_dates, quantile_table(errors, [level])[level], level, biz)
    intervals = interval_table(final, wide.series, test_dates, daily, biz.interval_levels)
    _rounded(plan).to_csv(cfg.paths.reorder_plan, index=False)
    _rounded(intervals).to_csv(cfg.paths.forecast_intervals, index=False)
    if "id" in frame.columns and frame["is_test"].any():
        build_submission(frame, final, wide.series, test_dates).round(4).to_csv(cfg.paths.submission, index=False)

    first_review = plan[plan["order_date"] == plan["order_date"].min()].assign(
        buffer_pct=lambda t: 100 * t["safety_stock"] / t["forecast_units"].clip(lower=1e-9))
    at_risk = first_review.sort_values("buffer_pct", ascending=False).head(10)
    summary: Dict[str, Any] = {
        "assumptions": {"lead_time_days": biz.lead_time_days, "review_days": biz.review_days,
                        "holding_cost_per_unit_day": biz.holding_cost_per_unit_day,
                        "stockout_cost_per_unit": biz.stockout_cost_per_unit, "target_fill_rate": biz.target_fill_rate,
                        "target_fill_rate_used": None if np.isnan(target) else round(float(target), 5),
                        "windows_simulated": list(range(biz.first_eval_window, len(windows)))},
        "selected_variant": None if selected is None else selected["variant"],
        "inventory_at_target_fill_rate": {k: (None if np.isnan(v) else round(v, 3)) for k, v in matched.items()},
        "inventory_saving_pct_at_target": {
            "vs_benchmark": _pct_saving(matched[benchmark], matched[MODEL_NAME]),
            "vs_best_baseline": _pct_saving(matched[baseline], matched[MODEL_NAME])},
        "cost_optimal": {name: {"service_level": float(row["service_level"]), "fill_rate": round(float(row["fill_rate"]), 4),
                                "avg_inventory": round(float(row["avg_inventory"]), 3),
                                "total_cost": round(float(row["total_cost"]), 5)} for name, row in optimal.iterrows()},
        "cost_saving_pct_at_optimum": {
            "vs_benchmark": _pct_saving(float(optimal.loc[benchmark, "total_cost"]), float(optimal.loc[MODEL_NAME, "total_cost"])),
            "vs_best_baseline": _pct_saving(float(optimal.loc[baseline, "total_cost"]), float(optimal.loc[MODEL_NAME, "total_cost"]))},
        "coverage_check": {name: [{"target": float(r.service_level), "achieved": round(float(r.coverage), 4)}
                                  for r in group.itertuples()] for name, group in frontier.groupby("forecaster")},
        "plan": {"service_level": level, "orders_per_series": int(plan.groupby(["store", "item"]).size().iloc[0]),
                 "total_forecast_units": round(float(final.sum()), 1),
                 "total_safety_stock_first_review": round(float(first_review["safety_stock"].sum()), 1),
                 "most_buffered_series": [{"store": int(r.store), "item": int(r.item), "buffer_pct": round(float(r.buffer_pct), 1)}
                                          for r in at_risk.itertuples()]},
    }
    cfg.paths.business_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figures, dpi = cfg.paths.figures_dir, cfg.eda.figure_dpi
    plot_frontier(frontier, figures / "business_frontier.png", dpi)
    plot_costs(frontier, figures / "business_costs.png", dpi)
    plot_coverage(frontier, figures / "business_coverage.png", dpi)
    plot_fan_chart(wide, final, intervals, test_dates, figures / "forecast_fan_chart.png", dpi)
    logger.info("Inventory at %.2f%% fill rate: model %.2f vs benchmark %.2f units per series-day",
                100 * target, matched[MODEL_NAME], matched[benchmark])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Step 8: inventory decisions from the forecasts.")
    parser.add_argument("--prepared-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    cfg = ForecastConfig()
    if args.reports_dir:
        cfg = replace(cfg, paths=replace(cfg.paths, reports_dir=args.reports_dir))
    configure_logging(args.log_level)
    run_business(cfg, load_prepared(args.prepared_path or cfg.paths.prepared_parquet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
