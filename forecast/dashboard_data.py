"""Data behind the dashboard: backtest export, the compact bundle and the chart functions.

The dashboard only reads a small bundle (``dashboard_data/``) built from the pipeline outputs, so it
can be deployed without the raw data or the models. Everything here is plain pandas/matplotlib so it
can be tested without Streamlit.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend so the functions work without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast.config import ForecastConfig  # noqa: E402
from forecast.data import load_prepared  # noqa: E402
from forecast.splits import ValidationWindow  # noqa: E402
from forecast.wide import WideData, to_wide, window_arrays  # noqa: E402

logger = logging.getLogger(__name__)
HISTORY_DAYS = 400
BACKTEST_NAMES = {"seasonal_naive_364": "benchmark", "month_weekday_profile_multi_year": "baseline", "selected_model": "model"}
GBM_VARIANTS = ("gbm_default", "gbm_tuned", "gbm_tuned_calibrated", "gbm_tuned_blend", "gbm_tuned_calibrated_blend")


def save_backtest(records: Mapping[str, Sequence], wide: WideData, windows: Sequence[ValidationWindow], path: Path) -> None:
    """Write actuals and the three forecasters' forecasts for every validation window (long format, gzip CSV).

    ``records`` maps forecaster name to a list of ``(actual, forecast)`` arrays, one per window.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = pd.DataFrame(list(wide.series), columns=["store", "item"])
    frames = []
    for index, window in enumerate(windows):
        arrays = window_arrays(wide, window)
        n_days = len(arrays.future_dates)
        part = pd.DataFrame({
            "window_end": window.valid_end.date().isoformat(),
            "store": np.repeat(keys["store"].to_numpy(), n_days), "item": np.repeat(keys["item"].to_numpy(), n_days),
            "date": np.tile(arrays.future_dates.to_numpy(), len(keys)),
            "horizon_day": np.tile(np.arange(1, n_days + 1), len(keys)),
            "actual": arrays.actual.ravel(),
        })
        for name, label in BACKTEST_NAMES.items():
            if name in records:
                part[label] = records[name][index][1].ravel()
        frames.append(part)
    table = pd.concat(frames, ignore_index=True)
    numeric = [c for c in table.columns if c in ("actual", *BACKTEST_NAMES.values())]
    table[numeric] = table[numeric].round(2)  # dates and ids are left alone
    table.to_csv(path, index=False, compression="gzip")


def export_bundle(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None, out_dir: Optional[Path] = None) -> Path:
    """Collect the files the dashboard needs into one small folder.

    Raises:
        FileNotFoundError: If a required pipeline output is missing (the message says which step creates it).
    """
    out = Path(out_dir) if out_dir is not None else cfg.paths.dashboard_dir
    out.mkdir(parents=True, exist_ok=True)
    required = {cfg.paths.forecast_intervals: "run_business", cfg.paths.reorder_plan: "run_business",
                cfg.paths.backtest_forecasts: "run_business", cfg.paths.business_summary: "run_business",
                cfg.paths.step7_summary: "run_tuning", cfg.paths.baseline_summary: "run_baselines"}
    for path, step in required.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} not found; run `python -m forecast.{step}` first")
    frame = load_prepared(cfg.paths.prepared_parquet) if df is None else df
    wide = to_wide(frame, cfg)
    tail = slice(max(0, wide.n_days - HISTORY_DAYS), wide.n_days)
    keys = pd.DataFrame(list(wide.series), columns=["store", "item"])
    n_days = len(wide.dates[tail])
    history = pd.DataFrame({"date": np.tile(wide.dates[tail].to_numpy(), len(keys)),
                            "store": np.repeat(keys["store"].to_numpy(), n_days),
                            "item": np.repeat(keys["item"].to_numpy(), n_days),
                            "sales": wide.values[:, tail].ravel().astype(int)})
    history.to_csv(out / "history.csv.gz", index=False, compression="gzip")
    for source, name in ((cfg.paths.forecast_intervals, "forecast_intervals.csv"), (cfg.paths.reorder_plan, "reorder_plan.csv"),
                         (cfg.paths.backtest_forecasts, "backtest_forecasts.csv.gz")):
        shutil.copy(source, out / name)
    summaries = {"baseline": json.loads(Path(cfg.paths.baseline_summary).read_text()),
                 "step7": json.loads(Path(cfg.paths.step7_summary).read_text()),
                 "business": json.loads(Path(cfg.paths.business_summary).read_text())}
    for optional, key in ((cfg.paths.statistical_summary, "statistical"), (cfg.paths.gbm_summary, "gbm")):
        if Path(optional).is_file():
            summaries[key] = json.loads(Path(optional).read_text())
    if Path(cfg.paths.monitoring_log).is_file():
        shutil.copy(cfg.paths.monitoring_log, out / "monitoring_log.csv")
    (out / "summaries.json").write_text(json.dumps(summaries), encoding="utf-8")
    logger.info("Dashboard bundle written to %s", out)
    return out


@dataclass(frozen=True)
class Bundle:
    """Everything the dashboard shows, loaded into memory."""

    history: pd.DataFrame
    forecast: pd.DataFrame
    plan: pd.DataFrame
    backtest: pd.DataFrame
    summaries: Dict[str, Any]
    monitoring: pd.DataFrame


def load_bundle(path: Path) -> Bundle:
    """Read a bundle written by :func:`export_bundle`.

    Raises:
        FileNotFoundError: If the folder or one of its core files is missing.
    """
    path = Path(path)
    for name in ("history.csv.gz", "forecast_intervals.csv", "reorder_plan.csv", "backtest_forecasts.csv.gz", "summaries.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"{path / name} not found. Run `python -m forecast.dashboard_data` first.")
    monitoring = path / "monitoring_log.csv"
    return Bundle(
        history=pd.read_csv(path / "history.csv.gz", parse_dates=["date"]),
        forecast=pd.read_csv(path / "forecast_intervals.csv", parse_dates=["date"]),
        plan=pd.read_csv(path / "reorder_plan.csv", parse_dates=["order_date"]),
        backtest=pd.read_csv(path / "backtest_forecasts.csv.gz", parse_dates=["date"]),
        summaries=json.loads((path / "summaries.json").read_text(encoding="utf-8")),
        monitoring=pd.read_csv(monitoring) if monitoring.is_file() else pd.DataFrame(),
    )


def series_slice(frame: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
    """Rows of one store-item series."""
    return frame[(frame["store"] == store) & (frame["item"] == item)]


def model_comparison(summaries: Mapping[str, Any]) -> pd.DataFrame:
    """One table of mean WAPE per model across the steps, best first, with the selected model flagged.

    Equal scores are ordered simplest first, so a tie never puts a more complex variant above the one that was
    actually selected (all 500 series where available).
    """
    rows = []
    for name, value in summaries["baseline"]["baselines"].items():
        rows.append({"model": name, "family": "baseline", "mean_wape": value["mean_wape"], "complexity": 0})
    for rank, name in enumerate(GBM_VARIANTS, start=1):
        wape = summaries["step7"]["variants_mean_wape"].get(name)
        if wape is not None:
            rows.append({"model": name, "family": "gradient boosting", "mean_wape": wape, "complexity": rank})
    table = pd.DataFrame(rows).drop_duplicates("model")
    benchmark = float(table.loc[table["model"] == "seasonal_naive_364", "mean_wape"].iloc[0])
    table["vs_benchmark_pct"] = (100 * (1 - table["mean_wape"] / benchmark)).round(1)
    table["selected"] = table["model"] == summaries["step7"].get("selected_variant")
    return table.sort_values(["mean_wape", "complexity"]).drop(columns="complexity").reset_index(drop=True)


def backtest_errors(backtest: pd.DataFrame, by: Any, model: str = "model") -> pd.DataFrame:
    """WAPE and bias of one forecaster grouped by a column (store, item, window_end, weekday ...)."""
    work = backtest.assign(weekday=backtest["date"].dt.dayofweek, month=backtest["date"].dt.month)
    work["abs_err"] = (work[model] - work["actual"]).abs()
    work["signed"] = work[model] - work["actual"]
    grouped = work.groupby(by)[["abs_err", "signed", "actual"]].sum()
    return pd.DataFrame({"wape": 100 * grouped["abs_err"] / grouped["actual"],
                         "bias": 100 * grouped["signed"] / grouped["actual"]}).reset_index()


def at_risk_table(plan: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Series whose first order needs the biggest safety buffer relative to expected demand."""
    first = plan[plan["order_date"] == plan["order_date"].min()].copy()
    first["buffer_pct"] = 100 * first["safety_stock"] / first["forecast_units"].clip(lower=1e-9)
    return first.sort_values("buffer_pct", ascending=False).head(top)[
        ["store", "item", "forecast_units", "safety_stock", "order_up_to", "buffer_pct"]].reset_index(drop=True)


def plot_series_forecast(history: pd.DataFrame, forecast: pd.DataFrame, title: str, history_days: int = 120) -> plt.Figure:
    """Recent actuals and the forecast with 80% and 95% intervals for one series."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    recent = history.sort_values("date").tail(history_days)
    ax.plot(recent["date"], recent["sales"], color="black", linewidth=1, label="actual")
    ax.plot(forecast["date"], forecast["forecast"], color="tab:blue", label="forecast")
    for level, alpha in ((95, 0.15), (80, 0.3)):
        if f"lower_{level}" in forecast:
            ax.fill_between(forecast["date"], forecast[f"lower_{level}"], forecast[f"upper_{level}"], color="tab:blue",
                            alpha=alpha, label=f"{level}% interval")
    ax.set_title(title)
    ax.set_ylabel("units per day")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_backtest_series(backtest: pd.DataFrame, window_end: str, title: str) -> plt.Figure:
    """Actual versus model and benchmark forecasts for one series in one validation window."""
    part = backtest[backtest["window_end"] == window_end].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(part["date"], part["actual"], color="black", linewidth=1, label="actual")
    for column, color in (("benchmark", "tab:orange"), ("model", "tab:blue")):
        if column in part:
            ax.plot(part["date"], part[column], color=color, label=column)
    ax.set_title(title)
    ax.set_ylabel("units per day")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_error_bars(table: pd.DataFrame, key: str, title: str) -> plt.Figure:
    """Bar chart of WAPE by a grouping column."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(table[key].astype(str), table["wape"], color="tab:blue")
    ax.set_ylabel("WAPE (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_store_item_heatmap(table: pd.DataFrame) -> plt.Figure:
    """WAPE per store (rows) and item (columns); the argument is the output of ``backtest_errors(by=['store','item'])``."""
    grid = table.pivot(index="store", columns="item", values="wape")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    image = ax.imshow(grid.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("item")
    ax.set_ylabel("store")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def main() -> int:
    """CLI entry point: ``python -m forecast.dashboard_data``."""
    from forecast.logging_utils import configure_logging

    configure_logging("INFO")
    export_bundle(ForecastConfig())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
