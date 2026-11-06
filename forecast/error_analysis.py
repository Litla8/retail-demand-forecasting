"""Error analysis: where is the model weak? By series, store, item, horizon day and calendar.

Everything works on the actuals and forecasts of all validation windows laid side by side
([n_series, n_windows * horizon]) and is vectorised, so it runs instantly on 500 series.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend so the pipeline runs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast.baselines import Forecaster  # noqa: E402
from forecast.splits import ValidationWindow  # noqa: E402
from forecast.wide import WideData, window_arrays  # noqa: E402

DOM_BINS = [0, 7, 14, 21, 31]
DOM_LABELS = ["1-7", "8-14", "15-21", "22-31"]


@dataclass
class RecordingForecaster:
    """Wraps a forecaster and remembers its output per history length, so the backtest's forecasts
    can be analysed afterwards without fitting again."""

    forecaster: Forecaster
    records: Dict[int, np.ndarray] = field(default_factory=dict)

    def __call__(self, history: np.ndarray, history_dates: pd.DatetimeIndex,
                 future_dates: pd.DatetimeIndex) -> np.ndarray:
        forecast = self.forecaster(history, history_dates, future_dates)
        self.records[history.shape[1]] = forecast
        return forecast


def stack_windows(wide: WideData, windows: Sequence[ValidationWindow],
                  recorder: RecordingForecaster) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Lay all windows side by side: actuals, forecasts, dates and days-ahead.

    Raises:
        KeyError: If the forecaster was not run on one of the windows.
    """
    actuals, forecasts, dates, horizon_days = [], [], [], []
    for window in windows:
        arrays = window_arrays(wide, window)
        actuals.append(arrays.actual)
        forecasts.append(recorder.records[arrays.history.shape[1]])
        dates.append(arrays.future_dates)
        horizon_days.append(np.arange(1, len(arrays.future_dates) + 1))
    return (np.concatenate(actuals, axis=1), np.concatenate(forecasts, axis=1),
            dates[0].append(dates[1:]) if len(dates) > 1 else dates[0], np.concatenate(horizon_days))


def series_table(actual: np.ndarray, forecast: np.ndarray, series: pd.Index) -> pd.DataFrame:
    """Per-series totals and WAPE/bias across every window (columns: store, item, volume, ...)."""
    if actual.shape != forecast.shape or actual.shape[0] != len(series):
        raise ValueError("actual, forecast and series must align")
    keys = pd.DataFrame(list(series), columns=["store", "item"])
    return keys.assign(
        volume=actual.mean(axis=1),
        abs_err_sum=np.abs(forecast - actual).sum(axis=1),
        actual_sum=actual.sum(axis=1),
        signed_sum=(forecast - actual).sum(axis=1),
    ).assign(wape=lambda t: 100 * t["abs_err_sum"] / t["actual_sum"],
             bias=lambda t: 100 * t["signed_sum"] / t["actual_sum"])


def group_wape(table: pd.DataFrame, key: str) -> pd.DataFrame:
    """WAPE and bias per store or per item (sums first, then the ratio, so large series weigh more)."""
    grouped = table.groupby(key)[["abs_err_sum", "actual_sum", "signed_sum"]].sum()
    return pd.DataFrame({"wape": 100 * grouped["abs_err_sum"] / grouped["actual_sum"],
                         "bias": 100 * grouped["signed_sum"] / grouped["actual_sum"]}).reset_index()


def _column_groups(actual: np.ndarray, forecast: np.ndarray, keys: Sequence[Any]) -> pd.DataFrame:
    """WAPE and bias per group of forecast days (columns are summed over all series first)."""
    frame = pd.DataFrame({"key": list(keys), "err": np.abs(forecast - actual).sum(axis=0),
                          "act": actual.sum(axis=0), "signed": (forecast - actual).sum(axis=0)})
    grouped = frame.groupby("key", observed=True)[["err", "act", "signed"]].sum()
    return pd.DataFrame({"wape": 100 * grouped["err"] / grouped["act"],
                         "bias": 100 * grouped["signed"] / grouped["act"]}).reset_index()


def horizon_table(actual: np.ndarray, forecast: np.ndarray, horizon_days: np.ndarray) -> pd.DataFrame:
    """WAPE and bias by days ahead (1..horizon)."""
    return _column_groups(actual, forecast, horizon_days).rename(columns={"key": "day"})


def calendar_tables(actual: np.ndarray, forecast: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    """WAPE and bias by weekday (Monday = 0), month and week of the month."""
    dom = pd.cut(dates.day.to_numpy(), DOM_BINS, labels=DOM_LABELS)
    return {
        "weekday": _column_groups(actual, forecast, dates.dayofweek.to_numpy()).rename(columns={"key": "weekday"}),
        "month": _column_groups(actual, forecast, dates.month.to_numpy()).rename(columns={"key": "month"}),
        "day_of_month": _column_groups(actual, forecast, dom).rename(columns={"key": "group"}),
    }


def analyze(actual: np.ndarray, forecast: np.ndarray, baseline_forecast: np.ndarray, dates: pd.DatetimeIndex,
            horizon_days: np.ndarray, series: pd.Index) -> Dict[str, Any]:
    """Full error analysis. Returns JSON-friendly summaries plus the tables needed for plots."""
    table = series_table(actual, forecast, series)
    base_table = series_table(actual, baseline_forecast, series)
    by_horizon = horizon_table(actual, forecast, horizon_days)
    calendar = calendar_tables(actual, forecast, dates)
    by_store, by_item = group_wape(table, "store"), group_wape(table, "item")
    slope_per_90 = float(np.polyfit(by_horizon["day"], by_horizon["wape"], 1)[0] * 90)
    ranked = table.sort_values("wape")

    def rows(frame: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
        return frame[list(columns)].round(3).to_dict(orient="records")

    summary = {
        "series_wape": {"min": round(float(table["wape"].min()), 2), "median": round(float(table["wape"].median()), 2),
                        "max": round(float(table["wape"].max()), 2), "std": round(float(table["wape"].std()), 2)},
        "wape_vs_volume_spearman": round(float(table["volume"].corr(table["wape"], method="spearman")), 3),
        "worst_series": rows(ranked.tail(5).iloc[::-1], ["store", "item", "volume", "wape", "bias"]),
        "best_series": rows(ranked.head(5), ["store", "item", "volume", "wape", "bias"]),
        "share_series_better_than_baseline": round(float((table["wape"] < base_table["wape"]).mean()), 3),
        "by_store": rows(by_store, ["store", "wape", "bias"]),
        "items_wape": {"min": round(float(by_item["wape"].min()), 2), "max": round(float(by_item["wape"].max()), 2),
                       "worst_5": rows(by_item.sort_values("wape").tail(5).iloc[::-1], ["item", "wape", "bias"])},
        "horizon_day": {"first_day_wape": round(float(by_horizon["wape"].iloc[0]), 2),
                        "last_day_wape": round(float(by_horizon["wape"].iloc[-1]), 2),
                        "trend_points_over_horizon": round(slope_per_90, 3)},
        "by_weekday": rows(calendar["weekday"], ["weekday", "wape", "bias"]),
        "by_month": rows(calendar["month"], ["month", "wape", "bias"]),
        "by_day_of_month": rows(calendar["day_of_month"], ["group", "wape", "bias"]),
    }
    return {"summary": summary, "series": table, "horizon": by_horizon, "calendar": calendar}


def plot_store_item_heatmap(table: pd.DataFrame, path: Path, dpi: int) -> None:
    """Heatmap of WAPE per store (rows) and item (columns)."""
    grid = table.pivot(index="store", columns="item", values="wape")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    image = ax.imshow(grid.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("item")
    ax.set_ylabel("store")
    ax.set_title("WAPE (%) per store and item")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_volume_scatter(table: pd.DataFrame, path: Path, dpi: int) -> None:
    """WAPE against average daily sales per series (small series are noisier)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(table["volume"], table["wape"], s=14, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("average daily sales (log scale)")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Error by series volume")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_horizon_calendar(horizon: pd.DataFrame, calendar: Dict[str, pd.DataFrame], path: Path, dpi: int) -> None:
    """WAPE by days ahead, by weekday and by month."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(horizon["day"], horizon["wape"])
    axes[0].set_xlabel("days ahead")
    axes[0].set_title("WAPE by horizon day")
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], calendar["weekday"]["wape"])
    axes[1].set_title("WAPE by weekday")
    axes[2].bar(calendar["month"]["month"].astype(str), calendar["month"]["wape"])
    axes[2].set_title("WAPE by month")
    for ax in axes:
        ax.set_ylabel("WAPE (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
