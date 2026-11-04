"""Step 4 pipeline: run the baselines on every validation window and report the results.

Run:  python -m forecast.run_baselines
Writes reports/baseline_results.csv, reports/baseline_summary.json and three figures.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend so the pipeline runs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast.backtest import noise_floor, run_backtest, summarize_backtest  # noqa: E402
from forecast.baselines import Forecaster, build_baselines  # noqa: E402
from forecast.config import ForecastConfig  # noqa: E402
from forecast.data import load_prepared  # noqa: E402
from forecast.logging_utils import configure_logging  # noqa: E402
from forecast.splits import make_validation_windows  # noqa: E402
from forecast.wide import WideData, to_wide, window_arrays  # noqa: E402

logger = logging.getLogger(__name__)


def plot_wape_by_window(scores: pd.DataFrame, floor_by_window: Dict[str, float], benchmark: str,
                        path: Path, dpi: int) -> None:
    """Grouped bars of WAPE per window and baseline, with the estimated noise floor as a marker."""
    table = scores[scores["scope"] == "all"].pivot(index="window_end", columns="baseline", values="wape")
    table = table[sorted(table.columns, key=lambda c: -table[c].mean())]
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.8 / len(table.columns)
    for i, name in enumerate(table.columns):
        positions = np.arange(len(table)) + i * width
        ax.bar(positions, table[name], width, label=name, edgecolor="black" if name == benchmark else None)
    centre = np.arange(len(table)) + 0.4 - width / 2
    for i, window_end in enumerate(table.index):  # one black line per window across its group of bars
        ax.hlines(floor_by_window[window_end], i - width / 2, i + 0.8 - width / 2, colors="black",
                  linewidth=2.5, label="estimated noise floor" if i == 0 else None)
    ax.set_xticks(centre, table.index)
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Baseline accuracy by validation window (lower is better; benchmark outlined)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_wape_by_horizon(scores: pd.DataFrame, path: Path, dpi: int) -> None:
    """WAPE per horizon bucket (days ahead), averaged over windows, one line per baseline."""
    table = scores[scores["scope"] != "all"].groupby(["scope", "baseline"])["wape"].mean().unstack("baseline")
    table = table.loc[sorted(table.index, key=lambda s: int(s.split("-")[0]))]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in table.columns:
        ax.plot(table.index, table[name], marker="o", label=name)
    ax.set_xlabel("days ahead")
    ax.set_ylabel("mean WAPE (%)")
    ax.set_title("Accuracy by forecast horizon")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_total_forecast(wide: WideData, window, forecasters: Dict[str, Forecaster], path: Path, dpi: int) -> None:
    """Total daily sales: actual versus selected baselines for one validation window."""
    arrays = window_arrays(wide, window)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(arrays.future_dates, arrays.actual.sum(axis=0), color="black", linewidth=2, label="actual")
    for name, forecaster in forecasters.items():
        forecast = forecaster(arrays.history, arrays.history_dates, arrays.future_dates)
        ax.plot(arrays.future_dates, forecast.sum(axis=0), linewidth=1.3, label=name)
    ax.set_title(f"Total daily sales, window ending {window.valid_end.date()}")
    ax.set_ylabel("units")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_baselines(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Run the Step 4 backtest end to end, write the outputs and return the summary."""
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    frame = load_prepared(cfg.paths.prepared_parquet) if df is None else df
    wide = to_wide(frame, cfg)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)
    forecasters = build_baselines(cfg.baselines)

    scores = run_backtest(wide, windows, forecasters, cfg)
    floor = {w.valid_end.date().isoformat(): noise_floor(*_floor_inputs(wide, w))["estimated_floor_wape"]
             for w in windows}
    summary = summarize_backtest(scores, cfg, floor)

    scores.round(4).to_csv(cfg.paths.baseline_results, index=False)
    cfg.paths.baseline_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures, dpi = cfg.paths.figures_dir, cfg.eda.figure_dpi
    plot_wape_by_window(scores, floor, cfg.benchmark, figures / "baseline_wape_by_window.png", dpi)
    plot_wape_by_horizon(scores, figures / "baseline_wape_by_horizon.png", dpi)
    shown = {name: forecasters[name] for name in dict.fromkeys([cfg.benchmark, summary["best_model"]])}
    plot_total_forecast(wide, windows[0], shown, figures / "baseline_total_forecast.png", dpi)
    logger.info("Best baseline %s (mean WAPE %.2f%%); estimated noise floor %.2f%%", summary["best_model"],
                summary["baselines"][summary["best_model"]]["mean_wape"], summary["noise_floor"]["mean"])
    return summary


def _floor_inputs(wide: WideData, window):
    """(actuals, future dates) of one window, for the noise-floor estimate."""
    arrays = window_arrays(wide, window)
    return arrays.actual, arrays.future_dates


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the Step 4 baseline backtest.")
    parser.add_argument("--prepared-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = ForecastConfig()
    if args.reports_dir:
        cfg = replace(cfg, paths=replace(cfg.paths, reports_dir=args.reports_dir))
    configure_logging(args.log_level)
    run_baselines(cfg, load_prepared(args.prepared_path or cfg.paths.prepared_parquet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
