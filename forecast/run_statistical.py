"""Step 5 pipeline: compare statistical models with the baselines on a sample of series.

Run:  python -m forecast.run_statistical            (add --n-series 5 for a quicker run)
Writes reports/statistical_results.csv, reports/statistical_summary.json and three figures.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from forecast.backtest import noise_floor, run_backtest, summarize_backtest
from forecast.baselines import Forecaster, build_baselines
from forecast.config import ForecastConfig
from forecast.data import load_prepared
from forecast.logging_utils import configure_logging
from forecast.run_baselines import _floor_inputs, plot_total_forecast, plot_wape_by_horizon, plot_wape_by_window
from forecast.splits import make_validation_windows
from forecast.statistical import SeriesModelForecaster, build_statistical_models, select_series_by_volume
from forecast.wide import to_wide

logger = logging.getLogger(__name__)


def run_statistical(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None,
                    models: Optional[Mapping[str, Forecaster]] = None) -> Dict[str, Any]:
    """Backtest baselines and statistical models on the sampled series and write the outputs.

    Args:
        cfg: Project configuration.
        df: Prepared table from Step 2; read from parquet when omitted.
        models: Forecasters to use instead of the configured statistical models (used by tests).

    Returns:
        The summary dict that is also written to JSON.

    Raises:
        RuntimeError: If no statistical model could be built.
    """
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    frame = load_prepared(cfg.paths.prepared_parquet) if df is None else df
    wide = to_wide(frame, cfg)
    sample = select_series_by_volume(wide, cfg.statistical.n_series)
    windows = make_validation_windows(sample.dates[0], sample.dates[-1], cfg)

    baselines = build_baselines(cfg.baselines)
    statistical: Mapping[str, Forecaster] = build_statistical_models(cfg.statistical) if models is None else models
    if not statistical:
        raise RuntimeError("No statistical model could be built; install statsmodels and/or prophet")
    forecasters = {**baselines, **statistical}

    scores = run_backtest(sample, windows, forecasters, cfg)
    floor = {w.valid_end.date().isoformat(): noise_floor(*_floor_inputs(sample, w))["estimated_floor_wape"]
             for w in windows}
    summary = summarize_backtest(scores, cfg, floor)

    means = {name: v["mean_wape"] for name, v in summary["baselines"].items()}
    best_baseline = min(baselines, key=means.__getitem__)
    best_statistical = min(statistical, key=means.__getitem__)
    summary["best_baseline"] = best_baseline
    summary["best_statistical"] = best_statistical
    summary["statistical_vs_best_baseline_pct"] = {
        name: round(100 * (1 - means[name] / means[best_baseline]), 2) for name in statistical}
    summary["sample"] = {"series": [[int(store), int(item)] for store, item in sample.series],
                         "n_series": sample.n_series}
    summary["fits"] = {name: {"failures": model.failures, "seconds": round(model.fit_seconds, 1)}
                       for name, model in statistical.items() if isinstance(model, SeriesModelForecaster)}

    scores.round(4).to_csv(cfg.paths.statistical_results, index=False)
    cfg.paths.statistical_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures, dpi = cfg.paths.figures_dir, cfg.eda.figure_dpi
    plot_wape_by_window(scores, floor, cfg.benchmark, figures / "statistical_wape_by_window.png", dpi)
    plot_wape_by_horizon(scores, figures / "statistical_wape_by_horizon.png", dpi)
    shown = {name: forecasters[name] for name in dict.fromkeys([cfg.benchmark, best_baseline, best_statistical])}
    plot_total_forecast(sample, windows[0], shown, figures / "statistical_total_forecast.png", dpi)
    logger.info("Best baseline %s (%.2f%%); best statistical model %s (%.2f%%) on %d series", best_baseline,
                means[best_baseline], best_statistical, means[best_statistical], sample.n_series)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the Step 5 statistical-model comparison.")
    parser.add_argument("--prepared-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--n-series", type=int, default=None, help="Number of sampled series (default 10)")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of statistical models to run")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = ForecastConfig()
    statistical = cfg.statistical
    if args.n_series:
        statistical = replace(statistical, n_series=args.n_series)
    if args.models:
        statistical = replace(statistical, models=tuple(args.models))
    paths = replace(cfg.paths, reports_dir=args.reports_dir) if args.reports_dir else cfg.paths
    configure_logging(args.log_level)
    cfg = replace(cfg, statistical=statistical, paths=paths)
    run_statistical(cfg, load_prepared(args.prepared_path or cfg.paths.prepared_parquet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
