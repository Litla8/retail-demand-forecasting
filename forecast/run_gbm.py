"""Step 6 pipeline: global gradient-boosting model versus every baseline on all 500 series.

Run:  python -m forecast.run_gbm
Writes reports/gbm_results.csv, reports/gbm_summary.json and figures in reports/figures/.
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
import pandas as pd  # noqa: E402

from forecast.backtest import noise_floor, run_backtest, summarize_backtest  # noqa: E402
from forecast.baselines import Forecaster, build_baselines  # noqa: E402
from forecast.config import ForecastConfig  # noqa: E402
from forecast.data import load_prepared  # noqa: E402
from forecast.gbm import GbmForecaster  # noqa: E402
from forecast.logging_utils import configure_logging  # noqa: E402
from forecast.run_baselines import _floor_inputs, plot_total_forecast, plot_wape_by_horizon, plot_wape_by_window  # noqa: E402
from forecast.splits import make_validation_windows  # noqa: E402
from forecast.wide import to_wide  # noqa: E402

logger = logging.getLogger(__name__)
MODEL_NAME = "gbm"


def plot_importance(importance: Dict[str, float], path: Path, dpi: int) -> None:
    """Horizontal bars of gain-based feature importance (share of total gain)."""
    total = sum(importance.values())
    ordered = sorted(importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([name for name, _ in ordered], [100 * value / total for _, value in ordered], color="tab:blue")
    ax.set_xlabel("share of total gain (%)")
    ax.set_title("Feature importance (latest validation window)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_gbm(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None, gbm: Optional[Forecaster] = None) -> Dict[str, Any]:
    """Backtest the baselines and the global boosting model on every window and write the outputs.

    Args:
        cfg: Project configuration.
        df: Prepared table from Step 2; read from parquet when omitted.
        gbm: Forecaster to use instead of the configured boosting model (used by tests).
    """
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    frame = load_prepared(cfg.paths.prepared_parquet) if df is None else df
    wide = to_wide(frame, cfg)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)

    baselines = build_baselines(cfg.baselines)
    model = gbm if gbm is not None else GbmForecaster(cfg.gbm, cfg.horizon_days, cfg.random_state)
    forecasters = {**baselines, MODEL_NAME: model}

    scores = run_backtest(wide, windows, forecasters, cfg)
    floor = {w.valid_end.date().isoformat(): noise_floor(*_floor_inputs(wide, w))["estimated_floor_wape"]
             for w in windows}
    summary = summarize_backtest(scores, cfg, floor)

    means = {name: v["mean_wape"] for name, v in summary["baselines"].items()}
    best_baseline = min(baselines, key=means.__getitem__)
    summary["best_baseline"] = best_baseline
    summary["model"] = {
        "name": MODEL_NAME, "engine": cfg.gbm.engine, "mean_wape": means[MODEL_NAME],
        "vs_best_baseline_pct": round(100 * (1 - means[MODEL_NAME] / means[best_baseline]), 2),
        "vs_benchmark_pct": summary["baselines"][MODEL_NAME]["mean_improvement_vs_benchmark_pct"],
        "fit_seconds": round(getattr(model, "fit_seconds", 0.0), 1),
        "feature_importance_share_pct": None,
    }
    importance = getattr(model, "importance", None)
    figures, dpi = cfg.paths.figures_dir, cfg.eda.figure_dpi
    if importance:
        total = sum(importance.values())
        summary["model"]["feature_importance_share_pct"] = {k: round(100 * v / total, 2) for k, v in
                                                             sorted(importance.items(), key=lambda kv: -kv[1])}
        plot_importance(importance, figures / "gbm_feature_importance.png", dpi)

    scores.round(4).to_csv(cfg.paths.gbm_results, index=False)
    cfg.paths.gbm_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_wape_by_window(scores, floor, cfg.benchmark, figures / "gbm_wape_by_window.png", dpi)
    plot_wape_by_horizon(scores, figures / "gbm_wape_by_horizon.png", dpi)
    shown = {name: forecasters[name] for name in dict.fromkeys([cfg.benchmark, best_baseline, MODEL_NAME])}
    plot_total_forecast(wide, windows[0], shown, figures / "gbm_total_forecast.png", dpi)
    logger.info("gbm (%s) mean WAPE %.2f%% vs best baseline %s %.2f%% (%+.1f%%)", cfg.gbm.engine, means[MODEL_NAME],
                best_baseline, means[best_baseline], summary["model"]["vs_best_baseline_pct"])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the Step 6 gradient-boosting backtest.")
    parser.add_argument("--prepared-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--engine", choices=("lightgbm", "hist_gb"), default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = ForecastConfig()
    if args.engine:
        cfg = replace(cfg, gbm=replace(cfg.gbm, engine=args.engine))
    if args.reports_dir:
        cfg = replace(cfg, paths=replace(cfg.paths, reports_dir=args.reports_dir))
    configure_logging(args.log_level)
    run_gbm(cfg, load_prepared(args.prepared_path or cfg.paths.prepared_parquet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
