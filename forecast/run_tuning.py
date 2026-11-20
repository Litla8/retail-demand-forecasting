"""Step 7 pipeline: tune the boosting model, test bias calibration and blending, analyse the errors.

Run:  python -m forecast.run_tuning
Writes reports/step7_results.csv, reports/step7_summary.json, reports/tuning_trials.csv,
models/gbm_params.json (the selected configuration) and figures in reports/figures/.
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
from forecast.baselines import build_baselines  # noqa: E402
from forecast.config import ForecastConfig  # noqa: E402
from forecast.data import load_prepared  # noqa: E402
from forecast.ensemble import StackedModels  # noqa: E402
from forecast.error_analysis import (  # noqa: E402
    RecordingForecaster,
    analyze,
    plot_horizon_calendar,
    plot_store_item_heatmap,
    plot_volume_scatter,
    stack_windows,
)
from forecast.gbm import GbmForecaster  # noqa: E402
from forecast.logging_utils import configure_logging  # noqa: E402
from forecast.run_baselines import _floor_inputs, plot_wape_by_window  # noqa: E402
from forecast.splits import make_validation_windows  # noqa: E402
from forecast.statistical import select_series_by_volume  # noqa: E402
from forecast.tuning import tune_gbm  # noqa: E402
from forecast.wide import WideData, to_wide  # noqa: E402

logger = logging.getLogger(__name__)
BEST_BASELINE = "month_weekday_profile_multi_year"
# From simplest to most complex; a later variant is chosen only if it clearly beats the earlier ones.
COMPLEXITY_ORDER = ("gbm_default", "gbm_tuned", "gbm_tuned_calibrated", "gbm_tuned_blend", "gbm_tuned_calibrated_blend")


def select_variant(means: Dict[str, float], min_gain: float) -> str:
    """Pick the simplest variant that is not clearly beaten (needs ``min_gain`` WAPE points to switch up)."""
    chosen = COMPLEXITY_ORDER[0]
    for name in COMPLEXITY_ORDER[1:]:
        if means[name] < means[chosen] - min_gain:
            chosen = name
    return chosen


def plot_trials(trials: pd.DataFrame, path: Path, dpi: int) -> None:
    """Trial scores with the running best, to show whether the search found anything."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trials["trial"], trials["value"], "o", alpha=0.7, label="trial")
    ax.plot(trials["trial"], trials["value"].cummin(), color="tab:red", label="best so far")
    ax.axhline(float(trials["value"].iloc[0]), color="grey", linestyle="--", label="default (trial 0)")
    ax.set_xlabel("trial")
    ax.set_ylabel("mean WAPE on the tuning windows (%)")
    ax.set_title("Hyperparameter search")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _tuning_windows(wide: WideData, cfg: ForecastConfig, first_valid_start: pd.Timestamp):
    """Windows used to score trials; they must finish before the first validation window starts."""
    ends = [pd.Timestamp(e) for e in cfg.tune.tuning_window_ends]
    if max(ends) >= first_valid_start:
        raise ValueError(f"Tuning windows must end before the first validation window starts "
                         f"({first_valid_start.date()}); got {max(ends).date()}")
    tuning_cfg = replace(cfg, validation_window_ends=tuple(cfg.tune.tuning_window_ends))
    return make_validation_windows(wide.dates[0], wide.dates[-1], tuning_cfg)


def run_tuning(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None,
               max_series: Optional[int] = None) -> Dict[str, Any]:
    """Run Step 7 end to end and return the summary that is also written to JSON.

    Args:
        cfg: Project configuration.
        df: Prepared table from Step 2; read from parquet when omitted.
        max_series: Use only this many evenly spaced series (for quick test runs).
    """
    for folder in (cfg.paths.reports_dir, cfg.paths.figures_dir, cfg.paths.models_dir):
        folder.mkdir(parents=True, exist_ok=True)
    frame = load_prepared(cfg.paths.prepared_parquet) if df is None else df
    wide = to_wide(frame, cfg)
    if max_series:
        wide = select_series_by_volume(wide, max_series)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)
    tuning_windows = _tuning_windows(wide, cfg, windows[0].valid_start)

    tuned = tune_gbm(wide, tuning_windows, cfg, cfg.tune.n_trials, cfg.tune.search)
    trials = pd.DataFrame(tuned.trials)
    trials.round(5).to_csv(cfg.paths.tuning_trials, index=False)
    tuned_cfg = replace(cfg.gbm, **tuned.best_params)
    logger.info("Tuning (%s): default %.3f -> best %.3f WAPE on the tuning windows", tuned.search,
                tuned.default_value, tuned.best_value)

    baselines = build_baselines(replace(cfg.baselines, names=(cfg.benchmark, BEST_BASELINE)))
    gbm_default = GbmForecaster(cfg.gbm, cfg.horizon_days, cfg.random_state)
    gbm_tuned = GbmForecaster(tuned_cfg, cfg.horizon_days, cfg.random_state)
    stack = StackedModels(gbm_tuned, baselines[BEST_BASELINE], cfg.tune)
    recorders = {
        "gbm_default": RecordingForecaster(gbm_default),
        "gbm_tuned": RecordingForecaster(stack.variant("raw")),
        "gbm_tuned_calibrated": RecordingForecaster(stack.variant("calibrated")),
        "gbm_tuned_blend": RecordingForecaster(stack.variant("blend")),
        "gbm_tuned_calibrated_blend": RecordingForecaster(stack.variant("calibrated_blend")),
        BEST_BASELINE: RecordingForecaster(baselines[BEST_BASELINE]),
    }
    forecasters = {cfg.benchmark: baselines[cfg.benchmark], **recorders}

    scores = run_backtest(wide, windows, forecasters, cfg)
    floor = {w.valid_end.date().isoformat(): noise_floor(*_floor_inputs(wide, w))["estimated_floor_wape"]
             for w in windows}
    summary = summarize_backtest(scores, cfg, floor)
    means = {name: v["mean_wape"] for name, v in summary["baselines"].items()}
    selected = select_variant(means, cfg.tune.min_gain_points)

    actual, forecast, dates, horizon_days = stack_windows(wide, windows, recorders[selected])
    baseline_forecast = stack_windows(wide, windows, recorders[BEST_BASELINE])[1]
    analysis = analyze(actual, forecast, baseline_forecast, dates, horizon_days, wide.series)

    diagnostics = [{"window_end": str(wide.dates[d["history_days"] + cfg.horizon_days - 1].date()), **d}
                   for d in stack.diagnostics]
    summary.update({
        "tuning": {**tuned.as_dict(), "tuning_windows": [w.valid_end.date().isoformat() for w in tuning_windows]},
        "variants_mean_wape": {name: means[name] for name in COMPLEXITY_ORDER},
        "improvement_vs_default_points": {name: round(means["gbm_default"] - means[name], 3) for name in COMPLEXITY_ORDER},
        "selected_variant": selected,
        "min_gain_points": cfg.tune.min_gain_points,
        "stack_diagnostics": diagnostics,
        "error_analysis": analysis["summary"],
        "series_used": int(wide.n_series),
    })
    saved_gbm = cfg.gbm if selected == "gbm_default" else tuned_cfg  # the default model keeps its own settings
    selected_config = {
        "variant": selected, "gbm": {**vars(saved_gbm)}, "baseline": BEST_BASELINE,
        "calibrate": "calibrated" in selected, "blend": "blend" in selected,
        "tuning_search": tuned.search, "mean_wape": means[selected],
    }
    cfg.paths.gbm_params.write_text(json.dumps(selected_config, indent=2, default=str), encoding="utf-8")
    scores.round(4).to_csv(cfg.paths.step7_results, index=False)
    cfg.paths.step7_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figures, dpi = cfg.paths.figures_dir, cfg.eda.figure_dpi
    plot_trials(trials, figures / "tuning_trials.png", dpi)
    plot_wape_by_window(scores, floor, cfg.benchmark, figures / "step7_wape_by_window.png", dpi)
    plot_store_item_heatmap(analysis["series"], figures / "error_store_item_heatmap.png", dpi)
    plot_volume_scatter(analysis["series"], figures / "error_volume_scatter.png", dpi)
    plot_horizon_calendar(analysis["horizon"], analysis["calendar"], figures / "error_horizon_calendar.png", dpi)
    logger.info("Selected %s (mean WAPE %.3f); default was %.3f", selected, means[selected], means["gbm_default"])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Step 7: tuning, calibration, blending, error analysis.")
    parser.add_argument("--prepared-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--engine", choices=("lightgbm", "hist_gb"), default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--search", choices=("optuna", "random"), default=None)
    parser.add_argument("--max-series", type=int, default=None, help="Use a sample of series for a quick run")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = ForecastConfig()
    if args.engine:
        cfg = replace(cfg, gbm=replace(cfg.gbm, engine=args.engine))
    tune = cfg.tune
    if args.n_trials:
        tune = replace(tune, n_trials=args.n_trials)
    if args.search:
        tune = replace(tune, search=args.search)
    paths = cfg.paths
    if args.reports_dir:
        paths = replace(paths, reports_dir=args.reports_dir)
    configure_logging(args.log_level)
    cfg = replace(cfg, tune=tune, paths=paths)
    run_tuning(cfg, load_prepared(args.prepared_path or cfg.paths.prepared_parquet), max_series=args.max_series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
