"""Tests for forecast.run_baselines (synthetic prepared table instead of the Kaggle files)."""
from __future__ import annotations

import json

import pandas as pd

from forecast.config import ForecastConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.run_baselines import run_baselines


def test_run_baselines_writes_results_summary_and_figures(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    summary = run_baselines(cfg, prepared)
    results = pd.read_csv(cfg.paths.baseline_results)
    assert len(results) == len(cfg.baselines.names) * 3 * 4
    assert set(results["scope"]) == {"all", "1-7", "8-14", "15-28"}
    saved = json.loads(cfg.paths.baseline_summary.read_text())
    assert saved["best_baseline"] == summary["best_baseline"] and saved["benchmark"] == cfg.benchmark
    figures = {p.name for p in cfg.paths.figures_dir.glob("baseline_*.png")}
    assert figures == {"baseline_wape_by_window.png", "baseline_wape_by_horizon.png", "baseline_total_forecast.png"}


def test_run_baselines_is_idempotent(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    first = run_baselines(cfg, prepared)
    second = run_baselines(cfg, prepared)  # same folders, no cleanup in between
    assert first == second
