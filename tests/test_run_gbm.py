"""Tests for forecast.run_gbm (small synthetic data and a small boosting model)."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from forecast.config import ForecastConfig, GbmConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.run_gbm import run_gbm


@pytest.fixture
def gbm_cfg(cfg: ForecastConfig) -> ForecastConfig:
    return replace(cfg, gbm=replace(GbmConfig(engine="hist_gb"), n_estimators=30, min_child_samples=20))


def test_run_gbm_writes_outputs_and_compares_with_every_baseline(raw_frames: RawFrames, gbm_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, gbm_cfg)[0]
    summary = run_gbm(gbm_cfg, prepared)
    assert summary["model"]["name"] == "gbm" and summary["model"]["engine"] == "hist_gb"
    assert summary["best_baseline"] in summary["baselines"] and "gbm" in summary["baselines"]
    assert summary["model"]["feature_importance_share_pct"] is None  # only LightGBM reports gain importance
    results = pd.read_csv(gbm_cfg.paths.gbm_results)
    assert len(results) == (len(gbm_cfg.baselines.names) + 1) * 3 * 4
    assert json.loads(gbm_cfg.paths.gbm_summary.read_text())["model"]["mean_wape"] == summary["model"]["mean_wape"]
    figures = {p.name for p in gbm_cfg.paths.figures_dir.glob("gbm_*.png")}
    assert figures == {"gbm_wape_by_window.png", "gbm_wape_by_horizon.png", "gbm_total_forecast.png"}


def test_run_gbm_is_idempotent(raw_frames: RawFrames, gbm_cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, gbm_cfg)[0]
    first = run_gbm(gbm_cfg, prepared)
    second = run_gbm(gbm_cfg, prepared)
    assert first["model"]["mean_wape"] == second["model"]["mean_wape"]
    assert first["baselines"]["gbm"]["wape_by_window"] == second["baselines"]["gbm"]["wape_by_window"]
