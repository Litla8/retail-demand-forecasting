"""Tests for forecast.prepare (the raw-file load is replaced with synthetic frames)."""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from forecast.config import ForecastConfig
from forecast.data import RawFrames
from forecast.features import CALENDAR_FEATURES
from forecast.prepare import build_prepared_frame, run_prepare


def test_build_prepared_frame_shape_columns_and_report(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    grid, report = build_prepared_frame(raw_frames, cfg)
    assert grid.shape[0] == 6 * 1028 and report["rows_prepared"] == grid.shape[0]
    assert report["series"] == 6 and report["rows_test"] == 6 * 28
    assert set(CALENDAR_FEATURES) <= set(grid.columns)
    assert report["train_memory_mb_downcast"] < report["train_memory_mb_raw"]
    assert report["missing_days_in_train"] == 0 and len(report["validation_windows"]) == 3
    json.dumps(report)  # the report must be JSON-serialisable


def test_build_is_deterministic(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    first, _ = build_prepared_frame(raw_frames, cfg)
    second, _ = build_prepared_frame(raw_frames, cfg)
    pd.testing.assert_frame_equal(first, second)


def test_build_fails_loudly_on_bad_test_horizon(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    shorter = RawFrames(raw_frames.train, raw_frames.test.iloc[:-6], raw_frames.submission.iloc[:-6])
    with pytest.raises(ValueError):
        build_prepared_frame(shorter, cfg)


def test_run_prepare_writes_parquet_and_report(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    pytest.importorskip("pyarrow")
    with patch("forecast.prepare.load_raw", return_value=raw_frames) as mocked:
        first = run_prepare(cfg)
        second = run_prepare(cfg)  # re-run into the same folders: must be idempotent
    assert mocked.call_count == 2
    saved = pd.read_parquet(cfg.paths.prepared_parquet)
    assert len(saved) == first["rows_prepared"] == second["rows_prepared"]
    assert json.loads(cfg.paths.quality_report.read_text())["series"] == 6
