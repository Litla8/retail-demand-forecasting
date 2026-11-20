"""Tests for forecast.horizon."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from forecast.config import ForecastConfig
from forecast.data import RawFrames, clean_test, clean_train, validate_submission, validate_test
from forecast.horizon import horizon_frame, make_horizon_files, matches


@pytest.fixture
def raw_cfg(cfg: ForecastConfig, raw_frames: RawFrames, tmp_path: Path) -> ForecastConfig:
    paths = replace(cfg.paths, train_csv=tmp_path / "train.csv", test_csv=tmp_path / "test.csv",
                    sample_submission_csv=tmp_path / "sample_submission.csv")
    raw_frames.train.to_csv(paths.train_csv, index=False)
    return replace(cfg, paths=paths)


def test_horizon_frame_layout(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    frame = horizon_frame(raw_frames.train, cfg)
    assert len(frame) == 6 * cfg.horizon_days and list(frame.columns) == ["id", "date", "store", "item"]
    assert frame["date"].min() == pd.Timestamp("2015-09-28") and frame["id"].tolist() == list(range(len(frame)))
    assert frame["date"].is_monotonic_increasing  # date-major, like the Kaggle file
    with pytest.raises(ValueError, match="needs the columns"):
        horizon_frame(raw_frames.train.drop(columns=["item"]), cfg)


def test_generated_files_pass_the_pipeline_checks(raw_cfg: ForecastConfig) -> None:
    assert make_horizon_files(raw_cfg) is True
    train = clean_train(pd.read_csv(raw_cfg.paths.train_csv), raw_cfg)[0]
    test = clean_test(pd.read_csv(raw_cfg.paths.test_csv), raw_cfg)
    validate_test(test, train, raw_cfg)
    validate_submission(pd.read_csv(raw_cfg.paths.sample_submission_csv), test, raw_cfg)


def test_existing_consistent_files_are_left_alone_and_new_data_triggers_a_rewrite(raw_cfg: ForecastConfig) -> None:
    make_horizon_files(raw_cfg)
    before = raw_cfg.paths.test_csv.read_text()
    assert make_horizon_files(raw_cfg) is False and raw_cfg.paths.test_csv.read_text() == before  # idempotent
    train = pd.read_csv(raw_cfg.paths.train_csv)
    extra = train[train["date"] == train["date"].max()].assign(date="2015-09-28")
    pd.concat([train, extra]).to_csv(raw_cfg.paths.train_csv, index=False)  # one more day of history arrives
    assert make_horizon_files(raw_cfg) is True
    assert pd.read_csv(raw_cfg.paths.test_csv)["date"].min() == "2015-09-29"
    wanted = horizon_frame(pd.read_csv(raw_cfg.paths.train_csv), raw_cfg)
    assert matches(pd.read_csv(raw_cfg.paths.test_csv), wanted, raw_cfg)


def test_missing_train_file(cfg: ForecastConfig, tmp_path: Path) -> None:
    missing = replace(cfg, paths=replace(cfg.paths, train_csv=tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        make_horizon_files(missing)
