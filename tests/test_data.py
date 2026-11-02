"""Tests for forecast.data."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.config import ForecastConfig
from forecast.data import (
    RawFrames,
    build_full_grid,
    clean_test,
    clean_train,
    load_raw,
    validate_submission,
    validate_test,
    validate_train,
)


def test_load_raw_reads_three_files(raw_frames: RawFrames, cfg: ForecastConfig, tmp_path: Path) -> None:
    paths = replace(cfg.paths, train_csv=tmp_path / "train.csv", test_csv=tmp_path / "test.csv",
                    sample_submission_csv=tmp_path / "sample_submission.csv")
    raw_frames.train.to_csv(paths.train_csv, index=False)
    raw_frames.test.to_csv(paths.test_csv, index=False)
    raw_frames.submission.to_csv(paths.sample_submission_csv, index=False)
    loaded = load_raw(paths)
    assert loaded.train.shape == raw_frames.train.shape and loaded.test.shape == raw_frames.test.shape


def test_load_raw_missing_file_fails_loudly(cfg: ForecastConfig, tmp_path: Path) -> None:
    paths = replace(cfg.paths, train_csv=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        load_raw(paths)


def test_validate_train_accepts_clean_and_rejects_bad(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    validate_train(raw_frames.train, cfg)  # must not raise
    with pytest.raises(ValueError, match="missing required columns"):
        validate_train(raw_frames.train.drop(columns=["item"]), cfg)
    with_nan = raw_frames.train.copy()
    with_nan.loc[0, "sales"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_train(with_nan, cfg)
    duplicated = pd.concat([raw_frames.train, raw_frames.train.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_train(duplicated, cfg)


def test_clean_train_downcasts_sorts_and_keeps_zeros(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    train = raw_frames.train.copy()
    train.loc[3, "sales"] = 0
    shuffled = train.sample(frac=1.0, random_state=42)
    cleaned, stats = clean_train(shuffled, cfg)
    assert str(cleaned["store"].dtype) == "int8" and str(cleaned["item"].dtype) == "int8"
    assert str(cleaned["sales"].dtype) in {"int8", "int16"}
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert stats["zero_sales_rows"] >= 1 and stats["memory_mb_after"] < stats["memory_mb_before"]
    ordered = cleaned.sort_values(["store", "item", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, ordered)  # already sorted by series then date


def test_negative_sales_policy(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    train = raw_frames.train.copy()
    train.loc[5, "sales"] = -4
    with pytest.raises(ValueError, match="negative"):
        clean_train(train, cfg)
    clipped, stats = clean_train(train, replace(cfg, negative_sales="clip"))
    assert clipped["sales"].min() == 0 and stats["negative_sales_rows"] == 1


def test_validate_test_rules(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    train, _ = clean_train(raw_frames.train, cfg)
    test = clean_test(raw_frames.test, cfg)
    validate_test(test, train, cfg)  # must not raise
    with pytest.raises(ValueError, match="horizon_days"):
        validate_test(test[test["date"] < test["date"].max()], train, cfg)
    gap = test.assign(date=test["date"] + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="day after"):
        validate_test(gap, train, cfg)
    missing_series = test[~((test["store"] == 1) & (test["item"] == 1))]
    with pytest.raises(ValueError, match="different store-item series"):
        validate_test(missing_series, train, cfg)
    non_contiguous = test[test["date"] != test["date"].min() + pd.Timedelta(days=3)]
    with pytest.raises(ValueError, match="contiguous"):
        validate_test(non_contiguous, train, cfg)


def test_validate_submission(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    test = clean_test(raw_frames.test, cfg)
    validate_submission(raw_frames.submission, test, cfg)
    with pytest.raises(ValueError, match="do not match"):
        validate_submission(raw_frames.submission.iloc[:-1], test, cfg)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_submission(raw_frames.submission.drop(columns=["sales"]), test, cfg)


def test_full_grid_is_complete_sorted_and_flags_test(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    train, _ = clean_train(raw_frames.train, cfg)
    test = clean_test(raw_frames.test, cfg)
    grid = build_full_grid(train, test, cfg)
    assert grid.shape[0] == 6 * (1000 + 28)
    assert grid.groupby(["store", "item"])["date"].is_monotonic_increasing.all()
    assert int(grid["is_test"].sum()) == 6 * 28
    assert grid.loc[grid["is_test"], "sales"].isna().all() and grid.loc[grid["is_test"], "id"].notna().all()
    assert grid.loc[~grid["is_test"], "id"].isna().all()
    assert not grid["is_missing_day"].any()


def test_full_grid_flags_missing_days_instead_of_filling(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    train, _ = clean_train(raw_frames.train, cfg)
    test = clean_test(raw_frames.test, cfg)
    dropped = train.drop(index=[10, 11, 5000]).reset_index(drop=True)
    grid = build_full_grid(dropped, test, cfg)
    assert grid.shape[0] == 6 * (1000 + 28)  # the grid is restored to complete
    assert int(grid["is_missing_day"].sum()) == 3
    assert grid.loc[grid["is_missing_day"], "sales"].isna().all()  # NaN, not a silent zero
