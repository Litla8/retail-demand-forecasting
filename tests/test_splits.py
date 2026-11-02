"""Tests for forecast.splits."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from forecast.config import ForecastConfig
from forecast.data import RawFrames, build_full_grid, clean_test, clean_train
from forecast.splits import ValidationWindow, final_split, make_validation_windows, split_window

FIRST, LAST = pd.Timestamp("2013-01-01"), pd.Timestamp("2015-09-27")


@pytest.fixture
def grid(raw_frames: RawFrames, cfg: ForecastConfig) -> pd.DataFrame:
    train, _ = clean_train(raw_frames.train, cfg)
    return build_full_grid(train, clean_test(raw_frames.test, cfg), cfg)


def test_window_dates_are_exact(cfg: ForecastConfig) -> None:
    windows = make_validation_windows(FIRST, LAST, cfg)
    assert len(windows) == 3
    last = windows[-1]
    assert last.valid_end == pd.Timestamp("2015-09-27") and last.horizon_days == 28
    assert last.valid_start == pd.Timestamp("2015-08-31")
    assert last.train_end == pd.Timestamp("2015-08-30")  # the day before validation starts
    assert all(a.valid_end < b.valid_end for a, b in zip(windows, windows[1:]))


def test_window_validation_errors(cfg: ForecastConfig) -> None:
    with pytest.raises(ValueError, match="after the last labelled day"):
        make_validation_windows(FIRST, pd.Timestamp("2015-08-01"), cfg)
    with pytest.raises(ValueError, match="duplicates"):
        make_validation_windows(FIRST, LAST, replace(cfg, validation_window_ends=("2015-03-31", "2015-03-31")))
    with pytest.raises(ValueError, match="days of history"):
        make_validation_windows(FIRST, LAST, replace(cfg, validation_window_ends=("2013-06-30",)))


def test_split_has_no_leakage_and_exact_horizon(grid: pd.DataFrame, cfg: ForecastConfig) -> None:
    for window in make_validation_windows(FIRST, LAST, cfg):
        train, valid = split_window(grid, window, cfg)
        assert train["date"].max() < valid["date"].min()
        assert valid["date"].nunique() == cfg.horizon_days
        assert not train["is_test"].any() and not valid["is_test"].any()
        assert len(valid) == 6 * cfg.horizon_days  # every series, every validation day


def test_split_is_repeatable(grid: pd.DataFrame, cfg: ForecastConfig) -> None:
    window = make_validation_windows(FIRST, LAST, cfg)[0]
    a_train, a_valid = split_window(grid, window, cfg)
    b_train, b_valid = split_window(grid, window, cfg)
    pd.testing.assert_frame_equal(a_valid, b_valid)
    assert len(a_train) == len(b_train)


def test_split_rejects_missing_actuals(grid: pd.DataFrame, cfg: ForecastConfig) -> None:
    window = make_validation_windows(FIRST, LAST, cfg)[-1]
    broken = grid.copy()
    broken.loc[broken["date"] == window.valid_end, "sales"] = float("nan")
    with pytest.raises(ValueError, match="missing actual"):
        split_window(broken, window, cfg)
    with pytest.raises(ValueError, match="empty"):
        split_window(grid, ValidationWindow(pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-02"),
                                            pd.Timestamp("2012-01-29")), cfg)


def test_final_split_separates_test_horizon(grid: pd.DataFrame) -> None:
    history, test = final_split(grid)
    assert len(history) + len(test) == len(grid) and test["is_test"].all()
    assert history["date"].max() < test["date"].min()
