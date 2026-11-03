"""Tests for forecast.wide."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.config import ForecastConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.splits import make_validation_windows
from forecast.wide import to_wide, window_arrays


@pytest.fixture
def prepared(raw_frames: RawFrames, cfg: ForecastConfig) -> pd.DataFrame:
    return build_prepared_frame(raw_frames, cfg)[0]


def test_to_wide_shape_order_and_labels(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    wide = to_wide(prepared, cfg)
    assert wide.values.shape == (6, 1000) and wide.n_series == 6 and wide.n_days == 1000
    assert wide.dates[0] == pd.Timestamp("2013-01-01") and wide.dates[-1] == pd.Timestamp("2015-09-27")
    assert wide.series.is_monotonic_increasing  # rows sorted by (store, item)
    row = wide.series.get_loc((1, 2))
    expected = prepared[(prepared["store"] == 1) & (prepared["item"] == 2) & ~prepared["is_test"]]["sales"].to_numpy()
    assert np.array_equal(wide.values[row], expected)  # the row holds that series in date order


def test_to_wide_rejects_missing_history(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    broken = prepared.copy()
    broken.loc[broken.index[10], "sales"] = np.nan
    with pytest.raises(ValueError, match="missing sales"):
        to_wide(broken, cfg)


def test_window_arrays_alignment_and_no_leakage(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    wide = to_wide(prepared, cfg)
    window = make_validation_windows(wide.dates[0], wide.dates[-1], cfg)[0]
    arrays = window_arrays(wide, window)
    assert arrays.history_dates[-1] == window.train_end
    assert arrays.future_dates[0] == window.valid_start and arrays.future_dates[-1] == window.valid_end
    assert arrays.history.shape[1] == len(arrays.history_dates)
    assert arrays.actual.shape == (6, cfg.horizon_days)
    assert arrays.history_dates[-1] < arrays.future_dates[0]  # history never reaches the validation days
    start = wide.dates.get_loc(window.valid_start)
    assert np.array_equal(arrays.actual, wide.values[:, start:start + cfg.horizon_days])


def test_window_outside_data_is_rejected(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    from forecast.splits import ValidationWindow

    wide = to_wide(prepared, cfg)
    outside = ValidationWindow(pd.Timestamp("2016-01-01"), pd.Timestamp("2016-01-02"), pd.Timestamp("2016-01-29"))
    with pytest.raises(ValueError, match="outside"):
        window_arrays(wide, outside)
