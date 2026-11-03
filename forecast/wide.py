"""Series x day matrices: the fast layout used by baselines and backtests.

One row per store-item series and one column per day lets every baseline be a few numpy
operations over all 500 series at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast.config import ForecastConfig
from forecast.splits import ValidationWindow


@dataclass(frozen=True)
class WideData:
    """Labelled sales as a [n_series, n_days] matrix with its row and column labels."""

    values: np.ndarray
    dates: pd.DatetimeIndex
    series: pd.Index

    @property
    def n_series(self) -> int:
        """Number of series (rows)."""
        return int(self.values.shape[0])

    @property
    def n_days(self) -> int:
        """Number of days (columns)."""
        return int(self.values.shape[1])


@dataclass(frozen=True)
class WindowArrays:
    """Everything a baseline needs for one validation window."""

    history: np.ndarray
    history_dates: pd.DatetimeIndex
    future_dates: pd.DatetimeIndex
    actual: np.ndarray


def to_wide(df: pd.DataFrame, cfg: ForecastConfig) -> WideData:
    """Pivot the labelled (non-test) rows of the prepared table into a series x day matrix.

    Raises:
        ValueError: If any labelled day is missing or the dates are not contiguous daily.
    """
    keys = list(cfg.granularity)
    labelled = df.loc[~df["is_test"], [cfg.date_col, *keys, cfg.target_col]]
    pivot = labelled.pivot(index=keys, columns=cfg.date_col, values=cfg.target_col).sort_index()
    pivot = pivot.sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Labelled history contains missing sales; fill or flag them first")
    dates = pd.DatetimeIndex(pivot.columns)
    if len(dates) > 1 and not (np.diff(dates.to_numpy()) == np.timedelta64(1, "D")).all():
        raise ValueError("Dates must be contiguous daily")
    return WideData(values=values, dates=dates, series=pivot.index)


def window_arrays(wide: WideData, window: ValidationWindow) -> WindowArrays:
    """Cut one validation window out of the matrix: history up to ``train_end`` and the actuals.

    Raises:
        ValueError: If the window dates are not inside the available data.
    """
    for stamp in (window.train_end, window.valid_start, window.valid_end):
        if stamp not in wide.dates:
            raise ValueError(f"Window date {stamp.date()} is outside the available data")
    train_stop = wide.dates.get_loc(window.train_end) + 1
    valid_start = wide.dates.get_loc(window.valid_start)
    valid_stop = wide.dates.get_loc(window.valid_end) + 1
    return WindowArrays(
        history=wide.values[:, :train_stop],
        history_dates=wide.dates[:train_stop],
        future_dates=wide.dates[valid_start:valid_stop],
        actual=wide.values[:, valid_start:valid_stop],
    )
