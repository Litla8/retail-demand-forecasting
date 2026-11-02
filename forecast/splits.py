"""Time-based validation windows. Random splits are never used on this data."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from forecast.config import ForecastConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationWindow:
    """One expanding-window fold: train up to ``train_end``, validate on the next days."""

    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    @property
    def horizon_days(self) -> int:
        """Number of days in the validation window."""
        return (self.valid_end - self.valid_start).days + 1

    def as_dict(self) -> Dict[str, str]:
        """ISO-formatted dates for JSON reports."""
        return {
            "train_end": self.train_end.date().isoformat(),
            "valid_start": self.valid_start.date().isoformat(),
            "valid_end": self.valid_end.date().isoformat(),
        }


def make_validation_windows(
    first_date: pd.Timestamp, last_train_date: pd.Timestamp, cfg: ForecastConfig
) -> List[ValidationWindow]:
    """Create expanding-window folds ending on each configured date.

    Raises:
        ValueError: On duplicate ends, a window ending after the last labelled day, or a
            window with less than ``min_history_days`` of training history.
    """
    ends = [pd.Timestamp(e) for e in cfg.validation_window_ends]
    if len(set(ends)) != len(ends):
        raise ValueError("validation_window_ends contains duplicates")
    windows: List[ValidationWindow] = []
    for end in sorted(ends):
        if end > last_train_date:
            raise ValueError(f"Validation window ends {end.date()} after the last labelled day "
                             f"{last_train_date.date()}")
        valid_start = end - pd.Timedelta(days=cfg.horizon_days - 1)
        train_end = valid_start - pd.Timedelta(days=1)
        history = (train_end - first_date).days + 1
        if history < cfg.min_history_days:
            raise ValueError(f"Window ending {end.date()} has only {history} days of history; "
                             f"{cfg.min_history_days} required")
        windows.append(ValidationWindow(train_end, valid_start, end))
    return windows


def split_window(
    df: pd.DataFrame, window: ValidationWindow, cfg: ForecastConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the prepared table into train and validation parts for one window.

    Test-horizon rows are never included. The split is verified: training strictly precedes
    validation, validation spans exactly ``horizon_days`` days, and has no missing actuals.

    Raises:
        ValueError: If any of those guarantees is violated.
    """
    date = df[cfg.date_col]
    labelled = ~df["is_test"]
    train = df[labelled & (date <= window.train_end)]
    valid = df[labelled & (date >= window.valid_start) & (date <= window.valid_end)]
    if train.empty or valid.empty:
        raise ValueError("Window produced an empty train or validation set")
    if train[cfg.date_col].max() >= valid[cfg.date_col].min():
        raise ValueError("Leakage: training data overlaps the validation window")
    if valid[cfg.date_col].nunique() != cfg.horizon_days:
        raise ValueError("Validation window does not contain horizon_days distinct days")
    if valid[cfg.target_col].isna().any():
        raise ValueError("Validation window contains missing actual sales")
    return train, valid


def final_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all labelled history, test horizon) for the final fit and forecast."""
    return df[~df["is_test"]], df[df["is_test"]]
