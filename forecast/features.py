"""Feature engineering for the demand-forecasting models (calendar features first)."""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

CALENDAR_FEATURES = (
    "dayofweek", "day", "month", "year", "quarter", "dayofyear",
    "weekofyear", "is_weekend", "is_month_start", "is_month_end", "t",
)


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a copy of ``df`` with compact calendar features derived from the date.

    The dataset has no holiday, price or promotion columns, so the calendar is the only
    external signal available. All features are vectorised and stored in small dtypes.
    ``t`` counts days since the earliest date and carries the long-run trend.

    Raises:
        ValueError: If ``date_col`` is missing or not a datetime column.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing date column {date_col!r}")
    if not is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"{date_col!r} must be a datetime column, got {df[date_col].dtype}")

    out = df.copy()
    dates = out[date_col]
    out["dayofweek"] = dates.dt.day_of_week.astype("int8")  # Monday = 0
    out["day"] = dates.dt.day.astype("int8")
    out["month"] = dates.dt.month.astype("int8")
    out["year"] = dates.dt.year.astype("int16")
    out["quarter"] = dates.dt.quarter.astype("int8")
    out["dayofyear"] = dates.dt.day_of_year.astype("int16")
    out["weekofyear"] = dates.dt.isocalendar().week.astype("int8")
    out["is_weekend"] = (dates.dt.day_of_week >= 5).astype("int8")
    out["is_month_start"] = dates.dt.is_month_start.astype("int8")
    out["is_month_end"] = dates.dt.is_month_end.astype("int8")
    out["t"] = (dates - dates.min()).dt.days.astype("int16")
    return out
