"""Tests for forecast.holidays."""
from __future__ import annotations

import pandas as pd
import pytest

from forecast.holidays import holiday_calendar


def _date(calendar: pd.DataFrame, name: str, year: int) -> str:
    row = calendar[(calendar["holiday"] == name) & (calendar["date"].dt.year == year)]
    return row["date"].iloc[0].strftime("%Y-%m-%d")


def test_known_holiday_dates() -> None:
    calendar = holiday_calendar(pd.Timestamp("2013-01-01"), pd.Timestamp("2017-12-31"))
    assert _date(calendar, "Thanksgiving", 2014) == "2014-11-27"
    assert _date(calendar, "Black Friday", 2014) == "2014-11-28"
    assert _date(calendar, "Memorial Day", 2015) == "2015-05-25"
    assert _date(calendar, "Labor Day", 2016) == "2016-09-05"
    assert _date(calendar, "Easter Sunday", 2017) == "2017-04-16"
    assert _date(calendar, "Mother's Day", 2013) == "2013-05-12"
    assert _date(calendar, "Christmas Day", 2013) == "2013-12-25"


def test_calendar_shape_sorting_and_window() -> None:
    calendar = holiday_calendar(pd.Timestamp("2013-01-01"), pd.Timestamp("2017-12-31"))
    assert len(calendar) == 5 * 18 and calendar["date"].is_monotonic_increasing
    narrow = holiday_calendar(pd.Timestamp("2014-12-01"), pd.Timestamp("2014-12-31"))
    assert set(narrow["holiday"]) == {"Christmas Eve", "Christmas Day", "New Year's Eve"}


def test_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="before start"):
        holiday_calendar(pd.Timestamp("2015-01-01"), pd.Timestamp("2014-01-01"))
