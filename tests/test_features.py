"""Tests for forecast.features."""
from __future__ import annotations

import pandas as pd
import pytest

from forecast.features import CALENDAR_FEATURES, add_calendar_features


def test_known_calendar_values() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2017-12-31", "2018-01-01"])})
    out = add_calendar_features(df).set_index(df["date"].dt.strftime("%Y-%m-%d"))
    sunday, monday = out.loc["2017-12-31"], out.loc["2018-01-01"]
    assert (sunday["dayofweek"], sunday["is_weekend"], sunday["month"], sunday["dayofyear"]) == (6, 1, 12, 365)
    assert sunday["is_month_end"] == 1 and sunday["is_month_start"] == 0
    assert (monday["dayofweek"], monday["is_weekend"], monday["year"], monday["is_month_start"]) == (0, 0, 2018, 1)
    assert sunday["t"] == 0 and monday["t"] == 1  # days since the earliest date


def test_adds_all_features_with_small_dtypes_and_keeps_input() -> None:
    df = pd.DataFrame({"date": pd.date_range("2016-02-27", periods=5), "sales": [1, 2, 3, 4, 5]})
    out = add_calendar_features(df)
    assert set(CALENDAR_FEATURES) <= set(out.columns)
    assert str(out["dayofweek"].dtype) == "int8" and str(out["year"].dtype) == "int16"
    assert "sales" in out.columns and "dayofweek" not in df.columns  # input is not mutated
    assert out.loc[2, "day"] == 29 and out.loc[2, "month"] == 2  # 2016 is a leap year


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="datetime"):
        add_calendar_features(pd.DataFrame({"date": ["2017-01-01"]}))
    with pytest.raises(ValueError, match="Missing date column"):
        add_calendar_features(pd.DataFrame({"when": pd.to_datetime(["2017-01-01"])}))
