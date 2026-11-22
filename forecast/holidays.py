"""A transparent holiday calendar (US dates) used to test whether holidays affect demand.

The dataset does not say which country it comes from, so this calendar is a hypothesis, not a
fact. Step 3 measures each holiday's effect; features are only added for holidays that matter.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from dateutil.easter import easter

# Monday = 0 ... Sunday = 6, matching pandas.
MONDAY, THURSDAY, SUNDAY = 0, 3, 6


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """The n-th given weekday of a month (n = 1 for the first)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.day_of_week) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    """The last given weekday of a month."""
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.day_of_week - weekday) % 7)


def holiday_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return one row per holiday occurrence between ``start`` and ``end`` (inclusive).

    Raises:
        ValueError: If ``end`` is before ``start``.
    """
    if end < start:
        raise ValueError("end must not be before start")
    rows: List[Tuple[str, pd.Timestamp]] = []
    for year in range(start.year, end.year + 1):
        thanksgiving = _nth_weekday(year, 11, THURSDAY, 4)
        rows += [
            ("New Year's Day", pd.Timestamp(year=year, month=1, day=1)),
            ("Martin Luther King Day", _nth_weekday(year, 1, MONDAY, 3)),
            ("Valentine's Day", pd.Timestamp(year=year, month=2, day=14)),
            ("Presidents' Day", _nth_weekday(year, 2, MONDAY, 3)),
            ("Easter Sunday", pd.Timestamp(easter(year))),
            ("Mother's Day", _nth_weekday(year, 5, SUNDAY, 2)),
            ("Memorial Day", _last_weekday(year, 5, MONDAY)),
            ("Father's Day", _nth_weekday(year, 6, SUNDAY, 3)),
            ("Independence Day", pd.Timestamp(year=year, month=7, day=4)),
            ("Labor Day", _nth_weekday(year, 9, MONDAY, 1)),
            ("Columbus Day", _nth_weekday(year, 10, MONDAY, 2)),
            ("Halloween", pd.Timestamp(year=year, month=10, day=31)),
            ("Veterans Day", pd.Timestamp(year=year, month=11, day=11)),
            ("Thanksgiving", thanksgiving),
            ("Black Friday", thanksgiving + pd.Timedelta(days=1)),
            ("Christmas Eve", pd.Timestamp(year=year, month=12, day=24)),
            ("Christmas Day", pd.Timestamp(year=year, month=12, day=25)),
            ("New Year's Eve", pd.Timestamp(year=year, month=12, day=31)),
        ]
    calendar = pd.DataFrame(rows, columns=["holiday", "date"])
    calendar = calendar[(calendar["date"] >= start) & (calendar["date"] <= end)]
    return calendar.sort_values("date").reset_index(drop=True)
