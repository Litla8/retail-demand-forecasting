"""Tests for forecast.baselines (hand-built series with known answers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import baselines as B
from forecast.config import BaselineConfig

DAYS = pd.date_range("2014-01-01", "2016-12-31", freq="D")  # 1,096 days, ends on a Saturday
HISTORY_DATES = DAYS
FUTURE = pd.date_range("2017-01-01", periods=10, freq="D")
MONTH_LEVEL = np.arange(1, 13) * 10.0
WEEKDAY_FACTOR = 1 + np.arange(7) / 10


def _ramp() -> np.ndarray:
    """Two series whose value equals the day index (second series is ten times larger)."""
    t = np.arange(len(DAYS), dtype=float)
    return np.vstack([t, 10 * t])


def test_naive_and_moving_average() -> None:
    history = _ramp()
    assert np.array_equal(B.naive(history, DAYS, FUTURE), np.tile(history[:, -1:], (1, 10)))
    flat = B.moving_average(history, DAYS, FUTURE, window=7)
    assert flat.shape == (2, 10) and np.allclose(flat[0], history[0, -7:].mean())


def test_seasonal_naive_7_matches_weekdays() -> None:
    weekday_value = np.tile(DAYS.day_of_week.to_numpy(dtype=float) * 100, (2, 1))  # value depends on weekday only
    forecast = B.seasonal_naive_7(weekday_value, DAYS, FUTURE)
    assert np.array_equal(forecast[0], FUTURE.day_of_week.to_numpy() * 100.0)  # each day copies its weekday
    ramp = B.seasonal_naive_7(_ramp(), DAYS, FUTURE)
    assert ramp[0, 0] == len(DAYS) - 7 and ramp[0, 6] == len(DAYS) - 1  # step 1 copies 7 days back, step 7 the last day


def test_seasonal_moving_average_averages_same_weekdays() -> None:
    history = _ramp()
    forecast = B.seasonal_moving_average(history, DAYS, FUTURE, weeks=2)
    one_week = B.seasonal_naive_7(history, DAYS, FUTURE)
    offsets = np.array([[3.5], [35.0]])  # mean of (x, x - 7) is x - 3.5 for the unit ramp, x - 35 for the 10x ramp
    assert np.allclose(forecast, one_week - offsets)


def test_seasonal_naive_364_copies_the_same_weekday_last_year() -> None:
    history = _ramp()
    forecast = B.seasonal_naive_364(history, DAYS, FUTURE)
    assert np.array_equal(forecast[0], np.arange(len(DAYS) - 364, len(DAYS) - 364 + 10, dtype=float))
    source = DAYS[len(DAYS) - 364: len(DAYS) - 364 + 10]
    assert (source.day_of_week.to_numpy() == FUTURE.day_of_week.to_numpy()).all()  # same weekday
    with pytest.raises(ValueError, match="too short"):
        B.seasonal_naive_364(history[:, :300], DAYS[:300], pd.date_range(DAYS[299] + pd.Timedelta(days=1), periods=5))


def test_growth_ratio_recovers_a_known_growth_rate() -> None:
    flat = np.ones((1, 1000))
    flat[:, 700:] *= 1.1  # the last 91 days (909-999) are 10% above the same days a year earlier (545-635)
    assert B.growth_ratio(flat, window=91)[0] == pytest.approx(1.1)
    grown = B.seasonal_naive_364_growth(_ramp(), DAYS, FUTURE, window=91)
    assert np.allclose(grown, B.seasonal_naive_364(_ramp(), DAYS, FUTURE) * B.growth_ratio(_ramp(), 91)[:, None])
    with pytest.raises(ValueError, match="at least"):
        B.growth_ratio(np.ones((1, 400)), window=91)


def test_month_weekday_profile_returns_last_years_cell_mean() -> None:
    month_level = np.arange(1, 13) * 10.0
    weekday_factor = 1 + DAYS.day_of_week.to_numpy() / 10
    history = np.tile(month_level[DAYS.month.to_numpy() - 1] * weekday_factor, (2, 1))
    forecast = B.month_weekday_profile(history, DAYS, FUTURE)
    expected = month_level[FUTURE.month.to_numpy() - 1] * (1 + FUTURE.day_of_week.to_numpy() / 10)
    assert np.allclose(forecast[0], expected)  # last year's level and weekday factor, noise-free
    with_growth = B.month_weekday_profile_growth(history, DAYS, FUTURE, window=91)
    assert np.allclose(with_growth, forecast * B.growth_ratio(history, 91)[:, None])  # profile x growth factor


def test_multi_year_profile_follows_a_steadily_growing_staircase() -> None:
    year_growth = 1.1 ** (DAYS.year.to_numpy() - 2014)  # every year is 10% above the one before
    pattern = MONTH_LEVEL[DAYS.month.to_numpy() - 1] * WEEKDAY_FACTOR[DAYS.day_of_week.to_numpy()]
    history = np.tile(pattern * year_growth, (2, 1))
    truth = (MONTH_LEVEL[FUTURE.month.to_numpy() - 1] * WEEKDAY_FACTOR[FUTURE.day_of_week.to_numpy()]) * 1.1 ** 3
    forecast = B.month_weekday_profile_multi_year(history, DAYS, FUTURE, years=2, window=91)
    assert np.allclose(forecast[0], truth, rtol=0.05)  # both past years point to the same 2017 level
    with pytest.raises(ValueError, match="year\\(s\\) back"):
        B.month_weekday_profile_multi_year(history, DAYS, FUTURE, years=5, window=91)


def test_profile_needs_a_matching_cell_last_year() -> None:
    short = pd.date_range("2016-06-01", "2016-12-31", freq="D")
    with pytest.raises(ValueError, match="no matching"):
        B.month_weekday_profile(np.ones((1, len(short))), short, FUTURE)


def test_input_checks() -> None:
    history = _ramp()
    with pytest.raises(ValueError, match="day after"):
        B.naive(history, DAYS, FUTURE + pd.Timedelta(days=3))
    with pytest.raises(ValueError, match="match history_dates"):
        B.naive(history[:, :-1], DAYS, FUTURE)
    with pytest.raises(ValueError, match="finite"):
        B.naive(np.where(history > 5, np.nan, history), DAYS, FUTURE)
    with pytest.raises(ValueError, match="longer than"):
        B.moving_average(history[:, :5], DAYS[:5], pd.date_range("2014-01-06", periods=3), window=28)


def test_build_baselines_respects_the_config() -> None:
    built = B.build_baselines(BaselineConfig())
    assert list(built) == list(BaselineConfig().names)
    only = B.build_baselines(BaselineConfig(names=("naive", "seasonal_naive_364")))
    assert list(only) == ["naive", "seasonal_naive_364"]
    with pytest.raises(ValueError, match="subset"):
        BaselineConfig(names=("naive", "magic"))
    with pytest.raises(ValueError, match="positive"):
        BaselineConfig(growth_window=0)
