"""Baseline forecasters. Every advanced model has to beat these.

Each baseline maps ``(history, history_dates, future_dates)`` to a [n_series, horizon] matrix and
only ever looks at ``history``, so nothing from the validation period can leak in. All of them
are vectorised over the 500 series.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict

import numpy as np
import pandas as pd

from forecast.config import BaselineConfig

YEAR_LAG = 364  # 52 whole weeks: the same weekday one year earlier
Forecaster = Callable[[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex], np.ndarray]


def check_inputs(history: np.ndarray, history_dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex) -> None:
    """Validate shapes, finiteness and that the future starts the day after the history ends.

    Raises:
        ValueError: On any violation.
    """
    if history.ndim != 2 or history.shape[1] != len(history_dates):
        raise ValueError("history must be [n_series, n_days] and match history_dates")
    if not np.isfinite(history).all():
        raise ValueError("history must be finite")
    if len(future_dates) == 0:
        raise ValueError("future_dates must not be empty")
    if future_dates[0] != history_dates[-1] + pd.Timedelta(days=1):
        raise ValueError("future_dates must start the day after history ends")
    if len(future_dates) > 1 and not (np.diff(future_dates.to_numpy()) == np.timedelta64(1, "D")).all():
        raise ValueError("future_dates must be contiguous daily")


def _same_weekday_positions(n_hist: int, horizon: int, weeks_back: int = 0) -> np.ndarray:
    """Column indices of the most recent same weekday for each forecast step.

    Step h (1-based) falls on absolute day ``n_hist - 1 + h``; subtracting ceil(h / 7) weeks lands
    on the latest day of the same weekday that is already in the history.
    """
    steps = np.arange(1, horizon + 1)
    return n_hist - 1 + steps - 7 * np.ceil(steps / 7).astype(int) - 7 * weeks_back


def naive(history: np.ndarray, history_dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Repeat the last observed day for the whole horizon."""
    check_inputs(history, history_dates, future_dates)
    return np.repeat(history[:, -1:], len(future_dates), axis=1)


def moving_average(history: np.ndarray, history_dates: pd.DatetimeIndex,
                   future_dates: pd.DatetimeIndex, window: int) -> np.ndarray:
    """Repeat the mean of the last ``window`` days (a flat forecast, ignoring the weekday)."""
    check_inputs(history, history_dates, future_dates)
    if window > history.shape[1]:
        raise ValueError("moving-average window is longer than the history")
    return np.repeat(history[:, -window:].mean(axis=1, keepdims=True), len(future_dates), axis=1)


def seasonal_naive_7(history: np.ndarray, history_dates: pd.DatetimeIndex,
                     future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Repeat the last observed week: each future day copies the latest same weekday."""
    check_inputs(history, history_dates, future_dates)
    if history.shape[1] < 7:
        raise ValueError("need at least 7 days of history")
    return history[:, _same_weekday_positions(history.shape[1], len(future_dates))]


def seasonal_moving_average(history: np.ndarray, history_dates: pd.DatetimeIndex,
                            future_dates: pd.DatetimeIndex, weeks: int) -> np.ndarray:
    """Average the latest ``weeks`` same-weekday values; this smooths the one-day noise."""
    check_inputs(history, history_dates, future_dates)
    if history.shape[1] < 7 * weeks:
        raise ValueError("history is shorter than the requested number of weeks")
    stacked = [history[:, _same_weekday_positions(history.shape[1], len(future_dates), w)] for w in range(weeks)]
    return np.mean(stacked, axis=0)


def _lag_364_positions(n_hist: int, horizon: int) -> np.ndarray:
    """Column indices of the same weekday one year (364 days) before each forecast step."""
    positions = n_hist - 1 + np.arange(1, horizon + 1) - YEAR_LAG
    if horizon > YEAR_LAG or positions.min() < 0:
        raise ValueError("history is too short for a 364-day lag at this horizon")
    return positions


def seasonal_naive_364(history: np.ndarray, history_dates: pd.DatetimeIndex,
                       future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Same weekday one year earlier. This is the project's benchmark."""
    check_inputs(history, history_dates, future_dates)
    return history[:, _lag_364_positions(history.shape[1], len(future_dates))]


def growth_ratio(history: np.ndarray, window: int, years_back: int = 1) -> np.ndarray:
    """Per-series growth factor: the last ``window`` days / the same days ``years_back`` x 364 days earlier.

    Raises:
        ValueError: If the history is too short for the requested comparison.
    """
    n = history.shape[1]
    lag = YEAR_LAG * years_back
    if n < window + lag:
        raise ValueError(f"need at least {window + lag} days of history for a {window}-day growth ratio "
                         f"{years_back} year(s) back")
    recent = history[:, n - window:].sum(axis=1)
    earlier = history[:, n - window - lag: n - lag].sum(axis=1)
    return np.divide(recent, earlier, out=np.ones_like(recent), where=earlier > 0)


def seasonal_naive_364_growth(history: np.ndarray, history_dates: pd.DatetimeIndex,
                              future_dates: pd.DatetimeIndex, window: int) -> np.ndarray:
    """The 364-day benchmark scaled by each series' recent year-over-year growth."""
    base = seasonal_naive_364(history, history_dates, future_dates)
    return base * growth_ratio(history, window)[:, None]


def month_weekday_profile(history: np.ndarray, history_dates: pd.DatetimeIndex,
                          future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Last year's average for the same month and weekday.

    The EDA found that the level is flat inside a month, so averaging the 4-5 matching days of
    last year removes most of the one-day noise that the plain 364-day lag copies.

    Raises:
        ValueError: If a forecast day has no matching day last year in the history.
    """
    check_inputs(history, history_dates, future_dates)
    same_cell = (
        (history_dates.year.to_numpy()[None, :] == future_dates.year.to_numpy()[:, None] - 1)
        & (history_dates.month.to_numpy()[None, :] == future_dates.month.to_numpy()[:, None])
        & (history_dates.day_of_week.to_numpy()[None, :] == future_dates.day_of_week.to_numpy()[:, None])
    ).astype(float)
    counts = same_cell.sum(axis=1, keepdims=True)
    if (counts == 0).any():
        raise ValueError("some forecast days have no matching month and weekday in last year's history")
    return history @ (same_cell / counts).T  # [n_series, horizon]


def month_weekday_profile_growth(history: np.ndarray, history_dates: pd.DatetimeIndex,
                                 future_dates: pd.DatetimeIndex, window: int) -> np.ndarray:
    """The month-and-weekday profile scaled by each series' recent year-over-year growth."""
    return month_weekday_profile(history, history_dates, future_dates) * growth_ratio(history, window)[:, None]


def month_weekday_profile_multi_year(history: np.ndarray, history_dates: pd.DatetimeIndex,
                                     future_dates: pd.DatetimeIndex, years: int, window: int) -> np.ndarray:
    """Average of ``years`` one-year-style forecasts, each from a different past year.

    For k = 1..years the same month-and-weekday average from k years back is scaled by the growth
    measured against k years back, and the results are averaged. Pooling several years cuts the
    noise in each year's 4-5 day average, which is why it beats the single-year profile.

    Raises:
        ValueError: If the history is too short or a forecast day has no matching day in a past year.
    """
    check_inputs(history, history_dates, future_dates)
    forecasts = []
    for k in range(1, years + 1):
        same_cell = (
            (history_dates.year.to_numpy()[None, :] == future_dates.year.to_numpy()[:, None] - k)
            & (history_dates.month.to_numpy()[None, :] == future_dates.month.to_numpy()[:, None])
            & (history_dates.day_of_week.to_numpy()[None, :] == future_dates.day_of_week.to_numpy()[:, None])
        ).astype(float)
        counts = same_cell.sum(axis=1, keepdims=True)
        if (counts == 0).any():
            raise ValueError(f"some forecast days have no matching month and weekday {k} year(s) back")
        cell_mean = history @ (same_cell / counts).T  # [n_series, horizon]
        forecasts.append(cell_mean * growth_ratio(history, window, years_back=k)[:, None])
    return np.mean(forecasts, axis=0)


def build_baselines(cfg: BaselineConfig) -> Dict[str, Forecaster]:
    """Instantiate the configured baselines with their parameters bound."""
    available: Dict[str, Forecaster] = {
        "naive": naive,
        "moving_average": partial(moving_average, window=cfg.moving_average_window),
        "seasonal_naive_7": seasonal_naive_7,
        "seasonal_moving_average": partial(seasonal_moving_average, weeks=cfg.seasonal_weeks),
        "seasonal_naive_364": seasonal_naive_364,
        "seasonal_naive_364_growth": partial(seasonal_naive_364_growth, window=cfg.growth_window),
        "month_weekday_profile": month_weekday_profile,
        "month_weekday_profile_growth": partial(month_weekday_profile_growth, window=cfg.growth_window),
        "month_weekday_profile_multi_year": partial(month_weekday_profile_multi_year, years=cfg.profile_years,
                                                    window=cfg.growth_window),
    }
    return {name: available[name] for name in cfg.names}
