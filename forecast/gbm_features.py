"""Leakage-safe features for the global gradient-boosting model.

A row is (series, origin, target day): everything is computed from sales up to the origin, the
target is the target day's sales divided by the series' trailing-year average (the scale), and the
same code builds the training rows and the forecast rows, so the two always agree.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from forecast.config import GbmConfig

YEAR_LAG = 364
MAX_SAFE_HORIZON = 330  # past this, last year's same-month cell would contain days after the origin
EPSILON = 1e-9


def feature_names(cfg: GbmConfig) -> List[str]:
    """Column names of the feature matrix, in order."""
    names = ["log_scale", *[f"mean{w}_ratio" for w in cfg.trend_windows], *[f"growth{w}" for w in cfg.trend_windows],
             "growth_year", "lag364"]
    for k in range(1, cfg.profile_years + 1):
        names += [f"cell{k}", f"month{k}"]
    names += ["month", "dayofweek", "dayofmonth", "horizon"]
    if cfg.use_series_ids:
        names += ["series_id"]
    return names


class FeatureBuilder:
    """Precomputes prefix sums and month/weekday averages of one history matrix.

    Args:
        history: Sales, shape [n_series, n_days], daily and contiguous.
        dates: The history's dates.
        cfg: Gradient-boosting settings.

    Raises:
        ValueError: If the history is not a finite [n_series, n_days] matrix matching ``dates``.
    """

    def __init__(self, history: np.ndarray, dates: pd.DatetimeIndex, cfg: GbmConfig) -> None:
        if history.ndim != 2 or history.shape[1] != len(dates):
            raise ValueError("history must be [n_series, n_days] and match dates")
        if not np.isfinite(history).all():
            raise ValueError("history must be finite")
        self.history, self.dates, self.cfg = history, dates, cfg
        n_series, n_days = history.shape
        self.prefix = np.concatenate([np.zeros((n_series, 1)), np.cumsum(history, axis=1)], axis=1)
        self.first_year = int(dates.year.min())
        self.n_years = int(dates.year.max()) - self.first_year + 1
        year, month = dates.year.to_numpy() - self.first_year, dates.month.to_numpy() - 1
        cell_id = (year * 12 + month) * 7 + dates.dayofweek.to_numpy()
        month_id = year * 12 + month
        self.cell_sum = np.stack([np.bincount(cell_id, weights=row, minlength=self.n_years * 84) for row in history])
        self.cell_count = np.bincount(cell_id, minlength=self.n_years * 84).astype(float)
        self.month_sum = np.stack([np.bincount(month_id, weights=row, minlength=self.n_years * 12) for row in history])
        self.month_count = np.bincount(month_id, minlength=self.n_years * 12).astype(float)

    def _window_mean(self, series: np.ndarray, origin: np.ndarray, window: int, back: int = 0) -> np.ndarray:
        """Mean of the ``window`` days ending ``back`` days before the origin (NaN if not enough history)."""
        end = origin + 1 - back
        start = end - window
        valid = start >= 0
        safe_end, safe_start = np.where(valid, end, window), np.where(valid, start, 0)
        mean = (self.prefix[series, safe_end] - self.prefix[series, safe_start]) / window
        return np.where(valid, mean, np.nan)

    def _lookup(self, sums: np.ndarray, counts: np.ndarray, series: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Average from a sum/count table at ``index`` (NaN when out of range or empty)."""
        size = len(counts)
        inside = (index >= 0) & (index < size)
        safe = np.where(inside, index, 0)
        count = counts[safe]
        ok = inside & (count > 0)
        return np.where(ok, sums[series, safe] / np.maximum(count, 1.0), np.nan)

    def features(self, series: np.ndarray, origin: np.ndarray, target_dates: pd.DatetimeIndex,
                 horizon: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Feature matrix and scale for rows (series[i], origin[i], target_dates[i]).

        Returns:
            (X, scale): ``X`` has the columns of :func:`feature_names`; ``scale`` is the series'
            trailing-year average at the origin, used to normalise the target and to rescale forecasts.

        Raises:
            ValueError: If an origin is before ``min_origin`` or a horizon exceeds the leakage-safe limit.
        """
        cfg = self.cfg
        if origin.min() < cfg.min_origin:
            raise ValueError(f"origins must be at least {cfg.min_origin}")
        if horizon.min() < 1 or horizon.max() > MAX_SAFE_HORIZON:
            raise ValueError(f"horizon must be between 1 and {MAX_SAFE_HORIZON}")

        scale = np.maximum(self._window_mean(series, origin, cfg.level_window), EPSILON)
        columns = {"log_scale": np.log(scale)}
        for w in cfg.trend_windows:
            columns[f"mean{w}_ratio"] = self._window_mean(series, origin, w) / scale
        for w in cfg.trend_windows:
            recent, earlier = self._window_mean(series, origin, w), self._window_mean(series, origin, w, YEAR_LAG)
            columns[f"growth{w}"] = np.where(earlier > 0, recent / np.maximum(earlier, EPSILON), np.nan)
        recent, earlier = scale, self._window_mean(series, origin, cfg.level_window, YEAR_LAG)
        columns["growth_year"] = np.where(earlier > 0, recent / np.maximum(earlier, EPSILON), np.nan)

        target_index = origin + horizon  # absolute day index of the target
        lag = target_index - YEAR_LAG
        in_history = lag < self.history.shape[1]
        columns["lag364"] = np.where(in_history & (lag >= 0),
                                     self.history[series, np.clip(lag, 0, self.history.shape[1] - 1)], np.nan) / scale

        year = target_dates.year.to_numpy() - self.first_year
        month, weekday = target_dates.month.to_numpy() - 1, target_dates.dayofweek.to_numpy()
        for k in range(1, cfg.profile_years + 1):
            cell = self._lookup(self.cell_sum, self.cell_count, series, ((year - k) * 12 + month) * 7 + weekday)
            month_mean = self._lookup(self.month_sum, self.month_count, series, (year - k) * 12 + month)
            columns[f"cell{k}"], columns[f"month{k}"] = cell / scale, month_mean / scale
        columns["month"] = month + 1
        columns["dayofweek"] = weekday
        columns["dayofmonth"] = target_dates.day.to_numpy()
        columns["horizon"] = horizon
        if cfg.use_series_ids:
            columns["series_id"] = series
        return pd.DataFrame(columns)[feature_names(cfg)], scale

    def training_rows(self, max_horizon: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """One training row per (series, target day) with a random horizon in 1..max_horizon.

        Returns:
            (X, y, weights): ``y`` is sales / scale and the weights are the scales, so large series
            count in proportion to their volume (matching WAPE).

        Raises:
            ValueError: If the history is too short to build any row.
        """
        n_series, n_days = self.history.shape
        targets = np.arange(self.cfg.min_origin + 1, n_days)
        if len(targets) == 0:
            raise ValueError("history is too short to build training rows")
        series = np.repeat(np.arange(n_series), len(targets))
        target_index = np.tile(targets, n_series)
        horizon = rng.integers(1, max_horizon + 1, size=len(target_index))
        origin = target_index - horizon
        keep = origin >= self.cfg.min_origin
        series, target_index, horizon, origin = series[keep], target_index[keep], horizon[keep], origin[keep]
        features, scale = self.features(series, origin, self.dates[target_index], horizon)
        return features, self.history[series, target_index] / scale, scale

    def forecast_rows(self, future_dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, np.ndarray]:
        """Rows for forecasting ``future_dates`` from the last history day (the origin)."""
        n_series, n_days = self.history.shape
        horizon = np.tile(np.arange(1, len(future_dates) + 1), n_series)
        series = np.repeat(np.arange(n_series), len(future_dates))
        origin = np.full(len(series), n_days - 1)
        return self.features(series, origin, pd.DatetimeIndex(np.tile(future_dates.to_numpy(), n_series)), horizon)
