"""Bias calibration and blending of the boosting model with a baseline.

Both corrections are decided on the last ``horizon`` days of the history the model is given (a
pseudo-forecast made without those days), so they never use the days being forecast.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from forecast.baselines import Forecaster
from forecast.config import TuneConfig
from forecast.metrics import wape

logger = logging.getLogger(__name__)

VARIANTS = ("raw", "calibrated", "blend", "calibrated_blend")


@dataclass(frozen=True)
class StackComponents:
    """Everything needed to build every variant for one history."""

    raw: np.ndarray
    baseline: np.ndarray
    factor: float
    weight_raw: float
    weight_calibrated: float


def best_blend_weight(actual: np.ndarray, model: np.ndarray, baseline: np.ndarray, step: float) -> float:
    """Weight of ``model`` (0..1) that minimises WAPE of ``w * model + (1 - w) * baseline``.

    Ties go to the larger weight, i.e. toward the boosting model.
    """
    grid = np.linspace(1.0, 0.0, int(round(1.0 / step)) + 1)
    errors = [wape(actual, w * model + (1.0 - w) * baseline) for w in grid]
    return float(grid[int(np.argmin(errors))])


class StackedModels:
    """Computes calibrated and blended forecasts from one boosting model and one baseline.

    The expensive parts (an inner fit and the final fit) run once per history and are cached, so the
    four variants cost two fits per validation window.
    """

    def __init__(self, gbm: Forecaster, baseline: Forecaster, cfg: TuneConfig) -> None:
        self.gbm, self.baseline, self.cfg = gbm, baseline, cfg
        self.diagnostics: List[Dict[str, Any]] = []
        self._cache: Dict[Tuple[int, float], StackComponents] = {}

    def components(self, history: np.ndarray, history_dates: pd.DatetimeIndex,
                   future_dates: pd.DatetimeIndex) -> StackComponents:
        """Fit (or reuse) the inner and final models for this history.

        Raises:
            ValueError: If the history is not longer than the forecast horizon.
        """
        key = (history.shape[1], round(float(history.sum()), 3))
        if key in self._cache:
            return self._cache[key]
        horizon, n_days = len(future_dates), history.shape[1]
        inner_end = n_days - horizon
        if inner_end < 1:
            raise ValueError("history must be longer than the forecast horizon")
        inner_history, inner_dates = history[:, :inner_end], history_dates[:inner_end]
        inner_future, inner_actual = history_dates[inner_end:], history[:, inner_end:]

        inner_model = self.gbm(inner_history, inner_dates, inner_future)
        inner_baseline = self.baseline(inner_history, inner_dates, inner_future)
        low, high = self.cfg.calibration_clip
        factor = float(np.clip(inner_actual.sum() / max(inner_model.sum(), 1e-9), low, high))
        weight_raw = best_blend_weight(inner_actual, inner_model, inner_baseline, self.cfg.blend_step)
        weight_cal = best_blend_weight(inner_actual, factor * inner_model, inner_baseline, self.cfg.blend_step)

        components = StackComponents(
            raw=self.gbm(history, history_dates, future_dates),
            baseline=self.baseline(history, history_dates, future_dates),
            factor=factor, weight_raw=weight_raw, weight_calibrated=weight_cal)
        self._cache[key] = components
        self.diagnostics.append({"history_days": n_days, "calibration_factor": round(factor, 4),
                                 "blend_weight_raw": weight_raw, "blend_weight_calibrated": weight_cal})
        logger.info("stack: factor %.4f, blend weights %.1f (raw) / %.1f (calibrated)", factor, weight_raw, weight_cal)
        return components

    def variant(self, kind: str) -> Callable[[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex], np.ndarray]:
        """Forecaster for one variant: 'raw', 'calibrated', 'blend' or 'calibrated_blend'.

        Raises:
            ValueError: For an unknown variant name.
        """
        if kind not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")

        def forecast(history: np.ndarray, history_dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex) -> np.ndarray:
            c = self.components(history, history_dates, future_dates)
            if kind == "raw":
                return c.raw
            if kind == "calibrated":
                return c.factor * c.raw
            if kind == "blend":
                return c.weight_raw * c.raw + (1.0 - c.weight_raw) * c.baseline
            return c.weight_calibrated * (c.factor * c.raw) + (1.0 - c.weight_calibrated) * c.baseline

        return forecast
