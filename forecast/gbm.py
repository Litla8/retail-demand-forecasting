"""Global gradient-boosting forecaster (LightGBM or scikit-learn's histogram boosting).

One model is trained across all series. It is refitted from scratch inside every call using only
the history it is given, so a backtest cannot leak the validation period into the training data.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from forecast.baselines import check_inputs
from forecast.config import GbmConfig
from forecast.gbm_features import MAX_SAFE_HORIZON, FeatureBuilder, feature_names

logger = logging.getLogger(__name__)


def build_regressor(cfg: GbmConfig, random_state: int) -> Any:
    """Create the boosting regressor for the configured engine.

    Raises:
        ImportError: If LightGBM is requested but not installed.
    """
    if cfg.engine == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples, subsample=cfg.subsample, subsample_freq=1,
            colsample_bytree=cfg.colsample, reg_lambda=cfg.reg_lambda, random_state=random_state,
            n_jobs=cfg.n_jobs, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=cfg.n_estimators, learning_rate=cfg.learning_rate, max_leaf_nodes=cfg.num_leaves,
        min_samples_leaf=cfg.min_child_samples, l2_regularization=cfg.reg_lambda, early_stopping=False,
        random_state=random_state)


class GbmForecaster:
    """Callable ``(history, history_dates, future_dates) -> forecast`` backed by a global boosting model.

    Attributes:
        importance: Gain-based feature importance of the latest fit (LightGBM only; None otherwise).
        fit_seconds: Total time spent fitting and predicting.
    """

    def __init__(self, cfg: GbmConfig, max_horizon: int, random_state: int = 42) -> None:
        if not 1 <= max_horizon <= MAX_SAFE_HORIZON:
            raise ValueError(f"max_horizon must be between 1 and {MAX_SAFE_HORIZON}")
        self.cfg, self.max_horizon, self.random_state = cfg, max_horizon, random_state
        self.importance: Optional[Dict[str, float]] = None
        self.fit_seconds = 0.0

    def __call__(self, history: np.ndarray, history_dates: pd.DatetimeIndex,
                 future_dates: pd.DatetimeIndex) -> np.ndarray:
        check_inputs(history, history_dates, future_dates)
        if len(future_dates) > self.max_horizon:
            raise ValueError(f"forecast horizon {len(future_dates)} exceeds max_horizon {self.max_horizon}")
        began = time.perf_counter()
        builder = FeatureBuilder(history, history_dates, self.cfg)
        rng = np.random.default_rng(self.random_state)  # same seed every call -> repeatable fits
        features, target, weight = builder.training_rows(self.max_horizon, rng)
        model = build_regressor(self.cfg, self.random_state)
        model.fit(features, target, sample_weight=weight)

        rows, scale = builder.forecast_rows(future_dates)
        ratio = np.asarray(model.predict(rows), dtype=float)
        forecast = np.maximum(ratio * scale, 0.0).reshape(history.shape[0], len(future_dates))
        if self.cfg.engine == "lightgbm":
            gains = model.booster_.feature_importance(importance_type="gain")
            self.importance = {name: float(g) for name, g in zip(feature_names(self.cfg), gains)}
        self.fit_seconds += time.perf_counter() - began
        logger.info("gbm (%s): trained on %d rows, forecast %d series x %d days in %.1fs", self.cfg.engine,
                    len(features), history.shape[0], len(future_dates), time.perf_counter() - began)
        return forecast
