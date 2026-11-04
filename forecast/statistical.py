"""Statistical forecasting models fitted one series at a time: Holt-Winters, SARIMA(X) and Prophet.

All of them are wrapped as ``(history, history_dates, future_dates) -> forecast`` so they plug into
the same backtest as the baselines. statsmodels and prophet are imported lazily, so the rest of
the project works without them.
"""
from __future__ import annotations

import importlib
import logging
import time
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from forecast.baselines import check_inputs, seasonal_naive_7
from forecast.config import StatisticalConfig
from forecast.holidays import holiday_calendar
from forecast.wide import WideData

logger = logging.getLogger(__name__)

SingleSeriesModel = Callable[[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex], np.ndarray]
REQUIRED_LIBRARY = {
    "holt_winters": "statsmodels", "sarima": "statsmodels", "sarimax_month": "statsmodels",
    "prophet_default": "prophet", "prophet_month": "prophet", "prophet_month_holidays": "prophet",
}


def select_series_by_volume(wide: WideData, n: int) -> WideData:
    """Pick ``n`` series at evenly spaced sales volumes (lowest to highest), deterministically.

    Raises:
        ValueError: If ``n`` is not between 1 and the number of series.
    """
    if not 1 <= n <= wide.n_series:
        raise ValueError(f"n must be between 1 and {wide.n_series}")
    order = np.argsort(wide.values.mean(axis=1), kind="stable")
    picks = np.round(np.linspace(0, wide.n_series - 1, n)).astype(int)
    positions = np.sort(order[picks])
    return WideData(values=wide.values[positions], dates=wide.dates, series=wide.series[positions])


def month_dummies(dates: pd.DatetimeIndex) -> np.ndarray:
    """[len(dates), 11] indicator columns for February..December (January is the baseline).

    The EDA showed the yearly pattern is a staircase by month, so these give a model the steps
    that smooth yearly seasonality cannot represent.
    """
    months = dates.month.to_numpy()[:, None]
    return (months == np.arange(2, 13)[None, :]).astype(float)


def holt_winters_forecast(y: np.ndarray, dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Holt-Winters exponential smoothing: damped additive trend and weekly seasonality.

    Seasonality is multiplicative when every observation is positive, otherwise additive.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal = "mul" if (y > 0).all() else "add"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence warnings are expected and counted via the fallback
        fitted = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=seasonal, seasonal_periods=7,
                                      initialization_method="estimated").fit()
    return np.asarray(fitted.forecast(len(future_dates)), dtype=float)


def sarimax_forecast(y: np.ndarray, dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex,
                     order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int],
                     maxiter: int, use_month: bool) -> np.ndarray:
    """SARIMA with a weekly seasonal difference, optionally with month dummies as regressors."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    exog = month_dummies(dates) if use_month else None
    exog_future = month_dummies(future_dates) if use_month else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=maxiter)
        forecast = result.forecast(steps=len(future_dates), exog=exog_future)
    return np.asarray(forecast, dtype=float)


def _quiet_prophet_logs() -> None:
    """Prophet and cmdstanpy log every fit at INFO level, which floods the console."""
    for name in ("prophet", "cmdstanpy"):
        logging.getLogger(name).setLevel(logging.WARNING)


def prophet_forecast(y: np.ndarray, dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex,
                     use_month: bool, use_holidays: bool) -> np.ndarray:
    """Prophet with weekly seasonality; yearly Fourier seasonality unless month dummies are used.

    Args:
        use_month: Replace the smooth yearly seasonality by month dummy regressors.
        use_holidays: Add the Step 3 holiday calendar (which the EDA found to have no effect).
    """
    from prophet import Prophet

    _quiet_prophet_logs()
    history = pd.DataFrame({"ds": dates, "y": y})
    future = pd.DataFrame({"ds": future_dates})
    regressors = [f"month_{m}" for m in range(2, 13)] if use_month else []
    if use_month:
        history[regressors] = month_dummies(dates)
        future[regressors] = month_dummies(future_dates)
    holidays: Optional[pd.DataFrame] = None
    if use_holidays:
        holidays = holiday_calendar(dates.min(), future_dates.max()).rename(columns={"date": "ds"})[["holiday", "ds"]]

    model = Prophet(yearly_seasonality=not use_month, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode="multiplicative", holidays=holidays)
    for name in regressors:
        model.add_regressor(name)
    model.fit(history)
    return np.asarray(model.predict(future)["yhat"], dtype=float)


@dataclass
class SeriesModelForecaster:
    """Fits one model per series (row) and forecasts the horizon, counting failed fits.

    A failed or non-finite fit falls back to the 7-day seasonal naive forecast and is counted in
    ``failures``, so one bad series cannot stop a long run (and the count is reported). Forecasts
    are clipped at zero because sales cannot be negative.
    """

    name: str
    model: SingleSeriesModel
    fit_days: int
    failures: int = 0
    fit_seconds: float = 0.0

    def __call__(self, history: np.ndarray, history_dates: pd.DatetimeIndex,
                 future_dates: pd.DatetimeIndex) -> np.ndarray:
        check_inputs(history, history_dates, future_dates)
        start = max(0, history.shape[1] - self.fit_days)
        dates = history_dates[start:]
        fallback = seasonal_naive_7(history, history_dates, future_dates)
        forecast = np.empty((history.shape[0], len(future_dates)))
        began = time.perf_counter()
        for row in range(history.shape[0]):
            try:
                result = np.maximum(np.asarray(self.model(history[row, start:], dates, future_dates), dtype=float), 0.0)
                if result.shape != (len(future_dates),) or not np.isfinite(result).all():
                    raise ValueError(f"model returned an invalid forecast of shape {result.shape}")
                forecast[row] = result
            except ImportError:
                raise  # a missing library is a setup problem, not a bad series
            except Exception as exc:  # noqa: BLE001  (library fits raise many unrelated exception types)
                self.failures += 1
                logger.warning("%s failed on series %d (%s); using seasonal naive 7", self.name, row, exc)
                forecast[row] = fallback[row]
        elapsed = time.perf_counter() - began
        self.fit_seconds += elapsed
        logger.info("%s: %d series x %d days in %.1fs (%d failures so far)", self.name, history.shape[0],
                    len(future_dates), elapsed, self.failures)
        return forecast


def build_statistical_models(cfg: StatisticalConfig) -> Dict[str, SeriesModelForecaster]:
    """Instantiate the configured models, skipping those whose library is not installed."""
    sarima = dict(order=cfg.sarima_order, seasonal_order=cfg.sarima_seasonal_order, maxiter=cfg.sarima_maxiter)
    definitions: Dict[str, Tuple[SingleSeriesModel, int]] = {
        "holt_winters": (holt_winters_forecast, cfg.ets_fit_days),
        "sarima": (partial(sarimax_forecast, use_month=False, **sarima), cfg.sarima_fit_days),
        "sarimax_month": (partial(sarimax_forecast, use_month=True, **sarima), cfg.sarima_fit_days),
        "prophet_default": (partial(prophet_forecast, use_month=False, use_holidays=False), cfg.prophet_fit_days),
        "prophet_month": (partial(prophet_forecast, use_month=True, use_holidays=False), cfg.prophet_fit_days),
        "prophet_month_holidays": (partial(prophet_forecast, use_month=True, use_holidays=True), cfg.prophet_fit_days),
    }
    built: Dict[str, SeriesModelForecaster] = {}
    for name in cfg.models:
        library = REQUIRED_LIBRARY[name]
        try:
            importlib.import_module(library)
        except ImportError:
            logger.warning("Skipping %s: %s is not installed", name, library)
            continue
        model, fit_days = definitions[name]
        built[name] = SeriesModelForecaster(name=name, model=model, fit_days=fit_days)
    return built
