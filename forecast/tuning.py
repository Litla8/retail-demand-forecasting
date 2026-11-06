"""Hyperparameter search for the global boosting model.

Trials are scored on windows that end before the first validation window starts, so tuning never
sees validation data. The default configuration is always trial 0, so the search can only match or
improve on it.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from forecast.config import ForecastConfig, GbmConfig
from forecast.gbm import GbmForecaster
from forecast.metrics import wape
from forecast.splits import ValidationWindow
from forecast.wide import WideData, window_arrays

logger = logging.getLogger(__name__)

# name -> (low, high, kind); "log" kinds are sampled log-uniformly
SEARCH_SPACE: Dict[str, Tuple[float, float, str]] = {
    "num_leaves": (8, 128, "int_log"),
    "min_child_samples": (20, 500, "int_log"),
    "learning_rate": (0.02, 0.2, "float_log"),
    "n_estimators": (100, 600, "int"),
    "subsample": (0.5, 1.0, "float"),
    "colsample": (0.5, 1.0, "float"),
    "reg_lambda": (1e-3, 10.0, "float_log"),
}


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a search: best parameters, their score, the default's score and every trial."""

    best_params: Dict[str, Any]
    best_value: float
    default_value: float
    trials: List[Dict[str, Any]]
    search: str

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly view without the full trial list."""
        out = asdict(self)
        out.pop("trials")
        out["n_trials"] = len(self.trials)
        return out


def default_params(gbm_cfg: GbmConfig) -> Dict[str, Any]:
    """The tunable values of the current configuration."""
    return {name: getattr(gbm_cfg, name) for name in SEARCH_SPACE}


def random_params(rng: np.random.Generator) -> Dict[str, Any]:
    """Draw one parameter set from the search space."""
    params: Dict[str, Any] = {}
    for name, (low, high, kind) in SEARCH_SPACE.items():
        value = float(np.exp(rng.uniform(np.log(low), np.log(high)))) if kind.endswith("log") else float(rng.uniform(low, high))
        params[name] = int(round(value)) if kind.startswith("int") else value
    return params


def score_params(wide: WideData, windows: Sequence[ValidationWindow], gbm_cfg: GbmConfig,
                params: Dict[str, Any], horizon: int, seed: int) -> float:
    """Mean WAPE of a boosting model with ``params`` across the given windows."""
    model = GbmForecaster(replace(gbm_cfg, **params), horizon, seed)
    errors = []
    for window in windows:
        arrays = window_arrays(wide, window)
        errors.append(wape(arrays.actual, model(arrays.history, arrays.history_dates, arrays.future_dates)))
    return float(np.mean(errors))


def _optuna_suggest(trial: Any, name: str) -> Any:
    """Suggest one parameter from an Optuna trial according to SEARCH_SPACE."""
    low, high, kind = SEARCH_SPACE[name]
    if kind.startswith("int"):
        return trial.suggest_int(name, int(low), int(high), log=kind.endswith("log"))
    return trial.suggest_float(name, low, high, log=kind.endswith("log"))


def tune_gbm(wide: WideData, windows: Sequence[ValidationWindow], cfg: ForecastConfig,
             n_trials: int, search: str) -> TuningResult:
    """Search the boosting hyperparameters and return the best configuration found.

    Args:
        wide: Sales matrix.
        windows: Tuning windows (must lie before every validation window).
        cfg: Project configuration (its ``gbm`` section is the starting point).
        n_trials: Number of trials including the default configuration.
        search: "optuna" or "random"; optuna falls back to random if it is not installed.

    Raises:
        ValueError: If ``n_trials`` is not positive.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    base, horizon, seed = cfg.gbm, cfg.horizon_days, cfg.random_state
    defaults = default_params(base)
    trials: List[Dict[str, Any]] = []

    def evaluate(number: int, params: Dict[str, Any]) -> float:
        value = score_params(wide, windows, base, params, horizon, seed)
        trials.append({"trial": number, "value": value, **params})
        logger.info("trial %d: WAPE %.3f %s", number, value, {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()})
        return value

    if search == "optuna":
        try:
            import optuna
        except ImportError:
            logger.warning("optuna is not installed; using random search instead")
            search = "random"
    if search == "optuna":
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.enqueue_trial(defaults)  # the default configuration is trial 0
        study.optimize(lambda trial: evaluate(trial.number, {n: _optuna_suggest(trial, n) for n in SEARCH_SPACE}),
                       n_trials=n_trials)
    else:
        rng = np.random.default_rng(seed)
        evaluate(0, defaults)
        for number in range(1, n_trials):
            evaluate(number, random_params(rng))

    best = min(trials, key=lambda t: t["value"])
    return TuningResult(best_params={name: best[name] for name in SEARCH_SPACE}, best_value=float(best["value"]),
                        default_value=float(trials[0]["value"]), trials=trials, search=search)
