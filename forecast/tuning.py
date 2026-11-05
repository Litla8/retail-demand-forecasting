"""
Step 7: Optuna tuning, trailing bias calibration, weighted ensembling, and error analysis
by store, item (category proxy), and forecast horizon (days 1-28, 29-60, 61-90).

Designed to plug directly into the `forecast/` package alongside `gbm.py` and `run_gbm.py`,
while remaining 100% self-contained so it never fails if internal helper signatures vary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Core Metrics (matching forecast/metrics.py)
# ==============================================================================
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error in percent."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(np.abs(y_true))
    if denom <= 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error in percent."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bias_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage bias: (sum(pred) - sum(true)) / sum(true) * 100."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(y_true)
    if denom <= 0:
        return 0.0
    return float((np.sum(y_pred) - denom) / denom * 100.0)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "wape": round(wape(y_true, y_pred), 4),
        "smape": round(smape(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "bias_pct": round(bias_pct(y_true, y_pred), 4),
    }


# ==============================================================================
# 2. Vectorized Baselines & Leakage-Safe Step 6 Features (500 Series x 4 Windows)
# ==============================================================================
VALIDATION_WINDOWS = [
    ("2017-03-31", "2017-01-01", "2017-03-31"),
    ("2017-06-30", "2017-04-02", "2017-06-30"),
    ("2017-09-30", "2017-07-03", "2017-09-30"),
    ("2017-12-31", "2017-10-03", "2017-12-31"),
]


def compute_window_baselines(
    df: pd.DataFrame, origin_date: pd.Timestamp, val_start: pd.Timestamp, val_end: pd.Timestamp
) -> pd.DataFrame:
    """
    Compute the 364-day seasonal naive benchmark and the 2-year pooled month-weekday
    profile (with trailing 91-day YoY growth scaling) using ONLY data <= origin_date.
    """
    hist = df[df["date"] <= origin_date].copy()
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()

    hist["month"] = hist["date"].dt.month
    hist["weekday"] = hist["date"].dt.dayofweek
    val["month"] = val["date"].dt.month
    val["weekday"] = val["date"].dt.dayofweek
    val["horizon"] = (val["date"] - origin_date).dt.days

    # 1. 364-day seasonal naive benchmark
    lookup_364 = hist[["store", "item", "date", "sales"]].copy()
    lookup_364["target_date"] = lookup_364["date"] + pd.Timedelta(days=364)
    val = val.merge(
        lookup_364[["store", "item", "target_date", "sales"]].rename(
            columns={"target_date": "date", "sales": "pred_benchmark_364"}
        ),
        on=["store", "item", "date"],
        how="left",
    )

    # Trailing 91-day YoY growth factor up to origin_date
    recent_91_start = origin_date - pd.Timedelta(days=90)
    prev_91_end = origin_date - pd.Timedelta(days=364)
    prev_91_start = prev_91_end - pd.Timedelta(days=90)

    rec_mean = (
        hist[(hist["date"] >= recent_91_start) & (hist["date"] <= origin_date)]
        .groupby(["store", "item"])["sales"]
        .mean()
    )
    prv_mean = (
        hist[(hist["date"] >= prev_91_start) & (hist["date"] <= prev_91_end)]
        .groupby(["store", "item"])["sales"]
        .mean()
    )
    growth_1y = (rec_mean / prv_mean.replace(0, np.nan)).fillna(1.0).clip(0.8, 1.25)

    # Prior year (Y-1: last 365 days before origin) and two years back (Y-2: 365 to 730 days before origin)
    y1_cut = origin_date - pd.Timedelta(days=365)
    y2_cut = origin_date - pd.Timedelta(days=730)

    hist_y1 = hist[(hist["date"] > y1_cut) & (hist["date"] <= origin_date)]
    hist_y2 = hist[(hist["date"] > y2_cut) & (hist["date"] <= y1_cut)]

    # 2-year growth factor (origin vs 2 years ago)
    prv2_91_end = origin_date - pd.Timedelta(days=728)
    prv2_91_start = prv2_91_end - pd.Timedelta(days=90)
    prv2_mean = (
        hist[(hist["date"] >= prv2_91_start) & (hist["date"] <= prv2_91_end)]
        .groupby(["store", "item"])["sales"]
        .mean()
    )
    growth_2y = (rec_mean / prv2_mean.replace(0, np.nan)).fillna(growth_1y ** 2).clip(0.75, 1.35)

    prof_y1 = (
        hist_y1.groupby(["store", "item", "month", "weekday"])["sales"]
        .mean()
        .rename("mw_y1")
        .reset_index()
    )
    prof_y2 = (
        hist_y2.groupby(["store", "item", "month", "weekday"])["sales"]
        .mean()
        .rename("mw_y2")
        .reset_index()
    )
    month_y1 = (
        hist_y1.groupby(["store", "item", "month"])["sales"]
        .mean()
        .rename("m_y1")
        .reset_index()
    )
    month_y2 = (
        hist_y2.groupby(["store", "item", "month"])["sales"]
        .mean()
        .rename("m_y2")
        .reset_index()
    )

    val = val.merge(prof_y1, on=["store", "item", "month", "weekday"], how="left")
    val = val.merge(prof_y2, on=["store", "item", "month", "weekday"], how="left")
    val = val.merge(month_y1, on=["store", "item", "month"], how="left")
    val = val.merge(month_y2, on=["store", "item", "month"], how="left")
    val = val.merge(growth_1y.rename("g1"), on=["store", "item"], how="left")
    val = val.merge(growth_2y.rename("g2"), on=["store", "item"], how="left")

    # 1-year and 2-year pooled month-weekday profile predictions
    val["pred_profile_1y"] = val["mw_y1"] * val["g1"]
    val["pred_profile_2y"] = 0.5 * (val["mw_y1"] * val["g1"] + val["mw_y2"] * val["g2"])
    val["pred_profile_2y"] = val["pred_profile_2y"].fillna(val["pred_profile_1y"]).fillna(
        val["pred_benchmark_364"]
    )

    return val


def build_gbm_dataset_for_window(
    df: pd.DataFrame,
    origin_date: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build leakage-safe training and validation matrices matching the Step 6 design:
    - Target scaled by series' trailing 364-day average (`scale_364`)
    - Sample weights equal to `scale_364` (aligning loss with volume-weighted WAPE)
    - Features: log_scale, mw_y1_rel, mw_y2_rel, m_y1_rel, m_y2_rel, lag_364_rel,
      growth_28, growth_91, growth_364, month, weekday, day, horizon
    """
    # Use the validation window and 4 historical 90-day blocks prior to origin_date for training
    train_blocks = []
    for k in range(1, 5):
        b_end = origin_date - pd.Timedelta(days=90 * (k - 1))
        b_start = b_end - pd.Timedelta(days=89)
        b_origin = b_start - pd.Timedelta(days=1)
        if (df["date"] <= b_origin - pd.Timedelta(days=730)).any():
            blk = compute_window_baselines(df, b_origin, b_start, b_end)
            # Trailing 364-day scale at block origin
            h_sub = df[(df["date"] > b_origin - pd.Timedelta(days=364)) & (df["date"] <= b_origin)]
            scale = h_sub.groupby(["store", "item"])["sales"].mean().rename("scale_364")
            blk = blk.merge(scale, on=["store", "item"], how="left")
            train_blocks.append(blk)

    train_df = pd.concat(train_blocks, ignore_index=True)

    val_df = compute_window_baselines(df, origin_date, val_start, val_end)
    h_val = df[(df["date"] > origin_date - pd.Timedelta(days=364)) & (df["date"] <= origin_date)]
    val_scale = h_val.groupby(["store", "item"])["sales"].mean().rename("scale_364")
    val_df = val_df.merge(val_scale, on=["store", "item"], how="left")

    feature_cols = [
        "mw_y1_rel",
        "mw_y2_rel",
        "m_y1_rel",
        "m_y2_rel",
        "lag_364_rel",
        "g1",
        "g2",
        "log_scale",
        "month",
        "weekday",
        "day",
        "horizon",
    ]

    for d in (train_df, val_df):
        s = d["scale_364"].replace(0, 1.0)
        d["mw_y1_rel"] = d["mw_y1"] / s
        d["mw_y2_rel"] = d["mw_y2"] / s
        d["m_y1_rel"] = d["m_y1"] / s
        d["m_y2_rel"] = d["m_y2"] / s
        d["lag_364_rel"] = d["pred_benchmark_364"] / s
        d["log_scale"] = np.log1p(s)
        d["day"] = d["date"].dt.day
        d["y_rel"] = d["sales"] / s

    train_df = train_df.dropna(subset=feature_cols + ["y_rel", "scale_364"]).reset_index(drop=True)
    return train_df, val_df, feature_cols


# ==============================================================================
# 3. Optuna Tuning, Bias Calibration & Weighted Blending
# ==============================================================================
@dataclass
class TuningResult:
    best_params: Dict
    best_blend_weight_lgbm: float
    window_metrics: pd.DataFrame
    predictions_df: pd.DataFrame


def fit_and_predict_gbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    params: Dict,
    engine: str = "lightgbm",
) -> np.ndarray:
    """Fit global gradient boosting model and return uncalibrated unit predictions on val_df."""
    X_tr = train_df[feature_cols]
    y_tr = train_df["y_rel"].to_numpy()
    w_tr = train_df["scale_364"].to_numpy()
    X_va = val_df[feature_cols]

    if engine == "lightgbm":
        try:
            import lightgbm as lgb

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            pred_rel = model.predict(X_va)
            return np.clip(pred_rel * val_df["scale_364"].to_numpy(), 0.0, None)
        except ImportError:
            logger.warning("LightGBM not installed; falling back to HistGradientBoostingRegressor.")

    from sklearn.ensemble import HistGradientBoostingRegressor

    hgb_params = {
        "loss": "absolute_error" if params.get("objective") in ("mae", "l1") else "squared_error",
        "max_leaf_nodes": params.get("num_leaves", 31),
        "max_depth": params.get("max_depth", 6),
        "min_samples_leaf": params.get("min_child_samples", 100),
        "learning_rate": params.get("learning_rate", 0.05),
        "l2_regularization": params.get("reg_lambda", 1.0),
        "max_iter": params.get("n_estimators", 250),
        "random_state": 42,
    }
    model = HistGradientBoostingRegressor(**hgb_params)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_rel = model.predict(X_va)
    return np.clip(pred_rel * val_df["scale_364"].to_numpy(), 0.0, None)


def estimate_bias_calibration_factor(
    train_df: pd.DataFrame, feature_cols: List[str], params: Dict, engine: str = "lightgbm"
) -> float:
    """
    Estimate multiplicative bias calibration factor on the most recent training block
    (the 90 days immediately preceding the forecast origin) so there is zero leakage.
    """
    latest_origin = train_df["date"].max()
    cutoff = latest_origin - pd.Timedelta(days=89)
    inner_tr = train_df[train_df["date"] < cutoff]
    inner_va = train_df[train_df["date"] >= cutoff]
    if len(inner_tr) < 1000 or len(inner_va) < 1000:
        return 1.0

    raw_preds = fit_and_predict_gbm(inner_tr, inner_va, feature_cols, params, engine=engine)
    denom = float(np.sum(raw_preds))
    if denom <= 0:
        return 1.0
    factor = float(np.sum(inner_va["sales"].to_numpy()) / denom)
    return float(np.clip(factor, 0.94, 1.04))


def tune_hyperparameters_optuna(
    window_datasets: List[Tuple[str, pd.DataFrame, pd.DataFrame, List[str]]],
    n_trials: int = 15,
    engine: str = "lightgbm",
) -> Dict:
    """
    Tune LightGBM hyperparameters using Optuna (or clean grid fallback if Optuna is absent)
    on historical inner splits to minimize mean WAPE.
    """
    default_untuned = {
        "objective": "regression",
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "learning_rate": 0.06,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.1,
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": trial.suggest_categorical("objective", ["mae", "regression"]),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 60, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 180, 400),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            # Evaluate on inner validation split of first two windows for speed & no test leakage
            wapes = []
            for _, tr_df, _, fcols in window_datasets[:2]:
                latest = tr_df["date"].max()
                c_date = latest - pd.Timedelta(days=89)
                in_tr = tr_df[tr_df["date"] < c_date]
                in_va = tr_df[tr_df["date"] >= c_date]
                preds = fit_and_predict_gbm(in_tr, in_va, fcols, params, engine=engine)
                wapes.append(wape(in_va["sales"].to_numpy(), preds))
            return float(np.mean(wapes))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)
        best = default_untuned.copy()
        best.update(study.best_params)
        logger.info("Optuna best inner-CV WAPE: %.3f%% | Params: %s", study.best_value, best)
        return best

    except ImportError:
        logger.info("Optuna not installed; using pre-validated L1/MAE regularized parameters.")
        return {
            "objective": "mae",
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 120,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.5,
            "reg_lambda": 2.5,
            "n_estimators": 280,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }


def find_optimal_blend_weight(
    y_true: np.ndarray, pred_lgbm: np.ndarray, pred_stat: np.ndarray
) -> float:
    """Find optimal weight `w` in `w * pred_lgbm + (1 - w) * pred_stat` minimizing WAPE."""
    grid = np.linspace(0.0, 1.0, 101)
    scores = [wape(y_true, w * pred_lgbm + (1.0 - w) * pred_stat) for w in grid]
    best_w = float(grid[int(np.argmin(scores))])
    res = minimize(
        lambda w: wape(
            y_true,
            float(np.clip(w[0], 0.0, 1.0)) * pred_lgbm
            + (1.0 - float(np.clip(w[0], 0.0, 1.0))) * pred_stat,
        ),
        x0=[best_w],
        bounds=[(0.0, 1.0)],
        method="L-BFGS-B",
    )
    return float(np.round(np.clip(res.x[0], 0.0, 1.0), 3))


# ==============================================================================
# 4. Granular Error Breakdown (by Store, Item/Category, and Horizon)
# ==============================================================================
def compute_error_breakdown(
    preds_df: pd.DataFrame, group_col: str, pred_col: str = "pred_ensemble"
) -> pd.DataFrame:
    """Evaluate WAPE, sMAPE, MAE, RMSE, Bias, and daily average volume by group."""
    rows = []
    for g_val, grp in preds_df.groupby(group_col, observed=True):
        y_t = grp["sales"].to_numpy()
        y_p = grp[pred_col].to_numpy()
        y_b = grp["pred_benchmark_364"].to_numpy()
        m = compute_all_metrics(y_t, y_p)
        b_wape = wape(y_t, y_b)
        rows.append(
            {
                group_col: g_val,
                "mean_daily_sales": round(float(np.mean(y_t)), 2),
                "wape": round(m["wape"], 2),
                "smape": round(m["smape"], 2),
                "mae": round(m["mae"], 2),
                "rmse": round(m["rmse"], 2),
                "bias_pct": round(m["bias_pct"], 2),
                "benchmark_wape": round(b_wape, 2),
                "vs_benchmark_pct": round((b_wape - m["wape"]) / max(b_wape, 1e-6) * 100.0, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("wape", ascending=False).reset_index(drop=True)