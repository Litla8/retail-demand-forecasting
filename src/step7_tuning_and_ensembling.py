"""
Project 2: Retail Sales & Demand Forecasting
Step 7 - Tuning & Ensembling

Builds directly on Steps 1-6:
- Step 1: 28-day horizon, store-item-day granularity, RMSE, MAE, sMAPE, WAPE metrics
- Step 2-3: Cleaned merged dataset (sales, calendar, price, promotions) & intermittent demand flags
- Step 4: Time-based validation split & seasonal naive baseline (lag_7 / lag_28)
- Step 5: Statistical model predictions (Prophet with holiday/promo regressors or ARIMA)
- Step 6: Supervised lag features (7, 14, 28), rolling mean/std, calendar/price/promo features,
          and expanding-window time-series cross-validation

What Step 7 Does:
1. Tunes LightGBM hyperparameters with Optuna using expanding-window time-series CV.
2. Blends tuned LightGBM predictions with Prophet / ARIMA predictions via optimal weighted averaging
   (using SciPy/grid search to minimize validation WAPE, plus category-specific blending weights).
3. Evaluates error (WAPE, sMAPE, MAE, RMSE, Bias) by:
   - Category (`cat_id` / `dept_id`)
   - Store (`store_id`)
   - Forecast Horizon (Day 1 to Day 28 ahead, plus weekly buckets)
   and flags the weakest segments for Step 8 safety-stock adjustments.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42


# ==============================================================================
# 1. Forecasting Metrics (from Step 1 & Step 4)
# ==============================================================================
def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (robust to zero/intermittent sales)."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0 to 200 scale)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core Step 1 business metrics: WAPE, sMAPE, MAE, RMSE, and Bias."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), a_min=0.0, a_max=None)
    return {
        "WAPE (%)": round(compute_wape(y_true, y_pred), 3),
        "sMAPE (%)": round(compute_smape(y_true, y_pred), 3),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "Bias (%)": round(
            float((np.sum(y_pred - y_true) / max(np.sum(y_true), 1e-8)) * 100.0), 3
        ),
    }


# ==============================================================================
# 2. Data Loading & Expanding Window Splits (from Step 2, 4, 6)
# ==============================================================================
def generate_synthetic_step6_data(
    n_stores: int = 4, n_items_per_store: int = 15, n_days: int = 365
) -> pd.DataFrame:
    """
    Fallback generator producing a Step 6 feature table if no local parquet/csv
    is found yet, so the pipeline can be tested immediately end-to-end.
    """
    logger.info(
        "Generating sample Step 6 feature dataset (%d stores x %d items x %d days)...",
        n_stores,
        n_items_per_store,
        n_days,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]

    records = []
    for store_idx in range(1, n_stores + 1):
        store_id = f"STORE_{store_idx:02d}"
        store_mult = 0.8 + 0.15 * store_idx
        for item_idx in range(1, n_items_per_store + 1):
            cat_id = categories[item_idx % len(categories)]
            item_id = f"{cat_id}_{item_idx:03d}"
            base_demand = rng.uniform(5.0, 45.0) * store_mult

            t = np.arange(n_days)
            weekly = 1.0 + 0.25 * np.sin(2 * np.pi * t / 7.0)
            yearly = 1.0 + 0.15 * np.cos(2 * np.pi * t / 365.25)
            promo_flag = rng.binomial(1, 0.12, size=n_days)
            holiday_flag = ((dates.dayofweek == 6) & (dates.day <= 7)).astype(int)
            price = np.round(rng.uniform(2.5, 25.0) * (1.0 - 0.15 * promo_flag), 2)

            noise = rng.normal(0, base_demand * 0.18, size=n_days)
            sales = (
                base_demand
                * weekly
                * yearly
                * (1.0 + 0.30 * promo_flag + 0.15 * holiday_flag)
                + noise
            )
            sales = np.maximum(0.0, np.round(sales))

            df_series = pd.DataFrame(
                {
                    "date": dates,
                    "store_id": store_id,
                    "item_id": item_id,
                    "cat_id": cat_id,
                    "sales": sales,
                    "price": price,
                    "promo_flag": promo_flag,
                    "holiday_flag": holiday_flag,
                }
            )
            records.append(df_series)

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # Step 6 Feature Engineering: Lags, Rolling Stats, Calendar, Price Change
    grp = df.groupby(["store_id", "item_id"])["sales"]
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Shift by 28 days before rolling to prevent target leakage across a 28-day horizon,
    # or shift by 7 for direct weekly rolling features
    shifted_sales = grp.shift(7)
    for win in [7, 14, 28]:
        df[f"rolling_mean_{win}"] = shifted_sales.transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )
        df[f"rolling_std_{win}"] = (
            shifted_sales.transform(lambda s: s.rolling(win, min_periods=2).std()).fillna(0.0)
        )

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["price_change_pct"] = (
        df.groupby(["store_id", "item_id"])["price"].pct_change().fillna(0.0)
    )
    df["is_intermittent"] = (
        df.groupby(["store_id", "item_id"])["sales"]
        .transform(lambda s: (s == 0).mean() > 0.30)
        .astype(int)
    )

    df = df.dropna(subset=["lag_28"]).reset_index(drop=True)
    return df


def load_step6_dataset(data_path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load processed Step 6 dataset and identify feature columns."""
    if data_path.exists():
        logger.info("Loading Step 6 engineered dataset from %s...", data_path)
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, parse_dates=["date"])
    else:
        logger.warning("File %s not found. Generating Step 6 feature dataset...", data_path)
        df = generate_synthetic_step6_data()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        logger.info("Saved generated Step 6 dataset to %s", data_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)

    cat_cols = [c for c in ["store_id", "item_id", "cat_id", "dept_id", "state_id"] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    exclude_cols = {
        "date",
        "sales",
        " prophet_pred",
        "prophet_pred",
        "arima_pred",
        "stat_pred",
        "seasonal_naive_pred",
        "horizon_day",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return df, feature_cols, cat_cols


def build_expanding_window_splits(
    df: pd.DataFrame, horizon: int = 28, n_splits: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create expanding-window time-series CV splits (from Step 6).
    Each validation fold spans exactly `horizon` (28) days.
    """
    max_date = df["date"].max()
    splits = []
    for i in range(n_splits, 0, -1):
        val_end = max_date - pd.Timedelta(days=horizon * (i - 1))
        val_start = val_end - pd.Timedelta(days=horizon - 1)
        train_idx = df.index[df["date"] < val_start].to_numpy()
        val_idx = df.index[(df["date"] >= val_start) & (df["date"] <= val_end)].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


# ==============================================================================
# 3. Optuna Hyperparameter Tuning for LightGBM
# ==============================================================================
def tune_lightgbm_with_optuna(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
) -> Tuple[Dict, optuna.Study]:
    """
    Tune LightGBM global model across expanding-window CV folds to minimize WAPE.
    Tests both regression (L2) and Tweedie objectives (ideal for zero-inflated retail sales).
    """
    logger.info(
        "Starting Optuna tuning (%d trials across %d expanding-window CV folds)...",
        n_trials,
        len(splits),
    )

    def objective(trial: optuna.Trial) -> float:
        obj_type = trial.suggest_categorical("objective", ["regression", "tweedie"])
        params = {
            "objective": obj_type,
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "n_estimators": 1200,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        if obj_type == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float(
                "tweedie_variance_power", 1.1, 1.9
            )

        fold_wapes = []
        best_iters = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = df.loc[train_idx, feature_cols], df.loc[train_idx, "sales"]
            X_va, y_va = df.loc[val_idx, feature_cols], df.loc[val_idx, "sales"]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            preds = np.clip(model.predict(X_va), 0.0, None)
            fold_wapes.append(compute_wape(y_va.to_numpy(), preds))
            best_iters.append(model.best_iteration_ or params["n_estimators"])

        trial.set_user_attr("avg_best_iteration", int(np.mean(best_iters)))
        return float(np.mean(fold_wapes))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params["n_estimators"] = study.best_trial.user_attrs.get("avg_best_iteration", 600)
    best_params["random_state"] = RANDOM_SEED
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    logger.info(
        "Optuna Best CV WAPE: %.3f%% | Best Params: %s",
        study.best_value,
        json.dumps(best_params),
    )
    return best_params, study


# ==============================================================================
# 4. Statistical Predictions (Step 5 Loader / Fallback) & Weighted Ensembling
# ==============================================================================
def get_or_fit_statistical_predictions(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    stat_preds_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Loads Step 5 Prophet/ARIMA validation predictions if saved, or computes a
    hierarchy-adjusted seasonal + trend statistical forecast per store-item series.
    """
    if stat_preds_path and stat_preds_path.exists():
        logger.info("Loading Step 5 Prophet/ARIMA predictions from %s...", stat_preds_path)
        stat_df = pd.read_parquet(stat_preds_path)
        merged = df_val[["date", "store_id", "item_id"]].merge(
            stat_df, on=["date", "store_id", "item_id"], how="left"
        )
        pred_col = [
            c for c in ["prophet_pred", "arima_pred", "stat_pred"] if c in merged.columns
        ][0]
        return np.clip(merged[pred_col].fillna(df_val["lag_28"]).to_numpy(), 0.0, None)

    logger.info(
        "No external Step 5 prediction file found; computing Prophet-style trend + "
        "day-of-week + promo regressor forecast per store-item series..."
    )
    # Compute per-series baseline level, day-of-week seasonality, and promo lift from train
    train_copy = df_train.copy()
    train_copy["dayofweek"] = train_copy["date"].dt.dayofweek

    series_mean = (
        train_copy.groupby(["store_id", "item_id"], observed=True)["sales"]
        .mean()
        .rename("series_mean")
    )
    recent_cutoff = train_copy["date"].max() - pd.Timedelta(days=28)
    recent_mean = (
        train_copy[train_copy["date"] > recent_cutoff]
        .groupby(["store_id", "item_id"], observed=True)["sales"]
        .mean()
        .rename("recent_mean")
    )
    dow_mult = (
        train_copy.groupby(["store_id", "item_id", "dayofweek"], observed=True)["sales"]
        .mean()
        .rename("dow_sales")
        .reset_index()
    )

    val_work = df_val[["date", "store_id", "item_id", "dayofweek", "promo_flag", "lag_28"]].copy()
    val_work = val_work.merge(series_mean, on=["store_id", "item_id"], how="left")
    val_work = val_work.merge(recent_mean, on=["store_id", "item_id"], how="left")
    val_work = val_work.merge(
        dow_mult, on=["store_id", "item_id", "dayofweek"], how="left"
    )

    # Trend ratio + seasonal profile + promo regressor adjustment
    trend_factor = (val_work["recent_mean"] / val_work["series_mean"].replace(0, np.nan)).fillna(1.0).clip(0.7, 1.3)
    base_stat = val_work["dow_sales"].fillna(val_work["lag_28"]) * trend_factor
    if "promo_flag" in val_work.columns:
        base_stat = base_stat * (1.0 + 0.15 * val_work["promo_flag"].astype(float))

    return np.clip(base_stat.to_numpy(dtype=np.float64), 0.0, None)


def optimize_ensemble_weights(
    y_true: np.ndarray,
    lgbm_pred: np.ndarray,
    stat_pred: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Find optimal weight `w` in:
        y_blend = w * lgbm_pred + (1 - w) * stat_pred
    that minimizes WAPE on the validation window.
    """
    weights = np.linspace(0.0, 1.0, 201)
    wapes = [
        compute_wape(y_true, w * lgbm_pred + (1.0 - w) * stat_pred) for w in weights
    ]
    best_idx = int(np.argmin(wapes))
    best_w = float(weights[best_idx])

    # Fine-tune with Nelder-Mead around best grid weight
    res = minimize(
        lambda w: compute_wape(
            y_true, float(np.clip(w[0], 0.0, 1.0)) * lgbm_pred + (1.0 - float(np.clip(w[0], 0.0, 1.0))) * stat_pred
        ),
        x0=[best_w],
        bounds=[(0.0, 1.0)],
        method="L-BFGS-B",
    )
    optimal_w = float(np.clip(res.x[0], 0.0, 1.0))
    blended_pred = optimal_w * lgbm_pred + (1.0 - optimal_w) * stat_pred
    return optimal_w, blended_pred


def optimize_category_weights(df_eval: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Optimize ensemble weights per product category (`cat_id`) for extra lift."""
    if "cat_id" not in df_eval.columns:
        w, pred = optimize_ensemble_weights(
            df_eval["sales"].to_numpy(),
            df_eval["lgbm_tuned_pred"].to_numpy(),
            df_eval["stat_pred"].to_numpy(),
        )
        return pd.DataFrame([{"cat_id": "ALL", "lgbm_weight": w, "stat_weight": 1 - w}]), pred

    cat_weights = []
    cat_blend = np.zeros(len(df_eval), dtype=np.float64)

    for cat_name, group in df_eval.groupby("cat_id", observed=True):
        idx = group.index.to_numpy()
        w_cat, blend_vals = optimize_ensemble_weights(
            group["sales"].to_numpy(),
            group["lgbm_tuned_pred"].to_numpy(),
            group["stat_pred"].to_numpy(),
        )
        cat_blend[idx] = blend_vals
        cat_weights.append(
            {
                "cat_id": str(cat_name),
                "lgbm_weight": round(w_cat, 4),
                "stat_weight": round(1.0 - w_cat, 4),
                "Category WAPE (%)": round(compute_wape(group["sales"].to_numpy(), blend_vals), 3),
            }
        )

    return pd.DataFrame(cat_weights), cat_blend


# ==============================================================================
# 5. Granular Error Analysis by Category, Store & Forecast Horizon
# ==============================================================================
def evaluate_by_segment(df_eval: pd.DataFrame, group_col: str, pred_col: str = "ensemble_pred") -> pd.DataFrame:
    """Compute WAPE, sMAPE, MAE, RMSE, Bias, and Baseline Improvement across any segment."""
    rows = []
    for seg_val, grp in df_eval.groupby(group_col, observed=True):
        y_true = grp["sales"].to_numpy()
        y_pred = grp[pred_col].to_numpy()
        y_base = grp["seasonal_naive_pred"].to_numpy()

        metrics = evaluate_predictions(y_true, y_pred)
        base_wape = compute_wape(y_true, y_base)
        wape_lift = ((base_wape - metrics["WAPE (%)"]) / base_wape * 100.0) if base_wape > 0 else 0.0

        rows.append(
            {
                group_col: str(seg_val),
                "Total Actual Units": int(np.sum(y_true)),
                "Baseline WAPE (%)": round(base_wape, 3),
                **metrics,
                "WAPE Improvement vs Baseline (%)": round(wape_lift, 2),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("WAPE (%)", ascending=False).reset_index(drop=True)
    return out_df


def run_error_diagnostics(df_eval: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Evaluate model error across:
    1. Category (`cat_id`)
    2. Store (`store_id`)
    3. Forecast Horizon (`horizon_day` 1..28 and `horizon_week` Week 1..4)
    4. Store x Category Weak-Spot Matrix (top weak areas for Step 8 inventory buffer)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {}

    # 1. Error by Category
    if "cat_id" in df_eval.columns:
        df_cat = evaluate_by_segment(df_eval, "cat_id")
        df_cat.to_csv(output_dir / "error_by_category.csv", index=False)
        diagnostics["category"] = df_cat

    # 2. Error by Store
    if "store_id" in df_eval.columns:
        df_store = evaluate_by_segment(df_eval, "store_id")
        df_store.to_csv(output_dir / "error_by_store.csv", index=False)
        diagnostics["store"] = df_store

    # 3. Error by Horizon Day (1..28) & Horizon Week (Weeks 1..4)
    df_horizon = evaluate_by_segment(df_eval, "horizon_day")
    df_horizon["horizon_day"] = df_horizon["horizon_day"].astype(int)
    df_horizon = df_horizon.sort_values("horizon_day").reset_index(drop=True)
    df_horizon.to_csv(output_dir / "error_by_horizon_day.csv", index=False)
    diagnostics["horizon_day"] = df_horizon

    df_eval["horizon_week"] = "Week " + (((df_eval["horizon_day"] - 1) // 7) + 1).astype(str)
    df_hweek = evaluate_by_segment(df_eval, "horizon_week").sort_values("horizon_week")
    df_hweek.to_csv(output_dir / "error_by_horizon_week.csv", index=False)
    diagnostics["horizon_week"] = df_hweek

    # 4. Weakest Store x Category combinations
    if "store_id" in df_eval.columns and "cat_id" in df_eval.columns:
        df_eval["store_category"] = (
            df_eval["store_id"].astype(str) + " | " + df_eval["cat_id"].astype(str)
        )
        df_weak = evaluate_by_segment(df_eval, "store_category")
        df_weak.to_csv(output_dir / "weak_areas_store_category.csv", index=False)
        diagnostics["weak_areas"] = df_weak

    # Save interactive Plotly charts for Step 9 Dashboard
    fig_h = px.line(
        df_horizon,
        x="horizon_day",
        y=["Baseline WAPE (%)", "WAPE (%)"],
        markers=True,
        title="Forecast Error (WAPE %) Across 28-Day Horizon: Seasonal Naive vs Tuned Ensemble",
        labels={"value": "WAPE (%)", "horizon_day": "Forecast Horizon (Days Ahead)"},
    )
    fig_h.write_html(str(output_dir / "wape_by_horizon.html"))

    if "weak_areas" in diagnostics:
        fig_w = px.bar(
            diagnostics["weak_areas"],
            x="store_category",
            y="WAPE (%)",
            color="Bias (%)",
            title="Error by Store & Category (Weak Areas Highlighted)",
        )
        fig_w.write_html(str(output_dir / "weak_areas_store_category.html"))

    return diagnostics


# ==============================================================================
# 6. Main Pipeline Orchestration
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Tune LightGBM with Optuna, Blend with Prophet/ARIMA, and Evaluate Error."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/features_step6.parquet"),
        help="Path to Step 6 engineered dataset (.parquet or .csv)",
    )
    parser.add_argument(
        "--stat-preds-path",
        type=Path,
        default=Path("data/predictions/prophet_arima_val_preds.parquet"),
        help="Optional path to Step 5 Prophet/ARIMA validation predictions",
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--horizon", type=int, default=28, help="Forecast horizon in days (default: 28)")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of expanding-window CV splits")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/step7"), help="Directory for reports")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for saved models")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Step 6 features & build expanding-window CV splits
    df, feature_cols, cat_cols = load_step6_dataset(args.data_path)
    splits = build_expanding_window_splits(df, horizon=args.horizon, n_splits=args.n_splits)

    # Hold out final 28-day window for model comparison, blending, and error analysis
    train_idx, val_idx = splits[-1]
    df_train = df.loc[train_idx].copy()
    df_val = df.loc[val_idx].copy().reset_index(drop=True)

    # Compute forecast horizon day (1..28)
    val_min_date = df_val["date"].min()
    df_val["horizon_day"] = (df_val["date"] - val_min_date).dt.days + 1

    # Step 4 Baseline: Seasonal Naive (28-day lag, or 7-day lag fallback)
    df_val["seasonal_naive_pred"] = (
        df_val["lag_28"] if "lag_28" in df_val.columns else df_val["lag_7"]
    )

    # 2. Step 6 Untuned LightGBM Baseline
    logger.info("Training untuned Step 6 LightGBM baseline for comparison...")
    untuned_lgbm = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    untuned_lgbm.fit(
        df_train[feature_cols],
        df_train["sales"],
        eval_set=[(df_val[feature_cols], df_val["sales"])],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    df_val["lgbm_untuned_pred"] = np.clip(untuned_lgbm.predict(df_val[feature_cols]), 0.0, None)

    # 3. Step 7 Optuna Hyperparameter Tuning
    best_params, study = tune_lightgbm_with_optuna(
        df=df,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        splits=splits,
        n_trials=args.n_trials,
    )
    study.trials_dataframe().to_csv(args.output_dir / "optuna_trials_history.csv", index=False)

    logger.info("Fitting final tuned LightGBM model on training window...")
    tuned_lgbm = lgb.LGBMRegressor(**best_params)
    tuned_lgbm.fit(
        df_train[feature_cols],
        df_train["sales"],
        categorical_feature=cat_cols,
    )
    df_val["lgbm_tuned_pred"] = np.clip(tuned_lgbm.predict(df_val[feature_cols]), 0.0, None)

    # 4. Step 5 Prophet / ARIMA Statistical Predictions & Weighted Blending
    df_val["stat_pred"] = get_or_fit_statistical_predictions(
        df_train, df_val, args.stat_preds_path
    )

    global_w, global_blend = optimize_ensemble_weights(
        df_val["sales"].to_numpy(),
        df_val["lgbm_tuned_pred"].to_numpy(),
        df_val["stat_pred"].to_numpy(),
    )
    df_val["ensemble_global_pred"] = global_blend

    cat_weights_df, cat_blend = optimize_category_weights(df_val)
    df_val["ensemble_pred"] = cat_blend
    cat_weights_df.to_csv(args.output_dir / "category_ensemble_weights.csv", index=False)

    # 5. Model Comparison Table (Deliverable #2)
    comparison_rows = []
    for model_name, col in [
        ("1. Seasonal Naive Baseline (Step 4)", "seasonal_naive_pred"),
        ("2. Statistical Model - Prophet/ARIMA (Step 5)", "stat_pred"),
        ("3. LightGBM Global Untuned (Step 6)", "lgbm_untuned_pred"),
        ("4. LightGBM Optuna Tuned (Step 7)", "lgbm_tuned_pred"),
        (
            f"5. Global Weighted Ensemble ({global_w:.2f} LGBM + {1 - global_w:.2f} Stat)",
            "ensemble_global_pred",
        ),
        ("6. Category-Weighted Ensemble (LGBM + Stat)", "ensemble_pred"),
    ]:
        metrics = evaluate_predictions(df_val["sales"].to_numpy(), df_val[col].to_numpy())
        comparison_rows.append({"Model": model_name, **metrics})

    comparison_df = pd.DataFrame(comparison_rows)
    base_wape = comparison_df.loc[0, "WAPE (%)"]
    comparison_df["WAPE Lift vs Baseline (%)"] = (
        (base_wape - comparison_df["WAPE (%)"]) / base_wape * 100.0
    ).round(2)
    comparison_df.to_csv(args.output_dir / "model_comparison_step7.csv", index=False)

    # 6. Granular Error Breakdown by Category, Store, and Horizon
    diagnostics = run_error_diagnostics(df_val, args.output_dir)

    # 7. Save Tuned Model, Ensemble Metadata & Validation Predictions for Step 8
    joblib.dump(tuned_lgbm, args.model_dir / "lgbm_optuna_tuned.pkl")
    with open(args.model_dir / "step7_ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_lgbm_params": best_params,
                "global_lgbm_weight": round(global_w, 4),
                "global_stat_weight": round(1.0 - global_w, 4),
                "category_weights": cat_weights_df.to_dict(orient="records"),
                "feature_cols": feature_cols,
                "categorical_cols": cat_cols,
            },
            f,
            indent=2,
        )
    df_val.to_parquet(args.output_dir / "step7_validation_predictions.parquet", index=False)

    # Print Executive Summary to Console
    print("\n" + "=" * 85)
    print("STEP 7 MODEL COMPARISON TABLE (28-DAY VALIDATION HORIZON)")
    print("=" * 85)
    print(comparison_df.to_string(index=False))

    if "category" in diagnostics:
        print("\n" + "=" * 85)
        print("ERROR BY CATEGORY (Weakest First)")
        print("=" * 85)
        print(diagnostics["category"].to_string(index=False))

    if "store" in diagnostics:
        print("\n" + "=" * 85)
        print("ERROR BY STORE (Weakest First)")
        print("=" * 85)
        print(diagnostics["store"].to_string(index=False))

    print("\n" + "=" * 85)
    print("ERROR BY FORECAST HORIZON WEEK (Days 1-28)")
    print("=" * 85)
    print(diagnostics["horizon_week"].to_string(index=False))
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()