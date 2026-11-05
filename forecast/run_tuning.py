"""
Step 7 CLI Pipeline: `python -m forecast.run_tuning`

Runs Optuna hyperparameter tuning + bias calibration on the global LightGBM model,
blends predictions with the 2-year month-weekday profile (and Prophet/statistical
predictions if present), evaluates error by store, item, and horizon (days 1-28,
29-60, 61-90), and writes reports and figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast.tuning import (
    VALIDATION_WINDOWS,
    build_gbm_dataset_for_window,
    compute_all_metrics,
    compute_error_breakdown,
    estimate_bias_calibration_factor,
    find_optimal_blend_weight,
    fit_and_predict_gbm,
    tune_hyperparameters_optuna,
    wape,
)

logger = logging.getLogger(__name__)


def load_prepared_sales(prepared_path: Path, raw_dir: Path) -> pd.DataFrame:
    """Load `data/processed/sales_prepared.parquet` or `data/raw/train.csv`."""
    if prepared_path.exists():
        logger.info("Loading prepared data from %s", prepared_path)
        df = pd.read_parquet(prepared_path)
    elif (raw_dir / "train.csv").exists():
        logger.info("Loading raw train.csv from %s", raw_dir / "train.csv")
        df = pd.read_csv(raw_dir / "train.csv", parse_dates=["date"])
    else:
        raise FileNotFoundError(
            f"Could not find {prepared_path} or {raw_dir / 'train.csv'}. "
            "Run `python -m forecast.prepare` first."
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df[["date", "store", "item", "sales"]].copy()


def save_step7_figures(
    summary_df: pd.DataFrame,
    store_err: pd.DataFrame,
    item_err: pd.DataFrame,
    horizon_err: pd.DataFrame,
    figures_dir: Path,
    docs_images_dir: Path,
) -> None:
    """Generate and save `tuning_ensemble_wape.png` and `error_analysis_breakdown.png`."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    docs_images_dir.mkdir(parents=True, exist_ok=True)

    # 1. WAPE by validation window across models
    win_cols = ["2017-03-31", "2017-06-30", "2017-09-30", "2017-12-31"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    x = np.arange(len(win_cols))
    width = 0.18

    models_to_plot = [
        ("two-year month-weekday profile", "#7f8c8d"),
        ("global LightGBM (Step 6, untuned)", "#3498db"),
        ("LightGBM tuned (Optuna + calibration)", "#e67e22"),
        ("Weighted ensemble (LGBM + profile)", "#2ecc71"),
    ]
    for idx, (m_name, color) in enumerate(models_to_plot):
        row = summary_df[summary_df["model"] == m_name]
        if not row.empty:
            vals = [float(row[c].iloc[0]) for c in win_cols]
            ax.bar(x + (idx - 1.5) * width, vals, width, label=m_name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(win_cols)
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Step 7: WAPE (%) Across Validation Windows (500 Series)")
    ax.set_ylim(8.0, 14.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=8.5)
    fig.tight_layout()

    for out_dir in (figures_dir, docs_images_dir):
        fig.savefig(out_dir / "tuning_ensemble_wape.png")
    plt.close(fig)

    # 2. Error breakdown by Store, Item volume, and Horizon bucket
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=140)

    # Panel A: Store WAPE
    s_sorted = store_err.sort_values("wape", ascending=True)
    axes[0].barh(
        ["Store " + str(s) for s in s_sorted["store"]],
        s_sorted["wape"],
        color="#2980b9",
    )
    axes[0].set_xlabel("WAPE (%)")
    axes[0].set_title("Error by Store")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    # Panel B: Item WAPE vs Mean Daily Sales
    axes[1].scatter(
        item_err["mean_daily_sales"],
        item_err["wape"],
        color="#c0392b",
        alpha=0.8,
        edgecolors="k",
    )
    axes[1].set_xlabel("Mean Daily Sales (Units/Day)")
    axes[1].set_ylabel("WAPE (%)")
    axes[1].set_title("Item WAPE vs Sales Volume (50 Items)")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel C: Horizon buckets (1-28, 29-60, 61-90)
    h_order = ["Days 1-28", "Days 29-60", "Days 61-90"]
    h_plot = horizon_err.set_index("horizon_bucket").reindex(h_order).reset_index()
    axes[2].bar(h_plot["horizon_bucket"], h_plot["wape"], color="#27ae60", width=0.5)
    axes[2].set_ylabel("WAPE (%)")
    axes[2].set_title("Error by Forecast Horizon")
    axes[2].set_ylim(0, max(float(h_plot["wape"].max()) * 1.25, 12.0))
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    for out_dir in (figures_dir, docs_images_dir):
        fig.savefig(out_dir / "error_analysis_breakdown.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Optuna tuning, weighted ensembling and error analysis by store/item/horizon."
    )
    parser.add_argument(
        "--prepared-path",
        type=Path,
        default=Path("data/processed/sales_prepared.parquet"),
        help="Path to prepared sales parquet file",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Fallback directory containing raw train.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write CSV/JSON reports and figures",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "hist_gb"],
        help="Gradient boosting engine (default: lightgbm)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna trials (default: 15)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (INFO, DEBUG, WARNING)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t0 = time.perf_counter()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_prepared_sales(args.prepared_path, args.raw_dir)

    # 1. Build leakage-safe datasets for each of the 4 validation windows
    logger.info("Building leakage-safe feature datasets for 4 validation windows...")
    window_datasets = []
    for win_name, start_str, end_str in VALIDATION_WINDOWS:
        val_start = pd.Timestamp(start_str)
        val_end = pd.Timestamp(end_str)
        origin_date = val_start - pd.Timedelta(days=1)
        tr_df, va_df, fcols = build_gbm_dataset_for_window(df, origin_date, val_start, val_end)
        va_df["window"] = win_name
        window_datasets.append((win_name, tr_df, va_df, fcols))

    # 2. Tune LightGBM hyperparameters with Optuna
    logger.info("Tuning global LightGBM hyperparameters (n_trials=%d)...", args.n_trials)
    tuned_params = tune_hyperparameters_optuna(
        window_datasets, n_trials=args.n_trials, engine=args.engine
    )

    untuned_params = {
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

    # 3. Fit untuned Step 6 model, tuned + bias-calibrated Step 7 model, and blend per window
    all_val_dfs = []
    blend_weights = []

    for idx, (win_name, tr_df, va_df, fcols) in enumerate(window_datasets):
        logger.info("Scoring validation window %s...", win_name)
        va_df = va_df.copy()

        # Untuned Step 6 LightGBM
        va_df["pred_gbm_untuned"] = fit_and_predict_gbm(
            tr_df, va_df, fcols, untuned_params, engine=args.engine
        )

        # Tuned Step 7 LightGBM + trailing bias calibration
        raw_tuned = fit_and_predict_gbm(
            tr_df, va_df, fcols, tuned_params, engine=args.engine
        )
        cal_factor = estimate_bias_calibration_factor(
            tr_df, fcols, tuned_params, engine=args.engine
        )
        va_df["pred_gbm_tuned"] = raw_tuned * cal_factor

        # Estimate blend weight on preceding validation windows (or 0.80 default on first window)
        if idx == 0:
            w_lgbm = 0.80
        else:
            prev_concat = pd.concat(all_val_dfs, ignore_index=True)
            w_lgbm = find_optimal_blend_weight(
                prev_concat["sales"].to_numpy(),
                prev_concat["pred_gbm_tuned"].to_numpy(),
                prev_concat["pred_profile_2y"].to_numpy(),
            )
        blend_weights.append(w_lgbm)
        va_df["pred_ensemble"] = (
            w_lgbm * va_df["pred_gbm_tuned"] + (1.0 - w_lgbm) * va_df["pred_profile_2y"]
        )
        all_val_dfs.append(va_df)

    full_val = pd.concat(all_val_dfs, ignore_index=True)
    avg_w_lgbm = float(np.round(np.mean(blend_weights), 2))

    # 4. Build Window Summary Table matching README.md
    model_cols = [
        ("364-day benchmark", "pred_benchmark_364"),
        ("two-year month-weekday profile", "pred_profile_2y"),
        ("global LightGBM (Step 6, untuned)", "pred_gbm_untuned"),
        ("LightGBM tuned (Optuna + calibration)", "pred_gbm_tuned"),
        ("Weighted ensemble (LGBM + profile)", "pred_ensemble"),
    ]

    summary_rows: List[Dict] = []
    for label, col in model_cols:
        row_dict: Dict = {"model": label}
        win_wapes = []
        win_biases = []
        win_smapes = []
        for win_name, _, _ in VALIDATION_WINDOWS:
            sub = full_val[full_val["window"] == win_name]
            m = compute_all_metrics(sub["sales"].to_numpy(), sub[col].to_numpy())
            row_dict[win_name] = round(m["wape"], 2)
            win_wapes.append(m["wape"])
            win_biases.append(m["bias_pct"])
            win_smapes.append(m["smape"])
        row_dict["mean_wape"] = round(float(np.mean(win_wapes)), 2)
        row_dict["mean_smape"] = round(float(np.mean(win_smapes)), 2)
        row_dict["mean_bias_pct"] = round(float(np.mean(win_biases)), 2)
        summary_rows.append(row_dict)

    summary_df = pd.DataFrame(summary_rows)
    bench_mean = float(summary_df.loc[0, "mean_wape"])
    summary_df["vs_benchmark_pct"] = (
        (bench_mean - summary_df["mean_wape"]) / bench_mean * 100.0
    ).round(1)

    # 5. Error Analysis by Store, Item, Horizon Bucket, and Month
    full_val["horizon_bucket"] = pd.cut(
        full_val["horizon"],
        bins=[0, 28, 60, 95],
        labels=["Days 1-28", "Days 29-60", "Days 61-90"],
    ).astype(str)

    store_err = compute_error_breakdown(full_val, "store", "pred_ensemble")
    item_err = compute_error_breakdown(full_val, "item", "pred_ensemble")
    horizon_err = compute_error_breakdown(full_val, "horizon_bucket", "pred_ensemble")
    month_err = compute_error_breakdown(full_val, "month", "pred_ensemble")

    # 6. Write Reports & Figures
    summary_df.to_csv(args.reports_dir / "tuning_ensemble_results.csv", index=False)
    store_err.to_csv(args.reports_dir / "error_by_store.csv", index=False)
    item_err.to_csv(args.reports_dir / "error_by_item.csv", index=False)
    horizon_err.to_csv(args.reports_dir / "error_by_horizon.csv", index=False)
    month_err.to_csv(args.reports_dir / "error_by_month.csv", index=False)
    full_val.to_parquet(args.reports_dir / "step7_predictions.parquet", index=False)

    with open(args.reports_dir / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": tuned_params,
                "average_lgbm_blend_weight": avg_w_lgbm,
                "summary_table": summary_rows,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            },
            f,
            indent=2,
        )

    save_step7_figures(
        summary_df=summary_df,
        store_err=store_err,
        item_err=item_err,
        horizon_err=horizon_err,
        figures_dir=args.reports_dir / "figures",
        docs_images_dir=Path("docs/images"),
    )

    print("\n" + "=" * 88)
    print("STEP 7: TUNING & ENSEMBLING RESULTS (ALL 500 SERIES, 4 VALIDATION WINDOWS, WAPE %)")
    print("=" * 88)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 88)
    print("ERROR BY STORE (WEAKEST FIRST)")
    print("=" * 88)
    print(store_err.to_string(index=False))

    print("\n" + "=" * 88)
    print("ERROR BY ITEM — TOP 5 WEAKEST & TOP 5 STRONGEST ITEMS")
    print("=" * 88)
    print(pd.concat([item_err.head(5), item_err.tail(5)]).to_string(index=False))

    print("\n" + "=" * 88)
    print("ERROR BY FORECAST HORIZON BUCKET (DAYS 1-28, 29-60, 61-90)")
    print("=" * 88)
    print(horizon_err.to_string(index=False))
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()