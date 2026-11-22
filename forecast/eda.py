"""Step 3: time series EDA for the Store Item Demand data.

Run:  python -m forecast.eda
Reads the prepared table from Step 2 and writes figures to reports/figures/ plus a JSON
summary to reports/eda_summary.json. STL/MSTL decomposition and the ADF/KPSS tests need
statsmodels; pass ``--skip-statsmodels`` to run everything else without it.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend so the pipeline runs without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from forecast.config import ForecastConfig  # noqa: E402
from forecast.data import load_prepared  # noqa: E402
from forecast.holidays import holiday_calendar  # noqa: E402
from forecast.logging_utils import configure_logging  # noqa: E402
from forecast.timeseries import (  # noqa: E402
    adf_kpss,
    autocorrelation,
    decomposition_strengths,
    mstl_components,
    partial_autocorrelation,
    stationarity_table,
)

logger = logging.getLogger(__name__)


def labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with known sales (everything except the test horizon)."""
    return df.loc[~df["is_test"]]


def daily_total(df: pd.DataFrame, cfg: ForecastConfig) -> pd.Series:
    """Total units sold per day across all stores and items, on a regular daily index."""
    total = labelled_rows(df).groupby(cfg.date_col)[cfg.target_col].sum().astype(float)
    return total.asfreq("D")


def annual_summary(total: pd.Series) -> pd.DataFrame:
    """Yearly total sales and year-over-year growth in percent."""
    yearly = total.groupby(total.index.year).sum()
    return pd.DataFrame({"year": yearly.index, "total_sales": yearly.to_numpy(),
                         "growth_pct": yearly.pct_change().to_numpy() * 100})


def seasonal_index(labelled: pd.DataFrame, level_col: str, cfg: ForecastConfig) -> pd.DataFrame:
    """Average sales per level (day of week, month) relative to each year's average.

    Dividing by the same year's mean removes growth, so years are comparable. An index of
    1.20 means 20% above that year's average.
    """
    per_level = labelled.groupby(["year", level_col])[cfg.target_col].mean().unstack("year")
    annual_mean = labelled.groupby("year")[cfg.target_col].mean()
    index = per_level.div(annual_mean, axis=1)
    index.columns = [str(c) for c in index.columns]
    index["average"] = index.mean(axis=1)
    return index


def yearly_shape(total: pd.Series) -> pd.DataFrame:
    """Smoothed day-of-year profile (7-day mean / annual mean) per year, to check stability."""
    smooth = total.rolling(7, center=True).mean()
    annual_mean = total.groupby(total.index.year).transform("mean")
    relative = smooth / annual_mean
    shape = relative.groupby([relative.index.year, relative.index.day_of_year]).mean().unstack(0)
    return shape.iloc[:365]


def same_month_weekday_ratio(total: pd.Series) -> pd.Series:
    """Sales on each day divided by the mean of the same weekday in the same month.

    This data has a flat level inside each month and a sharp step at every month boundary, so
    comparing a day with the same weekday in the *same month* removes the weekday pattern, the
    season and the trend. (Comparing with weeks either side would mix in the neighbouring month
    and invent holiday effects, for example around Thanksgiving.) The day itself is excluded
    from its own reference.
    """
    grouped = total.groupby([total.index.year, total.index.month, total.index.day_of_week])
    others = (grouped.transform("sum") - total) / (grouped.transform("count") - 1)
    return total / others


def holiday_effects(total: pd.Series, cfg: ForecastConfig) -> Tuple[pd.DataFrame, float]:
    """Average holiday effect with a noise yardstick taken from ordinary days.

    Returns:
        A table per holiday (mean ratio, effect %, z-score, material flag) and the standard
        deviation of the same ratio on all days, which measures ordinary day-to-day noise.
    """
    ratio = same_month_weekday_ratio(total)
    calendar = holiday_calendar(total.index.min(), total.index.max())
    occurrences = calendar[calendar["date"].isin(total.index)].copy()
    occurrences["ratio"] = ratio.reindex(pd.DatetimeIndex(occurrences["date"])).to_numpy()
    occurrences = occurrences.dropna(subset=["ratio"])
    noise_std = max(float(ratio.std()), 1e-12)  # guard: a perfectly flat series has zero noise

    table = occurrences.groupby("holiday")["ratio"].agg(["mean", "min", "max", "count"])
    table["effect_pct"] = (table["mean"] - 1.0) * 100
    table["z"] = (table["mean"] - 1.0) / (noise_std / np.sqrt(table["count"]))
    table["material"] = ((table["effect_pct"].abs() >= cfg.eda.holiday_material_pct)
                         & (table["z"].abs() >= cfg.eda.holiday_material_z))
    return table.sort_values("effect_pct").reset_index(), noise_std


def intermittency_table(wide: pd.DataFrame, adi_threshold: float, cv2_threshold: float) -> pd.DataFrame:
    """Syntetos-Boylan demand classes per series (rows = series, columns = days).

    ADI is the average gap between non-zero days; CV2 is the squared coefficient of variation
    of the non-zero sizes. Smooth/erratic series have ADI below the threshold.
    """
    values = wide.to_numpy(dtype=float)
    nonzero = values > 0
    n_nonzero = nonzero.sum(axis=1)
    safe = np.where(n_nonzero > 0, n_nonzero, 1)
    mean_nz = (values * nonzero).sum(axis=1) / safe
    var_nz = ((values ** 2) * nonzero).sum(axis=1) / safe - mean_nz ** 2
    adi = np.where(n_nonzero > 0, values.shape[1] / safe, np.inf)
    cv2 = np.divide(var_nz, mean_nz ** 2, out=np.zeros(len(values)), where=mean_nz > 0)
    demand_class = np.select(
        [n_nonzero == 0, (adi < adi_threshold) & (cv2 < cv2_threshold),
         (adi < adi_threshold), (cv2 < cv2_threshold)],
        ["no demand", "smooth", "erratic", "intermittent"], default="lumpy")
    return pd.DataFrame({"adi": adi, "cv2": cv2, "zero_share": 1.0 - nonzero.mean(axis=1),
                         "demand_class": demand_class}, index=wide.index)


def profile_spread(labelled: pd.DataFrame, level_col: str, cfg: ForecastConfig) -> Dict[str, float]:
    """How much the seasonal shape differs between series.

    Each series gets a profile (average sales per level divided by its own mean). The spread
    is the standard deviation of those profiles across series; the correlation compares every
    series profile with the average profile. Low spread and correlation near 1 mean all series
    share one seasonal shape, which a single global model can learn.
    """
    keys = list(cfg.granularity)
    matrix = labelled.groupby([*keys, level_col])[cfg.target_col].mean().unstack(level_col)
    profiles = matrix.div(matrix.mean(axis=1), axis=0)
    correlation = profiles.corrwith(profiles.mean(axis=0), axis=1)
    return {"max_std_across_series": float(profiles.std(axis=0).max()),
            "mean_std_across_series": float(profiles.std(axis=0).mean()),
            "min_correlation_with_average": float(correlation.min()),
            "median_correlation_with_average": float(correlation.median())}


def growth_distribution(labelled: pd.DataFrame, cfg: ForecastConfig) -> Dict[str, float]:
    """Last-year growth per series (latest full year versus the year before), in percent."""
    keys = list(cfg.granularity)
    yearly = labelled.groupby([*keys, "year"])[cfg.target_col].sum().unstack("year")
    last, previous = yearly.columns[-1], yearly.columns[-2]
    growth = (yearly[last] / yearly[previous] - 1.0) * 100
    return {"year": int(last), "median_pct": float(growth.median()),
            "p5_pct": float(growth.quantile(0.05)), "p95_pct": float(growth.quantile(0.95))}


# ---------------------------------------------------------------------------- plots
def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Tighten layout, save and close a figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_trend(total: pd.Series, annual: pd.DataFrame, path: Path, dpi: int) -> None:
    """Daily total with its 365-day average, next to yearly totals and growth."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    left.plot(total.index, total, color="tab:blue", alpha=0.35, linewidth=0.7, label="daily total")
    left.plot(total.index, total.rolling(365, center=True).mean(), color="tab:red", linewidth=2,
              label="365-day average")
    left.set_title("Total daily sales (all stores and items)")
    left.set_ylabel("units")
    left.legend()
    bars = right.bar(range(len(annual)), annual["total_sales"] / 1e6, color="tab:blue")
    right.set_xticks(range(len(annual)), [str(y) for y in annual["year"]])  # explicit labels avoid categorical axis
    for bar, growth in zip(bars, annual["growth_pct"]):
        label = "" if np.isnan(growth) else f"{growth:+.1f}%"
        right.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")
    right.set_title("Yearly sales (millions) and growth")
    _save(fig, path, dpi)


def plot_seasonality(dow: pd.DataFrame, month: pd.DataFrame, shape: pd.DataFrame, path: Path, dpi: int) -> None:
    """Weekly, monthly and day-of-year profiles, one thin line per year."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for ax, table, title in ((axes[0], dow, "Day-of-week index"), (axes[1], month, "Month index")):
        for column in table.columns[:-1]:
            ax.plot(table.index, table[column], marker="o", markersize=3, linewidth=1, label=column)
        ax.plot(table.index, table["average"], color="black", linewidth=2.5, label="average")
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(title + " (1.0 = that year's average)")
        ax.legend(fontsize=7)
    axes[0].set_xticks(range(7), labels)
    axes[1].set_xticks(range(1, 13))
    for column in shape.columns:
        axes[2].plot(shape.index, shape[column], linewidth=1, label=str(column))
    axes[2].set_title("Day-of-year shape (7-day average / annual average)")
    axes[2].set_xlabel("day of year")
    axes[2].legend(fontsize=7)
    _save(fig, path, dpi)


def plot_decomposition(components: pd.DataFrame, path: Path, dpi: int) -> None:
    """MSTL components stacked: observed, trend, each seasonal term and residual."""
    fig, axes = plt.subplots(len(components.columns), 1, figsize=(13, 2.2 * len(components.columns)), sharex=True)
    for ax, column in zip(axes, components.columns):
        ax.plot(components.index, components[column], linewidth=0.8)
        ax.set_ylabel(column)
    axes[0].set_title("MSTL decomposition of total daily sales")
    _save(fig, path, dpi)


def plot_acf_pacf(results: Dict[str, Dict[str, np.ndarray]], n_obs: int, short: int, path: Path, dpi: int) -> None:
    """ACF (short and long lags) and PACF for each analysed series."""
    fig, axes = plt.subplots(len(results), 3, figsize=(16, 3.6 * len(results)), squeeze=False)
    bound = 1.96 / np.sqrt(n_obs)
    for row, (name, values) in enumerate(results.items()):
        panels = (("ACF, short lags", values["acf"][: short + 1]), ("ACF, long lags", values["acf"]),
                  ("PACF", values["pacf"]))
        for ax, (title, data) in zip(axes[row], panels):
            ax.vlines(np.arange(len(data)), 0, data, linewidth=1)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axhline(bound, color="tab:red", linestyle="--", linewidth=0.8)
            ax.axhline(-bound, color="tab:red", linestyle="--", linewidth=0.8)
            ax.set_title(f"{name}: {title}")
            ax.set_xlabel("lag (days)")
    _save(fig, path, dpi)


def plot_holidays(table: pd.DataFrame, path: Path, dpi: int) -> None:
    """Average holiday effect with min-max whiskers; material holidays in orange."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["tab:orange" if m else "tab:blue" for m in table["material"]]
    ax.barh(table["holiday"], table["effect_pct"], color=colors)
    ax.errorbar(table["effect_pct"], table["holiday"],
                xerr=[table["effect_pct"] - (table["min"] - 1) * 100, (table["max"] - 1) * 100 - table["effect_pct"]],
                fmt="none", ecolor="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("effect vs the same weekday in the same month (%)")
    ax.set_title("Holiday effects on total daily sales (orange = material, none found)")
    _save(fig, path, dpi)


def plot_store_item(matrix: pd.DataFrame, cv: pd.Series, path: Path, dpi: int) -> None:
    """Heatmap of average daily sales per store and item, plus the CV histogram."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    image = left.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    left.set_yticks(range(len(matrix.index)), matrix.index)
    left.set_xlabel("item")
    left.set_ylabel("store")
    left.set_title("Average daily sales per store and item")
    fig.colorbar(image, ax=left)
    right.hist(cv, bins=25, color="tab:blue")
    right.set_title("Coefficient of variation per series")
    right.set_xlabel("std / mean of daily sales")
    _save(fig, path, dpi)


def plot_intermittency(table: pd.DataFrame, adi_threshold: float, cv2_threshold: float, path: Path, dpi: int) -> None:
    """ADI versus CV2 scatter with the Syntetos-Boylan cut-offs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(table["adi"], table["cv2"], s=14, alpha=0.7)
    ax.axvline(adi_threshold, color="tab:red", linestyle="--")
    ax.axhline(cv2_threshold, color="tab:red", linestyle="--")
    ax.set_xlim(0.9, adi_threshold * 1.15)
    ax.set_ylim(0, cv2_threshold * 1.3)
    ax.set_xlabel("ADI (average days between sales)")
    ax.set_ylabel("CV2 of sale sizes")
    ax.set_title("Demand classes: every series sits in the 'smooth' corner")
    _save(fig, path, dpi)


# ------------------------------------------------------------------------- orchestration
def _records(df: pd.DataFrame, digits: int = 4) -> List[Dict[str, Any]]:
    """Rounded, JSON-friendly rows."""
    return json.loads(df.round(digits).to_json(orient="records", date_format="iso"))


def _stationarity_sections(total: pd.Series, wide: pd.DataFrame, cfg: ForecastConfig) -> Dict[str, Any]:
    """ADF/KPSS on the total series and on a random sample of individual series."""
    alpha = cfg.eda.significance_level
    rng = np.random.default_rng(cfg.random_state)
    n_sample = min(cfg.eda.stationarity_sample_series, len(wide))
    picked = rng.choice(len(wide), size=n_sample, replace=False)
    shares: Dict[str, float] = {}
    for name, lag in (("level", 0), ("diff(7)", 7)):
        verdicts = []
        for position in picked:
            series = wide.iloc[position]
            values = (series.diff(lag).dropna() if lag else series).to_numpy(dtype=float)
            verdicts.append(adf_kpss(values, alpha)["verdict"])
        shares[name] = float(np.mean([v == "stationary" for v in verdicts]))
    return {"total_series": _records(stationarity_table(total, alpha)),
            "sampled_series": n_sample, "share_stationary_in_sample": shares}


def run_eda(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None, use_statsmodels: bool = True) -> Dict[str, Any]:
    """Run the whole EDA, save figures and the JSON summary, and return the summary.

    Args:
        cfg: Project configuration.
        df: Prepared table from Step 2; read from parquet when omitted.
        use_statsmodels: Include MSTL decomposition and ADF/KPSS tests.
    """
    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    frame = load_prepared(cfg.paths.prepared_parquet) if df is None else df
    keys, dpi = list(cfg.granularity), cfg.eda.figure_dpi
    labelled = labelled_rows(frame)
    total = daily_total(frame, cfg)
    wide = labelled.pivot(index=keys, columns=cfg.date_col, values=cfg.target_col).astype(float)
    figures = cfg.paths.figures_dir

    annual = annual_summary(total)
    dow, month = seasonal_index(labelled, "dayofweek", cfg), seasonal_index(labelled, "month", cfg)
    shape = yearly_shape(total)
    plot_trend(total, annual, figures / "eda_trend.png", dpi)
    plot_seasonality(dow, month, shape, figures / "eda_seasonality.png", dpi)

    holidays, noise_std = holiday_effects(total, cfg)
    plot_holidays(holidays, figures / "eda_holidays.png", dpi)

    classes = intermittency_table(wide, cfg.eda.adi_threshold, cfg.eda.cv2_threshold)
    plot_intermittency(classes, cfg.eda.adi_threshold, cfg.eda.cv2_threshold, figures / "eda_intermittency.png", dpi)
    cv = wide.std(axis=1) / wide.mean(axis=1)
    matrix = labelled.groupby(keys)[cfg.target_col].mean().unstack(keys[-1])
    plot_store_item(matrix, cv, figures / "eda_store_item.png", dpi)

    typical = wide.iloc[int(np.argmin(np.abs(wide.mean(axis=1).to_numpy() - wide.mean(axis=1).median())))]
    long_lags = cfg.eda.acf_lags_long
    acf_inputs = {"total sales": total.to_numpy(), "median series": typical.to_numpy(dtype=float)}
    acf_results = {name: {"acf": autocorrelation(x, long_lags), "pacf": partial_autocorrelation(x, cfg.eda.pacf_lags)}
                   for name, x in acf_inputs.items()}
    plot_acf_pacf(acf_results, len(total), cfg.eda.acf_lags_short, figures / "eda_acf_pacf.png", dpi)
    checkpoints = (1, 7, 14, 28, 364, 365)
    acf_summary = {name: {f"lag_{k}": round(float(v["acf"][k]), 4) for k in checkpoints if k <= long_lags}
                   for name, v in acf_results.items()}

    summary: Dict[str, Any] = {
        "overview": {"labelled_rows": int(len(labelled)), "series": int(len(wide)),
                     "dates": [total.index.min().date().isoformat(), total.index.max().date().isoformat()],
                     "mean_daily_sales_per_series": round(float(wide.to_numpy().mean()), 2)},
        "trend": _records(annual),
        "growth_by_series": growth_distribution(labelled, cfg),
        "day_of_week_index": _records(dow.reset_index().rename(columns={"dayofweek": "dow"})),
        "month_index": _records(month.reset_index()),
        "day_of_week_peak_to_trough_pct": round(float((dow["average"].max() / dow["average"].min() - 1) * 100), 1),
        "month_peak_to_trough_pct": round(float((month["average"].max() / month["average"].min() - 1) * 100), 1),
        "shape_similarity": {"day_of_week": profile_spread(labelled, "dayofweek", cfg),
                             "month": profile_spread(labelled, "month", cfg)},
        "holidays": {"noise_std_pct": round(noise_std * 100, 2), "table": _records(holidays),
                     "material": holidays.loc[holidays["material"], "holiday"].tolist()},
        "intermittency": {"class_counts": classes["demand_class"].value_counts().to_dict(),
                          "series_with_any_zero": int((classes["zero_share"] > 0).sum()),
                          "max_zero_share": round(float(classes["zero_share"].max()), 6)},
        "variability": {"cv_median": round(float(cv.median()), 3), "cv_min": round(float(cv.min()), 3),
                        "cv_max": round(float(cv.max()), 3)},
        "acf": acf_summary,
    }
    if use_statsmodels:
        components = mstl_components(total, cfg.eda.seasonal_periods)
        plot_decomposition(components, figures / "eda_decomposition.png", dpi)
        summary["decomposition_strength"] = {k: round(v, 4) for k, v in decomposition_strengths(components).items()}
        summary["stationarity"] = _stationarity_sections(total, wide, cfg)
    else:
        logger.warning("statsmodels sections skipped (decomposition and ADF/KPSS)")
        summary["decomposition_strength"] = summary["stationarity"] = "skipped"

    cfg.paths.eda_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %d figures to %s and %s", len(list(figures.glob("eda_*.png"))), figures, cfg.paths.eda_summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the Step 3 time series EDA.")
    parser.add_argument("--prepared-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--skip-statsmodels", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = ForecastConfig()
    paths = cfg.paths
    if args.reports_dir:
        paths = replace(paths, reports_dir=args.reports_dir)
    configure_logging(args.log_level)
    cfg = replace(cfg, paths=paths)
    frame = load_prepared(args.prepared_path or cfg.paths.prepared_parquet)
    run_eda(cfg, frame, use_statsmodels=not args.skip_statsmodels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
