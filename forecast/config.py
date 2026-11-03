"""Central configuration for the demand-forecasting project.

Every decision made in the Step 1 project brief (horizon, granularity, metrics,
validation windows, success target) and every Step 2 data-preparation rule lives here so
later modules never hardcode them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

RANDOM_STATE: int = 42
NEGATIVE_SALES_POLICIES: Tuple[str, ...] = ("raise", "clip")


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the pipeline."""

    train_csv: Path = Path("data/raw/train.csv")
    test_csv: Path = Path("data/raw/test.csv")
    sample_submission_csv: Path = Path("data/raw/sample_submission.csv")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    @property
    def prepared_parquet(self) -> Path:
        """Prepared store-item-day table (train + test horizon) written by Step 2."""
        return self.processed_dir / "sales_prepared.parquet"

    @property
    def quality_report(self) -> Path:
        """JSON data-quality report written by Step 2."""
        return self.reports_dir / "data_quality.json"

    @property
    def figures_dir(self) -> Path:
        """Folder for every generated plot."""
        return self.reports_dir / "figures"

    @property
    def eda_summary(self) -> Path:
        """JSON summary of the Step 3 time series EDA."""
        return self.reports_dir / "eda_summary.json"


@dataclass(frozen=True)
class EdaConfig:
    """Settings for the Step 3 time series EDA.

    Attributes:
        seasonal_periods: Seasonal cycles to decompose (weekly and yearly, in days).
        acf_lags_short / acf_lags_long / pacf_lags: Lags shown in the ACF/PACF plots.
        holiday_material_pct / holiday_material_z: A holiday counts as material only if its
            average effect exceeds this percentage AND this many standard errors.
        adi_threshold / cv2_threshold: Syntetos-Boylan cut-offs for intermittent demand.
        stationarity_sample_series: Number of random series tested with ADF/KPSS.
        significance_level: Alpha for the stationarity tests.
        figure_dpi: Resolution of the saved figures.
    """

    seasonal_periods: Tuple[int, ...] = (7, 365)
    acf_lags_short: int = 42
    acf_lags_long: int = 400
    pacf_lags: int = 42
    holiday_material_pct: float = 3.0
    holiday_material_z: float = 3.0
    adi_threshold: float = 1.32
    cv2_threshold: float = 0.49
    stationarity_sample_series: int = 50
    significance_level: float = 0.05
    figure_dpi: int = 150

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if not self.seasonal_periods or list(self.seasonal_periods) != sorted(set(self.seasonal_periods)):
            raise ValueError("seasonal_periods must be strictly increasing")
        if min(self.seasonal_periods) < 2:
            raise ValueError("seasonal periods must be at least 2 days")
        if min(self.acf_lags_short, self.acf_lags_long, self.pacf_lags,
               self.stationarity_sample_series) < 1:
            raise ValueError("lag and sample settings must be positive")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError("significance_level must be strictly between 0 and 1")


@dataclass(frozen=True)
class ForecastConfig:
    """Business, evaluation and data-preparation settings.

    Attributes:
        horizon_days: Forecast horizon. 90 days matches the Kaggle test period
            (2018-01-01 to 2018-03-31) and a quarterly inventory-planning cycle.
        horizon_buckets: Upper edges (in days ahead) used to report error by horizon, so
            the 28-day view from the course brief is reported inside the 90-day forecast.
        granularity: Columns that identify one series (store x item = 500 series).
        primary_metric: Metric used to choose models (WAPE is robust and volume-weighted).
        validation_window_ends: Last day of each validation window of ``horizon_days`` days.
            The Q1 window matches the season of the test period; Q4 is the most recent data.
        min_history_days: Minimum training history required before every validation window
            (728 days gives two full years, enough for the 364-day benchmark and yearly lags).
        benchmark: Baseline every model must beat (same weekday one year earlier).
        target_improvement_pct: Required WAPE improvement over the benchmark.
        negative_sales: What to do with negative sales: "raise" (default, fail loudly) or
            "clip" to zero. Zero sales are valid demand and are always kept.
    """

    random_state: int = RANDOM_STATE
    date_col: str = "date"
    target_col: str = "sales"
    id_col: str = "id"
    granularity: Tuple[str, ...] = ("store", "item")
    horizon_days: int = 90
    horizon_buckets: Tuple[int, ...] = (28, 60, 90)
    primary_metric: str = "wape"
    secondary_metrics: Tuple[str, ...] = ("smape", "mae", "rmse", "bias")
    validation_window_ends: Tuple[str, ...] = (
        "2017-03-31",
        "2017-06-30",
        "2017-09-30",
        "2017-12-31",
    )
    min_history_days: int = 728
    benchmark: str = "seasonal_naive_364"
    target_improvement_pct: Tuple[float, float] = (10.0, 20.0)
    negative_sales: str = "raise"
    paths: PathsConfig = field(default_factory=PathsConfig)
    eda: EdaConfig = field(default_factory=EdaConfig)

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if not self.horizon_buckets or self.horizon_buckets[-1] != self.horizon_days:
            raise ValueError("The last horizon bucket edge must equal horizon_days")
        if list(self.horizon_buckets) != sorted(set(self.horizon_buckets)):
            raise ValueError("horizon_buckets must be strictly increasing")
        low, high = self.target_improvement_pct
        if not 0 <= low <= high:
            raise ValueError("target_improvement_pct must be an ordered (low, high) pair")
        if self.negative_sales not in NEGATIVE_SALES_POLICIES:
            raise ValueError(f"negative_sales must be one of {NEGATIVE_SALES_POLICIES}")
        if self.min_history_days < 1:
            raise ValueError("min_history_days must be positive")
