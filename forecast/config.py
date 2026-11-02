"""Central configuration for the demand-forecasting project.

Every decision made in the Step 1 project brief (horizon, granularity, metrics,
validation windows, success target) lives here so later modules never hardcode them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

RANDOM_STATE: int = 42


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the pipeline."""

    train_csv: Path = Path("data/raw/train.csv")
    test_csv: Path = Path("data/raw/test.csv")
    sample_submission_csv: Path = Path("data/raw/sample_submission.csv")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


@dataclass(frozen=True)
class ForecastConfig:
    """Business and evaluation settings from the project brief.

    Attributes:
        horizon_days: Forecast horizon. 90 days matches the Kaggle test period
            (2018-01-01 to 2018-03-31) and a quarterly inventory-planning cycle.
        horizon_buckets: Upper edges (in days ahead) used to report error by horizon, so
            the 28-day view from the course brief is reported inside the 90-day forecast.
        granularity: Columns that identify one series (store x item = 500 series).
        primary_metric: Metric used to choose models (WAPE is robust and volume-weighted).
        validation_window_ends: Last day of each 90-day validation window. The Q1 window
            matches the season of the test period; the Q4 window is the most recent data.
        benchmark: Baseline every model must beat (same weekday one year earlier).
        target_improvement_pct: Required WAPE improvement over the benchmark.
    """

    random_state: int = RANDOM_STATE
    date_col: str = "date"
    target_col: str = "sales"
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
    benchmark: str = "seasonal_naive_364"
    target_improvement_pct: Tuple[float, float] = (10.0, 20.0)
    paths: PathsConfig = field(default_factory=PathsConfig)

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
