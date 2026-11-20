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

    @property
    def baseline_results(self) -> Path:
        """CSV with every baseline x window x horizon-bucket score (Step 4)."""
        return self.reports_dir / "baseline_results.csv"

    @property
    def baseline_summary(self) -> Path:
        """JSON summary of the Step 4 baseline backtest."""
        return self.reports_dir / "baseline_summary.json"

    @property
    def statistical_results(self) -> Path:
        """CSV with every statistical model x window x horizon-bucket score (Step 5)."""
        return self.reports_dir / "statistical_results.csv"

    @property
    def statistical_summary(self) -> Path:
        """JSON summary of the Step 5 statistical-model backtest."""
        return self.reports_dir / "statistical_summary.json"

    @property
    def gbm_results(self) -> Path:
        """CSV with every gradient-boosting x window x horizon-bucket score (Step 6)."""
        return self.reports_dir / "gbm_results.csv"

    @property
    def gbm_summary(self) -> Path:
        """JSON summary of the Step 6 gradient-boosting backtest."""
        return self.reports_dir / "gbm_summary.json"

    @property
    def tuning_trials(self) -> Path:
        """CSV with every hyperparameter trial of Step 7."""
        return self.reports_dir / "tuning_trials.csv"

    @property
    def step7_results(self) -> Path:
        """CSV with every Step 7 variant x window x horizon-bucket score."""
        return self.reports_dir / "step7_results.csv"

    @property
    def step7_summary(self) -> Path:
        """JSON summary of Step 7: tuning, stacking and error analysis."""
        return self.reports_dir / "step7_summary.json"

    @property
    def gbm_params(self) -> Path:
        """JSON with the selected model configuration, used by the later steps."""
        return self.models_dir / "gbm_params.json"

    @property
    def business_results(self) -> Path:
        """CSV with the inventory simulation for every forecaster and service level (Step 8)."""
        return self.reports_dir / "business_results.csv"

    @property
    def business_summary(self) -> Path:
        """JSON summary of the Step 8 inventory analysis."""
        return self.reports_dir / "business_summary.json"

    @property
    def reorder_plan(self) -> Path:
        """Weekly reorder plan for the test period (Step 8)."""
        return self.reports_dir / "reorder_plan.csv"

    @property
    def forecast_intervals(self) -> Path:
        """Daily forecasts with prediction intervals for the test period (Step 8)."""
        return self.reports_dir / "forecast_intervals.csv"

    @property
    def submission(self) -> Path:
        """Kaggle-format submission (id, sales) for the test period."""
        return self.reports_dir / "submission.csv"

    @property
    def backtest_forecasts(self) -> Path:
        """Validation-window actuals and forecasts of the main forecasters, for the dashboard."""
        return self.reports_dir / "backtest_forecasts.csv.gz"

    @property
    def archive_dir(self) -> Path:
        """Folder with dated copies of earlier forecasts, evaluated by the monitor."""
        return self.reports_dir / "archive"

    @property
    def monitoring_log(self) -> Path:
        """Running log of how earlier forecasts performed on the actuals that arrived later."""
        return self.reports_dir / "monitoring_log.csv"

    @property
    def dashboard_dir(self) -> Path:
        """Compact, committable data bundle read by the dashboard."""
        return Path("dashboard_data")


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


BASELINE_NAMES: Tuple[str, ...] = (
    "naive",
    "moving_average",
    "seasonal_naive_7",
    "seasonal_moving_average",
    "seasonal_naive_364",
    "seasonal_naive_364_growth",
    "month_weekday_profile",
    "month_weekday_profile_growth",
    "month_weekday_profile_multi_year",
)


@dataclass(frozen=True)
class BaselineConfig:
    """Settings for the Step 4 baseline models.

    Attributes:
        names: Baselines to run, in reporting order.
        moving_average_window: Days averaged by the flat moving-average baseline.
        seasonal_weeks: Same-weekday values averaged by the seasonal moving average.
        growth_window: Days used to measure year-over-year growth (recent days versus the
            same days 364 days earlier) for the growth-adjusted baselines. 91 days (one
            quarter, about the forecast horizon) was chosen after comparing 28, 91, 182 and
            364 days on the validation windows; the mean WAPE differed by under 0.2 points.
        profile_years: Past years pooled by the multi-year month-weekday profile. 2 and 3 scored the
            same on average; 2 was better in the Q1 window, which is the season of the test period.
    """

    names: Tuple[str, ...] = BASELINE_NAMES
    moving_average_window: int = 28
    seasonal_weeks: int = 4
    growth_window: int = 91
    profile_years: int = 2

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        unknown = [n for n in self.names if n not in BASELINE_NAMES]
        if unknown or not self.names:
            raise ValueError(f"names must be a non-empty subset of {BASELINE_NAMES}; got {unknown}")
        if min(self.moving_average_window, self.seasonal_weeks, self.growth_window, self.profile_years) < 1:
            raise ValueError("window settings must be positive")


STATISTICAL_MODEL_NAMES: Tuple[str, ...] = (
    "holt_winters",
    "sarima",
    "sarimax_month",
    "prophet_default",
    "prophet_month",
    "prophet_month_holidays",
)


@dataclass(frozen=True)
class StatisticalConfig:
    """Settings for the Step 5 statistical models.

    These models are fitted one series at a time and are slow, so they run on a small sample of
    series chosen at evenly spaced sales volumes and are compared with the baselines on those
    same series.

    Attributes:
        models: Which models to run.
        n_series: Number of sampled series.
        ets_fit_days / sarima_fit_days / prophet_fit_days: Most recent days used to fit each family.
        sarima_order / sarima_seasonal_order: (p, d, q) and (P, D, Q, s) for SARIMA(X). The weekly
            seasonal difference (D = 1, s = 7) follows from the Step 3 stationarity tests.
        sarima_maxiter: Optimiser iterations per SARIMA fit.
    """

    models: Tuple[str, ...] = STATISTICAL_MODEL_NAMES
    n_series: int = 10
    ets_fit_days: int = 730
    sarima_fit_days: int = 730
    prophet_fit_days: int = 1460
    sarima_order: Tuple[int, int, int] = (1, 0, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)
    sarima_maxiter: int = 50

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        unknown = [m for m in self.models if m not in STATISTICAL_MODEL_NAMES]
        if unknown or not self.models:
            raise ValueError(f"models must be a non-empty subset of {STATISTICAL_MODEL_NAMES}; got {unknown}")
        if self.n_series < 1:
            raise ValueError("n_series must be positive")
        if self.ets_fit_days < 28:
            raise ValueError("ets_fit_days must be at least 28")
        # Month dummies need every month present in the fit window, so it must cover a full year.
        if min(self.sarima_fit_days, self.prophet_fit_days) < 400:
            raise ValueError("sarima_fit_days and prophet_fit_days must be at least 400")
        if len(self.sarima_order) != 3 or len(self.sarima_seasonal_order) != 4:
            raise ValueError("sarima_order needs 3 values and sarima_seasonal_order 4")
        if self.sarima_seasonal_order[3] != 7 or self.sarima_maxiter < 1:
            raise ValueError("the seasonal period must be 7 and sarima_maxiter positive")


GBM_ENGINES: Tuple[str, ...] = ("lightgbm", "hist_gb")


@dataclass(frozen=True)
class GbmConfig:
    """Settings for the Step 6 global gradient-boosting model.

    One row is one (series, forecast origin, target day). Features use only data up to the origin
    and the target is the day's sales divided by the series' trailing-year average, so all 500
    series share one scale.

    Attributes:
        engine: "lightgbm" or "hist_gb" (scikit-learn's histogram boosting, same algorithm family).
        level_window: Days in the trailing average used as the series scale (364 = 52 whole weeks).
        trend_windows: Recent windows (days) whose growth versus a year earlier is a feature.
        profile_years: Past years whose same-month and same-weekday averages are features.
        min_origin: Earliest origin index allowed (needs 364 + the longest trend window of history).
        use_series_ids: Add store and item as features (the EDA found one shared shape, so off).
        n_estimators, learning_rate, num_leaves, min_child_samples, subsample, colsample, reg_lambda:
            Boosting hyperparameters (Step 7 tunes them).
    """

    engine: str = "lightgbm"
    level_window: int = 364
    trend_windows: Tuple[int, ...] = (28, 91)
    profile_years: int = 2
    min_origin: int = 454
    use_series_ids: bool = False
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample: float = 0.8
    reg_lambda: float = 1.0
    n_jobs: int = -1

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if self.engine not in GBM_ENGINES:
            raise ValueError(f"engine must be one of {GBM_ENGINES}")
        if self.level_window < 7 or min(self.trend_windows) < 1 or self.profile_years < 1:
            raise ValueError("level_window must be at least 7 and the other windows/years positive")
        if self.min_origin < self.level_window + max(self.trend_windows) - 1:
            raise ValueError("min_origin must be at least level_window + max(trend_windows) - 1")
        if self.n_estimators < 1 or not 0 < self.learning_rate <= 1 or self.num_leaves < 2:
            raise ValueError("invalid boosting hyperparameters")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample <= 1:
            raise ValueError("subsample and colsample must be in (0, 1]")


TUNE_SEARCHES: Tuple[str, ...] = ("optuna", "random")


@dataclass(frozen=True)
class TuneConfig:
    """Settings for Step 7 (tuning, bias calibration, blending).

    Attributes:
        n_trials: Hyperparameter trials, counting the default configuration as trial 0.
        search: "optuna" (TPE sampler) or "random"; falls back to random if optuna is missing.
        tuning_window_ends: Last day of each 90-day window used to score trials. They must end
            before the first validation window starts, so tuning never sees validation data.
        blend_step: Grid step for the weight of the boosting model in a blend (0.1 -> 0, 0.1, ..., 1).
        calibration_clip: Allowed range of the multiplicative bias correction.
        min_gain_points: A more complex variant must beat the simpler one by this many WAPE points.
    """

    n_trials: int = 30
    search: str = "optuna"
    tuning_window_ends: Tuple[str, ...] = ("2016-09-30", "2016-12-31")
    blend_step: float = 0.1
    calibration_clip: Tuple[float, float] = (0.9, 1.1)
    min_gain_points: float = 0.05

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if self.search not in TUNE_SEARCHES:
            raise ValueError(f"search must be one of {TUNE_SEARCHES}")
        if self.n_trials < 1 or not self.tuning_window_ends:
            raise ValueError("n_trials must be positive and tuning_window_ends non-empty")
        if not 0 < self.blend_step <= 1:
            raise ValueError("blend_step must be in (0, 1]")
        low, high = self.calibration_clip
        if not 0 < low < 1 < high:
            raise ValueError("calibration_clip must satisfy 0 < low < 1 < high")
        if self.min_gain_points < 0:
            raise ValueError("min_gain_points must be non-negative")


@dataclass(frozen=True)
class BusinessConfig:
    """Settings and placeholder economics for the Step 8 inventory analysis.

    The dataset has no prices or costs, so the cost figures are assumptions to be replaced with the
    retailer's real numbers; the service-level versus inventory trade-off does not depend on them.

    Attributes:
        lead_time_days: Days between placing an order and receiving it.
        review_days: Days between orders (a periodic-review policy).
        service_levels: Quantile levels of demand over the protection interval tested for safety stock.
        target_fill_rate: Fill rate at which forecasters are compared on the inventory they need; if it lies
            outside the range every forecaster reaches, the middle of that common range is used instead.
        holding_cost_per_unit_day: Cost of holding one unit in stock for one day.
        stockout_cost_per_unit: Cost of one unit of unmet demand (lost margin).
        first_eval_window: Index of the first validation window that is simulated; earlier windows
            only provide the forecast errors used to size safety stock, so no window is calibrated on its own errors
            or on later ones.
        interval_levels: Central prediction-interval levels reported for the forecasts.
    """

    lead_time_days: int = 7
    review_days: int = 7
    service_levels: Tuple[float, ...] = (0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99)
    target_fill_rate: float = 0.995
    holding_cost_per_unit_day: float = 0.02
    stockout_cost_per_unit: float = 3.0
    first_eval_window: int = 1
    interval_levels: Tuple[float, ...] = (0.80, 0.95)

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if self.lead_time_days < 0 or self.review_days < 1:
            raise ValueError("lead_time_days must be >= 0 and review_days >= 1")
        if not self.service_levels or any(not 0.5 <= q < 1 for q in self.service_levels):
            raise ValueError("service_levels must be non-empty and in [0.5, 1)")
        if list(self.service_levels) != sorted(set(self.service_levels)):
            raise ValueError("service_levels must be strictly increasing")
        if not 0 < self.target_fill_rate < 1:
            raise ValueError("target_fill_rate must be strictly between 0 and 1")
        if self.holding_cost_per_unit_day < 0 or self.stockout_cost_per_unit < 0:
            raise ValueError("costs must be non-negative")
        if self.first_eval_window < 1:
            raise ValueError("first_eval_window must be at least 1 (window 0 only calibrates)")
        if any(not 0 < q < 1 for q in self.interval_levels):
            raise ValueError("interval_levels must be strictly between 0 and 1")

    @property
    def protection_days(self) -> int:
        """Days of demand an order must cover: lead time plus the review period."""
        return self.lead_time_days + self.review_days


@dataclass(frozen=True)
class MonitorConfig:
    """Settings for the monthly accuracy check of earlier forecasts.

    Attributes:
        min_days: Realised days needed before a forecast is judged.
        wape_tolerance: Alert when realised WAPE exceeds the backtest WAPE by more than this fraction.
        min_coverage_95: Alert when fewer than this share of actuals fall inside the nominal 95% interval
            (a floor well below 0.95, because the backtest showed the intervals run narrow).
    """

    min_days: int = 14
    wape_tolerance: float = 0.20
    min_coverage_95: float = 0.80

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if self.min_days < 1 or self.wape_tolerance < 0:
            raise ValueError("min_days must be positive and wape_tolerance non-negative")
        if not 0 < self.min_coverage_95 < 1:
            raise ValueError("min_coverage_95 must be strictly between 0 and 1")


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
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    gbm: GbmConfig = field(default_factory=GbmConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

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
