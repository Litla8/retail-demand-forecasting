"""Tests for forecast.config."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from forecast.config import (
    BASELINE_NAMES,
    STATISTICAL_MODEL_NAMES,
    GBM_ENGINES,
    BaselineConfig,
    BusinessConfig,
    EdaConfig,
    ForecastConfig,
    GbmConfig,
    PathsConfig,
    StatisticalConfig,
    TuneConfig,
)


def test_defaults_match_the_project_brief() -> None:
    cfg = ForecastConfig()
    assert cfg.horizon_days == 90 and cfg.horizon_buckets == (28, 60, 90)
    assert cfg.granularity == ("store", "item") and cfg.primary_metric == "wape"
    assert cfg.random_state == 42 and cfg.benchmark == "seasonal_naive_364"
    assert cfg.validation_window_ends[-1] == "2017-12-31"
    assert cfg.negative_sales == "raise" and cfg.min_history_days == 728


def test_derived_paths() -> None:
    paths = PathsConfig()
    assert paths.prepared_parquet == Path("data/processed/sales_prepared.parquet")
    assert paths.quality_report == Path("reports/data_quality.json")
    assert paths.figures_dir == Path("reports/figures")
    assert paths.eda_summary == Path("reports/eda_summary.json")
    assert paths.baseline_results == Path("reports/baseline_results.csv")
    assert paths.baseline_summary == Path("reports/baseline_summary.json")
    assert paths.statistical_results == Path("reports/statistical_results.csv")
    assert paths.statistical_summary == Path("reports/statistical_summary.json")
    assert paths.gbm_results == Path("reports/gbm_results.csv")
    assert paths.gbm_summary == Path("reports/gbm_summary.json")
    assert paths.tuning_trials == Path("reports/tuning_trials.csv")
    assert paths.step7_summary == Path("reports/step7_summary.json")
    assert paths.gbm_params == Path("models/gbm_params.json")
    assert paths.business_summary == Path("reports/business_summary.json")
    assert paths.reorder_plan == Path("reports/reorder_plan.csv") and paths.submission == Path("reports/submission.csv")


def test_business_defaults_and_validation() -> None:
    biz = BusinessConfig()
    assert biz.protection_days == 14 and biz.target_fill_rate == 0.995 and ForecastConfig().business == biz
    with pytest.raises(ValueError, match="service_levels"):
        BusinessConfig(service_levels=(0.9, 0.8))
    with pytest.raises(ValueError, match="target_fill_rate"):
        BusinessConfig(target_fill_rate=1.0)
    with pytest.raises(ValueError, match="first_eval_window"):
        BusinessConfig(first_eval_window=0)
    with pytest.raises(ValueError, match="costs"):
        BusinessConfig(stockout_cost_per_unit=-1.0)


def test_tune_defaults() -> None:
    tune = TuneConfig()
    assert tune.search == "optuna" and tune.n_trials == 30 and tune.tuning_window_ends == ("2016-09-30", "2016-12-31")
    assert ForecastConfig().tune == tune


def test_gbm_defaults() -> None:
    gbm = GbmConfig()
    assert gbm.engine == "lightgbm" and gbm.engine in GBM_ENGINES and gbm.profile_years == 2
    assert gbm.min_origin >= gbm.level_window + max(gbm.trend_windows) - 1 and not gbm.use_series_ids
    assert ForecastConfig().gbm == gbm


def test_statistical_defaults() -> None:
    stat = StatisticalConfig()
    assert stat.models == STATISTICAL_MODEL_NAMES and stat.n_series == 10
    assert stat.sarima_seasonal_order == (1, 1, 1, 7)  # weekly seasonal difference from the Step 3 tests
    assert ForecastConfig().statistical == stat


def test_baseline_defaults() -> None:
    baselines = BaselineConfig()
    assert baselines.names == BASELINE_NAMES and "seasonal_naive_364" in baselines.names
    assert baselines.growth_window == 91 and baselines.profile_years == 2
    assert ForecastConfig().benchmark in baselines.names and 'month_weekday_profile_multi_year' in baselines.names


def test_eda_defaults_and_validation() -> None:
    eda = EdaConfig()
    assert eda.seasonal_periods == (7, 365) and eda.adi_threshold == 1.32 and eda.cv2_threshold == 0.49
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(eda, seasonal_periods=(365, 7))
    with pytest.raises(ValueError, match="at least 2"):
        replace(eda, seasonal_periods=(1, 7))
    with pytest.raises(ValueError, match="significance_level"):
        replace(eda, significance_level=1.5)
    with pytest.raises(ValueError, match="positive"):
        replace(eda, stationarity_sample_series=0)


def test_inconsistent_settings_fail_loudly() -> None:
    with pytest.raises(ValueError, match="last horizon bucket"):
        replace(ForecastConfig(), horizon_days=28)
    with pytest.raises(ValueError, match="increasing"):
        replace(ForecastConfig(), horizon_buckets=(60, 28, 90))
    with pytest.raises(ValueError, match="ordered"):
        replace(ForecastConfig(), target_improvement_pct=(20.0, 10.0))
    with pytest.raises(ValueError, match="negative_sales"):
        replace(ForecastConfig(), negative_sales="ignore")
    with pytest.raises(ValueError, match="min_history_days"):
        replace(ForecastConfig(), min_history_days=0)
