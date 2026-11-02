"""Tests for forecast.config."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from forecast.config import ForecastConfig, PathsConfig


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
