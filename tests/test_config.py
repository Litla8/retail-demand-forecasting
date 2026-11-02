"""Tests for forecast.config."""
from __future__ import annotations

from dataclasses import replace

import pytest

from forecast.config import ForecastConfig


def test_defaults_match_the_project_brief() -> None:
    cfg = ForecastConfig()
    assert cfg.horizon_days == 90 and cfg.horizon_buckets == (28, 60, 90)
    assert cfg.granularity == ("store", "item") and cfg.primary_metric == "wape"
    assert cfg.random_state == 42 and cfg.benchmark == "seasonal_naive_364"
    assert cfg.validation_window_ends[-1] == "2017-12-31"


def test_inconsistent_settings_fail_loudly() -> None:
    with pytest.raises(ValueError, match="last horizon bucket"):
        replace(ForecastConfig(), horizon_days=28)
    with pytest.raises(ValueError, match="increasing"):
        replace(ForecastConfig(), horizon_buckets=(60, 28, 90))
    with pytest.raises(ValueError, match="ordered"):
        replace(ForecastConfig(), target_improvement_pct=(20.0, 10.0))
