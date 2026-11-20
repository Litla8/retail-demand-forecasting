"""Tests for forecast.monitor."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.business import interval_table
from forecast.config import ForecastConfig, MonitorConfig
from forecast.data import RawFrames
from forecast.monitor import archive_forecast, evaluate_archived, run_monitor
from forecast.prepare import build_prepared_frame
from forecast.wide import to_wide

CFG = MonitorConfig()


def _forecast_csv(path: Path, wide, scale: float, days: int = 28, spread: float = 0.15) -> Path:
    """Forecast for the last ``days`` known days, as a multiple of the actuals, with intervals around it."""
    dates = wide.dates[-days:]
    actual = wide.values[:, -days:]
    forecast = actual * scale
    errors = np.random.default_rng(42).normal(0, 1, 3000)
    table = interval_table(forecast, wide.series, dates, errors, (0.8, 0.95))
    root = spread * forecast.ravel()  # intervals proportional to the forecast, so coverage depends on how wrong it is
    for level, z in ((80, 1.28), (95, 1.96)):
        table[f"lower_{level}"] = forecast.ravel() - z * root
        table[f"upper_{level}"] = forecast.ravel() + z * root
    table.to_csv(path, index=False)
    return path


@pytest.fixture
def wide(raw_frames: RawFrames, cfg: ForecastConfig):
    return to_wide(build_prepared_frame(raw_frames, cfg)[0], cfg)


def test_good_forecast_raises_no_alert(wide, tmp_path) -> None:
    csv = _forecast_csv(tmp_path / "forecast_intervals_a.csv", wide, scale=1.0)
    result = evaluate_archived(csv, wide, expected_wape=10.0, cfg=CFG)
    assert result["wape"] == pytest.approx(0.0, abs=1e-6) and result["coverage_95"] == 1.0
    assert result["alert"] is False and result["reasons"] == "" and result["realized_days"] == 28


def test_bad_forecast_raises_alerts_for_error_and_coverage(wide, tmp_path) -> None:
    csv = _forecast_csv(tmp_path / "forecast_intervals_b.csv", wide, scale=1.8, spread=0.05)
    result = evaluate_archived(csv, wide, expected_wape=10.0, cfg=CFG)
    assert result["wape"] == pytest.approx(80.0) and result["bias_pct"] == pytest.approx(80.0)
    assert result["alert"] is True and "WAPE" in result["reasons"] and "95%" in result["reasons"]


def test_not_enough_realized_days_returns_none(wide, tmp_path) -> None:
    csv = _forecast_csv(tmp_path / "forecast_intervals_c.csv", wide, scale=1.0, days=7)
    assert evaluate_archived(csv, wide, 10.0, CFG) is None  # 7 days < the 14 needed
    future = pd.read_csv(csv, parse_dates=["date"]).assign(date=lambda t: t["date"] + pd.Timedelta(days=400))
    future.to_csv(csv, index=False)
    assert evaluate_archived(csv, wide, 10.0, CFG) is None  # nothing has been realised at all


def test_run_monitor_writes_a_repeatable_log(wide, raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    cfg.paths.archive_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.gbm_params.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.gbm_params.write_text(json.dumps({"mean_wape": 10.0}))
    _forecast_csv(cfg.paths.archive_dir / "forecast_intervals_2015-09-01.csv", wide, scale=1.0)
    _forecast_csv(cfg.paths.archive_dir / "forecast_intervals_2015-08-01.csv", wide, scale=1.8, spread=0.05)
    first = run_monitor(cfg, prepared)
    second = run_monitor(cfg, prepared)
    assert first == second and len(first) == 2 and [r["alert"] for r in first] == [True, False]  # sorted by file name
    log = pd.read_csv(cfg.paths.monitoring_log)
    assert len(log) == 2 and set(log["archive"]) == {"forecast_intervals_2015-08-01.csv", "forecast_intervals_2015-09-01.csv"}


def test_monitor_needs_the_selected_model_file(wide, cfg: ForecastConfig) -> None:
    with pytest.raises(FileNotFoundError, match="run_tuning"):
        run_monitor(cfg, None)


def test_archive_is_named_after_the_first_forecast_day(wide, cfg: ForecastConfig) -> None:
    with pytest.raises(FileNotFoundError, match="run_business"):
        archive_forecast(cfg)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    _forecast_csv(cfg.paths.forecast_intervals, wide, scale=1.0)
    target = archive_forecast(cfg)
    assert target.name == f"forecast_intervals_{wide.dates[-28].date().isoformat()}.csv" and target.is_file()
    assert archive_forecast(cfg) == target  # archiving twice is harmless


def test_monitor_config_validation() -> None:
    with pytest.raises(ValueError, match="min_days"):
        MonitorConfig(min_days=0)
    with pytest.raises(ValueError, match="min_coverage_95"):
        MonitorConfig(min_coverage_95=1.0)
