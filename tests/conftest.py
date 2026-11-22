"""Deterministic synthetic fixtures shared by the test suite (no Kaggle files needed)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.config import ForecastConfig
from forecast.data import RawFrames

N_DAYS = 1000  # 2013-01-01 .. 2015-09-27
HORIZON = 28


@pytest.fixture
def cfg(tmp_path: Path) -> ForecastConfig:
    """Small, fast config: 28-day horizon and windows that fit the 1,000-day fixture."""
    base = ForecastConfig()
    return replace(
        base,
        horizon_days=HORIZON,
        horizon_buckets=(7, 14, HORIZON),
        validation_window_ends=("2015-06-30", "2015-08-15", "2015-09-27"),
        min_history_days=364,
        paths=replace(base.paths, processed_dir=tmp_path / "processed", models_dir=tmp_path / "models",
                     reports_dir=tmp_path / "reports"),  # every output folder must be temporary
    )


@pytest.fixture
def raw_frames() -> RawFrames:
    """Raw-looking frames (string dates, int64 columns) for 6 series, like the CSV files."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2013-01-01", periods=N_DAYS, freq="D")
    series = pd.MultiIndex.from_product([[1, 2], [1, 2, 3]], names=["store", "item"]).to_frame(index=False)
    train = series.merge(pd.DataFrame({"date": dates}), how="cross")
    weekly = 1.0 + 0.2 * np.sin(2 * np.pi * train["date"].dt.day_of_week / 7)
    train["sales"] = rng.poisson(20 * weekly).astype("int64")
    train["date"] = train["date"].dt.strftime("%Y-%m-%d")  # CSV files store dates as text
    train = train[["date", "store", "item", "sales"]]

    test_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    test = series.merge(pd.DataFrame({"date": test_dates}), how="cross")
    test = test.sort_values(["store", "item", "date"]).reset_index(drop=True)
    test.insert(0, "id", np.arange(len(test), dtype="int64"))
    test["date"] = test["date"].dt.strftime("%Y-%m-%d")
    test = test[["id", "date", "store", "item"]]
    submission = pd.DataFrame({"id": test["id"], "sales": 52})
    return RawFrames(train, test, submission)
