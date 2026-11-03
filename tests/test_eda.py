"""Tests for forecast.eda (run on the small synthetic dataset)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forecast.config import ForecastConfig
from forecast.data import RawFrames
from forecast.eda import (
    annual_summary,
    daily_total,
    growth_distribution,
    holiday_effects,
    intermittency_table,
    labelled_rows,
    profile_spread,
    run_eda,
    same_month_weekday_ratio,
    seasonal_index,
)
from forecast.prepare import build_prepared_frame


@pytest.fixture
def prepared(raw_frames: RawFrames, cfg: ForecastConfig) -> pd.DataFrame:
    return build_prepared_frame(raw_frames, cfg)[0]


def test_daily_total_and_annual_summary(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    total = daily_total(prepared, cfg)
    assert len(total) == 1000 and total.index.freq is not None and not total.isna().any()
    assert total.sum() == labelled_rows(prepared)["sales"].sum()  # test rows never leak into the totals
    annual = annual_summary(total)
    assert annual["year"].tolist() == [2013, 2014, 2015] and np.isnan(annual["growth_pct"].iloc[0])
    assert annual["total_sales"].sum() == pytest.approx(total.sum())


def test_seasonal_index_is_relative_to_each_year(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    index = seasonal_index(labelled_rows(prepared), "dayofweek", cfg)
    assert index.shape == (7, 4) and list(index.columns) == ["2013", "2014", "2015", "average"]
    assert index["2013"].mean() == pytest.approx(1.0, abs=0.02)  # an index of 1.0 is the year average


def test_same_month_weekday_ratio_isolates_a_single_day() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2014-01-01", "2014-12-31", freq="D")
    total = pd.Series(100.0 * (1 + rng.normal(0, 0.002, len(index))), index=index)
    total[pd.Timestamp("2014-11-27")] *= 1.2  # a Thursday
    ratio = same_month_weekday_ratio(total)
    assert ratio[pd.Timestamp("2014-11-27")] == pytest.approx(1.2, abs=0.01)
    assert abs(ratio[pd.Timestamp("2014-11-20")] - 1.0) < 0.1  # the other Thursdays are unaffected


def test_holiday_effects_flags_only_the_injected_holiday(cfg: ForecastConfig) -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2013-01-01", "2016-12-31", freq="D")
    total = pd.Series(100.0 * (1 + rng.normal(0, 0.002, len(index))), index=index)
    for year in (2013, 2014, 2015, 2016):  # Thanksgiving is the 4th Thursday of November
        first = pd.Timestamp(year=year, month=11, day=1)
        thanksgiving = first + pd.Timedelta(days=(3 - first.dayofweek) % 7 + 21)
        total[thanksgiving] *= 1.25
    table, noise = holiday_effects(total, cfg)
    assert table.loc[table["material"], "holiday"].tolist() == ["Thanksgiving"]
    assert table.set_index("holiday").loc["Thanksgiving", "effect_pct"] == pytest.approx(25.0, abs=1.5)
    assert noise > 0


def test_intermittency_classes() -> None:
    days = 100
    smooth = np.full(days, 10.0) + np.arange(days) % 3
    intermittent = np.where(np.arange(days) % 4 == 0, 10.0, 0.0)
    steps = np.arange(days)
    lumpy = np.where(steps % 8 == 0, 1.0, np.where(steps % 8 == 4, 100.0, 0.0))  # sizes 1 and 100
    nothing = np.zeros(days)
    table = intermittency_table(pd.DataFrame([smooth, intermittent, lumpy, nothing]), 1.32, 0.49)
    assert table["demand_class"].tolist() == ["smooth", "intermittent", "lumpy", "no demand"]
    assert table["adi"].iloc[1] == pytest.approx(4.0) and table["zero_share"].iloc[1] == pytest.approx(0.75)


def test_profile_spread_and_growth(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    labelled = labelled_rows(prepared)
    spread = profile_spread(labelled, "dayofweek", cfg)
    assert spread["median_correlation_with_average"] > 0.5 and spread["max_std_across_series"] >= 0
    growth = growth_distribution(labelled, cfg)
    assert growth["year"] == 2015 and growth["p5_pct"] <= growth["median_pct"] <= growth["p95_pct"]


def test_run_eda_writes_figures_and_summary(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    summary = run_eda(cfg, prepared, use_statsmodels=False)
    expected = {"eda_trend", "eda_seasonality", "eda_holidays", "eda_intermittency", "eda_store_item", "eda_acf_pacf"}
    assert expected == {p.stem for p in cfg.paths.figures_dir.glob("eda_*.png")}
    saved = json.loads(cfg.paths.eda_summary.read_text())
    assert saved["overview"]["series"] == 6 and saved["intermittency"]["class_counts"] == {"smooth": 6}
    assert saved["decomposition_strength"] == "skipped" and summary["overview"] == saved["overview"]
    assert set(saved["acf"]["total sales"]) == {"lag_1", "lag_7", "lag_14", "lag_28", "lag_364", "lag_365"}


def test_run_eda_with_statsmodels(prepared: pd.DataFrame, cfg: ForecastConfig) -> None:
    pytest.importorskip("statsmodels")
    summary = run_eda(cfg, prepared, use_statsmodels=True)
    assert summary["decomposition_strength"]["seasonal_7"] > 0.5
    assert (cfg.paths.figures_dir / "eda_decomposition.png").is_file()
    assert {"level", "diff(1)", "diff(7)", "diff(7) then diff(1)"} == {
        r["transformation"] for r in summary["stationarity"]["total_series"]}
