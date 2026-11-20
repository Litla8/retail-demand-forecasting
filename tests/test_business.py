"""Tests for forecast.business (synthetic demand with known forecast quality)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.business import (
    build_submission,
    cost_optimal,
    evaluate_policies,
    interval_table,
    inventory_at_fill_rate,
    matched_target,
    reorder_plan,
)
from forecast.config import BusinessConfig, ForecastConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame

BIZ = BusinessConfig()


def _records(n_windows: int = 3, n_series: int = 40, horizon: int = 56):
    rng = np.random.default_rng(42)
    good, bad = [], []
    for _ in range(n_windows):
        actual = rng.poisson(50, size=(n_series, horizon)).astype(float)
        good.append((actual, np.full(actual.shape, 50.0)))
        series_error = np.exp(rng.normal(0, 0.3, size=(n_series, 1)))  # a level error that differs per series
        bad.append((actual, 50.0 * series_error * np.ones_like(actual)))
    return {"good": good, "bad": bad}


def test_frontier_shape_monotonicity_and_ordering() -> None:
    frontier = evaluate_policies(_records(), BIZ)
    assert len(frontier) == 2 * len(BIZ.service_levels)
    for _, group in frontier.groupby("forecaster"):
        group = group.sort_values("service_level")
        assert group["fill_rate"].is_monotonic_increasing and group["avg_inventory"].is_monotonic_increasing
    low = max(frontier.groupby("forecaster")["fill_rate"].min())
    high = min(frontier.groupby("forecaster")["fill_rate"].max())
    target = (low + high) / 2
    assert inventory_at_fill_rate(frontier, "good", target) < inventory_at_fill_rate(frontier, "bad", target)


def test_safety_stock_is_calibrated_out_of_sample() -> None:
    frontier = evaluate_policies(_records(n_windows=4), BIZ).query("forecaster == 'good'")
    assert (frontier["coverage"] - frontier["service_level"]).abs().max() < 0.08  # honest, not just in-sample
    assert frontier["service_level"].is_monotonic_increasing


def test_needs_a_window_to_calibrate_on() -> None:
    with pytest.raises(ValueError, match="first_eval_window"):
        evaluate_policies(_records(n_windows=1), BIZ)
    uneven = {"a": _records(2)["good"], "b": _records(3)["good"]}
    with pytest.raises(ValueError, match="same number"):
        evaluate_policies(uneven, BIZ)


def test_cost_optimal_and_interpolation_edge_cases() -> None:
    frontier = evaluate_policies(_records(), BIZ)
    best = cost_optimal(frontier)
    assert sorted(best["forecaster"]) == ["bad", "good"]
    for row in best.itertuples():
        same = frontier[frontier["forecaster"] == row.forecaster]
        assert row.total_cost == same["total_cost"].min()
    assert np.isnan(inventory_at_fill_rate(frontier, "good", 0.0001))  # outside the reached range


def test_reorder_plan_layout() -> None:
    series = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1)], names=["store", "item"])
    forecast = np.full((3, 28), 10.0)
    dates = pd.date_range("2018-01-01", periods=28)
    plan = reorder_plan(forecast, series, dates, quantile=1.0, service_level=0.9, cfg=BIZ)
    assert len(plan) == 3 * 3  # reviews on days 0, 7, 14; day 21 + lead 7 would arrive after the period
    assert (plan["order_up_to"] >= plan["forecast_units"]).all() and (plan["safety_stock"] >= 0).all()
    assert plan["forecast_units"].eq(140.0).all() and plan["order_date"].nunique() == 3
    assert plan["safety_stock"].iloc[0] == pytest.approx(np.sqrt(140.0))
    with pytest.raises(ValueError, match="n_series"):
        reorder_plan(forecast[:2], series, dates, 1.0, 0.9, BIZ)


def test_interval_table_orders_the_bounds() -> None:
    series = pd.MultiIndex.from_tuples([(1, 1), (2, 2)], names=["store", "item"])
    forecast = np.vstack([np.full(10, 20.0), np.full(10, 80.0)])
    errors = np.random.default_rng(42).normal(0, 1, 5000)
    table = interval_table(forecast, series, pd.date_range("2018-01-01", periods=10), errors, (0.8, 0.95))
    assert {"lower_80", "upper_80", "lower_95", "upper_95"} <= set(table.columns) and len(table) == 20
    assert (table["lower_95"] <= table["lower_80"]).all() and (table["upper_95"] >= table["upper_80"]).all()
    assert (table["lower_80"] <= table["forecast"]).all() and (table["forecast"] <= table["upper_80"]).all()
    wide = table[table["forecast"] == 80.0].iloc[0]
    narrow = table[table["forecast"] == 20.0].iloc[0]
    assert wide["upper_95"] - wide["lower_95"] > narrow["upper_95"] - narrow["lower_95"]  # bigger series, wider range


def test_submission_matches_the_test_rows(raw_frames: RawFrames, cfg: ForecastConfig) -> None:
    prepared = build_prepared_frame(raw_frames, cfg)[0]
    series = pd.MultiIndex.from_product([[1, 2], [1, 2, 3]], names=["store", "item"])
    dates = pd.DatetimeIndex(sorted(prepared.loc[prepared["is_test"], "date"].unique()))
    forecast = np.random.default_rng(42).uniform(5, 50, size=(6, len(dates)))
    submission = build_submission(prepared, forecast, series, dates)
    assert list(submission.columns) == ["id", "sales"] and len(submission) == 6 * 28
    assert submission["id"].is_unique and submission["id"].is_monotonic_increasing and (submission["sales"] >= 0).all()
    with pytest.raises(ValueError, match="cover exactly"):
        build_submission(prepared, forecast[:5], series[:5], dates)


def test_matched_target_stays_inside_the_range_every_forecaster_reaches() -> None:
    frontier = evaluate_policies(_records(), BIZ)
    spans = frontier.groupby("forecaster")["fill_rate"].agg(["min", "max"])
    low, high = spans["min"].max(), spans["max"].min()
    inside = (low + high) / 2
    assert matched_target(frontier, inside) == inside  # a reachable request is kept
    assert matched_target(frontier, 0.0001) == pytest.approx((low + high) / 2)  # an unreachable one is replaced
    assert not np.isnan(inventory_at_fill_rate(frontier, "good", matched_target(frontier, 0.99999)))
    disjoint = frontier.copy()
    disjoint.loc[disjoint["forecaster"] == "bad", "fill_rate"] = disjoint["fill_rate"] * 0 + 0.1
    assert np.isnan(matched_target(disjoint, 0.99))
