"""Tests for forecast.inventory."""
from __future__ import annotations

import numpy as np
import pytest

from forecast.inventory import (
    block_starts,
    interval_quantiles,
    protection_sums,
    quantile_table,
    simulate_policy,
    standardized_errors,
)


def test_blocks_and_protection_sums() -> None:
    assert block_starts(90, 7, 14).tolist() == list(range(0, 71, 7))  # 70 + 14 = 84 <= 90, 77 + 14 = 91 > 90
    matrix = np.arange(1.0, 11.0)[None, :]
    assert protection_sums(matrix, np.array([0, 3]), 4).tolist() == [[10.0, 22.0]]  # 1+2+3+4 and 4+5+6+7


def test_standardized_errors() -> None:
    actual = np.full((2, 28), 10.0)
    assert np.allclose(standardized_errors(actual, actual, 7, 14), 0.0)
    low = standardized_errors(actual, actual * 0.5, 7, 14)  # forecast half of actual -> positive errors
    assert (low > 0).all() and low.size == 2 * 3
    with pytest.raises(ValueError, match="Shape mismatch"):
        standardized_errors(actual, actual[:, :20], 7, 14)
    with pytest.raises(ValueError, match="shorter"):
        standardized_errors(actual[:, :10], actual[:, :10], 7, 14)


def test_quantile_table_and_intervals() -> None:
    errors = np.arange(-50.0, 51.0)  # a uniform spread so quantiles are easy to read
    assert quantile_table(errors, [0.5, 0.9]) == {0.5: 0.0, 0.9: 40.0}
    low, high = interval_quantiles(errors, [0.8])[0.8]
    assert low == pytest.approx(-40.0) and high == pytest.approx(40.0)
    with pytest.raises(ValueError, match="no errors"):
        quantile_table(np.array([]), [0.9])


def test_perfect_forecast_with_constant_demand_never_stocks_out() -> None:
    demand = np.full((3, 90), 10.0)
    result = simulate_policy(demand, demand.copy(), quantile=0.0, lead=7, review=7)
    assert result.fill_rate == pytest.approx(1.0) and result.lost == pytest.approx(0.0)
    assert result.sold + result.lost == pytest.approx(result.demand)  # every unit of demand is accounted for
    assert result.avg_inventory > 0
    zero_lead = simulate_policy(demand, demand.copy(), 0.0, lead=0, review=7)
    assert zero_lead.fill_rate == pytest.approx(1.0)


def test_under_forecasting_loses_sales_and_safety_stock_helps() -> None:
    actual = np.full((4, 90), 10.0)
    forecast = np.full((4, 90), 6.0)  # 40% too low
    plain = simulate_policy(actual, forecast, 0.0, 7, 7)
    padded = simulate_policy(actual, forecast, 3.0, 7, 7)
    assert plain.fill_rate < 0.8 and plain.lost > 0
    assert padded.fill_rate > plain.fill_rate and padded.avg_inventory > plain.avg_inventory


def test_more_safety_stock_means_higher_service_and_more_inventory() -> None:
    rng = np.random.default_rng(42)
    actual = rng.poisson(30, size=(60, 90)).astype(float)
    forecast = np.full(actual.shape, 30.0)  # unbiased forecast; the only problem is random noise
    results = [simulate_policy(actual, forecast, q, 7, 7) for q in (0.0, 1.0, 2.0, 3.0)]
    assert all(b.fill_rate >= a.fill_rate for a, b in zip(results, results[1:]))
    assert all(b.avg_inventory >= a.avg_inventory for a, b in zip(results, results[1:]))
    assert results[-1].fill_rate > 0.97


def test_simulation_input_validation() -> None:
    actual = np.ones((2, 30))
    with pytest.raises(ValueError, match="Shape mismatch"):
        simulate_policy(actual, actual[:, :20], 1.0, 7, 7)
    with pytest.raises(ValueError, match="lead"):
        simulate_policy(actual, actual, 1.0, -1, 7)
    with pytest.raises(ValueError, match="review"):
        simulate_policy(actual, actual, 1.0, 7, 0)
