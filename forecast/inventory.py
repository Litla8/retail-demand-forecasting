"""Inventory simulation and safety-stock sizing from forecast errors.

A periodic-review policy is simulated with lost sales: every ``review`` days it orders enough to bring
stock plus open orders up to the forecast demand over the protection interval (lead time + review
period) plus a safety stock. The safety stock is sized from the *empirical* distribution of past
forecast errors, standardised by sqrt(forecast) because the error variance grows with the mean
(the Step 7 volume-error finding). Everything is vectorised over series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

EPSILON = 1e-9


def block_starts(horizon: int, review: int, protection: int) -> np.ndarray:
    """Review days that still have a full protection interval inside the horizon."""
    return np.arange(0, horizon - protection + 1, review)


def protection_sums(matrix: np.ndarray, starts: np.ndarray, protection: int) -> np.ndarray:
    """Sum of each row over ``protection`` days from every start: shape [n_series, n_starts]."""
    prefix = np.concatenate([np.zeros((matrix.shape[0], 1)), np.cumsum(matrix, axis=1)], axis=1)
    return prefix[:, starts + protection] - prefix[:, starts]


def standardized_errors(actual: np.ndarray, forecast: np.ndarray, review: int, protection: int) -> np.ndarray:
    """(actual - forecast) / sqrt(forecast) for the demand over each protection interval, flattened.

    Raises:
        ValueError: If the shapes differ or the horizon is shorter than one protection interval.
    """
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")
    starts = block_starts(actual.shape[1], review, protection)
    if len(starts) == 0:
        raise ValueError("Horizon is shorter than one protection interval")
    a, f = protection_sums(actual, starts, protection), protection_sums(forecast, starts, protection)
    return ((a - f) / np.sqrt(np.maximum(f, EPSILON))).ravel()


def quantile_table(errors: np.ndarray, levels: Sequence[float]) -> Dict[float, float]:
    """Empirical quantile of the standardised errors at each service level."""
    if errors.size == 0:
        raise ValueError("no errors to calibrate on")
    return {float(q): float(np.quantile(errors, q)) for q in levels}


@dataclass(frozen=True)
class SimulationResult:
    """Totals of one simulation, summed over series and days."""

    demand: float
    sold: float
    lost: float
    inventory_days: float
    n_series: int
    n_days: int

    @property
    def fill_rate(self) -> float:
        """Share of demand that was met from stock."""
        return self.sold / self.demand if self.demand > 0 else 1.0

    @property
    def avg_inventory(self) -> float:
        """Average units on hand per series per day (end of day)."""
        return self.inventory_days / (self.n_series * self.n_days)


def simulate_policy(actual: np.ndarray, forecast: np.ndarray, quantile: float, lead: int, review: int) -> SimulationResult:
    """Simulate lost-sales periodic review with safety stock ``max(quantile * sqrt(F), 0)``.

    The simulation starts fully stocked, orders arrive ``lead`` days after they are placed, and orders
    stop once they could no longer arrive inside the horizon. Near the end the protection interval is
    truncated to the days that have a forecast.

    Raises:
        ValueError: If the shapes differ or the lead time or review period are invalid.
    """
    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")
    if lead < 0 or review < 1:
        raise ValueError("lead must be >= 0 and review >= 1")
    n_series, horizon = actual.shape
    protection = lead + review
    forecast_prefix = np.concatenate([np.zeros((n_series, 1)), np.cumsum(forecast, axis=1)], axis=1)
    arrivals = np.zeros((n_series, horizon + lead + 1))
    on_hand = np.zeros(n_series)
    sold = lost = inventory = 0.0
    for day in range(horizon):
        on_hand += arrivals[:, day]
        if day % review == 0 and day + lead < horizon:
            expected = forecast_prefix[:, min(day + protection, horizon)] - forecast_prefix[:, day]
            order_up_to = expected + np.maximum(quantile * np.sqrt(np.maximum(expected, EPSILON)), 0.0)
            if day == 0:
                on_hand = order_up_to.copy()  # start fully stocked so there is no cold-start stock-out
            position = on_hand + arrivals[:, day + 1:].sum(axis=1)  # on hand plus everything already ordered
            order = np.maximum(order_up_to - position, 0.0)
            if lead == 0:
                on_hand += order
            else:
                arrivals[:, day + lead] += order
        demand = actual[:, day]
        met = np.minimum(on_hand, demand)
        on_hand -= met
        sold += float(met.sum())
        lost += float((demand - met).sum())
        inventory += float(on_hand.sum())
    return SimulationResult(demand=float(actual.sum()), sold=sold, lost=lost, inventory_days=inventory,
                            n_series=n_series, n_days=horizon)


def interval_quantiles(errors: np.ndarray, levels: Sequence[float]) -> Dict[float, Tuple[float, float]]:
    """Lower and upper standardised-error quantiles of central prediction intervals at each level."""
    return {float(level): (float(np.quantile(errors, (1 - level) / 2)), float(np.quantile(errors, 1 - (1 - level) / 2)))
            for level in levels}
