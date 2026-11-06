"""Tests for forecast.ensemble (stand-in forecasters with known behaviour)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast.config import TuneConfig
from forecast.ensemble import VARIANTS, StackedModels, best_blend_weight

DATES = pd.date_range("2015-01-01", periods=300, freq="D")
HORIZON = 28


class CountingConstant:
    """Forecasts a constant level and counts how often it is called."""

    def __init__(self, level: float) -> None:
        self.level, self.calls = level, 0

    def __call__(self, history, history_dates, future_dates):
        self.calls += 1
        return np.full((history.shape[0], len(future_dates)), self.level)


def _setup(gbm_level: float):
    history = np.full((3, 300 - HORIZON), 100.0)
    dates = DATES[: 300 - HORIZON]
    future = DATES[300 - HORIZON:]
    full = np.full((3, 300), 100.0)
    return history, dates, future, full


def test_best_blend_weight() -> None:
    actual = np.full((2, 10), 100.0)
    assert best_blend_weight(actual, actual, actual * 1.3, 0.1) == 1.0  # the model is exact
    assert best_blend_weight(actual, actual * 1.3, actual, 0.1) == 0.0  # the baseline is exact
    assert best_blend_weight(actual, actual, actual, 0.1) == 1.0  # ties go to the boosting model
    mixed = best_blend_weight(actual, actual * 1.2, actual * 0.8, 0.1)
    assert mixed == pytest.approx(0.5)  # equal and opposite errors cancel at 50/50


def test_calibration_and_blending_use_only_the_last_horizon_days() -> None:
    _, _, future, full = _setup(105.0)
    gbm, baseline = CountingConstant(105.0), CountingConstant(100.0)
    stack = StackedModels(gbm, baseline, TuneConfig())
    dates_all = DATES[:300 - HORIZON]
    # the final 28 history days are the inner window; the forecast target follows them
    history = full[:, : 300 - HORIZON]
    future_dates = DATES[300 - HORIZON:]
    out = {kind: stack.variant(kind)(history, dates_all, future_dates) for kind in VARIANTS}
    c = stack.components(history, dates_all, future_dates)
    assert c.factor == pytest.approx(100 / 105)  # the model was 5% high on the inner window
    assert c.weight_raw == 0.0 and c.weight_calibrated == 1.0  # exact baseline vs exact calibrated model
    assert np.allclose(out["raw"], 105.0) and np.allclose(out["calibrated"], 100.0)
    assert np.allclose(out["blend"], 100.0) and np.allclose(out["calibrated_blend"], 100.0)
    assert gbm.calls == 2  # one inner fit and one final fit, shared by all four variants
    assert len(stack.diagnostics) == 1 and stack.diagnostics[0]["history_days"] == 300 - HORIZON


def test_calibration_factor_is_clipped() -> None:
    history = np.full((2, 272), 100.0)
    stack = StackedModels(CountingConstant(200.0), CountingConstant(100.0), TuneConfig())
    stack.variant("raw")(history, DATES[:272], DATES[272:])
    assert stack.components(history, DATES[:272], DATES[272:]).factor == pytest.approx(0.9)  # lower clip


def test_unknown_variant_and_short_history() -> None:
    stack = StackedModels(CountingConstant(1.0), CountingConstant(1.0), TuneConfig())
    with pytest.raises(ValueError, match="variant must be"):
        stack.variant("magic")
    with pytest.raises(ValueError, match="longer than"):
        stack.variant("raw")(np.ones((1, 10)), DATES[:10], pd.date_range("2015-01-11", periods=10))


def test_tune_config_validation() -> None:
    with pytest.raises(ValueError, match="search"):
        TuneConfig(search="grid")
    with pytest.raises(ValueError, match="calibration_clip"):
        TuneConfig(calibration_clip=(1.1, 1.2))
    with pytest.raises(ValueError, match="blend_step"):
        TuneConfig(blend_step=0.0)
    with pytest.raises(ValueError, match="n_trials"):
        TuneConfig(n_trials=0)
