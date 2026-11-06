"""Tests for forecast.tuning."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast.config import ForecastConfig, GbmConfig
from forecast.data import RawFrames
from forecast.prepare import build_prepared_frame
from forecast.splits import make_validation_windows
from forecast.tuning import SEARCH_SPACE, default_params, random_params, score_params, tune_gbm
from forecast.wide import to_wide


@pytest.fixture
def setup(raw_frames: RawFrames, cfg: ForecastConfig):
    small = replace(cfg, gbm=replace(GbmConfig(engine="hist_gb"), n_estimators=20, min_child_samples=20))
    wide = to_wide(build_prepared_frame(raw_frames, small)[0], small)
    windows = make_validation_windows(wide.dates[0], wide.dates[-1],
                                      replace(small, validation_window_ends=("2015-03-31", "2015-05-31")))
    return small, wide, windows


def test_random_params_stay_in_the_search_space_and_are_repeatable() -> None:
    first = [random_params(np.random.default_rng(7)) for _ in range(2)]
    assert first[0] == first[1]  # same seed, same draw
    rng = np.random.default_rng(42)
    for _ in range(50):
        params = random_params(rng)
        for name, (low, high, kind) in SEARCH_SPACE.items():
            assert low <= params[name] <= high, name
            assert isinstance(params[name], int) == kind.startswith("int"), name


def test_default_params_cover_the_search_space() -> None:
    assert set(default_params(GbmConfig())) == set(SEARCH_SPACE)
    assert default_params(GbmConfig())["num_leaves"] == 31


def test_score_params_is_finite_and_repeatable(setup) -> None:
    cfg, wide, windows = setup
    params = default_params(cfg.gbm)
    first = score_params(wide, windows, cfg.gbm, params, cfg.horizon_days, 42)
    assert np.isfinite(first) and 0 < first < 100
    assert first == score_params(wide, windows, cfg.gbm, params, cfg.horizon_days, 42)


def test_random_search_always_includes_the_default_and_never_gets_worse(setup) -> None:
    cfg, wide, windows = setup
    result = tune_gbm(wide, windows, cfg, n_trials=3, search="random")
    assert len(result.trials) == 3 and result.search == "random"
    assert {k: result.trials[0][k] for k in SEARCH_SPACE} == default_params(cfg.gbm)  # trial 0 is the default
    assert result.best_value <= result.default_value and set(result.best_params) == set(SEARCH_SPACE)
    again = tune_gbm(wide, windows, cfg, n_trials=3, search="random")
    assert again.best_params == result.best_params and again.best_value == result.best_value
    assert result.as_dict()["n_trials"] == 3 and "trials" not in result.as_dict()
    with pytest.raises(ValueError, match="n_trials"):
        tune_gbm(wide, windows, cfg, n_trials=0, search="random")


def test_optuna_search_enqueues_the_default(setup) -> None:
    pytest.importorskip("optuna")
    cfg, wide, windows = setup
    result = tune_gbm(wide, windows, cfg, n_trials=3, search="optuna")
    assert len(result.trials) == 3 and result.search == "optuna"
    assert {k: result.trials[0][k] for k in SEARCH_SPACE} == default_params(cfg.gbm)
    assert result.best_value <= result.default_value
