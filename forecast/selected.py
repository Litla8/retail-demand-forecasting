"""Rebuild the forecaster selected in Step 7 from ``models/gbm_params.json``."""
from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict

from forecast.baselines import Forecaster
from forecast.config import ForecastConfig, GbmConfig
from forecast.ensemble import StackedModels
from forecast.gbm import GbmForecaster


def load_selected(path: Path) -> Dict[str, Any]:
    """Read the saved Step 7 selection.

    Raises:
        FileNotFoundError: If Step 7 has not been run.
        ValueError: If the file lacks the expected keys.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m forecast.run_tuning` first.")
    selected = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("variant", "gbm", "calibrate", "blend") if k not in selected]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return selected


def gbm_config_from(selected: Dict[str, Any], base: GbmConfig) -> GbmConfig:
    """GbmConfig with the saved values (JSON lists are turned back into tuples)."""
    known = {f.name for f in fields(GbmConfig)}
    values = {k: (tuple(v) if isinstance(v, list) else v) for k, v in selected["gbm"].items() if k in known}
    return replace(base, **values)


def build_selected_forecaster(selected: Dict[str, Any], cfg: ForecastConfig, baseline: Forecaster) -> Forecaster:
    """The forecaster for the selected variant (plain, calibrated, blended or both)."""
    gbm = GbmForecaster(gbm_config_from(selected, cfg.gbm), cfg.horizon_days, cfg.random_state)
    calibrate, blend = bool(selected["calibrate"]), bool(selected["blend"])
    if not calibrate and not blend:
        return gbm
    kind = {(True, False): "calibrated", (False, True): "blend", (True, True): "calibrated_blend"}[(calibrate, blend)]
    return StackedModels(gbm, baseline, cfg.tune).variant(kind)
