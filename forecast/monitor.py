"""Monthly accuracy check: how did earlier forecasts do on the actuals that arrived later?

Run:  python -m forecast.monitor            (evaluate every archived forecast)
      python -m forecast.monitor --archive  (first save the current forecast to reports/archive/)
The exit code is 2 when any archived forecast breaches the alert thresholds, so a scheduler can flag it.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from forecast.config import ForecastConfig, MonitorConfig
from forecast.data import load_prepared
from forecast.logging_utils import configure_logging
from forecast.wide import WideData, to_wide

logger = logging.getLogger(__name__)
ARCHIVE_PATTERN = "forecast_intervals_*.csv"


def archive_forecast(cfg: ForecastConfig) -> Path:
    """Copy the current forecast to the archive, named after its first forecast day (idempotent).

    Raises:
        FileNotFoundError: If there is no current forecast yet.
    """
    source = Path(cfg.paths.forecast_intervals)
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found; run `python -m forecast.run_business` first")
    first_day = pd.read_csv(source, usecols=["date"], parse_dates=["date"])["date"].min().date().isoformat()
    cfg.paths.archive_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.paths.archive_dir / f"forecast_intervals_{first_day}.csv"
    shutil.copy(source, target)
    return target


def evaluate_archived(forecast_csv: Path, wide: WideData, expected_wape: float, cfg: MonitorConfig) -> Optional[Dict[str, Any]]:
    """Score one archived forecast on the actuals that are now available.

    Returns:
        None if fewer than ``min_days`` of the forecast period have actuals yet; otherwise the realised
        WAPE, bias, interval coverage and whether an alert threshold was breached.
    """
    forecast = pd.read_csv(forecast_csv, parse_dates=["date"])
    rows = wide.series.get_indexer(pd.MultiIndex.from_arrays([forecast["store"], forecast["item"]]))
    cols = wide.dates.get_indexer(pd.DatetimeIndex(forecast["date"]))
    known = (rows >= 0) & (cols >= 0)
    if known.sum() == 0 or forecast.loc[known, "date"].nunique() < cfg.min_days:
        return None
    seen = forecast[known]
    actual = wide.values[rows[known], cols[known]]
    predicted = seen["forecast"].to_numpy()
    wape = float(100 * np.abs(predicted - actual).sum() / actual.sum())
    covered = {level: float(np.mean((actual >= seen[f"lower_{level}"]) & (actual <= seen[f"upper_{level}"])))
               for level in (80, 95) if f"lower_{level}" in seen}
    reasons = []
    if wape > expected_wape * (1 + cfg.wape_tolerance):
        reasons.append(f"WAPE {wape:.1f}% is more than {100 * cfg.wape_tolerance:.0f}% above the backtest {expected_wape:.1f}%")
    if covered.get(95, 1.0) < cfg.min_coverage_95:
        reasons.append(f"95% interval covered only {100 * covered[95]:.0f}% of actuals")
    return {"archive": Path(forecast_csv).name, "first_day": seen["date"].min().date().isoformat(),
            "last_realized_day": seen["date"].max().date().isoformat(), "realized_days": int(seen["date"].nunique()),
            "wape": round(wape, 3), "bias_pct": round(float(100 * (predicted - actual).sum() / actual.sum()), 3),
            "coverage_80": round(covered.get(80, float("nan")), 4), "coverage_95": round(covered.get(95, float("nan")), 4),
            "expected_wape": round(float(expected_wape), 3), "alert": bool(reasons), "reasons": "; ".join(reasons)}


def run_monitor(cfg: ForecastConfig, df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """Evaluate every archived forecast and rewrite the monitoring log (stateless, so re-runs are identical).

    Raises:
        FileNotFoundError: If the selected-model file with the backtest WAPE is missing.
    """
    params = Path(cfg.paths.gbm_params)
    if not params.is_file():
        raise FileNotFoundError(f"{params} not found. Run `python -m forecast.run_tuning` first.")
    expected = float(json.loads(params.read_text(encoding="utf-8"))["mean_wape"])
    wide = to_wide(load_prepared(cfg.paths.prepared_parquet) if df is None else df, cfg)
    results = []
    for path in sorted(Path(cfg.paths.archive_dir).glob(ARCHIVE_PATTERN)):
        result = evaluate_archived(path, wide, expected, cfg.monitor)
        if result is None:
            logger.info("%s: not enough realised days yet", path.name)
            continue
        logger.log(logging.WARNING if result["alert"] else logging.INFO, "%s: WAPE %.2f%% coverage95 %.2f %s",
                   path.name, result["wape"], result["coverage_95"], result["reasons"])
        results.append(result)
    cfg.paths.monitoring_log.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(cfg.paths.monitoring_log, index=False)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns 2 when an alert was raised."""
    parser = argparse.ArgumentParser(description="Check earlier forecasts against the actuals that arrived later.")
    parser.add_argument("--archive", action="store_true", help="Save the current forecast to the archive and exit")
    parser.add_argument("--prepared-path", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    cfg = ForecastConfig()
    if args.reports_dir:
        cfg = replace(cfg, paths=replace(cfg.paths, reports_dir=args.reports_dir))
    configure_logging(args.log_level)
    if args.archive:
        logger.info("Archived to %s", archive_forecast(cfg))
        return 0
    results = run_monitor(cfg, load_prepared(args.prepared_path or cfg.paths.prepared_parquet))
    return 2 if any(r["alert"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
