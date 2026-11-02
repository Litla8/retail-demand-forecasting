"""Step 2 pipeline: load -> validate -> clean -> complete grid -> calendar -> save.

Run:  python -m forecast.prepare
Re-running overwrites the outputs with identical content (idempotent).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from forecast.config import ForecastConfig
from forecast.data import (
    RawFrames,
    build_full_grid,
    clean_test,
    clean_train,
    load_raw,
    validate_submission,
    validate_test,
    validate_train,
)
from forecast.features import add_calendar_features
from forecast.logging_utils import configure_logging
from forecast.splits import make_validation_windows

logger = logging.getLogger(__name__)


def build_prepared_frame(raw: RawFrames, cfg: ForecastConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate and transform the raw files into the prepared store-item-day table.

    Returns:
        The prepared frame and a JSON-serialisable data-quality report.

    Raises:
        ValueError: If any validation fails.
    """
    keys = list(cfg.granularity)
    validate_train(raw.train, cfg)
    train, stats = clean_train(raw.train, cfg)
    test = clean_test(raw.test, cfg)
    validate_test(test, train, cfg)
    validate_submission(raw.submission, test, cfg)

    grid = build_full_grid(train, test, cfg)
    n_series = int(grid[keys].drop_duplicates().shape[0])
    n_days = int(grid[cfg.date_col].nunique())
    if len(grid) != n_series * n_days:
        raise ValueError("Prepared grid is not a complete series x day table")
    # Lag and rolling features need every series sorted by date.
    if not grid.groupby(keys)[cfg.date_col].is_monotonic_increasing.all():
        raise ValueError("Prepared grid is not sorted by date within each series")

    grid = add_calendar_features(grid, cfg.date_col)
    windows = make_validation_windows(train[cfg.date_col].min(), train[cfg.date_col].max(), cfg)

    report: Dict[str, Any] = {
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "rows_prepared": int(len(grid)),
        "series": n_series,
        "days_total": n_days,
        "train_dates": [train[cfg.date_col].min().date().isoformat(),
                        train[cfg.date_col].max().date().isoformat()],
        "test_dates": [test[cfg.date_col].min().date().isoformat(),
                       test[cfg.date_col].max().date().isoformat()],
        "missing_days_in_train": int(grid["is_missing_day"].sum()),
        "negative_sales_rows": int(stats["negative_sales_rows"]),
        "zero_sales_rows": int(stats["zero_sales_rows"]),
        "train_memory_mb_raw": round(stats["memory_mb_before"], 2),
        "train_memory_mb_downcast": round(stats["memory_mb_after"], 2),
        "prepared_memory_mb": round(float(grid.memory_usage(deep=True).sum() / 1e6), 2),
        "sales_min": float(np.nanmin(grid[cfg.target_col])),
        "sales_max": float(np.nanmax(grid[cfg.target_col])),
        "validation_windows": [w.as_dict() for w in windows],
        "columns": list(grid.columns),
        "dtypes": {c: str(t) for c, t in grid.dtypes.items()},
    }
    if report["missing_days_in_train"]:
        logger.warning("%d series-days are missing from train and left as NaN",
                       report["missing_days_in_train"])
    logger.info("Prepared %d rows (%d series x %d days); train memory %.1f MB -> %.1f MB",
                len(grid), n_series, n_days, report["train_memory_mb_raw"],
                report["train_memory_mb_downcast"])
    return grid, report


def run_prepare(cfg: ForecastConfig) -> Dict[str, Any]:
    """Run Step 2 end to end and write the parquet table and JSON report."""
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    grid, report = build_prepared_frame(load_raw(cfg.paths), cfg)
    grid.to_parquet(cfg.paths.prepared_parquet, index=False)
    cfg.paths.quality_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", cfg.paths.prepared_parquet, cfg.paths.quality_report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prepare the Store Item Demand data.")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Folder holding train.csv, test.csv, sample_submission.csv")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    cfg = ForecastConfig()
    paths = cfg.paths
    if args.raw_dir:
        paths = replace(paths, train_csv=args.raw_dir / "train.csv", test_csv=args.raw_dir / "test.csv",
                        sample_submission_csv=args.raw_dir / "sample_submission.csv")
    if args.processed_dir:
        paths = replace(paths, processed_dir=args.processed_dir)
    if args.reports_dir:
        paths = replace(paths, reports_dir=args.reports_dir)
    configure_logging(args.log_level)
    run_prepare(replace(cfg, paths=paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
