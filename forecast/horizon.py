"""Regenerate test.csv and sample_submission.csv for the 90 days after the newest history.

The pipeline validates that the test period starts the day after the last training date. When new rows are
appended to train.csv, this keeps those two files consistent with it (and leaves the original Kaggle files
untouched when they already match).
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from forecast.config import ForecastConfig
from forecast.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def horizon_frame(train: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """The (id, date, store, item) rows for the ``horizon_days`` after the last training date, date-major.

    Raises:
        ValueError: If the training table lacks the required columns or is empty.
    """
    keys = list(cfg.granularity)
    missing = [c for c in (cfg.date_col, *keys) if c not in train.columns]
    if missing or train.empty:
        raise ValueError(f"train needs the columns {[cfg.date_col, *keys]} and at least one row; missing {missing}")
    last = pd.to_datetime(train[cfg.date_col]).max()
    series = train[keys].drop_duplicates().sort_values(keys)
    days = pd.DataFrame({cfg.date_col: pd.date_range(last + pd.Timedelta(days=1), periods=cfg.horizon_days, freq="D")})
    frame = days.merge(series, how="cross").sort_values([cfg.date_col, *keys]).reset_index(drop=True)
    frame.insert(0, cfg.id_col, np.arange(len(frame)))
    return frame


def matches(existing: pd.DataFrame, wanted: pd.DataFrame, cfg: ForecastConfig) -> bool:
    """True if an existing test file already covers exactly the wanted dates and series."""
    columns = [cfg.date_col, *cfg.granularity]
    if not set(columns) <= set(existing.columns) or len(existing) != len(wanted):
        return False
    left = existing[columns].assign(**{cfg.date_col: pd.to_datetime(existing[cfg.date_col])})
    right = wanted[columns]
    return left.sort_values(columns).reset_index(drop=True).equals(right.sort_values(columns).reset_index(drop=True))


def make_horizon_files(cfg: ForecastConfig) -> bool:
    """Write test.csv and sample_submission.csv if they do not already fit the history.

    Returns:
        True if the files were (re)written, False if they were already consistent.

    Raises:
        FileNotFoundError: If train.csv is missing.
    """
    paths = cfg.paths
    if not Path(paths.train_csv).is_file():
        raise FileNotFoundError(f"{paths.train_csv} not found")
    wanted = horizon_frame(pd.read_csv(paths.train_csv), cfg)
    if Path(paths.test_csv).is_file() and matches(pd.read_csv(paths.test_csv), wanted, cfg):
        logger.info("%s already covers %s to %s; left unchanged", paths.test_csv, wanted[cfg.date_col].min().date(),
                    wanted[cfg.date_col].max().date())
        return False
    wanted.assign(**{cfg.date_col: wanted[cfg.date_col].dt.strftime("%Y-%m-%d")}).to_csv(paths.test_csv, index=False)
    pd.DataFrame({cfg.id_col: wanted[cfg.id_col], cfg.target_col: 0}).to_csv(paths.sample_submission_csv, index=False)
    logger.info("Wrote %s and %s for %s to %s", paths.test_csv, paths.sample_submission_csv,
                wanted[cfg.date_col].min().date(), wanted[cfg.date_col].max().date())
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate the test and sample-submission files after a data refresh.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    cfg = ForecastConfig()
    if args.raw_dir:
        cfg = replace(cfg, paths=replace(cfg.paths, train_csv=args.raw_dir / "train.csv", test_csv=args.raw_dir / "test.csv",
                                         sample_submission_csv=args.raw_dir / "sample_submission.csv"))
    configure_logging(args.log_level)
    make_horizon_files(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
