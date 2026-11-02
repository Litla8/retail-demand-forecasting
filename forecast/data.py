"""Loading, validating and cleaning the Kaggle Store Item Demand files.

The dataset ships as three CSVs and has no calendar, price or promotion tables, so there is
nothing to merge: calendar features are derived from the date (see ``forecast.features``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from forecast.config import ForecastConfig, PathsConfig

logger = logging.getLogger(__name__)


class RawFrames(NamedTuple):
    """The three raw competition files."""

    train: pd.DataFrame
    test: pd.DataFrame
    submission: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV, failing with a clear message if it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path.resolve()}")
    return pd.read_csv(path)


def load_raw(paths: PathsConfig) -> RawFrames:
    """Read train.csv, test.csv and sample_submission.csv.

    Raises:
        FileNotFoundError: If any file is missing.
    """
    frames = RawFrames(_read_csv(paths.train_csv), _read_csv(paths.test_csv),
                       _read_csv(paths.sample_submission_csv))
    logger.info("Loaded train %s, test %s, submission %s",
                frames.train.shape, frames.test.shape, frames.submission.shape)
    return frames


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    """Raise ValueError listing any required columns that are absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def downcast_integers(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy with the given integer columns shrunk to the smallest safe int dtype.

    Raises:
        ValueError: If a column is not an integer column.
    """
    out = df.copy()
    for column in columns:
        if not is_integer_dtype(out[column]):
            raise ValueError(f"Column {column!r} must be integer typed, got {out[column].dtype}")
        out[column] = pd.to_numeric(out[column], downcast="integer")
    return out


def validate_train(df: pd.DataFrame, cfg: ForecastConfig) -> None:
    """Check the raw training table: columns, dtypes, nulls and duplicate rows.

    Raises:
        ValueError: On any violation.
    """
    keys = [cfg.date_col, *cfg.granularity]
    _require_columns(df, [*keys, cfg.target_col], "train")
    if df[[*keys, cfg.target_col]].isna().any().any():
        raise ValueError("train contains missing values")
    for column in (*cfg.granularity, cfg.target_col):
        if not is_numeric_dtype(df[column]):
            raise ValueError(f"train column {column!r} must be numeric, got {df[column].dtype}")
    if df.duplicated(keys).any():
        raise ValueError("train contains duplicate (date, store, item) rows")


def clean_train(df: pd.DataFrame, cfg: ForecastConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Parse dates, apply the negative-sales policy, downcast types and sort.

    Zero sales are valid demand and are kept. Negative sales are impossible here, so the
    default policy raises; ``negative_sales="clip"`` sets them to zero instead.

    Returns:
        The cleaned frame sorted by series then date, and a dict of cleaning statistics.
    """
    keys = list(cfg.granularity)
    memory_before = df.memory_usage(deep=True).sum() / 1e6
    out = df.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="raise")

    negatives = int((out[cfg.target_col] < 0).sum())
    if negatives:
        if cfg.negative_sales == "raise":
            raise ValueError(f"train contains {negatives} negative sales rows")
        logger.warning("Clipping %d negative sales values to zero", negatives)
        out[cfg.target_col] = out[cfg.target_col].clip(lower=0)
    zeros = int((out[cfg.target_col] == 0).sum())
    if zeros:
        logger.info("Kept %d rows with zero sales (valid demand)", zeros)

    out = downcast_integers(out, [*keys, cfg.target_col])
    out = out.sort_values([*keys, cfg.date_col]).reset_index(drop=True)
    stats = {
        "negative_sales_rows": negatives,
        "zero_sales_rows": zeros,
        "memory_mb_before": float(memory_before),
        "memory_mb_after": float(out.memory_usage(deep=True).sum() / 1e6),
    }
    return out, stats


def clean_test(df: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """Parse dates, downcast types and sort the competition test table.

    Raises:
        ValueError: If required columns are missing or ids are not integers.
    """
    keys = list(cfg.granularity)
    _require_columns(df, [cfg.id_col, cfg.date_col, *keys], "test")
    out = df.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="raise")
    out = downcast_integers(out, [cfg.id_col, *keys])
    return out.sort_values([*keys, cfg.date_col]).reset_index(drop=True)


def validate_test(test: pd.DataFrame, train: pd.DataFrame, cfg: ForecastConfig) -> None:
    """Check the cleaned test table against the cleaned training table.

    The test period must start the day after training ends, be contiguous, last exactly
    ``horizon_days`` and cover exactly the same series as training.

    Raises:
        ValueError: On any violation.
    """
    keys = list(cfg.granularity)
    if test[cfg.id_col].duplicated().any():
        raise ValueError("test ids are not unique")
    if test.duplicated([cfg.date_col, *keys]).any():
        raise ValueError("test contains duplicate (date, store, item) rows")

    first, last = test[cfg.date_col].min(), test[cfg.date_col].max()
    n_days = test[cfg.date_col].nunique()
    if n_days != (last - first).days + 1:
        raise ValueError("test dates are not contiguous")
    if n_days != cfg.horizon_days:
        raise ValueError(f"test covers {n_days} days but horizon_days is {cfg.horizon_days}")
    if first != train[cfg.date_col].max() + pd.Timedelta(days=1):
        raise ValueError("test must start the day after the last training date")

    compared = train[keys].drop_duplicates().merge(
        test[keys].drop_duplicates(), how="outer", indicator=True)
    if (compared["_merge"] != "both").any():
        raise ValueError("train and test contain different store-item series")
    if (test.groupby(keys).size() != cfg.horizon_days).any():
        raise ValueError("some series do not have exactly horizon_days test rows")


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame, cfg: ForecastConfig) -> None:
    """Check that the sample submission lists exactly the test ids.

    Raises:
        ValueError: If columns or ids do not match.
    """
    _require_columns(submission, [cfg.id_col, cfg.target_col], "sample_submission")
    if submission[cfg.id_col].duplicated().any():
        raise ValueError("sample_submission ids are not unique")
    if set(submission[cfg.id_col]) != set(test[cfg.id_col]):
        raise ValueError("sample_submission ids do not match test ids")


def build_full_grid(train: pd.DataFrame, test: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """Build one complete series x day table covering training history plus the test horizon.

    Every series gets every calendar day, so lag and rolling features can rely on a regular
    index. Training days with no record stay NaN and are flagged ``is_missing_day`` instead
    of being silently filled with zeros. Test rows carry NaN sales and their competition id.

    Returns:
        Columns: date, store, item, sales (float32), id (nullable Int32), is_test,
        is_missing_day; sorted by series then date.
    """
    keys = list(cfg.granularity)
    date, target, id_col = cfg.date_col, cfg.target_col, cfg.id_col
    series = train[keys].drop_duplicates().sort_values(keys)
    calendar = pd.DataFrame({date: pd.date_range(train[date].min(), test[date].max(), freq="D")})

    grid = series.merge(calendar, how="cross")
    grid = grid.merge(train[[date, *keys, target]], on=[date, *keys], how="left", validate="one_to_one")
    grid = grid.merge(test[[date, *keys, id_col]], on=[date, *keys], how="left", validate="one_to_one")

    grid["is_test"] = grid[date] > train[date].max()
    grid["is_missing_day"] = ~grid["is_test"] & grid[target].isna()
    grid[target] = grid[target].astype("float32")  # float so test rows can hold NaN
    grid[id_col] = grid[id_col].astype("Int32")  # nullable: ids exist only for test rows
    grid = downcast_integers(grid, keys)
    return grid.sort_values([*keys, date]).reset_index(drop=True)


def load_prepared(path: Path) -> pd.DataFrame:
    """Read the prepared parquet written by Step 2.

    Raises:
        FileNotFoundError: If Step 2 has not been run yet.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {path}. Run `python -m forecast.prepare`.")
    return pd.read_parquet(path)
