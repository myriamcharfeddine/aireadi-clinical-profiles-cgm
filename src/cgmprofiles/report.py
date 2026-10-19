"""Writing outputs: tracked aggregate tables and numbers, restricted participant-level frames."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import restricted_dir, results_dir


def jsonable(value):
    """Plain JSON types (NaN and infinities become null)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_numbers(name: str, numbers: dict) -> Path:
    """results/numbers/<name>.json: every manuscript number of one analysis, unrounded."""
    path = results_dir("numbers") / f"{name}.json"
    path.write_text(json.dumps(jsonable(numbers), indent=2, sort_keys=True) + "\n")
    return path


def read_numbers(name: str) -> dict:
    return json.loads((results_dir("numbers") / f"{name}.json").read_text())


def write_aggregate(frame: pd.DataFrame, name: str) -> Path:
    """Tracked aggregate table (results/aggregate/<name>.csv); counts must already be masked."""
    path = results_dir("aggregate") / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def read_aggregate(name: str) -> pd.DataFrame:
    return pd.read_csv(results_dir("aggregate") / f"{name}.csv")


def write_restricted(frame: pd.DataFrame, name: str) -> Path:
    """Participant-level frame under restricted/ (gitignored), parquet so floats round-trip exactly."""
    path = restricted_dir() / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return path


def read_restricted(name: str) -> pd.DataFrame:
    path = restricted_dir() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the earlier analyses first (bash run_all.sh)")
    return pd.read_parquet(path)


class Timer:
    """Wall-clock seconds since creation."""

    def __init__(self):
        self.start = time.time()

    def seconds(self) -> float:
        return round(time.time() - self.start, 1)
