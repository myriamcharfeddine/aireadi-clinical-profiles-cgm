"""Section 3.1 CGM outcomes: mean glucose, time in range (70 to 180 mg/dL) and coefficient of variation.

Windows start at the participant's first parquet row: the first 48 hours (elapsed time <= 48 h,
inclusive boundary) and the full record (every valid reading). SD uses ddof 1 and needs at
least two readings; CV = SD / mean x 100; TIR is the share of valid readings within 70 to 180
mg/dL, both bounds inclusive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import get_config
from .io import GLUCOSE, PID

CFG = get_config().cgm
WINDOWS: list[str] = list(CFG.windows.raw)
MEAN, TIR, CV = CFG.metrics
METRICS: list[str] = list(CFG.metrics)


def window_metrics(cgm: pd.DataFrame, window_hours: float | None) -> pd.DataFrame:
    """Per participant: valid readings, mean, SD, CV (%) and TIR (%) within the window."""
    if window_hours is None:
        selected = cgm
    else:
        elapsed = cgm["elapsed_h"]
        selected = cgm[(elapsed <= window_hours) & elapsed.ge(0)]
    valid = selected[selected[GLUCOSE].notna()]
    grouped = valid.groupby(PID)[GLUCOSE]
    out = pd.DataFrame({"n_readings": grouped.size(), MEAN: grouped.mean(), "sd_glucose": grouped.std(ddof=CFG.sd_ddof)})
    low, high = CFG.tir_mgdl
    out[TIR] = valid[GLUCOSE].between(low, high).groupby(valid[PID]).mean() * 100.0
    out.loc[out.n_readings < CFG.min_readings_for_metric, "sd_glucose"] = np.nan
    out[CV] = out["sd_glucose"] / out[MEAN] * 100.0
    out = out.reindex(cgm[PID].unique())
    out["n_readings"] = out["n_readings"].fillna(0).astype(int)
    return out.rename_axis(PID).reset_index()


def participant_metrics(labels: pd.DataFrame, cgm: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per participant and window, with the label columns attached."""
    frames = []
    for window, hours in CFG.windows.raw.items():
        metrics = window_metrics(cgm, hours)
        metrics["window"] = window
        frames.append(labels.merge(metrics, on=PID, how="left"))
    return pd.concat(frames, ignore_index=True)
