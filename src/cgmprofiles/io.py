"""Read the AI-READI parquet and split file (read-only) into analysis frames.

Rules (Methods 2.1 and 2.2): the cohort is the 1,591 participants in the split file; the
stratum is the AI-READI enrolment group; a laboratory value dated after the participant's
last CGM timestamp is set to missing before TG/HDL (mg/dL over mg/dL) is derived.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import data_path, get_config

CFG = get_config()
COL = CFG.columns
PID = COL.participant_id
TIMESTAMP = COL.timestamp
GLUCOSE = COL.glucose
FACTORS: list[str] = list(CFG.profiles.factors)
STRATA: list[str] = list(CFG.cohort.strata)
PRIMARY = CFG.cohort.primary
SPLIT_RANK = {name: i for i, name in enumerate(CFG.cohort.split_order)}
NO_DRUG_FLAG = "no_diabetes_drug_flag"


def read_columns(columns: list[str]) -> pd.DataFrame:
    """Selected parquet columns, participant ID as string."""
    table = pq.read_table(data_path("parquet"), columns=columns).to_pandas()
    table[PID] = table[PID].astype(str)
    return table


def load_split() -> pd.DataFrame:
    """Split file with its row order (which fixes fitting and bootstrap order)."""
    split = pd.read_csv(data_path("split"), dtype={PID: str})
    split["split_file_row"] = np.arange(len(split))
    return split[[PID, "split", "split_file_row"]]


def load_clinical_frame() -> pd.DataFrame:
    """One row per participant: six profile factors after the post-CGM laboratory rule, stratum, CGM end."""
    lab_dates = CFG.columns.lab_dates.raw
    raw_factors = [f for f in FACTORS if f != "tg_hdl_ratio"]
    columns = [PID, COL.study_group, TIMESTAMP, *raw_factors, COL.triglycerides, COL.hdl, *lab_dates.values()]
    columns = list(dict.fromkeys(columns))
    table = read_columns(columns)
    static = [c for c in columns if c not in (PID, TIMESTAMP)]
    varying = table.groupby(PID)[static].nunique(dropna=False).max()
    if (varying > 1).any():
        raise AssertionError(f"Static columns vary within participant: {varying[varying > 1].to_dict()}")
    cgm_end = pd.to_datetime(table[TIMESTAMP], utc=True).groupby(table[PID]).max().rename("cgm_end")
    frame = table.drop(columns=[TIMESTAMP]).groupby(PID, sort=False).first().join(cgm_end).reset_index()
    for value_column, date_column in lab_dates.items():
        dates = pd.to_datetime(frame[date_column], utc=True)
        frame.loc[dates.notna() & (dates > frame["cgm_end"]), value_column] = np.nan
    valid_ratio = frame[COL.hdl].gt(CFG.cohort.min_valid_hdl) & frame[COL.triglycerides].notna()
    frame["tg_hdl_ratio"] = np.where(valid_ratio, frame[COL.triglycerides] / frame[COL.hdl], np.nan)
    frame["canonical_stratum"] = frame[COL.study_group].map(CFG.cohort.study_group_to_stratum.raw)
    return frame


def load_cgm(participant_ids: list[str]) -> pd.DataFrame:
    """CGM rows sorted by participant and time, with hours elapsed since the participant's first row."""
    table = read_columns([PID, TIMESTAMP, GLUCOSE])
    table = table[table[PID].isin(set(participant_ids))]
    table = table.sort_values([PID, TIMESTAMP], kind="mergesort").reset_index(drop=True)
    first_row = table.groupby(PID)[TIMESTAMP].transform("min")
    table["elapsed_h"] = (table[TIMESTAMP] - first_row).dt.total_seconds() / CFG.cgm.seconds_per_hour
    return table


def load_static(columns: list[str]) -> pd.DataFrame:
    """First value per participant of static parquet columns, indexed by participant."""
    table = read_columns([PID, *columns])
    return table.groupby(PID).first()


def load_timezones(participant_ids: list[str]) -> pd.Series:
    """Each participant's own time zone (UAB participants are America/Chicago)."""
    return load_static([COL.timezone])[COL.timezone].loc[participant_ids]


def load_medications(participant_ids: list[str]) -> pd.DataFrame:
    """Ever-recorded medication flags (0/1) plus a no-diabetes-drug flag, in the given order."""
    flags = list(CFG.columns.medications)
    table = load_static([*flags, COL.any_diabetes_drug]).fillna(0).gt(0).astype(int)
    table[NO_DRUG_FLAG] = 1 - table[COL.any_diabetes_drug]
    return table.loc[participant_ids].reset_index(drop=True)


def ordered_stratum(labels: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """One stratum's participants in the analysis order: train, val, test, each in split-file order."""
    sub = labels[labels.canonical_stratum == stratum]
    return sub.sort_values(["split_rank", "split_file_row"]).reset_index(drop=True)
