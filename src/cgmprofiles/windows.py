"""Non-overlapping CGM windows (Methods 2.3, temporal reproducibility).

Windows are half-open in elapsed time from the participant's first row, so adjacent windows
share no reading: days 1 to 3 = [0, 72) h and days 4 to 6 = [72, 144) h. A participant is
eligible when each window holds at least 70% of its expected 5-minute readings. This tests
reproducibility within the monitoring period, not long-term stability.

The eligibility and transition counts are masked together (linked cells) because they share
margins; a statistic computed on a masked n is rounded before it is tracked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agreement import cross_table
from .cgm_features import READINGS_PER_HOUR
from .config import get_config
from .disclosure import MaskProblem
from .io import GLUCOSE, PID

CFG = get_config().temporal
WINDOW_NAMES: list[str] = list(CFG.window_names)
ROW_ALL, ROW_TRAIN, ROW_HELDOUT = "all", "train", "held-out (val + test)"
STATUS_COLUMNS = ["n", "eligible", "not_eligible", "fail_A_only", "fail_B_only", "fail_both"]
GROUPS = [1, 2, 3]
SETS = {"held-out": ROW_HELDOUT, "all eligible": ROW_ALL}


def window_bounds(cgm: pd.DataFrame, design: str) -> pd.DataFrame:
    """Per participant: start and end hours of windows A and B."""
    last = cgm.groupby(PID, sort=False)["elapsed_h"].max()
    out = pd.DataFrame(index=last.index)
    spec = CFG.designs[design]
    for name in WINDOW_NAMES:
        out[f"{name}_start"], out[f"{name}_end"] = spec[name]
    return out


def window_rows(cgm: pd.DataFrame, bounds: pd.DataFrame, window: str) -> np.ndarray:
    """Boolean mask over CGM rows inside one half-open window."""
    start = cgm[PID].map(bounds[f"{window}_start"]).to_numpy(float)
    end = cgm[PID].map(bounds[f"{window}_end"]).to_numpy(float)
    t = cgm["elapsed_h"].to_numpy(float)
    return (t >= start) & (t < end)


def eligibility(cgm: pd.DataFrame, design: str) -> pd.DataFrame:
    """Valid readings and coverage per window; eligible = both windows pass 70%."""
    bounds = window_bounds(cgm, design)
    valid = cgm[GLUCOSE].notna().to_numpy()
    out = pd.DataFrame(index=bounds.index)
    for name in WINDOW_NAMES:
        counts = pd.Series(valid & window_rows(cgm, bounds, name)).groupby(cgm[PID].to_numpy()).sum()
        out[f"{name}_valid"] = counts.reindex(out.index).fillna(0).astype(int)
        expected = (bounds[f"{name}_end"] - bounds[f"{name}_start"]) * READINGS_PER_HOUR
        out[f"{name}_passes"] = out[f"{name}_valid"] / expected >= CFG.min_window_coverage
    out["eligible"] = out.A_passes & out.B_passes
    return out.rename_axis(PID).reset_index()


def _status_counts(sub: pd.DataFrame) -> dict[str, int]:
    return {"n": len(sub), "eligible": int(sub.eligible.sum()), "not_eligible": int((~sub.eligible).sum()),
            "fail_A_only": int((~sub.A_passes & sub.B_passes).sum()), "fail_B_only": int((sub.A_passes & ~sub.B_passes).sum()),
            "fail_both": int((~sub.A_passes & ~sub.B_passes).sum())}


def eligibility_rows(frame: pd.DataFrame, site_column: str) -> list[tuple[str, str, pd.DataFrame]]:
    """Rows of the eligibility table: overall, profile, split, held-out, site."""
    rows = [("overall", ROW_ALL, frame)]
    rows += [("profile", level, frame[frame.profile == level]) for level in ["C1", "C2", "C3"]]
    rows += [("split", level, frame[frame.split == level]) for level in ["train", "val", "test"]]
    rows.append(("split", ROW_HELDOUT, frame[frame.split.isin(["val", "test"])]))
    rows += [("site", level, frame[frame[site_column] == level]) for level in sorted(frame[site_column].dropna().unique())]
    return rows


def eligibility_problem(frame: pd.DataFrame, site_column: str, upper_bound: int) -> tuple[MaskProblem, list[tuple[str, str]]]:
    """Linked masking problem for the eligibility table (the training-eligible n is kept shown when possible)."""
    problem = MaskProblem(upper_bound=upper_bound)
    rows = eligibility_rows(frame, site_column)
    for _, level, sub in rows:
        counts = _status_counts(sub)
        for column in STATUS_COLUMNS:
            problem.add((level, column), counts[column], protected=(level, column) == (ROW_TRAIN, "eligible"), fixed=column == "n")
        problem.equal((level, "n"), [(level, "eligible"), (level, "not_eligible")])
        problem.equal((level, "not_eligible"), [(level, c) for c in STATUS_COLUMNS[3:]])
    groups = {g: [lv for gg, lv, _ in rows if gg == g] for g in ("profile", "site")}
    for column in STATUS_COLUMNS:
        problem.equal((ROW_ALL, column), [(lv, column) for lv in groups["profile"]])
        problem.equal((ROW_ALL, column), [(lv, column) for lv in groups["site"]])
        problem.equal((ROW_ALL, column), [(lv, column) for lv in ("train", "val", "test")])
        problem.equal((ROW_ALL, column), [(ROW_TRAIN, column), (ROW_HELDOUT, column)])
        problem.equal((ROW_HELDOUT, column), [("val", column), ("test", column)])
    return problem, [(g, lv) for g, lv, _ in rows]


def add_transition_cells(problem: MaskProblem, labels_a: np.ndarray, labels_b: np.ndarray, is_train: np.ndarray) -> None:
    """Window A x window B group tables (held-out and all eligible) and training group sizes,
    linked to the eligibility cells so that masked n's stay masked."""
    heldout = ~is_train
    for set_name, row_key in SETS.items():
        member = heldout if set_name == "held-out" else np.ones(len(labels_a), bool)
        t = cross_table(labels_a[member], labels_b[member], len(GROUPS))
        n_masked = problem.is_masked((row_key, "eligible"))
        for i in GROUPS:
            for j in GROUPS:
                problem.add(("tr", set_name, i, j), t[i - 1, j - 1])
        for g in GROUPS:
            problem.add(("rowA", set_name, g), t[g - 1].sum())
            problem.add(("colB", set_name, g), t[:, g - 1].sum())
            problem.equal(("rowA", set_name, g), [("tr", set_name, g, j) for j in GROUPS])
            problem.equal(("colB", set_name, g), [("tr", set_name, i, g) for i in GROUPS])
        problem.equal((row_key, "eligible"), [("rowA", set_name, g) for g in GROUPS])
        problem.equal((row_key, "eligible"), [("colB", set_name, g) for g in GROUPS])
        if n_masked:  # percent agreement = diagonal / n, so the diagonal total must not be exact
            problem.add(("diag", set_name), np.trace(t), hidden=True, sensitive=True)
            problem.equal(("diag", set_name), [("tr", set_name, g, g) for g in GROUPS])
    for window, labels in (("A", labels_a), ("B", labels_b)):
        for g in GROUPS:
            problem.add((f"train{window}", g), int((is_train & (labels == g)).sum()))
            key_all = ("rowA" if window == "A" else "colB", "all eligible", g)
            key_heldout = ("rowA" if window == "A" else "colB", "held-out", g)
            problem.equal(key_all, [(f"train{window}", g), key_heldout])
        problem.equal((ROW_TRAIN, "eligible"), [(f"train{window}", g) for g in GROUPS])
