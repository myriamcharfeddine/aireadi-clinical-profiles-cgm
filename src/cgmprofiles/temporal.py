"""Temporal reproducibility (two non-overlapping windows): window features, agreement with the profiles per
window and the C2 contrast, with the record's seeds and draw order. Within the monitoring period only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from . import agreement as ag
from . import cgm_features as cf
from . import io, stats, windows as wn
from .cgm_metrics import METRICS, window_metrics
from .config import get_config
from .disclosure import MASK_COMPLEMENT, MASK_SMALL

CFG = get_config()
TCFG, SEEDS = CFG.temporal, CFG.seeds
N_BOOT, N_PERM = CFG.bootstrap.n_default, CFG.permutation.n_agreement
SITE, C2 = CFG.columns.site, CFG.profiles.c2
CONTRAST = "C2 minus rest"

def window_features(cgm: pd.DataFrame, design: str, zones: pd.Series, ids: list[str]) -> dict[str, pd.DataFrame]:
    """Extended features inside each window separately (runs and autocorrelations never cross a boundary);
    the 70% window coverage rule replaces the 1,000-reading floor of the full record."""
    bounds = wn.window_bounds(cgm, design)
    keep = cgm[io.PID].isin(set(ids)).to_numpy()
    out = {}
    for name in wn.WINDOW_NAMES:
        selected = cgm[keep & wn.window_rows(cgm, bounds, name)]
        records = [{io.PID: pid, **cf.participant_features(block, zones[pid], CFG.cgm.full_window, CFG.cgm.min_readings_for_metric)}
                   for pid, block in selected.groupby(io.PID, sort=False)]
        out[name] = pd.DataFrame(records).set_index(io.PID).loc[ids].reset_index()
    return out


def feature_gate(cgm: pd.DataFrame, design: str, ids: list[str], feats: dict[str, pd.DataFrame]) -> float:
    """Largest difference between window mean glucose, TIR and CV in the features and a direct recomputation."""
    bounds = wn.window_bounds(cgm, design)
    keep = cgm[io.PID].isin(set(ids)).to_numpy()
    worst = 0.0
    for name in wn.WINDOW_NAMES:
        direct = window_metrics(cgm[keep & wn.window_rows(cgm, bounds, name)], None).set_index(io.PID).loc[ids]
        saved = feats[name].set_index(io.PID).loc[ids]
        worst = max(worst, max(float((direct[m] - saved[m]).abs().max()) for m in METRICS))
    return worst


def t3_agreement(a: np.ndarray, b: np.ndarray, profile: np.ndarray, offset: int) -> dict:
    """Agreement of window A and window B CGM groups with C1 to C3, permutation nulls and paired bootstrap."""
    windows = {"A": a, "B": b}
    observed = {w: ag.agreement(g, profile) for w, g in windows.items()}
    rng = np.random.default_rng(SEEDS.t3_permutation + offset * 10)
    nulls = {w: ag.permutation_null(g, profile, rng, N_PERM) for w, g in windows.items()}
    rng = np.random.default_rng(SEEDS.t3_bootstrap + offset * 10)
    n = len(profile)
    draws = {w: {m: np.empty(N_BOOT) for m in ag.METRICS} for w in windows}
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        for w, g in windows.items():
            for m, v in ag.agreement(g[idx], profile[idx]).items():
                draws[w][m][i] = v
    rows = []
    for m in ag.METRICS:
        row = {"statistic": m}
        for w in windows:
            low, high = stats.percentile_ci(draws[w][m])
            row.update({f"{w}_value": observed[w][m], f"{w}_ci_low": low, f"{w}_ci_high": high, f"{w}_null_mean": float(nulls[w][m].mean()),
                        f"{w}_null_q975": float(np.quantile(nulls[w][m], 0.975)), f"{w}_p_permutation": stats.upper_p(nulls[w][m], observed[w][m])})
        low, high = stats.percentile_ci(draws["A"][m] - draws["B"][m])
        row.update({"difference_A_minus_B": observed["A"][m] - observed["B"][m], "difference_ci_low": low, "difference_ci_high": high})
        rows.append(row)
    return rows


def c2_contrast(cv_a: np.ndarray, cv_b: np.ndarray, cv_full: np.ndarray, profile: np.ndarray, offset: int) -> tuple[dict, dict]:
    """C2 minus rest CV per window (bootstrap stratified by profile, same draws for both windows) and on the full record."""
    rng = np.random.default_rng(SEEDS.t4_bootstrap + SEEDS.t4_contrast_offset + offset * 10)
    contrast = lambda v, p: float(np.nanmean(v[p == C2]) - np.nanmean(v[np.isin(p, [1, 3])]))  # noqa: E731
    draws = {"A": np.empty(N_BOOT), "B": np.empty(N_BOOT)}
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(np.flatnonzero(profile == c), size=int((profile == c).sum()), replace=True) for c in (1, 2, 3)])
        draws["A"][i], draws["B"][i] = contrast(cv_a[idx], profile[idx]), contrast(cv_b[idx], profile[idx])
    row = {"contrast": CONTRAST}
    for w, cv in (("A", cv_a), ("B", cv_b)):
        low, high = stats.percentile_ci(draws[w])
        row.update({f"{w}_difference": contrast(cv, profile), f"{w}_ci_low": low, f"{w}_ci_high": high})
    low, high = stats.percentile_ci(draws["A"] - draws["B"])
    row.update({"change_A_minus_B": row["A_difference"] - row["B_difference"], "change_ci_low": low, "change_ci_high": high})
    diff, r_low, r_high = stats.bootstrap_mean_difference(cv_full[profile == C2], cv_full[profile != C2],
                                                          np.random.default_rng(SEEDS.t4_reference + offset), N_BOOT)
    return row, {"contrast": CONTRAST, "window": "full record, same participants", "difference": diff, "ci_low": r_low, "ci_high": r_high}


def masked(value: str) -> bool:
    return value in (MASK_SMALL, MASK_COMPLEMENT)
