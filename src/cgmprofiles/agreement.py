"""Agreement between two partitions (ARI, NMI, bias-corrected Cramer's V, kappa), permutation
nulls, participant bootstraps, and the ICC(A,1) between windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f as f_dist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import get_config
from .stats import TAILS

METRICS = ["ARI", "NMI", "CramersV"]
N_RATERS = 2
ICC_ALPHA = 0.05


def cramers_v_bias_corrected(a: np.ndarray, b: np.ndarray) -> float:
    """Bergsma (2013) bias-corrected Cramer's V."""
    _, a_codes = np.unique(a, return_inverse=True)
    _, b_codes = np.unique(b, return_inverse=True)
    table = np.zeros((a_codes.max() + 1, b_codes.max() + 1))
    np.add.at(table, (a_codes, b_codes), 1)
    n = table.sum()
    statistic = chi2_contingency(table, correction=False)[0]
    r, c = table.shape
    phi2 = max(0.0, statistic / n - (r - 1) * (c - 1) / (n - 1))
    denominator = min(r - (r - 1) ** 2 / (n - 1) - 1, c - (c - 1) ** 2 / (n - 1) - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """ARI, NMI (arithmetic) and bias-corrected Cramer's V."""
    return {"ARI": adjusted_rand_score(a, b), "NMI": normalized_mutual_info_score(a, b, average_method="arithmetic"),
            "CramersV": cramers_v_bias_corrected(a, b)}


def permutation_null(groups: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n_perm: int,
                     within: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Agreement under label permutation (optionally within levels of ``within``)."""
    draws = {k: [] for k in METRICS}
    for _ in range(n_perm):
        shuffled = labels.copy()
        if within is None:
            shuffled = rng.permutation(shuffled)
        else:
            for level in np.unique(within):
                idx = np.flatnonzero(within == level)
                shuffled[idx] = rng.permutation(shuffled[idx])
        for k, v in agreement(groups, shuffled).items():
            draws[k].append(v)
    return {k: np.array(v) for k, v in draws.items()}


def bootstrap_agreement(groups: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, tuple[float, float]]:
    """Participant bootstrap CIs of the agreement statistics."""
    draws = {k: [] for k in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), size=len(groups))
        for k, v in agreement(groups[idx], labels[idx]).items():
            draws[k].append(v)
    return {k: tuple(np.percentile(v, TAILS)) for k, v in draws.items()}


# ---------------------------------------------------------------- fast statistics from a k x k table
def cross_table(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """k x k table of labels 1..k."""
    return np.bincount((a - 1) * k + (b - 1), minlength=k * k).reshape(k, k)


def _comb2(x):
    return x * (x - 1) / 2.0


def table_statistics(t: np.ndarray) -> dict[str, float]:
    """Percent agreement, Cohen's kappa, ARI and NMI from a square table (checked against sklearn)."""
    n = t.sum()
    rows, cols = t.sum(axis=1), t.sum(axis=0)
    observed = np.trace(t) / n
    expected = (rows * cols).sum() / n ** 2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else np.nan
    index = _comb2(t).sum()
    expected_index = _comb2(rows).sum() * _comb2(cols).sum() / _comb2(n)
    max_index = (_comb2(rows).sum() + _comb2(cols).sum()) / 2
    ari = (index - expected_index) / (max_index - expected_index) if max_index != expected_index else np.nan
    p = t / n
    pr, pc = rows / n, cols / n
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pr, pc)[nz])).sum()
    h_r = -(pr[pr > 0] * np.log(pr[pr > 0])).sum()
    h_c = -(pc[pc > 0] * np.log(pc[pc > 0])).sum()
    nmi = mi / ((h_r + h_c) / 2) if h_r + h_c > 0 else np.nan
    return {"percent_agreement": observed * 100, "kappa": kappa, "ARI": ari, "NMI": nmi}


def window_agreement(a: np.ndarray, b: np.ndarray, k: int, boot_seed: int, perm_seed: int) -> dict:
    """Agreement of the same participants' groups in two windows: bootstrap CIs (both labels travel
    with the participant) and a null that permutes window B labels across participants."""
    n_boot, n_perm = get_config().bootstrap.n_default, get_config().permutation.n_agreement
    observed = table_statistics(cross_table(a, b, k))
    n = len(a)
    rng = np.random.default_rng(boot_seed)
    boot = pd.DataFrame([table_statistics(cross_table(a[idx], b[idx], k)) for idx in rng.integers(0, n, size=(n_boot, n))])
    rng = np.random.default_rng(perm_seed)
    null = pd.DataFrame([table_statistics(cross_table(a, rng.permutation(b), k)) for _ in range(n_perm)])
    out = {}
    for name, value in observed.items():
        low, high = np.nanpercentile(boot[name], TAILS)
        out[name] = {"value": value, "ci_low": low, "ci_high": high, "null_mean": float(null[name].mean()),
                     "null_q975": float(null[name].quantile(0.975)), "p_permutation": float((1 + (null[name] >= value).sum()) / (n_perm + 1))}
    return out


# ---------------------------------------------------------------- ICC(A,1), McGraw and Wong (1996)
def icc_components(x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Two-way random effects, absolute agreement, single measurement; x has shape (..., n, 2)."""
    n, k = x.shape[-2], N_RATERS
    grand = x.mean(axis=(-2, -1), keepdims=True)
    row_means = x.mean(axis=-1, keepdims=True)
    col_means = x.mean(axis=-2, keepdims=True)
    ss_rows = k * ((row_means - grand) ** 2).sum(axis=(-2, -1))
    ss_cols = n * ((col_means - grand) ** 2).sum(axis=(-2, -1))
    ss_error = ((x - grand) ** 2).sum(axis=(-2, -1)) - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_error / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), msr, msc, mse


def icc_analytic_ci(icc: float, msr: float, msc: float, mse: float, n: int) -> tuple[float, float]:
    """McGraw and Wong analytic 95% interval for ICC(A,1)."""
    k = N_RATERS
    a = k * icc / (n * (1 - icc))
    b = 1 + k * icc * (n - 1) / (n * (1 - icc))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    f_low, f_high = f_dist.ppf(1 - ICC_ALPHA / 2, n - 1, v), f_dist.ppf(1 - ICC_ALPHA / 2, v, n - 1)
    low = n * (msr - f_low * mse) / (f_low * (k * msc + (k * n - k - n) * mse) + n * msr)
    high = n * (f_high * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f_high * msr)
    return float(low), float(high)


def icc_with_ci(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    """ICC(A,1) between two windows with a participant bootstrap and the analytic interval."""
    keep = ~(np.isnan(a) | np.isnan(b))
    x = np.column_stack([a[keep], b[keep]])
    n = len(x)
    icc, msr, msc, mse = (float(v) for v in icc_components(x))
    rng = np.random.default_rng(seed)
    boot = icc_components(x[rng.integers(0, n, size=(get_config().bootstrap.n_default, n))])[0]
    low, high = np.nanpercentile(boot, TAILS)
    analytic = icc_analytic_ci(icc, msr, msc, mse, n)
    return {"icc": icc, "ci_low": float(low), "ci_high": float(high), "analytic_ci_low": analytic[0],
            "analytic_ci_high": analytic[1], "n_pairs": n, "n_dropped_missing": int((~keep).sum())}
