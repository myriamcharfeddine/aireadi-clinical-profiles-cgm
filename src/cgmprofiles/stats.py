"""Group comparisons: Kruskal-Wallis rank eta squared, participant bootstraps, Dunn-Holm, ANOVA.

Each function draws from the generator it is given in a fixed order, so a seeded generator
reproduces the record's intervals exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, chi2_contingency, f_oneway, kruskal, norm

from .config import get_config

TAILS = tuple(get_config().bootstrap.ci_tails)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties given their average rank (1-based)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def kruskal_h(groups: list[np.ndarray]) -> tuple[float, float]:
    """Tie-corrected Kruskal-Wallis H and its chi-square p value."""
    pooled = np.concatenate(groups)
    n = len(pooled)
    ranks = average_ranks(pooled)
    h, position = 0.0, 0
    for group in groups:
        h += ranks[position:position + len(group)].sum() ** 2 / len(group)
        position += len(group)
    h = 12.0 / (n * (n + 1)) * h - 3.0 * (n + 1)
    _, ties = np.unique(pooled, return_counts=True)
    h /= 1.0 - (ties ** 3 - ties).sum() / (n ** 3 - n)
    return float(h), float(chi2.sf(h, len(groups) - 1))


def eta_squared_from_h(h: float, k: int, n: int) -> float:
    """Rank-based eta squared, (H - k + 1) / (n - k), clipped at 0."""
    return max(0.0, (h - k + 1) / (n - k))


def kruskal_eta_squared(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """Eta squared, p value and n for values across groups (missing values dropped)."""
    keep = ~np.isnan(values)
    values, groups = values[keep], groups[keep]
    present = [values[groups == g] for g in np.unique(groups)]
    h, p = kruskal(*present)
    return eta_squared_from_h(float(h), len(present), len(values)), float(p), int(len(values))


def eta_or_nan(values: np.ndarray, groups: np.ndarray, k: int) -> float:
    """Eta squared, or NaN when fewer than k groups are present."""
    keep = ~np.isnan(values)
    if len(np.unique(groups[keep])) < k:
        return np.nan
    return kruskal_eta_squared(values, groups)[0]


def percentile_ci(draws) -> tuple[float, float]:
    """Percentile interval of the non-missing draws."""
    draws = np.asarray(draws, float)
    draws = draws[~np.isnan(draws)]
    return tuple(float(x) for x in np.percentile(draws, TAILS))


def bootstrap_eta_squared(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """Participant bootstrap CI of eta squared; resamples missing a group are skipped."""
    keep = ~np.isnan(values)
    values, groups = values[keep], groups[keep]
    k = len(np.unique(groups))
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        if len(np.unique(groups[idx])) == k:
            draws.append(kruskal_eta_squared(values[idx], groups[idx])[0])
    return tuple(np.percentile(draws, TAILS))


def eta_squared_draws(values: np.ndarray, groups: np.ndarray, indices: list[np.ndarray], profiles: list[int]) -> np.ndarray:
    """Eta squared for each precomputed resample (shared across windows for paired intervals)."""
    draws = []
    for index in indices:
        v, g = values[index], groups[index]
        h, _ = kruskal_h([v[g == c] for c in profiles])
        draws.append(eta_squared_from_h(h, len(profiles), len(v)))
    return np.asarray(draws)


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    """Mean of a minus mean of b, with participants resampled within each group."""
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = a[rng.integers(0, len(a), size=len(a))].mean() - b[rng.integers(0, len(b), size=len(b))].mean()
    low, high = np.percentile(draws, TAILS)
    return float(a.mean() - b.mean()), float(low), float(high)


def anova_eta_squared(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """One-way ANOVA eta squared (SS between / SS total) and its F-test p value."""
    groups = [values[labels == c] for c in np.unique(labels)]
    grand = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    return float(ss_between / ((values - grand) ** 2).sum()), float(f_oneway(*groups).pvalue)


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p values."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def dunn(values: np.ndarray, labels: np.ndarray, pairs: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[float, float]]:
    """Tie-corrected Dunn z and two-sided p for each pair (positive z: first group ranks higher)."""
    n = len(values)
    ranks = average_ranks(values)
    _, ties = np.unique(values, return_counts=True)
    tie_term = (ties ** 3 - ties).sum() / (12.0 * (n - 1))
    out = {}
    for a, b in pairs:
        ra, rb = ranks[labels == a], ranks[labels == b]
        se = np.sqrt((n * (n + 1) / 12.0 - tie_term) * (1 / len(ra) + 1 / len(rb)))
        z = (ra.mean() - rb.mean()) / se
        out[(a, b)] = (float(z), float(2 * norm.sf(abs(z))))
    return out


def two_sided_p(null: np.ndarray, observed: float) -> float:
    """Two-sided permutation p value with the +1 correction."""
    upper = (1 + np.sum(null >= observed)) / (len(null) + 1)
    lower = (1 + np.sum(null <= observed)) / (len(null) + 1)
    return float(min(1.0, 2 * min(upper, lower)))


def upper_p(null: np.ndarray, observed: float) -> float:
    """One-sided (upper) permutation p value with the +1 correction."""
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def cramers_v(table: np.ndarray) -> tuple[float, float]:
    """Plain Cramer's V and chi-square p (no continuity correction)."""
    statistic, p, _, _ = chi2_contingency(table, correction=False)
    return float(np.sqrt(statistic / (table.sum() * (min(table.shape) - 1)))), float(p)
