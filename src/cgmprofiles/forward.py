"""Forward direction (Section 3.1): CGM outcomes across clinical profiles, with the record's seeds and draw order."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from . import io, stats
from .cgm_metrics import CV, METRICS, WINDOWS
from .config import get_config

CFG = get_config()
SEEDS = CFG.seeds
PROFILES = [1, 2, 3]
PAIRS = list(itertools.combinations(PROFILES, 2))
C2 = CFG.profiles.c2

def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["split_rank", "split_file_row"])


def forward_tests(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kruskal-Wallis per stratum, window and metric; pairwise bootstrap differences with seed 42 + running counter."""
    kw_rows, pair_rows, counter = [], [], 0
    for stratum in io.STRATA:
        for window in WINDOWS:
            frame = ordered(metrics[(metrics.canonical_stratum == stratum) & (metrics.window == window)])
            for metric in METRICS:
                groups = {int(c): frame.loc[frame.label == c, metric].dropna().to_numpy() for c in sorted(frame.label.unique())}
                clusters = sorted(groups)
                n = sum(len(g) for g in groups.values())
                h, p = stats.kruskal_h([groups[c] for c in clusters])
                kw_rows.append({"canonical_stratum": stratum, "window": window, "metric": metric, "n_clusters": len(clusters), "n_total": n,
                                "kruskal_h": h, "p_value": p, "eta_squared": stats.eta_squared_from_h(h, len(clusters), n)})
                for a, b in itertools.combinations(clusters, 2):
                    counter += 1
                    diff, low, high = stats.bootstrap_mean_difference(groups[a], groups[b], np.random.default_rng(SEEDS.forward_pairwise + counter),
                                                                      CFG.bootstrap.n_pairwise)
                    pair_rows.append({"canonical_stratum": stratum, "window": window, "metric": metric, "cluster_a": a, "cluster_b": b,
                                      "n_a": len(groups[a]), "n_b": len(groups[b]), "mean_diff_a_minus_b": diff, "ci_low": low, "ci_high": high,
                                      "ci_excludes_zero": bool(low > 0 or high < 0)})
    return pd.DataFrame(kw_rows), pd.DataFrame(pair_rows)


def c2_vs_rest(t2d_metrics: pd.DataFrame) -> pd.DataFrame:
    """C2 against C1 and C3 together, CV, both windows."""
    rows = []
    for counter, window in enumerate(WINDOWS, start=1):
        sub = t2d_metrics[t2d_metrics.window == window]
        c2, rest = sub.loc[sub.label == C2, CV].dropna().to_numpy(), sub.loc[sub.label != C2, CV].dropna().to_numpy()
        h, p = stats.kruskal_h([c2, rest])
        rng = np.random.default_rng(SEEDS.forward_pairwise + SEEDS.c2_vs_rest_offset + counter)
        diff, low, high = stats.bootstrap_mean_difference(c2, rest, rng, CFG.bootstrap.n_pairwise)
        rows.append({"window": window, "n_c2": len(c2), "n_rest": len(rest), "kruskal_h": h, "p_value": p,
                     "eta_squared": stats.eta_squared_from_h(h, 2, len(c2) + len(rest)), "mean_diff": diff, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def eta_intervals_and_tests(t2d: pd.DataFrame, by_window: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2,000 participant resamples shared by both windows (paired); ANOVA; Dunn with Holm (within 3 pairs and across 18)."""
    rng = np.random.default_rng(SEEDS.eta_bootstrap)
    n = len(t2d)
    indices = [rng.integers(0, n, size=n) for _ in range(CFG.bootstrap.n_default)]
    rows, dunn_rows, draws = [], [], {}
    for window in WINDOWS:
        frame = by_window[window]
        labels = frame.label.to_numpy()
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            eta, p, _ = stats.kruskal_eta_squared(values, labels)
            draws[(window, metric)] = (eta, stats.eta_squared_draws(values, labels, indices, PROFILES))
            low, high = np.percentile(draws[(window, metric)][1], stats.TAILS)
            anova_eta, anova_p = stats.anova_eta_squared(values, labels)
            rows.append({"window": window, "metric": metric, "n": n, "eta_squared": eta, "ci_low": low, "ci_high": high, "p_value": p,
                         "anova_eta_squared": anova_eta, "anova_p": anova_p})
            for (a, b), (z, p_raw) in stats.dunn(values, labels, PAIRS).items():
                dunn_rows.append({"window": window, "metric": metric, "pair": f"C{a} vs C{b}", "z": z, "p_raw": p_raw})
    for metric in METRICS:
        full, short = draws[(WINDOWS[1], metric)], draws[(WINDOWS[0], metric)]
        low, high = np.percentile(full[1] - short[1], stats.TAILS)
        rows.append({"window": "full minus 48h", "metric": metric, "n": n, "eta_squared": full[0] - short[0], "ci_low": low, "ci_high": high})
    dunn = pd.DataFrame(dunn_rows)
    dunn["p_holm_within_3"] = dunn.groupby(["window", "metric"]).p_raw.transform(lambda s: pd.Series(stats.holm(s.tolist()), index=s.index))
    dunn["p_holm_across_18"] = stats.holm(dunn.p_raw.tolist())
    return pd.DataFrame(rows), dunn


def sensitivity_subsets(by_window: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Eta squared (and ANOVA) in the subsets named in Methods 2.3."""
    margin = CFG.profiles.margin_threshold
    subsets = {"all": lambda f: f, "training only": lambda f: f[f.split == "train"], "validation and test only": lambda f: f[f.split != "train"],
               f"relative assignment margin > {margin:.2f}": lambda f: f[f.relative_margin > margin],
               "complete-case clinical variables": lambda f: f[f.n_missing_factors == 0]}
    rows = []
    for name, select in subsets.items():
        for window in WINDOWS:
            frame = select(by_window[window])
            for metric in METRICS:
                eta, p, n = stats.kruskal_eta_squared(frame[metric].to_numpy(float), frame.label.to_numpy())
                a_eta, a_p = stats.anova_eta_squared(frame[metric].to_numpy(float), frame.label.to_numpy())
                rows.append({"subset": name, "window": window, "metric": metric, "n": n, "eta_squared": eta, "p_value": p,
                             "anova_eta_squared": a_eta, "anova_p": a_p})
    return pd.DataFrame(rows)


def split_dependence(t2d: pd.DataFrame, by_window: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Observed (held-out minus training) eta squared against 2,000 random 325/133 partitions keeping profile counts."""
    rng = np.random.default_rng(SEEDS.split_partition)
    labels = t2d.label.to_numpy()
    is_train = (t2d.split == "train").to_numpy()
    train_counts = {c: int((is_train & (labels == c)).sum()) for c in PROFILES}
    partitions = []
    for _ in range(CFG.bootstrap.n_split_partitions):
        mask = np.zeros(len(t2d), dtype=bool)
        for c in PROFILES:
            mask[rng.choice(np.flatnonzero(labels == c), size=train_counts[c], replace=False)] = True
        partitions.append(mask)
    eta = lambda v, lab: stats.kruskal_eta_squared(v, lab)[0]  # noqa: E731
    rows = []
    for window in WINDOWS:
        for metric in METRICS:
            values = by_window[window][metric].to_numpy(float)
            train, rest = eta(values[is_train], labels[is_train]), eta(values[~is_train], labels[~is_train])
            null = np.array([eta(values[~m], labels[~m]) - eta(values[m], labels[m]) for m in partitions])
            low, high = np.percentile(null, stats.TAILS)
            rows.append({"window": window, "metric": metric, "eta_train": train, "eta_heldout": rest, "difference": rest - train,
                         "null_q025": low, "null_q975": high, "p_two_sided": stats.two_sided_p(null, rest - train)})
    return pd.DataFrame(rows)


def glp1_contrasts(by_window: dict[str, pd.DataFrame], glp1: np.ndarray) -> pd.DataFrame:
    """C2 minus C1 and C2 minus C3 CV within participants with and without a GLP-1/GIP flag (descriptive)."""
    seed, rows = SEEDS.glp1_contrast + 1, []
    for window in WINDOWS:
        frame = by_window[window].assign(glp1=glp1)
        for flag, subset in ((1, "GLP-1/GIP flag"), (0, "no GLP-1/GIP flag")):
            sub = frame[frame.glp1 == flag]
            groups = {c: sub.loc[sub.label == c, CV].to_numpy(float) for c in PROFILES}
            for other in (1, 3):
                seed += 1
                diff, low, high = stats.bootstrap_mean_difference(groups[C2], groups[other], np.random.default_rng(seed), CFG.bootstrap.n_default)
                rows.append({"subset": subset, "window": window, "contrast": f"C2 minus C{other}", "n_c2": len(groups[C2]),
                             "n_other": len(groups[other]), "mean_diff": diff, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)
