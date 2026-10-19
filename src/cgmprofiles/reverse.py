"""Reverse direction (Section 3.2): CGM groups, clinical differences, agreement and the B5 prediction grid,
with the record's seeds and draw order (B5 permutation seeds are the base seed plus the running row count)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from . import agreement as ag
from . import cgm_groups as cg
from . import io, prediction as pr, stats
from .cgm_metrics import MEAN, METRICS, WINDOWS
from .config import get_config

CFG = get_config()
SEEDS, CRIT = CFG.seeds, CFG.criteria
N_BOOT, N_PERM = CFG.bootstrap.n_default, CFG.permutation.n_agreement
HBA1C = CFG.columns.hba1c
PRIMARY_NAME = "extended_window_full"
PRIMARY = f"kmeans3_{PRIMARY_NAME}"

def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Stop: {message}")


def partitions(features: dict[str, pd.DataFrame], is_train: np.ndarray, splits: np.ndarray):
    """B2: k-means k = 3 and the silhouette-selected k for each feature set and window."""
    label_columns, diagnostics, centroid_rows, size_rows, stability = {}, [], [], [], None
    for window in WINDOWS:
        frame = features[window]
        raw_mean = frame[MEAN].to_numpy(float)
        for feature_set in ("minimal", "extended"):
            name = f"{feature_set}_{window}"
            prep = cg.fit_preprocessing(frame, is_train, feature_set, window)
            matrix = prep.pop("matrix")
            km = cg.kmeans_partition(matrix, is_train, raw_mean, CFG.cgm_groups.k_primary)
            sil, k_sil = cg.silhouette_scan(matrix[is_train])
            label_columns[f"kmeans3_{name}"] = km["labels"]
            label_columns[f"kmeans_silk{k_sil}_{name}"] = cg.kmeans_partition(matrix, is_train, raw_mean, k_sil)["labels"]
            if name == PRIMARY_NAME:
                stability = cg.bootstrap_stability(matrix, is_train, km["labels"], SEEDS.b2_stability)
                check(min(np.bincount(km["labels"][is_train])[1:]) >= CFG.cgm_groups.min_cluster_n, "primary CGM group below 20 training participants")
            diagnostics.append({"partition": name, "primary": name == PRIMARY_NAME, "features": ";".join(prep["columns"]), "log1p": ";".join(prep["log1p"]),
                                "profile_pcs": prep.get("profile_pcs"), "train_silhouette_k3": km["train_silhouette"],
                                **{f"silhouette_k{k}": v for k, v in sil.items()}, "silhouette_selected_k": k_sil})
            for s in ["train", "val", "test", "all"]:
                keep = np.ones(len(splits), bool) if s == "all" else splits == s
                size_rows.append({"partition": name, "split": s, **{f"G{g}": int((km["labels"][keep] == g).sum()) for g in (1, 2, 3)}})
            for g in (1, 2, 3):
                members = km["labels"] == g
                centroid_rows.append({"partition": name, "cluster": f"G{g}", "n": int(members.sum()),
                                      **{f"z_{c}": float(km["centroids"][g - 1, j]) for j, c in enumerate(prep["columns"])},
                                      **{f"median_{c}": float(frame.loc[members, c].median()) for c in prep["scalars"]}})
    return pd.DataFrame(label_columns), pd.DataFrame(diagnostics), pd.DataFrame(centroid_rows), pd.DataFrame(size_rows), stability


def clinical_differences(groups_by_partition: pd.DataFrame, clinical: pd.DataFrame, is_train: np.ndarray) -> pd.DataFrame:
    """B3: eta squared of each clinical variable across CGM groups; one bootstrap stream over partitions in order."""
    rng = np.random.default_rng(SEEDS.b3_bootstrap)
    subsets = {"all": np.ones(len(clinical), bool), "train only": is_train, "validation and test only": ~is_train}
    rows = []
    for column, role, k, descriptive in CFG.reverse.partitions:
        groups = groups_by_partition[column].to_numpy()
        for factor in io.FACTORS:
            for subset, keep in subsets.items():
                values = clinical[factor].to_numpy(float)[keep]
                eta, p, n = stats.kruskal_eta_squared(values, groups[keep])
                low, high = (np.nan, np.nan)
                if subset == "all" and not descriptive:
                    low, high = stats.bootstrap_eta_squared(values, groups[keep], rng, N_BOOT)
                rows.append({"partition": column, "role": role, "k": k, "factor": factor, "hba1c_flag": factor == HBA1C, "subset": subset,
                             "n": n, "eta_squared": eta, "ci_low": low, "ci_high": high, "p_value": np.nan if descriptive else p})
    return pd.DataFrame(rows)


def partition_agreement(groups_by_partition: pd.DataFrame, profile: np.ndarray, splits: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B4: agreement with C1 to C3, permutation nulls (plain and within split) and bootstrap CIs; contingency by split."""
    perm_rng, boot_rng = np.random.default_rng(SEEDS.b4_permutation), np.random.default_rng(SEEDS.b4_bootstrap)
    rows, tables = [], []
    for column, role, k, descriptive in CFG.reverse.partitions:
        groups = groups_by_partition[column].to_numpy()
        for s in ("all", "train", "val", "test"):
            keep = np.ones(len(profile), bool) if s == "all" else splits == s
            table = pd.crosstab(pd.Series(groups[keep], name="cgm"), pd.Series(profile[keep], name="profile"))
            tables += [{"partition": column, "split": s, "cgm_cluster": f"G{g}", **{f"C{c}": int(r.get(c, 0)) for c in (1, 2, 3)}} for g, r in table.iterrows()]
        observed = ag.agreement(groups, profile)
        null = ag.permutation_null(groups, profile, perm_rng, N_PERM)
        null_split = ag.permutation_null(groups, profile, perm_rng, N_PERM, within=splits)
        ci = ag.bootstrap_agreement(groups, profile, boot_rng, N_BOOT) if not descriptive else {m: (np.nan, np.nan) for m in observed}
        for metric, value in observed.items():
            rows.append({"partition": column, "role": role, "k": k, "metric": metric, "value": value, "ci_low": ci[metric][0], "ci_high": ci[metric][1],
                         "perm_p": np.nan if descriptive else stats.upper_p(null[metric], value),
                         "perm_p_within_split": np.nan if descriptive else stats.upper_p(null_split[metric], value),
                         "null_mean": float(null[metric].mean()), "null_q975": float(np.percentile(null[metric], 97.5))})
    return pd.DataFrame(rows), pd.DataFrame(tables)


def prediction(t2d: pd.DataFrame, features: dict[str, pd.DataFrame], metrics: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """B5, in the record's row order (permutation seeds are the base seed plus the running row count)."""
    profile = t2d.label.to_numpy().astype(int)
    classes = np.unique(profile)
    fold_list = pr.folds(profile)
    clinical = t2d[io.FACTORS].astype(float).reset_index(drop=True)
    rows, null_rows, verify = [], [], {}
    five = [f for f in io.FACTORS if f != HBA1C]
    jobs = [(w, m, label, f, metrics[w][m].to_numpy(float)) for w in WINDOWS for m in METRICS
            for label, f in (("six variables", io.FACTORS), ("five variables (no HbA1c)", five))]
    oofs = Parallel(n_jobs=CFG.prediction.n_jobs)(delayed(pr.clinical_to_cgm_oof)(clinical, f, y, fold_list) for *_, f, y in jobs)
    for (window, metric, label, _, y), oof in zip(jobs, oofs):
        r2 = r2_score(y, oof)
        low, high = pr.bootstrap_ci(lambda idx: r2_score(y[idx], oof[idx]), len(y), SEEDS.b5_bootstrap)
        if label == "six variables":
            verify[f"{window}/{metric}"] = abs(r2 - CFG.prediction.section32_m2_r2[window][metric])
        rows.append({"direction": "clinical to CGM", "feature_set": label, "window": window, "target": metric, "metric": "R2", "value": r2,
                     "ci_low": low, "ci_high": high, "perm_p": np.nan, "n_permutations": 0, "hyperparameter": ""})
    check(max(verify.values()) <= CFG.prediction.section32_tolerance, f"Section 3.2 R2 not reproduced on the rebuilt folds: {verify}")
    combos = [tuple(c) for c in CFG.prediction.combos]
    primary = combos[0]
    matrices = {c: pr.fold_matrices(features[c[1]], c[0], c[1], fold_list) for c in combos}
    hba1c_mats = pr.hba1c_matrices(clinical[HBA1C].to_numpy(float), fold_list)
    escalate = lambda p, combo: combo == primary and pr.needs_escalation(p)  # noqa: E731
    for combo in combos:
        for factor in io.FACTORS:
            y = clinical[factor].to_numpy(float)
            obs = ~np.isnan(y)
            oof, chosen = pr.ridge_cv(matrices[combo], y, fold_list)
            yo, po = y[obs], oof[:, 0][obs]
            r2 = r2_score(yo, po)
            low, high = pr.bootstrap_ci(lambda idx: r2_score(yo[idx], po[idx]), len(yo), SEEDS.b5_bootstrap)
            alpha, seed = pr.most_common(chosen), SEEDS.b5_permutation + len(rows)
            null = pr.ridge_permutation_null(matrices[combo], y, fold_list, alpha, CFG.permutation.n_prediction, seed)
            p, n_perm = stats.upper_p(null, r2), CFG.permutation.n_prediction
            if escalate(p, combo):
                null = pr.ridge_permutation_null(matrices[combo], y, fold_list, alpha, CFG.permutation.n_prediction_escalated, seed + 1)
                p, n_perm = stats.upper_p(null, r2), CFG.permutation.n_prediction_escalated
            rows.append({"direction": "CGM to clinical variable", "feature_set": combo[0], "window": combo[1], "target": factor, "metric": "R2",
                         "value": r2, "ci_low": low, "ci_high": high, "perm_p": p, "n_permutations": n_perm, "hyperparameter": f"alpha {alpha:g}"})
            if combo == primary:
                null_rows += [{"analysis": f"ridge R2 {factor}", "draw": i, "value": v} for i, v in enumerate(null)]
    specs = [(c, name, mats) for c in combos for name, mats in (("CGM features", matrices[c]),
                                                                ("HbA1c plus CGM features", [np.hstack([h, m]) for h, m in zip(hba1c_mats, matrices[c])]))]
    specs.append((("HbA1c", "none"), "HbA1c alone", hba1c_mats))
    fitted = Parallel(n_jobs=CFG.prediction.n_jobs)(delayed(pr.logistic_cv)(mats, profile, fold_list) for *_, mats in specs)
    proba = {(c, name): f[0] for (c, name, _), f in zip(specs, fitted)}
    chosen_c = {(c, name): pr.most_common(f[1]) for (c, name, _), f in zip(specs, fitted)}

    def metric_at(p_matrix, idx, metric):
        if metric == "balanced_accuracy":
            return balanced_accuracy_score(profile[idx], classes[p_matrix[idx].argmax(axis=1)])
        return roc_auc_score(profile[idx], p_matrix[idx], multi_class="ovr", average="macro", labels=classes)

    hba1c_only = proba[(("HbA1c", "none"), "HbA1c alone")]
    base = pr.classification_metrics(profile, hba1c_only, classes)
    for metric, value in base.items():
        low, high = pr.bootstrap_ci(lambda idx: metric_at(hba1c_only, idx, metric), len(profile), SEEDS.b5_bootstrap)
        rows.append({"direction": "CGM to C1/C2/C3", "feature_set": "HbA1c alone", "window": "", "target": "C1/C2/C3", "metric": metric, "value": value,
                     "ci_low": low, "ci_high": high, "perm_p": np.nan, "n_permutations": 0, "hyperparameter": f"C {chosen_c[(('HbA1c', 'none'), 'HbA1c alone')]:g}"})
    for combo in combos:
        cgm = proba[(combo, "CGM features")]
        observed = pr.classification_metrics(profile, cgm, classes)
        c_fixed, seed = chosen_c[(combo, "CGM features")], SEEDS.b5_permutation + 1000 + len(rows)
        null = pr.logistic_permutation_null(matrices[combo], profile, fold_list, c_fixed, CFG.permutation.n_prediction, seed)
        n_perm = CFG.permutation.n_prediction
        p_values = {m: stats.upper_p(null[m], v) for m, v in observed.items()}
        if any(escalate(p, combo) for p in p_values.values()):
            null = pr.logistic_permutation_null(matrices[combo], profile, fold_list, c_fixed, CFG.permutation.n_prediction_escalated, seed + 1)
            p_values, n_perm = {m: stats.upper_p(null[m], v) for m, v in observed.items()}, CFG.permutation.n_prediction_escalated
        both = proba[(combo, "HbA1c plus CGM features")]
        for metric, value in observed.items():
            low, high = pr.bootstrap_ci(lambda idx: metric_at(cgm, idx, metric), len(profile), SEEDS.b5_bootstrap)
            rows.append({"direction": "CGM to C1/C2/C3", "feature_set": combo[0], "window": combo[1], "target": "C1/C2/C3", "metric": metric, "value": value,
                         "ci_low": low, "ci_high": high, "perm_p": p_values[metric], "n_permutations": n_perm, "hyperparameter": f"C {c_fixed:g}",
                         "null_mean": float(null[metric].mean())})
            if combo == primary:
                null_rows += [{"analysis": f"C1/C2/C3 {metric}", "draw": i, "value": v} for i, v in enumerate(null[metric])]
            combined = metric_at(both, np.arange(len(profile)), metric)
            g_low, g_high = pr.bootstrap_ci(lambda idx: metric_at(both, idx, metric) - metric_at(hba1c_only, idx, metric), len(profile), SEEDS.b5_bootstrap)
            rows.append({"direction": "CGM to C1/C2/C3, variant (a)", "feature_set": combo[0], "window": combo[1], "target": "C1/C2/C3",
                         "metric": f"{metric} gain, HbA1c plus CGM minus HbA1c alone", "value": combined - base[metric], "ci_low": g_low, "ci_high": g_high,
                         "perm_p": np.nan, "n_permutations": 0, "hyperparameter": f"C {chosen_c[(combo, 'HbA1c plus CGM features')]:g}",
                         "hba1c_plus_cgm": combined})
    return pd.DataFrame(rows), pd.DataFrame(null_rows), verify
