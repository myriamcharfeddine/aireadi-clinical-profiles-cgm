"""Other glycemic strata and the pooled cohort (Section 3.3): S1 forward, S3 specificity, S2 reverse, B6 pooled,
with the record's seeds and draw order."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from . import agreement as ag
from . import cgm_features as cf
from . import cgm_groups as cg
from . import io, prediction as pr, profiles, stats
from .cgm_metrics import CV, MEAN, METRICS, WINDOWS, window_metrics
from .config import get_config

CFG = get_config()
SEEDS, SCFG = CFG.seeds, CFG.strata
N_BOOT, N_PERM = CFG.bootstrap.n_default, CFG.permutation.n_agreement
HBA1C, C2 = CFG.columns.hba1c, CFG.profiles.c2

def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Stop: {message}")


def forward_other_strata(metrics: pd.DataFrame) -> pd.DataFrame:
    """S1: eta squared with bootstrap CIs (one stream over strata, windows, metrics); Holm across strata."""
    rng, rows = np.random.default_rng(SEEDS.s1_bootstrap), []
    for stratum in SCFG.other:
        sub = metrics[metrics.canonical_stratum == stratum].sort_values(["split_rank", "split_file_row"])
        for window in WINDOWS:
            frame = sub[sub.window == window]
            groups = frame.label.to_numpy()
            for metric in METRICS:
                values = frame[metric].to_numpy(float)
                eta, p, _ = stats.kruskal_eta_squared(values, groups)
                low, high = stats.bootstrap_eta_squared(values, groups, rng, N_BOOT)
                rows.append({"canonical_stratum": stratum, "window": window, "metric": metric, "n": len(values), "k": len(np.unique(groups)),
                             "eta_squared": eta, "ci_low": low, "ci_high": high, "p_value": p, "exploratory": stratum in SCFG.exploratory})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["window", "metric"]).groups.items():
        out.loc[idx, "p_holm"] = stats.holm(out.loc[idx, "p_value"].tolist())
    return out


def specificity(cohort: pd.DataFrame, labels: pd.DataFrame, t2d_model: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S3: healthy and prediabetes participants assigned to the frozen T2D centroids; C2-assigned minus rest."""
    t2d = labels[labels.canonical_stratum == io.PRIMARY]
    threshold = float(np.percentile(profiles.assign_to_frozen_centroids(profiles.transform(t2d, t2d_model), t2d_model).nearest_distance,
                                    SCFG.out_of_range_percentile))
    parts = []
    for stratum in SCFG.specificity_strata:
        sub = cohort[cohort.canonical_stratum == stratum].sort_values("split_file_row").reset_index(drop=True)
        primary = profiles.assign_to_frozen_centroids(profiles.transform(sub, t2d_model), t2d_model)
        values = profiles.impute_and_log(sub, t2d_model)
        train = (sub.split == "train").to_numpy()
        within = profiles.assign_to_frozen_centroids((values - values[train].mean(axis=0)) / values[train].std(axis=0), t2d_model)
        out = sub[[io.PID, "split"]].assign(canonical_stratum=stratum, label_primary=primary.label.to_numpy(),
                                            relative_margin=primary.relative_margin.to_numpy(), out_of_range=primary.nearest_distance.to_numpy() > threshold,
                                            label_within_stratum=within.label.to_numpy(), n_missing_factors=sub[io.FACTORS].isna().sum(axis=1).to_numpy())
        parts.append(out)
    assignments = pd.concat(parts, ignore_index=True)
    cgm = io.load_cgm(assignments[io.PID].tolist())
    for window, hours in CFG.cgm.windows.raw.items():
        m = window_metrics(cgm, hours)
        assignments = assignments.merge(m[[io.PID, *METRICS]].rename(columns={k: f"{k}__{window}" for k in METRICS}), on=io.PID, how="left")
    margin = CFG.profiles.margin_threshold
    variants = {"primary (T2D scaler, all)": lambda s: (s, "label_primary"),
                f"relative margin > {margin:.2f}": lambda s: (s[s.relative_margin > margin], "label_primary"),
                "excluding out-of-range": lambda s: (s[~s.out_of_range], "label_primary"),
                "within-stratum standardization": lambda s: (s, "label_within_stratum")}
    rng, rows = np.random.default_rng(SEEDS.s3_bootstrap), []
    for stratum in SCFG.specificity_strata:
        sub_all = assignments[assignments.canonical_stratum == stratum]
        for variant, select in variants.items():
            sub, column = select(sub_all)
            is_c2 = (sub[column] == C2).to_numpy()
            descriptive = min(is_c2.sum(), (~is_c2).sum()) < SCFG.min_group_n
            for window in WINDOWS:
                for metric in METRICS:
                    values = sub[f"{metric}__{window}"].to_numpy(float)
                    groups = is_c2.astype(int)
                    eta, p = (stats.kruskal_eta_squared(values, groups)[:2] if is_c2.sum() and (~is_c2).sum() else (np.nan, np.nan))
                    row = {"canonical_stratum": stratum, "variant": variant, "window": window, "metric": metric, "n_c2": int(is_c2.sum()),
                           "n_rest": int((~is_c2).sum()), "eta_squared": eta, "eta_ci_low": np.nan, "eta_ci_high": np.nan, "p_value": np.nan,
                           "mean_diff_c2_minus_rest": float(values[is_c2].mean() - values[~is_c2].mean()) if is_c2.sum() else np.nan,
                           "diff_ci_low": np.nan, "diff_ci_high": np.nan, "descriptive_only": bool(descriptive)}
                    if not descriptive:
                        row["eta_ci_low"], row["eta_ci_high"] = stats.bootstrap_eta_squared(values, groups, rng, N_BOOT)
                        _, row["diff_ci_low"], row["diff_ci_high"] = stats.bootstrap_mean_difference(values[is_c2], values[~is_c2], rng, N_BOOT)
                        row["p_value"] = p
                    rows.append(row)
    spec = pd.DataFrame(rows)
    spec["p_holm"] = np.nan
    main = (spec.variant == "primary (T2D scaler, all)") & (spec.metric == CV) & ~spec.descriptive_only
    for window in WINDOWS:
        idx = spec[main & (spec.window == window)].index
        spec.loc[idx, "p_holm"] = stats.holm(spec.loc[idx, "p_value"].tolist())
    summary = assignments.groupby("canonical_stratum").agg(n=(io.PID, "size"), assigned_C2=("label_primary", lambda s: int((s == C2).sum())),
                                                           out_of_range=("out_of_range", "sum")).reset_index()
    return spec, summary


def reverse_one(stratum: str, sub: pd.DataFrame, features: pd.DataFrame, window: str, k: int, with_prediction: bool) -> dict:
    """S2 for one stratum, window and k: CGM groups, clinical differences, agreement, optional prediction."""
    is_train, profile = (sub.split == "train").to_numpy(), sub.label.to_numpy().astype(int)
    clinical = sub[io.FACTORS].astype(float).reset_index(drop=True)
    prep = cg.fit_preprocessing(features, is_train, "extended", window)
    matrix = prep.pop("matrix")
    groups = cg.kmeans_partition(matrix, is_train, features[MEAN].to_numpy(float), k)["labels"]
    descriptive = bool(pd.Series(groups[is_train]).value_counts().reindex(range(1, k + 1), fill_value=0).min() < SCFG.min_train_group_n)
    name = f"{stratum}__{window}__k{k}"
    common = {"partition": name, "stratum": stratum, "window": window, "k": k, "descriptive_only": descriptive}
    rng, b3 = np.random.default_rng(SEEDS.s2_b3), []
    for factor in io.FACTORS:
        values = clinical[factor].to_numpy(float)
        eta, p, n = stats.kruskal_eta_squared(values, groups)
        low, high = (np.nan, np.nan) if descriptive else stats.bootstrap_eta_squared(values, groups, rng, N_BOOT)
        b3.append({**common, "factor": factor, "hba1c_flag": factor == HBA1C, "n": n, "eta_squared": eta, "ci_low": low, "ci_high": high,
                   "p_value": np.nan if descriptive else p})
    observed = ag.agreement(groups, profile)
    null = ag.permutation_null(groups, profile, np.random.default_rng(SEEDS.s2_b4_permutation), N_PERM)
    ci = {m: (np.nan, np.nan) for m in observed} if descriptive else ag.bootstrap_agreement(groups, profile, np.random.default_rng(SEEDS.s2_b4_bootstrap), N_BOOT)
    b4 = [{**common, "metric": m, "value": v, "ci_low": ci[m][0], "ci_high": ci[m][1], "perm_p": np.nan if descriptive else stats.upper_p(null[m], v),
           "null_mean": float(null[m].mean()), "null_q975": float(np.percentile(null[m], 97.5))} for m, v in observed.items()]
    out = {"labels": groups, "b3": b3, "b4": b4, "b5": [], "sizes": {**common, "sizes": "/".join(str(int((groups == g).sum())) for g in range(1, k + 1))}}
    if not with_prediction:
        return out
    classes, fold_list = np.unique(profile), pr.folds(profile)
    matrices = pr.fold_matrices(features, "extended", window, fold_list)
    hba1c = pr.hba1c_matrices(clinical[HBA1C].to_numpy(float), fold_list)
    both = [np.hstack([h, m]) for h, m in zip(hba1c, matrices)]
    (p_cgm, c_cgm), (p_h, _), (p_b, _) = Parallel(n_jobs=3)(delayed(pr.logistic_cv)(m, profile, fold_list) for m in (matrices, hba1c, both))
    c_fixed = pr.most_common(c_cgm)
    observed, base, combined = (pr.classification_metrics(profile, p, classes) for p in (p_cgm, p_h, p_b))
    seed = SEEDS.s2_b5_permutation
    null_cls = pr.logistic_permutation_null(matrices, profile, fold_list, c_fixed, CFG.permutation.n_prediction, seed)
    p_vals, n_perm = {m: stats.upper_p(null_cls[m], v) for m, v in observed.items()}, CFG.permutation.n_prediction
    if pr.needs_escalation(p_vals["macro_auroc"]):
        null_cls = pr.logistic_permutation_null(matrices, profile, fold_list, c_fixed, CFG.permutation.n_prediction_escalated, seed + 1)
        p_vals, n_perm = {m: stats.upper_p(null_cls[m], v) for m, v in observed.items()}, CFG.permutation.n_prediction_escalated
    at = lambda proba, idx, metric: pr.classification_metrics(profile[idx], proba[idx], classes)[metric]  # noqa: E731
    for metric in ("balanced_accuracy", "macro_auroc"):
        low, high = pr.bootstrap_ci(lambda idx: at(p_cgm, idx, metric), len(profile), SEEDS.b5_bootstrap, skip_errors=True)
        g_low, g_high = pr.bootstrap_ci(lambda idx: at(p_b, idx, metric) - at(p_h, idx, metric), len(profile), SEEDS.b5_bootstrap, skip_errors=True)
        out["b5"].append({**common, "direction": "CGM to profile", "target": "profile", "metric": metric, "value": observed[metric], "ci_low": low,
                          "ci_high": high, "perm_p": p_vals[metric], "n_permutations": n_perm, "hba1c_alone": base[metric],
                          "hba1c_plus_cgm": combined[metric], "gain_over_hba1c": combined[metric] - base[metric], "gain_ci_low": g_low, "gain_ci_high": g_high})
    for factor in io.FACTORS:
        y = clinical[factor].to_numpy(float)
        obs = ~np.isnan(y)
        oof, chosen = pr.ridge_cv(matrices, y, fold_list)
        yo, po = y[obs], oof[:, 0][obs]
        r2 = r2_score(yo, po)
        low, high = pr.bootstrap_ci(lambda idx: r2_score(yo[idx], po[idx]), len(yo), SEEDS.b5_bootstrap, skip_errors=True)
        alpha, rseed = pr.most_common(chosen), seed + 100 + io.FACTORS.index(factor)
        null = pr.ridge_permutation_null(matrices, y, fold_list, alpha, CFG.permutation.n_prediction, rseed)
        p, n_perm_r = stats.upper_p(null, r2), CFG.permutation.n_prediction
        if pr.needs_escalation(p):
            null = pr.ridge_permutation_null(matrices, y, fold_list, alpha, CFG.permutation.n_prediction_escalated, rseed + 1)
            p, n_perm_r = stats.upper_p(null, r2), CFG.permutation.n_prediction_escalated
        out["b5"].append({**common, "direction": "CGM to clinical variable", "target": factor, "metric": "R2", "value": r2, "ci_low": low,
                          "ci_high": high, "perm_p": p, "n_permutations": n_perm_r})
    return out


def pooled(cohort: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """B6: CGM groups fit on the 1,131 training participants of all strata; agreement with diagnosis and profiles."""
    frame = cohort.merge(labels[[io.PID, "label"]], on=io.PID, how="left")
    frame["combined_label"] = np.where(frame.label.notna(), frame.canonical_stratum + "__C" + frame.label.fillna(0).astype(int).astype(str), None)
    ids = frame[io.PID].tolist()
    feats_all = cf.build_features(io.load_cgm(ids), io.load_timezones(ids), ids)[CFG.cgm.full_window]
    feats = feats_all[feats_all.passes_coverage].reset_index(drop=True)
    frame = frame.set_index(io.PID).loc[feats[io.PID]].reset_index()
    is_train = (frame.split == "train").to_numpy()
    matrix = cg.fit_preprocessing(feats, is_train, "extended", CFG.cgm.full_window)["matrix"]
    _, k_sil = cg.silhouette_scan(matrix[is_train])
    ks = sorted(set(list(CFG.reverse.pooled_k_fixed) + [k_sil]))
    parts = {k: cg.kmeans_partition(matrix, is_train, feats[MEAN].to_numpy(float), k)["labels"] for k in ks}
    perm_rng, boot_rng = np.random.default_rng(SEEDS.b6_permutation), np.random.default_rng(SEEDS.b6_bootstrap)
    stratum, combined = frame.canonical_stratum.to_numpy(), frame.combined_label.notna().to_numpy()
    t2d = (frame.canonical_stratum == io.PRIMARY).to_numpy()
    rows, sizes = [], []
    for k, lab in parts.items():
        sizes += [{"pooled_k": k, "stratum": s, **{f"G{c}": int(r.get(c, 0)) for c in range(1, k + 1)}} for s, r in pd.crosstab(frame.canonical_stratum, lab).iterrows()]
        targets = {"diagnostic stratum (4)": (lab, stratum, None),
                   "stratum-profile label (10)": (lab[combined], frame.combined_label.to_numpy()[combined], None),
                   "stratum-profile label (10), within-stratum null": (lab[combined], frame.combined_label.to_numpy()[combined], stratum[combined]),
                   "C1/C2/C3 inside the 458": (lab[t2d], frame.label.to_numpy()[t2d].astype(int), None)}
        for target, (a, b, within) in targets.items():
            observed = ag.agreement(a, b)
            null = ag.permutation_null(a, b, perm_rng, N_PERM, within=within)
            ci = ag.bootstrap_agreement(a, b, boot_rng, N_BOOT) if within is None else {m: (np.nan, np.nan) for m in observed}
            rows += [{"pooled_k": k, "target": target, "n": len(a), "metric": m, "value": v, "ci_low": ci[m][0], "ci_high": ci[m][1],
                      "perm_p": stats.upper_p(null[m], v), "null_mean": float(null[m].mean()), "null_q975": float(np.percentile(null[m], 97.5))}
                     for m, v in observed.items()]
    return pd.DataFrame(rows), pd.DataFrame(sizes), feats_all, {"silhouette_selected_k": k_sil, "n_train": int(is_train.sum())}
