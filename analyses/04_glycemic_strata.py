"""04 Other glycemic strata and the pooled cohort (Section 3.3; Fig 3; Fig 2D).

Question: does the correspondence differ in healthy, prediabetes and insulin-treated participants; does the
C2-associated lower CV transfer to participants assigned to the non-insulin T2D centroids; do pooled CGM
groups follow diagnosis more than profile?
Inputs:   restricted/profile_labels, restricted/participant_metrics (01); forward results (02); parquet.
Outputs:  restricted/strata_*; results/aggregate/strata_*.csv and pooled_*.csv; results/numbers/04_strata.json.
Paper:    Section 3.3 eta squared, specificity differences and Holm p, ARI and AUROC per stratum, insulin-treated
          C-peptide, pooled ARI with diagnosis and combined labels (Fig 2D).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cgmprofiles import cgm_features as cf
from cgmprofiles import io, profiles, stats
from cgmprofiles.cgm_metrics import CV, MEAN, METRICS, WINDOWS
from cgmprofiles.config import get_config
from cgmprofiles.report import Timer, read_aggregate, read_restricted, write_aggregate, write_numbers, write_restricted

CFG = get_config()
SCFG = CFG.strata
HBA1C, C2 = CFG.columns.hba1c, CFG.profiles.c2
GATE_TOLERANCE = CFG.gate_tolerance
from cgmprofiles.strata import check, forward_other_strata, specificity, reverse_one, pooled


def main() -> None:
    timer = Timer()
    labels, metrics = read_restricted("profile_labels"), read_restricted("participant_metrics")
    split, clinical = io.load_split(), io.load_clinical_frame()
    cohort = split.merge(clinical, on=io.PID, how="left", validate="one_to_one")
    _, models = profiles.build_profiles(clinical, split)

    s1 = forward_other_strata(metrics)
    spec, s3_summary = specificity(cohort, labels, models[io.PRIMARY])
    b6, b6_sizes, pooled_features, b6_info = pooled(cohort, labels)
    write_restricted(pooled_features, "pooled_features_window_full")
    write_restricted(b6_sizes, "pooled_cluster_by_stratum")
    full = metrics[metrics.window == CFG.cgm.full_window].drop_duplicates(io.PID).set_index(io.PID)
    gap = max(float((pooled_features.set_index(io.PID).loc[full.index, m] - full[m]).abs().max()) for m in METRICS)
    check(gap <= GATE_TOLERANCE, f"pooled features differ from Section 3.1 metrics by {gap}")

    results = []
    for stratum in SCFG.other:
        sub = io.ordered_stratum(labels, stratum)
        features = pooled_features.set_index(io.PID).loc[sub[io.PID]].reset_index()
        results.append(reverse_one(stratum, sub, features, CFG.cgm.full_window, SCFG.primary_k[stratum], with_prediction=True))
        ids = sub[io.PID].tolist()
        f48 = cf.build_features(io.load_cgm(ids), io.load_timezones(ids), ids)[CFG.cgm.short_window]
        keep = f48.passes_coverage.to_numpy()
        results.append(reverse_one(stratum, sub[keep].reset_index(drop=True), f48[keep].reset_index(drop=True), CFG.cgm.short_window,
                                   SCFG.primary_k[stratum], with_prediction=False))
    b3 = pd.DataFrame([row for r in results for row in r["b3"]])
    b4 = pd.DataFrame([row for r in results for row in r["b4"]])
    b5 = pd.DataFrame([row for r in results for row in r["b5"]])
    primary = b3.window == CFG.cgm.full_window
    b3["p_holm"] = np.nan
    for _, idx in b3[primary & ~b3.descriptive_only].groupby("factor").groups.items():
        b3.loc[idx, "p_holm"] = stats.holm(b3.loc[idx, "p_value"].tolist())
    b4["p_holm"] = np.nan
    ari_idx = b4[(b4.metric == "ARI") & (b4.window == CFG.cgm.full_window) & ~b4.descriptive_only].index
    b4.loc[ari_idx, "p_holm"] = stats.holm(b4.loc[ari_idx, "perm_p"].tolist())
    b5["p_holm"] = np.nan
    for _, idx in [("auroc", b5[b5.metric == "macro_auroc"].index), *b5[b5.direction == "CGM to clinical variable"].groupby("target").groups.items()]:
        b5.loc[idx, "p_holm"] = stats.holm(b5.loc[idx, "perm_p"].tolist())
    write_restricted(pd.DataFrame([r["sizes"] for r in results]), "strata_partition_sizes")

    for frame, name in ((s1, "strata_forward"), (spec, "strata_specificity"), (s3_summary, "strata_specificity_assignment"), (b3, "strata_clinical_differences"),
                        (b4, "strata_agreement"), (b5, "strata_prediction"), (b6, "pooled_agreement")):
        write_aggregate(frame, name)

    t2d_ref = read_aggregate("forward_c2_vs_rest").set_index("window")
    cv_rows = spec[(spec.variant == "primary (T2D scaler, all)") & (spec.metric == CV)].set_index(["canonical_stratum", "window"])
    s1i = s1.set_index(["canonical_stratum", "window", "metric"])
    b3i, b4i = b3.set_index(["partition", "factor"]), b4.set_index(["partition", "metric"])
    auc = b5[b5.metric == "macro_auroc"].set_index("stratum")
    b6i = b6[b6.metric == "ARI"].set_index(["pooled_k", "target"])
    t2d_only = read_aggregate("reverse_agreement").set_index(["partition", "metric"]).loc[("kmeans3_extended_window_full", "ARI"), "value"]
    full_w, short_w = CFG.cgm.full_window, CFG.cgm.short_window
    numbers = {
        "forward_mean_glucose_full": {s: s1i.loc[(s, full_w, MEAN), ["eta_squared", "ci_low", "ci_high"]].tolist() for s in SCFG.other},
        "forward_cv_max_healthy_prediabetes": float(s1i.loc[[(s, w, CV) for s in SCFG.specificity_strata for w in WINDOWS], "eta_squared"].max()),
        "assigned_c2": s3_summary.set_index("canonical_stratum").assigned_C2.to_dict(),
        "cv_c2_minus_rest": {f"{s}/{w}": cv_rows.loc[(s, w), ["mean_diff_c2_minus_rest", "diff_ci_low", "diff_ci_high", "p_value", "p_holm"]].tolist()
                             for s in SCFG.specificity_strata for w in WINDOWS},
        "cv_c2_minus_rest_t2d": {w: t2d_ref.loc[w, ["mean_diff", "ci_low", "ci_high"]].tolist() for w in WINDOWS},
        "prediabetes_share_of_t2d": {w: abs(cv_rows.loc[("pre_diabetes", w), "mean_diff_c2_minus_rest"]) / abs(t2d_ref.loc[w, "mean_diff"]) for w in WINDOWS},
        "ari_other_strata": {s: b4i.loc[(f"{s}__{full_w}__k{SCFG.primary_k[s]}", "ARI"), "value"] for s in SCFG.other},
        "auroc": {s: [auc.loc[s, "value"], auc.loc[s, "hba1c_alone"]] for s in SCFG.other},
        "insulin_c_peptide": {w: b3i.loc[(f"insulin_dependent__{w}__k3", "c_peptide_ngml_baseline"), ["eta_squared", "ci_low", "ci_high"]].tolist()
                              for w in (full_w, short_w)},
        "pooled_ari_diagnosis_k3": b6i.loc[(3, "diagnostic stratum (4)"), ["value", "ci_low", "ci_high", "perm_p"]].tolist(),
        "pooled_ari_t2d_profiles_k3": b6i.loc[(3, "C1/C2/C3 inside the 458"), "value"], "t2d_only_ari": t2d_only,
        "pooled_ari_combined_k3": b6i.loc[(3, "stratum-profile label (10)"), "value"],
        "pooled_within_null": b6i.loc[(3, "stratum-profile label (10), within-stratum null"), ["null_mean", "perm_p"]].tolist(),
        "pooled": b6_info, "runtime_s": timer.seconds()}
    write_numbers("04_strata", numbers)
    print(f"04 done in {timer.seconds()} s")


if __name__ == "__main__":
    main()
