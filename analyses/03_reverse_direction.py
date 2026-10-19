"""03 Reverse direction (Section 3.2; Table 3; Fig 2 A to C).

Question: grouping the 458 participants by CGM features alone, do the groups match C1 to C3 and differ in
clinical variables, and how well does CGM predict the clinical variables and the profiles?
Inputs:   restricted/profile_labels, restricted/participant_metrics (01); forward eta squared (02); parquet (CGM).
Outputs:  restricted/reverse_* (features, labels, contingency by split); results/aggregate/reverse_*.csv;
          results/numbers/03_reverse.json.
Paper:    CGM group sizes and medians; ARI, Cramer's V; Table 3 reverse rows; R2 for each clinical variable;
          clinical-to-CGM R2 with and without HbA1c; profile AUROC and balanced accuracy, gain over HbA1c;
          "3 of 15" partition-by-subset cells; silhouette-preferred k; full-record refit stability (Fig S1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cgmprofiles import cgm_features as cf
from cgmprofiles import io
from cgmprofiles.cgm_metrics import MEAN, METRICS, WINDOWS
from cgmprofiles.config import get_config
from cgmprofiles.report import Timer, read_aggregate, read_restricted, write_aggregate, write_numbers, write_restricted

CFG = get_config()
CRIT = CFG.criteria
HBA1C = CFG.columns.hba1c
PRIMARY_NAME = "extended_window_full"
PRIMARY = f"kmeans3_{PRIMARY_NAME}"
GATE_TOLERANCE = CFG.gate_tolerance
from cgmprofiles.reverse import check, partitions, clinical_differences, partition_agreement, prediction


def main() -> None:
    timer = Timer()
    labels, metrics = read_restricted("profile_labels"), read_restricted("participant_metrics")
    t2d = io.ordered_stratum(labels, io.PRIMARY)
    ids, is_train, splits = t2d[io.PID].tolist(), (t2d.split == "train").to_numpy(), t2d.split.to_numpy()
    profile = t2d.label.to_numpy().astype(int)
    by_window = {w: metrics[(metrics.canonical_stratum == io.PRIMARY) & (metrics.window == w)].set_index(io.PID).loc[ids] for w in WINDOWS}
    check(abs(cf.mage_sine_check() - 100) < 1.0, "MAGE sine check")

    features = cf.build_features(io.load_cgm(ids), io.load_timezones(ids), ids)
    gap = max(float((features[w].set_index(io.PID)[m] - by_window[w][m]).abs().max()) for w in WINDOWS for m in METRICS)
    check(gap <= GATE_TOLERANCE, f"minimal features differ from Section 3.1 metrics by {gap}")
    check(all(features[w].passes_coverage.all() for w in WINDOWS), "participants below the CGM coverage floor")
    for w in WINDOWS:
        write_restricted(features[w], f"reverse_features_{w}")

    groups, diagnostics, centroids, sizes, stability = partitions(features, is_train, splits)
    write_restricted(groups.assign(**{io.PID: ids, "clinical_profile": profile}), "reverse_partition_labels")
    write_restricted(sizes, "reverse_partition_sizes_all")
    b3 = clinical_differences(groups, t2d[io.FACTORS].astype(float).reset_index(drop=True), is_train)
    b4, contingency = partition_agreement(groups, profile, splits)
    write_restricted(contingency, "reverse_contingency_by_split")
    forward = read_aggregate("forward_eta_squared_ci")
    forward = forward[forward.window.isin(WINDOWS)]
    primary_b3 = b3[(b3.partition == PRIMARY) & (b3.subset == "all")]
    fr = pd.concat([pd.DataFrame({"direction": "forward: CGM metric across C1/C2/C3", "window": forward.window, "variable": forward.metric,
                                  "hba1c_flag": False, "eta_squared": forward.eta_squared, "ci_low": forward.ci_low, "ci_high": forward.ci_high, "n": forward.n}),
                    pd.DataFrame({"direction": "reverse: clinical variable across G1/G2/G3 (extended, full record)", "window": WINDOWS[1],
                                  "variable": primary_b3.factor, "hba1c_flag": primary_b3.hba1c_flag, "eta_squared": primary_b3.eta_squared,
                                  "ci_low": primary_b3.ci_low, "ci_high": primary_b3.ci_high, "n": primary_b3.n})], ignore_index=True)
    b5, nulls, verify = prediction(t2d, features, {w: by_window[w] for w in WINDOWS})

    write_aggregate(diagnostics, "reverse_partition_diagnostics")
    write_aggregate(centroids[centroids.partition == PRIMARY_NAME], "reverse_primary_groups")
    write_aggregate(sizes[sizes.partition == PRIMARY_NAME], "reverse_primary_group_sizes")
    write_aggregate(contingency[(contingency.partition == PRIMARY) & (contingency.split == "all")], "reverse_primary_contingency")
    write_aggregate(b3, "reverse_clinical_differences")
    write_aggregate(fr, "forward_vs_reverse_eta_squared")
    write_aggregate(b4, "reverse_agreement")
    write_aggregate(b5, "reverse_prediction")
    write_aggregate(nulls, "reverse_prediction_null_draws")
    write_aggregate(pd.DataFrame({"statistic": ["mean", "q025", "q975"], "value": [stability.mean(), np.percentile(stability, 2.5), np.percentile(stability, 97.5)]}),
                    "reverse_primary_refit_stability")

    age = b3[(b3.factor == "participants_age")]
    cells = age.eta_squared.to_numpy()
    p_b4 = b4[b4.partition == PRIMARY].set_index("metric")
    p_b3 = primary_b3.set_index("factor")
    medians = centroids[centroids.partition == PRIMARY_NAME].set_index("cluster")
    cls = b5[b5.target == "C1/C2/C3"]
    pick = lambda fs, w, m: cls[(cls.feature_set == fs) & (cls.window == w) & (cls.metric == m)].iloc[0]  # noqa: E731
    ridge = b5[(b5.direction == "CGM to clinical variable") & (b5.feature_set == "extended") & (b5.window == WINDOWS[1])].set_index("target")
    m2 = b5[b5.direction == "clinical to CGM"].set_index(["window", "target", "feature_set"]).value
    numbers = {
        "group_sizes": medians.n.to_dict(), "median_mean_glucose": medians[f"median_{MEAN}"].to_dict(),
        "median_cv": medians["median_coefficient_of_variation_pct"].to_dict(),
        "silhouette": diagnostics.set_index("partition").loc[PRIMARY_NAME, ["silhouette_k2", "silhouette_k3", "silhouette_selected_k"]].to_dict(),
        "ari": p_b4.loc["ARI", ["value", "ci_low", "ci_high", "perm_p"]].to_dict(), "cramers_v": p_b4.loc["CramersV", "value"],
        "reverse_eta": {f: p_b3.loc[f, ["eta_squared", "ci_low", "ci_high"]].tolist() for f in io.FACTORS},
        "non_hba1c_eta_range": [p_b3.drop(index=HBA1C).eta_squared.min(), p_b3.drop(index=HBA1C).eta_squared.max()],
        "age_cells_at_or_above_medium": int((cells >= CRIT.medium_eta).sum()), "age_cells_total": int(len(cells)),
        "cgm_to_clinical_r2": {f: ridge.loc[f, ["value", "ci_low", "ci_high", "perm_p"]].tolist() for f in io.FACTORS},
        "clinical_to_cgm_r2": {f"{w}/{m}": [m2[(w, m, "six variables")], m2[(w, m, "five variables (no HbA1c)")]] for w in WINDOWS for m in METRICS},
        "profile_auroc": pick("extended", WINDOWS[1], "macro_auroc")[["value", "ci_low", "ci_high", "perm_p"]].tolist(),
        "profile_balanced_accuracy": pick("extended", WINDOWS[1], "balanced_accuracy")[["value", "ci_low", "ci_high", "perm_p"]].tolist(),
        "hba1c_alone_auroc": pick("HbA1c alone", "", "macro_auroc").value,
        "hba1c_plus_cgm_auroc": pick("extended", WINDOWS[1], "macro_auroc gain, HbA1c plus CGM minus HbA1c alone").hba1c_plus_cgm,
        "auroc_gain": pick("extended", WINDOWS[1], "macro_auroc gain, HbA1c plus CGM minus HbA1c alone")[["value", "ci_low", "ci_high"]].tolist(),
        "chance_balanced_accuracy": 1 / 3, "refit_stability_mean": float(stability.mean()),
        "section32_r2_max_difference": max(verify.values()), "runtime_s": timer.seconds()}
    write_numbers("03_reverse", numbers)
    print(f"03 done in {timer.seconds()} s")


if __name__ == "__main__":
    main()
