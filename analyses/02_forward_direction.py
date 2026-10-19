"""02 Forward direction (Section 3.1; Table 2; Fig 1 values).

Question: do the clinical profiles differ in mean glucose, time in range and CV (first 48 h, full record)?
Inputs:   restricted/profile_labels, restricted/participant_metrics (from 01).
Outputs:  results/aggregate/forward_*.csv, results/numbers/02_forward.json.
Paper:    Table 2 (eta squared with 2,000-resample CIs, pattern); Section 3.1 pairwise differences (1,000 resamples,
          record convention); Dunn-Holm; sensitivity subsets (training only, held-out, margin > 0.20, complete case),
          ANOVA; split dependence (2,000 stratified 325/133 partitions); GLP-1/GIP-stratified C2 contrasts.
"""
from __future__ import annotations


import pandas as pd

from cgmprofiles import io
from cgmprofiles.cgm_metrics import CV, METRICS, WINDOWS
from cgmprofiles.config import get_config
from cgmprofiles.report import Timer, read_restricted, write_aggregate, write_numbers

CFG = get_config()
C2 = CFG.profiles.c2
from cgmprofiles.forward import forward_tests, c2_vs_rest, eta_intervals_and_tests, sensitivity_subsets, split_dependence, glp1_contrasts


def main() -> None:
    timer = Timer()
    labels, metrics = read_restricted("profile_labels"), read_restricted("participant_metrics")
    t2d = io.ordered_stratum(labels, io.PRIMARY)
    t2d_metrics = metrics[metrics.canonical_stratum == io.PRIMARY]
    by_window = {w: t2d_metrics[t2d_metrics.window == w].set_index(io.PID).loc[t2d[io.PID]].reset_index() for w in WINDOWS}
    t2d_metrics = pd.concat(by_window.values(), keys=WINDOWS, names=["window_key"]).reset_index(drop=True)

    kw, pairs = forward_tests(metrics)
    c2rest = c2_vs_rest(t2d_metrics)
    eta_ci, dunn = eta_intervals_and_tests(t2d, by_window)
    subsets = sensitivity_subsets(by_window)
    split = split_dependence(t2d, by_window)
    glp1 = glp1_contrasts(by_window, io.load_medications(t2d[io.PID].tolist())[CFG.columns.glp1].to_numpy())
    for frame, name in ((kw, "forward_kruskal_wallis"), (pairs, "forward_pairwise_differences"), (c2rest, "forward_c2_vs_rest"),
                        (eta_ci, "forward_eta_squared_ci"), (dunn, "forward_dunn_holm"), (subsets, "forward_sensitivity_subsets"),
                        (split, "forward_split_dependence"), (glp1, "forward_glp1_contrasts")):
        write_aggregate(frame, name)

    t2d_kw = kw[kw.canonical_stratum == io.PRIMARY].set_index(["window", "metric"])
    t2d_pairs = pairs[pairs.canonical_stratum == io.PRIMARY].set_index(["window", "metric", "cluster_a", "cluster_b"])
    pick = lambda w, m, a, b: t2d_pairs.loc[(w, m, a, b), ["mean_diff_a_minus_b", "ci_low", "ci_high"]].tolist()  # noqa: E731
    sub = subsets.set_index(["subset", "window", "metric"]).eta_squared
    numbers = {
        "eta_squared": {f"{w}/{m}": t2d_kw.loc[(w, m), "eta_squared"] for w in WINDOWS for m in METRICS},
        "eta_squared_ci": {f"{r.window}/{r.metric}": [r.eta_squared, r.ci_low, r.ci_high] for r in eta_ci.itertuples() if r.window in WINDOWS},
        "max_p_t2d": float(t2d_kw.p_value.max()),
        "dunn_holm_all_pairs_same_verdict": bool(((dunn.p_holm_within_3 < CFG.criteria.alpha) == (dunn.p_holm_across_18 < CFG.criteria.alpha)).all()),
        "cv_c1_minus_c3": {w: pick(w, CV, 1, 3) for w in WINDOWS}, "cv_c1_minus_c2": {w: pick(w, CV, 1, 2) for w in WINDOWS},
        "cv_c2_minus_c3": {w: pick(w, CV, 2, 3) for w in WINDOWS},
        "cv_eta_training_only": {w: sub[("training only", w, CV)] for w in WINDOWS},
        "cv_eta_heldout": {w: sub[("validation and test only", w, CV)] for w in WINDOWS},
        "split_dependence_p": {f"{r.window}/{r.metric}": r.p_two_sided for r in split.itertuples()},
        "glp1_cv_c2_minus_c1": {f"{r.window}/{r.subset}": [r.mean_diff, r.ci_low, r.ci_high] for r in glp1.itertuples() if r.contrast == "C2 minus C1"},
        "c2_vs_rest": {r.window: [r.mean_diff, r.ci_low, r.ci_high, r.eta_squared] for r in c2rest.itertuples()},
        "n_resamples_pairwise": CFG.bootstrap.n_pairwise, "n_resamples_eta": CFG.bootstrap.n_default, "runtime_s": timer.seconds()}
    write_numbers("02_forward", numbers)
    print(f"02 done in {timer.seconds()} s")


if __name__ == "__main__":
    main()
