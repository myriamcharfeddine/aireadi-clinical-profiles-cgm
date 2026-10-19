"""05 Temporal reproducibility (Methods 2.3; temporal paragraph of Section 3.2; Supplementary Fig S1).

Question: are CGM groups reproducible across two non-overlapping 3-day windows (days 1 to 3, days 4 to 6), and does
their agreement with C1 to C3 stay low in each window? Within the monitoring period, not long-term stability.
Inputs:   restricted/profile_labels, restricted/participant_metrics (01); parquet (CGM, site).
Outputs:  restricted/temporal_*; results/aggregate/temporal_*.csv; results/numbers/05_temporal.json.
Paper:    Abstract and Discussion: ARI 0.42 between windows (held-out), ARI 0.030 and 0.036 with the profiles;
          Fig S1 (A ARI, B ICC by feature for 3- and 4-day windows, C C2 minus rest CV per window and full record).
Disclosure: eligible counts are masked (some are 1 to 4 or would reveal them); a statistic whose n is masked is
          rounded (2 decimals; percent agreement to whole percent) before it is tracked.
"""
from __future__ import annotations

import pandas as pd

from cgmprofiles import agreement as ag
from cgmprofiles import cgm_groups as cg
from cgmprofiles import io, windows as wn
from cgmprofiles.cgm_metrics import CV
from cgmprofiles.config import get_config
from cgmprofiles.disclosure import MASK_COMPLEMENT, is_small, masked_count
from cgmprofiles.report import Timer, read_restricted, write_aggregate, write_numbers, write_restricted

CFG = get_config()
TCFG, SEEDS = CFG.temporal, CFG.seeds
SITE, C2 = CFG.columns.site, CFG.profiles.c2
GATE_TOLERANCE = CFG.gate_tolerance
from cgmprofiles.temporal import window_features, feature_gate, t3_agreement, c2_contrast, masked


def main() -> None:
    timer = Timer()
    labels, metrics = read_restricted("profile_labels"), read_restricted("participant_metrics")
    t2d = io.ordered_stratum(labels, io.PRIMARY)
    ids = t2d[io.PID].tolist()
    t2d = t2d.join(io.load_static([SITE]), on=io.PID)
    t2d["profile"] = "C" + t2d.label.astype(int).astype(str)
    full = metrics[(metrics.canonical_stratum == io.PRIMARY) & (metrics.window == CFG.cgm.full_window)].set_index(io.PID)
    t2d["full_cv"] = t2d[io.PID].map(full[CV])
    cgm, zones = io.load_cgm(ids), io.load_timezones(ids)

    agreement_rows, icc_rows, t3_rows, contrast_rows, reference_rows, mask_rows, shown = [], [], [], [], [], [], {}
    for design, spec in CFG.temporal.designs.raw.items():
        offset = spec["seed_offset"]
        frame = t2d.merge(wn.eligibility(cgm, design), on=io.PID, how="left", validate="one_to_one")
        eligible = frame[frame.eligible].reset_index(drop=True)
        e_ids, is_train = eligible[io.PID].tolist(), (eligible.split == "train").to_numpy()
        feats = window_features(cgm, design, zones, e_ids)
        for name, f in feats.items():
            write_restricted(f, f"temporal_features_{design}_{name}")
        gap = feature_gate(cgm, design, e_ids, feats)
        if gap > GATE_TOLERANCE:
            raise SystemExit(f"Stop: {design} window features differ from a direct recomputation by {gap}")
        frozen = cg.fit_frozen(feats["A"], is_train, CFG.cgm.full_window)
        labels_a = frozen["partition"]["labels"]
        labels_b = cg.nearest_labels(cg.apply_frozen(frozen["info"], feats["B"]), frozen["partition"]["centroids"])
        if min((is_train & (labels_a == g)).sum() for g in wn.GROUPS) < CFG.cgm_groups.min_cluster_n:
            raise SystemExit(f"Stop: a {design} CGM group has fewer than 20 training participants in window A")
        write_restricted(pd.DataFrame({io.PID: e_ids, "split": eligible.split, "profile": eligible.profile, "G_window_A": labels_a,
                                       "G_window_B_frozen": labels_b}), f"temporal_labels_{design}")

        problem, _ = wn.eligibility_problem(frame, SITE, len(t2d))
        problem.solve()
        problem.fixed.update(k for k, state in problem.state.items() if state == "")
        wn.add_transition_cells(problem, labels_a, labels_b, is_train)
        problem.solve()
        mask_rows.append({"design": design, **problem.check()})
        shown[design] = {s: problem.shown((key, "eligible")) for s, key in wn.SETS.items()} | {"train": problem.shown((wn.ROW_TRAIN, "eligible"))}

        heldout = ~is_train
        result = ag.window_agreement(labels_a[heldout], labels_b[heldout], len(wn.GROUPS), SEEDS.t2_bootstrap + offset * 10, SEEDS.t2_permutation + offset * 10)
        for metric, r in result.items():
            agreement_rows.append({"design": design, "set": "held-out", "n": shown[design]["held-out"], "variant": "frozen window A pipeline", "metric": metric, **r})
        info = frozen["info"]
        xa, xb = cg.clustering_scale(info, feats["A"]), cg.clustering_scale(info, feats["B"])
        for f_i, feature in enumerate(info["columns"]):
            r = ag.icc_with_ci(xa[feature].to_numpy(float), xb[feature].to_numpy(float), SEEDS.t2_icc_bootstrap + offset * 1000 + f_i)
            n_shown = not masked(shown[design]["all eligible"])
            r["n_pairs"] = r["n_pairs"] if n_shown and not is_small(r["n_dropped_missing"]) else MASK_COMPLEMENT
            r["n_dropped_missing"] = masked_count(r["n_dropped_missing"]) if n_shown else MASK_COMPLEMENT
            icc_rows.append({"design": design, "set": "all eligible", "scale": "clustering scale", "feature": feature, "log1p": feature in info["log1p"], **r})

        profile = eligible.label.to_numpy(int)
        t3_rows += [{"design": design, **row} for row in t3_agreement(labels_a, labels_b, profile, offset)]
        cv = {name: feats[name][CV].to_numpy(float) for name in wn.WINDOW_NAMES}
        row, ref = c2_contrast(cv["A"], cv["B"], eligible.full_cv.to_numpy(float), profile, offset)
        contrast_rows.append({"design": design, **row})
        reference_rows.append({"design": design, **ref})

    agreement = pd.DataFrame(agreement_rows)
    write_restricted(agreement, "temporal_agreement_unrounded")
    columns = ["value", "ci_low", "ci_high", "null_mean", "null_q975"]
    is_masked = agreement["n"].map(masked)
    for metric in ("percent_agreement", "kappa", "ARI", "NMI"):
        decimals = TCFG.masked_percent_decimals if metric == "percent_agreement" else TCFG.masked_decimals
        rows = is_masked & (agreement.metric == metric)
        agreement.loc[rows, columns] = agreement.loc[rows, columns].round(decimals)
    agreement["rounded_because_n_masked"] = is_masked
    t3, contrasts, references = pd.DataFrame(t3_rows), pd.DataFrame(contrast_rows), pd.DataFrame(reference_rows)
    for frame, name in ((agreement, "temporal_agreement"), (pd.DataFrame(icc_rows), "temporal_icc"), (t3, "temporal_profile_agreement"),
                        (contrasts, "temporal_c2_contrast"), (references, "temporal_c2_full_record"), (pd.DataFrame(mask_rows), "temporal_masking_check")):
        write_aggregate(frame, name)
    if any(r["n_recoverable"] for r in mask_rows):
        raise SystemExit("Stop: a masked temporal count is recoverable")

    primary = TCFG.primary
    pick = lambda m: agreement[(agreement.design == primary) & (agreement.metric == m)].iloc[0]  # noqa: E731
    t3p = t3[(t3.design == primary) & (t3.statistic == "ARI")].iloc[0]
    c = contrasts[contrasts.design == primary].iloc[0]
    write_numbers("05_temporal", {
        "heldout_ari": pick("ARI")[["value", "ci_low", "ci_high"]].tolist(), "heldout_kappa": pick("kappa")[["value", "ci_low", "ci_high"]].tolist(),
        "heldout_ari_rounded_because_n_masked": bool(pick("ARI").rounded_because_n_masked),
        "profile_ari_window_a": t3p.A_value, "profile_ari_window_b": t3p.B_value,
        "c2_minus_rest_cv": {"A": [c.A_difference, c.A_ci_low, c.A_ci_high], "B": [c.B_difference, c.B_ci_low, c.B_ci_high]},
        "shown_counts": shown, "runtime_s": timer.seconds()})
    print(f"05 done in {timer.seconds()} s")


if __name__ == "__main__":
    main()
