"""Compare every aggregate output with the record repository (Phase 3 verification).

    python tools/compare_with_record.py --record /path/to/clinical-profiles-paper-repro

For each pair (new output, record output) the rows are matched on key columns and the largest
absolute difference is reported per value column. Deterministic quantities must agree to 1e-9;
bootstrap and permutation quantities must agree exactly when the RNG order is preserved (reported
with the same 1e-9 tolerance). Writes results/verification/record_comparison.csv; prints a summary.
The record's participant-level files are read from its restricted/ folder (never copied).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "aggregate"
RESTRICTED = ROOT / "restricted"
TOLERANCE = 1e-9
RESAMPLING_WORDS = ("ci_", "null", "perm", "p_permutation", "q025", "q975", "p_two_sided", "_low", "_high", "draw", "stability")


def kind(column: str) -> str:
    return "bootstrap/permutation" if any(w in column for w in RESAMPLING_WORDS) else "deterministic"


def compare(name: str, new: pd.DataFrame, record: pd.DataFrame, keys: list[str], columns: dict[str, str]) -> list[dict]:
    merged = new.merge(record, on=keys, how="inner", suffixes=("", "__record"))
    rows = []
    for new_col, rec_col in columns.items():
        rec_name = rec_col if rec_col != new_col else f"{rec_col}__record"
        a = pd.to_numeric(merged[new_col], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[rec_name], errors="coerce").to_numpy(float)
        both = ~(np.isnan(a) & np.isnan(b))
        diff = np.abs(a[both] - b[both])
        diff = np.where(np.isnan(diff), np.inf, diff)
        worst = float(diff.max()) if len(diff) else float("nan")
        rows.append({"comparison": name, "column": new_col, "kind": kind(new_col), "rows_new": len(new), "rows_record": len(record),
                     "rows_matched": len(merged), "values_compared": int(both.sum()), "max_abs_difference": worst,
                     "match": bool(len(merged) and worst <= TOLERANCE)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path, help="root of the record repository (read-only)")
    rec = parser.parse_args().record
    out_dir = rec / "01_raw_cgm_differences"
    rv, st, tp, au = out_dir / "reverse_direction/outputs", out_dir / "strata/outputs", out_dir / "temporal/outputs", out_dir / "audit"
    read = lambda p: pd.read_csv(p, low_memory=False)  # noqa: E731
    new = lambda n: pd.read_csv(AGG / f"{n}.csv", low_memory=False)  # noqa: E731
    rows = []

    # 01 profiles and participant metrics (participant level, restricted on both sides)
    labels = pd.read_parquet(RESTRICTED / "participant_metrics.parquet")
    labels["participant_id"] = labels["participant_id"].astype(str)
    ref = read(rec / "restricted/01_raw_cgm_differences/audit/recomputed_participant_metrics.csv").astype({"participant_id": str})
    rows += compare("01 labels, margins and CGM metrics (participant level)", labels, ref, ["participant_id", "window"],
                    {c: c for c in ["label", "nearest_distance", "relative_margin", "n_readings", "mean_glucose", "time_in_range_pct", "coefficient_of_variation_pct"]})

    # 02 forward
    rob = read(au / "ROBUSTNESS_31.csv")
    rows += compare("02 Kruskal-Wallis, all strata", new("forward_kruskal_wallis"), read(out_dir / "outputs/tables/kruskal_wallis_effect_sizes.csv"),
                    ["canonical_stratum", "window", "metric"], {"kruskal_h": "kruskal_h", "p_value": "p_value", "eta_squared": "eta_squared"})
    rows += compare("02 pairwise differences (1,000 resamples)", new("forward_pairwise_differences"),
                    read(out_dir / "outputs/tables/pairwise_bootstrap_mean_differences.csv"),
                    ["canonical_stratum", "window", "metric", "cluster_a", "cluster_b"], {"mean_diff_a_minus_b": "mean_diff_a_minus_b", "ci_low": "ci_low", "ci_high": "ci_high"})
    rows += compare("02 C2 versus rest", new("forward_c2_vs_rest"), read(out_dir / "outputs/tables/t2d_c2_vs_rest_cv_binary_test.csv"), ["window"],
                    {"mean_diff": "mean_diff_c2_minus_rest", "ci_low": "ci_low", "ci_high": "ci_high", "eta_squared": "eta_squared", "p_value": "p_value"})
    eta_rec = rob[rob.analysis == "eta_squared_bootstrap_ci"].rename(columns={"value": "eta_rec", "ci_low": "ci_low_rec", "ci_high": "ci_high_rec"})
    rows += compare("02 eta squared, 2,000-resample CIs (Table 2)", new("forward_eta_squared_ci"), eta_rec, ["window", "metric"],
                    {"eta_squared": "eta_rec", "ci_low": "ci_low_rec", "ci_high": "ci_high_rec"})
    anova = rob[(rob.analysis == "test_choice") & (rob.statistic == "one-way ANOVA eta squared")].rename(columns={"value": "anova_rec", "p_value": "anova_p_rec"})
    rows += compare("02 ANOVA", new("forward_eta_squared_ci"), anova, ["window", "metric"], {"anova_eta_squared": "anova_rec", "anova_p": "anova_p_rec"})
    dunn = rob[rob.analysis == "dunn_holm"].assign(pair=lambda f: f.statistic.str.extract(r"(C\d vs C\d)")[0]).rename(columns={"value": "z_rec", "p_value": "p_rec"})
    rows += compare("02 Dunn tests", new("forward_dunn_holm"), dunn, ["window", "metric", "pair"], {"z": "z_rec", "p_raw": "p_rec"})
    names = {"all 458 (paper)": "all", "train only": "training only", "validation and test only": "validation and test only",
             "relative assignment margin > 0.20": "relative assignment margin > 0.20", "complete-case clinical variables": "complete-case clinical variables"}
    subs = rob[rob.analysis == "eta_squared_subset"].assign(subset=lambda f: f.subset.map(names)).rename(columns={"value": "eta_rec", "p_value": "p_rec"})
    rows += compare("02 sensitivity subsets", new("forward_sensitivity_subsets"), subs, ["subset", "window", "metric"], {"eta_squared": "eta_rec", "p_value": "p_rec"})
    split = rob[rob.analysis == "a5_split_dependence"].rename(columns={"value": "diff_rec", "ci_low": "q025_rec", "ci_high": "q975_rec", "p_value": "p_rec"})
    rows += compare("02 split dependence", new("forward_split_dependence"), split, ["window", "metric"],
                    {"difference": "diff_rec", "null_q025": "q025_rec", "null_q975": "q975_rec", "p_two_sided": "p_rec"})
    glp = rob[rob.analysis == "a5_cv_within_glp1_stratum"].assign(contrast=lambda f: f.statistic.str.extract(r"(C2 minus C\d)")[0])
    rows += compare("02 GLP-1/GIP contrasts", new("forward_glp1_contrasts"), glp.rename(columns={"value": "d_rec", "ci_low": "l_rec", "ci_high": "h_rec"}),
                    ["subset", "window", "contrast"], {"mean_diff": "d_rec", "ci_low": "l_rec", "ci_high": "h_rec"})

    # 03 reverse
    diag = read(rv / "B2_partition_diagnostics.csv")
    rows += compare("03 silhouette by k", new("reverse_partition_diagnostics"), diag, ["partition"],
                    {f"silhouette_k{k}": f"silhouette_k{k}" for k in range(2, 7)} | {"train_silhouette_k3": "train_silhouette_k3", "silhouette_selected_k": "silhouette_selected_k"})
    stab = new("reverse_primary_refit_stability").set_index("statistic").value
    stab_rec = diag.set_index("partition").loc["extended_window_full"]
    rows += compare("03 refit stability (primary)", pd.DataFrame({"k": [0], "stability_mean": [stab["mean"]], "stability_q025": [stab["q025"]], "stability_q975": [stab["q975"]]}),
                    pd.DataFrame({"k": [0], "m": [stab_rec.stability_ari_mean], "l": [stab_rec.stability_ari_q025], "h": [stab_rec.stability_ari_q975]}), ["k"],
                    {"stability_mean": "m", "stability_q025": "l", "stability_q975": "h"})
    cent = new("reverse_primary_groups")
    cent_cols = [c for c in cent.columns if c.startswith(("z_", "median_")) or c == "n"]
    rows += compare("03 primary CGM groups (sizes, medians, centroids)", cent, read(rv / "B2_centroids.csv"), ["partition", "cluster"], {c: c for c in cent_cols})
    cont = new("reverse_primary_contingency")
    rows += compare("03 primary contingency (Fig 2A)", cont, read(rv / "B4_contingency_tables.csv"), ["partition", "split", "cgm_cluster"], {c: c for c in ["C1", "C2", "C3"]})
    b3_rec = read(rv / "B3_clinical_by_cgm_cluster.csv")
    rows += compare("03 B3 clinical differences", new("reverse_clinical_differences"), b3_rec[b3_rec.data == "observed"], ["partition", "factor", "subset"],
                    {c: c for c in ["n", "eta_squared", "ci_low", "ci_high", "p_value"]})
    rows += compare("03 forward versus reverse (Table 3)", new("forward_vs_reverse_eta_squared"), read(rv / "B3_forward_vs_reverse_eta_squared.csv"),
                    ["direction", "window", "variable"], {c: c for c in ["eta_squared", "ci_low", "ci_high", "n"]})
    rows += compare("03 B4 agreement", new("reverse_agreement"), read(rv / "B4_agreement.csv"), ["partition", "metric"],
                    {c: c for c in ["value", "ci_low", "ci_high", "perm_p", "perm_p_within_split", "null_mean", "null_q975"]})
    b5 = new("reverse_prediction").fillna({"window": "none"})
    rows += compare("03 B5 prediction", b5, read(rv / "B5_prediction.csv").fillna({"window": "none"}), ["direction", "feature_set", "window", "target", "metric"],
                    {c: c for c in ["value", "ci_low", "ci_high", "perm_p", "n_permutations"]})
    rows += compare("03 B5 permutation null draws (Fig 2C)", new("reverse_prediction_null_draws"), read(rv / "B5_primary_null_draws.csv"), ["analysis", "draw"], {"value": "value"})

    # 04 strata and pooled
    rows += compare("04 S1 forward, other strata", new("strata_forward"), read(st / "S1_forward.csv"), ["canonical_stratum", "window", "metric"],
                    {"eta_squared": "eta_squared", "ci_low": "ci_low", "ci_high": "ci_high", "p_value": "p_value", "p_holm": "p_holm_F1"})
    rows += compare("04 S3 specificity", new("strata_specificity"), read(st / "S3_specificity.csv"), ["canonical_stratum", "variant", "window", "metric"],
                    {c: c for c in ["n_c2", "n_rest", "eta_squared", "eta_ci_low", "eta_ci_high", "p_value", "mean_diff_c2_minus_rest", "diff_ci_low", "diff_ci_high"]}
                    | {"p_holm": "p_holm_F6"})
    rows += compare("04 S3 assignment", new("strata_specificity_assignment"), read(st / "S3_assignment.csv"), ["canonical_stratum"],
                    {"n": "n", "assigned_C2": "assigned_C2", "out_of_range": "out_of_range"})
    s2_b3 = pd.concat([read(st / "S2_clinical_full.csv"), read(st / "S2_clinical_secondary.csv")])
    rows += compare("04 S2 clinical differences", new("strata_clinical_differences"), s2_b3, ["partition", "factor"],
                    {c: c for c in ["n", "eta_squared", "ci_low", "ci_high", "p_value"]} | {"p_holm": "p_holm_F2"})
    s2_b4 = pd.concat([read(st / "S2_agreement_full.csv"), read(st / "S2_agreement_secondary.csv")])
    rows += compare("04 S2 agreement", new("strata_agreement"), s2_b4, ["partition", "metric"],
                    {c: c for c in ["value", "ci_low", "ci_high", "perm_p", "null_mean", "null_q975"]} | {"p_holm": "p_holm_F3"})
    s2_b5 = read(st / "S2_prediction.csv")
    rows += compare("04 S2 prediction", new("strata_prediction"), s2_b5, ["partition", "target", "metric"],
                    {c: c for c in ["value", "ci_low", "ci_high", "perm_p", "n_permutations", "hba1c_alone", "hba1c_plus_cgm", "gain_over_hba1c",
                                    "gain_ci_low", "gain_ci_high", "p_holm"]})
    rows += compare("04 B6 pooled agreement (Fig 2D)", new("pooled_agreement"), read(rv / "B6_agreement.csv"), ["pooled_k", "target", "metric"],
                    {c: c for c in ["n", "value", "ci_low", "ci_high", "perm_p", "null_mean", "null_q975"]})

    # 05 temporal (unrounded agreement from the record's restricted folder)
    t2_new = pd.read_parquet(RESTRICTED / "temporal_agreement_unrounded.parquet")
    t2_rec = read(rec / "restricted/01_raw_cgm_differences/temporal/T2_agreement_unrounded.csv")
    rows += compare("05 held-out agreement between windows (unrounded)", t2_new.drop(columns="n"), t2_rec.drop(columns="n"), ["design", "set", "variant", "metric"],
                    {c: c for c in ["value", "ci_low", "ci_high", "null_mean", "null_q975", "p_permutation"]})
    rows += compare("05 held-out agreement as tracked (rounded)", new("temporal_agreement").drop(columns="n"), read(tp / "T2_agreement.csv").drop(columns="n"),
                    ["design", "set", "variant", "metric"], {c: c for c in ["value", "ci_low", "ci_high", "null_mean", "null_q975"]})
    icc_rec = read(tp / "T2_icc.csv")
    icc_rec = icc_rec[icc_rec.scale.str.startswith("clustering")].assign(scale="clustering scale")
    rows += compare("05 ICC between windows", new("temporal_icc"), icc_rec, ["design", "set", "scale", "feature"],
                    {c: c for c in ["icc", "ci_low", "ci_high", "analytic_ci_low", "analytic_ci_high"]})
    t3_cols = [f"{w}_{c}" for w in "AB" for c in ("value", "ci_low", "ci_high", "null_mean", "null_q975", "p_permutation")] + ["difference_A_minus_B", "difference_ci_low", "difference_ci_high"]
    rows += compare("05 agreement with profiles per window", new("temporal_profile_agreement"), read(tp / "T3_clinical_agreement.csv"), ["design", "statistic"], {c: c for c in t3_cols})
    c_cols = [f"{w}_{c}" for w in "AB" for c in ("difference", "ci_low", "ci_high")] + ["change_A_minus_B", "change_ci_low", "change_ci_high"]
    rows += compare("05 C2 minus rest CV per window", new("temporal_c2_contrast"), read(tp / "T4_cv_contrasts.csv"), ["design", "contrast"], {c: c for c in c_cols})
    rows += compare("05 C2 minus rest CV, full record", new("temporal_c2_full_record"), read(tp / "T4_reference_full_record.csv").dropna(subset=["contrast"]),
                    ["design", "contrast"], {c: c for c in ["difference", "ci_low", "ci_high"]})

    table = pd.DataFrame(rows)
    out = ROOT / "results" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "record_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.groupby("comparison", sort=False).agg(columns=("column", "size"), matched=("rows_matched", "max"), new=("rows_new", "max"),
                                                       worst=("max_abs_difference", "max"), all_match=("match", "all")).to_string())
    print(f"\n{int(table.match.sum())} of {len(table)} column comparisons within {TOLERANCE}")


if __name__ == "__main__":
    main()
