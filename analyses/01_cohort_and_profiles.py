"""01 Cohort and clinical profiles (Methods 2.1 and 2.2; Table 1; Supplementary Tables S1 and S2).

Question: who is in the cohort, and which clinical profile is each participant assigned to?
Inputs:   AI-READI parquet and split file (AIREADI_DATA_DIR); frozen centroids in results/profile_derivation/.
Outputs:  restricted/profile_labels, restricted/participant_metrics (participant level, gitignored);
          results/tables/table1_baseline, tableS1_baseline_full, tableS2_baseline_other_strata (.csv, .tex);
          results/numbers/01_cohort.json.
Paper:    Methods 2.1 and 2.2 counts; Table 1; Table 1 footnote (missing values); Section 3.1 GLP-1 shares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cgmprofiles import io, profiles, tables
from cgmprofiles.cgm_metrics import participant_metrics
from cgmprofiles.config import get_config, restricted_dir, results_dir
from cgmprofiles.report import Timer, write_numbers, write_restricted

CFG = get_config()
COL = CFG.columns
TCFG = CFG.tables


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Stop: {message}")


def baseline_frame(labels: pd.DataFrame, clinical: pd.DataFrame, cgm: pd.DataFrame) -> pd.DataFrame:
    """Labels plus the descriptive variables of the baseline tables (observed values, post-CGM rule for LDL)."""
    medications = list(COL.medications)
    extra = io.load_static([COL.sex, COL.race, COL.site, COL.systolic_bp, COL.ldl, COL.ldl_date, *medications, COL.any_diabetes_drug])
    frame = labels.join(extra, on=io.PID).merge(clinical[[io.PID, "cgm_end"]], on=io.PID, how="left")
    ldl_dates = pd.to_datetime(frame[COL.ldl_date], utc=True)
    frame.loc[ldl_dates.notna() & (ldl_dates > frame.cgm_end), COL.ldl] = np.nan
    for column in medications:
        frame[column] = frame[column].fillna(0).gt(0).astype(int)
    frame[io.NO_DRUG_FLAG] = 1 - frame[COL.any_diabetes_drug].fillna(0).gt(0).astype(int)
    frame["female"] = np.where(frame[COL.sex].isna(), np.nan, (frame[COL.sex] == CFG.cohort.female_code).astype(float))
    frame["cgm_days"] = frame[io.PID].map(cgm.groupby(io.PID).elapsed_h.max() / 24.0)
    frame["valid_readings"] = frame[io.PID].map(cgm[cgm[io.GLUCOSE].notna()].groupby(io.PID).size())
    return frame


def latex_baseline(table: pd.DataFrame, cols: list[str], caption: str, label: str, footnote: str) -> str:
    out = table[["characteristic", *cols, "p_value"]].rename(columns={"characteristic": "Characteristic", "p_value": "P"})
    return tables.simple_latex(out, caption, label, footnote, "l" + "c" * (len(cols) + 1))


def main() -> None:
    timer = Timer()
    split, clinical = io.load_split(), io.load_clinical_frame()
    n_parquet = io.read_columns([io.PID])[io.PID].nunique()
    check(n_parquet == CFG.cohort.n_parquet and len(split) == CFG.cohort.n_split, f"cohort sizes {n_parquet}, {len(split)}")
    check(split.split.value_counts().to_dict() == CFG.cohort.split_counts.raw, "split counts")
    cohort = split.merge(clinical, on=io.PID, how="left", validate="one_to_one")
    stratum_n = cohort.canonical_stratum.value_counts().to_dict()
    check(stratum_n == CFG.cohort.stratum_n.raw, f"stratum sizes {stratum_n}")

    labels, models = profiles.build_profiles(clinical, split)
    centroid_difference = profiles.compare_with_frozen_centroids(models, results_dir("profile_derivation", "centroids"))
    check(max(centroid_difference.values()) == 0.0, f"refit centroids differ from the frozen centroids: {centroid_difference}")
    counts = {s: {int(k): int(v) for k, v in labels[labels.canonical_stratum == s].label.value_counts().sort_index().items()} for s in io.STRATA}
    check(counts == {s: {int(k): v for k, v in c.items()} for s, c in CFG.profiles.expected_counts.raw.items()}, f"profile counts {counts}")
    write_restricted(labels, "profile_labels")

    cgm = io.load_cgm(labels[io.PID].tolist())
    write_restricted(participant_metrics(labels, cgm), "participant_metrics")

    frame = baseline_frame(labels, clinical, cgm)
    sections = [(section, [tuple(item) for item in items]) for section, items in TCFG.baseline_sections]
    decimals, paper_labels = TCFG.decimals.raw, TCFG.paper_labels
    rendered, masking = {}, []
    for stratum in [io.PRIMARY, *CFG.strata.other]:
        cols, blocks = tables.build_baseline_blocks(frame[frame.canonical_stratum == stratum], sections)
        masking += tables.masking_report(blocks, cols, stratum)
        rendered[stratum] = {"cols": cols, "blocks": blocks,
                             "record": tables.render_baseline(cols, blocks, "record"),
                             "paper": tables.render_baseline(cols, blocks, "paper", decimals, paper_labels),
                             "table1": tables.render_baseline(cols, blocks, "paper", decimals, paper_labels, keep=list(TCFG.table1_rows))}
    masking = pd.DataFrame(masking)
    check(not masking.recoverable.any(), "a masked baseline cell is recoverable")
    masking.to_csv(restricted_dir() / "baseline_masking_check.csv", index=False)
    for stratum, r in rendered.items():
        r["record"].to_csv(restricted_dir() / f"baseline_record_style_{stratum}.csv", index=False)

    out = results_dir("tables")
    primary = rendered[io.PRIMARY]
    missing = {b["label"]: b["missing"] for b in primary["blocks"] if b["kind"] == "cont" and b["label"] in TCFG.table1_rows and b["missing"]}
    footnote_missing = "Missing values: " + ", ".join(f"{paper_labels.get(k, k).split(',')[0]} {v}" for k, v in missing.items()) + "."
    footnote = (f"Continuous variables are median [IQR], categorical variables n (%). Triglyceride-to-HDL ratio computed from mg/dL values. "
                f"{footnote_missing} P values compare profiles (Kruskal-Wallis or chi-square test) and are descriptive, without correction "
                f"for multiple comparisons. Race categories with fewer than 5 participants in any profile are combined. Site, blood pressure, "
                f"LDL cholesterol, other medications, CGM coverage and the data split are in Supplementary Table S1.")
    table1 = primary["table1"].drop(columns=["section", "missing_n", "test"])
    table1.to_csv(out / "table1_baseline.csv", index=False)
    (out / "table1_baseline.tex").write_text(latex_baseline(primary["table1"], primary["cols"],
        f"Baseline characteristics by clinical profile in {CFG.labels.strata_long['t2d_oral_non_insulin']}.", "tab:baseline", footnote))
    primary["paper"].to_csv(out / "tableS1_baseline_full.csv", index=False)
    (out / "tableS1_baseline_full.tex").write_text(latex_baseline(primary["paper"], primary["cols"],
        f"Full baseline characteristics by clinical profile in {CFG.labels.strata_long['t2d_oral_non_insulin']}.", "tab:baseline-full",
        "Cells with 1 to 4 participants are shown as <5, and further cells as NR where needed to prevent recovery. Missing values per row "
        "are in the CSV version. P values were computed from unmasked counts and are descriptive."))
    other = pd.concat([rendered[s]["paper"].assign(stratum=CFG.labels.strata_short[s]) for s in CFG.strata.other])
    other.to_csv(out / "tableS2_baseline_other_strata.csv", index=False)
    (out / "tableS2_baseline_other_strata.tex").write_text("\n".join(
        latex_baseline(rendered[s]["paper"], rendered[s]["cols"], f"Baseline characteristics by clinical profile: {CFG.labels.strata_short[s]}.",
                       f"tab:baseline-{s}", "Same conventions as Supplementary Table S1.") for s in CFG.strata.other))

    t2d = io.ordered_stratum(labels, io.PRIMARY)
    glp1 = frame[frame.canonical_stratum == io.PRIMARY].groupby("label")[COL.glp1].mean() * 100
    write_numbers("01_cohort", {
        "n_parquet": n_parquet, "n_cohort": len(split), "stratum_n": stratum_n,
        "t2d_n": len(t2d), "t2d_train_n": int((t2d.split == "train").sum()), "t2d_assigned_frozen_n": int((t2d.split != "train").sum()),
        "profile_counts": counts, "k": CFG.profiles.k.raw, "kmeans_n_init": CFG.profiles.kmeans_n_init,
        "healthy_complete_case_n": len(labels[labels.canonical_stratum == "healthy"]),
        "refit_centroid_max_abs_difference": centroid_difference,
        "glp1_percent_by_profile": {f"C{int(k)}": v for k, v in glp1.items()},
        "table1_missing": missing, "masked_cells": int(len(masking)), "masked_cells_recoverable": int(masking.recoverable.sum()),
        "runtime_s": timer.seconds()})
    print(f"01 done in {timer.seconds()} s: {len(labels)} participants assigned; centroid difference {max(centroid_difference.values())}")


if __name__ == "__main__":
    main()
