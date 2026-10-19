# Clinical profiles and free-living glucose in type 2 diabetes (AI-READI)

This is the analysis code for Paper 1, *Clinical profiles and free-living glucose in type 2 diabetes: a bidirectional analysis in the AI-READI cohort* (Charfeddine and Patel; draft of 19 October 2026 in `docs/`). Every table, figure and quoted number comes from one script here. [`PAPER_MAP.md`](PAPER_MAP.md) says which.

## Question

Clinical subtypes of type 2 diabetes are built from variables such as age, BMI and HbA1c. Do they match distinct patterns of free-living glucose?

- **Forward.** Three clinical profiles (C1, C2, C3) were derived in 458 participants with type 2 diabetes treated without insulin. Do they differ in glucose measured by continuous glucose monitoring (CGM)?
- **Reverse.** Grouping the same participants by CGM alone, do the groups recover the profiles and their clinical variables?
- **Other strata.** The analysis is repeated across all glycemic strata (1,591 participants) and across two non-overlapping 3-day windows.

No deep learning model is used.

## Main findings (from the manuscript)

1. **Modest forward differences.** The profiles differ modestly in glucose: rank η² is 0.042 to 0.078 for mean glucose, time in range and coefficient of variation, and the distributions overlap substantially. C3 has the highest glucose. C2 (younger, higher BMI) has the lowest variability.
2. **Weak reverse agreement.** CGM groups agree weakly with the profiles (ARI 0.039). They differ mainly in HbA1c (η² 0.422; other variables 0.002 to 0.061). CGM features predict HbA1c (R² 0.59) but not C-peptide or TG/HDL.
3. **Some profile information in CGM.** CGM still predicts profile membership above chance (macro AUROC 0.649) and adds to HbA1c (AUROC 0.605 to 0.660).
4. **Diagnosis dominates in the pooled cohort.** Pooled CGM groups follow diagnosis (ARI 0.160) far more than profile. The C2 lower-variability pattern is 37% to 44% as large in prediabetes and absent in healthy participants. CGM groups are moderately reproducible across 3-day windows (ARI 0.42), while their agreement with the profiles stays low in each window (0.030, 0.036).

## How to run

```bash
conda env create -f environment.yml          # CPU only, no torch; creates env paper1-cgm
conda activate paper1-cgm
export AIREADI_DATA_DIR=/path/to/AI-READI    # folder holding enriched_multimodal/ and experiment_c_split_adapt6h_seed42/
bash run_all.sh                              # analyses 01 to 06, then pytest
```

The full run takes about 28 minutes on 10 CPU cores, most of it in 03 (8 min) and 04 (17 min) (`docs/VERIFICATION.md` gives the time per script). Without the data, `pytest` still runs the unit, figure, paper-number and hygiene tests on the tracked outputs. The tests that need data are skipped.

## Repository map

| Path | What it holds |
|---|---|
| `analyses/01_cohort_and_profiles.py` | Cohort, clinical profiles, Table 1, Supplementary Tables S1 and S2 |
| `analyses/02_forward_direction.py` | Section 3.1: Table 2 and Fig 1 values, pairwise differences, sensitivity subsets, split dependence, GLP-1/GIP contrasts |
| `analyses/03_reverse_direction.py` | Section 3.2: CGM groups, Table 3, agreement, prediction (Fig 2 A to C) |
| `analyses/04_glycemic_strata.py` | Section 3.3: other strata, specificity check, pooled cohort (Fig 3, Fig 2D) |
| `analyses/05_temporal_reproducibility.py` | Two 3-day windows (Supplementary Fig S1) |
| `analyses/06_figures_and_tables.py` | Draws every figure and Tables 2 and 3 from saved outputs |
| `src/cgmprofiles/` | The importable package: data loading, profiles, CGM metrics and features, statistics, prediction, disclosure control, tables, plotting |
| `config/paper1.yaml` | Every constant: paths, seeds, thresholds, windows, resample counts, labels, colours |
| `results/` | Tracked aggregate outputs: `tables/`, `figures/`, `numbers/` (each quoted number), `aggregate/`, `profile_derivation/` (frozen centroids) |
| `tests/` | Paper numbers, figures, statistics, disclosure, profile refit, repository hygiene |
| `docs/` | Manuscript, methods notes, decision log, provenance, pre-registered designs, verification, manuscript fixes |

## Not included

- **The AI-READI data**, which is under the AI-READI data use agreement.
- **Participant-level outputs.** Profile assignments, per-participant metrics and features are written to `restricted/`, which is gitignored and never committed.
- **Small cells.** Tracked tables mask counts of 1 to 4 (`docs/small_cell_policy.md`).

The full history of how each result was first produced, including audits and superseded scripts, is in the record repository named in `docs/provenance.md`.
