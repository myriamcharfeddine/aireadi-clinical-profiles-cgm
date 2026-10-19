# Verification

Phase 3 compared this repository with the record repository (`SSM-CGM-stream-AI`, commit `a36acdb`) and with the manuscript (`paper1_draft_2026-10-19.pdf`). It used one clean end-to-end run of `run_all.sh` in env `paper1-cgm`, with `restricted/` and every generated output deleted first, on 19 October 2026.

## Summary

- **Manuscript numbers.** 77 quoted numbers were checked at the rounding used in the text. 74 match the outputs. 3 are manuscript errors, listed in `MANUSCRIPT_FIXES.md` (items 8, 9 and 10, all double rounding). Table 1 is checked cell by cell; one cell is a manuscript error (item 1). Tables 2 and 3 are checked row by row; Table 3 has the error of item 8.
- **Record outputs.** 186 of 190 value columns over 32 output comparisons match the record within 1e-9. Most deterministic quantities match exactly or to about 1e-14 (floating-point summation order). Bootstrap and permutation intervals match exactly, because the order of random draws is preserved.
- **The exception.** The 4 non-matching columns all come from one model: the exploratory insulin-treated "HbA1c plus CGM" classifier in S2. The combined AUROC is 0.550021 here against 0.549984 in the record (difference 3.6e-5). Its gain CI bounds differ by 1.1e-4 and 7.9e-5. Nothing in the manuscript uses these values. Every other S2 column matches to 1e-14.
- **Cause.** The record stored the pooled CGM features as CSV and read them back, which perturbs 11,988 values in the last bits. This logistic model stops at its iteration limit, so it is sensitive to that. Rerunning this repository's code on CSV-round-tripped features reproduces the record exactly (0.5499844541817508). This repository stores intermediates as parquet, which round-trips exactly. The difference is below the 0.001 threshold set for this check. The author accepted it on 19 October 2026.
- **Reproducing the paper.** `bash run_all.sh` reproduces every number, table and figure quoted in the manuscript. The one value that differs from the record is not quoted. Four manuscript values (items 1, 8, 9 and 10 of `MANUSCRIPT_FIXES.md`) were rounded twice by hand in the draft. The code gives the exact values and their correct rounding, so it is the manuscript text that needs to change. `tests/test_paper_numbers.py` fails if any other quoted number stops matching.
- **Baseline tables.** Table 1 and Supplementary Tables S1 and S2, rendered in the record's own format, are identical to the record's tracked CSVs cell for cell: values, `<5` and `NR` masks, and P values.
- **Profiles.** Refitting reproduces the frozen centroids exactly (maximum difference 0.0) in all four strata, with 0 label differences. Labels, assignment margins and CGM metrics match the record's participant-level audit file to 6e-14 (3,130 participant-window rows).
- **Hygiene.**
  - The participant-ID scan finds 0 hits.
  - The small-cell check finds no count of 1 to 4 in any tracked CSV.
  - No torch import and no absolute path in code, config or scripts.
  - No file over 5 MB.
  - All tests pass: 126 passed, plus 4 expected failures, which are the manuscript errors.

## Runtime per script (clean run, 10 CPU cores)

| Script | Seconds |
|---|---|
| 01_cohort_and_profiles.py | 17 |
| 02_forward_direction.py | 33 |
| 03_reverse_direction.py | 475 |
| 04_glycemic_strata.py | 1043 |
| 05_temporal_reproducibility.py | 87 |
| 06_figures_and_tables.py | 7 |
| **Total** | **1662** (about 28 min) |

## Manuscript numbers

Columns: location, the manuscript text, the record value at the same rounding, the new value, and the result. The record value equals the new value at this rounding wherever the source comparison below matches, which holds for every row.

| Location | Manuscript | Record | New | Result |
|---|---|---|---|---|
| 2.1 processed participants | 1,892 | 1,892 | 1,892 | match |
| 2.1 passed preprocessing | 1,591 | 1,591 | 1,591 | match |
| 2.1 strata | 564/405/458/164 | 564/405/458/164 | 564/405/458/164 | match |
| 2.2 training and frozen assignment | 325/133 | 325/133 | 325/133 | match |
| 2.2 k-means initializations | 100 | 100 | 100 | match |
| Table 1 n | 192/111/155 | 192/111/155 | 192/111/155 | match |
| Table 1 footnote missing | 30/25/23/1 | 30/25/23/1 | 30/25/23/1 | match |
| 3.1 GLP-1/GIP shares | 57/32/34 | 57/32/34 | 57/32/34 | match |
| Abstract eta squared range | 0.042-0.078 | 0.042-0.078 | 0.042-0.078 | match |
| 3.1 full-record range | 0.052-0.078 | 0.052-0.078 | 0.052-0.078 | match |
| 3.1 all p < 0.0001 | True | True | True | match |
| Table 2 mean glucose 48 h | 0.068 [0.029, 0.120] | 0.068 [0.029, 0.120] | 0.068 [0.029, 0.120] | match |
| Table 2 mean glucose full | 0.057 [0.021, 0.109] | 0.057 [0.021, 0.109] | 0.057 [0.021, 0.109] | match |
| Table 2 TIR 48 h | 0.064 [0.025, 0.116] | 0.064 [0.025, 0.116] | 0.064 [0.025, 0.116] | match |
| Table 2 TIR full | 0.078 [0.038, 0.133] | 0.078 [0.038, 0.133] | 0.078 [0.038, 0.133] | match |
| Table 2 CV 48 h | 0.042 [0.015, 0.087] | 0.042 [0.015, 0.087] | 0.042 [0.015, 0.087] | match |
| Table 2 CV full | 0.052 [0.023, 0.100] | 0.052 [0.023, 0.100] | 0.052 [0.023, 0.100] | match |
| 3.1 C1-C3 CV 48 h | -0.76 [-2.06, 0.63] | -0.76 [-2.06, 0.63] | -0.76 [-2.06, 0.63] | match |
| 3.1 C1-C3 CV full | -1.37 [-2.49, -0.21] | -1.37 [-2.49, -0.21] | -1.37 [-2.49, -0.21] | match |
| 3.1 C2 lower than C1, 48 h | 2.26 [0.99, 3.56] | 2.26 [0.99, 3.56] | 2.26 [0.99, 3.56] | match |
| 3.1 C2 lower than C3, 48 h | -3.02 | -3.02 | -3.02 | match |
| 3.1 CV eta training only, full | 0.021 | 0.021 | 0.021 | match |
| 3.1 CV eta held-out, full | 0.139 | 0.139 | 0.139 | match |
| 3.1 CV eta training only, 48 h | 0.013 | 0.013 | 0.013 | match |
| 3.1 CV eta held-out, 48 h | 0.111 | 0.111 | 0.111 | match |
| 3.1 split p full / 48 h | 0.043/0.084 | 0.043/0.084 | 0.043/0.084 | match |
| 3.1 C2-C1 CV without GLP-1/GIP flag | -2.77 [-4.46, -1.05] | -2.77 [-4.46, -1.05] | -2.77 [-4.46, -1.05] | match |
| 3.1 C2-C1 CV with GLP-1/GIP flag | 0.51 [-0.99, 2.02] | 0.51 [-0.99, 2.02] | 0.51 [-0.99, 2.02] | match |
| 3.2 CGM group sizes | 188/182/88 | 188/182/88 | 188/182/88 | match |
| 3.2 median mean glucose | 120/147/185 | 120/147/185 | 120/147/185 | match |
| 3.2 median CV | 16.4/21.1/24.8 | 16.4/21.1/24.8 | 16.4/21.1/24.8 | match |
| 3.2 silhouette favoured two groups | 2 | 2 | 2 | match |
| Abstract and 3.2 ARI | 0.039 [0.020, 0.068] | 0.039 [0.020, 0.068] | 0.039 [0.020, 0.068] | match |
| 3.2 ARI permutation p | 0.001 | 0.001 | 0.001 | match |
| 3.2 Cramer's V | 0.205 | 0.205 | 0.205 | match |
| 3.2 and Table 3 HbA1c | 0.422 [0.343, 0.497] | 0.422 [0.343, 0.497] | 0.422 [0.343, 0.497] | match |
| Abstract non-HbA1c range | 0.002-0.061 | 0.002-0.061 | 0.002-0.061 | match |
| Table 3 age | 0.061 [0.024, 0.113] | 0.061 [0.024, 0.113] | 0.061 [0.024, 0.113] | match |
| Table 3 BMI | 0.020 [0.001, 0.060] | 0.020 [0.001, 0.060] | 0.020 [0.001, 0.060] | match |
| Table 3 C-peptide | 0.002 [0.000, 0.029] | 0.002 [0.000, 0.029] | 0.002 [0.000, 0.029] | match |
| Table 3 TG/HDL | 0.025 [0.004, 0.069] | 0.025 [0.004, 0.069] | 0.025 [0.004, 0.069] | match |
| Table 3 waist-to-hip | 0.028 [0.005, 0.069] | 0.028 [0.004, 0.069] | 0.028 [0.004, 0.069] | manuscript error (MANUSCRIPT_FIXES.md item 8) |
| 3.2 R2 HbA1c | 0.590 [0.455, 0.682] | 0.590 [0.455, 0.682] | 0.590 [0.455, 0.682] | match |
| Abstract R2 HbA1c | 0.59 | 0.59 | 0.59 | match |
| 3.2 R2 age, BMI, waist-to-hip | 0.104/0.062/0.050 | 0.104/0.062/0.050 | 0.104/0.062/0.050 | match |
| 3.2 R2 C-peptide, TG/HDL | 0.000/0.002 | 0.000/0.002 | 0.000/0.002 | match |
| 3.2 clinical to mean glucose and TIR | 0.51/0.51 | 0.51/0.51 | 0.51/0.51 | match |
| 3.2 without HbA1c | 0.02/0.04 | 0.02/0.04 | 0.02/0.04 | match |
| 3.2 CV with and without HbA1c | 0.079/0.051 | 0.079/0.051 | 0.079/0.051 | match |
| Abstract and 3.2 AUROC | 0.649 [0.612, 0.686] | 0.649 [0.612, 0.686] | 0.649 [0.612, 0.686] | match |
| 3.2 balanced accuracy | 0.448 [0.400, 0.496] | 0.448 [0.400, 0.496] | 0.448 [0.400, 0.496] | match |
| 3.2 chance balanced accuracy | 0.333 | 0.333 | 0.333 | match |
| 3.2 permutation p (both metrics) | 0.005/0.005 | 0.005/0.005 | 0.005/0.005 | match |
| Abstract and 3.2 AUROC HbA1c to combined | 0.605/0.660 | 0.605/0.660 | 0.605/0.660 | match |
| 3.2 AUROC gain | 0.055 [0.017, 0.094] | 0.055 [0.017, 0.094] | 0.055 [0.017, 0.094] | match |
| 3.2 3 of 15 combinations | 3 of 15 | 3 of 15 | 3 of 15 | match |
| 3.3 healthy mean glucose | 0.021 [0.003, 0.054] | 0.021 [0.003, 0.053] | 0.021 [0.003, 0.053] | manuscript error (MANUSCRIPT_FIXES.md item 9) |
| 3.3 prediabetes mean glucose | 0.071 [0.028, 0.124] | 0.071 [0.028, 0.124] | 0.071 [0.028, 0.124] | match |
| 3.3 CV negligible | <= 0.002 | <= 0.002 | <= 0.002 | match |
| 3.3 assigned to C2 | 88/89 | 88/89 | 88/89 | match |
| 3.3 healthy C2 minus rest | 0.17 [-0.77, 1.12] | 0.17 [-0.77, 1.12] | 0.17 [-0.77, 1.12] | match |
| 3.3 prediabetes C2 minus rest | -1.04 [-1.88, -0.12] | -1.04 [-1.88, -0.12] | -1.04 [-1.88, -0.12] | match |
| 3.3 T2D C2 minus rest | -2.39 [-3.41, -1.43] | -2.39 [-3.41, -1.43] | -2.39 [-3.41, -1.43] | match |
| 3.3 Holm p prediabetes | 0.065 | 0.065 | 0.065 | match |
| Abstract prediabetes share (approximately 40%) | 37%-44% | 37%-44% | 37%-44% | match |
| 3.3 ARI other strata | 0.004-0.013 | 0.004-0.013 | 0.004-0.013 | match |
| Abstract ARI healthy and prediabetes | <= 0.013 | <= 0.013 | <= 0.013 | match |
| 3.3 AUROC healthy, prediabetes | 0.685/0.741 | 0.685/0.741 | 0.685/0.741 | match |
| 3.3 AUROC HbA1c alone | 0.664/0.628 | 0.664/0.628 | 0.664/0.628 | match |
| 3.3 insulin-treated C-peptide full | 0.085 [0.019, 0.197] | 0.085 [0.019, 0.197] | 0.085 [0.019, 0.197] | match |
| 3.3 insulin-treated C-peptide 48 h | 0.021 | 0.021 | 0.021 | match |
| Abstract and 3.3 pooled ARI with stratum | 0.160 [0.136, 0.187] | 0.160 [0.136, 0.187] | 0.160 [0.136, 0.187] | match |
| 3.3 pooled ARI p | 0.001 | 0.001 | 0.001 | match |
| 3.3 pooled versus separate T2D clustering | 0.037/0.039 | 0.036/0.039 | 0.036/0.039 | manuscript error (MANUSCRIPT_FIXES.md item 10) |
| 3.3 combined labels and null | 0.093/0.088/0.001 | 0.093/0.088/0.001 | 0.093/0.088/0.001 | match |
| Abstract windows ARI | 0.42 | 0.42 | 0.42 | match |
| Abstract profile ARI per window | 0.030/0.036 | 0.030/0.036 | 0.030/0.036 | match |

## Aggregate outputs against the record

One row per output comparison: rows matched, kind of quantities, largest absolute difference over all compared columns, and whether every column is within 1e-9. The column-level detail is in `results/verification/record_comparison.csv`.

| Comparison | Rows | Kind | Max abs difference | Within 1e-9 |
|---|---|---|---|---|
| 01 labels, margins and CGM metrics (participant level) | 3130 | deterministic | 5.7e-14 | yes |
| 02 Kruskal-Wallis, all strata | 24 | deterministic | 0.0e+00 | yes |
| 02 pairwise differences (1,000 resamples) | 48 | bootstrap/permutation, deterministic | 1.4e-14 | yes |
| 02 C2 versus rest | 2 | bootstrap/permutation, deterministic | 4.0e-15 | yes |
| 02 eta squared, 2,000-resample CIs (Table 2) | 9 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 02 ANOVA | 6 | deterministic | 0.0e+00 | yes |
| 02 Dunn tests | 18 | deterministic | 1.3e-23 | yes |
| 02 sensitivity subsets | 30 | deterministic | 2.6e-23 | yes |
| 02 split dependence | 6 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 02 GLP-1/GIP contrasts | 8 | bootstrap/permutation, deterministic | 8.9e-16 | yes |
| 03 silhouette by k | 4 | deterministic | 0.0e+00 | yes |
| 03 refit stability (primary) | 1 | bootstrap/permutation | 0.0e+00 | yes |
| 03 primary CGM groups (sizes, medians, centroids) | 3 | deterministic | 0.0e+00 | yes |
| 03 primary contingency (Fig 2A) | 3 | deterministic | 0.0e+00 | yes |
| 03 B3 clinical differences | 90 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 03 forward versus reverse (Table 3) | 12 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 03 B4 agreement | 15 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 03 B5 prediction | 54 | bootstrap/permutation, deterministic | 4.0e-16 | yes |
| 03 B5 permutation null draws (Fig 2C) | 3200 | deterministic | 5.1e-16 | yes |
| 04 S1 forward, other strata | 18 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 04 S3 specificity | 48 | bootstrap/permutation, deterministic | 4.0e-15 | yes |
| 04 S3 assignment | 2 | deterministic | 0.0e+00 | yes |
| 04 S2 clinical differences | 36 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 04 S2 agreement | 18 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 04 S2 prediction | 24 | bootstrap/permutation, deterministic | 1.1e-04 | **no** |
| 04 B6 pooled agreement (Fig 2D) | 36 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 05 held-out agreement between windows (unrounded) | 8 | bootstrap/permutation, deterministic | 9.9e-17 | yes |
| 05 held-out agreement as tracked (rounded) | 8 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 05 ICC between windows | 32 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 05 agreement with profiles per window | 6 | bootstrap/permutation, deterministic | 0.0e+00 | yes |
| 05 C2 minus rest CV per window | 2 | bootstrap/permutation, deterministic | 3.8e-15 | yes |
| 05 C2 minus rest CV, full record | 2 | bootstrap/permutation, deterministic | 0.0e+00 | yes |

## Not compared, by design

- **Outputs dropped under decision D4.** They are listed in `decisions.md`.
- **The record's chunk summaries.** They contain no number of their own.
- **Temporal statistics whose n is masked.** These are compared unrounded, using the record's restricted file, and again as tracked, at 2 decimals.
