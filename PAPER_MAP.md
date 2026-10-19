# Paper map

Each manuscript item, the script that produces it and the tracked output that holds it. Item names follow the 19 October 2026 draft (`docs/paper1_draft_2026-10-19.pdf`). Every quoted number is checked against these outputs by `tests/test_paper_numbers.py`. Numbers are stored unrounded in `results/numbers/<script>.json`.

## Tables and figures

| Item | Script | Output |
|---|---|---|
| Table 1 | `analyses/01_cohort_and_profiles.py` | `results/tables/table1_baseline.{csv,tex}` |
| Supplementary Table S1 (full Table 1) | `01` | `results/tables/tableS1_baseline_full.{csv,tex}` |
| Supplementary Table S2 (other strata) | `01` | `results/tables/tableS2_baseline_other_strata.{csv,tex}` |
| Table 2 | values from `02`; drawn by `06` | `results/tables/table2_forward.{csv,tex}` |
| Table 3 | values from `02` (forward) and `03` (reverse); drawn by `06` | `results/tables/table3_bidirectional.{csv,tex}` |
| Fig 1 | values from `02` (participant points from `restricted/`); drawn by `06` | `results/figures/fig1_forward.{pdf,png}` |
| Fig 2 A to C | `03`; drawn by `06` | `results/figures/fig2_reverse.{pdf,png}` |
| Fig 2 D | `04` (pooled cohort); combined with A to C by `06` | same file |
| Fig 3 | `04` (other strata), `02` and `03` (non-insulin T2D reference); drawn by `06` | `results/figures/fig3_strata.{pdf,png}` |
| Supplementary Fig S1 | `05` (temporal), `03` (refit-stability reference line); drawn by `06` | `results/figures/figS1_temporal.{pdf,png}` |
| Fig 1, print version | same values as Fig 1; drawn by `06` | `results/figures/print/fig1_forward_print.{pdf,png}` |
| Fig 2, print version | same values as Fig 2 (A to C from `03`, D from `04`); drawn by `06` | `results/figures/print/fig2_reverse_print.{pdf,png}` |
| Fig 3, print version | same values as Fig 3; drawn by `06` | `results/figures/print/fig3_strata_print.{pdf,png}` |

Each figure has a `_plotted.json` file next to it holding the plotted values. `tests/test_figures.py` checks those values against the source tables. It also checks that each print version shows exactly the same values as its manuscript version.

The print versions are the newer 16 cm, 8 pt redraws of Figures 1 to 3, ported from the record's `paper1_overleaf/figures/`. They contain no new numbers, so either version can go into the manuscript. Supplementary Fig S1 already uses the print style.

## Numbers quoted in the text

The JSON key comes first, then where it comes from in `results/aggregate/`.

### Methods 2.1 and 2.2 (`01`, `results/numbers/01_cohort.json`)

- **1,892 processed, 1,591 passed; 564 / 405 / 458 / 164 by stratum:** `n_parquet`, `n_cohort`, `stratum_n`.
- **325 training participants, 133 assigned to frozen centroids:** `t2d_train_n`, `t2d_assigned_frozen_n`.
- **100 initializations; three profiles (diabetes strata), two (healthy, prediabetes):** `kmeans_n_init`, `k`.
- **Table 1 footnote missing values (30, 25, 23, 1):** `table1_missing`.
- **Section 3.1 GLP-1/GIP shares (57%, 32%, 34%):** `glp1_percent_by_profile`.

### Section 3.1 (`02`, `results/numbers/02_forward.json`)

- **Abstract η² 0.042 to 0.078; full-record 0.052 to 0.078; Table 2:** `eta_squared`, `eta_squared_ci`. Source: `forward_eta_squared_ci.csv` (2,000 resamples).
- **All omnibus p < 0.0001:** `max_p_t2d`. Source: `forward_kruskal_wallis.csv`.
- **Table 2 between-profile patterns:** `forward_pairwise_differences.csv`. The Dunn-Holm support comes from `forward_dunn_holm.csv`.
- **C1 − C3 CV −0.76 [−2.06, 0.63] (48 h) and −1.37 [−2.49, −0.21] (full); C1 − C2 2.26 [0.99, 3.56]; C2 − C3 −3.02:** `cv_c1_minus_c3`, `cv_c1_minus_c2`, `cv_c2_minus_c3`. Source: `forward_pairwise_differences.csv` (1,000 resamples within groups; see `docs/MANUSCRIPT_FIXES.md`).
- **CV η² 0.021 / 0.139 (full) and 0.013 / 0.111 (48 h):** `cv_eta_training_only`, `cv_eta_heldout`. Source: `forward_sensitivity_subsets.csv`.
- **Split p = 0.043 (full), 0.084 (48 h):** `split_dependence_p`. Source: `forward_split_dependence.csv`.
- **C2 − C1 CV −2.77 [−4.46, −1.05] without the GLP-1/GIP flag, 0.51 [−0.99, 2.02] with it:** `glp1_cv_c2_minus_c1`. Source: `forward_glp1_contrasts.csv`.

### Section 3.2 (`03`, `results/numbers/03_reverse.json`)

- **Groups 188 / 182 / 88; median mean glucose 120 / 147 / 185; median CV 16.4 / 21.1 / 24.8:** `group_sizes`, `median_mean_glucose`, `median_cv`. Source: `reverse_primary_groups.csv`.
- **Silhouette favoured two groups:** `silhouette`. Source: `reverse_partition_diagnostics.csv`.
- **ARI 0.039 [0.020, 0.068], p = 0.001; Cramér's V 0.205:** `ari`, `cramers_v`. Source: `reverse_agreement.csv`.
- **HbA1c η² 0.422; other variables 0.002 to 0.061; Table 3 reverse rows:** `reverse_eta`, `non_hba1c_eta_range`. Source: `forward_vs_reverse_eta_squared.csv`.
- **R² HbA1c 0.590 [0.455, 0.682], age 0.104, BMI 0.062, waist-to-hip 0.050, C-peptide 0.000, TG/HDL 0.002:** `cgm_to_clinical_r2`. Source: `reverse_prediction.csv`.
- **Clinical to mean glucose and TIR R² about 0.51, falling to 0.02 and 0.04 without HbA1c; CV 0.079 to 0.051:** `clinical_to_cgm_r2`.
- **AUROC 0.649 [0.612, 0.686]; balanced accuracy 0.448 [0.400, 0.496]; chance 0.333; p = 0.005:** `profile_auroc`, `profile_balanced_accuracy`, `chance_balanced_accuracy`.
- **AUROC 0.605 → 0.660, gain 0.055 [0.017, 0.094]:** `hba1c_alone_auroc`, `hba1c_plus_cgm_auroc`, `auroc_gain`.
- **"3 of 15 partition and participant-subset combinations":** `age_cells_at_or_above_medium`, `age_cells_total`. Source: the age rows of `reverse_clinical_differences.csv`.

### Section 3.3 (`04`, `results/numbers/04_strata.json`)

- **Healthy η² 0.021 [0.003, 0.054]; prediabetes 0.071 [0.028, 0.124]; CV η² ≤ 0.002:** `forward_mean_glucose_full`, `forward_cv_max_healthy_prediabetes`. Source: `strata_forward.csv`.
- **88 healthy and 89 prediabetes participants assigned to C2:** `assigned_c2`. Source: `strata_specificity_assignment.csv`.
- **C2 minus rest: +0.17 [−0.77, 1.12] in healthy, −1.04 [−1.88, −0.12] in prediabetes (Holm p = 0.065), −2.39 [−3.41, −1.43] in non-insulin T2D:** `cv_c2_minus_rest`, `cv_c2_minus_rest_t2d`. Source: `strata_specificity.csv`, `forward_c2_vs_rest.csv`.
- **"Approximately 40%" (37% at 48 h, 44% full record):** `prediabetes_share_of_t2d`.
- **ARI 0.004 to 0.013 in other strata; AUROC 0.685 and 0.741 against 0.664 and 0.628 with HbA1c alone:** `ari_other_strata`, `auroc`. Source: `strata_agreement.csv`, `strata_prediction.csv`.
- **Insulin-treated C-peptide 0.085 [0.019, 0.197] full record, 0.021 at 48 h:** `insulin_c_peptide`. Source: `strata_clinical_differences.csv`.
- **Pooled ARI with stratum 0.160 [0.136, 0.187], p = 0.001; 0.037 against 0.039; combined labels 0.093 against null mean 0.088, p = 0.001 (Fig 2D):** `pooled_*`, `t2d_only_ari`. Source: `pooled_agreement.csv`.

### Temporal (`05`, `results/numbers/05_temporal.json`)

- **ARI 0.42 between windows (held-out):** `heldout_ari`. It is rounded to 2 decimals because the held-out n is masked. Source: `temporal_agreement.csv`.
- **ARI with the profiles 0.030 and 0.036:** `profile_ari_window_a`, `profile_ari_window_b`. Source: `temporal_profile_agreement.csv`.
