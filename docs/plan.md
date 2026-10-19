# Plan: standalone repository for Paper 1

> Historical: the Phase 1 plan (19 October 2026), kept for the record. Where it differs from the repository, `decisions.md` is authoritative (decisions D1 to D9 changed several proposals below).

Paper: "Clinical profiles and free-living glucose in type 2 diabetes: a bidirectional analysis in the AI-READI cohort" (draft of 19 October 2026).

Status: Phase 1, plan only. Nothing has been ported. This folder is not a git repository yet.

- **Record repository (read-only):** `~/CGM/clinical-profiles-paper-repro`, branch `main` at `5aecdae`. Every provenance entry below refers to this commit.
- **Manuscript:** `01_raw_cgm_differences/Clinical_profiles_and_free_living_glucose_in_type_2_diabetes__a_bidirectional_analysis_in_the_AI_READI_cohort__2_ (6).pdf`, committed in `5aecdae`. It will be copied to `docs/paper1_draft_2026-10-19.pdf`.

## 0. Findings from Step 0 that shape the plan

1. **The figures in the PDF are the original summary figures, not the "paper" redraws.**

   | Paper item | Drawn by | Not used in the PDF |
   |---|---|---|
   | Fig 1 | `code/outputs/phase0_extensions/scripts/plot_phase0_raw_t2d_main_figure.py` (title "Raw CGM behavior across clinical profiles…") | `plot_fig1_raw_profiles_paper.py` |
   | Fig 2 | `reverse_direction/plot_reverse_figure.py` (`reverse_direction_summary.pdf`; confirmed by you) | `plot_fig2_paper.py`, `plot_fig2_correspondence.py` |
   | Fig 3 | `strata/plot_strata_figure.py` (`strata_summary.pdf`) | `plot_fig3_paper.py` |

   The "superseded" labels in the brief are therefore reversed for these figures; see Decision D1.
2. **Tables 2 and 3 have no generating script.** They were typed into the manuscript from `ROBUSTNESS_31.csv` and `B3_forward_vs_reverse_eta_squared.csv`. The new repo will generate them.
3. **Table 1 in the PDF was edited by hand from the generated `.tex`.** The PDF uses integer medians with half-up rounding (46.5 → 47), en-dash ranges, shorter race labels and a P of 0.007 instead of 0.0067. See D2.
4. **No Results text on temporal reproducibility and no supplementary section in the PDF yet.** The temporal numbers (0.42, 0.030, 0.036) appear only in the Abstract, Methods and Discussion. Fig S1 (v2) exists only as a file.
5. **`paper1-level-dynamics` does not exist**, locally or on the remote, and there is no `level_dynamics/` folder. `06_level_vs_dynamics.py` is dropped.
6. **The frozen profiles can be regenerated exactly.** A read-only probe (2026-10-19, env `ssmcgm`) refit the iterative imputer, log1p rule and scaler, then k-means (n_init 100, seed 42) on training participants in split-file order for all four strata. Compared with the frozen pipelines and centroid JSONs:
   - centroid difference 0.0, scaler difference 0.0, same log list;
   - 0 label mismatches out of 538 / 405 / 458 / 164;
   - 4.7 s in total.

   So the new repo needs no joblib pipeline and no assignment CSV. It refits, then asserts equality with the tracked centroid JSONs (aggregate, 6 numbers per profile).
7. **The record's tracked outputs contain unmasked counts of 1 to 4** that must not be copied as they are:
   - `B4_contingency_tables.csv` (split-level and minimal-feature partitions);
   - `B6_cluster_by_stratum.csv` (healthy, pooled k = 3, G3 = 2);
   - `B2_partition_sizes.csv`;
   - `S2_contingency_secondary.csv`.

   The record repository stays as it is (read-only), but you may want to fix it separately.
8. **`gh` is not installed** on this machine. Phase 4 needs it, or you create the repository on the web. See D8.

## 1. Tree

```
aireadi-clinical-profiles-cgm/
  README.md
  PAPER_MAP.md
  environment.yml          python 3.10, numpy 2.2.6, pandas 2.3.3, scipy 1.15.2, scikit-learn 1.7.2,
                           statsmodels 0.14.6, matplotlib 3.10.9, pyarrow 24.0.0, pyyaml 6.0.3,
                           joblib 1.5.3, threadpoolctl, pytest (versions pinned to env ssmcgm; no torch, no seaborn)
  pyproject.toml           installs src/cgmprofiles
  run_all.sh
  config/
    paper1.yaml
  src/cgmprofiles/
    __init__.py
    config.py              loads paper1.yaml once; resolves AIREADI_DATA_DIR and the restricted dir
    io.py                  parquet columns, split, clinical frame (post-CGM lab rule, TG/HDL mg/dL), timezones, medication flags
    profiles.py            imputation, log1p rule, scaling, k-means, HbA1c display order, frozen-centroid assignment, margins, check against tracked centroids
    cgm_metrics.py         windows from the first record; mean glucose, TIR 70 to 180, CV
    cgm_features.py        extended and minimal features: TBR/TAR, SD, MAGE, ACF, Lomb-Scargle share, daily-profile PCA, per-participant local time
    cgm_groups.py          fit-on-train preprocessing and k-means, nearest centroid, silhouette, GMM check, refit stability, Hungarian matching
    stats.py               Kruskal-Wallis H by hand, rank eta squared, participant bootstrap (eta, mean difference), Dunn-Holm, ARI, NMI, bias-corrected Cramér's V, kappa, permutation nulls, ICC
    prediction.py          repeated stratified CV (5 x 20), M2 fold imputation, ridge, multinomial logistic, paired bootstrap, permutation nulls with escalation
    windows.py             non-overlapping temporal windows, eligibility, warm-up variants
    disclosure.py          one milp-based small-cell masker (named cells plus linear equalities) and recovery check, used by tables and temporal outputs
    tables.py              Table 1, Table 2, Table 3, Tables S1 and S2 (CSV and tabularx .tex)
    plotting/
      style.py             palette, fonts, stratum labels and colours from config, panel helpers, save
      figures.py           fig1_forward, fig2_reverse (A to C from 03, D from 04), fig3_strata, figS1_temporal
      alternates.py        fig2_correspondence, figS_reverse_supplement, print redraws of Fig 1 and Fig 3 (see D1)
  analyses/
    01_cohort_and_profiles.py
    02_forward_direction.py
    03_reverse_direction.py
    04_glycemic_strata.py
    05_temporal_reproducibility.py
    06_figures_and_tables.py       draws every figure and table from saved aggregate outputs only (no parquet read)
  results/
    tables/                table1_baseline, table2_forward, table3_bidirectional, tableS1_baseline_full, tableS2_baseline_other_strata (.csv, .tex)
    figures/               fig1_forward, fig2_reverse, fig3_strata, figS1_temporal (.pdf, .png); alternates/ (see D1)
    numbers/               01_cohort.json … 05_temporal.json, every number quoted in the manuscript, unrounded
    aggregate/             analysis tables (masked), e.g. forward_eta.csv, reverse_prediction.csv, strata_specificity.csv
    profile_derivation/    frozen centroids JSON and k-selection records (aggregate, copied, see D3)
  restricted/              gitignored: participant-level intermediates written by the analyses
  tests/
    test_paper_numbers.py  every number in results/numbers vs the record and the manuscript text
    test_profiles.py       refit equals tracked centroids (skipped without data)
    test_cgm_metrics.py    synthetic TIR, CV, MAGE (sine of known amplitude), window boundaries
    test_stats.py          eta squared vs scipy kruskal, Cramér's V, ARI edge cases
    test_disclosure.py     masking and recovery on toy tables
    test_figures.py        port of verify_paper1_figures.py: plotted estimates read back vs source CSV
    test_repo_hygiene.py   no torch import, no absolute path, no file over 5 MB, participant-ID scan, small-cell scan
  tools/
    scan_participant_ids.py   ported from the record's tools/
  docs/
    paper1_draft_2026-10-19.pdf
    methods_notes.md
    decisions.md
    provenance.md
    small_cell_policy.md
    VERIFICATION.md        written in Phase 3
    preregistration/       README.md (index), reverse_SPEC.md, strata_SPEC.md, temporal_SPEC.md (unchanged copies)
  .gitignore
```

Changes from the tree in the brief, with reasons:
- `06_figures_and_tables.py` is added. Fig 2 combines outputs from 03 and 04, and Tables 2 and 3 do the same. Drawing everything in one place, from saved aggregates, lets a reviewer redraw every figure in seconds without the data. Scripts 01 to 05 then compute only.
- `tables.py` is added to `src`, so that analysis scripts only orchestrate.
- `plotting/` is a small subpackage instead of one `plotting.py`. One file with five figures plus alternates would exceed about 400 lines.
- `results/aggregate/` is added for the aggregate tables that back the numbers and figures (masked). `results/profile_derivation/` holds the frozen-profile records.
- `tools/scan_participant_ids.py` is added for the pre-commit scan.

`run_all.sh` runs 01 to 06, then `pytest`. Estimated runtime, from the record's logged times: 01 and 02 under 2 min; 03 about 6 to 10 min (B5 permutation nulls); 04 about 6 min; 05 about 9 min (323 s + 182 s logged); 06 under 1 min. That is roughly 25 to 30 min on CPU. Measured times go in `VERIFICATION.md`.

### config/paper1.yaml (outline)

```yaml
data:            # relative to $AIREADI_DATA_DIR
  parquet: enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet
  split: experiment_c_split_adapt6h_seed42/split_participants.csv
  cohort: enriched_multimodal/cohort.csv
  parquet_sha256: 1ee72b66...e446
restricted_dir: restricted          # inside the repo, gitignored
cohort: {n_parquet: 1892, n_split: 1591, strata: {...}, primary: t2d_oral_non_insulin}
profiles: {factors: [...], skew_log_threshold: 1.0, imputer_max_iter: 20, kmeans_n_init: 100, seed: 42, k: {healthy: 2, ...}}
cgm: {tir_mgdl: [70, 180], tbr_mgdl: 70, tar_mgdl: [180, 250], short_window_h: 48, min_readings: 2, sd_ddof: 1}
bootstrap: {n_forward_pairwise: 1000, n_default: 2000, ci_level: 0.95}
permutation: {n_agreement: 1000, n_prediction: 200, n_prediction_escalated: 1000, escalate_p: [0.01, 0.10]}
cv: {folds: 5, repeats: 20, seed: 42}
seeds: {...}     # every named seed from the record, verbatim, e.g. B3_bootstrap: 20261020
temporal: {window_days: 3, designs: [...], coverage_floor: ...}
criteria: {...}  # pre-registered thresholds (eta 0.06, ARI 0.10, AUROC 0.65, Cramér's V 0.20)
disclosure: {small_min: 1, small_max: 4, mask_small: "<5", mask_complement: "NR"}
strata_labels:
  healthy:              {short: Healthy, long: healthy}
  pre_diabetes:         {short: Prediabetes, long: prediabetes}
  t2d_oral_non_insulin: {short: Non-insulin T2D, long: type 2 diabetes treated without insulin}
  insulin_dependent:    {short: Insulin-treated, long: insulin-treated type 2 diabetes}
strata_colors: {...}    # four distinct, colour-blind-safe, no bright red (proposal in D5)
profile_colors: {...}   # C1/C2/C3 as in the current figures
```

## 2. Mapping: every current Paper 1 file

Paths are relative to the record repository; `01/` is `01_raw_cgm_differences/`.

### Profile derivation

| Current file | New location | Reason |
|---|---|---|
| `00_shared/code/scripts/within_subtype_phase0.py` | dropped; its Gate 0 counts move into `analyses/01` | Viability bootstraps and k pre-screen are Study 2 scaffolding. The paper quotes only the cohort counts, which 01 recomputes and asserts. |
| `00_shared/code/scripts/within_subtype_phase1.py` | `profiles.py` (`fit_profile_pipeline`, `order_by_hba1c`); k-selection dropped (D3) | Only the fit for the chosen k is needed to reproduce the labels (probe: exact). |
| `00_shared/code/scripts/within_subtype_phase2.py` | `profiles.py` (`assign_to_frozen_centroids`), about 15 lines | Nearest-centroid assignment only. Mantel, h0, PCA embedding and permutation code are Paper 2 and dropped. No torch is involved, but it imports phase1 and seaborn. |
| `00_shared/code/ssmcgm/analysis/within_subtype_config.py` | `config/paper1.yaml` (Paper 1 constants only) | Removes absolute paths and the `ssmcgm` package dependency. |
| `00_shared/outputs/profile_derivation/models/centroids/*.json` | `results/profile_derivation/centroids/` | Reference for the exact-refit check (aggregate). |
| `00_shared/outputs/profile_derivation/phase0_viability/*.json`, `phase1_clinical_clustering/*.json`, `cluster_size_validation_summary.csv`, `missing_data_sensitivity_ari.csv`, `decisions/GATE_A/B_REPORT.md`, `MANIFEST.json` | `results/profile_derivation/` (copied, see D3) | Frozen record of k choice and missing-data strategy. Not rerun. |
| `00_shared/outputs/profile_derivation/phase0_viability/participant_counts.csv` | dropped (01 recomputes) | Counts regenerated. |
| `restricted/00_shared/...` (joblib pipelines, validation and test assignments, exclusions) | not copied | Regenerated by refit (Step 0, point 6). |

### Section 3.1 forward

| Current file | New location | Reason |
|---|---|---|
| `01/audit/recompute_31.py` | `io.py`, `profiles.py`, `cgm_metrics.py`, `stats.py`; orchestrated by `analyses/01` and `02` | Clean independent implementation; the audit shows it reproduces the originals exactly. Paper-value comparison block → `tests/test_paper_numbers.py`. |
| `01/code/scripts/glycemic_metrics_by_clinical_cluster.py` | dropped | Reproduced exactly by `recompute_31.py` (RECOMPUTE_31.csv, all rows match). Its G1 to G3 figures are not in the paper. |
| `01/code/scripts/t2d_c2_centroid_and_binary_split.py` | C2 versus rest → `stats.py` + `analyses/02`; raw-space readout dropped | `recompute_31.c2_vs_rest` reproduces it. The raw-space z table is not quoted in this manuscript. |
| `01/code/scripts/t2d_exception_triptych.py` | constants → `config/paper1.yaml` | Only TIR bounds, 48 h window and column names were used. |
| `01/code/scripts/t2d_cv_focus_figure.py` | dropped | Figure G4, not in the paper. |
| `01/code/outputs/phase0_extensions/scripts/plot_phase0_raw_t2d_main_figure.py` | `plotting/figures.py::fig1_forward` | Draws manuscript Fig 1. The jitter seed rule (42 + 10·row + col) is kept so that point positions are identical. |
| `01/code/scripts/plot_fig1_raw_profiles_paper.py` | `plotting/alternates.py` or dropped (D1) | Print redraw, not in the PDF. |
| `01/audit/robustness_31.py` | `stats.py` (eta bootstrap CI, Dunn-Holm, test choice) + `analyses/02` | Source of the Table 2 CIs and the Dunn-Holm statement. k-means stability rows go into `analyses/01` as a sensitivity. |
| `01/audit/a5_split_medication_31.py` | `analyses/02` (split dependence, GLP-1/GIP contrasts) | Source of 0.021/0.139, p 0.043/0.084, −2.77 and 0.51. |
| `01/audit/pipeline_checks_31.py` | `analyses/01` (cohort, coverage and window checks as assertions) | Gate checks become asserted facts in `numbers/01_cohort.json`. |
| `01/audit/compare_rerun_31.py` | dropped | Audit scaffolding (rerun against SSM-CGM outputs). |
| `01/audit/run_section31_from_repo.py` | dropped | Audit scaffolding (runs the original scripts from SSM-CGM paths). |
| `01/audit/inputs/phase1_cluster_characterization.csv` | dropped | Only needed by the original Section 3.1 scripts. |
| `01/audit/*.csv`, `PIPELINE_CHECKS_31.json` | not copied; used as the reference in Phase 3 | Record outputs. |

### Section 3.2 reverse direction

| Current file | New location | Reason |
|---|---|---|
| `01/reverse_direction/features.py` | `cgm_features.py` | Feature definitions unchanged. |
| `01/reverse_direction/clustering.py` | `cgm_groups.py` | |
| `01/reverse_direction/reverse_tests.py` | `stats.py` (bootstrap eta, agreement, permutation null, refit bootstrap) | Single bootstrap and eta implementations. |
| `01/reverse_direction/prediction.py` | `prediction.py` | |
| `01/reverse_direction/population.py` | `analyses/04` (pooled B6) via `cgm_groups.py` and `stats.py` | The pooled analysis gives Fig 2D and the Section 3.3 paragraph. |
| `01/reverse_direction/run_reverse.py` | `analyses/03_reverse_direction.py` | Chunks 1 to 3 run in one pass. `--chunk` flags dropped. |
| `01/reverse_direction/plot_reverse_figure.py` | `plotting/figures.py::fig2_reverse` | Draws manuscript Fig 2. |
| `01/reverse_direction/plot_fig2_correspondence.py` | `plotting/alternates.py::fig2_correspondence`, `figS_reverse_supplement` | Kept as alternates, as you asked, with their captions. |
| `01/reverse_direction/plot_fig2_paper.py` | dropped (or alternates, D1) | Third Fig 2 variant, not in the PDF. |
| `01/reverse_direction/outputs/CHUNK*_SUMMARY.md`, `chunk*_summary.json` | dropped | Chunk logs. |

### Section 3.3 strata

| Current file | New location | Reason |
|---|---|---|
| `01/strata/run_strata.py` | `analyses/04_glycemic_strata.py` | S0 gate, S1 forward, S3 specificity. |
| `01/strata/s2_reverse.py` | `analyses/04` using `cgm_groups.py`, `stats.py`, `prediction.py` | Its private `boot_ci` and `logistic_null` duplicate prediction.py. Merged, with seeds kept. |
| `01/strata/split_composition.py` | `analyses/01` (split composition check) | It is a cohort property. |
| `01/strata/plot_strata_figure.py` | `plotting/figures.py::fig3_strata` | Draws manuscript Fig 3. |
| `01/strata/plot_fig3_paper.py` | `plotting/alternates.py` or dropped (D1) | Print redraw, not in the PDF. |
| `01/strata/outputs/CHUNK*.md`, `chunk*_summary.json` | dropped | Chunk logs. |

### Temporal

| Current file | New location | Reason |
|---|---|---|
| `01/temporal/windows.py` | `windows.py` | |
| `01/temporal/pipeline.py` | `cgm_groups.py` (frozen window-A pipeline) + `stats.py` (agreement inference, ICC) | |
| `01/temporal/window_tests.py` | `stats.py` + `analyses/05` | Its eta bootstrap is replaced by the shared one, with seeds kept. |
| `01/temporal/small_cells.py` | `disclosure.py` | Becomes the single masker (it is the general one). |
| `01/temporal/run_temporal.py` | `analyses/05_temporal_reproducibility.py` | |
| `01/temporal/plot_temporal_summary_v2.py` | `plotting/figures.py::figS1_temporal` | v2 three-panel design. |
| `01/temporal/plot_temporal_figure.py` | dropped | v1, superseded by v2. |
| `01/temporal/PAPER_TEXT_PROPOSALS.md`, `outputs/CHUNK*` | dropped | Text proposals and chunk logs. |

### Tables, style, verification, docs

| Current file | New location | Reason |
|---|---|---|
| `01/tables/make_baseline_tables.py` | `tables.py` (Table 1, S1, S2) + `disclosure.py` | Its milp masker merges with temporal's. Fisher-Freeman-Halton Monte Carlo seed is kept. |
| `02_continuous_vs_labels/code/scripts/phase0_ext/figure_main_config.py` | `plotting/style.py` | Removes the dependency on a Paper 2 folder. |
| `01/code/scripts/paper1_figure_style.py` | `plotting/style.py` | Merged. |
| `01/code/scripts/verify_paper1_figures.py` | `tests/test_figures.py` | The figure read-back check becomes a test. |
| `01/reverse_direction/SPEC.md`, `01/strata/SPEC.md`, `01/temporal/SPEC.md` | `docs/preregistration/` (unchanged) | |
| `RESULTS.md`, `STRATA_RESULTS.md`, `TEMPORAL_RESULTS.md` | condensed into `docs/decisions.md` and `docs/methods_notes.md` | Their numbers live in `results/numbers/`. |
| `01/audit/AUDIT_31.md`, `PIPELINE_31.md`, `RUNBOOK.md` | `docs/methods_notes.md` (definitions: windows, coverage, row order, TG/HDL units); findings list dropped | The audit stays in the record. |
| `docs/SMALL_CELL_POLICY.md` | `docs/small_cell_policy.md` (updated paths) | |
| `paper1_overleaf/` (now under `01/`) | replaced by `results/tables` and `results/figures` | Generated, never hand-copied. |
| `docs/PAPER_TEXT_TODO.md`, `docs/Paper_draft.pdf`, `docs/paper_draft.txt` | dropped | Older drafts are not the reference. |
| `tools/scan_participant_ids.py` | `tools/scan_participant_ids.py` | Pre-commit scan. |
| `01/logs/glycemic_metrics/*` | dropped | Logs of the original run. |

**Counts:**
- Source files that move into the new structure: about 32 `.py` (some split across several modules).
- New package modules: 13; analysis scripts: 6.
- Dropped outright: 11 scripts (`within_subtype_phase0`, `glycemic_metrics_by_clinical_cluster`, `t2d_cv_focus_figure`, `t2d_exception_triptych` (constants kept), `compare_rerun_31`, `run_section31_from_repo`, `plot_temporal_figure`, `plot_fig2_paper` (D1), the phase1 k-selection body, the phase2 h0 body, `glycemic_metrics` G1 to G3 plotting) plus all chunk summaries and logs.

## 3. Dropped files and why (summary)

- **Audit scaffolding:** `run_section31_from_repo.py`, `compare_rerun_31.py`, `RERUN_COMPARISON_31.csv`. They exist to compare against the SSM-CGM checkout, which the new repo never touches.
- **Chunk logs:** all `CHUNK*_SUMMARY.md` and `chunk*_summary.json`. The gated-chunk history belongs to the record.
- **Superseded figure scripts:** the temporal v1 plot, figures G1 to G4, and the unused Fig 1, 2 and 3 redraws (D1).
- **Original Section 3.1 scripts:** `glycemic_metrics_by_clinical_cluster.py` and `t2d_c2_centroid_and_binary_split.py`. `recompute_31.py` reproduces every row (RECOMPUTE_31.csv: all match; pairwise CI bounds identical under the same seeding).
- **`t2d_exception_triptych.py`:** only its constants are kept.
- **Paper 2 parts of `within_subtype_phase2.py`:** h0 and hidden-state metrics, Mantel tests, embedding PCA.
- **Phase 0 and Phase 1 k-selection runs:** frozen records kept (D3).

## 4. Paper map (from the 19 October PDF)

"Record source" is the record file the number was traced to. "New output" is where the new repo will write it. All values were found in a record output unless marked.

### Abstract, Methods 2.1 and 2.2

| Number | Record source | New output (script) |
|---|---|---|
| 1,892 processed; 1,591 passed; 564 / 405 / 458 / 164 | `PIPELINE_CHECKS_31.json` `cohort`; split file | `numbers/01_cohort.json` (01) |
| 325 train / 133 remaining | `PIPELINE_CHECKS_31.json` | `01_cohort.json` (01) |
| k = 3 (diabetes strata), k = 2 (healthy, prediabetes); 100 initializations | frozen manifest; config | `01_cohort.json` (01) |
| η² 0.042–0.078 | `kruskal_wallis_effect_sizes.csv` | `02_forward.json` (02) |
| ARI 0.039; η² 0.002–0.061; R² 0.59; AUROC 0.649; 0.605 → 0.660 | `B4_agreement.csv`, `B3_clinical_by_cgm_cluster.csv`, `B5_prediction.csv` (0.660 is only in the `note` string "HbA1c plus CGM 0.6598") | `03_reverse.json` (03), stored numerically |
| ARI ≤ 0.013; "approximately 40%"; ARI 0.160 | `S2_agreement_full.csv`, `S3_specificity.csv` (44% full, 37% 48 h), `B6_agreement.csv` | `04_strata.json` (04) |
| ARI 0.42; 0.030 and 0.036 | `T2_agreement.csv` (held-out, frozen window A), `T3_clinical_agreement.csv` | `05_temporal.json` (05) |

### Section 3.1, Table 1, Table 2, Fig 1

| Number | Record source | New output |
|---|---|---|
| Table 1, every cell (n, medians [IQR], n (%), P, missing counts HbA1c 30, C-peptide 25, TG/HDL 23, WHR 1) | `paper1_overleaf/table1_baseline_by_profile.csv` (+ hand formatting, D2) | `tables/table1_baseline.{csv,tex}` (06 from 01) |
| "all p < 0.0001" (max T2D p 2.9e-5) | `kruskal_wallis_effect_sizes.csv` | `02_forward.json` |
| Table 2: 0.068 [0.029, 0.120], 0.057 [0.021, 0.109], 0.064 [0.025, 0.116], 0.078 [0.038, 0.133], 0.042 [0.015, 0.087], 0.052 [0.023, 0.100] | `ROBUSTNESS_31.csv` `eta_squared_bootstrap_ci` (2,000 resamples; identical to `B3_forward_vs_reverse_eta_squared.csv`) | `tables/table2_forward` (06 from 02) |
| Table 2 pattern column; "supported by Dunn tests with Holm" | `pairwise_bootstrap_mean_differences.csv`; `ROBUSTNESS_31.csv` `dunn_holm` | `02_forward.json` |
| C1−C3 CV −0.76 [−2.06, 0.63] (48 h), −1.37 [−2.49, −0.21] (full); C1−C2 2.26 [0.99, 3.56]; C2−C3 3.02 | `pairwise_bootstrap_mean_differences.csv` (**1,000** resamples) | `02_forward.json` |
| η² 0.021 / 0.139 (full), 0.013 / 0.111 (48 h); p 0.043 / 0.084 | `ROBUSTNESS_31.csv` `eta_squared_subset`, `a5_split_dependence` | `02_forward.json` |
| GLP-1/GIP 57% / 32% / 34% | Table 1 CSV | `01_cohort.json` |
| −2.77 [−4.46, −1.05]; 0.51 [−0.99, 2.02] | `ROBUSTNESS_31.csv` `a5_cv_within_glp1_stratum` | `02_forward.json` |
| Fig 1 n labels and six η² annotations | `kruskal_wallis_effect_sizes.csv` | `figures/fig1_forward` |

### Section 3.2, Table 3, Fig 2

| Number | Record source | New output |
|---|---|---|
| Groups 188 / 182 / 88; median mean glucose 120 / 147 / 185; median CV 16.4 / 21.1 / 24.8 | `B2_centroids.csv` | `03_reverse.json` |
| Silhouette favoured two groups | `B2_partition_diagnostics.csv` (0.248 at k = 2 vs 0.164 at k = 3) | `03_reverse.json` |
| ARI 0.039 [0.020, 0.068], p = 0.001; Cramér's V 0.205 | `B4_agreement.csv` | `03_reverse.json` |
| HbA1c η² 0.422 [0.343, 0.497]; Table 3 reverse rows | `B3_forward_vs_reverse_eta_squared.csv` | `tables/table3_bidirectional` |
| R² 0.590 [0.455, 0.682]; 0.104; 0.062; 0.050; 0.000; 0.002 | `B5_prediction.csv` | `03_reverse.json` |
| R² ≈ 0.51 → 0.02 / 0.04; CV 0.079 → 0.051 | `B5_prediction.csv` | `03_reverse.json` |
| AUROC 0.649 [0.612, 0.686]; balanced accuracy 0.448 [0.400, 0.496]; chance 0.333; p = 0.005; 0.605 → 0.660; gain 0.055 [0.017, 0.094] | `B5_prediction.csv` | `03_reverse.json` |
| "3 of 15 partition and participant-subset combinations" | **not stored**; derivable from `B3_clinical_by_cgm_cluster.csv` age rows (5 partitions × {all, train, val+test}) | computed in 03 and stored |
| Fig 2A counts (85 66 37 / 82 28 72 / 25 17 46); 2B; 2C; 2D | `B4_contingency_tables.csv`, `B3_forward_vs_reverse…`, `B5_prediction.csv`, `B6_agreement.csv` | `figures/fig2_reverse` |

### Section 3.3, Fig 3

| Number | Record source | New output |
|---|---|---|
| Healthy 0.021 [0.003, 0.054]; prediabetes 0.071 [0.028, 0.124]; CV η² ≤ 0.002 | `S1_forward.csv` | `04_strata.json` |
| 88 healthy and 89 prediabetes assigned to C2 | `S3_assignment.csv` | `04_strata.json` |
| +0.17 [−0.77, 1.12]; −1.04 [−1.88, −0.12]; −2.39 [−3.41, −1.43]; Holm p 0.065 | `S3_specificity.csv`; T2D row from `t2d_c2_vs_rest_cv_binary_test.csv` (**1,000** resamples) | `04_strata.json` |
| ARI 0.004–0.013; AUROC 0.685 / 0.741 vs 0.664 / 0.628 | `S2_agreement_full.csv`, `S2_prediction.csv` | `04_strata.json` |
| Insulin C-peptide 0.085 [0.019, 0.197] vs 0.021 at 48 h | `S2_clinical_full.csv`, `S2_clinical_secondary.csv` | `04_strata.json` |
| Pooled ARI 0.160 [0.136, 0.187], p = 0.001; 0.037 vs 0.039; 0.093 vs null mean 0.088, p = 0.001 | `B6_agreement.csv` | `04_strata.json` |

### Numbers with no direct output

- **0.660** (HbA1c plus CGM AUROC): only inside a text note. The new repo stores it as a number.
- **"3 of 15":** computed, not stored. The new repo stores it.
- **"Approximately 40%":** a rounding of 44% (full) and 37% (48 h). Stored as both shares.
- **Table 1 cells:** formatting differs from the generated table (D2). Each cell is checked in Phase 3.

### Possible manuscript issues seen so far (to confirm in Phase 3, not fixed)

1. **Bootstrap count.** Methods says CIs come from 2,000 bootstrap resamples, which holds for η². The pairwise mean-difference CIs in 3.1 (−0.76, −1.37, 2.26) and the T2D C2-minus-rest reference (−2.39 [−3.41, −1.43], also in Fig 3C) use 1,000 resamples. The healthy and prediabetes S3 CIs use 2,000.
2. **Broken reference.** The Table 1 footnote cites "Supplementary Table ??".
3. **Feature list.** The Methods list of CGM features omits SD, the 24 h autocorrelation and the Lomb-Scargle circadian power share, all of which are in the primary partition.
4. **Old stratum label.** The Fig 2 and Fig 3 titles and panel labels use "T2D oral" and "T2D oral non-insulin". The redraw fixes this.
5. **Missing temporal text.** There is no temporal paragraph in the Results, and Fig S1 is not referenced, although the Abstract and Discussion use the temporal numbers.

## 5. Risks

### Random number streams

Every record script uses named, fixed seeds. Most bootstraps draw from a single generator in loop order, so a value depends on everything drawn before it in that loop. The rule for the port:
- Each analysis keeps the record's loop order exactly, including loops over partitions that the paper does not quote (for example the B3 secondary partitions share `b3_rng` with the primary). Nothing is dropped from inside a seeded loop. A loop body that is not needed can only be removed when it has its own generator.
- Seeds move verbatim into `config.seeds`. Offset rules are kept as named functions, for example:
  - `seed + counter` for pairwise differences;
  - `SEED + 900000 + counter` for C2 versus rest;
  - temporal `offset * 10 + …` and `offset * 1000 + s_i * 100 + f_i + 50·held_out`.
- One shared `bootstrap_eta(values, groups, rng, n)` replaces four copies. The copies are in `reverse_tests`, `run_strata`, `robustness_31` and `window_tests`, and also `bootstrap_mean_difference`. The port works because all copies draw the same sequence (`rng.integers` per resample, within-group or whole-sample as now). Phase 2 first checks that the copies are draw-for-draw identical. Any copy that differs (whole-sample against within-group resampling) is kept as a separately named function rather than merged silently.
- Parallel permutation nulls already use `SeedSequence.spawn`, so the result does not depend on `n_jobs`.
- k-means and iterative imputation depend on row order and library versions. Row order is kept (split-file order), and versions are pinned to the `ssmcgm` env.

**Expected outcome:** identical bootstrap and permutation results. If a difference above 0.001 appears, I stop and report it, as the brief says.

### Temporal outputs are rounded in the record

Where an n is masked, `T2_agreement.csv` and others round statistics to 2 decimals by design. The comparison at 1e-9 is against the record's unrounded participant-level and aggregate files under `restricted/01_raw_cgm_differences/temporal/`, where they exist. Otherwise it is against the rounded value at its rounding.

### Masking behaviour

There are two milp maskers today: `make_baseline_tables` and `temporal/small_cells`. Merging them could change which complementary cell gets masked as NR, because the choice of cell depends on iteration order. The test therefore requires that the masked tables equal the record's masked tables cell for cell, and that recovery remains impossible (0 recoverable). If the merged masker picks a different NR cell, I keep the record's choice rule.

Tables not masked in the record (B4, B6, B2, S2 contingency) will be masked in the new repo. Those tracked files will therefore differ from the record by design. That is a disclosure fix, not a numerical change, and it does not touch Fig 2A, whose cells are all ≥ 17.

### Hard-to-untangle dependencies

- **`within_subtype_phase2.py`:** only its about 15-line nearest-centroid step is needed. Resolved: the refit probe shows `profiles.py` reproduces every label without it.
- **Figure style in a Paper 2 folder:** `figure_main_config.py` is merged into `plotting/style.py`. The fonts, sizes and profile colours of Fig 1 to Fig 3 are kept. Stratum colours and labels change (D5).
- **Cross-imports in the record:** strata imports `robustness_31`, `run_reverse` and `s2_reverse`; temporal imports reverse `features` and `robustness_31`; reverse imports `recompute_31`. These all become package imports.
- **Intermediates:** analyses 04 and 05 read participant-level intermediates written by 03 (CGM features). The new repo writes these to `restricted/` inside the repo (gitignored). `run_all.sh` order enforces the dependency, and each script fails with a clear message if its input is missing.
- **Seaborn:** the record's Fig 1 uses seaborn violins. Dropping seaborn means redrawing the violins in matplotlib, which could change the KDE bandwidth and so the violin shapes. Proposal: keep seaborn (pinned 0.13.2) for Fig 1 only (D6).

### Environment

The `ssmcgm` env has torch installed. The new package never imports it, and the hygiene test greps for it. A fresh env from `environment.yml` (D7) proves the repo runs without torch or a GPU.

## 6. Decisions I need from you

- **D1. Figure variants.** The PDF uses the original summary figures for Fig 1, 2 and 3. Proposal:
  - restyle those three (labels and colours only) as the canonical `fig1_forward`, `fig2_reverse` and `fig3_strata`;
  - keep `fig2_correspondence` and `figS_reverse_supplement` under `results/figures/alternates/`, as you asked.

  Should the print redraws of Fig 1 and Fig 3 (`fig1_forward_cgm_by_profile`, `fig3_strata_summary`) and `plot_fig2_paper` also go into alternates, or be dropped? My recommendation is to keep only the two you named and drop the rest.
- **D2. Table 1 format.** Proposal: generate Table 1 exactly in the manuscript's format, with these rules written into `docs/decisions.md`:
  - integer age;
  - half-up rounding;
  - en-dash IQR;
  - short race labels;
  - P to 3 decimals.

  The full generated table becomes Table S1. The alternative is to keep the generated format and edit the manuscript.
- **D3. Profile k-selection.** Proposal: do not rerun the Phase 0 and Phase 1 k-selection (about 2,100 lines of Study 2 code, with bootstraps). Copy its aggregate records to `results/profile_derivation/` as frozen inputs, and have `01` refit the chosen k and assert equality with the frozen centroids. The alternative is to port k-selection as well, adding about 400 lines and several minutes.
- **D4. Sensitivity analyses quoted only in Methods.** Training-only and complete-case subsets, margin > 0.20, ANOVA, two-group CGM clustering, and k-means stability. Proposal: port all of them, since Methods names them, and record them in `numbers/`. Confirm.
- **D5. Stratum colours.** Proposal, from the Okabe-Ito colour-blind-safe set:

  | Stratum | Colour |
  |---|---|
  | Healthy | blue `#0072B2` |
  | Prediabetes | orange `#E69F00` |
  | Non-insulin T2D | bluish green `#009E73` |
  | Insulin-treated | reddish purple `#CC79A7` |

  Profile colours C1/C2/C3 stay as in the current figures. Long label for insulin-treated: "insulin-treated type 2 diabetes"?
- **D6. Seaborn.** Keep seaborn as a pinned dependency for the Fig 1 violins, so the shapes stay identical? The alternative is matplotlib only, with violin shapes changing slightly.
- **D7. Environment.** Install the package editable into `ssmcgm` (fast, but torch is present there), or create a light env `paper1` from `environment.yml`? My recommendation is a new light env, which proves the repo has no torch dependency.
- **D8. GitHub.** `gh` is not installed. Should I install it (conda-forge `gh`, then `gh auth login` interactively by you), or will you create the empty private repository `aireadi-clinical-profiles-cgm` on the web so that I push with `git`? Also confirm the name.
- **D9. Commit attribution.** The record repository uses no Claude co-author trailers. Same for the new repo?
