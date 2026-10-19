# Provenance

This repository was rebuilt on 19 October 2026 from the record repository `myriamcharfeddine/SSM-CGM-stream-AI` (private), branch `main` at commit `a36acdb`. The record is the frozen archive of how every result was first produced, including chunk logs, audits and superseded scripts. Nothing here was copied from its git history. The code was ported module by module, keeping the same seeds and the same order of random draws. `docs/VERIFICATION.md` shows that the outputs match.

Columns: new file, then the record file(s) it was ported from, the record commit that last changed that file, and the first 12 characters of its sha256.

## Package (`src/cgmprofiles/`)

| New file | Record source | Commit | sha256 |
|---|---|---|---|
| `io.py` | `01_raw_cgm_differences/audit/recompute_31.py` (`load_split`, `load_clinical_frame`, `load_cgm`); `reverse_direction/features.py` (`load_timezones`); `reverse_direction/run_reverse.py` (`pq_meds`) | a9ea17e, eb9b544, 551e652 | 8b556eee7c05 |
| `profiles.py` | `00_shared/code/scripts/within_subtype_phase1.py` (`fit_transform_matrix`, k-means fit, HbA1c display order); `within_subtype_phase2.py` (nearest-centroid step only); `audit/recompute_31.py` (`impute_by_hand`, `transform_by_hand`, `assign_nearest`, `assign_labels`) | 6f1534e, a9ea17e | 7b7b7041d041, 4d6b65500295, 8b556eee7c05 |
| `cgm_metrics.py` | `audit/recompute_31.py` (`window_metrics`, `participant_metrics`) | a9ea17e | 8b556eee7c05 |
| `cgm_features.py` | `reverse_direction/features.py` | eb9b544 | e699d70e3da8 |
| `cgm_groups.py` | `reverse_direction/clustering.py` (GMM and BIC scans dropped); `temporal/pipeline.py` (frozen window pipeline) | eb9b544, 08b452d | f703cb3b5395, 9f8cb60cb1c5 |
| `stats.py` | `audit/recompute_31.py` (ranks, Kruskal-Wallis, eta squared, bootstrap difference); `audit/robustness_31.py` (ANOVA, Holm, Dunn); `reverse_direction/reverse_tests.py` (`bootstrap_eta`); `audit/a5_split_medication_31.py` (`two_sided_p`) | a9ea17e, 6f432c7 | 8b556eee7c05, 8ebbffcd18fd, 8fd27e948080, 30cd612f5cf0 |
| `agreement.py` | `reverse_direction/reverse_tests.py` (agreement, nulls, bootstrap); `temporal/pipeline.py` (table statistics, window agreement, ICC) | 6f432c7, 08b452d | 8fd27e948080, 9f8cb60cb1c5 |
| `prediction.py` | `reverse_direction/prediction.py`; `strata/s2_reverse.py` (two-class AUROC, bootstrap that skips failed resamples) | 551e652, 7f43b4e | 070fb2f40337, 9b4d49859465 |
| `windows.py` | `temporal/windows.py` (days 1 to 3 and days 4 to 6 designs); `temporal/run_temporal.py` (eligibility and transition masking) | 08b452d | 5757526e3ccb, a79c91373c94 |
| `disclosure.py` | `tables/make_baseline_tables.py` (block masker); `temporal/small_cells.py` (linked masker) | 747afad, 08b452d | f83baa4b3437, c8a3301e1fe9 |
| `tables.py` | `tables/make_baseline_tables.py` | 747afad | f83baa4b3437 |
| `plotting/style.py` | `02_continuous_vs_labels/code/scripts/phase0_ext/figure_main_config.py`; `01_raw_cgm_differences/code/scripts/paper1_figure_style.py` | 6f1534e, f2b7a8f | 86362ff94af1, ce14382494f9 |
| `plotting/fig_forward.py` | `code/outputs/phase0_extensions/scripts/plot_phase0_raw_t2d_main_figure.py` | 6f1534e | e357c6d82a7c |
| `plotting/fig_reverse.py` | `reverse_direction/plot_reverse_figure.py` | ffefb30 | 58a096fdec1a |
| `plotting/fig_strata.py` | `strata/plot_strata_figure.py` | 2ba9f4d | fe99b3ed9a0c |
| `plotting/fig_temporal.py` | `temporal/plot_temporal_summary_v2.py` | 3706911 | 6805774e3662 |
| `plotting/print_style.py` | `01_raw_cgm_differences/code/scripts/paper1_figure_style.py` | f2b7a8f | ce14382494f9 |
| `plotting/print_forward.py` | `01_raw_cgm_differences/code/scripts/plot_fig1_raw_profiles_paper.py` | f2b7a8f | 4bfeca000ea1 |
| `plotting/print_reverse.py` | `01_raw_cgm_differences/reverse_direction/plot_fig2_paper.py` | ffefb30 | ac5c55603e23 |
| `plotting/print_strata.py` | `01_raw_cgm_differences/strata/plot_fig3_paper.py` | f2b7a8f | b248596e0017 |
| `plotting/fig_reverse_alternates.py` | `01_raw_cgm_differences/reverse_direction/plot_fig2_correspondence.py` (and its two caption files) | ffefb30 | bf3aece70fa2 |

## Analyses

| New file | Record source |
|---|---|
| `analyses/01_cohort_and_profiles.py` | `audit/pipeline_checks_31.py` (cohort checks), `within_subtype_phase1.py` (fit), `tables/make_baseline_tables.py` (`main`) |
| `analyses/02_forward_direction.py` | `audit/recompute_31.py` (`forward_tests`, `c2_vs_rest`), `audit/robustness_31.py` (Methods-named parts), `audit/a5_split_medication_31.py` (A5.1 and the GLP-1/GIP contrasts) |
| `analyses/03_reverse_direction.py` | `reverse_direction/run_reverse.py` chunks 1 to 3 |
| `analyses/04_glycemic_strata.py` | `strata/run_strata.py` chunks 1, 2a and 2b; `strata/s2_reverse.py`; `reverse_direction/population.py` (B6) |
| `analyses/05_temporal_reproducibility.py` | `temporal/run_temporal.py` chunks 1 to 3; `temporal/window_tests.py` (T3, T4 contrast, full-record reference) |
| `analyses/06_figures_and_tables.py` | the four figure scripts above; Tables 2 and 3 had no script in the record (they were typed into the manuscript) |
| `tests/test_figures.py` | replaces `code/scripts/verify_paper1_figures.py` (ffefb30), which checked the print redraws that are not in this repository |
| `tools/scan_participant_ids.py` | `tools/scan_participant_ids.py` (bad35d3), data path from the config |

## Frozen inputs copied unchanged

| New file | Record file | Commit |
|---|---|---|
| `results/profile_derivation/centroids/*.json` | `00_shared/outputs/profile_derivation/models/centroids/*.json` (frozen 2026-08-03) | 6f1534e |
| `results/profile_derivation/k_selection_frozen.json` | `00_shared/outputs/profile_derivation/phase1_clinical_clustering/k_selection_summary.json` | 6f1534e |
| `docs/preregistration/reverse_direction_SPEC.md` | `01_raw_cgm_differences/reverse_direction/SPEC.md` | 8cc3f63 |
| `docs/preregistration/glycemic_strata_SPEC.md` | `01_raw_cgm_differences/strata/SPEC.md` | bb893ec |
| `docs/preregistration/temporal_SPEC.md` | `01_raw_cgm_differences/temporal/SPEC.md` | c6d824b |
| `docs/small_cell_policy.md` | `docs/SMALL_CELL_POLICY.md` | 747afad |
| `docs/paper1_draft_2026-10-19.pdf` | `01_raw_cgm_differences/Clinical_profiles_..._(6).pdf` | 5aecdae |

## Data

| Input | Path under `AIREADI_DATA_DIR` | sha256 |
|---|---|---|
| Processed AI-READI parquet | `enriched_multimodal/final_multimodal_dataset_20260515_184339.parquet` | `1ee72b66689d…e446` |
| Participant split | `experiment_c_split_adapt6h_seed42/split_participants.csv` | `dd92db0652a3…` (identical to the record's restricted copy) |

The streaming forecaster studied in Paper 2 (SSM-CGM-Stream, designed by Shakson Isaac) plays no part in this repository.
