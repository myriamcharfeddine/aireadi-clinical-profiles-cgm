# Other glycemic strata: specification (not run)

Status: design written 2026-10-15 on branch `paper1-strata` (from `main` at 23420b3). Decisions and approval to run received 2026-10-15 (see the end of this file). Runs in two chunks, with a stop after each.

## Purpose

Section 3.1 and the reverse-direction test (`../reverse_direction/`) cover the T2D oral non-insulin stratum. This analysis applies the same forward and reverse tests to the three other strata, each with its own frozen profiles. It also replaces the paper's specificity sentence ("the corresponding coefficient of variation pattern was absent in the healthy and prediabetes groups", finding S31-03) with a direct test: assign healthy and prediabetes participants to the T2D centroids and compare CV.

No model hidden state is used. SSM-CGM-Stream (the forecasting architecture designed by Shakson Isaac, Isaac, Collin & Patel, arXiv:2510.04386) plays no part.

## Rules

- New files only, under `01_raw_cgm_differences/strata/`. Participant-level outputs go to `restricted/01_raw_cgm_differences/strata/`.
- `docs/PAPER_TEXT_TODO.md` is append-only. `main` is not pushed.
- Named constants and fixed seeds throughout. No causal language. Figures follow the style guide (sentence case, `STRATUM_COLORS`).
- CPU only. Ask before any step projected to exceed 10 minutes.
- All code imports `audit/recompute_31.py` and the reverse-direction modules (`features.py`, `clustering.py`, `reverse_tests.py`, `prediction.py`). It imports no Section 3.1 or `within_subtype_*` module.

## Strata, profiles and Gate 0

Profiles are the frozen ones from `00_shared/outputs/profile_derivation/` (no refit), applied by hand with `recompute_31.assign_labels`.

| Stratum | k | Missing-data rule | Expected n (train / val / test) | Profile sizes, all (train) | Status |
|---|---|---|---|---|---|
| Healthy | 2 | complete case | 538 of 564 (383 / 80 / 75) | 325 / 213 (238 / 145) | primary |
| Prediabetes | 2 | iterative imputation | 405 (288 / 61 / 56) | 184 / 221 (128 / 160) | primary |
| Insulin-treated | 3 | iterative imputation | 164 (116 / 25 / 23) | 53 / 79 / 32 (39 / 55 / 22) | exploratory |

**Gate 0 (per stratum, before any test).** The run stops for a stratum if any of these fails:

- n, split sizes and profile sizes equal the table above;
- labels equal `restricted/01_raw_cgm_differences/outputs/tables/participant_glycemic_metrics_by_cluster.csv` (0 mismatches);
- the three metrics match it to 1e-9 in both windows.

Part A already found 0 mismatches in all four strata (`audit/RECOMPUTE_31_participant_summary.csv`), so this is a re-check.

**Small groups.** In the insulin-treated stratum the validation-plus-test part has C1 14, C2 24 and C3 10 participants. Any analysis restricted to that part is therefore descriptive only (see stop conditions).

## S1. Forward direction, per stratum

Metrics as in Section 3.1, from `recompute_31.window_metrics`: mean glucose, TIR 70 to 180 mg/dL, CV (SD ddof 1); 48 h window (elapsed ≤ 48 h from the first row) and full record.

1. **Reproduction first.** Recompute the 12 existing rows of `01_raw_cgm_differences/outputs/tables/kruskal_wallis_effect_sizes.csv` for these strata. Each must match to 1e-9 in H, p and eta squared. Also reproduce the existing pairwise bootstrap rows of `pairwise_bootstrap_mean_differences.csv`, using the original seeding convention (1,000 draws, seed 42 plus a running counter). A mismatch is a stop condition.
2. **Kruskal-Wallis and eta squared** = (H − k + 1)/(n − k), clipped at 0, with a participant bootstrap 95% CI (`N_BOOTSTRAP = 2000`; resample participants across the stratum, so labels travel with participants; percentile interval).
3. **Pairwise comparisons.**
   - k = 2 strata: bootstrap mean difference C1 minus C2, resampling participants within group, 2,000 draws, 95% percentile CI.
   - Insulin-treated (k = 3): Dunn tests (tie-corrected) with Holm correction over its 3 pairs, plus bootstrap mean differences for each pair.
4. **Split dependence (as A5.1).** Eta squared in train only and in validation plus test, and their difference. The reference distribution comes from 2,000 random partitions that keep each profile's train count; two-sided p. Descriptive only for the insulin-treated stratum.

## S2. Reverse direction, per stratum

Same pipeline as Part B, with every fitted step restricted to the stratum.

- **Features.** Extended CGM feature set, full record, as in `reverse_direction/features.py`: minimal metrics, TBR, TAR > 180 and > 250, SD, MAGE (moving-average version, per contiguous run), daily-profile PCA, autocorrelation at 1, 3 and 24 h, and circadian Lomb-Scargle share. Coverage floor: 1,000 valid readings. The full-record features for all 1,591 participants already exist (`restricted/.../reverse_direction/pooled_features_window_full.csv`, from B6). They are reused after checking that the minimal features match `recompute_31` to 1e-9. The 48 h window is not run (primary is full record only), which keeps the runtime down.
- **Preprocessing.** Fit on the stratum's training participants only: daily-profile PCA (80% target, cap 4 components), log1p for |skew| > 1, standardization. Applied unchanged to validation and test.
- **Clustering.** k-means (`n_init` 100, seed 42) fit on the stratum's training participants, nearest centroid for the rest; display labels by increasing train-member mean glucose.
  - Primary k = the stratum's profile k: healthy 2, prediabetes 2, insulin-treated 3.
  - Sensitivity: k = 3 for healthy and prediabetes. Also reported but not used in any reading: silhouette-selected k over 2 to 6, and bootstrap stability (200 train resamples, ARI against the frozen fit).
- **Small CGM groups.** Any CGM group with fewer than 20 training participants makes that partition descriptive only: point estimates without p values, excluded from the readings.
- **B3.** For each of the six clinical variables (observed values after the post-CGM laboratory rule): Kruskal-Wallis across CGM groups, eta squared with a 2,000-draw participant bootstrap CI. HbA1c is shown in its own row. Clinical medians per CGM group. Medication flags per CGM group (descriptive).
- **B4.** Agreement with the stratum's own profiles: ARI, NMI and bias-corrected Cramér's V. Null: 1,000 permutations of the profile labels. Uncertainty: 2,000 participant bootstrap draws with labels fixed. Contingency tables.
- **B5.** Participant-held-out `RepeatedStratifiedKFold(5 folds, 20 repeats, seed 42)` within the stratum, stratified by profile. There are no Section 3.2 folds outside T2D. Every CGM transform is refit inside each outer fold.
  - CGM features to profile: multinomial or binary logistic, L2, `class_weight` balanced, C from an inner 5-fold CV on 10^-3 to 10^3. Report balanced accuracy and macro AUROC (for k = 2, AUROC of the positive class).
    - Reference models: HbA1c alone, and HbA1c plus CGM. HbA1c missing values take the train-fold mean.
    - Null: 200 label permutations with C fixed at the most frequent choice. Any primary p in [0.01, 0.10] is rerun with 1,000 permutations, as in Part B.
  - CGM features to each clinical variable: ridge, alpha from an inner 5-fold CV. Out-of-fold R² with a 2,000-draw bootstrap CI, and a 200-permutation null with the same escalation rule.

## S3. Specificity check: healthy and prediabetes against the T2D profiles

- **Assignment.** The frozen T2D pipeline is applied unchanged to all 564 healthy and 405 prediabetes participants: `t2d_oral_non_insulin_pipeline.joblib` (iterative imputer fit on T2D training participants, then log1p of HbA1c, C-peptide and TG/HDL, then the T2D scaler). Each participant is assigned to the nearest frozen T2D centroid. The T2D imputer fills missing values for healthy participants too, so all 564 are assigned (the healthy stratum's own profiles are complete-case). This is stated as a caveat: imputation uses regressions fit on T2D participants.
- **Descriptives, per stratum.**
  - Counts assigned to C1-like, C2-like and C3-like groups.
  - The relative assignment margin (d2 − d1)/d1.
  - The distance to the nearest centroid, compared with the distribution of the same distance in the 458 T2D participants. Participants beyond the 95th percentile of the T2D distances are flagged as out of range (`OUT_OF_RANGE_PERCENTILE = 95`).
- **Main test, per stratum and window.** CV in the C2-assigned group against everyone else in the stratum:
  - two-group Kruskal-Wallis eta squared, with a 2,000-draw participant bootstrap CI;
  - bootstrap mean difference (C2-assigned minus rest; 2,000 draws, resampling within group, 95% percentile CI).

  These are reported next to the T2D C2-versus-rest values: eta squared 0.039 (48 h) and 0.036 (full); difference −2.60 [−3.76, −1.34] and −2.39 [−3.41, −1.43].
- **Secondary.** Mean glucose and TIR for the same comparison.
- **Sensitivities** (main test repeated):
  1. Relative margin > 0.20 only.
  2. Excluding out-of-range participants.
  3. Within-stratum standardization. After the T2D imputer and log1p, each variable is standardized with the stratum's own training-participant mean and SD instead of the T2D scaler. Participants are then assigned to the nearest T2D centroid. This asks about relative profile shape within the stratum rather than absolute HbA1c or BMI level.
- **Stop rule for S3.** If the C2-assigned group has fewer than 20 participants in a stratum, the main test there is descriptive only. This is likely in the healthy stratum, given its low HbA1c and BMI, and would leave the specificity reading for that stratum undetermined.
- **Interpretive note (not a criterion).** Healthy and prediabetes participants have few or no diabetes medication flags. A C2-like CV difference there would therefore not depend on the GLP-1/GIP pattern seen in T2D (S31-02). The data cannot separate the explanations further.

## Multiplicity: Holm across strata within each family

Holm is applied to p values across the strata within each family listed below. Each metric-by-window or variable cell is its own family, with one test per stratum.

| Family | Tests per family | Families |
|---|---|---|
| F1. S1 forward Kruskal-Wallis | 3 (healthy, prediabetes, insulin-treated) | 6 (3 metrics × 2 windows) |
| F2. S2 B3 clinical differences, primary partition | 3 | 6 (one per clinical variable) |
| F3. S2 B4 agreement, ARI permutation p | 3 | 1 |
| F4. S2 B5 profile prediction, AUROC permutation p | 3 | 1 |
| F5. S2 B5 clinical prediction, R² permutation p | 3 | 6 |
| F6. S3 main test, CV C2-assigned vs rest, Kruskal-Wallis p | 2 (healthy, prediabetes) | 2 (windows) |

Within the insulin-treated stratum, Dunn pairs are Holm-corrected over its 3 pairs. Sensitivity analyses and secondary metrics (S3 mean glucose and TIR) are not corrected and are labelled as such.

**CIs and Holm.** Criteria stated through a CI use the unadjusted 95% interval and also require the Holm-adjusted p of the same family to be below 0.05. For example, "CI excluding 0" for the S3 main test means the bootstrap CI excludes 0 and the F6 Holm-adjusted Kruskal-Wallis p < 0.05.

## Pre-registered reading

Written before any result exists.

### Specificity (S3 main test, CV, C2-assigned minus rest, both windows)

| Reading | Rule |
|---|---|
| **Specificity supported** | In both healthy and prediabetes, in both windows: the difference CI includes 0, or the point estimate is less than half the T2D difference (that is, between −1.30 and +1.30 at 48 h, and between −1.20 and +1.20 full record). |
| **Not specific** | In at least one of the two strata, in at least one window: the difference has the same sign as T2D (negative), its CI excludes 0, and the F6 Holm-adjusted p < 0.05. |
| **Inconclusive** | Anything else: an opposite-sign difference with CI excluding 0; a CI including 0 but a point estimate at least half the T2D value in one stratum while the other qualifies; or a stratum where the main test is descriptive only (C2-assigned group under 20). Reported with each criterion's status. |

The reading uses the primary assignment (T2D scaler, all participants). Sensitivities 1 to 3 are reported beside it and do not change the reading.

### Per-stratum reverse direction (S2, primary partition)

The Part B criteria apply unchanged (`../reverse_direction/SPEC.md`):

- **Clinical difference beyond HbA1c:** any non-HbA1c variable with eta squared ≥ 0.06 and CI lower bound > 0.01, plus F2 Holm-adjusted p < 0.05.
- **Agreement above null:** ARI ≥ 0.10 with permutation p < 0.05 (F3 Holm-adjusted).
- **Prediction above chance:** macro AUROC ≥ 0.65 with permutation p < 0.05 (F4 Holm-adjusted), and a gain over HbA1c alone whose bootstrap CI excludes 0.
- **Readings per stratum:**
  - reciprocal weakness: ARI CI upper < 0.10 and Cramér's V < 0.20, no clinical difference beyond HbA1c, and prediction not above chance;
  - asymmetry: any one of the three criteria above;
  - mixed: anything else.

The insulin-treated reading is labelled exploratory. The Part B population-effect reading does not apply here; B6 already covered the pooled cohort.

### Forward direction (S1)

Descriptive. Eta squared with CIs and Holm-adjusted p (F1) by stratum, compared with T2D. There is no pass or fail rule.

## Stop conditions

Stop, report, and wait if:

- Gate 0 fails for a stratum (counts, labels or metrics), or the S1 reproduction of existing rows fails;
- a group used in a primary test has fewer than 20 participants. The analysis then continues as descriptive only for that test, and the reason is listed. A failed Gate 0 stops the stratum entirely;
- any step is projected to run longer than 10 minutes.

## Planned outputs (after approval)

| File | Contents |
|---|---|
| `strata/outputs/S1_forward.csv` | Kruskal-Wallis, bootstrap CIs, Holm p |
| `strata/outputs/S1_pairwise.csv` | Pairwise bootstrap differences, Dunn |
| `strata/outputs/S1_split_dependence.csv` | Train vs validation-and-test |
| `strata/outputs/S1_reproduction.csv` | Existing rows against recomputed |
| `strata/outputs/S2_partitions.csv` | Sizes, silhouette, stability |
| `strata/outputs/S2_clinical.csv` | B3 |
| `strata/outputs/S2_agreement.csv` | B4 |
| `strata/outputs/S2_prediction.csv` | B5 |
| `strata/outputs/S3_assignment.csv` | Counts, margins, distance distributions |
| `strata/outputs/S3_specificity.csv` | Main test, secondary metrics, sensitivities, T2D reference |

Participant-level labels and features go to `restricted/01_raw_cgm_differences/strata/`.

`strata/STRATA_RESULTS.md` will hold, in order:

- a plain-language summary (5 sentences at most);
- the tables;
- the reading against the criteria above;
- the proposed paper text for Section 3.3, also appended to `docs/PAPER_TEXT_TODO.md`.

The summary figure `strata/outputs/figures/strata_summary.{png,pdf}` will have two panels:

1. eta squared with CIs by stratum, forward and reverse side by side, HbA1c on its own axis;
2. the specificity check: CV difference (C2-assigned minus rest) with CI for healthy, prediabetes and T2D, both windows, with the half-T2D band.

Note: the paper currently has no Section 3.3 for this; the proposed text will say where it fits.

## Runtime estimate (CPU, 12 cores, from the Part A and Part B timings)

| Part | Work | Estimate |
|---|---|---|
| Loading and Gate 0 | parquet, labels, metrics for 1,565 participants | about 30 s |
| S1 | Reproduction; 2,000-draw Kruskal-Wallis bootstraps (3 strata × 3 metrics × 2 windows); pairwise bootstraps; Dunn; 2,000 split partitions × 18 cells | 1 to 2 min |
| S3 | T2D-pipeline assignment of 969 participants (three variants); 2,000-draw bootstraps for 2 strata × 3 metrics × 2 windows × 4 variants | about 1 min |
| **Chunk 1 (S1 + S3)** | | **2 to 4 min** |
| S2 features | reuse B6 full-record features after the check (recomputing instead would add about 3 min) | under 30 s |
| S2 clustering | 5 partitions, silhouette scans, 200-resample stability | about 1.5 min |
| S2 B3 and B4 | bootstraps and permutations, 5 partitions | about 1.5 min |
| S2 B5 | logistic with inner CV, 100 outer folds × 3 models × 3 strata, in parallel; 200-permutation nulls; ridge for 6 variables × 3 strata | 3 to 5 min, plus 1 to 2 min per escalated null |
| **Chunk 2 (S2)** | | **6 to 9 min**, nearer 10 if several nulls escalate; I would stop and ask before exceeding 10 |

Proposed order: chunk 1 = S1 + S3, then a stop and summary; chunk 2 = S2, then a stop, the summary, `STRATA_RESULTS.md` and the figure.

## Decisions (2026-10-15)

1. **CI and Holm:** criteria stated through a CI require both the unadjusted 95% CI and a Holm-adjusted p < 0.05 in the same family (as written above).
2. **Specificity readings:** the three readings stand (supported / not specific / inconclusive).
3. **S3 healthy:** all 564 healthy participants are assigned through the T2D pipeline, including the 26 without a profile of their own; the imputation caveat is stated.
4. **S2 window:** full record is primary. The 48 h window is added as a secondary partition with the same k, for each stratum. Features for the 48 h window are computed with `reverse_direction/features.py` for the three strata (they do not exist from B6). The 48 h partitions are reported beside the primary and do not enter any reading. If the chunk 2 estimate exceeds 10 minutes, it is split (S2 full record first, then the 48 h secondary) and approval is asked.
