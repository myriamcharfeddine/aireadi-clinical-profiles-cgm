# Reverse-direction test: specification (not run)

Status: design written 2026-10-13; decisions taken 2026-10-14 (see the end of this file). Approved to run in three chunks, with a stop and a go after each chunk.

## Question

Section 3.1 starts from the clinical profiles C1, C2 and C3 and asks whether observed glucose differs between them. This test reverses the direction: if participants are grouped from CGM data alone, do the groups match C1/C2/C3, and do the six clinical variables differ between the CGM groups? A weak forward result does not imply a weak reverse result. The two directions condition on different partitions, so either one can be strong while the other is weak.

No model hidden state is used. SSM-CGM-Stream (the forecasting architecture designed by Shakson Isaac, Isaac, Collin & Patel, arXiv:2510.04386) plays no part.

## Fixed inputs, taken from Part A

| Item | Value | Source |
|---|---|---|
| Participants | the 458 of `t2d_oral_non_insulin` (train 325, val 69, test 64), labels C1/C2/C3 = 192/111/155 | `recompute_31.assign_labels` |
| Split | `restricted/00_shared/data/experiment_c_split_adapt6h_seed42/split_participants.csv` | `recompute_31.load_split` |
| CGM | `cgm_glucose_mean`, 5-minute grid, NaN for missing | `recompute_31.load_cgm` |
| 48 h window | elapsed ≤ 48 h from the participant's first row (inclusive, 577 slots) | `recompute_31.window_metrics` |
| Full record | every row of the untrimmed record | same |
| Clinical variables | the six factors after the post-CGM laboratory rule; TG/HDL as a mg/dL ratio | `recompute_31.load_clinical_frame` |
| Seeds | 42 unless stated | |

**Gate 0 (before any new computation).** Recompute the 458 labels and the three Section 3.1 metrics with `recompute_31.py`, and assert:

- n = 458;
- 0 label mismatches against `restricted/01_raw_cgm_differences/outputs/tables/participant_glycemic_metrics_by_cluster.csv`;
- maximum metric difference ≤ 1e-9 in both windows.

If any check fails, stop (see stop conditions).

New code goes in `01_raw_cgm_differences/reverse_direction/`: `features.py`, `clustering.py`, `reverse_tests.py`, `prediction.py`, `population.py` and `run_reverse.py`. All of them import `recompute_31.py` and none import Section 3.1 or `within_subtype_*` modules. Participant-level outputs go to `restricted/01_raw_cgm_differences/reverse_direction/`, and aggregate tables to `reverse_direction/outputs/`. Figures follow the style guide: sentence case, `STRATUM_COLORS` from `within_subtype_config`. CGM clusters are drawn in neutral greys or a separate ordered palette, so they cannot be confused with C1/C2/C3.

## B1. CGM features

All features are computed per participant and window from valid readings only.

### Coverage and segment rules

Part A found high coverage (2 of 458 below 80% in the first 48 h, none under 48 h of data) and 17 interior gaps over 2 h in 13 participants.

- `MIN_COVERAGE_48H = 0.70` of 576 expected readings, and `MIN_READINGS_FULL = 1000`. Participants below either threshold get missing features for that window. Part A suggests nobody falls below; if anyone does, they are listed.
- Primary analyses keep everyone who passes. A sensitivity analysis drops participants below 80% coverage (2 expected).
- Summary features (mean, SD, CV, TIR, TBR, TAR) pool all valid readings in the window, across gaps, as Section 3.1 does.
- Sequence features (MAGE, autocorrelation, spectral) are computed within contiguous runs only. A run is split at any gap longer than `MAX_INTERPOLATION_GAP_MIN = 30` minutes; this is the CGM gap threshold used for the cohort segmentation (`cohort_selection_metadata.json`). Gaps of 30 minutes or less are linearly interpolated for these features only. Runs shorter than `MIN_RUN_H = 6` hours are dropped. Per-run values are averaged, weighted by run length.
- Clock time: `timestamp_local` stores every participant in America/Los_Angeles time, while the 719 UAB participants (all strata) are on America/Chicago (`timezone` column). Hour-of-day features must convert each participant's timestamps to their own `timezone` before taking the hour. Elapsed-time windows are unaffected.

### Minimal set (3 features)

Mean glucose, TIR 70 to 180 mg/dL, and CV (SD ddof 1 / mean). These are identical to Section 3.1 and are checked in Gate 0.

### Extended set

| Feature | Definition |
|---|---|
| Minimal set | as above |
| TBR < 70 | % of valid readings below 70 mg/dL |
| TAR > 180 | % of valid readings above 180 mg/dL |
| TAR > 250 | % of valid readings above 250 mg/dL |
| SD | SD of valid readings, ddof 1 |
| MAGE | Moving-average algorithm of Fernandes et al. (2022), as in iglu `mage_ma`: short moving average of 5 readings and long moving average of 32 readings. Their crossings delimit excursions; the peak or nadir between crossings is taken from the raw series; excursions whose amplitude exceeds 1 SD of the run are kept; MAGE is the mean amplitude of kept excursions, up and down pooled. It is computed per contiguous run (above) and reimplemented in the script with a unit test on a synthetic series of known amplitude. No R dependency. |
| Mean daily profile | Hourly means by local clock hour (0 to 23) over all days in the window, giving 24 values. An empty hour takes the participant's window mean. The values are centred on the participant's own mean, so the profile carries shape rather than level, which is already in mean glucose. PCA is fit on the 325 training participants only and applied to everyone. Components are kept up to 80% cumulative variance, capped at `MAX_PROFILE_PCS = 4`. The variance retained is reported in the output, and the loadings are plotted. |
| Variability spectrum | Autocorrelation of the run-wise mean-centred series at lags of 1 h, 3 h and 24 h (pairwise-complete readings within runs), plus the share of Lomb-Scargle power in periods of 20 to 28 h relative to 2 to 28 h (gaps allowed, no interpolation). The 24 h autocorrelation and the circadian share are computed for the full record only: the 48 h window holds at most two cycles, so in that window these two are omitted. |

### Transformations

Fit on the training participants only, mirroring the clinical pipeline:

1. log1p for any feature with |skew| > 1 on train (TBR, TAR > 250 and MAGE are expected to qualify);
2. a `StandardScaler`.

Both fitted transforms are saved (joblib, restricted) and applied unchanged to val and test.

## B2. CGM clustering

- **Primary.** k-means with k = 3, `n_init` = 100, `random_state` = 42, fit on the 325 training participants' standardized features; val and test are assigned to the nearest centroid. This is run for each window (48 h, full) and each feature set (minimal, extended), giving 4 partitions. The primary partition is **extended, full record**: Section 2.5 names full-record CV the primary outcome, and the extended set is the richer description. The other three are secondary. CGM clusters get display labels G1, G2 and G3 by increasing centroid mean glucose, the analogue of the HbA1c ordering of C1 to C3.
- **Sensitivity.**
  - Silhouette-selected k over 2 to 6 on train, with ties within 0.02 going to the smaller k (`SILHOUETTE_TOLERANCE` from the clinical pipeline).
  - A Gaussian mixture (full covariance, `n_init` = 10, `random_state` = 42) at k = 3 and at BIC-selected k over 2 to 6, fit on train, with val and test assigned by maximum posterior.
- **Report.**
  - sizes by split;
  - centroids in standardized units and as raw medians per cluster;
  - train silhouette;
  - bootstrap stability (200 train resamples, refit, ARI against the frozen fit);
  - ARI between the k-means and GMM partitions.

## B3. Clinical differences across CGM clusters

For each of the six clinical variables and each CGM partition:

- **Test.** Kruskal-Wallis on observed values; participants missing that variable are excluded from that test. Eta squared = (H − k + 1)/(n − k), clipped at 0, with a participant-bootstrap 95% CI (2,000 resamples, labels travel with participants). A sensitivity analysis uses the frozen-pipeline imputed values for the 32 participants with a missing variable.
- **Side by side.** A table with one row per variable or metric shows the forward result (eta squared of each CGM metric across C1/C2/C3, from Part A, with its CI from `ROBUSTNESS_31.csv`) beside the reverse result (eta squared of each clinical variable across G1/G2/G3).
- **HbA1c** is shown in its own row and flagged. Groups formed on glucose are expected to differ in HbA1c because HbA1c reflects mean glucose over about three months, so its difference is not evidence for the profiles.
- **Descriptive only.** Medication flags (GLP-1/GIP, metformin, SGLT2, sulfonylurea, insulin) by CGM cluster, with chi-square and Cramér's V. Part A found GLP-1/GIP concentrated in C2 (S31-02).
- **By split.** Each eta squared is also reported for train only and for val and test only, given the split dependence found in Part A (S31-01).

## B4. Agreement between the two partitions

For the primary partition and each secondary one:

- the 3 × 3 contingency table (counts and row percentages), by split and overall;
- ARI, NMI (arithmetic normalisation) and Cramér's V (bias-corrected, Bergsma 2013);
- **Null distribution.** 1,000 permutations of the C1/C2/C3 labels across the 458 participants, with the CGM labels fixed; p = (1 + number of permuted values ≥ observed) / 1,001. A second null permutes within split, so that split-level differences cannot drive agreement.
- **Uncertainty.**
  - Primary: 2,000 participant bootstrap resamples with both labels fixed (percentile CI).
  - Secondary: 200 resamples that refit the CGM k-means on the resampled training participants and reassign everyone. This captures clustering instability.

## B5. Symmetric prediction

**Folds.** Participant-held-out cross-validation reusing the Section 3.2 splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)` over the pooled 458, stratified by C1/C2/C3, on the Section 3.2 population in its row order (`02_continuous_vs_labels/code/scripts/phase0_ext/a_cluster_vs_continuous.py` `run_repeated_cv`). Reuse is verified first: rebuild the population and reproduce the published full-record M2 R² (mean glucose 0.5105, TIR 0.5103, CV 0.079; `02_continuous_vs_labels/outputs/tables/table1_model_comparison.csv`) to 1e-4. If that fails, use the same splitter on the `recompute_31` order and say so.

**Inside each fold.** Every transform is refit on the fold's training rows: the log1p choice, the scaler, the daily-profile PCA, and imputation for clinical predictors (iterative imputer, as in Section 3.2). Out-of-fold predictions are averaged over the 20 repeats.

| Direction | Model | Metric | Null |
|---|---|---|---|
| Clinical to CGM metrics | reuse Section 3.2 (M2: six continuous variables, linear regression) | out-of-fold R² (published) | as published |
| CGM features to each clinical variable | ridge regression; alpha from an inner 5-fold CV on the grid 10^-3 to 10^3 (13 values) | out-of-fold R² per variable, participant-bootstrap CI | 200 permutations of the target (see note) |
| CGM features to C1/C2/C3 | multinomial logistic regression, L2; C from an inner 5-fold CV on 10^-3 to 10^3 (13 values); class weights balanced | balanced accuracy (chance 1/3) and macro one-vs-rest AUROC (chance 0.5), participant-bootstrap CI | 200 permutations of the labels (see note) |

Note on the null: each permutation reruns the full 5 × 20 cross-validation with the hyperparameter fixed at the value selected most often in the unpermuted run. This keeps the cost bounded; the choice is stated in the output. `N_PERMUTATIONS_DEFAULT = 200`. Escalation rule (applies to every 200-permutation null on the primary partition; the B4 nulls already use 1,000): if a p value falls in [0.01, 0.10], that one null is rerun with `N_PERMUTATIONS_ESCALATED = 1000` and the 1,000-permutation p value is the one reported.

Each direction is run for both windows and both feature sets.

**Without HbA1c.** HbA1c ties glucose to the profiles by construction, so each direction is also run without it:

- clinical to CGM: refit with the five non-HbA1c variables;
- CGM to clinical: the five non-HbA1c targets are already reported separately; HbA1c is reported but flagged;
- CGM to C1/C2/C3: two variants.
  - (a) Incremental value over HbA1c: a model with HbA1c alone, against HbA1c plus CGM features. Report the gain in balanced accuracy and AUROC with a paired bootstrap CI.
  - (b) Not run (decision 2026-10-14). Possible future work: labels from a five-variable re-derivation of the profiles without HbA1c, using the frozen pipeline settings. That would be a new clustering of the training participants.

## B6. Population check (all four strata, 1,591 participants)

- **Features.** As B1, for every participant in the split file; all 1,591 have CGM.
- **Clustering.** The B2 procedure fit on the 1,131 training participants, at k = 3, at k = 4 (the number of strata), and at silhouette-selected k.
- **Agreement (B4 measures)** with:
  - the diagnostic stratum (4 groups);
  - the combined stratum-profile label: healthy 2 + prediabetes 2 + T2D oral 3 + insulin-dependent 3 = 10 groups. The 26 healthy participants without a profile (complete case) are excluded for this label.
- **Separating diagnosis from profile.**
  - A permutation null that shuffles the combined label within stratum, so that only within-stratum profile structure can create agreement.
  - Agreement of the pooled CGM clusters with C1/C2/C3 inside the 458 only.
- **Clinical differences (B3)** across the pooled CGM clusters, overall and within stratum.

## Pre-registered interpretation

Written before any Part B result exists. Criteria are applied to the primary partition (extended features, full record, k-means k = 3). The secondary partitions are reported but do not decide the reading unless all four disagree with the primary.

Definitions:

- **Clinical differences beyond HbA1c**: any of the five non-HbA1c variables with eta squared ≥ 0.06 (Cohen's medium) and bootstrap CI lower bound > 0.01.
- **Agreement above null**: ARI ≥ 0.10 with permutation p < 0.05.
- **Prediction above chance**: macro AUROC ≥ 0.65 with permutation p < 0.05, and the B5 variant (a) gain over HbA1c with a CI excluding 0.

Readings:

1. **Reciprocal weakness.** All of the following:
   - ARI 95% CI upper bound < 0.10 and Cramér's V < 0.20;
   - no clinical difference beyond HbA1c;
   - CGM to C1/C2/C3 macro AUROC < 0.65, or not above its permutation null.

   Section 3.1's conclusion (modest, overlapping links between profiles and glucose) then holds in both directions.
2. **Asymmetry.** Any clinical difference beyond HbA1c, or agreement above null, or prediction above chance. Section 3.1 is then a one-directional result, and the paper must say that glucose-defined groups carry clinical structure that the forward test did not show.
3. **Population effect.** In B6: ARI with diagnostic stratum ≥ 0.10 (p < 0.05), while the T2D criteria meet reading 1 and the within-stratum permutation null is not rejected. The cohort-level link between glucose and clinical state is then mainly explained by diagnosis.
4. **Mixed.** Anything not covered above (for example, agreement above null but no clinical difference beyond HbA1c). This is reported as mixed, naming each criterion met or not, and is not forced into readings 1 to 3.

## Stop conditions

Stop, write what was found, and wait if:

- any CGM cluster in the primary partition has fewer than `MIN_CLUSTER_N = 20` participants among the 325 training participants or among the 458;
- Gate 0 fails: participants are not the 458, labels or windows do not match Part A, or the minimal features differ from Section 3.1 by more than 1e-9;
- the Section 3.2 folds cannot be rebuilt and the fallback is needed (report it, then continue only with approval);
- any step is projected to run longer than 10 minutes.

## Runtime estimate (CPU, 12 cores, from the Part A timings)

Data loading takes about 10 s (Part A).

| Step | Estimate |
|---|---|
| B1 features, 458 participants × 2 windows (MAGE, autocorrelation and Lomb-Scargle per run) | 20 to 40 s |
| B1 features, 1,591 participants (B6) | 1 to 2 min |
| B2 k-means and GMM, k scans, 200-resample stability, 4 partitions | about 1 min |
| B3 Kruskal-Wallis plus 2,000 bootstraps, 6 variables × 4 partitions × 3 subsets | about 1 min |
| B4 1,000 + 1,000 permutations, 2,000 bootstraps, 200 refit bootstraps, 4 partitions | about 1 min |
| B5 cross-validation with inner CV (100 outer folds × 2 windows × 2 feature sets × directions) | 2 to 3 min |
| B5 permutation nulls (200 × 100 folds, fixed hyperparameters, parallel) | 4 to 7 min |
| B6 clustering, agreement and nulls on 1,591 | 1 to 2 min |
| **Total** | **about 12 to 18 min.** B5 permutation nulls dominate. |

The run would be split into three approved chunks: B1 and B2; B3, B4 and B6; B5. Each chunk stays under 10 minutes.

## Decisions (2026-10-14)

1. **Primary partition**: extended features, full record. The other three partitions (extended 48 h, minimal full record, minimal 48 h) are secondary.
2. **B5 variant (b)**: not run; kept above as possible future work. Variant (a), incremental value over HbA1c, is run.
3. **Coverage floor**: 70% of 576 readings in the first 48 h, and 1,000 valid readings for the full record.
4. **Permutations**: 200 with fixed hyperparameters; any primary-partition p value in [0.01, 0.10] is rerun with 1,000 permutations.

Chunks: 1 = B1 and B2; 2 = B3, B4 and B6; 3 = B5. Each chunk is committed on `section31-audit` and followed by a stop and a summary against the criteria above.
