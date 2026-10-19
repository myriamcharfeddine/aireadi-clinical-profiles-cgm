# Methods notes

Definitions and conventions in plain language, with the module that implements each.

## Cohort (`io.py`)

- **Who is in the cohort.** The split file lists the 1,591 participants (of 1,892 in the processed parquet) who passed preprocessing for the forecasting pipeline. That file is the cohort: no other exclusion is applied.
- **Strata.** The stratum is the AI-READI enrolment group. "Non-insulin T2D" is the group "oral medication and/or non-insulin injectable medication controlled", so it includes participants on injectable GLP-1 agonists (178 of 458 have a GLP-1 or GIP/GLP-1 flag).
- **Laboratory values.** A value dated after the participant's last CGM reading is set to missing.
- **TG/HDL** is the ratio of the two mg/dL values (about 2.3 times the mmol/L ratio). HDL must be above 0.
- **Medication flags** record whether a class was ever recorded, with no dose or timing. DPP-4 inhibitors are not recorded in the processed data.

## Clinical profiles (`profiles.py`)

- **Fitting.** Per stratum, on training participants in split-file order:
  - iterative imputation (20 rounds, seed 42), or complete cases for healthy;
  - log(1 + x) for factors with |skew| > 1 (HbA1c, C-peptide and TG/HDL in non-insulin T2D; BMI as well in healthy);
  - standardization;
  - k-means with 100 initializations and seed 42.
- **Numbering.** C1, C2, … are ordered by the training members' mean raw HbA1c.
- **Assignment.** Every participant, training ones included, gets the nearest frozen centroid.
- **Relative assignment margin.** (second-nearest distance minus nearest distance) / nearest distance. The sensitivity subset keeps margins above 0.20.
- **Frozen profiles.** The profiles were frozen on 2026-08-03. Refitting reproduces the frozen centroids exactly in all four strata, so they are rebuilt rather than stored.

## CGM outcomes (`cgm_metrics.py`)

- **Windows.** Both start at the participant's first row in the parquet, which is a complete 5-minute grid with missing glucose as empty rows.
  - **First 48 hours:** elapsed time 0 to 48 h, both bounds included.
  - **Full record:** every valid reading, across any interior gaps.
- **Metrics.**
  - Mean glucose.
  - **TIR:** the share of valid readings from 70 to 180 mg/dL inclusive.
  - **CV:** SD (ddof 1) / mean × 100. It needs at least 2 readings.

## CGM features and groups (`cgm_features.py`, `cgm_groups.py`)

- **Extended features.**
  - Mean glucose, TIR, CV, TBR < 70, TAR > 180, TAR > 250 and SD.
  - MAGE: moving-average method, 5- and 32-reading means, excursions > 1 SD.
  - Autocorrelation at 1 h and 3 h, plus 24 h over the full record.
  - The Lomb-Scargle share of power in the 20 to 28 h band, over the full record.
  - Up to 4 principal components of the 24-hour daily profile, built in each participant's own time zone and keeping 80% of the variance.
- **Minimal features.** Mean glucose, TIR and CV.
- **Runs.** MAGE and autocorrelations use runs that break at gaps over 30 minutes. Runs shorter than 6 h are dropped.
- **Coverage floor.** 70% of 576 readings in 48 h, and 1,000 readings over the full record.
- **CGM groups.**
  - Preprocessing is fitted on training participants: profile PCA, training-median imputation, log(1 + x) for |skew| > 1, standardization.
  - k-means (100 initializations), then nearest centroid for the other participants.
  - G1, G2, G3 are ordered by mean glucose.
  - The silhouette scan covers k = 2 to 6 and keeps the smallest k within 0.02 of the best.

## Statistics (`stats.py`, `agreement.py`)

- **Rank-based η².** (H − k + 1) / (n − k), from the tie-corrected Kruskal-Wallis H, clipped at 0.
- **Bootstrap intervals.** Percentile intervals over participant resamples:
  - **η²:** 2,000 resamples. In the forward direction the same resamples are shared by both windows.
  - **Pairwise mean differences and C2 versus rest:** 1,000 resamples within each group, seed 42 plus a running counter. This is the record convention, and Section 3.1 quotes these intervals.
  - **Other differences:** 2,000 resamples.
- **Agreement.**
  - ARI and NMI (arithmetic normalization).
  - Bias-corrected Cramér's V (Bergsma 2013).
  - Permutation nulls with 1,000 label permutations and a one-sided p of (1 + count) / (1 + 1,000).
- **Pairwise tests.** Dunn tests are tie-corrected, with Holm correction within each metric and window.

## Prediction (`prediction.py`)

- **Cross-validation.** Five folds repeated 20 times, stratified by profile, seed 42. Every CGM transform is refit inside each training fold.
- **Hyperparameters.** Inner 5-fold CV over 13 values from 10⁻³ to 10³ for the ridge alpha and the logistic C.
- **Logistic model.** Class-balanced.
- **Clinical to CGM.** The Section 3.2 linear model, with iterative imputation per fold (seed 42 + fold index). It reproduces the published Section 3.2 R² to within 4e-7.
- **Permutation nulls.** 200 permutations, or 1,000 when the primary p lies between 0.01 and 0.10, with the hyperparameter fixed at its most frequent value.

## Temporal windows (`windows.py`)

- **Windows.** Half-open intervals from the first row: [0, 72) h and [72, 144) h, so adjacent windows share no reading.
- **Eligibility.** A participant is eligible if both windows have at least 70% of their expected readings.
- **Frozen pipeline.** Fitted on window A training participants and applied unchanged to window B.
- **ICC.** ICC(A,1), two-way random effects with absolute agreement, on the clustering scale.

## Small cells (`disclosure.py`, `small_cell_policy.md`)

- **Masking.** Counts of 1 to 4 are shown as "<5". Further cells are shown as "NR" while any masked value could be worked out from the totals. Two integer programs per masked cell confirm that none can be recovered.
- **Rounding.** A statistic whose n is masked is rounded before it is tracked.
