# Temporal reproducibility across non-overlapping CGM windows: specification (not run)

Status: design written 2026-10-15 on branch `paper1-temporal` (from `main` at 747afad). Approved 2026-10-15 with the decisions recorded at the end of this file. Runs in chunks, with a stop after each.

## Purpose and scope

Section 3.1 compares the first 48 h with the full record. The two windows share observations: the first 48 h is part of the full record. Agreement between them is therefore not an independent check. This analysis uses two windows that share no readings and asks whether, inside the same monitoring period, the CGM groups, the clinical-CGM agreement and the C2 variability contrast come out the same in both.

**Scope statement (repeated in every output).** This tests reproducibility within the monitoring period of about 10 days, not long-term phenotype stability. Both windows come from the same sensor wear, with the same clinical measurements and the same medication status.

No model hidden state is used. SSM-CGM-Stream (the forecasting architecture designed by Shakson Isaac, Isaac, Collin & Patel, arXiv:2510.04386) plays no part.

## Rules

- New files only, under `01_raw_cgm_differences/temporal/`. Participant-level outputs (features, labels, transition rows, fitted transforms) go to `restricted/01_raw_cgm_differences/temporal/`. Tracked files hold aggregates only.
- Small-cell rule (`docs/SMALL_CELL_POLICY.md`): any count from 1 to 4 in a tracked table or figure is shown as "<5", with complementary masking where a margin would reveal it. Statistics are computed from the unmasked data in `restricted/`.
- Named constants and fixed seeds throughout. Participant bootstrap CIs (labels and features travel with the participant). No causal language. No em dashes. Figures follow the style guide (sentence case, `figure_main_config` colours; CGM groups in a neutral ordered palette, never the C1/C2/C3 colours).
- CPU only. Ask before any step projected to exceed 10 minutes.
- `docs/PAPER_TEXT_TODO.md` is not touched by this analysis (decision 2026-10-15, second set, item 4); paper text goes to `temporal/PAPER_TEXT_PROPOSALS.md`. `main` is not pushed.
- Code imports `audit/recompute_31.py` and the reverse-direction modules (`features.py`, `clustering.py`, `reverse_tests.py`). It imports no Section 3.1 or `within_subtype_*` module.

## Fixed inputs

| Item | Value | Source |
|---|---|---|
| Participants | the 458 of `t2d_oral_non_insulin` (train 325, val 69, test 64), C1/C2/C3 = 192/111/155 | `recompute_31.assign_labels` |
| Split | `experiment_c_split_adapt6h_seed42` | `recompute_31.load_split` |
| CGM | `cgm_glucose_mean`, 5-minute grid, NaN for missing | `recompute_31.load_cgm` |
| Time origin | `elapsed_h` from the participant's first row, as in Section 3.1 | same |
| Clock time | converted per participant to the `timezone` column (UAB is America/Chicago, stored as America/Los_Angeles) | `features.load_timezones` |
| Clinical variables | the six factors after the post-CGM laboratory rule | `recompute_31.load_clinical_frame` |
| Sex, site, GLP-1/GIP flag | `demo_sex_at_birth`, `participants_clinical_site`, `med_glp1_or_gip_glp1` | as in `strata/split_composition.py` and `reverse_tests.MEDICATION_COLUMNS` |
| Seeds | 42 for clustering; separate named seeds for each bootstrap and permutation stream | |

Record length is already known from `audit/PIPELINE_CHECKS_31.json` (aggregate): in the 458, record hours have minimum 109, 25th percentile 203, median 225, maximum 259. So about three quarters of the records reach 192 h, before coverage is applied. The 60% stop rule below is therefore a real risk, not a formality.

**Gate 0.** Recompute the 458 labels and the Section 3.1 metrics with `recompute_31.py`: n = 458, 0 label mismatches, metric difference ≤ 1e-9 in both Section 3.1 windows. A failure stops the run.

## T0. Windows and eligibility

### Window definitions

All windows are half-open in elapsed time, so adjacent windows share no reading.

| Design | Window A | Window B | Length each | Role |
|---|---|---|---|---|
| **Days 1 to 3 vs days 4 to 6 (`days3`)** | 0 ≤ t < 72 h | 72 ≤ t < 144 h | 72 h | **primary** (from 2026-10-15, second decision set) |
| Days 1 to 4 vs days 5 to 8 (`days4`) | 0 ≤ t < 96 h | 96 ≤ t < 192 h | 96 h | sensitivity (primary in the first version of this file) |
| Halves | 0 ≤ t < T/2 | T/2 ≤ t ≤ T, T = the participant's last row | varies by participant (54 to 129 h) | sensitivity |
| No warm-up, 3-day (`no_warmup_days3`) | 24 ≤ t < 72 h | 72 ≤ t < 144 h | A 48 h, B 72 h | sensitivity |
| No warm-up, 4-day (`no_warmup_days4`) | 24 ≤ t < 96 h | 96 ≤ t < 192 h | A 72 h, B 96 h | sensitivity |
| 3-day windows on the 4-day-eligible set (`days3_on_days4_set`) | as `days3` | as `days3` | 72 h | sensitivity: participants eligible in both `days3` and `days4` |

Note: the Section 3.1 48 h window is inclusive (≤ 48 h, 577 slots). Here the boundaries are half-open, so window A has exactly 1,152 slots.

### Eligibility

- `READINGS_PER_HOUR = 12`; expected readings = window length in hours × 12 (A and B primary: 1,152 each).
- `MIN_WINDOW_COVERAGE = 0.70` of expected readings (the B1 coverage floor), applied **separately** to A and to B. A participant is eligible for a design only if both windows pass.
- In the halves design, expected readings use each half's own length.
- Participants whose record ends before the end of window B fail B automatically (their coverage is counted against the full expected length).

### Reported before any analysis (chunk 1 stops here)

1. **Counts qualifying**, for each design: overall, by profile (C1/C2/C3), by split, and by site. Also the number failing A only, B only, and both.
2. **Representativeness**: standardized mean differences (SMD) between the qualifying subset and the primary 458, for age, sex, BMI, HbA1c, C-peptide, TG/HDL, waist-to-hip, GLP-1/GIP flag, site, and full-record mean glucose, TIR and CV.
   - Continuous: difference in means over the pooled SD. Binary: difference in proportions over the pooled binomial SD. Site (three levels): one SMD per level against the rest, plus the multivariate SMD of Yang and Dalton (2012).
   - The qualifying subset is part of the 458, so this comparison understates differences. The SMD between qualifying and non-qualifying participants is reported beside it.
   - `SMD_FLAG = 0.10` is flagged in the table (descriptive; no test).
3. **Weekday versus weekend**: share of valid readings that fall on Saturday or Sunday (local clock time, per-participant timezone), in A and B, as median and IQR across participants and as the pooled share. Expected share under uniform wear: 2/7.

### Primary stratum and secondary strata

- Primary: T2D oral non-insulin, frozen C1/C2/C3.
- Secondary, only if runtime allows (chunk 4): healthy (k = 2, complete case, 538) and prediabetes (k = 2, 405), each with its own frozen k = 2 profiles. CGM groups use k = 2 for these strata, matching the strata analysis primary (`../strata/SPEC.md`). T4 uses the single contrast C1 minus C2 in place of the C2 contrasts. All secondary results are reported separately and do not enter the T2D readings.

## T1. CGM features and frozen pipeline

### Features

The extended feature set of Part B (`reverse_direction/features.py`), computed separately in window A and window B with the same definitions and gap rules:

- mean glucose, TIR 70 to 180, CV, TBR < 70, TAR > 180, TAR > 250, SD;
- MAGE (moving-average version, per contiguous run; runs split at gaps over 30 min, runs under 6 h dropped);
- autocorrelation at 1 h, 3 h and 24 h (within runs);
- circadian share of Lomb-Scargle power (20 to 28 h over 2 to 28 h);
- the 24-value mean daily profile, centred on the participant's window mean, for PCA.

The 24 h autocorrelation and circadian share are included because each primary window holds 4 days. In the short design (3 days) they are also included. In the halves design, half lengths go down to 54 h, so these two are dropped there (decision 3). Window subsetting is done before feature computation, so runs, MAGE and autocorrelations never cross a window boundary.

`features.participant_features` takes a window name and applies the 48 h coverage rule internally. The new module wraps it with explicit start and end hours and its own coverage rule, rather than editing it.

### Frozen pipeline, fit on window A

Fit on the **training participants eligible in the design** (both windows pass), using their window A features only:

1. daily-profile PCA (80% cumulative variance target, at most 4 components);
2. log1p for any feature with |skew| > 1 on those participants;
3. `StandardScaler`;
4. k-means, k = 3, `n_init = 100`, `random_state = 42`.

Then every eligible participant (all splits) is assigned to the nearest frozen centroid **in both windows**, using the same PCA, log1p choice and scaler. Group labels are aligned by construction. Display labels G1 to G3 are ordered by the window A training-member mean glucose.

Fitted objects are saved with joblib in `restricted/`.

**Stop rule.** If any CGM group has fewer than `MIN_CLUSTER_N = 20` training participants in window A, stop and report.

### Sensitivity: refit on window B

Refit the whole pipeline (PCA, log1p choice, scaler, k-means) on the window B training participants. Match its groups to the window A groups by Hungarian assignment on the centroids, with centroid distance computed in the window A standardized space (B centroids are mapped back to raw feature medians, then through the A transforms). Report the matching cost and the ARI between the two fitted partitions on the same participants.

Sensitivity designs (short, halves, no warm-up) repeat the frozen-pipeline fit on their own window A.

## T2. Reproducibility of CGM groups (A versus B)

**Participant sets.** Primary: held-out participants (validation and test) eligible in the design. Their window A labels are out of sample, as are their window B labels. Secondary: all eligible participants. For training participants, the window A labels come from the fitting data, so agreement in this set is expected to be higher.

### Group-level agreement

- Percent agreement, Cohen's kappa (unweighted), ARI, NMI (arithmetic normalisation).
- The 3 × 3 transition table (A rows, B columns), counts and row percentages, masked under the small-cell rule in tracked outputs.
- Bootstrap: `N_BOOTSTRAP = 2000` participant resamples, both labels fixed per participant, percentile 95% CI.
- Permutation null: `N_PERMUTATIONS = 1000` shuffles of the window B labels across participants (window A labels fixed). This breaks the pairing while keeping group sizes. p = (1 + count of null values ≥ observed) / 1,001.

### Feature-level reproducibility

- ICC(2,1): two-way random effects, absolute agreement, single measurement (McGraw and Wong 1996, ICC(A,1)), between A and B, for each scalar feature and each profile PC score (scores from the window A PCA).
- Computed on the scale used for clustering (after the log1p choice, before standardization). Raw-scale ICC is reported beside it.
- CI: participant bootstrap, 2,000 draws, percentile. The analytic F-based CI is reported as a check.
- Sets: all eligible participants (primary for ICC, since features do not depend on any fit) and held-out only.

### Benchmarks shown side by side (descriptive, not tested against each other)

| Benchmark | Value | Source | Caveat |
|---|---|---|---|
| A versus B ARI, held-out | this analysis | T2 | |
| Bootstrap refit stability of the primary Part B partition | 0.80 [0.46, 0.94] | `reverse_direction/outputs/B2_partition_diagnostics.csv` | full record, 458 participants, train resamples |
| ARI of CGM groups with C1/C2/C3 | 0.039 [0.020, 0.068] | `reverse_direction/outputs/B4_agreement.csv` | full record, 458 participants |
| Bootstrap refit stability in window A (new) | this analysis, same 200-resample procedure | T2 | added so the stability benchmark uses the same window length and participants |

## T3. Clinical-CGM agreement in each window

In window A and in window B separately, on the same eligible participants (all splits, as in Part B4):

- ARI, NMI and bias-corrected Cramér's V between CGM groups and C1/C2/C3.
  - Null: 1,000 permutations of the C1/C2/C3 labels, CGM labels fixed, per window.
  - CI: 2,000 participant bootstrap draws.
- Kruskal-Wallis eta squared of each of the six clinical variables across G1/G2/G3, (H − k + 1)/(n − k) clipped at 0, with a 2,000-draw participant bootstrap CI. HbA1c is shown in its own row and flagged: groups formed on glucose are expected to differ in HbA1c.
- **Paired bootstrap of the difference between windows.** In each draw the same participants are resampled for both windows, and the statistic in A minus the statistic in B is recorded. Applied to ARI, NMI, Cramér's V and each eta squared. 2,000 draws, percentile CI.

## T4. Forward direction and the C2 variability contrast in each window

In window A and in window B separately, on the same eligible participants:

- Kruskal-Wallis eta squared of mean glucose, TIR and CV across C1/C2/C3, with a 2,000-draw participant bootstrap CI.
- CV contrasts (mean difference, percentage points): C2 minus C1, C2 minus C3, and C2 minus rest. Bootstrap CI: 2,000 draws resampling within group (the Section 3.1 convention), 95% percentile.
- Dunn tests (tie-corrected) for the three pairs, Holm correction over the three pairs, per window.
- **Paired participant bootstrap of the change** in each contrast (A minus B): participants resampled once per draw, stratified by profile so group sizes stay fixed, same participants in both windows. 2,000 draws.
- **Reference.** The full-record C2 minus rest value for the same eligible participants (recomputed here), and the published 458 value −2.39 [−3.41, −1.43].
- **GLP-1/GIP stratification (descriptive only).** C2 minus rest CV, in each window, within flag = 1 and within flag = 0. Point estimates with bootstrap CIs; no tests and no reading. Group counts 1 to 4 are masked, and a contrast is not shown when either group has fewer than 5 participants.

## Pre-registered reading

Written before any result exists. Applied to the T2D primary design (days 1 to 3 versus days 4 to 6, `days3`; changed from days 1 to 4 versus days 5 to 8 on 2026-10-15 before any chunk 2 result, see Decisions), held-out participants for T2, all eligible for T3 and T4. Sensitivity designs are reported beside the primary and do not change the reading.

| Reading | Rule |
|---|---|
| **CGM groups reproducible** | held-out A versus B ARI ≥ 0.50 and kappa ≥ 0.60 |
| **CGM groups moderate** | held-out A versus B ARI from 0.30 to below 0.50 |
| **CGM groups unstable** | held-out A versus B ARI < 0.30 |
| **Distinct groupings** | CGM groups reproducible or moderate, and ARI with C1/C2/C3 < 0.10 in both windows. CGM structure would then be stable within the monitoring period but would not match the profiles. |
| **Temporal instability as an explanation** | CGM groups unstable (ARI < 0.30) |
| **C2 contrast reproducible** | C2 minus rest CV has the same sign in both windows and its CI excludes 0 in both |
| **C2 contrast not reproducible** | the CI includes 0 in either window, or the signs differ |
| **Clinical-CGM agreement consistent** | ARI with C1/C2/C3 < 0.10 in both windows, and the paired difference CI includes 0 |

Thresholds use point estimates; CIs are reported beside them. Named constants: `ARI_REPRODUCIBLE = 0.50`, `KAPPA_REPRODUCIBLE = 0.60`, `ARI_MODERATE = 0.30`, `ARI_PROFILE_AGREEMENT = 0.10`.

**Gap in the rules (decided 2026-10-15).** ARI ≥ 0.50 with kappa < 0.60 is read as moderate.

## Stop conditions

Stop, report and wait if:

- Gate 0 fails;
- fewer than 60% of the 458 (fewer than 275) are eligible in both primary windows (checked in chunk 1: about 98% for `days3`, 84.9% for `days4`);
- any CGM group has fewer than 20 training participants in window A (primary design `days3`);
- any step is projected to run longer than 10 minutes.

## Planned outputs (after approval)

| File | Contents |
|---|---|
| `temporal/outputs/T0_eligibility.csv` | counts by design, profile, split, site; masked |
| `temporal/outputs/T0_representativeness.csv` | SMDs, qualifying vs 458 and vs non-qualifying |
| `temporal/outputs/T0_weekday_weekend.csv` | weekend share in A and B |
| `temporal/outputs/T1_pipeline.csv` | PCA variance, log1p choices, group sizes by split, refit-on-B matching |
| `temporal/outputs/T2_agreement.csv` | agreement measures, CIs, nulls, benchmarks; all designs |
| `temporal/outputs/T2_transition.csv` | 3 × 3 tables, masked |
| `temporal/outputs/T2_icc.csv` | ICC per feature, both scales, both sets |
| `temporal/outputs/T3_clinical_agreement.csv` | per window, plus paired differences |
| `temporal/outputs/T4_forward_contrast.csv` | per window, plus paired changes, GLP-1/GIP strata |
| `temporal/outputs/CHUNK*_SUMMARY.md` | one per chunk |
| `restricted/01_raw_cgm_differences/temporal/` | features per window, labels, fitted transforms |

`temporal/TEMPORAL_RESULTS.md`, in order:

1. a plain-language summary first, opening with the scope statement;
2. the eligibility table;
3. the T2 to T4 tables;
4. the reading against each criterion, met or not, with numbers;
5. proposed paper text for Methods and Results, written to `temporal/PAPER_TEXT_PROPOSALS.md` (not to `docs/PAPER_TEXT_TODO.md`).

One figure, `temporal/outputs/figures/temporal_summary.{png,pdf}`, four panels:

- (A) the A versus B transition heatmap (held-out, row percentages, masked cells shown as "<5"), with ARI and kappa;
- (B) ICC per CGM feature with CIs;
- (C) ARI with C1/C2/C3 in window A and in window B, each against its permutation null;
- (D) the C2 minus rest CV contrast in windows A and B with CIs, with the full-record value on the same participants as a reference line.

The caption repeats the scope statement.

## Runtime estimate (CPU, 12 cores, from the Part B and strata timings)

| Part | Work | Estimate |
|---|---|---|
| Loading and Gate 0 | parquet, labels, Section 3.1 metrics | about 30 s |
| T0 | eligibility for 4 designs, SMDs, weekday shares | under 30 s |
| T1 features | extended features for 458 participants × 8 windows (4 designs × A and B) | 1.5 to 3 min |
| T1 pipeline | frozen fit per design, refit on B, Hungarian matching, 200-resample stability in A | about 1 min |
| T2 | 2,000 bootstraps and 1,000 permutations × 2 sets × 4 designs; ICC bootstraps for about 18 features | about 1 min |
| T3 | permutations and bootstraps per window, paired bootstraps, 6 eta squared | about 1 min |
| T4 | Kruskal-Wallis, Dunn, contrast bootstraps, paired bootstraps, GLP-1 strata | under 1 min |
| Secondary strata | features for 943 participants × 2 windows, then T1 to T4 at k = 2 | 3 to 5 min |
| Figure and results file | | under 1 min |

### Chunk plan (each under 10 minutes; stop and summary after each)

1. **Chunk 1: Gate 0 and T0** (about 1 min). Eligibility, representativeness, weekday share. Check the 60% stop rule. Stop.
2. **Chunk 2: T1 and T2** for all four designs (4 to 6 min). Check the group-size stop rule. Stop.
3. **Chunk 3: T3 and T4** (about 2 min). Stop.
4. **Chunk 4 (optional): secondary strata** (3 to 5 min). Only if approved after chunk 3.
5. **Chunk 5: TEMPORAL_RESULTS.md, figure, PAPER_TEXT_PROPOSALS.md** (about 1 min).

Each chunk is committed on `paper1-temporal`, without Claude co-author trailers.

## Questions for approval (answered, see Decisions)

1. **Gap in the reading.** ARI ≥ 0.50 with kappa < 0.60: read as moderate (proposed)?
2. **Fitting set.** Fit the window A pipeline on training participants eligible in both windows (proposed, so A and B are compared on one fixed set), or on all training participants eligible in A?
3. **Halves design.** Drop the 24 h autocorrelation and circadian share there because halves can be as short as 54 h (proposed), or keep them?
4. **CI with Holm.** The strata analysis required a Holm-adjusted p < 0.05 alongside each CI criterion. This design does not, as written in the request. Keep it without Holm (proposed), or add it to the C2 contrast rule?
5. **Secondary strata.** CGM groups at k = 2 for healthy and prediabetes, with C1 minus C2 as the T4 contrast (proposed)?

## Decisions (2026-10-15)

1. **Gap in the reading.** Held-out ARI ≥ 0.50 with kappa < 0.60 is read as **moderate**.
2. **Fitting set.** The window A pipeline is fit on the training participants eligible in both windows of the design. The number of such training participants is reported for each design.
3. **Halves design.** The 24 h autocorrelation and the circadian share are dropped there.
4. **No Holm correction for the C2 contrast rule.** The rule requires the CI to exclude 0 in both windows. That is an intersection-union test: the claim is accepted only if each of the two component tests rejects at its own level, and the overall type I error is then at most the level of each component, without adjustment (Berger 1982). The rule is therefore already conservative, and Holm would make it more so with no gain in error control. Unadjusted p values (two-group Kruskal-Wallis for C2 versus rest, and Dunn for the pairs) are reported alongside each CI.
5. **Secondary strata (healthy, prediabetes).** CGM groups at k = 2. The T4 variability contrast is C1 minus C2 on each stratum's own profiles. Secondary contrast: C2-assigned minus rest, with participants assigned to the frozen T2D centroids through the frozen T2D pipeline (as in the strata S3 specificity check), so that it matches the specificity check in the paper.

**Eligibility fallback (pre-approved).** If fewer than 60% of the 458 (fewer than 275) qualify for days 1 to 4 versus days 5 to 8, chunk 1 is also reported in full for days 1 to 3 versus days 4 to 6, and the run stops so the primary design can be chosen before chunk 2. Chunk 1 computes eligibility for all four designs in any case.

**Files outside this analysis.** The uncommitted edit to `01_raw_cgm_differences/README.md` and the untracked Figure 1 and Figure 2 files carried over from `main` are left untouched and out of every commit on this branch.

## Decisions (2026-10-15, second set, before chunk 2)

Taken after chunk 1 (eligibility, representativeness, weekday share) and before any CGM feature, group or outcome was computed.

1. **Primary design: days 1 to 3 versus days 4 to 6 (`days3`).** Days 1 to 4 versus days 5 to 8 (`days4`) becomes a sensitivity design. The choice rests only on eligibility and calendar balance; no outcome had been seen:
   - about 98% eligible against 84.9% (the exact 3-day figure is masked, decision 2);
   - the 4-day design excludes participants selectively (SMD qualifying vs non-qualifying: GLP-1/GIP −0.355, full-record CV 0.302, site multivariate 0.515), which bears directly on the C2 contrast;
   - the weekend share of valid readings is balanced in the 3-day design (33.8% in A vs 31.7% in B) but not in the 4-day design (34.4% vs 20.6%).

   Caveat: 3-day windows give noisier per-participant features than 4-day windows. That biases A-versus-B agreement downward, so the primary reading is conservative. The reading rules now name `days3` as primary; the thresholds are unchanged.
2. **Held-out n in `days3`.** Fewer than 5 held-out participants are excluded in this design. The held-out n, and any count that would let that number be derived (overall eligible n given the shown training n, the n of the all-eligible analysis set, transition-table totals), is masked. Masking uses one integer-program check across the linked tables (`small_cells.py`). The training (fitting-set) n stays shown, as decision 2 of the first set asks.
3. **No-warm-up sensitivity** follows the primary: hours 24 to 72 versus days 4 to 6 (`no_warmup_days3`). The 4-day version (`no_warmup_days4`) is kept beside it.
4. **Added sensitivity (`days3_on_days4_set`).** The 3-day windows on participants eligible in both the 3-day and the 4-day design, with its own frozen pipeline fit on its training participants. The 3-day and 4-day results can then be compared on the same people. Its representativeness check is reported in chunk 1 (rerun).
5. **Concurrent edits elsewhere.** This branch does not touch, stage or commit `paper1_overleaf/`, `docs/PAPER_TEXT_TODO.md` or the figure scripts. Before each commit, `git status` is checked so that only `temporal/` files are staged (plus the separate scan-tool fix, committed on its own).

Chunk 1 is rerun with these design names and the linked masking. Its numbers do not change.

## Decisions (2026-10-16, third set, after chunk 3)

1. **Chunk 4 (healthy and prediabetes): not run.** The prediabetes contrast (−1.04 full record) is too small to be informative in 3-day windows.
2. **Branch history.** The first chunk 1 commit contained unmasked 3-day counts. It was never pushed, and the branch history was rewritten so that no commit on `paper1-temporal` contains them. The final files are unchanged.
3. **Chunk 5.** Paper text goes to `temporal/PAPER_TEXT_PROPOSALS.md`. Nothing is written to `docs/PAPER_TEXT_TODO.md` or `paper1_overleaf/`.
