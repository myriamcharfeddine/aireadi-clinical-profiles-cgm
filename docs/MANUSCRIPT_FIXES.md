# Manuscript fixes

These are discrepancies between the 19 October 2026 draft (`paper1_draft_2026-10-19.pdf`) and the outputs of this repository, found in Phase 3. Each item gives the exact current text and a proposed correction. Every number in a proposed correction comes from `results/numbers/` and is checked by `tests/test_paper_numbers.py`. No result changes. The fixes concern rounding, wording, references, labels and one missing paragraph.

## 1. Table 1: C-peptide lower quartile in C1

- **Current:** "C-peptide, ng/mL | 2.3 [1.7–3.1]" (C1 column).
- **Proposed:** "2.3 [1.6–3.1]".
- **Why:** the lower quartile is 1.6475. The draft rounded it twice: first to 1.65 in the 2-decimal generated table, then by hand to 1.7. Rounded once, it is 1.6. Every other Table 1 cell matches. `results/tables/table1_baseline.tex` holds the corrected table.

## 2. Methods 2.3: number of bootstrap resamples

- **Current:** "We reported rank-based η² with 95% confidence intervals from 2,000 participant bootstrap resamples."
- **Proposed:** "We reported rank-based η² with 95% confidence intervals from 2,000 participant bootstrap resamples. Pairwise mean differences between profiles, including C2 against the other profiles in type 2 diabetes, used 1,000 resamples drawn within each profile."
- **Why:** the η² intervals (Table 2, Table 3, Section 3.3) use 2,000 resamples. The pairwise CV differences quoted in Section 3.1 (−0.76 [−2.06, 0.63], −1.37 [−2.49, −0.21], 2.26 [0.99, 3.56]) use 1,000, and so does the type 2 diabetes C2-minus-rest reference (−2.39 [−3.41, −1.43]) in Section 3.3 and Fig 3C. The healthy and prediabetes specificity intervals use 2,000. Rerunning with 2,000 would change the intervals in the second decimal, so the text should describe what was done rather than the numbers change.

## 3. Table 1 footnote: unresolved reference

- **Current:** "Site, blood pressure, LDL cholesterol, other medications, CGM coverage and the data split are in Supplementary Table ??."
- **Proposed:** "… are in Supplementary Table S1." (`results/tables/tableS1_baseline_full.tex`; Supplementary Table S2 holds the other strata).

## 4. Methods 2.3: incomplete list of CGM features

- **Current:** "We clustered CGM features comprising glucose summaries, time below 70 mg/dL, time above 180 and 250 mg/dL, mean amplitude of glycemic excursions, daily-profile components and autocorrelation summaries."
- **Proposed:** "We clustered CGM features comprising mean glucose, time in range, coefficient of variation and standard deviation, time below 70 mg/dL, time above 180 and 250 mg/dL, mean amplitude of glycemic excursions, up to four daily-profile principal components, autocorrelation at 1 and 3 hours and, over the full record, at 24 hours, and the share of Lomb-Scargle power at periods of 20 to 28 hours."
- **Why:** the primary partition (16 features) also contains the 24-hour autocorrelation and the circadian power share, which the current list does not mention.

## 5. Figures 2 and 3: stratum label "T2D oral"

- **Current:**
  - Fig 2 title: "Reverse direction: grouping by CGM first, T2D oral non-insulin (n = 458)".
  - Fig 3 legend and panel C: "T2D oral", "T2D oral (own profiles)".
- **Proposed:** replace with the regenerated `results/figures/fig2_reverse.pdf` and `fig3_strata.pdf`, which use "Non-insulin T2D" and "Insulin-treated T2D", and the new stratum colours (`decisions.md`, visual changes).
- **Note:** Fig 1's title now ends "in type 2 diabetes treated without insulin". The Fig 2 caption needs no change.

## 6. Results: the temporal paragraph is missing

The Abstract (ARI 0.42, 0.030 and 0.036), Methods 2.3 and the Discussion use the temporal analysis, but no Results paragraph reports it.

- **Proposed:** add at the end of Section 3.2.

> Across two non-overlapping 3-day windows (days 1 to 3 and days 4 to 6), CGM group assignments were moderately reproducible in held-out participants (ARI 0.42, 95% CI [0.30, 0.58]; kappa 0.63 [0.52, 0.74]). Agreement between CGM groups and the clinical profiles remained low in each window (ARI 0.030 [0.013, 0.057] and 0.036 [0.017, 0.064]). Glucose level was more reproducible between windows than timing features. C2 had a lower coefficient of variation than the other profiles in both windows (−2.33 [−3.47, −1.21] and −2.31 [−3.47, −1.16] percentage points), close to the full-record difference in the same participants (−2.38 [−3.39, −1.30]).

About 98% of participants were eligible. The exact count is withheld under the small-cell rule, and the held-out statistics are rounded to two decimals for the same reason.

## 7. Abstract: "approximately 40%"

- **Current:** "the C2-associated reduction in variability was approximately 40% as large in prediabetes".
- **Proposed:** "the C2-associated reduction in variability was 37% to 44% as large in prediabetes" (37% at 48 hours, 44% over the full record).

## 8. Table 3: waist-to-hip ratio interval

- **Current:** "Waist-to-hip ratio | 0.028 [0.005, 0.069]".
- **Proposed:** "0.028 [0.004, 0.069]".
- **Why:** the lower bound is 0.004454. The draft took 0.0045 from a 4-decimal summary and rounded it up.

## 9. Section 3.3: healthy mean-glucose interval

- **Current:** "with full-record η² of 0.021 (95% CI [0.003, 0.054])".
- **Proposed:** "0.021 (95% CI [0.003, 0.053])".
- **Why:** the upper bound is 0.053478. This is the same double rounding: 0.0535, then 0.054.

## 10. Section 3.3: pooled agreement inside type 2 diabetes

- **Current:** "yielded similarly low agreement with C1–C3 as clustering that stratum separately (ARI 0.037 versus 0.039)".
- **Proposed:** "(ARI 0.036 versus 0.039)".
- **Why:** the pooled k = 3 ARI with C1 to C3 is 0.036452. The record's `RESULTS.md` rounded 0.0365 up to 0.037. This affects the Fig 2D text only; the plotted value is unchanged.

Items 1, 8, 9 and 10 have the same cause: the draft rounded values that had already been rounded. `tests/test_paper_numbers.py` marks each of them as an expected failure that names its item here, so the tests pass while the errors stay visible. When the manuscript is corrected, remove each entry from `KNOWN_ERRORS` (or `KNOWN_TABLE1_ERRORS`) and update the expected text.

## Checked and correct

Every other number in the Abstract, Results, Tables 1 to 3 and the figure captions matches the outputs at the rounding used in the text. `tests/test_paper_numbers.py` checks them all.
