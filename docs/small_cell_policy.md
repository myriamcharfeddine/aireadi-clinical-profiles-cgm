# Small-cell policy for descriptive tables

Written 2026-10-15 for the paper's baseline tables (`paper1_overleaf/table1_baseline.tex`, `table_s1_baseline_other_strata.tex`), produced by `01_raw_cgm_differences/tables/make_baseline_tables.py`.

## What the AI-READI licence says

The AI-READI Data License Agreement, version 2.0 (licensor Washington University in St. Louis, published 7 November 2025; [Zenodo record](https://zenodo.org/records/17555036); version 1.0: [Zenodo record](https://zenodo.org/records/10642459)), states **no numeric small-cell threshold**. The relevant clauses:

- **3E, Derivative Data.** Data that has been "modified, excerpted, encrypted, condensed, encoded, translated or otherwise altered, such that it contains Data or Data may be derived from it" is "considered to be Data subject to all restrictions described herein".
- **3F, Publications.** "Data may not be reproduced in papers, articles, presentations, analyses, reports or publications ('Papers') except that small representative samples of Data may be reproduced in up to five images or figures per Paper for illustrative purposes only."
- **4B, Additional use restrictions.** Licensee shall not use the Data "to attempt to identify or contact any Data Subject or group of Data Subjects" or "to extract or extrapolate any identifying information about a Data Subject".

The licence also refers to the NIH security best practices for controlled-access data. That document concerns data security and was not found to set a cell-size threshold. No threshold is stated in this repository or in the local dataset folders.

## Rule used

- **Masking.** Counts of 1 to 4 participants are masked and shown as `<5`; the percentage is masked with the count. True zeros are shown.
- **Race categories.** Any race category with a cell of 1 to 4 is combined into "Other, multiple or not reported"; other categories are kept as recorded. Remaining cells of 1 to 4 are masked.
- **Complementary masking.** While any masked value could be worked out from what is shown, one more cell is masked and shown as `NR`. "What is shown" means the other cells, the All column, the row and column totals, the n row, the "Missing, n" column and the percentages.
- **P values** are computed from the unmasked counts and are descriptive.
- **Unmasked versions** exist only under `restricted/` (gitignored) and are never tracked.

## Recovery check

For every masked cell, two integer programs (`scipy.optimize.milp`) find its smallest and largest possible value over all tables consistent with what is shown. The constraints are:

- the profiles sum to All in each row;
- for exhaustive categorical blocks, the categories sum to each column's total;
- visible cells are fixed;
- `<5` cells lie in 1 to 4;
- `NR` cells lie between 5 and the column denominator.

A masked value counts as recoverable if its minimum equals its maximum.

**Result:** 56 masked cells across Table 1 and Table S1 (27 `<5`, 29 `NR`); **0 recoverable**. The per-cell ranges are in `restricted/01_raw_cgm_differences/tables_draft/masking_recovery_check.csv`. A separate scan of the tracked CSVs found no visible count of 1 to 4.

## Comparison thresholds in other data agreements

- PEDSnet: 5; PCORnet: 11 ([Duke cell size suppression requirements](https://duke.atlassian.net/wiki/spaces/PD/pages/33588869/Cell+Size+Suppression+Requirements)).
- CMS: values of 1 to 10 may not be shown, including through percentages or derivations ([CMS cell suppression policy](https://www.hhs.gov/guidance/document/cms-cell-suppression-policy); [ResDAC summary](https://resdac.org/node/1506)).
- SEER-MHOS: masking reviewed before publication ([SEER-MHOS small cell sizes](https://healthcaredelivery.cancer.gov/seer-mhos/support/small_cell_sizes.html)).

## Changing the threshold

The threshold can be raised by changing one constant, `SMALL_MAX`, in `01_raw_cgm_differences/tables/make_baseline_tables.py`. The `<n` label, the footnote, the race combination rule and the recovery check all follow it. Rerun with `--write-tracked` afterwards.
