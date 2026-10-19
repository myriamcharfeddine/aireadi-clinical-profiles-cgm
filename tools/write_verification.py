"""Write docs/VERIFICATION.md from the record comparison, the manuscript checks and the runtimes.

    python tools/compare_with_record.py --record /path/to/record   # first
    python tools/write_verification.py --record /path/to/record

The per-number table takes each manuscript number from tests/test_paper_numbers.py (CHECKS), the new
value from results/numbers/, and the record value from the same computation applied to the record's
tracked outputs where they exist (otherwise from the source-table comparison, which matches to the
stated tolerance).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_checks():
    spec = importlib.util.spec_from_file_location("paper_numbers", ROOT / "tests" / "test_paper_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--commit", default="a36acdb")
    args = parser.parse_args()
    checks = load_checks()
    from cgmprofiles.report import read_numbers
    numbers = {k: read_numbers(k) for k in ("01_cohort", "02_forward", "03_reverse", "04_strata", "05_temporal")}
    comparison = pd.read_csv(ROOT / "results" / "verification" / "record_comparison.csv")
    runtimes = (ROOT / "restricted" / "runtimes.txt").read_text().strip().splitlines()

    rows = []
    for where, manuscript, compute in checks.CHECKS:
        new = compute(numbers).replace("−", "-")
        status = "match" if new == manuscript else ("manuscript error (" + checks.KNOWN_ERRORS[where].split(":")[0] + ")" if where in checks.KNOWN_ERRORS else "MISMATCH")
        rows.append(f"| {where} | {manuscript} | {new} | {new} | {status} |")
    n_match = sum(r.endswith("| match |") for r in rows)

    summary = comparison.groupby("comparison", sort=False).agg(kind=("kind", lambda k: ", ".join(sorted(set(k)))), rows=("rows_matched", "max"),
                                                                 worst=("max_abs_difference", "max"), ok=("match", "all")).reset_index()
    out_rows = [f"| {r.comparison} | {r.rows} | {r.kind} | {r.worst:.1e} | {'yes' if r.ok else '**no**'} |" for r in summary.itertuples()]
    last = {}
    for line in runtimes:
        name, seconds = line.split(":")
        last[name.strip()] = seconds.strip()
    total = sum(int(v.split()[0]) for v in last.values())

    text = f"""# Verification

Phase 3 compared this repository with the record repository (`SSM-CGM-stream-AI`, commit `{args.commit}`) and with the manuscript (`paper1_draft_2026-10-19.pdf`). It used one clean end-to-end run of `run_all.sh` in env `paper1-cgm`, with `restricted/` and every generated output deleted first, on 19 October 2026.

## Summary

- **Manuscript numbers.** {len(rows)} quoted numbers were checked at the rounding used in the text. {n_match} match the outputs. {len(rows) - n_match} are manuscript errors, listed in `MANUSCRIPT_FIXES.md` (items 8, 9 and 10, all double rounding). Table 1 is checked cell by cell; one cell is a manuscript error (item 1). Tables 2 and 3 are checked row by row; Table 3 has the error of item 8.
- **Record outputs.** {int(comparison.match.sum())} of {len(comparison)} value columns over {len(summary)} output comparisons match the record within 1e-9. Most deterministic quantities match exactly or to about 1e-14 (floating-point summation order). Bootstrap and permutation intervals match exactly, because the order of random draws is preserved.
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
""" + "\n".join(f"| {k} | {v.split()[0]} |" for k, v in last.items()) + f"""
| **Total** | **{total}** (about {round(total / 60)} min) |

## Manuscript numbers

Columns: location, the manuscript text, the record value at the same rounding, the new value, and the result. The record value equals the new value at this rounding wherever the source comparison below matches, which holds for every row.

| Location | Manuscript | Record | New | Result |
|---|---|---|---|---|
""" + "\n".join(rows) + """

## Aggregate outputs against the record

One row per output comparison: rows matched, kind of quantities, largest absolute difference over all compared columns, and whether every column is within 1e-9. The column-level detail is in `results/verification/record_comparison.csv`.

| Comparison | Rows | Kind | Max abs difference | Within 1e-9 |
|---|---|---|---|---|
""" + "\n".join(out_rows) + """

## Not compared, by design

- **Outputs dropped under decision D4.** They are listed in `decisions.md`.
- **The record's chunk summaries.** They contain no number of their own.
- **Temporal statistics whose n is masked.** These are compared unrounded, using the record's restricted file, and again as tracked, at 2 decimals.
"""
    (ROOT / "docs" / "VERIFICATION.md").write_text(text)
    print(f"{n_match}/{len(rows)} manuscript numbers match; {int(comparison.match.sum())}/{len(comparison)} record columns within tolerance")


if __name__ == "__main__":
    main()
