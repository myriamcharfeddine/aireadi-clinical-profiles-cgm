"""06 Figures and Tables 2 and 3, from saved outputs only (no parquet read, no refit).

Question: none; this script draws. Fig 2 combines analyses 03 (panels A to C) and 04 (panel D).
Inputs:   results/aggregate/*.csv (02 to 05); restricted/participant_metrics (Fig 1 points only).
Outputs:  results/figures/figure_1_*/, figure_2_*/, figure_3_*/ (one folder per manuscript figure: .pdf, .png,
          make_figure_N.py, source_data/); results/tables/table2_forward, table3_bidirectional (.csv, .tex).
Paper:    Fig 1, Fig 2, Fig 3, Table 2, Table 3.
"""
from __future__ import annotations

import pandas as pd

from cgmprofiles import figures, io, tables
from cgmprofiles.cgm_metrics import METRICS, WINDOWS
from cgmprofiles.config import get_config, results_dir
from cgmprofiles.report import Timer, read_aggregate

CFG = get_config()
LABELS = CFG.labels.variables
T2D_LONG = CFG.labels.strata_long["t2d_oral_non_insulin"]


def table2() -> pd.DataFrame:
    eta = read_aggregate("forward_eta_squared_ci").set_index(["window", "metric"])
    pairs = read_aggregate("forward_pairwise_differences")
    pairs = pairs[pairs.canonical_stratum == io.PRIMARY]
    rows = [{"Outcome": LABELS[m], "First 48 hours": tables.fmt_ci(*eta.loc[(WINDOWS[0], m), ["eta_squared", "ci_low", "ci_high"]]),
             "Full record": tables.fmt_ci(*eta.loc[(WINDOWS[1], m), ["eta_squared", "ci_low", "ci_high"]]),
             "Between-profile pattern": tables.between_profile_pattern(pairs, m, WINDOWS, {1: "C1", 2: "C2", 3: "C3"})} for m in METRICS]
    return pd.DataFrame(rows)


def table3() -> pd.DataFrame:
    fr = read_aggregate("forward_vs_reverse_eta_squared")
    forward = fr[fr.direction.str.startswith("forward") & (fr.window == WINDOWS[1])]
    reverse = fr[fr.direction.str.startswith("reverse")].sort_values("hba1c_flag", kind="mergesort")
    rows = [{"Direction": "Forward" if i == 0 else "", "Variable": LABELS[r.variable], "Eta squared [95% CI]": tables.fmt_ci(r.eta_squared, r.ci_low, r.ci_high)}
            for i, r in enumerate(forward.itertuples())]
    rows += [{"Direction": "Reverse" if i == 0 else "", "Variable": LABELS[r.variable], "Eta squared [95% CI]": tables.fmt_ci(r.eta_squared, r.ci_low, r.ci_high)}
             for i, r in enumerate(reverse.itertuples())]
    return pd.DataFrame(rows)


def main() -> None:
    timer = Timer()
    out = results_dir("tables")
    t2 = table2()
    t2.to_csv(out / "table2_forward.csv", index=False)
    (out / "table2_forward.tex").write_text(tables.simple_latex(t2, f"CGM differences across clinical profiles in {T2D_LONG} (n = 458). Rank-based "
        "$\\eta^2$ with 95\\% participant bootstrap confidence intervals (2,000 resamples). All omnibus comparisons had p $<$ 0.0001. Pairwise "
        "patterns were supported by Dunn tests with Holm correction.", "tab:forward", column_spec="lccp{4.2cm}"))
    t3 = table3()
    t3.to_csv(out / "table3_bidirectional.csv", index=False)
    (out / "table3_bidirectional.tex").write_text(tables.simple_latex(t3, f"Effect sizes in both directions in {T2D_LONG}. Forward comparisons assess "
        "full-record CGM outcomes across clinical profiles. Reverse comparisons assess clinical variables across CGM groups derived from extended "
        "full-record features (k = 3). Values are rank-based $\\eta^2$ with 95\\% participant bootstrap confidence intervals.", "tab:bidirectional"))

    for draw in figures.FIGURES.values():
        draw()
    print(f"06 done in {timer.seconds()} s")


if __name__ == "__main__":
    main()
