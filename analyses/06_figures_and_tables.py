"""06 Figures and Tables 2 and 3, from saved outputs only (no parquet read, no refit).

Question: none; this script draws. Fig 2 combines analyses 03 (panels A to C) and 04 (panel D).
Inputs:   results/aggregate/*.csv (02 to 05); restricted/participant_metrics (Fig 1 points only).
Outputs:  results/figures/fig1_forward, fig2_reverse, fig3_strata, figS1_temporal (.pdf, .png, _plotted.json);
          results/tables/table2_forward, table3_bidirectional (.csv, .tex).
Paper:    Fig 1, Fig 2, Fig 3, Supplementary Fig S1, Table 2, Table 3.
"""
from __future__ import annotations

import json

import pandas as pd

from cgmprofiles import io, tables
from cgmprofiles.cgm_metrics import METRICS, WINDOWS
from cgmprofiles.config import get_config, results_dir
from cgmprofiles.plotting import fig_forward, fig_reverse, fig_strata, fig_temporal, style
from cgmprofiles.report import Timer, jsonable, read_aggregate, read_restricted

CFG = get_config()
LABELS = CFG.labels.variables
T2D_LONG = CFG.labels.strata_long["t2d_oral_non_insulin"]


def save_figure(fig, plotted: dict, stem: str) -> None:
    out = results_dir("figures")
    style.save(fig, out, stem)
    (out / f"{stem}_plotted.json").write_text(json.dumps(jsonable({str(k): v for k, v in plotted.items()}), indent=1) + "\n")


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

    metrics = read_restricted("participant_metrics")
    kw = read_aggregate("forward_kruskal_wallis")
    fig, plotted = fig_forward.draw(metrics[metrics.canonical_stratum == io.PRIMARY], kw[kw.canonical_stratum == io.PRIMARY],
                                    f"Raw CGM behavior across clinical profiles in {T2D_LONG}")
    save_figure(fig, plotted, "fig1_forward")

    agreement = read_aggregate("reverse_agreement").set_index(["partition", "metric"])
    contingency = read_aggregate("reverse_primary_contingency").set_index("cgm_cluster")[["C1", "C2", "C3"]].astype(int)
    fig, plotted = fig_reverse.draw(contingency, agreement.loc[("kmeans3_extended_window_full", "ARI")], read_aggregate("forward_vs_reverse_eta_squared"),
                                    read_aggregate("reverse_prediction").fillna({"window": ""}), read_aggregate("reverse_prediction_null_draws"),
                                    read_aggregate("pooled_agreement"), CFG.criteria, int(contingency.to_numpy().sum()))
    save_figure(fig, plotted, "fig2_reverse")

    s1 = read_aggregate("strata_forward")
    eta = read_aggregate("forward_eta_squared_ci")
    forward = pd.concat([s1[s1.window == WINDOWS[1]][["canonical_stratum", "metric", "eta_squared", "ci_low", "ci_high"]],
                         eta[eta.window == WINDOWS[1]].assign(canonical_stratum=io.PRIMARY)[["canonical_stratum", "metric", "eta_squared", "ci_low", "ci_high"]]])
    s2 = read_aggregate("strata_clinical_differences")
    s2 = s2[s2.window == WINDOWS[1]].rename(columns={"stratum": "canonical_stratum"})
    fr = read_aggregate("forward_vs_reverse_eta_squared")
    fr = fr[fr.direction.str.startswith("reverse")].assign(canonical_stratum=io.PRIMARY).rename(columns={"variable": "factor"})
    cols = ["canonical_stratum", "factor", "eta_squared", "ci_low", "ci_high"]
    spec = read_aggregate("strata_specificity")
    spec = spec[(spec.variant == "primary (T2D scaler, all)") & (spec.metric == "coefficient_of_variation_pct")]
    fig, plotted = fig_strata.draw(forward, pd.concat([s2[cols], fr[cols]]), spec, read_aggregate("forward_c2_vs_rest"), CFG.criteria.medium_eta)
    save_figure(fig, plotted, "fig3_strata")

    primary, days4 = CFG.temporal.primary, "days4"
    t2 = read_aggregate("temporal_agreement")
    t2 = t2[t2.design == primary].set_index("metric")
    t3t = read_aggregate("temporal_profile_agreement")
    icc = read_aggregate("temporal_icc")
    icc_p = icc[icc.design == primary].set_index("feature")
    icc_p = icc_p.assign(icc_days4=icc[icc.design == days4].set_index("feature").icc.reindex(icc_p.index))
    stability = read_aggregate("reverse_primary_refit_stability").set_index("statistic").value["mean"]
    fig, plotted = fig_temporal.draw({"ARI": t2.loc["ARI"], "kappa": t2.loc["kappa"]}, t3t[(t3t.design == primary) & (t3t.statistic == "ARI")].iloc[0], icc_p,
                                     read_aggregate("temporal_c2_contrast").query("design == @primary").iloc[0],
                                     read_aggregate("temporal_c2_full_record").query("design == @primary").iloc[0], stability, CFG.criteria.ari)
    save_figure(fig, plotted, "figS1_temporal")
    print(f"06 done in {timer.seconds()} s")


if __name__ == "__main__":
    main()
