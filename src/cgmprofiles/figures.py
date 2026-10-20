"""Manuscript figures, one folder each under results/figures/ (names in config/paper1.yaml, section figures).

Each folder holds the figure files (.pdf, .png), a make_figure_N.py that redraws only that figure, and
source_data/ with the aggregate tables the figure plots and the plotted values (<stem>_plotted.json).
Participant-level values (the Fig 1 points) stay in restricted/ and are never copied here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import io
from .cgm_metrics import WINDOWS
from .config import get_config, results_dir
from .plotting import fig_forward, fig_forward_profiles, fig_reverse, fig_reverse_correspondence, fig_strata, style
from .report import jsonable, read_aggregate, read_restricted

CFG = get_config()
FIG = CFG.figures
PRIMARY_PARTITION = "kmeans3_extended_window_full"
T2D_LONG = CFG.labels.strata_long["t2d_oral_non_insulin"]
CAPTION_FIGURE_1_PROFILES = (
    r"\caption{Observed CGM outcomes across the three clinical profiles in " + T2D_LONG.replace("type 2", "type~2")
    + r" ($N=458$; C1 $n=192$, C2 $n=111$, C3 $n=155$). Columns show mean glucose, time in range (70--180~mg/dL) and glucose variability "
    r"(coefficient of variation). The top row (A--C) uses the full CGM record; the bottom row (D--F) uses the first 48 hours, with the same "
    r"y-axis limits within each column. Full-record coefficient of variation (C) is the primary outcome. Shaded shapes are kernel density "
    r"estimates of each profile's distribution, light points are individual participants, white boxes span the interquartile range and the "
    r"black bar marks the median. Effect sizes are reported in Table~2.}")


MAKE_SCRIPT = '''"""Redraw {title} into this folder from the saved outputs.

    python {path}

Run the analyses first (bash run_all.sh). The drawing code is cgmprofiles.figures.{function}, and the values
it plots are in source_data/.
"""
from cgmprofiles.figures import {function}

if __name__ == "__main__":
    {function}()
'''


def folder(figure: str) -> Path:
    """results/figures/<folder of this figure>/ with its make_figure_N.py, created on demand."""
    out = results_dir("figures", FIG[figure].folder)
    number = figure.split("_")[1]
    title = f"Figure {number} ({FIG[figure].title})"
    script = out / f"make_figure_{number}.py"
    script.write_text(MAKE_SCRIPT.format(title=title, path=f"results/figures/{FIG[figure].folder}/{script.name}", function=figure))
    return out


def _save(fig, plotted: dict, figure: str, stem_key: str, sources: dict[str, pd.DataFrame], caption: str | None = None) -> None:
    out = folder(figure)
    stem = FIG[figure][stem_key]
    style.save(fig, out, stem)
    data = out / "source_data"
    data.mkdir(exist_ok=True)
    (data / f"{stem}_plotted.json").write_text(json.dumps(jsonable({str(k): v for k, v in plotted.items()}), indent=1) + "\n")
    for name, frame in sources.items():
        frame.to_csv(data / f"{name}.csv", index=False)
    if caption:
        (out / f"{stem}_caption.tex").write_text(caption + "\n")


def figure_1() -> None:
    """Fig 1: CGM outcomes across C1 to C3 (summary version and profile version)."""
    metrics = read_restricted("participant_metrics")
    metrics = metrics[metrics.canonical_stratum == io.PRIMARY]
    kw = read_aggregate("forward_kruskal_wallis")
    kw = kw[kw.canonical_stratum == io.PRIMARY]
    fig, plotted = fig_forward.draw(metrics, kw, f"Raw CGM behavior across clinical profiles in {T2D_LONG}")
    _save(fig, plotted, "figure_1", "summary", {"kruskal_wallis_eta_squared": kw})
    fig, plotted = fig_forward_profiles.draw(metrics)
    _save(fig, plotted, "figure_1", "profiles", {}, CAPTION_FIGURE_1_PROFILES)


def figure_2() -> None:
    """Fig 2: CGM groups against clinical profiles (summary version, panels A to C from 03 and D from 04; correspondence version)."""
    agreement = read_aggregate("reverse_agreement")
    ari = agreement.set_index(["partition", "metric"]).loc[(PRIMARY_PARTITION, "ARI")]
    contingency_table = read_aggregate("reverse_primary_contingency")
    contingency = contingency_table.set_index("cgm_cluster")[["C1", "C2", "C3"]].astype(int)
    forward_reverse = read_aggregate("forward_vs_reverse_eta_squared")
    b5 = read_aggregate("reverse_prediction").fillna({"window": ""})
    nulls = read_aggregate("reverse_prediction_null_draws")
    pooled = read_aggregate("pooled_agreement")
    fig, plotted = fig_reverse.draw(contingency, ari, forward_reverse, b5, nulls, pooled, CFG.criteria, int(contingency.to_numpy().sum()))
    _save(fig, plotted, "figure_2", "summary", {
        "cgm_group_by_profile_counts": contingency_table, "agreement_primary_partition": agreement[agreement.partition == PRIMARY_PARTITION],
        "forward_vs_reverse_eta_squared": forward_reverse, "prediction_of_profiles": b5[b5.target == "C1/C2/C3"],
        "prediction_permutation_null": nulls[nulls.analysis.str.startswith("C1/C2/C3")], "pooled_cohort_agreement": pooled})
    b3 = read_aggregate("reverse_clinical_differences")
    b3 = b3[(b3.partition == PRIMARY_PARTITION) & (b3.subset == "all")]
    groups = read_aggregate("reverse_primary_groups")
    fig, plotted = fig_reverse_correspondence.draw_correspondence(contingency, groups, ari, b3.set_index("factor"), b5)
    _save(fig, plotted, "figure_2", "correspondence", {
        "cgm_group_by_profile_counts": contingency_table, "cgm_group_medians": groups[["partition", "cluster", "n", "median_mean_glucose"]],
        "clinical_differences_across_cgm_groups": b3, "prediction_of_profiles": b5[b5.target == "C1/C2/C3"],
        "prediction_of_clinical_variables": b5[(b5.direction == "CGM to clinical variable") & (b5.feature_set == "extended") & (b5.window == WINDOWS[1])]},
          fig_reverse_correspondence.CAPTION_CORRESPONDENCE)


def figure_3() -> None:
    """Fig 3: correspondence across glycemic strata (non-insulin T2D values from 02 and 03, other strata from 04); also a version with panels A and B only."""
    cols = ["canonical_stratum", "metric", "eta_squared", "ci_low", "ci_high"]
    s1, eta = read_aggregate("strata_forward"), read_aggregate("forward_eta_squared_ci")
    forward = pd.concat([s1[s1.window == WINDOWS[1]][cols], eta[eta.window == WINDOWS[1]].assign(canonical_stratum=io.PRIMARY)[cols]], ignore_index=True)
    s2 = read_aggregate("strata_clinical_differences")
    s2 = s2[s2.window == WINDOWS[1]].rename(columns={"stratum": "canonical_stratum"})
    fr = read_aggregate("forward_vs_reverse_eta_squared")
    fr = fr[fr.direction.str.startswith("reverse")].assign(canonical_stratum=io.PRIMARY).rename(columns={"variable": "factor"})
    rcols = ["canonical_stratum", "factor", "eta_squared", "ci_low", "ci_high"]
    reverse = pd.concat([s2[rcols], fr[rcols]], ignore_index=True)
    spec = read_aggregate("strata_specificity")
    spec = spec[(spec.variant == "primary (T2D scaler, all)") & (spec.metric == "coefficient_of_variation_pct")]
    t2d_c2 = read_aggregate("forward_c2_vs_rest")
    fig, plotted = fig_strata.draw(forward, reverse, spec, t2d_c2, CFG.criteria.medium_eta)
    _save(fig, plotted, "figure_3", "summary", {"forward_eta_squared_full_record": forward, "reverse_eta_squared_full_record": reverse,
                                                "c2_minus_rest_cv_other_strata": spec, "c2_minus_rest_cv_non_insulin_t2d": t2d_c2})
    fig, plotted = fig_strata.draw(forward, reverse, spec, t2d_c2, CFG.criteria.medium_eta, with_panel_c=False)
    _save(fig, plotted, "figure_3", "summary_panels_a_b", {})


FIGURES = {"figure_1": figure_1, "figure_2": figure_2, "figure_3": figure_3}
