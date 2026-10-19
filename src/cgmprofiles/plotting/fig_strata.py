"""Figure 3: clinical-CGM correspondence across glycemic strata.

A: forward eta squared (full record) across each stratum's own profiles; B: reverse eta squared
across each stratum's CGM groups (primary k), HbA1c on its own axis; C: CV difference, C2-assigned
minus rest, for healthy and prediabetes participants assigned to the frozen non-insulin T2D
centroids, and for non-insulin T2D itself, in both windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import style
from .style import plt

STRATA = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
METRICS = [("mean_glucose", "Mean glucose"), ("time_in_range_pct", "Time in range"), ("coefficient_of_variation_pct", "CV")]
FACTORS = [("participants_age", "Age"), ("bmi_baseline", "BMI"), ("c_peptide_ngml_baseline", "C-peptide"),
           ("tg_hdl_ratio", "TG/HDL"), ("waist_to_hip_ratio_baseline", "Waist-to-hip")]
HBA1C = "hba1c_percent_baseline"
DODGE = 0.18
FIGURE_SIZE_IN = (7.4, 5.4)


def _dot_panel(ax, table: pd.DataFrame, key: str, items, title: str, medium_eta: float) -> dict:
    base = np.arange(len(items))[::-1].astype(float)
    plotted = {}
    for j, stratum in enumerate(STRATA):
        sub = table[table.canonical_stratum == stratum].set_index(key)
        for (name, _), yy in zip(items, base + (1.5 - j) * DODGE):
            r = sub.loc[name]
            ax.errorbar(r.eta_squared, yy, xerr=[[r.eta_squared - r.ci_low], [r.ci_high - r.eta_squared]], fmt="o", color=style.STRATUM_COLORS[stratum],
                        markersize=3, elinewidth=style.CI_LINEWIDTH * 0.8, label=style.STRATUM_SHORT[stratum] if name == items[0][0] else None)
            plotted[(stratum, name)] = [r.eta_squared, r.ci_low, r.ci_high]
    ax.axvline(medium_eta, color=style.COLOR_NULL, linestyle=style.REFERENCE_STYLE, linewidth=style.REFERENCE_LINEWIDTH)
    ax.set_yticks(base, [label for _, label in items])
    ax.set_xlim(0, None)
    ax.set_xlabel("Eta squared [95% CI]", fontsize=style.AXIS_LABEL_PT)
    ax.set_title(title, fontsize=style.PANEL_TITLE_PT + 1, fontweight="bold")
    style.boxed_axes(ax)
    return plotted


def draw(forward: pd.DataFrame, reverse: pd.DataFrame, specificity: pd.DataFrame, t2d_c2: pd.DataFrame, medium_eta: float) -> tuple[plt.Figure, dict]:
    """forward / reverse: canonical_stratum, metric or factor, eta_squared, ci_low, ci_high (full record, all strata);
    specificity: primary-variant CV rows (canonical_stratum, window, difference and CI); t2d_c2: T2D C2 minus rest by window."""
    t2d = style.STRATUM_SHORT["t2d_oral_non_insulin"]
    t2d_text = style.in_sentence(t2d)
    style.setup_base()
    fig = plt.figure(figsize=FIGURE_SIZE_IN, facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.12, right=0.98, top=0.86, bottom=0.09, wspace=0.42, hspace=0.55, height_ratios=[1.25, 1])
    ax_a = fig.add_subplot(grid[0, 0])
    sub = grid[0, 1].subgridspec(2, 1, height_ratios=[5, 1.3], hspace=0.75)
    ax_b, ax_b2 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    ax_c = fig.add_subplot(grid[1, :])
    plotted = {"A": _dot_panel(ax_a, forward, "metric", METRICS, "Forward: CGM metrics across profiles\n(full record)", medium_eta)}
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, fontsize=style.LEGEND_PT - 0.5, frameon=False)
    plotted["B"] = _dot_panel(ax_b, reverse, "factor", FACTORS, "Reverse: clinical variables across\nCGM groups (full record, primary k)", medium_eta)
    hb = reverse[reverse.factor == HBA1C].set_index("canonical_stratum")
    for j, stratum in enumerate(STRATA):
        r = hb.loc[stratum]
        ax_b2.errorbar(r.eta_squared, (1.5 - j) * DODGE, xerr=[[r.eta_squared - r.ci_low], [r.ci_high - r.eta_squared]], fmt="s",
                       color=style.STRATUM_COLORS[stratum], markersize=3, elinewidth=style.CI_LINEWIDTH * 0.8)
        plotted["B"][(stratum, HBA1C)] = [r.eta_squared, r.ci_low, r.ci_high]
    ax_b2.set_yticks([0], ["HbA1c\n(expected)"])
    ax_b2.set_ylim(-0.5, 0.5)
    ax_b2.set_xlim(0, 0.6)
    style.boxed_axes(ax_b2)

    spec = specificity.set_index(["canonical_stratum", "window"])
    ref = t2d_c2.set_index("window")
    groups = [("healthy", f"Healthy\n(assigned to {t2d_text} centroids)"), ("pre_diabetes", f"Prediabetes\n(assigned to {t2d_text} centroids)"),
              ("t2d_oral_non_insulin", f"{t2d}\n(own profiles)")]
    windows = [("window_48h", "48 h", "o"), ("window_full", "full record", "D")]
    plotted["C"] = {}
    for i, (stratum, _) in enumerate(groups):
        for w, (window, _, marker) in enumerate(windows):
            if stratum == "t2d_oral_non_insulin":
                r = ref.loc[window]
                value, low, high = r.mean_diff, r.ci_low, r.ci_high
            else:
                r = spec.loc[(stratum, window)]
                value, low, high = r.mean_diff_c2_minus_rest, r.diff_ci_low, r.diff_ci_high
            color = style.STRATUM_COLORS[stratum]
            ax_c.errorbar(i + (w - 0.5) * 0.25, value, yerr=[[value - low], [high - value]], fmt=marker, color=color, markersize=4,
                          elinewidth=style.CI_LINEWIDTH, markerfacecolor=color if w == 0 else "white")
            plotted["C"][(stratum, window)] = [value, low, high]
    for w, (window, _, _) in enumerate(windows):
        half = ref.loc[window, "mean_diff"] / 2
        ax_c.axhspan(half, -half, xmin=0, xmax=1, color=style.COLOR_NULL, alpha=0.08 if w == 0 else 0.0)
    half = abs(ref.loc["window_48h", "mean_diff"]) / 2
    ax_c.text(2.45, -half + 0.05, f"half the {t2d_text} difference (48 h)", fontsize=style.ANNOTATION_PT, color=style.COLOR_NULL, ha="right", va="bottom")
    ax_c.axhline(0, color=style.COLOR_ZERO, linewidth=style.REFERENCE_LINEWIDTH)
    ax_c.set_xticks(range(len(groups)), [label for _, label in groups])
    ax_c.set_xlim(-0.5, 2.5)
    ax_c.set_ylabel("CV difference, C2 minus rest\n(percentage points, 95% CI)", fontsize=style.AXIS_LABEL_PT)
    ax_c.set_title(f"Specificity check: lower CV of C2 is largest in {t2d_text}, smaller in prediabetes, absent in healthy",
                   fontsize=style.PANEL_TITLE_PT + 1, fontweight="bold")
    for wlabel, marker, face in (("48 h", "o", style.COLOR_NULL), ("full record", "D", "white")):
        ax_c.plot([], [], marker, color=style.COLOR_NULL, markerfacecolor=face, label=wlabel)
    ax_c.legend(fontsize=style.LEGEND_PT - 1, loc="lower left", frameon=False, title="Window", title_fontsize=style.LEGEND_PT - 1)
    style.boxed_axes(ax_c)
    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        ax.text(-0.1 if ax is not ax_c else -0.05, 1.06, letter, transform=ax.transAxes, fontsize=style.LETTER_PT, fontweight="bold", va="bottom")
    fig.suptitle("Clinical profiles and CGM across glycemic strata", fontsize=style.PANEL_TITLE_PT + 3, fontweight="bold")
    return fig, plotted
