"""Figure 2: agreement and prediction between CGM groups and clinical profiles.

A: CGM groups G1 to G3 against C1 to C3 (primary partition) with ARI; B: forward against reverse
eta squared, HbA1c on its own axis; C: profile classification from CGM, HbA1c and both, against
the permutation null (from analyses/03); D: pooled 1,591 CGM groups against diagnosis, combined
stratum-profile labels and C1 to C3 (from analyses/04).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import style
from .style import plt

COLOR_FORWARD = style.NEUTRAL_DARK
COLOR_REVERSE = style.NEUTRAL_MID
COLOR_HBA1C = style.COLOR_NULL
COLOR_THRESHOLD = style.NEUTRAL_DARK
HEATMAP_CMAP = LinearSegmentedColormap.from_list("grey", ["#FFFFFF", style.NEUTRAL_MID])
FIGURE_SIZE_IN = (7.2, 5.6)
ETA_X_MAX = 0.15
HBA1C_X_RANGE = (0.30, 0.52)
CHANCE_BA = 1 / 3
SHORT = {"mean_glucose": "Mean glucose", "time_in_range_pct": "Time in range", "coefficient_of_variation_pct": "CV",
         "participants_age": "Age", "bmi_baseline": "BMI", "c_peptide_ngml_baseline": "C-peptide", "tg_hdl_ratio": "TG/HDL",
         "waist_to_hip_ratio_baseline": "Waist-to-hip", "hba1c_percent_baseline": "HbA1c"}
WINDOW_SHORT = {"window_48h": "48 h", "window_full": "full"}


def _letter(ax, text) -> None:
    ax.text(-0.12, 1.06, text, transform=ax.transAxes, fontsize=style.LETTER_PT, fontweight="bold", va="bottom")


def _panel_a(ax, table: pd.DataFrame, ari: pd.Series) -> dict:
    counts = table.to_numpy()
    ax.imshow(counts, cmap=HEATMAP_CMAP, vmin=0, vmax=counts.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=style.ANNOTATION_PT + 1,
                    color="white" if counts[i, j] > counts.max() * 0.6 else "black")
    ax.set_xticks(range(3), list(table.columns))
    ax.set_yticks(range(3), [f"{g}\n(n={int(table.loc[g].sum())})" for g in table.index])
    for tick, profile in zip(ax.get_xticklabels(), table.columns):
        tick.set_color(style.PROFILE_COLORS[profile])
        tick.set_fontweight("bold")
    ax.set_title(f"CGM groups vs clinical profiles\nARI {ari.value:.3f} [{ari.ci_low:.3f}, {ari.ci_high:.3f}]", fontsize=style.PANEL_TITLE_PT + 1, fontweight="bold")
    ax.set_xlabel("Clinical profile", fontsize=style.AXIS_LABEL_PT)
    ax.set_ylabel("CGM group (extended, full record)", fontsize=style.AXIS_LABEL_PT)
    ax.tick_params(labelsize=style.TICK_PT)
    return {"counts": counts.tolist(), "ARI": [ari.value, ari.ci_low, ari.ci_high]}


def _panel_b(ax, ax_hba1c, table: pd.DataFrame, medium_eta: float) -> dict:
    forward = table[table.direction.str.startswith("forward")]
    reverse = table[table.direction.str.startswith("reverse") & ~table.hba1c_flag]
    hba1c = table[table.direction.str.startswith("reverse") & table.hba1c_flag].iloc[0]
    rows = [(f"{SHORT[r.variable]}, {WINDOW_SHORT[r.window]}", r, COLOR_FORWARD) for r in forward.itertuples()]
    rows += [(SHORT[r.variable], r, COLOR_REVERSE) for r in reverse.itertuples()]
    positions = np.arange(len(rows))[::-1].astype(float)
    positions[: len(forward)] += 0.6
    plotted = {}
    for y, (label, r, color) in zip(positions, rows):
        ax.errorbar(r.eta_squared, y, xerr=[[r.eta_squared - r.ci_low], [r.ci_high - r.eta_squared]], fmt="o", color=color,
                    markersize=3.5, elinewidth=style.CI_LINEWIDTH, capsize=0)
        plotted[label] = [r.eta_squared, r.ci_low, r.ci_high]
    ax.axvline(medium_eta, color=COLOR_THRESHOLD, linestyle=style.REFERENCE_STYLE, linewidth=style.REFERENCE_LINEWIDTH)
    ax.text(medium_eta + 0.0015, positions.min() + 1.6, f"{medium_eta:.2f}", color=COLOR_THRESHOLD, fontsize=style.ANNOTATION_PT)
    ax.set_yticks(positions, [label for label, _, _ in rows])
    ax.set_xlim(0, ETA_X_MAX)
    ax.set_xlabel("Eta squared [95% CI]", fontsize=style.AXIS_LABEL_PT)
    ax.set_title("Forward (CGM across C1 to C3) vs reverse\n(clinical across G1 to G3)", fontsize=style.PANEL_TITLE_PT + 1, fontweight="bold")
    ax.plot([], [], "o", color=COLOR_FORWARD, label="Forward")
    ax.plot([], [], "o", color=COLOR_REVERSE, label="Reverse")
    ax.legend(fontsize=style.LEGEND_PT - 1, loc="lower right", frameon=False)
    style.boxed_axes(ax)
    ax_hba1c.errorbar(hba1c.eta_squared, 0, xerr=[[hba1c.eta_squared - hba1c.ci_low], [hba1c.ci_high - hba1c.eta_squared]], fmt="s",
                      color=COLOR_HBA1C, markersize=3.5, elinewidth=style.CI_LINEWIDTH)
    ax_hba1c.set_xlim(*HBA1C_X_RANGE)
    ax_hba1c.set_yticks([0], ["HbA1c (reverse,\nexpected)"])
    ax_hba1c.set_ylim(-1, 1)
    style.boxed_axes(ax_hba1c)
    plotted["HbA1c"] = [hba1c.eta_squared, hba1c.ci_low, hba1c.ci_high]
    return plotted


def _panel_c(ax, b5: pd.DataFrame, nulls: pd.DataFrame, auroc_threshold: float) -> dict:
    cls = b5[b5.target == "C1/C2/C3"]
    plotted, tick_labels = {}, []
    for x, (metric, label, chance) in enumerate([("balanced_accuracy", "Balanced accuracy", CHANCE_BA), ("macro_auroc", "Macro AUROC", 0.5)]):
        parts = ax.violinplot(nulls[nulls.analysis == f"C1/C2/C3 {metric}"].value.to_numpy(), positions=[x], widths=0.55, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(style.COLOR_NULL)
            body.set_alpha(style.SHADE_ALPHA_HIGH + 0.15)
        cgm = cls[(cls.feature_set == "extended") & (cls.window == "window_full") & (cls.metric == metric)].iloc[0]
        ref = cls[(cls.feature_set == "HbA1c alone") & (cls.metric == metric)].iloc[0]
        gain = cls[(cls.feature_set == "extended") & (cls.window == "window_full") & cls.metric.str.startswith(f"{metric} gain")].iloc[0]
        both = ref.value + gain.value
        ax.errorbar(x - 0.12, cgm.value, yerr=[[cgm.value - cgm.ci_low], [cgm.ci_high - cgm.value]], fmt="o", color=COLOR_REVERSE,
                    markersize=4, elinewidth=style.CI_LINEWIDTH, label="CGM features" if x == 0 else None)
        ax.errorbar(x + 0.12, ref.value, yerr=[[ref.value - ref.ci_low], [ref.ci_high - ref.value]], fmt="s", color=COLOR_HBA1C,
                    markersize=3.5, elinewidth=style.CI_LINEWIDTH, label="HbA1c alone" if x == 0 else None)
        ax.plot(x + 0.24, both, "D", color=COLOR_FORWARD, markersize=3.5, label="HbA1c plus CGM" if x == 0 else None)
        ax.hlines(chance, x - 0.3, x + 0.3, color=style.COLOR_ZERO, linewidth=style.REFERENCE_LINEWIDTH, linestyle=":")
        tick_labels.append(f"{label}\n(permutation p = {cgm.perm_p:.3f})")
        plotted[metric] = {"cgm": [cgm.value, cgm.ci_low, cgm.ci_high], "hba1c": [ref.value, ref.ci_low, ref.ci_high], "both": both, "p": cgm.perm_p}
    ax.hlines(auroc_threshold, 0.7, 1.3, color=COLOR_THRESHOLD, linestyle=style.REFERENCE_STYLE, linewidth=style.REFERENCE_LINEWIDTH)
    ax.text(1.31, auroc_threshold, f"{auroc_threshold:.2f}", color=COLOR_THRESHOLD, fontsize=style.ANNOTATION_PT, va="center")
    ax.set_xticks([0, 1], tick_labels)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.22, 0.78)
    ax.set_ylabel("Cross-validated score", fontsize=style.AXIS_LABEL_PT)
    ax.set_title("Predicting C1 to C3 from CGM\n(grey: permutation null; dotted: chance)", fontsize=style.PANEL_TITLE_PT + 1, fontweight="bold")
    ax.legend(fontsize=style.LEGEND_PT - 1, loc="upper left", frameon=False)
    style.boxed_axes(ax)
    return plotted


def _panel_d(ax, b6: pd.DataFrame, ari_threshold: float) -> dict:
    b6 = b6[b6.metric == "ARI"]
    targets = [("diagnostic stratum (4)", "Diagnosis (4 strata)", style.NEUTRAL_DARK),
               ("stratum-profile label (10)", "Stratum and profile (10)", style.NEUTRAL_MID),
               ("C1/C2/C3 inside the 458", f"C1 to C3, {style.in_sentence(style.STRATUM_SHORT['t2d_oral_non_insulin'])} only", style.STRATUM_COLORS["t2d_oral_non_insulin"])]
    ks = sorted(b6.pooled_k.unique())
    plotted = {}
    for i, (target, label, color) in enumerate(targets):
        sub = b6[b6.target == target].set_index("pooled_k").loc[ks]
        ax.errorbar(np.arange(len(ks)) + (i - 1) * 0.24, sub.value, yerr=[sub.value - sub.ci_low, sub.ci_high - sub.value], fmt="o", color=color,
                    markersize=4, elinewidth=style.CI_LINEWIDTH, label=label)
        plotted[target] = sub[["value", "ci_low", "ci_high"]].to_numpy().tolist()
    within = b6[b6.target.str.contains("within-stratum")].set_index("pooled_k").loc[ks]
    ax.scatter(np.arange(len(ks)), within.null_mean, marker="_", s=80, color=style.COLOR_NULL, zorder=4, label="Within-stratum null mean")
    ax.axhline(ari_threshold, color=COLOR_THRESHOLD, linestyle=style.REFERENCE_STYLE, linewidth=style.REFERENCE_LINEWIDTH)
    ax.set_xticks(range(len(ks)), [f"k = {k}" for k in ks])
    ax.set_ylabel("ARI [95% CI]", fontsize=style.AXIS_LABEL_PT)
    ax.set_title("Pooled 1,591: CGM clusters vs diagnosis\nand vs profile", fontsize=style.PANEL_TITLE_PT + 1, fontweight="bold")
    ax.legend(fontsize=style.LEGEND_PT - 1.5, loc="upper right", frameon=False)
    ax.set_ylim(0, 0.24)
    style.boxed_axes(ax)
    plotted["within_null_mean"] = within.null_mean.tolist()
    return plotted


def draw(contingency: pd.DataFrame, ari: pd.Series, forward_reverse: pd.DataFrame, b5: pd.DataFrame, nulls: pd.DataFrame,
         b6: pd.DataFrame, criteria, n: int) -> tuple[plt.Figure, dict]:
    """contingency: G x C counts (primary, all participants); ari: value, ci_low, ci_high."""
    style.setup_base()
    fig = plt.figure(figsize=FIGURE_SIZE_IN, facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.13, right=0.98, top=0.90, bottom=0.08, wspace=0.55, hspace=0.62)
    ax_a = fig.add_subplot(grid[0, 0])
    sub = grid[0, 1].subgridspec(2, 1, height_ratios=[6, 1], hspace=0.55)
    ax_b, ax_b2 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    ax_c, ax_d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    plotted = {"A": _panel_a(ax_a, contingency, ari), "B": _panel_b(ax_b, ax_b2, forward_reverse, criteria.medium_eta),
               "C": _panel_c(ax_c, b5, nulls, criteria.auroc), "D": _panel_d(ax_d, b6, criteria.ari)}
    for ax, text in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")):
        _letter(ax, text)
    fig.suptitle(f"Reverse direction: grouping by CGM first, {style.in_sentence(style.STRATUM_SHORT['t2d_oral_non_insulin'])} (n = {n})",
                 fontsize=style.PANEL_TITLE_PT + 3, fontweight="bold")
    return fig, plotted
