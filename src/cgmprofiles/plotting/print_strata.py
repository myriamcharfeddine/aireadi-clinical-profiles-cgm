"""Figure 3, print version (ported from the record's plot_fig3_paper.py), same values as Fig 3.

A: rank eta squared of the CGM outcomes across each stratum's profiles (full record); B: rank eta squared of the
clinical variables across each stratum's CGM groups, HbA1c on its own axis; C: CV difference, C2-assigned minus
others, for healthy and prediabetes (assigned to the non-insulin T2D centroids) and non-insulin T2D, both windows.
Strata keep their config colours with one marker each; no 0.06 line and no half-T2D band.
"""
from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D

from . import print_style as ps
from . import style
from .style import plt

HEIGHT_CM = 15.5
DODGE = 0.19
METRICS = [("mean_glucose", "Mean glucose"), ("time_in_range_pct", "Time in range"), ("coefficient_of_variation_pct", "Coefficient of variation")]
FACTORS = [("participants_age", "Age"), ("bmi_baseline", "BMI"), ("c_peptide_ngml_baseline", "C-peptide"),
           ("tg_hdl_ratio", "TG/HDL ratio"), ("waist_to_hip_ratio_baseline", "Waist-to-hip ratio")]
HBA1C = "hba1c_percent_baseline"
ETA_X_A, ETA_X_B, HBA1C_X = (-0.006, 0.15), (-0.008, 0.20), (-0.02, 0.60)
C_GROUPS = ["healthy", "pre_diabetes", "t2d_oral_non_insulin"]
C_WINDOWS = [("window_full", "Full record", "o", True, 6.0, 1.3), ("window_48h", "48 h", "D", False, 4.2, 0.8)]
C_OFFSET = 0.13
A_AXES_X_CM, B_AXES_X_CM, B_LABEL_X_CM = 4.15, 11.1, 8.3


def _hci(ax, x, y, low, high, stratum, size=ps.MARKER_SIZE - 0.5):
    ax.errorbar(x, y, xerr=[[x - low], [high - x]], fmt=ps.STRATUM_MARKER[stratum], color=style.STRATUM_COLORS[stratum], markersize=size,
                elinewidth=ps.CI_LINEWIDTH, capsize=0, markeredgewidth=0.6, markeredgecolor="white", zorder=3)


def _dot_panel(ax, table: pd.DataFrame, key: str, items, xlim) -> dict:
    plotted = {}
    for j, stratum in enumerate(ps.STRATA):
        sub = table[table.canonical_stratum == stratum].set_index(key)
        for row, (name, _) in enumerate(items):
            r = sub.loc[name]
            _hci(ax, r.eta_squared, -row + (1.5 - j) * DODGE, r.ci_low, r.ci_high, stratum)
            plotted[(stratum, name)] = [r.eta_squared, r.ci_low, r.ci_high]
    ax.set_yticks([-r for r in range(len(items))], [label for _, label in items])
    ax.set_ylim(-len(items) + 0.45, 0.55)
    ax.set_xlim(*xlim)
    ax.axvline(0, color=ps.INK, linewidth=ps.ZERO_LINEWIDTH, zorder=1)
    ax.set_xlabel("Rank-based effect size (η²)")
    ps.axes_style(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return plotted


def _panel_c(ax, specificity: pd.DataFrame, t2d_c2: pd.DataFrame) -> dict:
    spec, ref = specificity.set_index(["canonical_stratum", "window"]), t2d_c2.set_index("window")
    plotted = {}
    for i, stratum in enumerate(C_GROUPS):
        for w, (window, _, marker, filled, size, lw) in enumerate(C_WINDOWS):
            if stratum == "t2d_oral_non_insulin":
                r = ref.loc[window]
                value, low, high = r.mean_diff, r.ci_low, r.ci_high
            else:
                r = spec.loc[(stratum, window)]
                value, low, high = r.mean_diff_c2_minus_rest, r.diff_ci_low, r.diff_ci_high
            color = style.STRATUM_COLORS[stratum]
            ax.errorbar(i + (w - 0.5) * 2 * C_OFFSET, value, yerr=[[value - low], [high - value]], fmt=marker, color=color, markersize=size,
                        elinewidth=lw, capsize=0, markerfacecolor=color if filled else "white", markeredgewidth=0.9, zorder=3)
            plotted[(stratum, window)] = [value, low, high]
    ax.axhline(0, color=ps.INK, linewidth=ps.ZERO_LINEWIDTH, zorder=1)
    ax.set_xticks(range(len(C_GROUPS)), [style.STRATUM_SHORT[s] for s in C_GROUPS])
    ax.set_xlim(-0.5, len(C_GROUPS) - 0.5)
    ax.set_ylim(-4.0, 2.0)
    ax.set_ylabel("CV difference, C2 − others\n(percentage points)")
    ps.axes_style(ax, grid_axis="y")
    ax.tick_params(axis="x", length=0)
    handles = [Line2D([], [], marker=m, linestyle="none", color=style.NEUTRAL_DARK, markerfacecolor=style.NEUTRAL_DARK if f else "white",
                      markersize=s, label=lab) for _, lab, m, f, s, _ in C_WINDOWS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.5)
    return plotted


def draw(forward: pd.DataFrame, reverse: pd.DataFrame, specificity: pd.DataFrame, t2d_c2: pd.DataFrame) -> tuple[plt.Figure, dict]:
    """Same inputs as fig_strata.draw (full-record eta squared for all strata, primary-variant CV specificity, T2D C2 minus rest)."""
    ps.setup()
    fig = ps.figure(HEIGHT_CM)
    box = lambda *c: ps.box(HEIGHT_CM, *c)  # noqa: E731
    ax_a = fig.add_axes(box(A_AXES_X_CM, 8.55, 7.45, 13.55))
    ax_b, ax_h = fig.add_axes(box(B_AXES_X_CM, 9.85, 15.45, 13.55)), fig.add_axes(box(B_AXES_X_CM, 7.75, 15.45, 8.55))
    ax_c = fig.add_axes(box(2.35, 1.0, 13.0, 5.25))
    plotted = {"A": _dot_panel(ax_a, forward, "metric", METRICS, ETA_X_A)}
    ax_a.set_xticks([0, 0.05, 0.10, 0.15])
    plotted["B"] = _dot_panel(ax_b, reverse, "factor", FACTORS, ETA_X_B)
    ax_b.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    hb = reverse[reverse.factor == HBA1C].set_index("canonical_stratum")
    for j, stratum in enumerate(ps.STRATA):
        r = hb.loc[stratum]
        _hci(ax_h, r.eta_squared, (1.5 - j) * DODGE * 1.15, r.ci_low, r.ci_high, stratum)
        plotted["B"][(stratum, HBA1C)] = [r.eta_squared, r.ci_low, r.ci_high]
    ax_h.set_yticks([0], ["HbA1c"])
    ax_h.set_ylim(-0.5, 0.5)
    ax_h.set_xlim(*HBA1C_X)
    ax_h.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax_h.axvline(0, color=ps.INK, linewidth=ps.ZERO_LINEWIDTH, zorder=1)
    ax_h.set_xlabel("η², HbA1c (separate scale)")
    ps.axes_style(ax_h, grid_axis="x")
    ax_h.spines["left"].set_visible(False)
    ax_h.tick_params(axis="y", length=0)
    plotted["C"] = _panel_c(ax_c, specificity, t2d_c2)
    handles = [Line2D([], [], marker=ps.STRATUM_MARKER[s], linestyle="-", linewidth=ps.CI_LINEWIDTH, color=style.STRATUM_COLORS[s],
                      markeredgecolor="white", markeredgewidth=0.6, markersize=ps.MARKER_SIZE, label=style.STRATUM_SHORT[s]) for s in ps.STRATA]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 15.05 / HEIGHT_CM), ncol=4, frameon=False, handlelength=1.6,
               handletextpad=0.4, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)
    for x_cm, y_cm, letter, title in ((0.0, 14.3, "A", "CGM differences across clinical profiles"),
                                      (B_LABEL_X_CM - 0.5, 14.3, "B", "Clinical differences across CGM groups"),
                                      (0.0, 6.05, "C", "C2-associated variability across strata")):
        ps.fig_text(fig, HEIGHT_CM, x_cm, y_cm, letter, bold=True, size=ps.LETTER_PT)
        ps.fig_text(fig, HEIGHT_CM, x_cm + 0.5, y_cm, title, bold=True, size=ps.TITLE_PT)
    ps.check_fonts(fig)
    return fig, plotted
