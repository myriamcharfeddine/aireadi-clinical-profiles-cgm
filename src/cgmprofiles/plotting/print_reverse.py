"""Figure 2, print version (ported from the record's plot_fig2_paper.py), same values as Fig 2.

A: CGM groups by clinical profile, counts and row percentages; B: rank eta squared, CGM outcomes across
C1 to C3 (full record filled, 48 h open) and clinical variables across G1 to G3, HbA1c on its own axis;
C: profile classification from HbA1c, CGM and both; D: pooled ARI across the number of CGM groups.
No threshold lines; permutation nulls and p values belong in the caption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from . import print_style as ps
from . import style
from .style import plt

HEIGHT_CM = 18.5
B_AXES_X_CM, B_LABEL_X_CM, B_LETTER_X_CM = 12.0, 8.45, 7.95
D_LABEL_X_CM, D_LETTER_X_CM = 9.05, 8.55
ETA_X, HBA1C_X = (-0.006, 0.15), (0.30, 0.52)
HEAT_MAX_PCT = 60
DARK, MID, GREY = style.NEUTRAL_DARK, style.NEUTRAL_MID, style.COLOR_NULL
T2D_GREEN = style.STRATUM_COLORS["t2d_oral_non_insulin"]
HEAT_CMAP = LinearSegmentedColormap.from_list("grey", ["#FFFFFF", MID])
CGM_LABELS = {"mean_glucose": "Mean glucose", "time_in_range_pct": "Time in range", "coefficient_of_variation_pct": "Coefficient of variation"}
CLINICAL_LABELS = {"participants_age": "Age", "bmi_baseline": "BMI", "c_peptide_ngml_baseline": "C-peptide",
                   "tg_hdl_ratio": "TG/HDL ratio", "waist_to_hip_ratio_baseline": "Waist-to-hip ratio"}
WINDOWS = [("window_full", "Full record", True), ("window_48h", "48 h", False)]
ARI_SERIES = [("diagnostic stratum (4)", "Enrolment stratum", "o", DARK),
              ("stratum-profile label (10)", "Stratum + clinical profile", "s", MID),
              ("C1/C2/C3 inside the 458", f"Clinical profile within {style.in_sentence(style.STRATUM_SHORT['t2d_oral_non_insulin'])}", "^", T2D_GREEN)]


def _hci(ax, x, y, low, high, marker, color, filled=True):
    ax.errorbar(x, y, xerr=[[x - low], [high - x]], fmt=marker, color=color, markersize=ps.MARKER_SIZE, elinewidth=ps.CI_LINEWIDTH,
                capsize=0, markerfacecolor=color if filled else "white", markeredgewidth=0.9, zorder=3)


def _vci(ax, x, y, low, high, marker, color):
    ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt=marker, color=color, markersize=ps.MARKER_SIZE, elinewidth=ps.CI_LINEWIDTH,
                capsize=0, markerfacecolor=color, markeredgewidth=0.9, zorder=3)


def _panel_a(fig, ax, cax, table: pd.DataFrame, ari: pd.Series) -> dict:
    counts = table.to_numpy()
    row_pct = 100 * counts / counts.sum(axis=1, keepdims=True)
    image = ax.imshow(row_pct, cmap=HEAT_CMAP, vmin=0, vmax=HEAT_MAX_PCT, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{counts[i, j]}\n{row_pct[i, j]:.0f}%", ha="center", va="center", fontsize=ps.FONT_PT, linespacing=1.15,
                    color="white" if row_pct[i, j] > HEAT_MAX_PCT * 0.55 else ps.INK)
    ax.set_xticks(range(3), list(table.columns))
    for tick in ax.get_xticklabels():
        tick.set_color(style.PROFILE_COLORS[tick.get_text()])
        tick.set_fontweight("bold")
    ax.set_yticks(range(3), [f"{g} (n = {int(table.loc[g].sum())})" for g in table.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Clinical profile")
    ax.set_ylabel("CGM group")
    ps.fig_text(fig, HEIGHT_CM, 0.75, 17.35, f"ARI {ari.value:.3f} (95% CI {ari.ci_low:.3f} to {ari.ci_high:.3f})")
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Within CGM group (%)", fontsize=ps.FONT_PT)
    bar.set_ticks([0, 20, 40, 60])
    bar.ax.tick_params(labelsize=ps.FONT_PT, length=2, width=ps.SPINE_LINEWIDTH)
    bar.outline.set_linewidth(ps.SPINE_LINEWIDTH)
    return {"counts": counts.tolist(), "row_percent": row_pct.tolist()}


def _panel_b(fig, ax, ax_h, table: pd.DataFrame) -> dict:
    forward = table[table.direction.str.startswith("forward")].set_index(["variable", "window"])
    reverse = table[table.direction.str.startswith("reverse")].set_index("variable")
    row, ticks, labels, headers, plotted = 0.0, [], [], [(0.0, "CGM outcomes across clinical profiles")], {}
    row -= 1.0
    for variable, label in CGM_LABELS.items():
        for k, (window, _, filled) in enumerate(WINDOWS):
            r = forward.loc[(variable, window)]
            _hci(ax, r.eta_squared, row - 0.17 + 0.34 * k, r.ci_low, r.ci_high, "o", DARK, filled)
            plotted[f"{variable}/{window}"] = [r.eta_squared, r.ci_low, r.ci_high]
        ticks.append(row)
        labels.append(label)
        row -= 1.0
    row -= 0.3
    headers.append((row, "Clinical variables across CGM groups"))
    row -= 1.0
    for variable, label in CLINICAL_LABELS.items():
        r = reverse.loc[variable]
        _hci(ax, r.eta_squared, row, r.ci_low, r.ci_high, "s", MID)
        plotted[variable] = [r.eta_squared, r.ci_low, r.ci_high]
        ticks.append(row)
        labels.append(label)
        row -= 1.0
    ax.set_yticks(ticks, labels)
    ax.set_ylim(row + 0.4, 0.6)
    ax.set_xlim(*ETA_X)
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.axvline(0, color=ps.INK, linewidth=ps.ZERO_LINEWIDTH, zorder=1)
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    for y, text in headers:
        ax.text(B_LABEL_X_CM / ps.WIDTH_CM, y, text, transform=trans, ha="left", va="center", fontsize=ps.FONT_PT, fontstyle="italic",
                color=ps.INK, zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xlabel("Rank-based effect size (η²)")
    ps.axes_style(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", linestyle="none", color=DARK, markerfacecolor=DARK if f else "white", markersize=ps.MARKER_SIZE, label=lab)
               for _, lab, f in WINDOWS]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(B_LABEL_X_CM / ps.WIDTH_CM, 17.35 / HEIGHT_CM), ncol=2, frameon=False,
               handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)
    hb = reverse.loc["hba1c_percent_baseline"]
    _hci(ax_h, hb.eta_squared, 0, hb.ci_low, hb.ci_high, "s", GREY)
    plotted["hba1c_percent_baseline"] = [hb.eta_squared, hb.ci_low, hb.ci_high]
    ax_h.set_xlim(*HBA1C_X)
    ax_h.set_xticks([0.30, 0.35, 0.40, 0.45, 0.50])
    ax_h.set_yticks([0], ["HbA1c"])
    ax_h.set_ylim(-0.7, 0.7)
    ax_h.set_xlabel("η², HbA1c (separate scale)")
    ps.axes_style(ax_h, grid_axis="x")
    ax_h.tick_params(axis="y", length=0)
    return plotted


def _panel_c(fig, axes, b5: pd.DataFrame) -> dict:
    cls = b5[b5.target == "C1/C2/C3"]
    specs = [("balanced_accuracy", "Balanced accuracy", 1 / 3, "1/3", (0.30, 0.52), [0.30, 0.40, 0.50]),
             ("macro_auroc", "Macro AUROC", 0.5, "0.50", (0.47, 0.72), [0.50, 0.60, 0.70])]
    rows, plotted = {"HbA1c": 2, "CGM": 1, "HbA1c + CGM": 0}, {}
    for ax, (metric, title, chance, chance_label, xlim, xticks) in zip(axes, specs):
        ref = cls[(cls.feature_set == "HbA1c alone") & (cls.metric == metric)].iloc[0]
        cgm = cls[(cls.feature_set == "extended") & (cls.window == "window_full") & (cls.metric == metric)].iloc[0]
        gain = cls[(cls.feature_set == "extended") & (cls.window == "window_full") & cls.metric.str.startswith(f"{metric} gain")].iloc[0]
        _hci(ax, ref.value, rows["HbA1c"], ref.ci_low, ref.ci_high, "s", GREY)
        _hci(ax, cgm.value, rows["CGM"], cgm.ci_low, cgm.ci_high, "o", MID)
        ax.plot(gain.hba1c_plus_cgm, rows["HbA1c + CGM"], "D", color=DARK, markerfacecolor=DARK, markeredgewidth=0.9, markersize=ps.MARKER_SIZE, zorder=3)
        ax.axvline(chance, color=ps.INK, linestyle=ps.CHANCE_STYLE, linewidth=ps.ZERO_LINEWIDTH, zorder=1)
        ax.text(chance, 2.55, f"Chance {chance_label}", ha="left", va="bottom", fontsize=ps.FONT_PT, color=ps.INK)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_ylim(-0.6, 3.2)
        ax.set_yticks(list(rows.values()), list(rows))
        ax.set_xlabel(title)
        ps.axes_style(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)
        plotted[metric] = {"hba1c": [ref.value, ref.ci_low, ref.ci_high], "cgm": [cgm.value, cgm.ci_low, cgm.ci_high], "both": gain.hba1c_plus_cgm,
                           "gain": [gain.value, gain.ci_low, gain.ci_high]}
    axes[1].tick_params(axis="y", labelleft=False)
    g = plotted["macro_auroc"]["gain"]
    ps.fig_text(fig, HEIGHT_CM, 0.5, 1.75, f"Macro AUROC gain of HbA1c + CGM over HbA1c:\n{g[0]:+.3f} (95% CI {g[1]:.3f} to {g[2]:.3f})",
                va="center", linespacing=1.3)
    return plotted


def _panel_d(fig, ax, b6: pd.DataFrame) -> dict:
    b6 = b6[b6.metric == "ARI"]
    ks = sorted(b6.pooled_k.unique())
    offsets = {name: (i - 1) * 0.22 for i, (name, *_) in enumerate(ARI_SERIES)}
    plotted = {}
    for target, _, marker, color in ARI_SERIES:
        sub = b6[b6.target == target].set_index("pooled_k").loc[ks]
        for x, k in enumerate(ks):
            r = sub.loc[k]
            _vci(ax, x + offsets[target], r.value, r.ci_low, r.ci_high, marker, color)
        plotted[target] = sub[["value", "ci_low", "ci_high"]].to_numpy().tolist()
    within = b6[b6.target.str.contains("within-stratum")].set_index("pooled_k").loc[ks]
    xs = np.arange(len(ks)) + offsets["stratum-profile label (10)"]
    ax.hlines(within.null_mean, xs - 0.13, xs + 0.13, color=GREY, linewidth=1.8, zorder=4)
    ax.axhline(0, color=ps.INK, linewidth=ps.ZERO_LINEWIDTH, zorder=1)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(-0.01, 0.20)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel("Number of CGM groups")
    ax.set_ylabel("Adjusted Rand index")
    ps.axes_style(ax, grid_axis="y")
    handles = [Line2D([], [], marker=m, linestyle="none", color=c, markersize=ps.MARKER_SIZE, label=lab) for _, lab, m, c in ARI_SERIES]
    handles.insert(2, Line2D([], [], color=GREY, linewidth=1.8, label="Within-stratum permutation mean"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(D_LABEL_X_CM / ps.WIDTH_CM, 2.25 / HEIGHT_CM), ncol=1, frameon=False,
               handletextpad=0.5, borderaxespad=0.0, borderpad=0.0, handlelength=1.4, labelspacing=0.35)
    plotted["within_null_mean"] = within.null_mean.tolist()
    return plotted


def draw(contingency: pd.DataFrame, ari: pd.Series, forward_reverse: pd.DataFrame, b5: pd.DataFrame, b6: pd.DataFrame) -> tuple[plt.Figure, dict]:
    ps.setup()
    fig = ps.figure(HEIGHT_CM)
    box = lambda *c: ps.box(HEIGHT_CM, *c)  # noqa: E731
    ax_a, cax = fig.add_axes(box(2.75, 11.7, 6.35, 16.85)), fig.add_axes(box(6.55, 11.7, 6.75, 16.85))
    ax_b, ax_h = fig.add_axes(box(B_AXES_X_CM, 11.95, 15.45, 16.85)), fig.add_axes(box(B_AXES_X_CM, 9.85, 15.45, 10.4))
    ax_c1, ax_c2 = fig.add_axes(box(2.35, 3.55, 5.05, 7.35)), fig.add_axes(box(5.4, 3.55, 8.1, 7.35))
    ax_d = fig.add_axes(box(10.3, 3.55, 15.6, 7.35))
    plotted = {"A": _panel_a(fig, ax_a, cax, contingency, ari), "B": _panel_b(fig, ax_b, ax_h, forward_reverse),
               "C": _panel_c(fig, (ax_c1, ax_c2), b5), "D": _panel_d(fig, ax_d, b6)}
    for x_cm, y_cm, letter, title in ((0.0, 17.95, "A", "Overlap between clinical and CGM groups"),
                                      (B_LETTER_X_CM, 17.95, "B", "Differences across the two groupings"),
                                      (0.0, 8.15, "C", "Prediction of clinical profiles"),
                                      (D_LETTER_X_CM, 8.15, "D", "Agreement across glycemic strata")):
        ps.fig_text(fig, HEIGHT_CM, x_cm, y_cm, letter, bold=True, size=ps.LETTER_PT)
        ps.fig_text(fig, HEIGHT_CM, x_cm + 0.5, y_cm, title, bold=True, size=ps.TITLE_PT)
    ps.check_fonts(fig)
    return fig, plotted
