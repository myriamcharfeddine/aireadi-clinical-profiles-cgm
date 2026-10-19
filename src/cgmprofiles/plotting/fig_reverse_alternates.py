"""Alternative Figure 2 (correspondence, four panels) and its supplementary figure, ported from the record's
plot_fig2_correspondence.py. Same aggregate values as Fig 2; no new numbers.

fig2_correspondence: A profile composition of CGM groups (each column shaded in its profile colour), B clinical
differences across CGM groups (HbA1c on its own scale), C profile prediction from HbA1c, CGM and both with the
paired AUROC gain, D prediction of clinical variables from CGM (R2).
figS_reverse_supplement: A classification scores against the permutation null, B pooled-cohort ARI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from . import style
from .style import plt

WIDTH_CM, HEIGHT_CM, SUPP_HEIGHT_CM = 16.0, 14.5, 7.5
FONT_PT, TITLE_PT, LETTER_PT, MIN_FONT_PT = 7.0, 8.0, 9.5, 7.0
BLACK, GREY = style.COLOR_ZERO, style.COLOR_NULL
T2D_GREEN = style.STRATUM_COLORS["t2d_oral_non_insulin"]
PERCENT_MAX = 60
PROFILES = ["C1", "C2", "C3"]
HBA1C = "hba1c_percent_baseline"
VARIABLES = [("participants_age", "Age"), ("bmi_baseline", "BMI"), ("waist_to_hip_ratio_baseline", "Waist-to-hip ratio"),
             ("tg_hdl_ratio", "TG/HDL ratio"), ("c_peptide_ngml_baseline", "C-peptide"), (HBA1C, "HbA1c")]
GLUCOSE_DESCRIPTOR = {"G1": "lower", "G2": "intermediate", "G3": "higher"}
ETA_LABEL = "Rank-based effect size (η²)"
T2D_SHORT = style.in_sentence(style.STRATUM_SHORT["t2d_oral_non_insulin"])
T2D_LONG = style.STRATUM_LONG["t2d_oral_non_insulin"]


def _axes_style(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, color=style.GRID_COLOR, linewidth=style.GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(style.SPINE_COLOR)
        spine.set_linewidth(style.SPINE_LINEWIDTH)
    ax.tick_params(labelsize=FONT_PT, width=style.SPINE_LINEWIDTH, length=2.5)


def _interval(ax, x, y, low, high, horizontal: bool, **kwargs) -> None:
    err = [[x - low], [high - x]] if horizontal else [[y - low], [high - y]]
    ax.errorbar(x, y, xerr=err if horizontal else None, yerr=None if horizontal else err, capsize=0, elinewidth=style.CI_LINEWIDTH, **kwargs)


def _shade(color: str, percent: float) -> np.ndarray:
    """White blended towards the profile colour in proportion to percent (0 to PERCENT_MAX)."""
    t = min(percent / PERCENT_MAX, 1.0)
    return 1 - t * (1 - np.array(to_rgb(color)))


def _panel_a(fig, ax, table: pd.DataFrame, medians: pd.Series, ari: pd.Series) -> dict:
    colors = [style.PROFILE_COLORS[p] for p in PROFILES]
    counts = table[PROFILES].to_numpy()
    percent = 100 * counts / counts.sum(axis=1, keepdims=True)
    rgb = np.stack([[_shade(colors[j], percent[i, j]) for j in range(3)] for i in range(3)])
    ax.imshow(rgb, aspect="auto")
    for i in range(3):
        for j in range(3):
            dark = float(np.dot(rgb[i, j], [0.2126, 0.7152, 0.0722])) < 0.45
            ax.text(j, i, f"{counts[i, j]}\n({percent[i, j]:.0f}%)", ha="center", va="center", fontsize=FONT_PT, color="white" if dark else "black")
    ax.set_xticks(range(3), PROFILES)
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_yticks(range(3), [f"{g} (n = {counts[i].sum()})\n{GLUCOSE_DESCRIPTOR[g]}, {medians[g]:.0f} mg/dL" for i, g in enumerate(table.index)])
    ax.set_xlabel("Clinical profile", fontsize=FONT_PT)
    ax.set_ylabel("CGM group", fontsize=FONT_PT)
    ax.tick_params(labelsize=FONT_PT, length=0)
    ax.set_title(f"Clinical profile composition\nof CGM groups\nARI {ari.value:.3f} [{ari.ci_low:.3f}, {ari.ci_high:.3f}]", fontsize=TITLE_PT, fontweight="bold")
    box = ax.get_position()
    cax = fig.add_axes([box.x1 + 0.012, box.y0, 0.03, box.height])
    levels = np.linspace(0, PERCENT_MAX, 121)
    cax.imshow(np.stack([[_shade(c, v) for c in colors] for v in levels]), aspect="auto", origin="lower", extent=(-0.5, 2.5, 0, PERCENT_MAX))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_yticks(range(0, PERCENT_MAX + 1, 20))
    cax.set_ylabel("Participants within CGM group (%)", fontsize=FONT_PT)
    cax.tick_params(labelsize=FONT_PT, length=2, width=style.SPINE_LINEWIDTH)
    for spine in cax.spines.values():
        spine.set_linewidth(style.SPINE_LINEWIDTH)
    return {"counts": counts.tolist(), "row_percent": percent.tolist()}


def _panel_b(ax, ax_h, b3: pd.DataFrame) -> dict:
    others = [(f, label) for f, label in VARIABLES if f != HBA1C]
    y = np.arange(len(others))[::-1]
    plotted = {}
    for yy, (factor, _) in zip(y, others):
        r = b3.loc[factor]
        _interval(ax, r.eta_squared, yy, r.ci_low, r.ci_high, True, fmt="o", color=BLACK, markersize=3.5)
        plotted[factor] = [r.eta_squared, r.ci_low, r.ci_high]
    ax.set_yticks(y, [label for _, label in others])
    ax.set_xlim(0, 0.13)
    ax.set_ylim(-0.6, len(others) - 0.4)
    ax.set_title("Clinical differences\nacross CGM groups", fontsize=TITLE_PT, fontweight="bold")
    _axes_style(ax)
    r = b3.loc[HBA1C]
    _interval(ax_h, r.eta_squared, 0, r.ci_low, r.ci_high, True, fmt="s", color=BLACK, markersize=3.5)
    plotted[HBA1C] = [r.eta_squared, r.ci_low, r.ci_high]
    ax_h.set_yticks([0], ["HbA1c"])
    ax_h.set_ylim(-0.8, 0.8)
    ax_h.set_xlim(0.30, 0.55)
    ax_h.set_xlabel(ETA_LABEL, fontsize=FONT_PT)
    ax_h.text(1.0, 1.12, "separate scale, 0.30 to 0.55", transform=ax_h.transAxes, ha="right", va="bottom", fontsize=FONT_PT, color=GREY, style="italic")
    _axes_style(ax_h)
    ax.set_xlabel(ETA_LABEL, fontsize=FONT_PT)
    return plotted


def _classification_rows(b5: pd.DataFrame, metric: str):
    cls = b5[b5.target == "C1/C2/C3"]
    ref = cls[(cls.feature_set == "HbA1c alone") & (cls.metric == metric)].iloc[0]
    cgm = cls[(cls.feature_set == "extended") & (cls.window == "window_full") & (cls.metric == metric)].iloc[0]
    gain = cls[(cls.feature_set == "extended") & (cls.window == "window_full") & cls.metric.str.startswith(f"{metric} gain")].iloc[0]
    return ref, cgm, gain


def _panel_c(ax, b5: pd.DataFrame) -> dict:
    hba1c, cgm, gain = _classification_rows(b5, "macro_auroc")
    combined = gain.hba1c_plus_cgm
    _interval(ax, 0, hba1c.value, hba1c.ci_low, hba1c.ci_high, False, fmt="s", color=GREY, markersize=4.5)
    _interval(ax, 1, cgm.value, cgm.ci_low, cgm.ci_high, False, fmt="o", color=BLACK, markersize=4.5)
    ax.plot(2, combined, "D", color=BLACK, markersize=4.5, zorder=4)
    for x, value in ((0, hba1c.value), (1, cgm.value)):
        ax.text(x + 0.12, value, f"{value:.3f}", fontsize=FONT_PT, va="center", ha="left")
    ax.text(2, combined - 0.014, f"{combined:.3f}\n(no stored\nCI)", fontsize=FONT_PT, va="top", ha="center")
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=style.REFERENCE_LINEWIDTH)
    ax.text(2.45, 0.504, "Chance (0.50)", fontsize=FONT_PT, color=GREY, ha="right", va="bottom")
    top = 0.725
    ax.plot([0, 0, 2, 2], [hba1c.ci_high + 0.012, top, top, combined + 0.012], color=BLACK, linewidth=0.7)
    ax.text(1, top + 0.006, f"Paired gain in AUROC\n+{gain.value:.3f} [{gain.ci_low:.3f}, {gain.ci_high:.3f}]", ha="center", va="bottom",
            fontsize=FONT_PT, fontweight="bold")
    ax.set_xticks(range(3), ["HbA1c\nalone", "CGM\nalone", "HbA1c\n+ CGM"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.46, 0.80)
    ax.set_ylabel("Macro AUROC (cross-validated)", fontsize=FONT_PT)
    ax.set_title("Prediction of clinical\nprofiles", fontsize=TITLE_PT, fontweight="bold")
    _axes_style(ax, "y")
    return {"hba1c": [hba1c.value, hba1c.ci_low, hba1c.ci_high], "cgm": [cgm.value, cgm.ci_low, cgm.ci_high], "both": combined,
            "gain": [gain.value, gain.ci_low, gain.ci_high]}


def _panel_d(ax, b5: pd.DataFrame) -> dict:
    r2 = b5[(b5.direction == "CGM to clinical variable") & (b5.feature_set == "extended") & (b5.window == "window_full")].set_index("target")
    y = np.arange(len(VARIABLES))[::-1]
    plotted = {}
    for yy, (factor, _) in zip(y, VARIABLES):
        r = r2.loc[factor]
        _interval(ax, r.value, yy, r.ci_low, r.ci_high, True, fmt="s" if factor == HBA1C else "o", color=BLACK, markersize=3.5)
        plotted[factor] = [r.value, r.ci_low, r.ci_high]
    ax.axvline(0, color=BLACK, linewidth=style.REFERENCE_LINEWIDTH)
    ax.set_yticks(y, [label for _, label in VARIABLES])
    ax.set_xlim(-0.08, 0.72)
    ax.set_xlabel("Cross-validated R² [95% CI]", fontsize=FONT_PT)
    ax.set_title("Prediction of clinical\nvariables from CGM", fontsize=TITLE_PT, fontweight="bold")
    _axes_style(ax)
    return plotted


def _panel_letters(fig, panels) -> None:
    """Letter at the left edge of each panel's y-axis labels, level with the top of its title."""
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    for ax, letter in panels:
        x0 = max(to_fig.transform(ax.yaxis.get_tightbbox(renderer).p0)[0], 0.005)
        y1 = to_fig.transform(ax.title.get_window_extent(renderer).p1)[1]
        fig.text(x0, y1, letter, fontsize=LETTER_PT, fontweight="bold", va="top", ha="left")


def _setup(height_cm: float):
    style.setup_base()
    plt.rcParams.update({"font.size": FONT_PT})
    return plt.figure(figsize=(WIDTH_CM / style.CM_PER_INCH, height_cm / style.CM_PER_INCH), facecolor="white")


def _check_fonts(fig) -> None:
    if style.smallest_font(fig) < MIN_FONT_PT:
        raise ValueError(f"Font below {MIN_FONT_PT} pt")


def draw_correspondence(contingency: pd.DataFrame, groups: pd.DataFrame, ari: pd.Series, b3: pd.DataFrame, b5: pd.DataFrame) -> tuple[plt.Figure, dict]:
    """contingency: G x C1..C3 counts (primary, all); groups: primary group summary with median_mean_glucose;
    b3: primary-partition clinical eta squared (all participants), indexed by factor."""
    fig = _setup(HEIGHT_CM)
    grid = fig.add_gridspec(2, 2, left=0.205, right=0.975, top=0.89, bottom=0.085, wspace=1.15, hspace=0.62, width_ratios=[1.0, 1])
    ax_a = fig.add_subplot(grid[0, 0])
    sub = grid[0, 1].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.95)
    ax_b, ax_h = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    ax_c, ax_d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    plotted = {"A": _panel_a(fig, ax_a, contingency, groups.set_index("cluster").median_mean_glucose, ari), "B": _panel_b(ax_b, ax_h, b3),
               "C": _panel_c(ax_c, b5), "D": _panel_d(ax_d, b5)}
    _panel_letters(fig, ((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")))
    _check_fonts(fig)
    return fig, plotted


def draw_supplement(b5: pd.DataFrame, nulls: pd.DataFrame, b6: pd.DataFrame, auroc_threshold: float) -> tuple[plt.Figure, dict]:
    """A: classification against the permutation null; B: pooled-cohort ARI with diagnosis and profiles."""
    fig = _setup(SUPP_HEIGHT_CM)
    grid = fig.add_gridspec(1, 2, left=0.09, right=0.975, top=0.80, bottom=0.22, wspace=0.32)
    ax_a, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    plotted, ticks = {"A": {}, "B": {}}, []
    for x, (metric, label, chance) in enumerate((("balanced_accuracy", "Balanced accuracy", 1 / 3), ("macro_auroc", "Macro AUROC", 0.5))):
        parts = ax_a.violinplot(nulls[nulls.analysis == f"C1/C2/C3 {metric}"].value.to_numpy(), positions=[x], widths=0.5, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(GREY)
            body.set_alpha(0.35)
        ref, cgm, gain = _classification_rows(b5, metric)
        _interval(ax_a, x - 0.12, ref.value, ref.ci_low, ref.ci_high, False, fmt="s", color=GREY, markersize=3.5, label="HbA1c alone" if x == 0 else None)
        _interval(ax_a, x + 0.04, cgm.value, cgm.ci_low, cgm.ci_high, False, fmt="o", color=BLACK, markersize=3.5, label="CGM alone" if x == 0 else None)
        ax_a.plot(x + 0.2, gain.hba1c_plus_cgm, "D", color=BLACK, markerfacecolor="white", markersize=3.5, label="HbA1c + CGM" if x == 0 else None)
        ax_a.hlines(chance, x - 0.3, x + 0.3, color=BLACK, linewidth=style.REFERENCE_LINEWIDTH, linestyle=":")
        ticks.append(f"{label}\np = {cgm.perm_p:.3f}")
        plotted["A"][metric] = {"cgm": [cgm.value, cgm.ci_low, cgm.ci_high], "hba1c": [ref.value, ref.ci_low, ref.ci_high], "both": gain.hba1c_plus_cgm,
                                "p": cgm.perm_p}
    ax_a.hlines(auroc_threshold, 0.65, 1.35, color=GREY, linestyle=style.REFERENCE_STYLE, linewidth=style.REFERENCE_LINEWIDTH)
    ax_a.text(1.37, auroc_threshold, f"{auroc_threshold:.2f}", color=GREY, fontsize=FONT_PT, va="center")
    ax_a.set_xticks([0, 1], ticks)
    ax_a.set_xlim(-0.5, 1.65)
    ax_a.set_ylim(0.22, 0.80)
    ax_a.set_ylabel("Cross-validated score", fontsize=FONT_PT)
    ax_a.set_xlabel("Permutation p, CGM alone", fontsize=FONT_PT)
    ax_a.set_title("Prediction of C1 to C3\nagainst the permutation null", fontsize=TITLE_PT, fontweight="bold")
    ax_a.legend(fontsize=FONT_PT, loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    _axes_style(ax_a)
    b6 = b6[b6.metric == "ARI"]
    targets = [("diagnostic stratum (4)", "Diagnosis (4 strata)", "o", BLACK), ("stratum-profile label (10)", "Stratum and profile (10)", "s", GREY),
               ("C1/C2/C3 inside the 458", f"C1 to C3, {T2D_SHORT} only", "D", T2D_GREEN)]
    ks = sorted(b6.pooled_k.unique())
    for i, (target, label, marker, color) in enumerate(targets):
        sub = b6[b6.target == target].set_index("pooled_k").loc[ks]
        ax_b.errorbar(np.arange(len(ks)) + (i - 1) * 0.24, sub.value, yerr=[sub.value - sub.ci_low, sub.ci_high - sub.value], fmt=marker, color=color,
                      markersize=3.5, markerfacecolor="white" if marker == "D" else color, elinewidth=style.CI_LINEWIDTH, capsize=0, label=label)
        plotted["B"][target] = sub[["value", "ci_low", "ci_high"]].to_numpy().tolist()
    within = b6[b6.target.str.contains("within-stratum")].set_index("pooled_k").loc[ks]
    ax_b.scatter(np.arange(len(ks)), within.null_mean, marker="_", s=70, color=BLACK, zorder=4, label="Within-stratum null mean")
    ax_b.set_xticks(range(len(ks)), [f"k = {k}" for k in ks])
    ax_b.set_ylim(0, 0.30)
    ax_b.set_ylabel("ARI [95% CI]", fontsize=FONT_PT)
    ax_b.set_title("Pooled cohort (n = 1,591): CGM clusters\nvs diagnosis and profile", fontsize=TITLE_PT, fontweight="bold")
    ax_b.legend(fontsize=FONT_PT, loc="upper right", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    _axes_style(ax_b)
    _panel_letters(fig, ((ax_a, "A"), (ax_b, "B")))
    _check_fonts(fig)
    return fig, plotted


CAPTION_CORRESPONDENCE = (
    r"\caption{Correspondence between clinical profiles and CGM-defined groups in " + T2D_LONG.replace("type 2", "type~2") + r" ($N=458$; full CGM record). "
    r"CGM groups G1--G3 come from $k$-means ($k=3$) on extended CGM features; row labels give each group's size and median participant mean glucose. "
    r"(A)~Clinical profile composition of each CGM group: cells show counts and, in parentheses, percentages within the CGM group; each column is shaded "
    r"in its profile colour from white (0\%) to full colour (60\%); agreement between the two partitions is given as the adjusted Rand index (ARI) with "
    r"95\% bootstrap CI. (B)~Kruskal--Wallis rank-based effect sizes ($\eta^2$, 95\% bootstrap CI) for clinical variables across CGM groups; HbA1c is shown "
    r"on a separate scale. (C)~Cross-validated macro AUROC for predicting C1--C3 from HbA1c alone, CGM features alone, and HbA1c plus CGM; bars are 95\% "
    r"bootstrap CIs, the bracket gives the paired bootstrap gain of HbA1c plus CGM over HbA1c alone, and no interval was stored for the combined model "
    r"itself. The dotted line marks chance (0.50). (D)~Cross-validated $R^2$ (95\% CI) for predicting each clinical variable from CGM features; the "
    r"vertical line marks $R^2=0$.}")
CAPTION_SUPPLEMENT = (
    r"\caption{Supplementary reverse-direction results. (A)~Cross-validated balanced accuracy and macro AUROC for predicting C1--C3 in "
    + T2D_LONG.replace("type 2", "type~2") + r" ($N=458$, full CGM record) from HbA1c alone (grey squares), CGM features alone (black circles; 95\% "
    r"bootstrap CI) and HbA1c plus CGM (open diamonds; point estimate only), against the permutation null for CGM alone (grey densities, 200 "
    r"permutations). Dotted lines mark chance; the dashed line marks the prespecified 0.65 AUROC criterion. (B)~Pooled cohort ($n=1{,}591$, all four "
    r"diagnostic strata): ARI (95\% CI) between pooled CGM clusters ($k=2$--4) and diagnostic stratum, stratum-specific clinical profile label, and C1--C3 "
    r"within the non-insulin type~2 diabetes stratum (green); black bars mark the mean of the within-stratum permutation null.}")
