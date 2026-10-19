"""Supplementary Figure S1 (temporal reproducibility, v2 three-panel design).

A: ARI between held-out participants' CGM groups in days 1 to 3 and days 4 to 6, and between CGM
groups and C1 to C3 in each window, against permutation nulls, the full-record refit stability and
the 0.10 threshold; B: ICC between windows per CGM feature, grouped by level, variability, and timing
and daily pattern (3-day windows filled, 4-day windows open); C: C2 minus rest CV in each window and
on the full record of the same participants. Within the monitoring period, not long-term stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import style
from .style import plt

WIDTH_CM, HEIGHT_CM = 16.0, 13.0
FONT_PT, TITLE_PT, LETTER_PT = 7.5, 8.0, 10.0
COLOR_REFERENCE = style.NEUTRAL_DARK
COLOR_MUTED = style.COLOR_NULL
COLOR_C2 = style.PROFILE_COLORS["C2"]
COLOR_INK = "#1A1A1A"
COLOR_LIGHT = "#C8C8C8"
LINE_W, MARKER = 0.9, 4.2
MINUS = "−"
FEATURE_GROUPS = {
    "Level": ["mean_glucose", "time_in_range_pct", "tar_180_pct", "tar_250_pct"],
    "Variability": ["sd_glucose", "coefficient_of_variation_pct", "mage", "tbr_70_pct"],
    "Timing and daily pattern": ["acf_1h", "acf_3h", "acf_24h", "profile_pc1", "profile_pc2", "profile_pc3", "profile_pc4", "circadian_power_share"],
}
FEATURE_LABELS = {
    "mean_glucose": "Mean glucose", "time_in_range_pct": "TIR 70 to 180", "tar_180_pct": "TAR > 180", "tar_250_pct": "TAR > 250",
    "sd_glucose": "SD", "coefficient_of_variation_pct": "CV", "mage": "MAGE", "tbr_70_pct": "TBR < 70",
    "acf_1h": "Autocorrelation 1 h", "acf_3h": "Autocorrelation 3 h", "acf_24h": "Autocorrelation 24 h",
    "profile_pc1": "Daily profile PC1", "profile_pc2": "Daily profile PC2", "profile_pc3": "Daily profile PC3",
    "profile_pc4": "Daily profile PC4", "circadian_power_share": "Circadian share",
}


def _fmt(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}".replace("-", MINUS)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=TITLE_PT, fontweight="bold", pad=6)


def _letter(fig, ax, text: str, dx_cm: float) -> None:
    box = ax.get_position()
    w, h = fig.get_size_inches() * style.CM_PER_INCH
    fig.text(box.x0 - dx_cm / w, box.y1 + 0.32 / h, text, fontsize=LETTER_PT, fontweight="bold", ha="left", va="bottom")


def _panel_a(ax, rows: list[dict], stability: float, ari_threshold: float) -> None:
    y = np.arange(len(rows))[::-1].astype(float)
    for yi, r in zip(y, rows):
        ax.plot([0, r["null"]], [yi, yi], color=COLOR_LIGHT, linewidth=5.5, solid_capstyle="butt", zorder=1)
        ax.hlines(yi, r["low"], r["high"], color=r["color"], linewidth=LINE_W, zorder=3)
        ax.plot(r["value"], yi, "o", color=r["color"], markersize=MARKER, zorder=4)
        ax.text(max(r["high"] + 0.025, ari_threshold + 0.02), yi, _fmt(r["value"], r["decimals"]), va="center", ha="left", fontsize=FONT_PT,
                color=r["color"] if r["note"] else COLOR_INK)
        if r["note"]:
            ax.text(r["value"], yi - 0.34, r["note"], va="top", ha="center", fontsize=FONT_PT, color=r["color"])
    ax.axvline(stability, color=COLOR_MUTED, linestyle="--", linewidth=0.8, zorder=2)
    ax.text(stability + 0.012, y.min() - 0.55, "Refit stability", ha="left", va="center", fontsize=FONT_PT, color=COLOR_MUTED)
    ax.axvline(ari_threshold, color=COLOR_MUTED, linestyle=":", linewidth=0.8, zorder=2)
    ax.text(ari_threshold + 0.012, y.min() - 0.55, "Threshold", ha="left", va="center", fontsize=FONT_PT, color=COLOR_MUTED)
    ax.plot([], [], color=COLOR_LIGHT, linewidth=5.5, solid_capstyle="butt", label="Permutation null, 97.5th percentile")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False, handlelength=1.2, borderaxespad=0.1, fontsize=FONT_PT)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_ylim(y.min() - 0.85, y.max() + 0.5)
    ax.set_xlabel("Adjusted Rand index (ARI)")
    style.open_axes(ax, "x", FONT_PT)
    ax.tick_params(axis="y", length=0)
    _title(ax, "CGM groups agree over time, not with clinical profiles")


def _panel_b(ax, icc: pd.DataFrame) -> None:
    labels, positions, separators, headers, y = [], [], [], [], 0.0
    for group, features in FEATURE_GROUPS.items():
        headers.append((group, y))
        y -= 1
        for f in icc.loc[features].sort_values("icc", ascending=False).index:
            r = icc.loc[f]
            ax.hlines(y, r.ci_low, r.ci_high, color=COLOR_INK, linewidth=LINE_W)
            ax.plot(r.icc, y, "o", color=COLOR_INK, markersize=MARKER, zorder=3)
            ax.plot(r.icc_days4, y, "o", markerfacecolor="white", markeredgecolor=COLOR_MUTED, markersize=MARKER, zorder=4)
            labels.append(FEATURE_LABELS[f])
            positions.append(y)
            y -= 1
        separators.append(y + 0.5)
    for s in separators[:-1]:
        ax.axhline(s, color=COLOR_LIGHT, linewidth=0.5)
    header_names = [g for g, _ in headers]
    ax.set_yticks(positions + [yy for _, yy in headers], labels + header_names)
    for tick in ax.get_yticklabels():
        if tick.get_text() in header_names:
            tick.set_color(COLOR_MUTED)
            tick.set_fontstyle("italic")
    ax.set_ylim(min(positions) - 0.7, 2.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ICC between windows")
    ax.plot([], [], "o", color=COLOR_INK, markersize=MARKER, label="3-day windows")
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor=COLOR_MUTED, markersize=MARKER, label="4-day windows")
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.1, handletextpad=0.2, fontsize=FONT_PT, ncol=2, columnspacing=0.8)
    style.open_axes(ax, "x", FONT_PT)
    ax.tick_params(axis="y", length=0)
    _title(ax, "Glucose level is more\nreproducible than timing")


def _panel_c(ax, points: list[dict], change: dict) -> None:
    xs = np.arange(len(points), dtype=float)
    ax.axhline(0, color=COLOR_INK, linewidth=0.8)
    for x, p in zip(xs, points):
        ax.vlines(x, p["low"], p["high"], color=p["color"], linewidth=LINE_W)
        ax.plot(x, p["value"], "o", color=p["color"], markersize=MARKER + 0.6, zorder=4)
        ax.text(x - 0.09, p["value"], _fmt(p["value"], 2), ha="right", va="center", fontsize=FONT_PT, color=COLOR_INK)
    ax.text(0.03, 0.97, f"Change between windows\n{_fmt(change['value'], 2)} [{_fmt(change['low'], 2)}, {_fmt(change['high'], 2)}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=FONT_PT, color=COLOR_MUTED)
    ax.set_xticks(xs, [p["label"] for p in points])
    ax.set_xlim(-0.75, len(points) - 0.4)
    ax.set_ylim(-5.0, 1.2)
    ax.set_ylabel("CV, C2 minus other profiles\n(percentage points)")
    style.open_axes(ax, "y", FONT_PT)
    _title(ax, "The lower variability of C2\nis stable across windows")


def draw(t2_heldout: dict, t3_ari: pd.Series, icc: pd.DataFrame, contrast: pd.Series, reference: pd.Series,
         stability: float, ari_threshold: float) -> tuple[plt.Figure, dict]:
    """t2_heldout: {'ARI': row, 'kappa': row} (held-out, frozen pipeline); t3_ari: ARI with C1 to C3 per window;
    icc: per feature (clustering scale, all eligible) with icc_days4; contrast / reference: C2 minus rest CV."""
    ari, kappa = t2_heldout["ARI"], t2_heldout["kappa"]
    rows = [
        {"label": "CGM groups, days 1 to 3 vs\ndays 4 to 6 (held-out)", "value": ari.value, "low": ari.ci_low, "high": ari.ci_high,
         "null": ari.null_q975, "color": COLOR_REFERENCE, "decimals": 2,
         "note": f"kappa {_fmt(kappa.value, 2)} [{_fmt(kappa.ci_low, 2)}, {_fmt(kappa.ci_high, 2)}]"},
        {"label": "CGM groups vs C1/C2/C3,\ndays 1 to 3", "value": t3_ari.A_value, "low": t3_ari.A_ci_low, "high": t3_ari.A_ci_high,
         "null": t3_ari.A_null_q975, "color": COLOR_MUTED, "decimals": 3, "note": ""},
        {"label": "CGM groups vs C1/C2/C3,\ndays 4 to 6", "value": t3_ari.B_value, "low": t3_ari.B_ci_low, "high": t3_ari.B_ci_high,
         "null": t3_ari.B_null_q975, "color": COLOR_MUTED, "decimals": 3, "note": ""},
    ]
    points = [{"label": "Days\n1 to 3", "value": contrast.A_difference, "low": contrast.A_ci_low, "high": contrast.A_ci_high, "color": COLOR_C2},
              {"label": "Days\n4 to 6", "value": contrast.B_difference, "low": contrast.B_ci_low, "high": contrast.B_ci_high, "color": COLOR_C2},
              {"label": "Full\nrecord", "value": reference.difference, "low": reference.ci_low, "high": reference.ci_high, "color": COLOR_REFERENCE}]
    change = {"value": contrast.change_A_minus_B, "low": contrast.change_ci_low, "high": contrast.change_ci_high}
    style.setup_print(FONT_PT, TITLE_PT)
    fig = plt.figure(figsize=(WIDTH_CM / style.CM_PER_INCH, HEIGHT_CM / style.CM_PER_INCH), facecolor="white")
    top = fig.add_gridspec(1, 1, left=0.255, right=0.975, bottom=0.70, top=0.935)
    bottom = fig.add_gridspec(1, 2, left=0.215, right=0.985, bottom=0.095, top=0.565, wspace=0.55, width_ratios=[0.9, 1.0])
    ax_a, ax_b, ax_c = fig.add_subplot(top[0, 0]), fig.add_subplot(bottom[0, 0]), fig.add_subplot(bottom[0, 1])
    _panel_a(ax_a, rows, stability, ari_threshold)
    _panel_b(ax_b, icc)
    _panel_c(ax_c, points, change)
    for ax, text, dx in ((ax_a, "A", 3.9), (ax_b, "B", 3.3), (ax_c, "C", 1.9)):
        _letter(fig, ax, text, dx)
    if style.smallest_font(fig) < style.MIN_FONT_PT_PRINT:
        raise ValueError("Font below 7 pt in Fig S1")
    plotted = {"A": [[r["value"], r["low"], r["high"], r["null"]] for r in rows], "B": icc[["icc", "ci_low", "ci_high", "icc_days4"]].to_dict("index"),
               "C": [[p["value"], p["low"], p["high"]] for p in points], "change": change, "stability": stability}
    return fig, plotted
