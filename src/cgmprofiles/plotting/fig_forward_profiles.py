"""Figure 1, profile version (the record's fig1_raw_cgm_profiles design; its drawing script was not kept in the
record, so this module re-implements it from the same participant values as Fig 1).

Full CGM record (A to C) above the first 48 hours (D to F); kernel density shapes of each profile, light points
for participants, white IQR boxes and black median bars; y limits shared within each outcome column; 16 x 11 cm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde

from . import print_style as ps
from . import style
from .style import plt

WINDOWS = [("window_full", "Full CGM record"), ("window_48h", "First 48 hours")]
OUTCOMES = [("mean_glucose", "Mean glucose", "Mean glucose (mg/dL)", 50),
            ("time_in_range_pct", "Time in range", "Time in 70–180 mg/dL (%)", 20),
            ("coefficient_of_variation_pct", "Glucose variability", "Coefficient of variation (%)", 10)]
CLUSTERS = [1, 2, 3]
HEIGHT_CM = 11.0
JITTER_SEED = 42            # seed + 10 x row + column, as in the record's Fig 1 scripts
VIOLIN_ALPHA, VIOLIN_HALF_WIDTH, KDE_BW = 0.38, 0.38, 0.8
POINT_ALPHA, POINT_SIZE, JITTER = 0.35, 1.6, 0.15
IQR_HALF_WIDTH, MEDIAN_HALF_WIDTH = 0.075, 0.16
PAD_FRACTION, MIN_PAD = 0.05, 1.0


def _limits(metrics: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Shared y limits per outcome: observed range plus 5% (at least 1 unit) on each side."""
    out = {}
    for metric, *_ in OUTCOMES:
        values = metrics[metric].dropna().to_numpy()
        pad = max(PAD_FRACTION * float(values.max() - values.min()), MIN_PAD)
        out[metric] = (float(values.min() - pad), float(values.max() + pad))
    return out


def _violin(ax, x: float, values: np.ndarray, color: str) -> None:
    """Width-normalized KDE cut at the observed range."""
    kde = gaussian_kde(values, bw_method=lambda k: k.scotts_factor() * KDE_BW)
    grid = np.linspace(values.min(), values.max(), 256)
    density = kde(grid)
    half = VIOLIN_HALF_WIDTH * density / density.max()
    ax.fill_betweenx(grid, x - half, x + half, facecolor=color, alpha=VIOLIN_ALPHA, edgecolor="none", zorder=1)
    ax.plot(np.r_[x - half, (x + half)[::-1]], np.r_[grid, grid[::-1]], color=color, linewidth=0.6, alpha=0.5, zorder=1)


def _boxed(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.grid(True, axis="y", color=ps.GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=ps.FONT_PT, width=0.6, length=2.5)


def _panel(ax, frame: pd.DataFrame, metric: str, limits, step, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    plotted = {}
    for x, cluster in enumerate(CLUSTERS):
        color = style.PROFILE_COLORS[f"C{cluster}"]
        values = frame.loc[frame.label.eq(cluster), metric].dropna().to_numpy()
        _violin(ax, x, values, color)
        ax.scatter(x + rng.uniform(-JITTER, JITTER, len(values)), values, s=POINT_SIZE, color=color, alpha=POINT_ALPHA,
                   edgecolors="none", rasterized=True, zorder=2)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.add_patch(plt.Rectangle((x - IQR_HALF_WIDTH, q1), 2 * IQR_HALF_WIDTH, q3 - q1, facecolor="white", edgecolor="#000000", linewidth=1.0, zorder=4))
        ax.hlines(median, x - MEDIAN_HALF_WIDTH, x + MEDIAN_HALF_WIDTH, color="#000000", linewidth=2.0, zorder=5)
        plotted[f"C{cluster}"] = {"n": len(values), "q1": float(q1), "median": float(median), "q3": float(q3)}
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlim(-0.6, len(CLUSTERS) - 0.4)
    ax.set_xticks(range(len(CLUSTERS)), [f"C{c}" for c in CLUSTERS])
    _boxed(ax)
    ax.tick_params(axis="x", length=0)
    return plotted


def draw(metrics: pd.DataFrame) -> tuple[plt.Figure, dict]:
    """metrics: non-insulin T2D participant rows with label, window and the three outcomes."""
    ps.setup()
    fig = ps.figure(HEIGHT_CM)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.925, bottom=0.065, wspace=0.42, hspace=0.32)
    limits = _limits(metrics)
    plotted = {}
    for i, (window, _) in enumerate(WINDOWS):
        frame = metrics[metrics.window.eq(window)]
        for j, (metric, title, ylabel, step) in enumerate(OUTCOMES):
            ax = axes[i, j]
            plotted[(window, metric)] = _panel(ax, frame, metric, limits[metric], step, JITTER_SEED + 10 * i + j)
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(title, fontsize=ps.TITLE_PT + 0.5, fontweight="bold", pad=8)
            ps.letter(fig, ax, "ABCDEF"[3 * i + j], dx_cm=1.0)
    for i, (_, row_label) in enumerate(WINDOWS):
        box = axes[i, 0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, row_label, rotation=90, ha="left", va="center", fontsize=ps.TITLE_PT + 0.5, fontweight="bold")
    ps.check_fonts(fig)
    return fig, plotted
