"""Figure 1, print version (ported from the record's plot_fig1_raw_profiles_paper.py): CGM outcomes across
C1 to C3, full record (A to C) above the first 48 hours (D to F), at 16 cm and 8 pt, fixed y limits per
outcome, no title and no eta squared annotations (Table 2 reports them). Same participant values as Fig 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde

from . import print_style as ps
from . import style
from .style import plt

WINDOWS = [("window_full", "Full record"), ("window_48h", "First 48 hours")]
OUTCOMES = [("mean_glucose", "Mean glucose", "Mean glucose (mg/dL)", (75, 350), 50),
            ("time_in_range_pct", "Time in range", "Time in range (%)", (-5, 105), 20),
            ("coefficient_of_variation_pct", "Glucose variability", "Coefficient of variation (%)", (5, 45), 10)]
CLUSTERS = [1, 2, 3]
HEIGHT_CM = 12.0
JITTER_SEED = 42            # seed + 10 x row + column, as in the record
VIOLIN_ALPHA, VIOLIN_HALF_WIDTH, KDE_BW = 0.22, 0.38, 0.8
POINT_ALPHA, POINT_SIZE, JITTER = 0.30, 1.6, 0.15
IQR_HALF_WIDTH, MEDIAN_HALF_WIDTH = 0.075, 0.16


def _violin(ax, x: float, values: np.ndarray, color: str) -> None:
    """Width-normalized KDE cut at the observed range (seaborn cut=0, density_norm='width')."""
    kde = gaussian_kde(values, bw_method=lambda k: k.scotts_factor() * KDE_BW)
    grid = np.linspace(values.min(), values.max(), 256)
    density = kde(grid)
    half = VIOLIN_HALF_WIDTH * density / density.max()
    ax.fill_betweenx(grid, x - half, x + half, facecolor=color, alpha=VIOLIN_ALPHA, edgecolor="none", zorder=1)
    ax.plot(np.r_[x - half, (x + half)[::-1]], np.r_[grid, grid[::-1]], color=color, linewidth=0.6, alpha=0.7, zorder=1)


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
        ax.add_patch(plt.Rectangle((x - IQR_HALF_WIDTH, q1), 2 * IQR_HALF_WIDTH, q3 - q1, facecolor="white", edgecolor=ps.INK, linewidth=0.9, zorder=4))
        ax.hlines(median, x - MEDIAN_HALF_WIDTH, x + MEDIAN_HALF_WIDTH, color=ps.INK, linewidth=2.0, zorder=5)
        plotted[f"C{cluster}"] = {"n": len(values), "q1": float(q1), "median": float(median), "q3": float(q3)}
    if frame[metric].min() < limits[0] or frame[metric].max() > limits[1]:
        raise ValueError(f"{metric} outside the fixed axis range {limits}")
    ax.set_ylim(*limits)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlim(-0.6, len(CLUSTERS) - 0.4)
    ax.set_xticks(range(len(CLUSTERS)), [f"C{c}" for c in CLUSTERS])
    ps.axes_style(ax, grid_axis="y")
    ax.tick_params(axis="x", length=0)
    return plotted


def draw(metrics: pd.DataFrame) -> tuple[plt.Figure, dict]:
    """metrics: T2D participant rows with label, window and the three outcomes."""
    ps.setup()
    fig = ps.figure(HEIGHT_CM)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.925, bottom=0.06, wspace=0.50, hspace=0.30)
    plotted = {}
    for i, (window, _) in enumerate(WINDOWS):
        frame = metrics[metrics.window.eq(window)]
        for j, (metric, title, ylabel, limits, step) in enumerate(OUTCOMES):
            ax = axes[i, j]
            plotted[(window, metric)] = _panel(ax, frame, metric, limits, step, JITTER_SEED + 10 * i + j)
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(title, fontsize=ps.TITLE_PT, fontweight="bold", pad=8)
            ps.letter(fig, ax, "ABCDEF"[3 * i + j], dx_cm=1.25)
    for i, (_, row_label) in enumerate(WINDOWS):
        box = axes[i, 0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, row_label, rotation=90, ha="left", va="center", fontsize=ps.TITLE_PT, fontweight="bold")
    ps.check_fonts(fig)
    return fig, plotted
