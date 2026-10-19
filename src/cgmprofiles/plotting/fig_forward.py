"""Figure 1: CGM outcomes across clinical profiles, first 48 hours (A to C) and full record (D to F).

Violins (seaborn, kept so the density shapes match the manuscript), jittered participant points,
medians and IQRs, and the Kruskal-Wallis rank eta squared of each panel. The participant values
are read from restricted/; the figure itself is tracked (individual points, no identifiers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection

from . import style
from .style import plt

WINDOW_LABELS = {"window_48h": "First 48 hours", "window_full": "Full CGM record"}
OUTCOMES = [("mean_glucose", "Mean glucose", "Mean glucose (mg/dL)"), ("time_in_range_pct", "Time in range", "Time in range (%)"),
            ("coefficient_of_variation_pct", "Coefficient of variation", "Coefficient of variation (%)")]
LETTERS = np.array(list("ABCDEF")).reshape(2, 3)
JITTER_SEED = 42                 # point positions: seed + 10 x row + column, as in the record
FIGURE_SIZE_MM = (220.0, 125.0)
VIOLIN_ALPHA = 0.52
VIOLIN_WIDTH = 0.80
VIOLIN_BW_ADJUST = 0.80
JITTER_WIDTH = 0.17
POINT_SIZE = style.POINT_MARKER_SIZE * 0.24
MEDIAN_SIZE = style.POINT_MARKER_SIZE * 0.72
IQR_LINEWIDTH = style.CI_LINEWIDTH * 1.35
COLOR_ANNOTATION = style.COLOR_NULL


def _limits(metrics: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for metric, _, _ in OUTCOMES:
        values = metrics[metric].dropna().to_numpy()
        pad = max(0.05 * float(values.max() - values.min()), 1.0)
        out[metric] = (float(values.min() - pad), float(values.max() + pad))
    return out


def _panel(ax, frame: pd.DataFrame, metric: str, effect: float, limits, seed: int, names: dict[int, str]) -> dict:
    clusters = sorted(names)
    order = [names[c] for c in clusters]
    palette = {names[c]: style.PROFILE_COLORS[f"C{c}"] for c in clusters}
    before = len(ax.collections)
    sns.violinplot(data=frame, x="cluster_label", y=metric, order=order, hue="cluster_label", hue_order=order, palette=palette,
                   legend=False, dodge=False, inner=None, cut=0, width=VIOLIN_WIDTH, bw_adjust=VIOLIN_BW_ADJUST, density_norm="width",
                   linecolor=style.SPINE_COLOR, linewidth=style.SPINE_LINEWIDTH, saturation=1.0, ax=ax, zorder=1)
    for collection in ax.collections[before:]:
        if isinstance(collection, PolyCollection):
            collection.set_alpha(VIOLIN_ALPHA)
            collection.set_edgecolor(style.SPINE_COLOR)
            collection.set_linewidth(style.SPINE_LINEWIDTH)
    rng = np.random.default_rng(seed)
    plotted = {}
    for x, cluster in enumerate(clusters):
        values = frame.loc[frame.label.eq(cluster), metric].dropna().to_numpy()
        ax.scatter(x + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(values)), values, s=POINT_SIZE, color=style.PROFILE_COLORS[f"C{cluster}"],
                   alpha=style.SHADE_ALPHA_LOW, edgecolors="none", linewidths=0, rasterized=True, zorder=2)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.vlines(x, q1, q3, color=style.COLOR_ZERO, linewidth=IQR_LINEWIDTH, zorder=4)
        ax.scatter([x], [median], s=MEDIAN_SIZE, facecolor="white", edgecolor=style.COLOR_ZERO, linewidth=style.SPINE_LINEWIDTH, zorder=5)
        plotted[f"C{cluster}"] = {"n": len(values), "median": float(median), "q1": float(q1), "q3": float(q3)}
    ax.set_ylim(*limits)
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(order)), order, fontsize=style.TICK_PT)
    ax.text(0.97, 0.96, rf"$\eta^2 = {effect:.3f}$", transform=ax.transAxes, ha="right", va="top", fontsize=style.ANNOTATION_PT,
            color=COLOR_ANNOTATION, zorder=6)
    style.boxed_axes(ax)
    return {"eta_squared": effect, **plotted}



def draw(metrics: pd.DataFrame, effects: pd.DataFrame, title: str) -> tuple[plt.Figure, dict]:
    """metrics: T2D participant rows (label, window, three outcomes); effects: window, metric, eta_squared."""
    counts = metrics[metrics.window == "window_full"].label.value_counts().sort_index()
    names = {int(c): f"C{int(c)} (n={int(n)})" for c, n in counts.items()}
    metrics = metrics.assign(cluster_label=pd.Categorical(metrics.label.map(names), categories=[names[c] for c in sorted(names)], ordered=True))
    effect = {(r.window, r.metric): r.eta_squared for r in effects.itertuples()}
    limits = _limits(metrics)
    sns.set_style("whitegrid")
    style.setup_base()
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_SIZE_MM[0] / 25.4, FIGURE_SIZE_MM[1] / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.84, wspace=0.28, hspace=0.40)
    plotted = {}
    for row, window in enumerate(WINDOW_LABELS):
        frame = metrics[metrics.window == window]
        for col, (metric, column_title, y_label) in enumerate(OUTCOMES):
            ax = axes[row, col]
            plotted[(window, metric)] = _panel(ax, frame, metric, effect[(window, metric)], limits[metric], JITTER_SEED + row * 10 + col, names)
            ax.set_ylabel(y_label, fontsize=style.AXIS_LABEL_PT)
            if row == 0:
                ax.set_title(column_title, fontsize=style.PANEL_TITLE_PT + 1.0, fontweight="bold", loc="center", pad=5)
            ax.text(-0.10, 1.04, LETTERS[row, col], transform=ax.transAxes, ha="left", va="bottom", fontsize=style.PANEL_TITLE_PT + 1.0,
                    fontweight="bold", color=style.COLOR_ZERO, clip_on=False)
    fig.suptitle(title, fontsize=style.PANEL_TITLE_PT + 3.0, fontweight="bold", y=0.965)
    for label, y in zip(WINDOW_LABELS.values(), [0.655, 0.295]):
        fig.text(0.018, y, label, ha="center", va="center", rotation=90, fontsize=style.AXIS_LABEL_PT, fontweight="bold", color=style.COLOR_ZERO)
    return fig, plotted
