"""Figure style shared by the paper figures (merges the record's figure_main_config and
paper1_figure_style). Colours and stratum labels come from config/paper1.yaml:

- profiles C1, C2, C3 keep their colours in every figure;
- each glycemic stratum has one colour (Okabe-Ito), never a profile colour and never bright red;
- everything else (forward or reverse blocks, prediction models, references) is neutral grey.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from ..config import get_config  # noqa: E402

CFG = get_config()
PROFILE_COLORS: dict[str, str] = dict(CFG.colors.profiles.raw)
STRATUM_COLORS: dict[str, str] = dict(CFG.colors.strata.raw)
STRATUM_SHORT: dict[str, str] = dict(CFG.labels.strata_short.raw)
STRATUM_LONG: dict[str, str] = dict(CFG.labels.strata_long.raw)
NEUTRAL_DARK, NEUTRAL_MID, NEUTRAL_LIGHT = CFG.colors.neutral_dark, CFG.colors.neutral_mid, CFG.colors.neutral_light
COLOR_NULL = CFG.colors.null_color
COLOR_ZERO = "#000000"


def in_sentence(label: str) -> str:
    """A short stratum label as it reads inside a sentence ('Non-insulin T2D' -> 'non-insulin T2D')."""
    return label[0].lower() + label[1:]

# Typography and lines (record values: figure_main_config and paper1_figure_style)
FONT_FAMILY = "sans-serif"
FONT_STACK = ["DejaVu Sans", "Arial", "Helvetica"]
MATHTEXT = "dejavusans"
PANEL_TITLE_PT = 6.5
AXIS_LABEL_PT = 7.5
TICK_PT = 6.5
ANNOTATION_PT = 6.5
LEGEND_PT = 7.0
LETTER_PT = 10.0
GRID_COLOR = "#D9D9D9"
GRID_LINEWIDTH = 0.5
SPINE_COLOR = "#000000"
SPINE_LINEWIDTH = 0.7
CI_LINEWIDTH = 1.3
REFERENCE_LINEWIDTH = 0.7
REFERENCE_STYLE = "--"
SHADE_ALPHA_LOW = 0.15
SHADE_ALPHA_HIGH = 0.25
POINT_MARKER_SIZE = 22
DPI_PNG = 300
CM_PER_INCH = 2.54
MIN_FONT_PT_PRINT = 7.0


def setup_base() -> None:
    """rcParams of the original summary figures (Figs 1 to 3), starting from matplotlib defaults so that
    no style (such as seaborn's, set for Fig 1) carries over from a figure drawn earlier."""
    plt.rcdefaults()
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.sans-serif": FONT_STACK, "mathtext.fontset": MATHTEXT,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def setup_base_fonts() -> None:
    """Font settings of setup_base without resetting other rcParams (applied after a seaborn style)."""
    plt.rcParams.update({"font.family": FONT_FAMILY, "font.sans-serif": FONT_STACK, "mathtext.fontset": MATHTEXT,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def setup_print(font_pt: float, title_pt: float) -> None:
    """rcParams of the 16 cm print style (Fig S1)."""
    setup_base()
    plt.rcParams.update({"font.size": font_pt, "axes.titlesize": title_pt, "axes.labelsize": font_pt, "xtick.labelsize": font_pt,
                         "ytick.labelsize": font_pt, "legend.fontsize": font_pt, "pdf.fonttype": 42, "axes.unicode_minus": True})


def boxed_axes(ax) -> None:
    """White panel, light grid, thin black spines (Figs 1 to 3)."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(labelsize=TICK_PT, width=SPINE_LINEWIDTH)


def open_axes(ax, grid_axis: str, font_pt: float) -> None:
    """Left and bottom spines only, grid on one axis (Fig S1)."""
    ax.set_facecolor("white")
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color="#E3E3E3", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#4D4D4D")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=font_pt, width=0.6, length=2.5, color="#4D4D4D")


def smallest_font(fig) -> float:
    """Smallest visible font size in a figure."""
    return min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible())


def save(fig, directory: Path, stem: str) -> list[Path]:
    """Write PDF and 300 dpi PNG with a white background."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}.pdf", directory / f"{stem}.png"]
    fig.savefig(paths[0], facecolor="white")
    fig.savefig(paths[1], dpi=DPI_PNG, facecolor="white")
    plt.close(fig)
    return paths
