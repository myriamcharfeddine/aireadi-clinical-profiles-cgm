"""Print style for the 16 cm print versions of Figures 1 to 3 (ported from the record's
paper1_figure_style.py). Profile colours C1 to C3 as in the manuscript figures; each glycemic
stratum has its config colour and its own marker; every other role is neutral grey or black.
"""
from __future__ import annotations

from . import style
from .style import plt

WIDTH_CM = 16.0
FONT_PT, LABEL_PT, TITLE_PT, LETTER_PT = 8.0, 8.0, 8.5, 10.0
MIN_FONT_PT = 7.0
INK = "#1A1A1A"
GRID_COLOR, GRID_LINEWIDTH = "#E3E3E3", 0.5
SPINE_COLOR, SPINE_LINEWIDTH = "#4D4D4D", 0.6
CI_LINEWIDTH, ZERO_LINEWIDTH, CHANCE_STYLE = 0.9, 0.8, ":"
MARKER_SIZE = 4.5
STRATA = ["healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"]
STRATUM_MARKER = {"healthy": "o", "pre_diabetes": "s", "t2d_oral_non_insulin": "D", "insulin_dependent": "^"}


def setup() -> None:
    """rcParams for 8 pt print text."""
    style.setup_print(FONT_PT, TITLE_PT)
    plt.rcParams.update({"axes.labelsize": LABEL_PT})


def figure(height_cm: float):
    """A 16 cm wide white figure."""
    return plt.figure(figsize=(WIDTH_CM / style.CM_PER_INCH, height_cm / style.CM_PER_INCH), facecolor="white")


def axes_style(ax, grid_axis: str = "both") -> None:
    """White background, light gridlines, left and bottom spines only."""
    ax.set_facecolor("white")
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(labelsize=FONT_PT, width=SPINE_LINEWIDTH, length=2.5, color=SPINE_COLOR)


def letter(fig, ax, text: str, dx_cm: float = 1.3, dy_cm: float = 0.35) -> None:
    """Panel letter a fixed physical distance left of and above the axes' top-left corner."""
    box = ax.get_position()
    w, h = fig.get_size_inches() * style.CM_PER_INCH
    fig.text(box.x0 - dx_cm / w, box.y1 + dy_cm / h, text, fontsize=LETTER_PT, fontweight="bold", ha="left", va="bottom")


def fig_text(fig, height_cm: float, x_cm: float, y_cm: float, text: str, bold: bool = False, size: float = FONT_PT, **kw):
    """Text at an absolute position in cm from the bottom-left corner."""
    return fig.text(x_cm / WIDTH_CM, y_cm / height_cm, text, fontsize=size, fontweight="bold" if bold else "normal",
                    ha=kw.pop("ha", "left"), va=kw.pop("va", "center"), color=INK, **kw)


def box(height_cm: float, x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Axes rectangle from cm coordinates (bottom-left origin)."""
    return [x0 / WIDTH_CM, y0 / height_cm, (x1 - x0) / WIDTH_CM, (y1 - y0) / height_cm]


def check_fonts(fig) -> None:
    if style.smallest_font(fig) < MIN_FONT_PT:
        raise ValueError(f"Font below {MIN_FONT_PT} pt")
