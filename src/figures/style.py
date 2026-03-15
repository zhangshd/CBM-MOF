"""
Publication style constants and helpers for CBM-MOF paper figures.

Target journal: Separation and Purification Technology (Elsevier).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from dataclasses import dataclass
from pathlib import Path

# ── Journal layout constants (Elsevier / SPT) ──────────────────────────────
SINGLE_COL_INCH = 3.35          # 8.5 cm
DOUBLE_COL_INCH = 6.89          # 17.5 cm
MAX_HEIGHT_INCH = 9.19          # 23.35 cm (full page)
DPI = 300
A4_TEXT_WIDTH_INCH = 6.26
WORD_TITLE_PT = 12.0
WORD_BODY_PT = 10.5
FONT_VISUAL_COMPENSATION = 1.36
TYPOGRAPHY_FINE_TUNE_PT = -0.5

# ── Typography tuned for direct placement in A4 Word documents ─────────────
# Figures are usually pasted close to full text width in A4 Word documents.
# The effective source font therefore needs a visual compensation factor rather
# than a one-to-one point mapping.


def _round_to_half_point(value: float) -> float:
    """Round font sizes to stable 0.5 pt steps."""
    return round(value * 2.0) / 2.0


def derive_word_equivalent_fonts(
    figure_width_inch: float,
    target_print_width_inch: float = A4_TEXT_WIDTH_INCH,
    target_title_pt: float = WORD_TITLE_PT,
    target_body_pt: float = WORD_BODY_PT,
    compensation: float = FONT_VISUAL_COMPENSATION,
) -> tuple[float, float]:
    """Map Word-equivalent target typography to source matplotlib sizes."""
    print_scale = min(1.0, target_print_width_inch / figure_width_inch)
    title_size = target_title_pt / (print_scale * compensation)
    body_size = target_body_pt / (print_scale * compensation)
    return (
        _round_to_half_point(title_size + TYPOGRAPHY_FINE_TUNE_PT),
        _round_to_half_point(body_size + TYPOGRAPHY_FINE_TUNE_PT),
    )


TITLE_FONT_SIZE, BODY_FONT_SIZE = derive_word_equivalent_fonts(DOUBLE_COL_INCH)
LABEL_FONT_SIZE = BODY_FONT_SIZE
TICK_FONT_SIZE = max(7.5, _round_to_half_point(BODY_FONT_SIZE - 0.5))
LEGEND_FONT_SIZE = TICK_FONT_SIZE
TITLE_EMPHASIS_LINEWIDTH = 0.6


@dataclass(frozen=True)
class PanelGridLayout:
    """Physical layout recipe for grid figures."""

    figure_width: float
    figure_height: float
    panel_width: float
    panel_height: float
    left: float
    right: float
    bottom: float
    top: float
    wspace: float
    hspace: float
    title_font: float
    body_font: float
    tick_font: float
    annotation_font: float
    marker_area: float


def compute_panel_grid_layout(
    nrows: int,
    ncols: int,
    figure_width_inch: float,
    *,
    left_margin_inch: float = 0.38,
    right_margin_inch: float = 0.08,
    top_margin_inch: float = 0.22,
    bottom_margin_inch: float = 0.42,
    gap_ratio_x: float = 0.14,
    gap_ratio_y: float = 0.08,
    panel_aspect: float = 0.98,
) -> PanelGridLayout:
    """Compute a coordinated layout from panel count and target figure width."""
    usable_width = figure_width_inch - left_margin_inch - right_margin_inch
    panel_width = usable_width / (ncols + (ncols - 1) * gap_ratio_x)
    panel_height = panel_width * panel_aspect
    gap_x = gap_ratio_x * panel_width
    gap_y = gap_ratio_y * panel_height
    figure_height = (
        top_margin_inch
        + bottom_margin_inch
        + nrows * panel_height
        + (nrows - 1) * gap_y
    )

    title_font, body_font = derive_word_equivalent_fonts(figure_width_inch)
    scale = panel_width / 1.35
    tick_font = max(7.5, _round_to_half_point(body_font - 0.5))
    annotation_font = max(7.0, _round_to_half_point(body_font - 1.0))
    marker_area = max(6.0, 8.0 * (scale ** 1.05))

    return PanelGridLayout(
        figure_width=figure_width_inch,
        figure_height=figure_height,
        panel_width=panel_width,
        panel_height=panel_height,
        left=left_margin_inch / figure_width_inch,
        right=1.0 - right_margin_inch / figure_width_inch,
        bottom=bottom_margin_inch / figure_height,
        top=1.0 - top_margin_inch / figure_height,
        wspace=gap_x / panel_width,
        hspace=gap_y / panel_height,
        title_font=title_font,
        body_font=body_font,
        tick_font=tick_font,
        annotation_font=annotation_font,
        marker_area=marker_area,
    )

# ── Model visual identity ───────────────────────────────────────────────────
NATURE_COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "cyan": "#56B4E9",
    "magenta": "#CA9161",
    "yellow": "#ECE133",
    "purple": "#949494",
}

MODEL_COLORS = {
    "CGCNN": NATURE_COLORS["green"],
    "MOFTransformer": NATURE_COLORS["orange"],
    "ALIGNN": NATURE_COLORS["blue"],
}

MODEL_MARKERS = {
    "CGCNN": "s",
    "MOFTransformer": "^",
    "ALIGNN": "D",
}

MODEL_ORDER = ["CGCNN", "MOFTransformer", "ALIGNN"]


# ── Task display metadata ───────────────────────────────────────────────────
TASK_LIST = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
    "QstCH4", "QstN2",
]

TASK_LABELS = {
    "AdsCH4_10kPa":   r"CH$_4$@10 kPa",
    "AdsCH4_100kPa":  r"CH$_4$@100 kPa",
    "AdsCH4_1000kPa": r"CH$_4$@1000 kPa",
    "AdsN2_10kPa":    r"N$_2$@10 kPa",
    "AdsN2_100kPa":   r"N$_2$@100 kPa",
    "AdsN2_1000kPa":  r"N$_2$@1000 kPa",
    "QstCH4":         r"$Q_{\mathrm{st,CH}_4}$",
    "QstN2":          r"$Q_{\mathrm{st,N}_2}$",
}

TASK_UNITS = {
    "AdsCH4_10kPa":   "mol/kg",
    "AdsCH4_100kPa":  "mol/kg",
    "AdsCH4_1000kPa": "mol/kg",
    "AdsN2_10kPa":    "mol/kg",
    "AdsN2_100kPa":   "mol/kg",
    "AdsN2_1000kPa":  "mol/kg",
    "QstCH4":         "kJ/mol",
    "QstN2":          "kJ/mol",
}


# ── rcParams for publication quality ────────────────────────────────────────
def set_publication_style():
    """Apply Elsevier / SPT compatible rcParams."""
    plt.style.use("default")
    mpl.rcParams.update({
        # Font
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size":        BODY_FONT_SIZE,
        "axes.labelsize":   LABEL_FONT_SIZE,
        "axes.titlesize":   TITLE_FONT_SIZE,
        "axes.titleweight": "bold",
        "xtick.labelsize":  TICK_FONT_SIZE,
        "ytick.labelsize":  TICK_FONT_SIZE,
        "legend.fontsize":  LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        # Ticks
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        # Axes
        "axes.linewidth":   0.5,
        "lines.linewidth":  0.8,
        "lines.markersize": 3,
        # Figure
        "figure.dpi":       DPI,
        "savefig.dpi":      DPI,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
        # Legend
        "legend.frameon":    False,
        "legend.handlelength": 1.2,
        # Mathtext
        "mathtext.default": "regular",
    })


def apply_uniform_text_emphasis(
    text_obj,
    *,
    linewidth: float = TITLE_EMPHASIS_LINEWIDTH,
    foreground: str = "black",
) -> None:
    """Apply a math-safe emphasis treatment to mixed text/math labels."""
    text_obj.set_path_effects(
        [
            pe.Stroke(linewidth=linewidth, foreground=foreground),
            pe.Normal(),
        ]
    )


def set_emphasized_title(
    ax,
    text: str,
    *,
    loc: str = "center",
    fontsize: float | None = None,
    color: str = "black",
    linewidth: float = TITLE_EMPHASIS_LINEWIDTH,
    **kwargs,
):
    """Set a title and apply uniform emphasis to both text and math fragments."""
    title_obj = ax.set_title(
        text,
        loc=loc,
        fontsize=fontsize,
        color=color,
        fontweight="normal",
        **kwargs,
    )
    apply_uniform_text_emphasis(
        title_obj,
        linewidth=linewidth,
        foreground=color,
    )
    return title_obj


def save_figure(fig, name: str, output_dir: str | Path,
                formats=("png",), tight_layout: bool = True):
    """Save figure in multiple formats with tight layout."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=DPI, bbox_inches="tight",
                    pad_inches=0.02)
        print(f"  Saved: {path}")
