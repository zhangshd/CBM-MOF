"""
Publication style constants and helpers for CBM-MOF paper figures.

Target journal: Separation and Purification Technology (Elsevier).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ── Journal layout constants (Elsevier / SPT) ──────────────────────────────
SINGLE_COL_INCH = 3.35          # 8.5 cm
DOUBLE_COL_INCH = 6.89          # 17.5 cm
MAX_HEIGHT_INCH = 9.19          # 23.35 cm (full page)
DPI = 300

# ── Model visual identity ───────────────────────────────────────────────────
MODEL_COLORS = {
    "XGBoost":        "#949494",   # grey
    "MOFTransformer": "#DE8F05",   # orange
    "ALIGNN":         "#029E73",   # green
    "CGCNN":          "#0173B2",   # blue  (SI only)
}

MODEL_MARKERS = {
    "XGBoost":        "o",
    "MOFTransformer": "^",
    "ALIGNN":         "D",
    "CGCNN":          "s",          # SI only
}

MODEL_ORDER = ["XGBoost", "MOFTransformer", "ALIGNN"]   # main-text order
MODEL_ORDER_SI = ["XGBoost", "CGCNN", "MOFTransformer", "ALIGNN"]


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
    "QstCH4":         r"$Q_{\mathrm{st}}$(CH$_4$)",
    "QstN2":          r"$Q_{\mathrm{st}}$(N$_2$)",
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
        "font.size":        7,
        "axes.labelsize":   8,
        "axes.titlesize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  6.5,
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


def save_figure(fig, name: str, output_dir: str | Path,
                formats=("pdf", "png")):
    """Save figure in multiple formats with tight layout."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=DPI, bbox_inches="tight",
                    pad_inches=0.02)
        print(f"  Saved: {path}")


# ── Convenience ──────────────────────────────────────────────────────────────
# Default output directory for paper figures
FIGURES_DIR = Path(__file__).resolve().parents[2] / "manuscript" / "figures"
# This resolves to CBM-MOF-paper/manuscript/figures/ when called from
# CBM-MOF/src/figures/style.py — but each script can override via --output_dir
