"""Figure S4: Distribution of 16 ML target labels in the integrated dataset.

Generates a 4×4 panel grid of histograms, one per label column, showing the
value distribution across all 21,976 MOFs in the training dataset.
"""

from __future__ import annotations

import argparse
import logging
import string
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "src" / "cgcnn" / "data" / "round2" / "integrated_ads_qst_metric_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures"

LABEL_COLUMNS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa", "QstCH4",
    "AdsN2_10kPa", "AdsN2_100kPa", "AdsN2_1000kPa", "QstN2",
    "PSA_WC_CH4", "PSA_WC_N2", "PSA_alpha_CH4_N2", "PSA_API_CH4",
    "VSA_WC_CH4", "VSA_WC_N2", "VSA_alpha_CH4_N2", "VSA_API_CH4",
]

_COLOR_CYCLE = list(NATURE_COLORS.values())


# ── Figure construction ─────────────────────────────────────────────────────


def make_figure(df: pd.DataFrame) -> plt.Figure:
    """Build the 4×4 histogram grid (Figure S4).

    Args:
        df: DataFrame containing all 16 label columns.

    Returns:
        Matplotlib Figure ready for saving.
    """
    set_publication_style()
    layout = compute_panel_grid_layout(nrows=4, ncols=4, figure_width_inch=DOUBLE_COL_INCH)

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )

    edge_lw = mpl.rcParams["axes.linewidth"]
    panel_letters = list(string.ascii_lowercase)

    for idx, col in enumerate(LABEL_COLUMNS):
        row, col_idx = divmod(idx, 4)
        ax = axes[row, col_idx]
        color = _COLOR_CYCLE[idx % len(_COLOR_CYCLE)]
        letter = panel_letters[idx]

        values = df[col].dropna()
        ax.hist(
            values,
            bins=50,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=edge_lw,
        )

        # Panel title
        ax.set_title(f"({letter}) {col}", loc="left", fontsize=layout.tick_font, fontweight="bold")

        # Axis labels (no xlabel — column name is in the panel title)
        # Only show ylabel on leftmost column
        if col_idx == 0:
            ax.set_ylabel("Frequency", fontsize=layout.tick_font)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=layout.tick_font)

        # Spine styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    return fig


# ── CLI entry ────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, load data, build figure, and save."""
    parser = argparse.ArgumentParser(
        description="Generate Figure S4: distribution of 16 ML target labels.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_CSV,
        help="Path to integrated label CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved figure (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    logger.info("Reading %s", args.csv)
    df = pd.read_csv(args.csv, usecols=LABEL_COLUMNS)
    logger.info("Loaded %d rows × %d label columns", len(df), len(LABEL_COLUMNS))

    fig = make_figure(df)
    save_figure(fig, "FigureS4_label_distributions", args.output_dir, tight_layout=True)

    # Summary statistics
    logger.info("=== Label distribution summary ===")
    for col in LABEL_COLUMNS:
        vals = df[col].dropna()
        logger.info(
            "  %-20s  n=%6d  min=%10.4f  median=%10.4f  max=%10.4f",
            col, len(vals), vals.min(), vals.median(), vals.max(),
        )


if __name__ == "__main__":
    main()
