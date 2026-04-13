"""Figure S2: Geometric screening distributions (PLD and GSA).

Generates a two-panel figure:
  (a) Histogram of Pore Limiting Diameter with screening cutoff at 3.0 Å.
  (b) Histogram of Gravimetric Surface Area with screening cutoff at 100 m²/g.
"""

from __future__ import annotations

import argparse
import logging
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
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "RAC_and_zeo_features_deduplicated.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures"

PLD_CUTOFF = 3.0    # Å
GSA_CUTOFF = 100.0  # m²/g


# ── Figure construction ─────────────────────────────────────────────────────


def make_figure(df: pd.DataFrame) -> plt.Figure:
    """Build the two-panel Figure S2.

    Args:
        df: DataFrame with at least ``Df`` (PLD) and ``GSA`` columns.

    Returns:
        Matplotlib Figure ready for saving.
    """
    set_publication_style()
    layout = compute_panel_grid_layout(nrows=1, ncols=2, figure_width_inch=DOUBLE_COL_INCH)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=layout.wspace,
    )

    total = len(df)
    hist_color = NATURE_COLORS["cyan"]
    cutoff_color = NATURE_COLORS["orange"]
    edge_lw = mpl.rcParams["axes.linewidth"]

    # ── Panel (a): PLD distribution ──────────────────────────────────────────
    ax1.hist(
        df["Df"].dropna(),
        bins=100,
        color=hist_color,
        alpha=0.5,
        edgecolor="black",
        linewidth=edge_lw,
    )
    ax1.axvline(
        PLD_CUTOFF,
        color=cutoff_color,
        linestyle="--",
        linewidth=1.5,
        label=f"Cutoff = {PLD_CUTOFF} Å",
    )

    ax1.set_xlabel("Pore Limiting Diameter (Å)", fontsize=layout.body_font)
    ax1.set_ylabel("Count", fontsize=layout.body_font)
    ax1.tick_params(axis="both", which="major", labelsize=layout.tick_font)
    ax1.legend(
        frameon=True,
        edgecolor="black",
        loc="upper right",
        framealpha=0.7,
        fontsize=layout.tick_font,
    )

    # Spine styling
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)

    # ── Panel (b): GSA distribution ──────────────────────────────────────────
    ax2.hist(
        df["GSA"].dropna(),
        bins=100,
        color=hist_color,
        alpha=0.5,
        edgecolor="black",
        linewidth=edge_lw,
    )
    ax2.axvline(
        GSA_CUTOFF,
        color=cutoff_color,
        linestyle="--",
        linewidth=1.5,
        label=f"Cutoff = {GSA_CUTOFF} m²/g",
    )

    ax2.set_xlabel("Gravimetric Surface Area (m²/g)", fontsize=layout.body_font)
    ax2.set_ylabel("Count", fontsize=layout.body_font)
    ax2.tick_params(axis="both", which="major", labelsize=layout.tick_font)
    ax2.legend(
        frameon=True,
        edgecolor="black",
        loc="upper right",
        framealpha=0.7,
        fontsize=layout.tick_font,
    )

    # Spine styling
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)

    # ── Cross-panel titles via fig.text ──────────────────────────────────────
    title_y = layout.top + 0.035
    for ax, label in [(ax1, "(a)"), (ax2, "(b)")]:
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.005, title_y, label,
            fontsize=layout.title_font, fontweight="bold",
            ha="left", va="bottom",
            transform=fig.transFigure,
        )

    return fig


# ── CLI entry ────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, load data, build figure, and save."""
    parser = argparse.ArgumentParser(
        description="Generate Figure S2: PLD and GSA screening distributions.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=FEATURES_CSV,
        help="Path to deduplicated features CSV (default: %(default)s)",
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
    df = pd.read_csv(args.csv, usecols=["Df", "GSA"])
    logger.info("Loaded %d rows", len(df))

    fig = make_figure(df)
    save_figure(fig, "FigureS2_pore_distribution", args.output_dir, tight_layout=True)

    # Summary to stdout
    total = len(df)
    pass_pld = int((df["Df"] > PLD_CUTOFF).sum())
    pass_gsa = int((df["GSA"] > GSA_CUTOFF).sum())
    logger.info("=== Screening summary ===")
    logger.info("  Total MOFs:         %s", f"{total:,}")
    logger.info("  PLD > %.1f Å:       %s / %s (%.1f%%)", PLD_CUTOFF, f"{pass_pld:,}", f"{total:,}", pass_pld / total * 100)
    logger.info("  GSA > %.1f m²/g:   %s / %s (%.1f%%)", GSA_CUTOFF, f"{pass_gsa:,}", f"{total:,}", pass_gsa / total * 100)


if __name__ == "__main__":
    main()
