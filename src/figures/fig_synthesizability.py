"""
Synthesizability screening: ∆LMFFL distribution histogram (NEW figure).

Usage:
    python src/figures/fig_synthesizability.py [--output_dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (
    set_publication_style, save_figure, SINGLE_COL_INCH,
)
from src.figures.data_loader import load_synthesizability


THRESHOLD = 4.4  # kJ/mol


def plot_synthesizability(output_dir: Path):
    set_publication_style()

    df = load_synthesizability()
    dlmffl = df["dLMFFL"].dropna().values

    fig, ax = plt.subplots(figsize=(SINGLE_COL_INCH, 2.5))

    # Histogram
    bins = np.linspace(-5, 42, 60)
    n, bin_edges, patches = ax.hist(
        dlmffl, bins=bins, color="#CCCCCC", edgecolor="white",
        linewidth=0.3, zorder=2,
    )

    # Color bars below threshold green
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge + (bin_edges[1] - bin_edges[0]) / 2 <= THRESHOLD:
            patch.set_facecolor("#029E73")
            patch.set_alpha(0.8)

    # Threshold line
    ax.axvline(x=THRESHOLD, color="#CC3311", linewidth=1.0, linestyle="--",
               zorder=3)
    ax.text(
        THRESHOLD + 0.5, ax.get_ylim()[1] * 0.85,
        f"Threshold = {THRESHOLD} kJ/mol",
        fontsize=6, color="#CC3311", va="top",
    )

    # Count annotation
    n_below = int(np.sum(dlmffl <= THRESHOLD))
    n_total = len(dlmffl)
    pct = n_below / n_total * 100
    ax.text(
        0.95, 0.92,
        f"{n_below}/{n_total} ({pct:.1f}%)\nbelow threshold",
        transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#029E73",
                  alpha=0.9, linewidth=0.5),
    )

    ax.set_xlabel(r"$\Delta$LMFFL (kJ/mol)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Synthesizability Assessment", fontsize=8, fontweight="bold",
                 pad=4)

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig, "FigX_synthesizability", output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="manuscript/figures")
    args = parser.parse_args()
    plot_synthesizability(Path(args.output_dir))
    print("Done: synthesizability distribution plot.")


if __name__ == "__main__":
    main()
