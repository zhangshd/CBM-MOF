"""
Top-100 PSA/VSA parity plots: ALIGNN vs XGBoost (replaces Fig 8).

Layout: two separate figures (8a: PSA, 8b: VSA), each 2 x 4 panels.
Row 1 = CH4 (3 uptakes + QstCH4), Row 2 = N2 (3 uptakes + QstN2).
Each panel shows ALIGNN scatter + R^2 annotation (ALIGNN & XGBoost).

Usage:
    python src/figures/fig8_top100_parity.py [--output_dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (
    set_publication_style, save_figure, DOUBLE_COL_INCH,
    MODEL_COLORS, MODEL_MARKERS, TASK_LIST, TASK_LABELS, TASK_UNITS,
)
from src.figures.data_loader import (
    load_alignn_predictions, r2_score, UPTAKE_TASKS,
)

# ── Task ordering for 2x4 layout ──────────────────────────────────────────
# Row 0: CH4 tasks (3 uptakes + Qst), Row 1: N2 tasks (3 uptakes + Qst)
PANEL_ORDER = [
    # Row 0
    ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa", "QstCH4"],
    # Row 1
    ["AdsN2_10kPa", "AdsN2_100kPa", "AdsN2_1000kPa", "QstN2"],
]

# Known XGBoost R^2 for top-100 (from model_comparison.md)
XGB_R2 = {
    "top_100_psa": {
        "AdsCH4_10kPa": 0.703, "AdsCH4_100kPa": 0.776,
        "AdsCH4_1000kPa": 0.809, "AdsN2_10kPa": 0.664,
        "AdsN2_100kPa": 0.768, "AdsN2_1000kPa": 0.886,
        "QstCH4": 0.654, "QstN2": 0.734,
    },
    "top_100_vsa": {
        "AdsCH4_10kPa": 0.688, "AdsCH4_100kPa": 0.439,
        "AdsCH4_1000kPa": 0.599, "AdsN2_10kPa": 0.642,
        "AdsN2_100kPa": 0.518, "AdsN2_1000kPa": 0.829,
        "QstCH4": 0.826, "QstN2": 0.752,
    },
}


def _plot_single_split(split: str, split_label: str,
                       output_dir: Path, fig_name: str):
    """Plot one 2x4 parity figure for a single top-100 split."""
    set_publication_style()

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_INCH, 4.0))

    df = load_alignn_predictions(split)
    xgb_r2 = XGB_R2[split]

    for row in range(2):
        for col in range(4):
            task = PANEL_ORDER[row][col]
            ax = axes[row, col]

            yt = df[f"{task}_true"].values
            yp = df[f"{task}_pred"].values
            r2_aln = r2_score(yt, yp)

            # ALIGNN scatter
            ax.scatter(
                yt, yp,
                c=MODEL_COLORS["ALIGNN"], marker=MODEL_MARKERS["ALIGNN"],
                s=12, alpha=0.6, linewidths=0, rasterized=True,
            )

            # Diagonal
            lo = min(yt.min(), yp.min())
            hi = max(yt.max(), yp.max())
            margin = (hi - lo) * 0.08
            lims = [lo - margin, hi + margin]
            ax.plot(lims, lims, "k--", linewidth=0.4, alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")

            # R^2 annotation: two separate calls, no layering
            r2_xgb = xgb_r2.get(task, float("nan"))
            # Call 1: ALN row with $R^2$ header (bold, white bbox)
            ax.text(
                0.05, 0.95,
                f"$R^2$:\nALN {r2_aln:.3f}",
                transform=ax.transAxes, fontsize=5.5,
                va="top", ha="left", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.8),
            )
            # Call 2: XGB row (normal weight, positioned below)
            ax.text(
                0.05, 0.80,
                f"XGB {r2_xgb:.3f}",
                transform=ax.transAxes, fontsize=5.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.8),
            )

            # Title (all panels)
            ax.set_title(TASK_LABELS[task], fontsize=7, pad=3)

            # Axis labels with units
            unit = TASK_UNITS[task]
            if row == 1:
                ax.set_xlabel(f"GCMC ({unit})", fontsize=6)
            if col == 0:
                ax.set_ylabel(f"Predicted ({unit})", fontsize=6)

            ax.tick_params(labelsize=5)

    fig.suptitle(f"{split_label}", fontsize=8, fontweight="bold", y=1.01)
    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    save_figure(fig, fig_name, output_dir)
    plt.close(fig)


def plot_top100_psa(output_dir: Path):
    """Generate Figure 8a: PSA Top-100 parity (2x4)."""
    _plot_single_split("top_100_psa", "PSA Top-100",
                       output_dir, "Figure8a_top100_psa")


def plot_top100_vsa(output_dir: Path):
    """Generate Figure 8b: VSA Top-100 parity (2x4)."""
    _plot_single_split("top_100_vsa", "VSA Top-100",
                       output_dir, "Figure8b_top100_vsa")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="manuscript/figures")
    args = parser.parse_args()
    out = Path(args.output_dir)
    plot_top100_psa(out)
    plot_top100_vsa(out)
    print("Done: top-100 PSA/VSA parity plots (Fig 8a/8b).")


if __name__ == "__main__":
    main()
