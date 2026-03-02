"""
Multi-model parity plots: 2×4 panels with 3 models overlaid (replaces Fig 4).

Usage:
    python src/figures/fig4_model_parity.py [--output_dir DIR]
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
    MODEL_COLORS, MODEL_MARKERS, MODEL_ORDER,
    TASK_LIST, TASK_LABELS, TASK_UNITS,
)
from src.figures.data_loader import (
    load_xgboost_predictions, load_mft_predictions,
    load_alignn_predictions, r2_score,
)


def plot_parity(output_dir: Path):
    set_publication_style()

    # Load data
    data = {
        "XGBoost":        load_xgboost_predictions(),
        "MOFTransformer": load_mft_predictions(),
        "ALIGNN":         load_alignn_predictions("test"),
    }

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_INCH, 4.2))
    axes = axes.flatten()

    for idx, task in enumerate(TASK_LIST):
        ax = axes[idx]

        # Track global range for diagonal line
        all_vals = []

        for model_name in MODEL_ORDER:
            df = data[model_name]
            yt = df[f"{task}_true"].values
            yp = df[f"{task}_pred"].values
            r2 = r2_score(yt, yp)
            all_vals.extend(yt)
            all_vals.extend(yp)

            ax.scatter(
                yt, yp,
                c=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                s=4, alpha=0.25, linewidths=0,
                label=f"{model_name} ($R^2$={r2:.3f})",
                rasterized=True,
            )

        # Diagonal y=x
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.05
        lims = [lo - margin, hi + margin]
        ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.6)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        # Labels
        unit = TASK_UNITS[task]
        label = TASK_LABELS[task]
        ax.set_title(label, fontsize=7, pad=3)
        if idx >= 4:
            ax.set_xlabel(f"GCMC ({unit})", fontsize=6.5)
        if idx % 4 == 0:
            ax.set_ylabel(f"Predicted ({unit})", fontsize=6.5)

        ax.tick_params(labelsize=5.5)

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3,
        fontsize=6.5, frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        markerscale=2.5,
    )

    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    save_figure(fig, "Figure4_model_parity", output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="manuscript/figures")
    args = parser.parse_args()
    plot_parity(Path(args.output_dir))
    print("Done: model parity plots (Fig 4).")


if __name__ == "__main__":
    main()
