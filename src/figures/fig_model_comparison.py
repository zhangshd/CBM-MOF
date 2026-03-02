"""
Model comparison heatmap: R² across 8 tasks for 4 models (main text).

Usage:
    python src/figures/fig_model_comparison.py [--output_dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (
    set_publication_style, save_figure, DOUBLE_COL_INCH,
    MODEL_COLORS, MODEL_ORDER_SI, TASK_LIST, TASK_LABELS,
)
from src.figures.data_loader import (
    load_xgboost_predictions, load_mft_predictions,
    load_alignn_predictions, load_cgcnn_predictions,
    r2_score, TASK_LIST as DL_TASKS,
)

# ── R² matrix (physical units, recomputed from data) ────────────────────────

def build_r2_matrix():
    """Return (model_names, task_names, R² array [n_models × n_tasks+1])."""
    loaders = {
        "XGBoost":        load_xgboost_predictions,
        "CGCNN":          lambda: load_cgcnn_predictions("symlog"),
        "MOFTransformer": load_mft_predictions,
        "ALIGNN":         lambda: load_alignn_predictions("test"),
    }
    tasks = DL_TASKS
    model_names = MODEL_ORDER_SI
    n_tasks = len(tasks)
    mat = np.zeros((len(model_names), n_tasks + 1))  # +1 for mean

    for i, name in enumerate(model_names):
        df = loaders[name]()
        for j, t in enumerate(tasks):
            mat[i, j] = r2_score(df[f"{t}_true"].values, df[f"{t}_pred"].values)
        mat[i, -1] = np.mean(mat[i, :-1])

    col_labels = [TASK_LABELS[t] for t in tasks] + ["Mean"]
    return model_names, col_labels, mat


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_heatmap(output_dir: Path):
    set_publication_style()

    model_names, col_labels, mat = build_r2_matrix()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_INCH, 2.2))

    # Custom diverging colormap: red (low) → yellow → green (high)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0.65, vmax=1.0)

    # Ticks
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=7)

    # Annotate each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            # Choose text color based on background brightness
            text_color = "white" if val < 0.78 else "black"
            fontweight = "bold" if j == mat.shape[1] - 1 else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6, color=text_color, fontweight=fontweight)

    # Separator line before Mean column
    ax.axvline(x=mat.shape[1] - 1.5, color="white", linewidth=2)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(r"$R^2$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(r"Test Set $R^2$ Comparison (Physical Units)", fontsize=8,
                 fontweight="bold", pad=6)

    save_figure(fig, "FigX_model_comparison_heatmap", output_dir)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="manuscript/figures")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    plot_heatmap(output_dir)
    print("Done: model comparison heatmap.")


if __name__ == "__main__":
    main()
