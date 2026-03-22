"""
Figure 13': Composition sensitivity — API(20:80) vs API(50:50) scatter.

Two-panel figure showing how MOF screening performance (API) changes
between the low-concentration (CH4:N2 = 20:80) and equimolar (50:50)
feed compositions. Experimental and hypothetical MOFs are distinguished;
ATC-Cu is highlighted as a benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Ensure src/figures is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    set_publication_style,
    save_figure,
    compute_panel_grid_layout,
    NATURE_COLORS,
    DOUBLE_COL_INCH,
    DPI,
)

# ── Constants ─────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
DATA_CSV = REPO / "results/alignn/model_ep150/composition_sensitivity/composition_sensitivity_results.csv"
DEFAULT_OUTPUT_DIR = REPO / "results" / "alignn" / "model_ep150" / "figures"

ATC_CU_ID = "CoRE-2020[Cu][pts]3[ASR]1"

EXP_PREFIXES = ("CoRE-", "MOSAEC-", "ARC-DB12-", "ARC-DB14-")

# Visual identity
COLOR_EXP = NATURE_COLORS["blue"]
COLOR_HYPO = NATURE_COLORS["orange"]
COLOR_ATC = "#D62728"  # matplotlib red


def _is_exp(mof_id: str) -> bool:
    return any(mof_id.startswith(p) for p in EXP_PREFIXES)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    # Re-derive is_exp from mof_id (more robust than trusting CSV column)
    df["is_exp"] = df["mof_id"].apply(_is_exp)
    return df


def _plot_panel(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label: str,
    layout,
):
    """Draw one scatter panel (PSA or VSA)."""
    hypo = df[~df["is_exp"]]
    exp = df[df["is_exp"]]
    atc = df[df["mof_id"] == ATC_CU_ID]

    ms = layout.marker_area

    # Hypothetical MOFs (background)
    ax.scatter(
        hypo[x_col], hypo[y_col],
        s=ms, alpha=0.45, color=COLOR_HYPO, edgecolors="none",
        label="Hypothetical", zorder=2, rasterized=True,
    )
    # Experimental MOFs (foreground)
    ax.scatter(
        exp[x_col], exp[y_col],
        s=ms * 1.3, alpha=0.70, color=COLOR_EXP, edgecolors="none",
        label="Experimental", zorder=3, rasterized=True,
    )
    # ATC-Cu benchmark
    if not atc.empty:
        ax.scatter(
            atc[x_col], atc[y_col],
            s=ms * 8, marker="*", color=COLOR_ATC,
            edgecolors="k", linewidths=0.3,
            label="ATC-Cu", zorder=5,
        )

    # 1:1 diagonal
    lo = min(df[x_col].min(), df[y_col].min()) * 0.9
    hi = max(df[x_col].max(), df[y_col].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="gray", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # Spearman rho
    rho, pval = stats.spearmanr(df[x_col], df[y_col])
    ax.text(
        0.05, 0.93,
        rf"$\rho_s$ = {rho:.3f}",
        transform=ax.transAxes,
        fontsize=layout.annotation_font,
        va="top",
    )

    # Labels
    ax.set_xlabel(r"API at CH$_4$:N$_2$ = 20:80", fontsize=layout.body_font)
    ax.set_ylabel(r"API at CH$_4$:N$_2$ = 50:50", fontsize=layout.body_font)
    ax.set_title(label, fontsize=layout.title_font, fontweight="bold", loc="left")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 13: Composition sensitivity scatter (20:80 vs 50:50)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the figure (default: results/alignn/model_ep150/figures).",
    )
    args = parser.parse_args()

    set_publication_style()
    df = load_data()
    print(f"Loaded {len(df)} MOFs  (exp={df['is_exp'].sum()}, hypo={(~df['is_exp']).sum()})")

    layout = compute_panel_grid_layout(1, 2, DOUBLE_COL_INCH)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left, right=layout.right,
        bottom=layout.bottom, top=layout.top,
        wspace=layout.wspace + 0.15,  # extra room for y-label
    )

    _plot_panel(axes[0], df, "PSA_API_2080", "PSA_API_5050", "(a) PSA", layout)
    _plot_panel(axes[1], df, "VSA_API_2080", "VSA_API_5050", "(b) VSA", layout)

    # Shared legend from right panel
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=layout.tick_font,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    save_figure(fig, "Figure13_composition_sensitivity", args.output_dir,
                formats=("png",), tight_layout=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
