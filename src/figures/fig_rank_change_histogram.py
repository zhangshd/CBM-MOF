"""
Figure 13b: Rank-change distribution histogram for composition sensitivity.

Two-panel stacked histogram showing the distribution of absolute rank changes
(|Drank|) when MOF rankings are compared between CH4:N2 = 20:80 and 50:50
compositions. Experimental and hypothetical MOFs are shown as stacked bars.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
TOP_CANDIDATES_DIR = REPO / "results/alignn/model_ep150/top_candidates"
DEFAULT_OUTPUT_DIR = REPO / "results" / "alignn" / "model_ep150" / "figures"

EXP_PREFIXES = ("CoRE-", "MOSAEC-", "ARC-DB12-", "ARC-DB14-")

# Visual identity (matches fig_composition_sensitivity.py)
COLOR_EXP = NATURE_COLORS["blue"]
COLOR_HYPO = NATURE_COLORS["orange"]

# Bin edges for |Drank| histogram
BIN_EDGES = [0, 5, 10, 20, 30, 40, 50, 100]
BIN_LABELS = ["0\u20135", "5\u201310", "10\u201320", "20\u201330", "30\u201340", "40\u201350", "50+"]

# Threshold for "significant shift"
SHIFT_THRESHOLD = 10


def _is_exp(mof_id: str) -> bool:
    return any(mof_id.startswith(p) for p in EXP_PREFIXES)


def _load_pool_ids(prefix: str) -> set[str]:
    """Load union of exp + hypo top-50 MOF IDs for a given process (psa or vsa)."""
    exp_csv = TOP_CANDIDATES_DIR / f"exp_top50_{prefix}.csv"
    hypo_csv = TOP_CANDIDATES_DIR / f"hypo_top50_{prefix}.csv"
    exp_ids = set(pd.read_csv(exp_csv, usecols=["mof_id"])["mof_id"])
    hypo_ids = set(pd.read_csv(hypo_csv, usecols=["mof_id"])["mof_id"])
    return exp_ids | hypo_ids


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load composition sensitivity results, filtered to per-process MOF pools.

    Returns:
        (df_psa, df_vsa): DataFrames filtered to PSA and VSA candidate pools.
    """
    df = pd.read_csv(DATA_CSV)
    # Re-derive is_exp from mof_id (more robust than trusting CSV column)
    df["is_exp"] = df["mof_id"].apply(_is_exp)

    psa_ids = _load_pool_ids("psa")
    vsa_ids = _load_pool_ids("vsa")

    df_psa = df[df["mof_id"].isin(psa_ids)].copy()
    df_vsa = df[df["mof_id"].isin(vsa_ids)].copy()

    return df_psa, df_vsa


def _bin_rank_changes(values: pd.Series) -> np.ndarray:
    """Bin absolute rank changes into predefined bins, return counts per bin."""
    abs_vals = values.abs()
    counts = np.zeros(len(BIN_LABELS), dtype=int)
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        if i == len(BIN_EDGES) - 2:
            # Last bin: open-ended (50+)
            counts[i] = (abs_vals >= lo).sum()
        else:
            counts[i] = ((abs_vals >= lo) & (abs_vals < hi)).sum()
    return counts


def _plot_panel(
    ax,
    df: pd.DataFrame,
    rank_col: str,
    panel_label: str,
    layout,
):
    """Draw one stacked histogram panel (PSA or VSA)."""
    exp_mask = df["is_exp"]
    hypo_mask = ~df["is_exp"]

    exp_counts = _bin_rank_changes(df.loc[exp_mask, rank_col])
    hypo_counts = _bin_rank_changes(df.loc[hypo_mask, rank_col])

    x = np.arange(len(BIN_LABELS))
    bar_width = 0.65

    # Stacked bars: hypothetical on bottom, experimental on top
    ax.bar(
        x, hypo_counts,
        width=bar_width, color=COLOR_HYPO, alpha=0.85,
        label="Hypothetical", zorder=2, edgecolor="white", linewidth=0.3,
    )
    ax.bar(
        x, exp_counts, bottom=hypo_counts,
        width=bar_width, color=COLOR_EXP, alpha=0.85,
        label="Experimental", zorder=2, edgecolor="white", linewidth=0.3,
    )

    # Vertical dashed line at threshold (between bin "0-5" and "5-10" is at x=1,
    # but threshold=10 falls at the boundary between bin index 1 and 2)
    threshold_x = 1.5  # boundary between "5-10" and "10-20" bins
    ax.axvline(
        threshold_x, ls="--", lw=0.8, color="#555555", zorder=3,
    )

    # Count MOFs shifting >= threshold
    abs_changes = df[rank_col].abs()
    n_shifted = (abs_changes >= SHIFT_THRESHOLD).sum()
    n_total = len(df)
    pct_shifted = 100.0 * n_shifted / n_total

    # Annotation text (upper-right, away from bars)
    ax.text(
        0.97, 0.92,
        f"|$\\Delta$rank| $\\geq$ {SHIFT_THRESHOLD}:\n{n_shifted}/{n_total} ({pct_shifted:.1f}%)",
        transform=ax.transAxes,
        fontsize=layout.annotation_font,
        va="top", ha="right",
        color="#555555",
    )

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=layout.tick_font)
    ax.set_xlabel(r"|$\Delta$rank|", fontsize=layout.body_font)
    ax.set_ylabel("Number of MOFs", fontsize=layout.body_font)
    ax.set_title(panel_label, fontsize=layout.title_font, fontweight="bold", loc="left")

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Integer y-ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 13b: Rank-change distribution histogram "
                    "(20:80 vs 50:50 composition sensitivity)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the figure (default: results/alignn/model_ep150/figures).",
    )
    args = parser.parse_args()

    set_publication_style()
    df_psa, df_vsa = load_data()

    for label, df in [("PSA", df_psa), ("VSA", df_vsa)]:
        n_exp = df["is_exp"].sum()
        n_hypo = (~df["is_exp"]).sum()
        print(f"{label}: {len(df)} MOFs  (exp={n_exp}, hypo={n_hypo})")

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

    _plot_panel(
        axes[0], df_psa,
        rank_col="PSA_rank_change",
        panel_label="(a) PSA Case",
        layout=layout,
    )
    _plot_panel(
        axes[1], df_vsa,
        rank_col="VSA_rank_change",
        panel_label="(b) VSA Case",
        layout=layout,
    )

    # Shared legend from right panel
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        fontsize=layout.tick_font,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    save_figure(fig, "Figure13_rank_change_distribution", args.output_dir,
                formats=("png",), tight_layout=False)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
