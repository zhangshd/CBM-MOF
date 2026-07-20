"""Composition-sensitivity rank-change histograms.

The script generates the two-panel API figure used as Figure 9 and a separate
four-panel Supporting Information figure for methane working capacity and
selectivity. Ranks compare CH4:N2 = 20:80 and 50:50 compositions, while the
displayed distributions track the process-specific 20:80 API top-100 sets.
Experimental and hypothetical MOFs are shown as stacked bars.
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

# Visual identity aligned with Figure 7
COLOR_EXP = NATURE_COLORS["green"]
COLOR_HYPO = NATURE_COLORS["purple"]

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
    *,
    ylabel: str,
    show_legend: bool,
    annotation_x: float = 0.72,
):
    """Draw one stacked histogram panel (PSA or VSA)."""
    exp_mask = df["is_exp"]
    hypo_mask = ~df["is_exp"]

    exp_counts = _bin_rank_changes(df.loc[exp_mask, rank_col])
    hypo_counts = _bin_rank_changes(df.loc[hypo_mask, rank_col])

    x = np.arange(len(BIN_LABELS))
    bar_width = 0.65

    # Stacked bars: experimental on bottom, hypothetical on top
    ax.bar(
        x, exp_counts,
        width=bar_width, color=COLOR_EXP, alpha=0.85,
        label="Experimental", zorder=2, edgecolor="black", linewidth=0.4,
    )
    ax.bar(
        x, hypo_counts, bottom=exp_counts,
        width=bar_width, color=COLOR_HYPO, alpha=0.85,
        label="Hypothetical", zorder=2, edgecolor="black", linewidth=0.4,
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

    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=layout.tick_font,
            handlelength=1.2,
            borderaxespad=0.2,
        )

    # Annotation text placed below the in-panel legend and aligned to its left edge
    ax.text(
        annotation_x, 0.70,
        f"|$\\Delta$rank| $\\geq$ {SHIFT_THRESHOLD}:\n{n_shifted}/{n_total} ({pct_shifted:.1f}%)",
        transform=ax.transAxes,
        fontsize=layout.tick_font,
        va="top", ha="left",
        color="#555555",
    )

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=layout.tick_font)
    ax.set_xlabel(r"|$\Delta$rank|", fontsize=layout.body_font)
    ax.set_ylabel(ylabel, fontsize=layout.body_font)
    ax.set_title(panel_label, fontsize=layout.title_font, fontweight="bold", loc="left")

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Integer y-ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))


def main():
    parser = argparse.ArgumentParser(
        description="Generate API and individual-metric rank-change histograms "
                    "for 20:80 vs 50:50 composition sensitivity."
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
        figsize=(DOUBLE_COL_INCH, 0.45 * DOUBLE_COL_INCH),
    )

    _plot_panel(
        axes[0], df_psa,
        rank_col="PSA_rank_change",
        panel_label="(a) PSA Case",
        layout=layout,
        ylabel="Number of MOFs",
        show_legend=True,
    )
    _plot_panel(
        axes[1], df_vsa,
        rank_col="VSA_rank_change",
        panel_label="(b) VSA Case",
        layout=layout,
        ylabel="",
        show_legend=False,
    )

    fig.tight_layout(w_pad=0.45)
    save_figure(fig, "Figure13_rank_change_distribution", args.output_dir,
                formats=("png",))
    plt.close(fig)

    si_layout = compute_panel_grid_layout(2, 2, DOUBLE_COL_INCH)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL_INCH, 0.88 * DOUBLE_COL_INCH),
    )

    panels = [
        (axes[0, 0], df_psa, "PSA_working_capacity_rank_change",
         "(a) PSA working capacity", "Number of MOFs", True),
        (axes[0, 1], df_vsa, "VSA_working_capacity_rank_change",
         "(b) VSA working capacity", "", False),
        (axes[1, 0], df_psa, "PSA_selectivity_rank_change",
         "(c) PSA selectivity", "Number of MOFs", False),
        (axes[1, 1], df_vsa, "VSA_selectivity_rank_change",
         "(d) VSA selectivity", "", False),
    ]
    for ax, df, rank_col, title, ylabel, show_legend in panels:
        _plot_panel(
            ax,
            df,
            rank_col=rank_col,
            panel_label=title,
            layout=si_layout,
            ylabel=ylabel,
            show_legend=show_legend,
            annotation_x=0.58,
        )

    fig.tight_layout(w_pad=0.45, h_pad=0.70)
    save_figure(
        fig,
        "FigureS13_metric_rank_change_distribution",
        args.output_dir,
        formats=("png",),
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
