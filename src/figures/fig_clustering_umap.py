"""Figure 2: Cluster-aware partitioning and stratified sampling UMAP visualization.

Panels
------
(a-e) Five views of the same UMAP coordinates. Each view highlights four or
    five clusters against a common gray background so that cluster identity does
    not depend on distinguishing 22 colors in one panel.
(f) Same projection highlighting the 21,976 sampled structures (train+val+test)
    against the full-library background.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_CSV = (
    PROJECT_ROOT / "data" / "processed" / "textural_screened" / "textural_screened_clustered_with_umap.csv"
)
STRAT_DIR = PROJECT_ROOT / "data" / "processed" / "stratified_datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures"

# Cluster IDs are zero-based here and displayed as one-based labels. The groups
# separate nearby robust centers to minimize label collisions without changing
# the cluster assignments.
CLUSTER_DISPLAY_GROUPS = (
    (9, 17, 19, 20, 21),
    (1, 7, 8, 14, 15),
    (0, 2, 4, 16),
    (6, 11, 12, 13),
    (3, 5, 10, 18),
)

# Six high-contrast colors from the Okabe-Ito family. Colors are reused between
# subpanels because the numeric labels provide the definitive cluster identity.
HIGHLIGHT_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#000000",  # black
)
BACKGROUND_COLOR = "#D9D9D9"
SAMPLED_COLOR = "#005A8D"

# Short point-space offsets separate the only close pair that remains after
# grouping. Their leader lines retain the exact anchor locations.
CLUSTER_LABEL_OFFSETS = {
    19: (8, -2),
    20: (-8, 2),
}


# ── Data loading ────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, set[str]]:
    """Load clustered UMAP coordinates and the union of stratified-sample IDs.

    Returns:
        df: DataFrame with columns ``CifId, Cluster, UMAP1, UMAP2``.
        sampled_ids: set of CifId strings that belong to train/val/test.
    """
    logger.info("Loading clustered UMAP data from %s", CLUSTER_CSV)
    df = pd.read_csv(CLUSTER_CSV, usecols=["CifId", "Cluster", "UMAP1", "UMAP2"])
    logger.info("  Loaded %d rows, %d unique clusters", len(df), df["Cluster"].nunique())

    sampled_ids: set[str] = set()
    for split in ("train", "val", "test"):
        csv_path = STRAT_DIR / f"{split}_set.csv"
        split_df = pd.read_csv(csv_path, usecols=["name"])
        sampled_ids.update(split_df["name"].tolist())
        logger.info("  %s: %d structures", split, len(split_df))
    logger.info("  Total sampled: %d", len(sampled_ids))
    return df, sampled_ids


def _validate_display_groups(clusters: np.ndarray) -> None:
    """Ensure that every observed cluster appears in exactly one display group."""
    configured = [cid for group in CLUSTER_DISPLAY_GROUPS for cid in group]
    observed = set(np.unique(clusters).astype(int))
    if len(configured) != len(set(configured)) or set(configured) != observed:
        raise ValueError(
            "CLUSTER_DISPLAY_GROUPS must contain every observed cluster exactly once."
        )


def _cluster_anchor(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    cluster_id: int,
) -> tuple[float, float]:
    """Return the observed point nearest a cluster's coordinate-wise median."""
    cluster_idx = np.flatnonzero(clusters == cluster_id)
    cluster_x = x[cluster_idx]
    cluster_y = y[cluster_idx]
    center_x = np.median(cluster_x)
    center_y = np.median(cluster_y)
    nearest = np.argmin((cluster_x - center_x) ** 2 + (cluster_y - center_y) ** 2)
    return float(cluster_x[nearest]), float(cluster_y[nearest])


# ── Figure construction ─────────────────────────────────────────────────────
def make_figure(df: pd.DataFrame, sampled_ids: set[str]) -> plt.Figure:
    """Build grouped cluster views and the stratified-sampling UMAP."""
    set_publication_style()

    layout = compute_panel_grid_layout(
        nrows=2,
        ncols=3,
        figure_width_inch=DOUBLE_COL_INCH,
        right_margin_inch=0.08,
        gap_ratio_x=0.16,
        gap_ratio_y=0.18,
        panel_aspect=0.92,
    )

    from matplotlib.gridspec import GridSpec

    fig_h = layout.figure_height
    fig = plt.figure(figsize=(DOUBLE_COL_INCH, fig_h))

    # Five cluster views and one binary sampling view form a regular 2 x 3 grid.
    gs = GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[1, 1, 1],
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )
    cluster_axes = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    )
    ax_b = fig.add_subplot(gs[1, 2])

    x = df["UMAP1"].values
    y = df["UMAP2"].values
    clusters = df["Cluster"].values

    _validate_display_groups(clusters)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(df))

    x_padding = 0.03 * (x.max() - x.min())
    y_padding = 0.03 * (y.max() - y.min())
    common_xlim = (x.min() - x_padding, x.max() + x_padding)
    common_ylim = (y.min() - y_padding, y.max() + y_padding)

    background_size = max(0.25, layout.marker_area * 0.035)
    highlight_size = max(0.45, layout.marker_area * 0.065)

    # ── Panels (a-e): four or five highlighted clusters per view ────────
    for panel_idx, (ax, group) in enumerate(
        zip(cluster_axes, CLUSTER_DISPLAY_GROUPS, strict=True)
    ):
        ax.scatter(
            x[order], y[order],
            c=BACKGROUND_COLOR, s=background_size, alpha=0.32,
            edgecolors="none", rasterized=True, zorder=1,
        )

        legend_handles = []
        for color, cid in zip(HIGHLIGHT_COLORS, group, strict=False):
            mask = clusters == cid
            ax.scatter(
                x[mask], y[mask],
                c=color, s=highlight_size, alpha=0.72,
                edgecolors="none", rasterized=True, zorder=2,
            )
            anchor_x, anchor_y = _cluster_anchor(x, y, clusters, cid)
            label_offset = CLUSTER_LABEL_OFFSETS.get(cid, (0, 0))
            ax.annotate(
                str(cid + 1), xy=(anchor_x, anchor_y),
                xytext=label_offset, textcoords="offset points",
                ha="center", va="center",
                fontsize=layout.annotation_font,
                fontweight="bold", color="black", zorder=5,
                bbox={
                    "boxstyle": "circle,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.6,
                    "alpha": 0.92,
                },
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": "black",
                        "linewidth": 0.45,
                        "shrinkA": 5,
                        "shrinkB": 1,
                    }
                    if label_offset != (0, 0)
                    else None
                ),
            )
            legend_handles.append(
                Line2D(
                    [0], [0], marker="o", color="none",
                    markerfacecolor=color, markeredgecolor="none",
                    markersize=3.5, linestyle="None", label=str(cid + 1),
                )
            )

        panel_label = chr(ord("a") + panel_idx)
        ax.set_title(f"({panel_label})", fontweight="bold", loc="left",
                     fontsize=layout.title_font)
        ax.legend(
            handles=legend_handles,
            loc="upper right", bbox_to_anchor=(1.0, 1.07),
            ncol=len(legend_handles), frameon=False,
            fontsize=max(6.0, layout.tick_font - 1.5),
            handlelength=0.5, handletextpad=0.15,
            columnspacing=0.45, borderaxespad=0.0,
        )
        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cluster_axes[3].set_xlabel("UMAP 1", fontsize=layout.body_font)
    cluster_axes[4].set_xlabel("UMAP 1", fontsize=layout.body_font)
    cluster_axes[0].set_ylabel("UMAP 2", fontsize=layout.body_font)
    cluster_axes[3].set_ylabel("UMAP 2", fontsize=layout.body_font)

    # ── Panel (f): sampled vs full library ──────────────────────────────
    is_sampled = df["CifId"].isin(sampled_ids).values
    is_unsampled = ~is_sampled

    ax_b.scatter(
        x[is_unsampled], y[is_unsampled],
        c="#BFBFBF", s=background_size, alpha=0.50,
        edgecolors="none", rasterized=True, zorder=1,
    )

    ax_b.scatter(
        x[is_sampled], y[is_sampled],
        c=SAMPLED_COLOR, s=background_size * 0.75, alpha=0.72,
        edgecolors="none", rasterized=True, zorder=2,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#BFBFBF",
               markeredgecolor="none", markersize=4, linestyle="None",
               label="Unsampled"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SAMPLED_COLOR,
               markeredgecolor="none", markersize=4, linestyle="None",
               label="Sampled"),
    ]
    sampling_legend = ax_b.legend(
        handles=legend_handles,
        title="Sampling",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        fontsize=layout.tick_font - 1,
        title_fontsize=layout.tick_font,
        frameon=True,
        fancybox=False,
        edgecolor="k",
        framealpha=0.7,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    sampling_legend.get_frame().set_linewidth(0.5)

    ax_b.set_xlabel("UMAP 1", fontsize=layout.body_font)
    ax_b.set_ylabel("UMAP 2", fontsize=layout.body_font)
    ax_b.set_xlim(common_xlim)
    ax_b.set_ylim(common_ylim)
    ax_b.tick_params(labelbottom=False, labelleft=False)
    ax_b.set_title("(f)", fontweight="bold", loc="left",
                   fontsize=layout.title_font)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    return fig


# ── CLI entry point ─────────────────────────────────────────────────────────
def main() -> None:
    """Parse arguments and produce Figure 2."""
    parser = argparse.ArgumentParser(
        description="Figure 2: UMAP clustering and stratified sampling visualization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure will be saved (default: %(default)s).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    df, sampled_ids = load_data()
    fig = make_figure(df, sampled_ids)
    save_figure(fig, "Figure02_clustering_umap", args.output_dir, tight_layout=False)
    plt.close(fig)
    logger.info("Done — Figure 2 saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
