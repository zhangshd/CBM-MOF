"""Figure 2: Cluster-aware partitioning and stratified sampling UMAP visualization.

Panels
------
(a) UMAP of 235,141 MOFs colored by 22 cluster assignments.
(b) Same projection highlighting the 21,976 sampled structures (train+val+test)
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
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


# ── Colormap for 22 clusters ───────────────────────────────────────────────
def _build_cluster_cmap(n_clusters: int = 22) -> mcolors.ListedColormap:
    """Create a qualitative colormap for *n_clusters* categories.

    Combines ``tab20`` (20 colors) with two extra distinguishable hues so that
    all 22 clusters get unique colors.
    """
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    extras = np.array([
        [0.40, 0.00, 0.40, 1.0],  # dark purple
        [0.00, 0.40, 0.40, 1.0],  # teal
    ])
    colors = np.vstack([tab20, extras])[:n_clusters]
    return mcolors.ListedColormap(colors, name="cluster22")


# ── Figure construction ─────────────────────────────────────────────────────
def make_figure(df: pd.DataFrame, sampled_ids: set[str]) -> plt.Figure:
    """Build the two-panel UMAP figure.

    Layout: [panel_a | legend | gap | panel_b]
    Legend sits snug against panel (a); panel (b) has no y-label.
    """
    set_publication_style()

    layout = compute_panel_grid_layout(
        nrows=1,
        ncols=2,
        figure_width_inch=DOUBLE_COL_INCH,
        right_margin_inch=0.08,
        panel_aspect=0.85,
    )

    from matplotlib.gridspec import GridSpec

    fig_h = layout.figure_height
    fig = plt.figure(figsize=(DOUBLE_COL_INCH, fig_h))

    # 4-column grid: panel_a (4) | legend (0.9) | gap (0.3) | panel_b (4)
    gs = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[4, 0.9, 0.3, 4],
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=0.0,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    # gs[0, 2] is the gap — no axis
    ax_b = fig.add_subplot(gs[0, 3])
    ax_legend.set_axis_off()

    x = df["UMAP1"].values
    y = df["UMAP2"].values
    clusters = df["Cluster"].values

    n_clusters = int(clusters.max()) + 1
    cmap = _build_cluster_cmap(n_clusters)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1), n_clusters)

    # ── Panel (a): cluster-colored UMAP ─────────────────────────────────
    rng = np.random.default_rng(42)
    order = rng.permutation(len(df))

    ax_a.scatter(
        x[order], y[order],
        c=clusters[order], cmap=cmap, norm=norm,
        s=layout.marker_area * 0.15, alpha=0.6,
        edgecolors="k", linewidths=0.1, rasterized=True,
    )

    # Direct labels preserve cluster identity in grayscale. Place each label
    # on the observed point nearest the cluster median to keep it in-region.
    label_offsets = {
        15: (8, 8),
        18: (-10, -4),
        19: (10, -8),
        20: (-10, 8),
    }
    for cid in sorted(np.unique(clusters)):
        cluster_idx = np.flatnonzero(clusters == cid)
        cluster_x = x[cluster_idx]
        cluster_y = y[cluster_idx]
        center_x = np.median(cluster_x)
        center_y = np.median(cluster_y)
        nearest = np.argmin((cluster_x - center_x) ** 2 + (cluster_y - center_y) ** 2)
        label = ax_a.annotate(
            str(cid + 1),
            xy=(cluster_x[nearest], cluster_y[nearest]),
            xytext=label_offsets.get(cid, (0, 0)),
            textcoords="offset points", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="black", zorder=5,
        )
        label.set_path_effects([
            path_effects.Stroke(linewidth=2.0, foreground="white"),
            path_effects.Normal(),
        ])

    ax_a.set_xlabel("UMAP 1", fontsize=layout.body_font)
    ax_a.set_ylabel("UMAP 2", fontsize=layout.body_font)
    ax_a.tick_params(labelbottom=False, labelleft=False)
    ax_a.set_title("(a) Cluster assignments", fontweight="bold", loc="left",
                    fontsize=layout.title_font)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ── Legend column (flush against panel a) ────────────────────────────
    unique_clusters = sorted(np.unique(clusters))
    legend_patches = []
    for cid in unique_clusters:
        cnt = int((clusters == cid).sum())
        pct = 100.0 * cnt / len(clusters)
        color = cmap(cid / max(n_clusters - 1, 1))
        legend_patches.append(
            Patch(facecolor=color, edgecolor="k", linewidth=0.3,
                  label=f"{cid + 1}: {pct:.2f}%")
        )
    ax_legend.legend(
        handles=legend_patches,
        title="Clusters (a)",
        loc="center left",
        ncol=1,
        fontsize=max(6.5, layout.tick_font - 1.0),
        title_fontsize=layout.tick_font,
        frameon=True,
        fancybox=False,
        edgecolor="k",
        framealpha=0.7,
        handlelength=0.8,
        handletextpad=0.3,
        labelspacing=0.2,
        borderpad=0.4,
    )

    # ── Panel (b): sampled vs full library ──────────────────────────────
    is_sampled = df["CifId"].isin(sampled_ids).values
    n_all = len(df)
    n_sampled = int(is_sampled.sum())
    is_unsampled = ~is_sampled
    n_unsampled = n_all - n_sampled

    ax_b.scatter(
        x[is_unsampled], y[is_unsampled],
        c="#AAAAAA", s=layout.marker_area * 0.45, alpha=0.55,
        edgecolors="none", rasterized=True, zorder=1,
    )

    highlight_color = "#E41A1C"
    ax_b.scatter(
        x[is_sampled], y[is_sampled],
        c=highlight_color, s=layout.marker_area * 0.08, alpha=0.45,
        edgecolors="none", rasterized=True, zorder=2,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#AAAAAA",
               markeredgecolor="none", markersize=4, linestyle="None",
               label=f"Unsampled ({n_unsampled:,})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=highlight_color,
               markeredgecolor="none", markersize=4, linestyle="None",
               label=f"Sampled ({n_sampled:,})"),
    ]
    ax_b.legend(
        handles=legend_handles,
        title="Sampling (b)",
        loc="upper right",
        fontsize=layout.tick_font - 1,
        title_fontsize=layout.tick_font,
        frameon=True,
        fancybox=False,
        edgecolor="k",
        framealpha=0.7,
        handletextpad=0.3,
    )

    ax_b.set_xlabel("UMAP 1", fontsize=layout.body_font)
    # Extend x-axis to give legend more room
    xmin_b, xmax_b = ax_b.get_xlim()
    ax_b.set_xlim(xmin_b, xmax_b + (xmax_b - xmin_b) * 0.12)
    # No y-label on panel (b) — same coordinate space as panel (a)
    ax_b.tick_params(labelbottom=False, labelleft=False)
    ax_b.set_title("(b) Stratified sample", fontweight="bold", loc="left",
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
