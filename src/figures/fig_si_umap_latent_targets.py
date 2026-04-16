"""
Generate SI Figure S6: 8-panel UMAP projection of latent embeddings colored by targets.

Replaces the former PCA version to be consistent with main-text Figure 2 (UMAP).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    DPI,
    LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    PANEL_ORDER,
    TASK_LABELS,
    save_figure,
    set_publication_style,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_LABELS = list("abcdefgh")


def load_latent_and_truth(deployment_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load and concatenate train/val/test latent features and ground truth."""
    feature_blocks: list[np.ndarray] = []
    truth_frames: list[pd.DataFrame] = []

    for split in ("train", "val", "test"):
        npz_path = deployment_dir / f"{split}_latent_features.npz"
        truth_csv = deployment_dir / f"{split}_groundtruth.csv"
        feature_blocks.append(
            np.load(npz_path, allow_pickle=True)["features"].astype(np.float32)
        )
        truth_frames.append(pd.read_csv(truth_csv))

    all_features = np.vstack(feature_blocks)
    all_truths = pd.concat(truth_frames, ignore_index=True)
    return all_features, all_truths


def compute_umap(features: np.ndarray, n_neighbors: int = 15,
                 min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Compute 2-D UMAP embedding."""
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,          # reproducibility
        verbose=True,
    )
    coords = reducer.fit_transform(features)
    return coords


def plot_umap_panel(
    ax,
    fig,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """Plot one UMAP panel with robust target coloring."""
    mask = np.isfinite(values)
    vmin = np.nanpercentile(values[mask], 2)
    vmax = np.nanpercentile(values[mask], 98)

    scatter = ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        c=values[mask],
        s=3.0,
        alpha=0.45,
        cmap="viridis",
        edgecolors="none",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    if show_xlabel:
        ax.set_xlabel("UMAP-1", fontsize=LABEL_FONT_SIZE)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("UMAP-2", fontsize=LABEL_FONT_SIZE)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.locator = plt.MaxNLocator(nbins=3)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)


def plot_si_umap_figure(
    output_dir: Path,
    deployment_dir: Path,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """Generate the SI 8-panel UMAP figure colored by all 8 targets."""
    print("Loading latent features and ground truth...")
    features, truth_df = load_latent_and_truth(deployment_dir)
    print(f"  Combined shape: features {features.shape}, truth {truth_df.shape}")

    print("Computing UMAP (this may take a minute)...")
    coords = compute_umap(features, n_neighbors=n_neighbors, min_dist=min_dist)
    print(f"  UMAP embedding shape: {coords.shape}")

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(DOUBLE_COL_INCH, 3.5),
    )

    idx = 0
    for row in range(2):
        for col in range(4):
            target = PANEL_ORDER[row][col]
            ax = axes[row, col]
            label_char = PANEL_LABELS[idx]
            title = f"({label_char}) {TASK_LABELS[target]}"

            plot_umap_panel(
                ax=ax,
                fig=fig,
                coords=coords,
                values=truth_df[target].to_numpy(),
                title=title,
                show_xlabel=(row == 1),
                show_ylabel=(col == 0),
            )
            idx += 1

    fig.subplots_adjust(
        left=0.07, right=0.95, bottom=0.08, top=0.95,
        wspace=0.35, hspace=-0.10,
    )
    save_figure(fig, "fig_umap_latent_targets", output_dir, tight_layout=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SI Figure S6: 8-panel UMAP of latent embeddings colored by targets."
    )
    parser.add_argument(
        "--deployment-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "deployment",
        help="Directory containing the train/val/test latent features and ground truth CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures",
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15,
        help="UMAP n_neighbors parameter (default: 15).",
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1,
        help="UMAP min_dist parameter (default: 0.1).",
    )
    args = parser.parse_args()

    set_publication_style()
    plot_si_umap_figure(
        output_dir=args.output_dir,
        deployment_dir=args.deployment_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    print("Done: SI UMAP figure generated.")


if __name__ == "__main__":
    main()
