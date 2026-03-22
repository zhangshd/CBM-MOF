"""
Generate UQ validation figures and threshold support table.

Main-text Figure 6: Single-panel SR sweep showing LSV calibration quality.
SI Figure: 8-panel PCA colored by all 8 prediction targets.
"""

from __future__ import annotations

import argparse
import json
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
    SINGLE_COL_INCH,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    MODEL_COLORS,
    NATURE_COLORS,
    PANEL_ORDER,
    TASK_LABELS,
    TASK_LIST,
    TASK_UNITS,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_LABELS = list("abcdefgh")


def load_latent_space_projection(deployment_dir: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load train/val/test embeddings and return their 2D PCA coordinates."""
    from sklearn.decomposition import PCA

    feature_blocks: list[np.ndarray] = []
    truth_frames: list[pd.DataFrame] = []

    for split in ("train", "val", "test"):
        npz_path = deployment_dir / f"{split}_latent_features.npz"
        truth_csv = deployment_dir / f"{split}_groundtruth.csv"
        feature_blocks.append(np.load(npz_path, allow_pickle=True)["features"].astype(np.float32))
        truth_frames.append(pd.read_csv(truth_csv))

    all_features = np.vstack(feature_blocks)
    all_truths = pd.concat(truth_frames, ignore_index=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_features)
    var_exp = pca.explained_variance_ratio_ * 100.0
    return coords, all_truths, var_exp


def load_threshold_payload(uq_dir: Path) -> dict:
    """Load the calibrated threshold payload produced by the UQ scripts."""
    threshold_json = uq_dir / "lsv_thresholds.json"
    with open(threshold_json) as f:
        payload = json.load(f)

    required = ["percentile", "composite_threshold", "composite_retain_fraction", "baseline_lsv_mean", "sr_sweep"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {threshold_json}: {missing}")

    per_target_key = f"per_target_p{payload['percentile']}_lsv_norm"
    if per_target_key not in payload:
        raise KeyError(f"Missing per-target threshold key in {threshold_json}: {per_target_key}")

    return payload


def build_threshold_table(payload: dict) -> pd.DataFrame:
    """Build a CSV-friendly threshold summary table for Note S2 and summaries."""
    percentile = int(payload["percentile"])
    per_target = payload[f"per_target_p{percentile}_lsv_norm"]
    baseline = payload["baseline_lsv_mean"]

    rows = []
    for target in TASK_LIST:
        rows.append(
            {
                "Target": target,
                "BaselineMeanRawLSV": float(baseline[target]),
                f"P{percentile}_LSV_norm": float(per_target[target]),
            }
        )
    return pd.DataFrame(rows)


def export_threshold_table(csv_path: Path, payload: dict) -> None:
    """Export the per-target LSV threshold table."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    build_threshold_table(payload).to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved: {csv_path}")


def plot_pca_panel(
    ax,
    fig,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    var_exp: np.ndarray,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """Plot one PCA panel with robust target coloring."""
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
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=LABEL_FONT_SIZE)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=LABEL_FONT_SIZE)
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


def _load_calibration_lsv_composite(uq_dir: Path) -> np.ndarray:
    """Load pre-computed composite LSV_norm for the val+test calibration set."""
    npy_path = uq_dir / "calibration_lsv_composite.npy"
    return np.load(npy_path)


def plot_sr_panel(ax, payload: dict, lsv_composite: np.ndarray) -> None:
    """Plot the SR sweep with LSV_norm x-axis and retention twin y-axis."""
    sr_payload = payload["sr_sweep"]
    ordered = sorted((int(k), v) for k, v in sr_payload.items())
    pcts = np.array([item[0] for item in ordered], dtype=float)
    sr = np.array([item[1]["sr"] if item[1]["sr"] is not None else np.nan
                   for item in ordered], dtype=float)
    retention = np.array([item[1]["retention"] for item in ordered], dtype=float)

    # Map percentiles to actual LSV_norm values
    lsv_thresholds = np.array([np.percentile(lsv_composite, p) if 0 < p < 100
                               else (lsv_composite.min() if p == 0 else lsv_composite.max())
                               for p in pcts])

    # Filter valid (non-nan SR)
    mask = np.isfinite(sr)
    lsv_x = lsv_thresholds[mask]
    sr_valid = sr[mask]
    ret_valid = retention[mask]

    # SR curve (left y-axis)
    ax.plot(
        lsv_x,
        sr_valid,
        color=MODEL_COLORS["ALIGNN"],
        lw=1.2,
        marker="D",
        ms=2.5,
        label=r"SR = MAE$_{out}$/MAE$_{in}$",
    )
    ax.set_xlabel(r"LSV$_{norm}$ cutoff", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Separation ratio", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_ylim(ax.get_ylim()[0]*0.9, ax.get_ylim()[1]*1.05)

    # Retention curve (right y-axis)
    ax2 = ax.twinx()
    ax2.plot(
        lsv_x,
        ret_valid * 100,
        color=NATURE_COLORS["orange"],
        lw=1.0,
        marker="o",
        ms=2.5,
        alpha=0.8,
        label="Retention (%)",
    )
    ax2.set_ylabel("Retention (%)", fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(labelsize=TICK_FONT_SIZE)
    ax2.set_ylim(-5, 105)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=LEGEND_FONT_SIZE, loc="lower right")


def plot_sr_figure(output_dir: Path, payload: dict,
                   lsv_composite: np.ndarray) -> None:
    """Generate the main-text Figure 6: single-panel SR sweep."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL_INCH, 2.8))
    plot_sr_panel(ax, payload, lsv_composite)
    save_figure(fig, "Figure06_uq_validation", output_dir)
    plt.close(fig)


def plot_si_pca_figure(
    output_dir: Path,
    deployment_dir: Path,
) -> None:
    """Generate the SI 8-panel PCA figure colored by all 8 targets."""
    coords, truth_df, var_exp = load_latent_space_projection(deployment_dir)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(DOUBLE_COL_INCH, 3.8),
    )

    idx = 0
    for row in range(2):
        for col in range(4):
            target = PANEL_ORDER[row][col]
            ax = axes[row, col]
            label_char = PANEL_LABELS[idx]
            title = f"({label_char}) {TASK_LABELS[target]}"

            plot_pca_panel(
                ax=ax,
                fig=fig,
                coords=coords,
                values=truth_df[target].to_numpy(),
                title=title,
                var_exp=var_exp,
                show_xlabel=(row == 1),
                show_ylabel=(col == 0),
            )
            idx += 1

    fig.subplots_adjust(
        left=0.07, right=0.95, bottom=0.10, top=0.93,
        wspace=0.35, hspace=-0.05,
    )
    save_figure(fig, "FigureS_uq_pca_targets", output_dir, tight_layout=False)
    plt.close(fig)


def generate_assets(
    output_dir: Path,
    threshold_csv: Path,
    deployment_dir: Path,
    uq_dir: Path,
) -> None:
    """Generate both the main-text SR figure and SI PCA figure, plus threshold CSV."""
    set_publication_style()
    payload = load_threshold_payload(uq_dir)

    # Compute calibration LSV composite for x-axis mapping
    lsv_composite = _load_calibration_lsv_composite(uq_dir)

    # Main-text Figure 6: single-panel SR sweep
    plot_sr_figure(output_dir, payload, lsv_composite)

    # SI Figure: 8-panel PCA
    plot_si_pca_figure(output_dir, deployment_dir)

    # Threshold CSV for SI Note S2
    export_threshold_table(threshold_csv, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 6 (main-text SR sweep) and SI PCA figure (8-panel latent-space)."
    )
    parser.add_argument(
        "--deployment-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "deployment",
        help="Directory containing the train/val/test latent features and ground truth CSVs.",
    )
    parser.add_argument(
        "--uq-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "uq",
        help="Directory containing the calibrated ep150 UQ outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures",
        help="Output directory for all figures.",
    )
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures" / "LSV_thresholds_ep150.csv",
        help="CSV path for the per-target LSV threshold summary.",
    )
    args = parser.parse_args()

    generate_assets(
        output_dir=args.output_dir,
        threshold_csv=args.threshold_csv,
        deployment_dir=args.deployment_dir,
        uq_dir=args.uq_dir,
    )
    print("Done: Figure 6 (SR sweep) + SI PCA figure + threshold CSV.")


if __name__ == "__main__":
    main()
