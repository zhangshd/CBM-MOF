"""
Generate the main-text UQ validation figure and threshold support table.

Figure layout:
  (a) Latent-space PCA colored by CH4 uptake at 10 kPa
  (b) Latent-space PCA colored by CH4 uptake at 1000 kPa
  (c) Latent-space PCA colored by Qst(CH4)
  (d) SR-based LSV cutoff selection using the calibrated ep150 threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    DPI,
    LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    MODEL_COLORS,
    save_figure,
    set_emphasized_title,
    set_publication_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_COLS = [
    "AdsCH4_10kPa",
    "AdsCH4_100kPa",
    "AdsCH4_1000kPa",
    "AdsN2_10kPa",
    "AdsN2_100kPa",
    "AdsN2_1000kPa",
    "QstCH4",
    "QstN2",
]
PCA_TARGETS = ["AdsCH4_10kPa", "AdsCH4_1000kPa", "QstCH4"]
PANEL_TITLES = {
    "AdsCH4_10kPa": r"(a) CH$_4$ uptake at 10 kPa (mol/kg)",
    "AdsCH4_1000kPa": r"(b) CH$_4$ uptake at 1000 kPa (mol/kg)",
    "QstCH4": r"(c) $Q_{\mathrm{st}}$(CH$_4$) (kJ/mol)",
    "SR": r"(d) SR-based cutoff selection",
}


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
    for target in TARGET_COLS:
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
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    var_exp: np.ndarray,
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
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_aspect("equal", adjustable="box")
    set_emphasized_title(ax, title, fontsize=TITLE_FONT_SIZE)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.048, pad=0.03)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)


def plot_sr_panel(ax, payload: dict) -> None:
    """Plot the SR sweep panel from the calibrated threshold JSON."""
    percentile = int(payload["percentile"])
    threshold = float(payload["composite_threshold"])
    retain_fraction = float(payload["composite_retain_fraction"])
    sr_payload = payload["sr_sweep"]
    if "pcts" in sr_payload:
        pcts = np.array(sr_payload["pcts"], dtype=float)
        sr = np.array(sr_payload["sr"], dtype=float)
    else:
        ordered = sorted((int(k), v) for k, v in sr_payload.items())
        pcts = np.array([item[0] for item in ordered], dtype=float)
        sr = np.array([item[1]["sr"] for item in ordered], dtype=float)

    ax.plot(
        pcts,
        sr,
        color=MODEL_COLORS["ALIGNN"],
        lw=1.2,
        marker="D",
        ms=3.0,
        label=r"SR = MAE$_{\mathrm{out}}$/MAE$_{\mathrm{in}}$",
    )
    ax.axvline(percentile, color="#CC4125", lw=0.9, ls="--", alpha=0.9)
    ax.axhline(1.0, color="#888888", lw=0.5, ls=":", alpha=0.6)
    ax.set_xlim(-2, 102)
    ax.set_xlabel(r"LSV$_{\mathrm{norm}}$ percentile cutoff", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Separation ratio", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    set_emphasized_title(ax, PANEL_TITLES["SR"], fontsize=TITLE_FONT_SIZE)

    sr_at_pct = sr[pcts == percentile]
    sr_value = float(sr_at_pct[0]) if len(sr_at_pct) > 0 else float("nan")
    ax.text(
        0.97,
        0.03,
        (
            f"p{percentile} threshold = {threshold:.3f}\n"
            f"Retention = {retain_fraction * 100:.1f}%\n"
            f"SR = {sr_value:.2f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=LEGEND_FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.88),
    )


def generate_assets(
    output_dir: Path,
    threshold_csv: Path,
    deployment_dir: Path,
    uq_dir: Path,
) -> None:
    """Generate the manuscript UQ figure and threshold CSV."""
    set_publication_style()
    coords, truth_df, var_exp = load_latent_space_projection(deployment_dir)
    payload = load_threshold_payload(uq_dir)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_INCH, 5.55))
    axes = axes.ravel()

    for ax, target in zip(axes[:3], PCA_TARGETS):
        plot_pca_panel(
            ax=ax,
            coords=coords,
            values=truth_df[target].to_numpy(),
            title=PANEL_TITLES[target],
            var_exp=var_exp,
        )

    plot_sr_panel(axes[3], payload)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.94, wspace=0.22, hspace=0.28)

    save_figure(fig, "Figure6", output_dir)
    plt.close(fig)
    export_threshold_table(threshold_csv, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the manuscript UQ validation figure.")
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
        default=PROJECT_ROOT / "results" / "figures",
        help="Output directory for the main-text figure.",
    )
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "summary" / "LSV_thresholds_ep150.csv",
        help="CSV path for the per-target LSV threshold summary.",
    )
    args = parser.parse_args()

    generate_assets(
        output_dir=args.output_dir,
        threshold_csv=args.threshold_csv,
        deployment_dir=args.deployment_dir,
        uq_dir=args.uq_dir,
    )
    print("Done: Figure6 and threshold CSV.")


if __name__ == "__main__":
    main()
