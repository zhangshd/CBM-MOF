"""Canonical UQ manuscript figure entrypoint for Figure S7/S8 and threshold support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignn.common.constants import TARGET_COLS
from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    DPI,
    LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    MODEL_COLORS,
    NATURE_COLORS,
    TASK_LABELS,
    TASK_LIST,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _save_path_figure(fig, out_path: Path) -> None:
    """Save a figure to an explicit path using publication defaults."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_calibration(
    lsv: np.ndarray,
    pred_orig: np.ndarray,
    truth_orig: np.ndarray,
    calib_results: dict,
    out_path: Path,
) -> None:
    """Scatter plot of per-target LSV vs absolute error."""
    set_publication_style()
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_INCH * 1.15, DOUBLE_COL_INCH * 0.72))
    axes = axes.ravel()

    for idx, col in enumerate(TARGET_COLS):
        ax = axes[idx]
        lsv_vals = lsv[:, idx]
        abs_error = np.abs(pred_orig[:, idx] - truth_orig[:, idx])
        mask = np.isfinite(lsv_vals) & np.isfinite(abs_error)
        ax.scatter(
            lsv_vals[mask],
            abs_error[mask],
            s=4,
            alpha=0.35,
            edgecolors="none",
            color=MODEL_COLORS["ALIGNN"],
        )
        ax.set_xlabel("LSV$_{\\rm norm}$")
        ax.set_ylabel("|Error|")
        ax.set_title(TASK_LABELS.get(col, col), fontsize=7)
        rho = calib_results[col]["rho"]
        label = f"$\\rho$={rho:.3f}" if rho is not None else "N/A"
        ax.text(0.98, 0.02, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

    fig.tight_layout()
    _save_path_figure(fig, out_path)


def plot_k_sweep(k_sweep: dict[int, dict[str, float | None]], out_path: Path) -> None:
    """Plot Spearman rho vs k for every target."""
    set_publication_style()
    k_values = sorted(k_sweep)
    xtick_labels = ["" if k == 3 else str(k) for k in k_values]
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_INCH, 3.8))
    axes = axes.ravel()
    for idx, col in enumerate(TARGET_COLS):
        ax = axes[idx]
        rho_values = [k_sweep[k].get(col) for k in k_values]
        valid_pairs = [(k, rho) for k, rho in zip(k_values, rho_values) if rho is not None]
        if valid_pairs:
            x_vals, y_vals = zip(*valid_pairs)
            ax.plot(x_vals, y_vals, color=MODEL_COLORS["ALIGNN"], lw=1.0, marker="D", ms=3.5)
        label_char = chr(ord("a") + idx)
        ax.set_title(f"({label_char}) {TASK_LABELS.get(col, col)}", fontsize=TITLE_FONT_SIZE, fontweight="bold")
        if idx >= 4:
            ax.set_xlabel("$k$", fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_xlabel("")
        if idx % 4 == 0:
            ax.set_ylabel("Spearman $\\rho$", fontsize=LABEL_FONT_SIZE)
        else:
            ax.set_ylabel("")
        ax.set_xticks(k_values)
        ax.set_xticklabels(xtick_labels)
        ax.set_ylim(-0.05, 1.0)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_path_figure(fig, out_path)


def _extract_sr_curve(sr_payload: dict, lsv_composite: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map percentile-based SR payload to actual LSV thresholds."""
    ordered = sorted((int(k), v) for k, v in sr_payload.items())
    pcts = np.array([item[0] for item in ordered], dtype=float)
    sr = np.array(
        [item[1]["sr"] if item[1]["sr"] is not None else np.nan for item in ordered],
        dtype=float,
    )
    retention = np.array([item[1]["retention"] for item in ordered], dtype=float)

    lsv_thresholds = np.array(
        [
            np.percentile(lsv_composite, p)
            if 0 < p < 100
            else (lsv_composite.min() if p == 0 else lsv_composite.max())
            for p in pcts
        ]
    )
    mask = np.isfinite(sr)
    return lsv_thresholds[mask], sr[mask], retention[mask]


def _load_calibration_lsv_composite(uq_dir: Path) -> np.ndarray:
    """Load pre-computed composite LSV_norm for the val+test calibration set."""
    npy_path = uq_dir / "calibration_lsv_composite.npy"
    return np.load(npy_path)


def plot_sr_axis(ax, payload: dict, lsv_composite: np.ndarray) -> None:
    """Plot the SR sweep on an existing axis using actual LSV_norm cutoffs."""
    lsv_x, sr_valid, ret_valid = _extract_sr_curve(payload["sr_sweep"], lsv_composite)
    grey = "#888888"

    # SR curve (left y-axis)
    ax.plot(
        lsv_x,
        sr_valid,
        color=MODEL_COLORS["ALIGNN"],
        lw=1.2,
        marker="D",
        ms=5.0,
        alpha=0.85,
    )
    ax.set_xlabel(r"LSV$_{norm}$ cutoff", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Separation ratio", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    sr_span = sr_valid.max() - sr_valid.min()
    sr_pad = max(0.08, 0.12 * sr_span)
    ax.set_ylim(0.0, sr_valid.max() + sr_pad)

    # Retention curve (right y-axis)
    ax2 = ax.twinx()
    ax2.fill_between(lsv_x, ret_valid * 100, alpha=0.12, color=grey, zorder=0)
    ax2.plot(
        lsv_x,
        ret_valid * 100,
        color=grey,
        lw=1.0,
        marker="o",
        ms=5.0,
        alpha=0.7,
    )
    ax2.set_ylabel("Retention (%)", fontsize=LABEL_FONT_SIZE, color=grey)
    ax2.tick_params(labelsize=TICK_FONT_SIZE, labelcolor=grey)
    ax2.set_ylim(-5, 105)


def plot_sr_analysis(payload: dict, lsv_composite: np.ndarray, out_path: Path) -> None:
    """Generate canonical SR-analysis figure (current Figure S8)."""
    layout = compute_panel_grid_layout(nrows=1, ncols=1, figure_width_inch=DOUBLE_COL_INCH)
    fig_w = DOUBLE_COL_INCH * 0.85
    fig_h = fig_w * 8.0 / 14.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
    )
    plot_sr_axis(ax, payload, lsv_composite)
    ax.set_xlabel("Mean LSV$_{\\rm norm}$ cutoff", fontsize=layout.body_font)
    ax.set_ylabel(
        r"Separation ratio ($\mathrm{MAE}_{out}/\mathrm{MAE}_{in}$)",
        fontsize=layout.body_font,
        color=MODEL_COLORS["ALIGNN"],
    )
    ax.tick_params(axis="both", labelsize=layout.tick_font)
    ax.tick_params(axis="y", labelcolor=MODEL_COLORS["ALIGNN"])
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.figure.axes[-1].spines["top"].set_visible(False)
    _save_path_figure(fig, out_path)


def plot_distribution_panel(lsv_train: np.ndarray, lsv_val: np.ndarray, lsv_test: np.ndarray, out_path: Path) -> None:
    """Plot composite and per-target normalized LSV distributions."""
    set_publication_style()
    green = MODEL_COLORS["ALIGNN"]
    orange = "#E07B00"
    red = "#CC4125"

    lsv_vt = np.vstack([lsv_val, lsv_test])
    composite_train = lsv_train.mean(axis=1)
    composite_vt = lsv_vt.mean(axis=1)

    ncols = 4
    nrows = (len(TARGET_COLS) + ncols - 1) // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COL_INCH, DOUBLE_COL_INCH * 0.55 * nrows))

    def _draw_panel(ax, train_vals, vt_vals, title):
        xmax = max(np.percentile(train_vals, 99.5), np.percentile(vt_vals, 99.5)) * 1.1
        bins = np.linspace(0, xmax, 55)
        ax.hist(train_vals, bins=bins, density=True, color=green, alpha=0.50, label="Train")
        ax.hist(vt_vals, bins=bins, density=True, color=orange, alpha=0.50, label="Val+Test")
        ax.axvline(np.mean(train_vals), color=green, lw=1.0, ls="--")
        ax.axvline(np.mean(vt_vals), color=orange, lw=1.0, ls="--")
        ax.axvline(np.percentile(vt_vals, 80), color=red, lw=0.9, ls=":")
        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("LSV$_{\\rm norm}$", fontsize=5.5)
        ax.set_ylabel("Density", fontsize=5.5)
        ax.tick_params(labelsize=5)

    _draw_panel(axes[0, 0], composite_train, composite_vt, "Composite LSV$_{\\rm norm}$")
    for idx in range(1, ncols):
        axes[0, idx].set_visible(False)

    for idx, col in enumerate(TARGET_COLS):
        row = 1 + idx // ncols
        col_index = idx % ncols
        _draw_panel(axes[row, col_index], lsv_train[:, idx], lsv_vt[:, idx], TASK_LABELS.get(col, col))

    fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.5)
    _save_path_figure(fig, out_path)


def generate_assets(
    output_dir: Path,
    threshold_csv: Path,
    uq_dir: Path,
) -> None:
    """Generate canonical UQ manuscript figures and threshold support table."""
    set_publication_style()
    payload = load_threshold_payload(uq_dir)

    # Compute calibration LSV composite for x-axis mapping
    lsv_composite = _load_calibration_lsv_composite(uq_dir)

    uq_json = uq_dir / "k_sensitivity_sweep.json"
    k_payload = json.loads(uq_json.read_text())
    k_sweep = {int(k): values for k, values in k_payload["k_sweep"].items()}

    # Canonical SI Figures S7 and S8
    plot_k_sweep(k_sweep, output_dir / "fig_k_sensitivity.png")
    plot_sr_analysis(payload, lsv_composite, output_dir / "fig_sr_analysis.png")

    # Threshold CSV for SI Note S2
    export_threshold_table(threshold_csv, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate canonical UQ manuscript figures (Figure S7/S8) and threshold support table."
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
        uq_dir=args.uq_dir,
    )
    print("Done: canonical Figure S7/S8 + threshold CSV generated.")


if __name__ == "__main__":
    main()
