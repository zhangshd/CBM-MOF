"""Publication plotting helpers for deployment-based UQ calibration."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.alignn.common.constants import TARGET_COLS
from src.figures.style import (
    DPI,
    DOUBLE_COL_INCH,
    MODEL_COLORS,
    SINGLE_COL_INCH,
    TASK_LABELS,
    TASK_UNITS,
    set_publication_style,
)


def save_figure(fig, out_path: Path) -> None:
    """Save a figure using the current publication defaults."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_calibration(lsv: np.ndarray, pred_orig: np.ndarray, truth_orig: np.ndarray, calib_results: dict, out_path: Path) -> None:
    """Scatter plot of per-target LSV vs absolute error."""
    set_publication_style()
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_INCH * 1.15, DOUBLE_COL_INCH * 0.72))
    axes = axes.ravel()

    for idx, col in enumerate(TARGET_COLS):
        ax = axes[idx]
        lsv_vals = lsv[:, idx]
        abs_error = np.abs(pred_orig[:, idx] - truth_orig[:, idx])
        mask = np.isfinite(lsv_vals) & np.isfinite(abs_error)
        ax.scatter(lsv_vals[mask], abs_error[mask], s=4, alpha=0.35, edgecolors="none", color=MODEL_COLORS["ALIGNN"])
        ax.set_xlabel("LSV$_{\\rm norm}$")
        ax.set_ylabel("|Error|")
        ax.set_title(TASK_LABELS.get(col, col), fontsize=7)
        rho = calib_results[col]["rho"]
        label = f"$\\rho$={rho:.3f}" if rho is not None else "N/A"
        ax.text(0.98, 0.02, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=6)

    fig.tight_layout()
    save_figure(fig, out_path)


def plot_pca_by_targets(all_features: np.ndarray, all_truths: np.ndarray, out_path: Path) -> None:
    """2D PCA scatter colored by each target."""
    set_publication_style()
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_features)
    variance = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_INCH * 1.35, DOUBLE_COL_INCH * 0.75))
    axes = axes.ravel()
    for idx, col in enumerate(TARGET_COLS):
        ax = axes[idx]
        values = all_truths[:, idx]
        mask = np.isfinite(values)
        plotted_values = values.copy()
        use_log = col.startswith("Ads") and np.all(values[mask] > 0)
        if use_log:
            plotted_values = np.where(mask & (values > 0), np.log10(values), np.nan)
        scatter = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=plotted_values[mask],
            s=2,
            alpha=0.35,
            cmap="viridis",
            edgecolors="none",
            vmin=np.nanpercentile(plotted_values[mask], 2),
            vmax=np.nanpercentile(plotted_values[mask], 98),
            rasterized=True,
        )
        cb = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5.5)
        unit = TASK_UNITS.get(col, "")
        suffix = r" (log$_{10}$)" if use_log else f" ({unit})" if unit else ""
        ax.set_title(f"{TASK_LABELS.get(col, col)}{suffix}", fontsize=7)
        ax.set_xlabel(f"PC1 ({variance[0]:.1f}%)", fontsize=6.5)
        ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)", fontsize=6.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, out_path)


def plot_k_sweep(k_sweep: dict[int, dict[str, float | None]], out_path: Path) -> None:
    """Plot Spearman rho vs k for every target."""
    from src.figures.style import LABEL_FONT_SIZE, TICK_FONT_SIZE
    set_publication_style()
    k_values = sorted(k_sweep)
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_INCH * 1.2, DOUBLE_COL_INCH * 0.72))
    axes = axes.ravel()
    for idx, col in enumerate(TARGET_COLS):
        ax = axes[idx]
        rho_values = [k_sweep[k].get(col) for k in k_values]
        valid_pairs = [(k, rho) for k, rho in zip(k_values, rho_values) if rho is not None]
        if valid_pairs:
            x_vals, y_vals = zip(*valid_pairs)
            ax.plot(x_vals, y_vals, color=MODEL_COLORS["ALIGNN"], lw=1.0, marker="D", ms=3.5)
        label_char = chr(ord("a") + idx)
        ax.set_title(f"({label_char}) {TASK_LABELS.get(col, col)}", fontsize=LABEL_FONT_SIZE, fontweight="bold", loc="left")
        ax.set_xlabel("$k$", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Spearman $\\rho$", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(k_values)
        ax.set_ylim(-0.05, 1.0)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, out_path)


def plot_sr_panel(sr_sweep: dict, threshold_value: float, recommended_pct: int, out_path: Path) -> None:
    """Plot the SR sweep with the selected percentile threshold highlighted."""
    from src.figures.style import LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE
    set_publication_style()
    green = MODEL_COLORS["ALIGNN"]
    orange = "#E07B00"
    grey = "#888888"

    percentiles = np.asarray(sr_sweep["percentiles"])
    sr = np.asarray(sr_sweep["sr"], dtype=float)
    retention = np.asarray(sr_sweep["retention"], dtype=float)
    valid = np.isfinite(sr)

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL_INCH * 1.15, SINGLE_COL_INCH * 0.85))
    ax2 = ax1.twinx()
    ax2.fill_between(percentiles, retention, alpha=0.12, color=grey, zorder=0)
    ax2.plot(percentiles, retention, color=grey, lw=0.8, alpha=0.7, zorder=1)
    ax2.set_ylabel("Retention fraction", color=grey, fontsize=LABEL_FONT_SIZE)
    ax2.set_ylim(0, 1.15)
    ax2.tick_params(axis="y", labelcolor=grey, labelsize=TICK_FONT_SIZE)

    ax1.plot(percentiles[valid], sr[valid], color=green, lw=1.3, marker="D", ms=3.0)
    ax1.axvline(recommended_pct, color=orange, lw=1.0, ls="--", alpha=0.9)
    ax1.axhline(1.0, color=grey, lw=0.5, ls=":", alpha=0.6)
    ax1.set_xlabel("LSV$_{\\rm norm}$ percentile cutoff", fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel("Separation Ratio (SR)", fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(labelsize=TICK_FONT_SIZE)
    ax1.set_xlim(-2, 102)
    ax1.set_ylim(bottom=0.0)

    index = list(sr_sweep["percentiles"]).index(recommended_pct)
    sr_value = sr_sweep["sr"][index]
    ax1.text(
        0.98,
        0.02,
        f"p{recommended_pct} = {threshold_value:.3f}\nSR = {sr_value:.2f}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=TICK_FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )
    fig.tight_layout(pad=0.5)
    save_figure(fig, out_path)


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
    save_figure(fig, out_path)
