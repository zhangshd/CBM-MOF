"""
lsv_dist_analysis.py
====================
Plot LSV_norm distributions (composite + per-target) for train vs val+test.

Generates lsv_norm_distribution.png showing histograms with labelled vertical
lines for: train mean, val+test mean, val+test p80, val+test p90.

Usage:
    conda run -n mofmthnn python src/alignn/lsv_dist_analysis.py \\
        --deployment-dir results/alignn/ep100_deployment \\
        --output-dir results/alignn/ep100_deployment
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "alignn"))
sys.path.insert(0, str(REPO_ROOT / "src" / "figures"))

from compute_uq import TARGET_COLS, compute_lsv
from style import set_publication_style, DOUBLE_COL_INCH, DPI, MODEL_COLORS

import faiss

# Friendly target labels for figure axes
TARGET_LABELS = {
    "AdsCH4_10kPa":   "CH$_4$@10 kPa",
    "AdsCH4_100kPa":  "CH$_4$@100 kPa",
    "AdsCH4_1000kPa": "CH$_4$@1000 kPa",
    "AdsN2_10kPa":    "N$_2$@10 kPa",
    "AdsN2_100kPa":   "N$_2$@100 kPa",
    "AdsN2_1000kPa":  "N$_2$@1000 kPa",
    "QstCH4":         "Q$_{\\rm st}$(CH$_4$)",
    "QstN2":          "Q$_{\\rm st}$(N$_2$)",
}


def plot_distributions(
    lsv_train: np.ndarray,
    lsv_val: np.ndarray,
    lsv_test: np.ndarray,
    out_path: Path,
) -> None:
    """
    Plot per-target + composite LSV_norm histograms.

    Vertical lines: train mean (green), val+test mean (orange),
    val+test p80 (orange dashed), val+test p90 (red dotted).
    """
    set_publication_style()

    GREEN  = MODEL_COLORS["ALIGNN"]
    ORANGE = "#E07B00"
    RED    = "#CC4125"

    lsv_vt = np.vstack([lsv_val, lsv_test])      # (N_val+N_test, T)
    composite_train = lsv_train.mean(axis=1)
    composite_vt    = lsv_vt.mean(axis=1)

    T    = len(TARGET_COLS)
    ncols = 4
    nrows = (T + ncols - 1) // ncols + 1          # first row = composite

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(DOUBLE_COL_INCH, DOUBLE_COL_INCH * 0.55 * nrows),
    )

    def _draw_panel(ax, tr_vals, vt_vals, title):
        xmax = max(np.percentile(tr_vals, 99.5), np.percentile(vt_vals, 99.5)) * 1.1
        bins = np.linspace(0, xmax, 55)
        ax.hist(tr_vals, bins=bins, density=True, color=GREEN,  alpha=0.50, label="Train")
        ax.hist(vt_vals, bins=bins, density=True, color=ORANGE, alpha=0.50, label="Val+Test")

        tr_mean = tr_vals.mean()
        vt_mean = vt_vals.mean()
        vt_p80  = np.percentile(vt_vals, 80)
        vt_p90  = np.percentile(vt_vals, 90)

        ax.axvline(tr_mean, color=GREEN,  lw=1.0, ls="--",
                   label=f"Train $\\mu$={tr_mean:.2f}")
        ax.axvline(vt_mean, color=ORANGE, lw=1.0, ls="--",
                   label=f"V+T $\\mu$={vt_mean:.2f}")
        ax.axvline(vt_p80,  color=RED,    lw=0.9, ls=":",
                   label=f"V+T p80={vt_p80:.2f}")
        ax.axvline(vt_p90,  color=RED,    lw=0.9, ls="-.",
                   label=f"V+T p90={vt_p90:.2f}")

        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("LSV$_{\\rm norm}$", fontsize=5.5)
        ax.set_ylabel("Density", fontsize=5.5)
        ax.tick_params(labelsize=5)
        leg = ax.legend(fontsize=4, frameon=True, loc="upper right")
        leg.get_frame().set_linewidth(0.2)

    # Row 0: composite
    _draw_panel(axes[0, 0], composite_train, composite_vt, "Composite LSV$_{\\rm norm}$")
    for j in range(1, ncols):
        axes[0, j].set_visible(False)

    # Rows 1+: per target
    for i, col in enumerate(TARGET_COLS):
        row = 1 + i // ncols
        ci  = i % ncols
        _draw_panel(
            axes[row, ci],
            lsv_train[:, i],
            lsv_vt[:, i],
            TARGET_LABELS[col],
        )

    # Hide trailing empty axes
    for idx in range(T, nrows * ncols - ncols):
        row = 1 + idx // ncols
        ci  = idx % ncols
        if row < nrows:
            axes[row, ci].set_visible(False)

    fig.suptitle(
        "LSV$_{\\rm norm}$ = LSV / mean(LSV$_{\\rm train}$) — ALIGNN ep100, k=10\n"
        "[Train mean ≈ 1  →  Val+Test mean ≈ 1 ✓]",
        fontsize=7, y=1.01,
    )
    fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LSV_norm distribution plots")
    parser.add_argument("--deployment-dir", type=Path,
                        default=REPO_ROOT / "results/alignn/ep100_deployment",
                        help="Directory with uncertainty_trees.pkl and latent feature npz files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to deployment-dir)")
    args = parser.parse_args()

    d = args.deployment_dir
    out_dir = args.output_dir or d
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load UQ trees ─────────────────────────────────────────────────────────
    print(f"Loading uncertainty_trees.pkl from {d} ...")
    with open(d / "uncertainty_trees.pkl", "rb") as f:
        payload = pickle.load(f)

    index             = faiss.deserialize_index(payload["index_bytes"])
    train_labels      = payload["train_labels_orig"]
    baseline_dist     = payload["baseline_dist"]
    k                 = payload["k"]
    baseline_lsv_mean = payload["baseline_lsv_mean"]

    # ── Load embeddings for all splits ────────────────────────────────────────
    def load_emb(name: str) -> np.ndarray:
        return np.load(d / f"{name}_latent_features.npz",
                       allow_pickle=True)["features"].astype("float32")

    train_emb = load_emb("train")
    val_emb   = load_emb("val")
    test_emb  = load_emb("test")

    print(f"  Splits: train={len(train_emb)}, val={len(val_emb)}, test={len(test_emb)}")

    # ── Compute LSV_norm for all splits ───────────────────────────────────────
    def lsv(emb):
        return compute_lsv(emb, train_labels, index, baseline_dist,
                           k=k, baseline_lsv_mean=baseline_lsv_mean)

    lsv_train = lsv(train_emb)
    lsv_val   = lsv(val_emb)
    lsv_test  = lsv(test_emb)

    # ── Print stats ───────────────────────────────────────────────────────────
    comp_train = lsv_train.mean(axis=1)
    comp_vt    = np.vstack([lsv_val, lsv_test]).mean(axis=1)

    print(f"\n=== Composite LSV_norm ===")
    print(f"  train    mean={comp_train.mean():.4f}  p80={np.percentile(comp_train,80):.4f}  "
          f"p90={np.percentile(comp_train,90):.4f}")
    print(f"  val+test mean={comp_vt.mean():.4f}  p80={np.percentile(comp_vt,80):.4f}  "
          f"p90={np.percentile(comp_vt,90):.4f}")

    print(f"\n=== Per-target LSV_norm (val+test) ===")
    lsv_vt = np.vstack([lsv_val, lsv_test])
    for i, col in enumerate(TARGET_COLS):
        v = lsv_vt[:, i]
        print(f"  {col:25s}  mean={v.mean():.4f}  p80={np.percentile(v,80):.4f}  "
              f"p90={np.percentile(v,90):.4f}")

    # ── Generate figure ───────────────────────────────────────────────────────
    out_fig = out_dir / "lsv_norm_distribution.png"
    plot_distributions(lsv_train, lsv_val, lsv_test, out_fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
