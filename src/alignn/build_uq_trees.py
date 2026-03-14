"""
build_uq_trees.py
=================
Task 1.1b: Build faiss UQ index from pre-computed latent embeddings and
validate via Spearman rho calibration.

Reads pre-computed results from Task 1.1a (npz + csv), builds a faiss
IndexFlatL2, computes LSV calibration, and serialises uncertainty_trees.pkl
for use in Task 1.1c (full-library screening).

Usage:
    conda activate mofmthnn
    cd /home/zhangsd/repos/CBM-MOF
    python src/alignn/build_uq_trees.py \\
        --input-dir  results/alignn/model_ep150/deployment \\
        --output-dir results/alignn/model_ep150/uq \\
        --k 10
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

# Add src/alignn to path so we can import from compute_uq
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
# Add src/figures to path for publication style
sys.path.insert(0, str(_SCRIPT_DIR.parent / "figures"))

from compute_uq import (
    TARGET_COLS,
    MIN_SPEARMAN_RHO,
    build_faiss_index,
    compute_baseline_dist,
    compute_lsv,
    calibrate_uq,
    plot_calibration,
)
from style import (
    SINGLE_COL_INCH,
    DOUBLE_COL_INCH,
    DPI,
    TASK_LABELS,
    TASK_UNITS,
    MODEL_COLORS,
    set_publication_style,
)

# Colour palette for LSV cutoff plots
_CLR_IN  = MODEL_COLORS["ALIGNN"]      # green: low-UQ (retained)
_CLR_OUT = "#CC4125"                   # red-ish: high-UQ (filtered)
_CLR_RET = "#7F7F7F"                   # grey: retention fraction fill


# ── Data loading ───────────────────────────────────────────────────────────────

def load_split(data_dir: Path, split: str) -> dict:
    """
    Load pre-computed embeddings + predictions/groundtruth from Task 1.1a.

    Returns dict with keys: mol_ids, features, preds, truths.
    All arrays in physical (original) space.
    """
    npz  = np.load(data_dir / f"{split}_latent_features.npz", allow_pickle=True)
    pred = pd.read_csv(data_dir / f"{split}_predictions.csv")
    gt   = pd.read_csv(data_dir / f"{split}_groundtruth.csv")

    mol_ids  = list(npz["mol_ids"])
    features = npz["features"].astype(np.float32)       # (N, 256)

    pred_arr = pred[TARGET_COLS].values.astype(np.float32)   # (N, 8)
    true_arr = gt[TARGET_COLS].values.astype(np.float32)     # (N, 8)

    assert features.shape[0] == pred_arr.shape[0], \
        f"[{split}] features/predictions row mismatch: {features.shape[0]} vs {pred_arr.shape[0]}"

    print(f"  [{split}] features={features.shape}, n_targets={pred_arr.shape[1]}")
    return {"mol_ids": mol_ids, "features": features,
            "preds": pred_arr, "truths": true_arr}


# ── Serialisation ──────────────────────────────────────────────────────────────

def save_uncertainty_trees(
    index_cpu,
    train_labels_orig: np.ndarray,
    baseline_dist: float,
    baseline_lsv_mean: np.ndarray,
    k: int,
    embedding_dim: int,
    out_path: Path,
) -> None:
    """
    Serialise faiss CPU index + metadata to pickle.

    Uses faiss.serialize_index so the pkl is portable (no GPU faiss needed
    at load time).

    baseline_lsv_mean: (T,) per-target mean raw LSV of the training set.
        Used to normalise query LSV scores: LSV_norm_t = LSV_t / mean(LSV_train_t).
        Training samples average LSV_norm ≈ 1; OOD samples >> 1.
    """
    import faiss

    index_bytes = faiss.serialize_index(index_cpu)
    payload = {
        "index_bytes":       index_bytes,
        "train_labels_orig": train_labels_orig,
        "baseline_dist":     baseline_dist,
        "baseline_lsv_mean": baseline_lsv_mean,
        "k":                 k,
        "embedding_dim":     embedding_dim,
        "target_cols":       TARGET_COLS,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"  uncertainty_trees.pkl saved: {out_path}  "
          f"(index_bytes={len(index_bytes)} B, "
          f"train_labels={train_labels_orig.shape})")


def verify_pkl(out_path: Path) -> bool:
    """Load pkl and do a smoke-test nearest-neighbour search."""
    import faiss

    with open(out_path, "rb") as f:
        payload = pickle.load(f)
    idx = faiss.deserialize_index(payload["index_bytes"])
    dim = payload["embedding_dim"]
    probe = np.random.randn(1, dim).astype(np.float32)
    D, I = idx.search(probe, 1)
    print(f"  PKL smoke-test: search OK — nearest dist={D[0,0]:.4f}, idx={I[0,0]}")
    return True


# ── PCA visualisation ─────────────────────────────────────────────────────────

def plot_pca_by_targets(
    all_features: np.ndarray,
    all_truths:   np.ndarray,
    out_path: Path,
) -> None:
    """
    2D PCA scatter coloured by each of the 8 target values.

    all_features : (N_total, 256)
    all_truths   : (N_total, 8)
    """
    print("  Computing PCA 2D...")
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_features)   # (N, 2)

    var_exp = pca.explained_variance_ratio_ * 100
    print(f"  PCA variance explained: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%")

    fig, axes = plt.subplots(2, 4,
                             figsize=(DOUBLE_COL_INCH * 1.35, DOUBLE_COL_INCH * 0.75))
    axes = axes.ravel()

    for i, col in enumerate(TARGET_COLS):
        ax   = axes[i]
        vals = all_truths[:, i]
        mask = np.isfinite(vals)
        v    = vals.copy()

        use_log = col.startswith("Ads") and np.all(v[mask] > 0)
        if use_log:
            v = np.where(mask & (v > 0), np.log10(v), np.nan)

        sc = ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=v[mask], s=2, alpha=0.35,
            cmap="viridis", edgecolors="none",
            vmin=np.nanpercentile(v[mask], 2),
            vmax=np.nanpercentile(v[mask], 98),
            rasterized=True,
        )
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5.5)

        unit  = TASK_UNITS.get(col, "")
        label = TASK_LABELS.get(col, col)
        suffix = r" (log$_{10}$)" if use_log else f" ({unit})" if unit else ""
        ax.set_title(f"{label}{suffix}", fontsize=7)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)", fontsize=6.5)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)", fontsize=6.5)

    fig.suptitle(
        "Latent Space PCA for ALIGNN ep150 (train + val + test, colored by target)",
        fontsize=8,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  PCA plot saved: {out_path}")


# ── LSV cutoff scan (3 metric variants) ───────────────────────────────────────

def _compute_r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination R² on a (N, T) subset."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _compute_mape(true: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%), skipping near-zero denominators."""
    denom = np.abs(true)
    valid = denom > eps
    if valid.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs(true[valid] - pred[valid]) / denom[valid]))


def plot_lsv_cutoff_scan(
    lsv:    np.ndarray,
    preds:  np.ndarray,
    truths: np.ndarray,
    metric: str,
    out_path: Path,
) -> None:
    """
    Sweep LSV percentile cutoff and plot a quality metric vs retention.

    Parameters
    ----------
    metric : "MAE" | "R2" | "MAPE"
    """
    assert metric in ("MAE", "R2", "MAPE"), f"Unknown metric: {metric}"

    lsv_score  = lsv.mean(axis=1)        # (N,) composite score
    # Include endpoints 0 and 100 to show the full range (no filtering → all filtered)
    percentiles = np.concatenate([[0], np.arange(5, 100, 5), [100]])
    # Minimum sample count for a meaningful metric (avoid noisy single-sample estimates)
    MIN_SAMPLES = 10

    y_in, y_out, retain = [], [], []

    for pct in percentiles:
        thresh = np.percentile(lsv_score, pct)
        lo     = lsv_score <= thresh     # confident (retained)
        hi     = ~lo

        if metric == "MAE":
            y_in.append(np.abs(preds[lo]  - truths[lo]).mean()  if lo.sum()  >= MIN_SAMPLES else np.nan)
            y_out.append(np.abs(preds[hi] - truths[hi]).mean()  if hi.sum()  >= MIN_SAMPLES else np.nan)
        elif metric == "R2":
            y_in.append(_compute_r2(truths[lo],  preds[lo])  if lo.sum()  >= MIN_SAMPLES else np.nan)
            y_out.append(_compute_r2(truths[hi], preds[hi])  if hi.sum()  >= MIN_SAMPLES else np.nan)
        elif metric == "MAPE":
            y_in.append(_compute_mape(truths[lo],  preds[lo])  if lo.sum()  >= MIN_SAMPLES else np.nan)
            y_out.append(_compute_mape(truths[hi], preds[hi])  if hi.sum()  >= MIN_SAMPLES else np.nan)

        retain.append(lo.mean())

    y_in  = np.array(y_in,  dtype=float)
    y_out = np.array(y_out, dtype=float)
    retain = np.array(retain, dtype=float)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(SINGLE_COL_INCH * 1.85, SINGLE_COL_INCH * 1.25))
    ax2 = ax1.twinx()

    # Shade retention area
    ax2.fill_between(percentiles, retain, alpha=0.12, color=_CLR_RET, zorder=0)
    ax2.plot(percentiles, retain, color=_CLR_RET, lw=0.8, alpha=0.7, zorder=1)
    ax2.set_ylabel("Retention fraction", color=_CLR_RET, fontsize=7)
    ax2.set_ylim(0, 1.15)
    ax2.tick_params(axis="y", labelcolor=_CLR_RET, labelsize=6.5)

    # Metric lines
    ax1.plot(percentiles, y_in,  color=_CLR_IN,  lw=1.2, marker="o",
             ms=3.5, markerfacecolor=_CLR_IN,  label="Low-UQ (retained)",  zorder=3)
    ax1.plot(percentiles, y_out, color=_CLR_OUT, lw=1.2, marker="s",
             ms=3.5, markerfacecolor=_CLR_OUT, linestyle="--",
             label="High-UQ (filtered)", zorder=3)

    # Mark 80th percentile reference line
    ax1.axvline(80, color="#555555", lw=0.6, linestyle=":", alpha=0.7)

    # Y-axis label
    ylabels = {
        "MAE":  "Mean Absolute Error (avg. 8 targets)",
        "R2":   r"$R^{2}$ (avg. 8 targets)",
        "MAPE": "MAPE (%, avg. 8 targets)",
    }
    ax1.set_xlabel("LSV percentile cutoff")
    ax1.set_ylabel(ylabels[metric])

    # R² y-limits
    if metric == "R2":
        finite = np.concatenate([y_in[np.isfinite(y_in)], y_out[np.isfinite(y_out)]])
        if len(finite):
            ylo = max(-0.1, np.nanmin(finite) - 0.05)
            ax1.set_ylim(ylo, min(1.02, np.nanmax(finite) + 0.05))
    elif metric == "MAPE":
        ax1.set_ylim(bottom=0)

    leg = ax1.legend(frameon=True, fontsize=6, loc="upper left",
                     handlelength=1.4, borderpad=0.5)
    leg.get_frame().set_linewidth(0.4)

    title_map = {"MAE": "MAE", "R2": "R²", "MAPE": "MAPE (%)"}
    ax1.set_title(
        f"LSV Cutoff Scan: {title_map[metric]} vs Retention (ALIGNN ep150, k=current)",
        fontsize=7,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  LSV cutoff scan ({metric}) saved: {out_path}")


# ── k sensitivity sweep ────────────────────────────────────────────────────────

def run_k_sweep(
    query_emb:         np.ndarray,
    train_emb:         np.ndarray,
    train_labels_orig: np.ndarray,
    index_cpu,
    preds:             np.ndarray,
    truths:            np.ndarray,
    k_values:          list,
    out_json:          Path,
    out_png:           Path,
) -> dict:
    """
    Sweep k ∈ k_values; compute Spearman ρ per target for each k.
    LSV normalisation is NOT applied here (sweep evaluates raw rank correlation,
    which is invariant to monotone transforms).
    Returns nested dict: {k: {target: rho}}.
    """
    sweep_results = {}

    for k in k_values:
        bd    = compute_baseline_dist(index_cpu, train_emb, k=k)
        lsv_k = compute_lsv(query_emb, train_labels_orig, index_cpu, bd, k=k)
        rhos  = {}
        for i, col in enumerate(TARGET_COLS):
            err  = np.abs(preds[:, i] - truths[:, i])
            mask = np.isfinite(lsv_k[:, i]) & np.isfinite(err)
            if mask.sum() >= 10:
                rho, _ = spearmanr(lsv_k[mask, i], err[mask])
                rhos[col] = float(rho)
            else:
                rhos[col] = None
        sweep_results[k] = rhos
        n_pass  = sum(1 for v in rhos.values() if v is not None and v > MIN_SPEARMAN_RHO)
        mean_r  = np.mean([v for v in rhos.values() if v is not None])
        print(f"  k={k:2d}  pass={n_pass}/{len(TARGET_COLS)}  mean_ρ={mean_r:.3f}")

    # Save JSON (keys as strings for JSON compatibility)
    with open(out_json, "w") as f:
        json.dump({
            "k_sweep": {str(k): v for k, v in sweep_results.items()},
            "min_rho_threshold": MIN_SPEARMAN_RHO,
        }, f, indent=2)
    print(f"  k-sweep JSON saved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4,
                             figsize=(DOUBLE_COL_INCH * 1.2, DOUBLE_COL_INCH * 0.72))
    axes = axes.ravel()

    for i, col in enumerate(TARGET_COLS):
        ax       = axes[i]
        rho_vals = [sweep_results[k].get(col) for k in k_values]
        valid_k  = [kv for kv, rv in zip(k_values, rho_vals) if rv is not None]
        valid_r  = [rv for rv in rho_vals if rv is not None]

        ax.plot(valid_k, valid_r, color=MODEL_COLORS["ALIGNN"],
                lw=1.0, marker="D", ms=3.5, markerfacecolor=MODEL_COLORS["ALIGNN"])
        ax.axhline(MIN_SPEARMAN_RHO, color="#CC4125", lw=0.7, linestyle="--",
                   label=f"ρ = {MIN_SPEARMAN_RHO}")
        ax.set_xlabel("k")
        ax.set_ylabel("Spearman ρ")
        ax.set_title(TASK_LABELS.get(col, col), fontsize=7)
        ax.set_xticks(k_values)
        ax.set_ylim(-0.05, 1.0)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        leg = ax.legend(fontsize=5.5, loc="lower right", handlelength=1.2)
        leg.get_frame().set_linewidth(0.3)

    fig.suptitle("k-NN Sensitivity Sweep: Spearman ρ per Target (ALIGNN ep150)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  k-sweep plot saved:  {out_png}")

    return sweep_results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir",  type=str, required=True,
                        help="Directory with Task 1.1a outputs (npz + csv)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write outputs")
    parser.add_argument("--k",          type=int, default=10,
                        help="Default k for k-NN LSV (default: 10)")
    parser.add_argument("--skip-pca",   action="store_true",
                        help="Skip PCA visualisation (slower)")
    args = parser.parse_args()

    # Apply publication-quality rcParams globally
    set_publication_style()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k = args.k

    print("=" * 60)
    print("Task 1.1b — Build UQ Trees")
    print("=" * 60)
    print(f"  input_dir  : {input_dir}")
    print(f"  output_dir : {output_dir}")
    print(f"  k          : {k}")

    # ── Step 1: Load pre-computed data ─────────────────────────────────────────
    print("\n[1/6] Loading pre-computed embeddings and predictions...")
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = load_split(input_dir, split)

    train_emb         = splits["train"]["features"]
    train_labels_orig = splits["train"]["truths"]

    # ── Step 2: Build faiss index ──────────────────────────────────────────────
    print("\n[2/6] Building faiss index...")
    import faiss
    index = build_faiss_index(train_emb)

    baseline_dist = compute_baseline_dist(index, train_emb, k=k)
    print(f"  Baseline k-NN L2 distance (training): {baseline_dist:.4f}")

    # Compute raw LSV on the training set itself to derive the normalisation baseline.
    # baseline_lsv_mean_t = mean raw LSV of all training samples for target t.
    # Normalised: LSV_norm_t = LSV_t / baseline_lsv_mean_t
    # → training samples average ≈ 1.0; OOD samples >> 1; semantically self-consistent.
    print("  Computing raw LSV on training set for baseline...")
    lsv_train_raw      = compute_lsv(train_emb, train_labels_orig, index, baseline_dist, k=k)
    baseline_lsv_mean  = lsv_train_raw.mean(axis=0).astype(np.float32)   # (T,)
    print(f"  Baseline mean raw LSV per target (normalisation denominator):")
    for col, v in zip(TARGET_COLS, baseline_lsv_mean):
        print(f"    {col:25s}: {v:.6f}")

    # Get CPU index for serialisation
    try:
        index_cpu = faiss.index_gpu_to_cpu(index)
        print("  Converted GPU index to CPU for serialisation")
    except Exception:
        index_cpu = index  # already CPU

    # ── Step 3: Compute LSV for val + test ────────────────────────────────────
    print("\n[3/6] Computing LSV for val and test splits...")
    for split_name in ("val", "test"):
        sd = splits[split_name]
        sd["lsv"] = compute_lsv(sd["features"], train_labels_orig, index, baseline_dist, k=k,
                                 baseline_lsv_mean=baseline_lsv_mean)

    comb_lsv    = np.vstack([splits["val"]["lsv"],    splits["test"]["lsv"]])
    comb_preds  = np.vstack([splits["val"]["preds"],  splits["test"]["preds"]])
    comb_truths = np.vstack([splits["val"]["truths"],  splits["test"]["truths"]])
    print(f"  Combined val+test: {comb_lsv.shape[0]} samples")

    # ── Step 4: Calibrate (Spearman ρ) ────────────────────────────────────────
    print("\n[4/6] Calibrating UQ (Spearman ρ)...")
    calib_results = calibrate_uq(comb_lsv, comb_preds, comb_truths)

    n_pass = sum(
        1 for v in calib_results.values()
        if v.get("rho") is not None and v["rho"] > MIN_SPEARMAN_RHO
    )
    valid_rhos = [v["rho"] for v in calib_results.values() if v.get("rho") is not None]
    mean_rho   = float(np.mean(valid_rhos)) if valid_rhos else 0.0

    if n_pass < 7:
        print(f"\n  WARNING: Only {n_pass}/8 targets pass ρ > {MIN_SPEARMAN_RHO}. "
              "Proceeding — review calibration.")

    # Save calibration JSON
    calib_summary = {
        "input_dir":           str(input_dir),
        "k":                   k,
        "baseline_dist":       float(baseline_dist),
        "n_train":             int(train_emb.shape[0]),
        "embedding_dim":       int(train_emb.shape[1]),
        "n_val":               int(splits["val"]["features"].shape[0]),
        "n_test":              int(splits["test"]["features"].shape[0]),
        "calibration":         calib_results,
        "n_pass_rho_threshold": n_pass,
        "mean_rho":            mean_rho,
    }
    calib_json = output_dir / "uq_calibration.json"
    with open(calib_json, "w") as f:
        json.dump(calib_summary, f, indent=2)
    print(f"\n  Calibration JSON saved: {calib_json}")

    # ── Step 5: Save uncertainty_trees.pkl ────────────────────────────────────
    print("\n[5/6] Saving uncertainty_trees.pkl...")
    pkl_path = output_dir / "uncertainty_trees.pkl"
    save_uncertainty_trees(
        index_cpu, train_labels_orig,
        baseline_dist, baseline_lsv_mean,
        k, int(train_emb.shape[1]),
        pkl_path,
    )
    verify_pkl(pkl_path)

    # ── Step 5b: Save absolute LSV threshold (80th pct of val+test composite) ──
    # The composite score is the mean of normalised per-target LSV_norm values.
    # The threshold must be fixed on the calibration set (val+test); the same
    # absolute value is used for full-library filtering in Task 1.1d so that
    # the percentile rank is anchored to the calibration distribution.
    lsv_composite = comb_lsv.mean(axis=1)          # (N,) composite score
    thresh_80     = float(np.percentile(lsv_composite, 80))
    n_retained    = int((lsv_composite <= thresh_80).sum())
    n_filtered    = int((lsv_composite >  thresh_80).sum())
    thresholds = {
        "description": (
            "LSV_norm filtering thresholds calibrated on val+test (n=1955). "
            "LSV_norm_t = LSV_t / mean(LSV_train_t) — dimensionless relative uncertainty. "
            "Training samples average LSV_norm ≈ 1; OOD samples >> 1. "
            "Apply composite_threshold to full-library composite LSV_norm scores."
        ),
        "calibration_set":        "val+test combined",
        "n_calibration":          int(len(lsv_composite)),
        "k":                      k,
        "lsv_normalised":         True,
        "percentile":             80,
        "composite_threshold":    thresh_80,
        "composite_retain_fraction": float(n_retained / len(lsv_composite)),
        "baseline_lsv_mean":      {col: float(v) for col, v in zip(TARGET_COLS, baseline_lsv_mean)},
        "note": (
            "Primary filter for Task 1.1d: flag MOFs where "
            "mean(LSV_norm_8targets) > composite_threshold."
        ),
    }
    thresh_path = output_dir / "lsv_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n  LSV threshold (80th pct, composite): {thresh_80:.4f}")
    print(f"  Retained: {n_retained}/{len(lsv_composite)} ({n_retained/len(lsv_composite)*100:.1f}%)")
    print(f"  Thresholds saved: {thresh_path}")

    # ── Step 6: Visualisations ─────────────────────────────────────────────────
    print("\n[6/6] Generating visualisations...")

    # 6a: calibration scatter (uses compute_uq.plot_calibration, style via rcParams)
    plot_calibration(
        comb_lsv, comb_preds, comb_truths, calib_results,
        out_path=output_dir / "uq_calibration.png",
    )

    # 6b: PCA latent space
    if not args.skip_pca:
        plot_pca_by_targets(
            all_features=np.vstack([splits["train"]["features"],
                                    splits["val"]["features"],
                                    splits["test"]["features"]]),
            all_truths=np.vstack([splits["train"]["truths"],
                                  splits["val"]["truths"],
                                  splits["test"]["truths"]]),
            out_path=output_dir / "latent_space_pca_by_targets.png",
        )

    # 6c: LSV cutoff scan — 3 metric variants
    for metric in ("MAE", "R2", "MAPE"):
        stem = f"ALIGNN_ep150_LSV_cutoff_{metric}"
        plot_lsv_cutoff_scan(
            comb_lsv, comb_preds, comb_truths,
            metric=metric,
            out_path=output_dir / f"{stem}.png",
        )

    # 6d: k sensitivity sweep
    K_VALUES = [3, 5, 10, 20, 50]
    print("\n  Running k sensitivity sweep...")
    run_k_sweep(
        query_emb          = np.vstack([splits["val"]["features"],
                                        splits["test"]["features"]]),
        train_emb          = train_emb,
        train_labels_orig  = train_labels_orig,
        index_cpu          = index_cpu,
        preds              = comb_preds,
        truths            = comb_truths,
        k_values          = K_VALUES,
        out_json          = output_dir / "k_sensitivity_sweep.json",
        out_png           = output_dir / "k_sensitivity_sweep.png",
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 1.1b SUMMARY")
    print("=" * 60)
    print(f"  k              : {k}")
    print(f"  baseline_dist  : {baseline_dist:.4f}")
    print(f"  LSV_norm       : enabled (divided by per-target training label variance)")
    print(f"  LSV threshold  : {thresh_80:.4f} (80th pct of val+test composite)")
    print(f"  n_train        : {train_emb.shape[0]}")
    print(f"  embed_dim      : {train_emb.shape[1]}")
    print(f"  Calibration    : {n_pass}/{len(valid_rhos)} targets ρ > {MIN_SPEARMAN_RHO}")
    print(f"  Mean ρ         : {mean_rho:.3f}")
    print(f"\nOutputs in {output_dir}:")
    print(f"  uncertainty_trees.pkl")
    print(f"  uq_calibration.json  (n_pass={n_pass})")
    print(f"  uq_calibration.png")
    if not args.skip_pca:
        print(f"  latent_space_pca_by_targets.png")
    print(f"  ALIGNN_ep150_LSV_cutoff_MAE.png")
    print(f"  ALIGNN_ep150_LSV_cutoff_R2.png")
    print(f"  ALIGNN_ep150_LSV_cutoff_MAPE.png")
    print(f"  k_sensitivity_sweep.json")
    print(f"  k_sensitivity_sweep.png")

    gate_ok = n_pass >= 7
    print(f"\n  Acceptance gate (≥7/8 ρ > {MIN_SPEARMAN_RHO}): {'PASS' if gate_ok else 'FAIL'}")
    print("\nDone.")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
