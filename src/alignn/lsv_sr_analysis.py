"""
lsv_sr_analysis.py
==================
Separation Ratio (SR) sweep analysis for LSV_norm cutoff selection.

SR = MAE_out / MAE_in — measures how much better the retained (low-UQ) group
is compared to the filtered (high-UQ) group as a function of LSV_norm percentile
cutoff. A higher SR means the UQ score is better at separating uncertain samples.

Usage:
    conda run -n mofmthnn python src/alignn/lsv_sr_analysis.py \\
        --deployment-dir results/alignn/ep100_deployment \\
        --output-dir results/alignn/ep100_deployment
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

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "alignn"))
sys.path.insert(0, str(REPO_ROOT / "src" / "figures"))

from compute_uq import TARGET_COLS, compute_lsv
from style import set_publication_style, SINGLE_COL_INCH, DPI, MODEL_COLORS

import faiss


# ── Helpers ───────────────────────────────────────────────────────────────────

def mae(t: np.ndarray, p: np.ndarray, min_n: int = 10) -> float:
    """Mean absolute error; returns NaN if fewer than min_n samples."""
    return float(np.abs(t - p).mean()) if len(t) >= min_n else float("nan")


def run_sr_sweep(
    lsv_score: np.ndarray,
    truths: np.ndarray,
    preds: np.ndarray,
    pcts: np.ndarray,
) -> dict:
    """
    Compute MAE_in, MAE_out, SR, and retention fraction for each percentile cutoff.

    Returns dict with arrays: mae_in, mae_out, sr, retention.
    """
    mae_in_arr, mae_out_arr, sr_arr, ret_arr = [], [], [], []

    for pct in pcts:
        thresh = np.percentile(lsv_score, pct)
        lo = lsv_score <= thresh
        hi = ~lo
        mi = mae(truths[lo], preds[lo])
        mo = mae(truths[hi], preds[hi])
        sr = mo / mi if (np.isfinite(mi) and np.isfinite(mo) and mi > 0) else float("nan")
        mae_in_arr.append(mi)
        mae_out_arr.append(mo)
        sr_arr.append(sr)
        ret_arr.append(float(lo.mean()))

    return {
        "mae_in": np.array(mae_in_arr),
        "mae_out": np.array(mae_out_arr),
        "sr": np.array(sr_arr),
        "retention": np.array(ret_arr),
    }


def plot_sr_panel(
    pcts: np.ndarray,
    sr: np.ndarray,
    recommended_pct: int,
    out_path: Path,
) -> None:
    """Single-panel SR plot with recommended cutoff marker."""
    set_publication_style()

    GREEN  = MODEL_COLORS["ALIGNN"]
    ORANGE = "#E07B00"
    GREY   = "#888888"

    valid = np.isfinite(sr)
    x_sr = pcts[valid]
    y_sr = sr[valid]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_INCH * 1.15, SINGLE_COL_INCH * 0.85))

    ax.plot(x_sr, y_sr, color=GREEN, lw=1.3, marker="D", ms=3.0,
            label="SR = MAE$_{\\rm out}$ / MAE$_{\\rm in}$")
    ax.axvline(recommended_pct, color=ORANGE, lw=1.0, ls="--", alpha=0.9,
               label=f"Recommended (p{recommended_pct})")
    ax.axhline(1.0, color=GREY, lw=0.5, ls=":", alpha=0.6)

    # Annotate SR at recommended percentile
    idx = np.where(pcts == recommended_pct)[0]
    if len(idx) > 0:
        sr_val = sr[idx[0]]
        if np.isfinite(sr_val):
            ax.annotate(
                f"SR = {sr_val:.2f}",
                xy=(recommended_pct, sr_val),
                xytext=(recommended_pct + 5, sr_val + 0.08),
                fontsize=6,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6),
                color=ORANGE,
            )

    ax.set_xlabel("LSV$_{\\rm norm}$ percentile cutoff")
    ax.set_ylabel("Separation Ratio (SR)")
    ax.set_title(
        "LSV$_{\\rm norm}$ SR Analysis — ALIGNN ep100\n"
        "(k=10, val+test, n=1955)",
        fontsize=7,
        pad=4,
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(bottom=0.0)

    leg = ax.legend(fontsize=6, loc="upper left", frameon=True)
    leg.get_frame().set_linewidth(0.3)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  SR figure saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LSV_norm SR sweep analysis")
    parser.add_argument("--deployment-dir", type=Path,
                        default=REPO_ROOT / "results/alignn/ep100_deployment",
                        help="Directory with uncertainty_trees.pkl and prediction CSVs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to deployment-dir)")
    parser.add_argument("--recommended-pct", type=int, default=90,
                        help="Recommended percentile cutoff to annotate (default: 90)")
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

    # ── Load embeddings + predictions ─────────────────────────────────────────
    val_emb  = np.load(d / "val_latent_features.npz",  allow_pickle=True)["features"].astype("float32")
    test_emb = np.load(d / "test_latent_features.npz", allow_pickle=True)["features"].astype("float32")

    lsv_v = compute_lsv(val_emb,  train_labels, index, baseline_dist,
                        k=k, baseline_lsv_mean=baseline_lsv_mean)
    lsv_t = compute_lsv(test_emb, train_labels, index, baseline_dist,
                        k=k, baseline_lsv_mean=baseline_lsv_mean)

    comb_lsv  = np.vstack([lsv_v, lsv_t])           # (N, T)
    lsv_score = comb_lsv.mean(axis=1)               # composite scalar per MOF

    preds  = np.vstack([
        pd.read_csv(d / "val_predictions.csv")[TARGET_COLS].values,
        pd.read_csv(d / "test_predictions.csv")[TARGET_COLS].values,
    ]).astype("float32")
    truths = np.vstack([
        pd.read_csv(d / "val_groundtruth.csv")[TARGET_COLS].values,
        pd.read_csv(d / "test_groundtruth.csv")[TARGET_COLS].values,
    ]).astype("float32")

    # ── Composite LSV_norm distribution stats ─────────────────────────────────
    print(f"\n=== LSV_norm composite distribution (n={len(lsv_score)}) ===")
    print(f"  mean = {lsv_score.mean():.4f}  std = {lsv_score.std():.4f}")
    for pct in [50, 75, 80, 85, 90, 95]:
        print(f"  p{pct:02d}  = {np.percentile(lsv_score, pct):.4f}")

    # ── SR sweep ──────────────────────────────────────────────────────────────
    pcts = np.concatenate([[0], np.arange(5, 100, 5), [100]])
    result = run_sr_sweep(lsv_score, truths, preds, pcts)
    sr      = result["sr"]
    mae_in  = result["mae_in"]
    mae_out = result["mae_out"]
    ret     = result["retention"]

    valid_sr = np.isfinite(sr)
    x_sr, y_sr = pcts[valid_sr], sr[valid_sr]

    print("\n=== SR sweep ===")
    print(f"{'pct':>4}  {'retain%':>7}  {'SR':>6}  {'MAE_in':>8}  {'MAE_out':>8}")
    for i in range(len(pcts)):
        if np.isfinite(sr[i]):
            print(f"  {pcts[i]:3.0f}  {ret[i]*100:7.1f}%  {sr[i]:6.3f}  "
                  f"{mae_in[i]:8.5f}  {mae_out[i]:8.5f}")

    sr_at_rec = sr[pcts == args.recommended_pct]
    sr_at_rec_val = float(sr_at_rec[0]) if len(sr_at_rec) > 0 and np.isfinite(sr_at_rec[0]) else float("nan")
    print(f"\n  SR at p{args.recommended_pct}: {sr_at_rec_val:.3f}")
    print(f"  SR max: {np.nanmax(y_sr):.3f} at pct={x_sr[np.nanargmax(y_sr)]:.0f}")

    # ── Update lsv_thresholds.json ────────────────────────────────────────────
    thresh_file = d / "lsv_thresholds.json"
    if thresh_file.exists():
        with open(thresh_file) as f:
            thresholds = json.load(f)
    else:
        thresholds = {}

    # Per-target threshold at recommended_pct
    per_target_thresh = {
        col: float(np.percentile(comb_lsv[:, i], args.recommended_pct))
        for i, col in enumerate(TARGET_COLS)
    }
    thresh_rec = float(np.percentile(lsv_score, args.recommended_pct))
    retain_rec = float((lsv_score <= thresh_rec).mean())

    thresholds["percentile"]                = args.recommended_pct
    thresholds["composite_threshold"]       = thresh_rec
    thresholds["composite_retain_fraction"] = retain_rec
    thresholds[f"per_target_p{args.recommended_pct}_lsv_norm"] = per_target_thresh
    thresholds["elbow_analysis"]["recommended_pct"]  = args.recommended_pct
    thresholds["elbow_analysis"][f"sr_at_p{args.recommended_pct}"] = float(sr_at_rec_val)
    thresholds["note"] = (
        f"Primary filter for Task 1.1d: flag MOFs where "
        f"mean(LSV_norm_8targets) > composite_threshold (= p{args.recommended_pct} of val+test)."
    )
    thresholds["sr_sweep"] = {
        "pcts":    [int(p) for p in pcts[valid_sr].tolist()],
        "sr":      [round(float(v), 4) for v in y_sr.tolist()],
        "mae_in":  [round(float(mae_in[i]), 6) for i in range(len(pcts)) if valid_sr[i]],
        "mae_out": [round(float(mae_out[i]), 6) for i in range(len(pcts)) if valid_sr[i]],
        "retain":  [round(float(ret[i]), 4) for i in range(len(pcts)) if valid_sr[i]],
    }

    with open(thresh_file, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n  lsv_thresholds.json updated.")

    # ── Generate single-panel SR figure ───────────────────────────────────────
    out_fig = out_dir / "ALN-s1e3-ep100_LSV_elbow_analysis.png"
    plot_sr_panel(pcts, sr, args.recommended_pct, out_fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
