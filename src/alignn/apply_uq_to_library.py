"""
apply_uq_to_library.py
======================
Task 1.1d: Compute LSV uncertainty scores for the full MOF library.

Reads batch NPZ files from Task 1.1c (full_library_inference/batches/),
applies the pre-built faiss UQ index (Task 1.1b), and outputs a single
CSV with per-MOF uncertainty scores and a high-UQ flag.

Algorithm:
  1. Load uncertainty_trees.pkl (faiss index + training labels + baseline)
  2. Stream over all batch_*_features.npz files (avoids loading all at once)
  3. Compute LSV_norm per target: LSV_t / baseline_lsv_mean[t]
  4. Compute composite: mean(LSV_norm) across all 8 targets
  5. Flag high-UQ: lsv_norm_composite > 2.2773 (absolute p90 threshold)

Output:
  results/alignn/full_library_inference/full_library_uq.csv
    columns: mof_id, <target>_lsv_norm (x8), lsv_norm_composite, flag_high_uq

Usage:
    cd /home/zhangsd/repos/CBM-MOF
    conda activate mofmthnn
    python src/alignn/apply_uq_to_library.py \\
        --uq-pkl     results/alignn/ep100_deployment/uncertainty_trees.pkl \\
        --input-dir  results/alignn/full_library_inference \\
        --output-dir results/alignn/full_library_inference

    # Dry-run on first 3 batches only:
    python src/alignn/apply_uq_to_library.py --max-batches 3
"""

import argparse
import pickle
import time
import warnings
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path("/home/zhangsd/repos/CBM-MOF")

UPTAKE_COLS = ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
               "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa"]
QST_COLS    = ["QstCH4", "QstN2"]
TARGET_COLS = UPTAKE_COLS + QST_COLS

# Absolute p90 threshold confirmed in Task 1.1b (DO NOT recompute from full library)
P90_THRESHOLD = 2.2773

LOG_INTERVAL = 10    # print progress every N batches


# ── LSV computation (adapted from compute_uq.py) ──────────────────────────────

def compute_lsv_norm(
    query_emb: np.ndarray,
    train_labels_orig: np.ndarray,
    index,
    baseline_dist: float,
    k: int,
    baseline_lsv_mean: np.ndarray,
) -> np.ndarray:
    """
    Compute normalised LSV (dimensionless) for a batch of query embeddings.

    Returns:
        lsv_norm -- (N, T) float32 dimensionless uncertainty scores
    """
    D, I = index.search(query_emb.astype(np.float32), k)

    sigma2 = max(baseline_dist ** 2, 1e-8)
    w = np.exp(-D / sigma2)                               # (N, k)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)        # row-normalize

    neighbor_labels = train_labels_orig[I]                # (N, k, T)
    weighted_mean   = (w[:, :, None] * neighbor_labels).sum(axis=1)  # (N, T)

    diff = neighbor_labels - weighted_mean[:, None, :]    # (N, k, T)
    lsv  = (w[:, :, None] * diff ** 2).sum(axis=1)        # (N, T)

    lsv_norm = lsv / (baseline_lsv_mean + 1e-12)          # (N, T)
    return lsv_norm.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--uq-pkl",
        type=str,
        default=str(REPO_ROOT / "results/alignn/ep100_deployment/uncertainty_trees.pkl"),
        help="Path to uncertainty_trees.pkl (from Task 1.1b)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(REPO_ROOT / "results/alignn/full_library_inference"),
        help="Directory containing batches/batch_*_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results/alignn/full_library_inference"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=P90_THRESHOLD,
        help=f"Absolute UQ threshold for flag_high_uq (default: {P90_THRESHOLD})",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Process only the first N batches (for dry-run/debug)",
    )
    args = parser.parse_args()

    uq_pkl_path = Path(args.uq_pkl)
    input_dir   = Path(args.input_dir)
    batches_dir = input_dir / "batches"
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("apply_uq_to_library.py  —  Task 1.1d")
    print(f"  UQ pkl        : {uq_pkl_path}")
    print(f"  Input dir     : {batches_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  UQ threshold  : {args.threshold}")
    print(f"  Start time    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── 1. Load UQ index (Task 1.1b output) ───────────────────────────────────
    print("\n[1/3] Loading UQ index...")
    with open(uq_pkl_path, "rb") as f:
        payload = pickle.load(f)

    index             = faiss.deserialize_index(payload["index_bytes"])
    train_labels_orig = payload["train_labels_orig"]   # (N_train, T) physical space
    baseline_dist     = float(payload["baseline_dist"])
    baseline_lsv_mean = np.array(payload["baseline_lsv_mean"], dtype=np.float32)  # (T,)
    k                 = int(payload["k"])
    stored_targets    = payload.get("target_cols", TARGET_COLS)

    print(f"  faiss index   : {index.ntotal} vectors, d={index.d}")
    print(f"  k             : {k}")
    print(f"  baseline_dist : {baseline_dist:.4f}")
    print(f"  baseline_lsv_mean: {baseline_lsv_mean.round(6).tolist()}")

    # Validate target column order matches
    if list(stored_targets) != list(TARGET_COLS):
        raise ValueError(
            f"Target column mismatch!\n  PKL: {stored_targets}\n  Script: {TARGET_COLS}"
        )

    # ── 2. Stream over all batch NPZ files ────────────────────────────────────
    print("\n[2/3] Computing LSV for all batches...")

    npz_files = sorted(batches_dir.glob("batch_*_features.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No batch_*_features.npz found in {batches_dir}")

    if args.max_batches is not None:
        npz_files = npz_files[: args.max_batches]
        print(f"  [dry-run] Limiting to {args.max_batches} batches.")

    n_files = len(npz_files)
    print(f"  Found {n_files} batch files to process.")

    all_rows = []
    t0 = time.time()
    total_mofs = 0

    for i, npz_path in enumerate(npz_files):
        # Progress logging
        if i % LOG_INTERVAL == 0 or i == n_files - 1:
            elapsed = time.time() - t0
            print(f"  Batch {i+1}/{n_files}  ({npz_path.name})  "
                  f"elapsed={elapsed:.0f}s  total_mofs={total_mofs}")

        data = np.load(npz_path, allow_pickle=True)
        emb      = data["features"].astype(np.float32)   # (N_i, 256)
        mol_ids  = data["mol_ids"].tolist()

        if emb.shape[0] == 0:
            continue

        lsv_norm = compute_lsv_norm(
            emb, train_labels_orig, index,
            baseline_dist, k, baseline_lsv_mean,
        )   # (N_i, T)

        composite = lsv_norm.mean(axis=1)   # (N_i,)

        # Build rows for this batch
        batch_df = pd.DataFrame(
            lsv_norm,
            columns=[f"{t}_lsv_norm" for t in TARGET_COLS],
        )
        batch_df.insert(0, "mof_id", mol_ids)
        batch_df["lsv_norm_composite"] = composite
        batch_df["flag_high_uq"] = (composite > args.threshold).astype(np.int8)

        all_rows.append(batch_df)
        total_mofs += len(mol_ids)

    # ── 3. Concatenate and save ────────────────────────────────────────────────
    print(f"\n[3/3] Saving results ({total_mofs} total MOFs)...")

    result_df = pd.concat(all_rows, ignore_index=True)

    out_csv = output_dir / "full_library_uq.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}  (shape={result_df.shape})")

    # Summary statistics
    n_flagged    = int(result_df["flag_high_uq"].sum())
    frac_flagged = n_flagged / max(len(result_df), 1)
    comp_vals    = result_df["lsv_norm_composite"]

    print(f"\n  Composite LSV_norm statistics:")
    print(f"    mean   : {comp_vals.mean():.4f}")
    print(f"    median : {comp_vals.median():.4f}")
    print(f"    p90    : {comp_vals.quantile(0.90):.4f}  (reference threshold: {args.threshold})")
    print(f"    max    : {comp_vals.max():.4f}")
    print(f"  High-UQ flagged: {n_flagged:,} / {len(result_df):,} ({frac_flagged:.1%})")

    elapsed_total = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"DONE — Task 1.1d")
    print(f"  Total MOFs processed : {total_mofs:,}")
    print(f"  High-UQ flagged      : {n_flagged:,} ({frac_flagged:.1%})")
    print(f"  Output               : {out_csv}")
    print(f"  Total elapsed        : {elapsed_total:.1f}s")
    print(f"  Finish time          : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
