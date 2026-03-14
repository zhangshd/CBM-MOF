"""
apply_uq_to_library.py
======================
Task 1.1d: Apply the canonical UQ payload to full-library latent features.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignn.common.paths import REPO_ROOT, resolve_model_paths
from src.alignn.uq.apply import score_full_library_batch
from src.alignn.uq.core import load_lsv_thresholds, load_uncertainty_payload
from src.alignn.uq.io import load_full_library_batch_features


DEFAULT_MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"


def load_composite_threshold(threshold_json: Path) -> float:
    """Backward-compatible helper for tests and downstream code."""
    return float(load_lsv_thresholds(threshold_json)["composite_threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Model-specific result directory containing uq/ and full_library_inference/.",
    )
    parser.add_argument(
        "--uq-pkl",
        type=Path,
        default=None,
        help="Optional override for uncertainty_trees.pkl.",
    )
    parser.add_argument(
        "--threshold-json",
        type=Path,
        default=None,
        help="Optional override for lsv_thresholds.json.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional override for full_library_inference directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional override for full_library_uq.csv.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Process only the first N batch feature files.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir.is_absolute() else REPO_ROOT / args.model_dir
    paths = resolve_model_paths(model_dir)
    uq_pkl = args.uq_pkl or (paths.uq_dir / "uncertainty_trees.pkl")
    threshold_json = args.threshold_json or (paths.uq_dir / "lsv_thresholds.json")
    inference_dir = args.input_dir or paths.inference_dir
    output_csv = args.output_csv or (inference_dir / "full_library_uq.csv")

    thresholds = load_lsv_thresholds(threshold_json)
    threshold = float(thresholds["composite_threshold"])
    payload = load_uncertainty_payload(uq_pkl)
    batch_features = load_full_library_batch_features(inference_dir / "batches")
    if args.max_batches is not None:
        batch_features = batch_features[: args.max_batches]

    t0 = time.time()
    frames = []
    total_rows = 0
    for index, (mol_ids, features) in enumerate(batch_features, start=1):
        frames.append(
            score_full_library_batch(
                mol_ids=mol_ids,
                embeddings=features,
                train_labels_orig=payload["train_labels_orig"],
                index=payload["index"],
                baseline_dist=float(payload["baseline_dist"]),
                baseline_lsv_mean=payload["baseline_lsv_mean"],
                k=int(payload["k"]),
                threshold=threshold,
            )
        )
        total_rows += len(mol_ids)
        print(f"Processed batch {index}/{len(batch_features)}; total_mofs={total_rows}")

    result = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    print("=" * 65)
    print(f"Saved {output_csv}")
    print(f"Rows: {len(result):,}")
    print(f"High-UQ flagged: {int(result['flag_high_uq'].sum()):,}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
