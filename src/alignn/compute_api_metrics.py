"""
compute_api_metrics.py
======================
Task 2.1: Concatenate full-library prediction batches, compute PSA/VSA
screening metrics, and merge persisted UQ scores.

Output:
    full_library_with_api.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignn.common.paths import REPO_ROOT, resolve_model_paths
from src.alignn.screening.metrics import calculate_separation_metrics


N_BATCHES = 24
TEST_BATCHES = 3
DEFAULT_MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"


def resolve_inference_paths(model_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve batch input, UQ CSV, and output CSV for one model directory."""
    paths = resolve_model_paths(model_dir)
    batch_dir = paths.inference_dir / "batches"
    uq_csv = paths.inference_dir / "full_library_uq.csv"
    out_csv = paths.inference_dir / "full_library_with_api.csv"
    return batch_dir, uq_csv, out_csv


def load_prediction_batches(batch_dir: Path, n_batches: int) -> pd.DataFrame:
    """Load and concatenate batch prediction CSVs."""
    frames = []
    for index in range(n_batches):
        csv_path = batch_dir / f"batch_{index:04d}_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing batch prediction CSV: {csv_path}")
        frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True)


def build_full_library_with_api(model_dir: Path, n_batches: int) -> pd.DataFrame:
    """Compute canonical API metrics and merge persisted UQ flags."""
    batch_dir, uq_csv, out_csv = resolve_inference_paths(model_dir)
    df_pred = load_prediction_batches(batch_dir, n_batches)
    df_api = calculate_separation_metrics(df_pred)

    if not uq_csv.exists():
        raise FileNotFoundError(f"Missing UQ CSV: {uq_csv}")
    df_uq = pd.read_csv(uq_csv)
    df_merged = df_api.merge(df_uq, on="mof_id", how="left")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_csv, index=False)
    return df_merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Model-specific results dir (e.g. results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Dry-run: load only the first {TEST_BATCHES} batch CSVs.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir.is_absolute() else REPO_ROOT / args.model_dir
    n_batches = TEST_BATCHES if args.test else N_BATCHES
    df = build_full_library_with_api(model_dir, n_batches)
    print(f"Saved full_library_with_api.csv for {model_dir} with shape {df.shape}")


if __name__ == "__main__":
    main()
