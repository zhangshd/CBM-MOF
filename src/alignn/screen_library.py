"""
screen_library.py
=================
Task 2.2: Apply the canonical library pre-screening logic on top of
``full_library_with_api.csv``.

Current filters:
  1. Remove MOFs flagged as high-UQ
  2. Remove fringe non-adsorbers by CH4 uptake floor at 1000 kPa

Output:
    full_library_screened.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignn.common.constants import UPTAKE_FLOOR_MOL_PER_KG
from src.alignn.common.paths import REPO_ROOT, resolve_model_paths


DEFAULT_MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"


def screen_full_library(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical UQ and uptake-floor filters."""
    filtered = df[df["flag_high_uq"] == 0].copy()
    filtered = filtered[filtered["AdsCH4_1000kPa"] >= UPTAKE_FLOOR_MOL_PER_KG].copy()
    return filtered.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Model-specific results dir (e.g. results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional override for full_library_with_api.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional override for full_library_screened.csv.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir.is_absolute() else REPO_ROOT / args.model_dir
    paths = resolve_model_paths(model_dir)
    input_csv = args.input_csv or (paths.inference_dir / "full_library_with_api.csv")
    output_csv = args.output_csv or (paths.inference_dir / "full_library_screened.csv")

    df = pd.read_csv(input_csv)
    screened = screen_full_library(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    screened.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} with shape {screened.shape}")


if __name__ == "__main__":
    main()
