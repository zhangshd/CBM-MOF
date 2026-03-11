"""
parse_top20_pure_component.py — Task 3.1a: Parse Top-20 BKT candidates'
pure-component GCMC results into standard CSV for isotherm fitting.

Parses RASPA3 output for 20 MOFs × 2 gases (CH4, N2) × 10 pressures using
the canonical MOF-HTS raspa3_result_parser, then outputs a CSV with the same
schema as atc_cu_pure_component.csv.

Usage:
    python src/alignn/parse_top20_pure_component.py
    python src/alignn/parse_top20_pure_component.py --model-dir results/alignn/model_ep150
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
GCMC_SRC = REPO_ROOT / "src" / "gcmc"

PURE_COMPONENTS = {"methane", "N2"}

STANDARD_COLUMNS = [
    "MofName",
    "GasName",
    "Temperature[K]",
    "Pressure[bar]",
    "AllComponents",
    "MoleculeFraction",
    "LoadingUnit",
    "AbsLoading",
    "ExcessLoading",
    "SimuDuration[h]",
    "FilePath",
    "Notes",
]


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_with_raspa3_parser(result_dir: Path) -> pd.DataFrame:
    """Load a RASPA3 result directory using the canonical MOF-HTS parser."""
    if str(GCMC_SRC) not in sys.path:
        sys.path.insert(0, str(GCMC_SRC))

    from raspa3_result_parser import RASPA3ResultParser

    parser = RASPA3ResultParser(result_dir)
    df = parser.parse_all_results()
    if df.empty:
        raise ValueError(f"No parsed rows found in {result_dir}")
    return df


def filter_pure_component(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only pure-component adsorption rows (MoleculeFraction == 1.0)."""
    _require_columns(df, STANDARD_COLUMNS)

    filtered = df.copy()
    filtered["AllComponents"] = filtered["AllComponents"].astype(str).str.strip()
    filtered["GasName"] = filtered["GasName"].astype(str).str.strip()

    filtered = filtered[filtered["GasName"].isin(PURE_COMPONENTS)]
    filtered = filtered[filtered["AllComponents"].isin(PURE_COMPONENTS)]
    filtered = filtered[np.isclose(filtered["MoleculeFraction"], 1.0)]

    filtered = filtered[STANDARD_COLUMNS].copy()
    filtered = filtered.sort_values(
        by=["MofName", "GasName", "Pressure[bar]"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.1a: Parse Top-20 pure-component GCMC results."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model-specific results dir (e.g. results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Output CSV path (default: auto-generated in isotherm_input/).",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = REPO_ROOT / md
    else:
        md = REPO_ROOT / "results" / "alignn" / "model_ep150"

    gcmc_base = md / "bkt_candidates" / "gcmc_pure_component"
    output_csv = Path(args.output_csv) if args.output_csv else (
        md / "bkt_candidates" / "isotherm_input" / "top20_pure_component.csv"
    )

    # Parse each gas directory
    frames = []
    for gas_dir in ["methane", "N2"]:
        result_dir = gcmc_base / gas_dir / "batch_000"
        if not result_dir.exists():
            print(f"[WARNING] Directory not found: {result_dir}")
            continue

        print(f"Parsing {gas_dir}: {result_dir}")
        parsed = load_with_raspa3_parser(result_dir)
        filtered = filter_pure_component(parsed)
        print(f"  Parsed rows: {len(parsed)}, filtered pure-component: {len(filtered)}")
        frames.append(filtered)

    if not frames:
        print("[ERROR] No data parsed from any gas directory.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        by=["MofName", "GasName", "Pressure[bar]"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    # Verify completeness
    n_mofs = combined["MofName"].nunique()
    n_gases = combined["GasName"].nunique()
    n_pressures_per = combined.groupby(["MofName", "GasName"]).size()

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total rows    : {len(combined)}")
    print(f"  Unique MOFs   : {n_mofs}")
    print(f"  Unique gases  : {n_gases} ({sorted(combined['GasName'].unique().tolist())})")
    print(f"  Pressures/MOF : min={n_pressures_per.min()}, max={n_pressures_per.max()}")

    # Check for duplicates
    dup_check = combined.duplicated(subset=["MofName", "GasName", "Pressure[bar]"])
    if dup_check.any():
        print(f"  WARNING: {dup_check.sum()} duplicate (MOF, gas, pressure) entries!")
    else:
        print(f"  No duplicates ✓")

    # Check expected count: 20 MOFs × 2 gases × 10 pressures = 400
    expected = 20 * 2 * 10
    if len(combined) != expected:
        print(f"  WARNING: Expected {expected} rows, got {len(combined)}")
    else:
        print(f"  Count matches expected {expected} ✓")
    print(f"{'='*60}")

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\nOutput CSV: {output_csv}")


if __name__ == "__main__":
    main()
