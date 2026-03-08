"""
parse_atc_cu_pure_component.py — Task 3.1b: Extract ATC-Cu pure-component
GCMC results for downstream isotherm fitting.

This script uses the canonical MOF-HTS RASPA3 parser, then filters the parsed
rows down to the ATC-Cu benchmark and pure-component adsorption only.

Usage:
    python src/alignn/parse_atc_cu_pure_component.py
    python src/alignn/parse_atc_cu_pure_component.py --result-dir /path/to/results
    python src/alignn/parse_atc_cu_pure_component.py --output-csv /path/to/out.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MOF_HTS_SRC = Path("/home/zhangsd/repos/MOF-HTS/src")

BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"
PURE_COMPONENTS = {"methane", "N2"}

RESULT_DIR_DEFAULT = Path(
    "/home/zhangsd/repos/MOF-HTS/results/cbm_screening/gcmc_ATC-Cu_DreidingTraPPEJson"
)
OUTPUT_CSV_DEFAULT = (
    REPO_ROOT
    / "results"
    / "alignn"
    / "model_ep150"
    / "bkt_candidates"
    / "isotherm_input"
    / "atc_cu_pure_component.csv"
)

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
    """Load benchmark result directory using the canonical MOF-HTS parser."""
    if str(MOF_HTS_SRC) not in sys.path:
        sys.path.insert(0, str(MOF_HTS_SRC))

    from gcmc.raspa3_result_parser import RASPA3ResultParser

    parser = RASPA3ResultParser(result_dir)
    df = parser.parse_all_results()
    if df.empty:
        raise ValueError(f"No parsed rows found in {result_dir}")
    return df


def filter_atc_cu_pure_component(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ATC-Cu benchmark rows for pure CH4 and pure N2 adsorption."""
    _require_columns(df, STANDARD_COLUMNS)

    filtered = df.copy()
    filtered["AllComponents"] = filtered["AllComponents"].astype(str).str.strip()
    filtered["GasName"] = filtered["GasName"].astype(str).str.strip()

    filtered = filtered[filtered["MofName"] == BENCHMARK_MOF]
    filtered = filtered[filtered["GasName"].isin(PURE_COMPONENTS)]
    filtered = filtered[filtered["AllComponents"].isin(PURE_COMPONENTS)]
    filtered = filtered[np.isclose(filtered["MoleculeFraction"], 1.0)]

    filtered = filtered[STANDARD_COLUMNS].copy()
    filtered = filtered.sort_values(
        by=["GasName", "Pressure[bar]"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.1b: Extract ATC-Cu pure-component GCMC results."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=str(RESULT_DIR_DEFAULT),
        help="ATC-Cu benchmark result directory to parse.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(OUTPUT_CSV_DEFAULT),
        help="Output CSV path for the filtered pure-component data.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_csv = Path(args.output_csv)

    print(f"Parsing ATC-Cu result dir: {result_dir}")
    parsed = load_with_raspa3_parser(result_dir)
    filtered = filter_atc_cu_pure_component(parsed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    print(f"Filtered rows: {len(filtered)}")
    print(f"Unique gases : {sorted(filtered['GasName'].unique().tolist())}")
    print(f"Output CSV   : {output_csv}")


if __name__ == "__main__":
    main()
