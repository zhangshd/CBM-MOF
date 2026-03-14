"""
parse_pure_component_results.py
===============================
Canonical pure-component parser for benchmark and candidate BKT isotherm inputs.

Supported modes:
  - benchmark: parse the ATC-Cu benchmark directory
  - candidates: parse the Top-20 candidate pure-component directories
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignn.common.paths import REPO_ROOT, resolve_model_paths


GCMC_SRC = REPO_ROOT / "src" / "gcmc"
BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"
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
    """Load a RASPA3 result directory using the canonical parser."""
    if str(GCMC_SRC) not in sys.path:
        sys.path.insert(0, str(GCMC_SRC))
    from raspa3_result_parser import RASPA3ResultParser

    parser = RASPA3ResultParser(result_dir)
    df = parser.parse_all_results()
    if df.empty:
        raise ValueError(f"No parsed rows found in {result_dir}")
    return df


def filter_pure_component(df: pd.DataFrame, benchmark_only: bool = False) -> pd.DataFrame:
    """Filter a parsed RASPA3 table to the canonical pure-component schema."""
    _require_columns(df, STANDARD_COLUMNS)
    filtered = df.copy()
    filtered["AllComponents"] = filtered["AllComponents"].astype(str).str.strip()
    filtered["GasName"] = filtered["GasName"].astype(str).str.strip()

    filtered = filtered[filtered["GasName"].isin(PURE_COMPONENTS)]
    filtered = filtered[filtered["AllComponents"].isin(PURE_COMPONENTS)]
    filtered = filtered[np.isclose(filtered["MoleculeFraction"], 1.0)]
    if benchmark_only:
        filtered = filtered[filtered["MofName"] == BENCHMARK_MOF]

    filtered = filtered[STANDARD_COLUMNS].copy()
    sort_cols = ["GasName", "Pressure[bar]"] if benchmark_only else ["MofName", "GasName", "Pressure[bar]"]
    return filtered.sort_values(by=sort_cols, ascending=True, kind="stable").reset_index(drop=True)


def parse_candidate_mode(model_dir: Path, output_csv: Path) -> None:
    """Parse Top-20 candidate pure-component results."""
    model_paths = resolve_model_paths(model_dir)
    gcmc_base = model_paths.bkt_candidates_dir / "gcmc_pure_component"
    frames = []
    for gas_dir in ["methane", "N2"]:
        result_dir = gcmc_base / gas_dir / "batch_000"
        if not result_dir.exists():
            continue
        frames.append(filter_pure_component(load_with_raspa3_parser(result_dir)))
    if not frames:
        raise FileNotFoundError(f"No candidate pure-component results found under {gcmc_base}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["MofName", "GasName", "Pressure[bar]"], kind="stable").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)


def parse_benchmark_mode(result_dir: Path, output_csv: Path) -> None:
    """Parse the ATC-Cu benchmark pure-component result directory."""
    parsed = load_with_raspa3_parser(result_dir)
    filtered = filter_pure_component(parsed, benchmark_only=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["benchmark", "candidates"],
        required=True,
        help="Which pure-component source to parse.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "results" / "alignn" / "model_ep150",
        help="Model-specific result directory for candidate parsing.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPO_ROOT / "results" / "cbm_screening" / "gcmc_ATC-Cu_DreidingTraPPEJson",
        help="Benchmark result directory for benchmark mode.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional override for the canonical output CSV path.",
    )
    args = parser.parse_args()

    if args.mode == "benchmark":
        output_csv = args.output_csv or (
            resolve_model_paths(args.model_dir).bkt_candidates_dir / "isotherm_input" / "atc_cu_pure_component.csv"
        )
        parse_benchmark_mode(args.result_dir, output_csv)
    else:
        output_csv = args.output_csv or (
            resolve_model_paths(args.model_dir).bkt_candidates_dir / "isotherm_input" / "top20_pure_component.csv"
        )
        parse_candidate_mode(args.model_dir, output_csv)

    print(f"Saved pure-component CSV: {output_csv}")


if __name__ == "__main__":
    main()
