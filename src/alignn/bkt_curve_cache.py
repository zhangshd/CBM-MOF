"""Utilities for saving and rebuilding breakthrough curve cache CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CURVE_CACHE_COLUMNS = ["mof", "process", "time_min", "CC0_CH4"]


def build_curve_dataframe(
    mof_name: str,
    process: str,
    time_min: Iterable[float],
    cc0_ch4: Iterable[float],
) -> pd.DataFrame:
    """Build a normalized breakthrough-curve DataFrame for one simulation."""
    df = pd.DataFrame(
        {
            "mof": mof_name,
            "process": process,
            "time_min": list(time_min),
            "CC0_CH4": list(cc0_ch4),
        }
    )
    return df[CURVE_CACHE_COLUMNS]


def collect_curve_csv_paths(bkt_dir: Path) -> list[Path]:
    """Find all per-run breakthrough curve CSV files under the BKT output tree."""
    return sorted(bkt_dir.glob("bkt_*/*/breakthrough_curve_data.csv"))


def rebuild_curve_cache(bkt_dir: Path, output_csv: Path | None = None) -> pd.DataFrame:
    """Merge per-run breakthrough curve CSV files into one cache table."""
    csv_paths = collect_curve_csv_paths(bkt_dir)
    if not csv_paths:
        raise FileNotFoundError(
            f"No breakthrough_curve_data.csv found under {bkt_dir}"
        )

    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        missing = [col for col in CURVE_CACHE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        dfs.append(df[CURVE_CACHE_COLUMNS].copy())

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(["process", "mof", "time_min"], kind="stable")
    merged = merged.reset_index(drop=True)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False)

    return merged
