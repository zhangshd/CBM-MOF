"""Select knee operating points from NSGA-II Pareto fronts.

Parses last-generation populations from SuperPSA NSGA-II result files,
filters feasible solutions, identifies Pareto-optimal points, and selects
the knee point using normalized distance-to-utopia.

Usage:
    python select_knee_points.py

Output: prints selected operating conditions (Var1-Var10) for PSA and VSA.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Result files
_REPO = Path(__file__).resolve().parents[3]
_PSA_FILE = _REPO / "src" / "SuperPSA" / "Results" / "opt_PSA_HR_002_ARC-DB0-m3_o10_o146_f0_fsc_repeat_20260401_215802.txt"
_VSA_FILE = _REPO / "src" / "SuperPSA" / "Results" / "opt_VSA_HR_004_ARC-DB0-m3_o25_o460_f0_fsc.sym.15_repeat_20260402_054821.txt"

# Column names (16 columns: 10 Vars + 2 Objs + 4 Cons)
_DATA_COLS = [
    "Var1", "Var2", "Var3", "Var4", "Var5",
    "Var6", "Var7", "Var8", "Var9", "Var10",
    "Obj1", "Obj2",
    "Cons1", "Cons2", "Cons3", "Cons4",
]


def parse_last_generation(filepath: Path) -> pd.DataFrame:
    """Parse last generation population from NSGA-II result file."""
    with open(filepath, "r") as f:
        content = f.read()

    # Find last generation block
    gen_pattern = re.compile(r"#Generation\s+(\d+)\s*/\s*(\d+)")
    gen_matches = list(gen_pattern.finditer(content))
    if not gen_matches:
        raise ValueError(f"No generation blocks found in {filepath.name}")

    last_gen_match = gen_matches[-1]
    last_gen_num = int(last_gen_match.group(1))
    logger.info("File: %s — last generation: %d", filepath.name, last_gen_num)

    # Extract data after #end and header
    remaining = content[last_gen_match.start():]
    end_idx = remaining.find("#end")
    if end_idx == -1:
        raise ValueError(f"No #end marker after last generation in {filepath.name}")

    after_end = remaining[end_idx + 4:]
    lines = after_end.strip().split("\n")

    data_lines = []
    header_found = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Var1"):
            header_found = True
            continue
        if line.startswith("#"):
            break
        if header_found:
            data_lines.append(line)

    rows = []
    for line in data_lines:
        parts = [p for p in line.split("\t") if p.strip()]
        if len(parts) < 16:
            continue
        try:
            row = [float(x) for x in parts[:16]]
            rows.append(row)
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No valid data rows in last generation of {filepath.name}")

    return pd.DataFrame(rows, columns=_DATA_COLS)


def filter_feasible(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to feasible solutions (all constraints <= 0)."""
    mask = (
        (df["Cons1"] <= 0) & (df["Cons2"] <= 0) &
        (df["Cons3"] <= 0) & (df["Cons4"] <= 0)
    )
    return df[mask].copy()


def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Find non-dominated solutions.

    Objectives: minimize Obj1 (= -productivity), minimize Obj2 (= energy).
    """
    obj = df[["Obj1", "Obj2"]].values
    n = len(obj)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j <= i on all objs and j < i on at least one
            if (obj[j] <= obj[i]).all() and (obj[j] < obj[i]).any():
                is_dominated[i] = True
                break

    return df[~is_dominated].copy()


def select_knee_utopia(pareto: pd.DataFrame) -> pd.Series:
    """Select knee point using normalized distance-to-utopia.

    Utopia: (max productivity = min Obj1, min energy = min Obj2).
    Normalize objectives to [0,1], find point closest to utopia.
    """
    obj1 = pareto["Obj1"].values  # -productivity (more negative = better)
    obj2 = pareto["Obj2"].values  # energy (lower = better)

    # Normalize to [0, 1]
    obj1_min, obj1_max = obj1.min(), obj1.max()
    obj2_min, obj2_max = obj2.min(), obj2.max()

    if obj1_max - obj1_min < 1e-12:
        obj1_norm = np.zeros_like(obj1)
    else:
        obj1_norm = (obj1 - obj1_min) / (obj1_max - obj1_min)

    if obj2_max - obj2_min < 1e-12:
        obj2_norm = np.zeros_like(obj2)
    else:
        obj2_norm = (obj2 - obj2_min) / (obj2_max - obj2_min)

    # Utopia in normalized space = (0, 0)
    dist = np.sqrt(obj1_norm**2 + obj2_norm**2)

    knee_idx = np.argmin(dist)
    return pareto.iloc[knee_idx]


def format_operating_point(row: pd.Series, mode: str, mat_name: str) -> str:
    """Format selected operating point for display."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  {mode} Knee Point — {mat_name}")
    lines.append(f"{'='*70}")

    # Objectives
    prod = -row["Obj1"]
    energy = row["Obj2"]
    lines.append(f"  Productivity : {prod:.4f} mol/(kg*h)")
    lines.append(f"  Energy       : {energy:.2f} kWh/ton")

    # Operating conditions
    var_names = [
        "P_H (Pa)", "t_ads (s)", "LR_ratio (-)", "v_feed (m/s)",
        "HR1_ratio (-)", "P_L (Pa)", "t_pres (s)", "t_CnCDepres (s)",
        "t_CoCDepres (s)", "HR2_ratio (-)",
    ]
    lines.append("\n  Operating Conditions:")
    for i, name in enumerate(var_names, 1):
        val = row[f"Var{i}"]
        lines.append(f"    x({i:2d}) = {val:15.6f}   % {name}")

    # MATLAB x-vector format
    vals = [row[f"Var{i}"] for i in range(1, 11)]
    matlab_str = ", ".join(f"{v:.6f}" for v in vals)
    lines.append(f"\n  MATLAB x vector:")
    lines.append(f"    x = [{matlab_str}];")

    # Constraints
    lines.append(f"\n  Constraints:")
    for i in range(1, 5):
        lines.append(f"    Cons{i} = {row[f'Cons{i}']:.6f}")

    return "\n".join(lines)


def main():
    results = {}

    for mode, filepath, mat_name in [
        ("PSA", _PSA_FILE, "ARC-o10"),
        ("VSA", _VSA_FILE, "ARC-o25.15"),
    ]:
        logger.info("Processing %s: %s", mode, filepath.name)

        # Parse last generation
        pop = parse_last_generation(filepath)
        logger.info("  Population size: %d", len(pop))

        # Filter feasible
        feasible = filter_feasible(pop)
        logger.info("  Feasible solutions: %d", len(feasible))

        if len(feasible) == 0:
            logger.error("  No feasible solutions found!")
            continue

        # Deduplicate by objective values
        feasible_dedup = feasible.drop_duplicates(subset=["Obj1", "Obj2"])
        logger.info("  Unique feasible solutions: %d", len(feasible_dedup))

        # Compute Pareto front
        pareto = compute_pareto_front(feasible_dedup)
        logger.info("  Pareto front size: %d", len(pareto))

        # Select knee
        knee = select_knee_utopia(pareto)
        results[mode] = knee

        # Display
        print(format_operating_point(knee, mode, mat_name))

        # Also show full Pareto front
        print(f"\n  Full Pareto Front ({len(pareto)} points):")
        print(f"    {'Prod (mol/kg/h)':>16s}  {'Energy (kWh/ton)':>16s}")
        for _, r in pareto.sort_values("Obj1").iterrows():
            print(f"    {-r['Obj1']:16.4f}  {r['Obj2']:16.2f}")

    return results


if __name__ == "__main__":
    main()
