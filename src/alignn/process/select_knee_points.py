"""Select knee operating points from NSGA-II Pareto fronts.

Parses last-generation populations from SuperPSA NSGA-II result files,
filters feasible solutions, identifies Pareto-optimal points, and selects
the knee point using normalized distance-to-utopia.

Usage:
    python select_knee_points.py

Output:
    - selected_knee_points.json (machine-readable, consumed by MATLAB and figure scripts)
    - stdout summary (human-readable)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
_REPO = Path(__file__).resolve().parents[3]
_RESULTS_DIR = _REPO / "src" / "SuperPSA" / "Results_extDSL_HR"
_RANKING_CSV = _REPO / "results" / "alignn" / "model_ep150" / "psa_optimization" / "material_ranking.csv"
_SUPERPSA_DATA = _REPO / "src" / "SuperPSA" / "data"
_OUTPUT_JSON = _REPO / "results" / "alignn" / "model_ep150" / "psa_optimization" / "selected_knee_points.json"

# Mode → config file mapping
_MODE_CONFIG = {
    "PSA": {
        "yaml_name": "ProcessConfig_PSA_HR.yaml",
        "csv_name": "Adsorbents_CH4N2_PSA.csv",
    },
    "VSA": {
        "yaml_name": "ProcessConfig_VSA_HR.yaml",
        "csv_name": "Adsorbents_CH4N2_VSA.csv",
    },
}

# Column names (16 columns: 10 Vars + 2 Objs + 4 Cons)
_DATA_COLS = [
    "Var1", "Var2", "Var3", "Var4", "Var5",
    "Var6", "Var7", "Var8", "Var9", "Var10",
    "Obj1", "Obj2",
    "Cons1", "Cons2", "Cons3", "Cons4",
]


def _find_result_file(results_dir: Path, mode: str, material_name: str) -> Path:
    """Find the NSGA-II result file for a given mode and material.

    Matches pattern: opt_{mode}_HR_*_{material_name}_*.txt
    The material name may contain special chars like [] so we search by substring.
    """
    candidates = []
    for f in results_dir.glob(f"opt_{mode}_HR_*.txt"):
        if material_name in f.name:
            candidates.append(f)
    if not candidates:
        raise FileNotFoundError(
            f"No result file for {mode}/{material_name} in {results_dir}"
        )
    if len(candidates) > 1:
        # Pick the latest by timestamp in filename
        candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _get_top1_materials(ranking_csv: Path) -> dict[str, str]:
    """Read material_ranking.csv and return Top-1 material name per mode."""
    df = pd.read_csv(ranking_csv)
    top1 = {}
    for mode in ["PSA", "VSA"]:
        mode_df = df[df["mode"] == mode].sort_values("global_rank")
        if not mode_df.empty:
            top1[mode] = mode_df.iloc[0]["material_name"]
    return top1


def _lookup_mat_idx(csv_path: Path, material_name: str) -> int:
    """Find 1-based row index of material in adsorbent CSV (for MATLAB)."""
    df = pd.read_csv(csv_path)
    matches = df.index[df["material_name"] == material_name].tolist()
    if not matches:
        raise ValueError(f"{material_name} not found in {csv_path.name}")
    return matches[0] + 1  # 1-based for MATLAB


def _make_short_name(material_name: str) -> str:
    """Create a filesystem-safe short name for profile CSV filenames."""
    # Known aliases
    if material_name == "CoRE-2020[Cu][pts]3[ASR]1":
        return "ATC-Cu"
    if material_name == "MOSAEC-YOBPOW_full_REPEAT":
        return "YOBPOW"
    if material_name == "CoRE-2010[Co][pts]3[ASR]2":
        return "CoRE-Co"
    # Generic: strip prefix + suffix, replace unsafe chars
    cleaned = re.sub(r"_(full_REPEAT|clean_repeat|repeat)$", "", material_name)
    cleaned = re.sub(r"^(CoRE-\d{4}|ARC-DB\d+-|MOSAEC-)", "", cleaned)
    return cleaned.replace("[", "").replace("]", "").replace(".", "_")


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
    json_cases = []

    # Dynamically resolve Top-1 materials from ranking CSV
    top1 = _get_top1_materials(_RANKING_CSV)
    logger.info("Top-1 materials: %s", top1)

    for mode in ["PSA", "VSA"]:
        mat_name = top1[mode]
        filepath = _find_result_file(_RESULTS_DIR, mode, mat_name)
        logger.info("Processing %s: %s (%s)", mode, mat_name, filepath.name)

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

        # Build JSON entry
        cfg = _MODE_CONFIG[mode]
        csv_path = _SUPERPSA_DATA / cfg["csv_name"]
        mat_idx = _lookup_mat_idx(csv_path, mat_name)
        short_name = _make_short_name(mat_name)
        x_vector = [float(knee[f"Var{i}"]) for i in range(1, 11)]

        json_cases.append({
            "mode": mode,
            "material_name": mat_name,
            "short_name": short_name,
            "mat_idx": mat_idx,
            "yaml_name": cfg["yaml_name"],
            "csv_name": cfg["csv_name"],
            "x_vector": x_vector,
            "productivity": float(-knee["Obj1"]),
            "energy": float(knee["Obj2"]),
        })

    # Save JSON
    _OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_JSON, "w") as f:
        json.dump(json_cases, f, indent=2)
    print(f"\nSaved: {_OUTPUT_JSON}")

    return results


if __name__ == "__main__":
    main()
