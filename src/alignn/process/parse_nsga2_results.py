"""Parse NSGA-II optimization results and analyze Pareto fronts.

Reads SuperPSA NSGA-II result files, extracts last-generation populations,
filters feasible solutions, computes per-material and global Pareto fronts,
and ranks materials by their contributions to the global front.

Output:
  - pareto_analysis.csv: All per-material Pareto-optimal feasible solutions
  - material_ranking.csv: Material ranking summary across modes
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Repository layout defaults
_REPO = Path(__file__).resolve().parents[3]  # .../CBM-MOF
_RESULTS_DIR = _REPO / "src" / "SuperPSA" / "Results"
_PSA_CSV = _REPO / "src" / "SuperPSA" / "data" / "Adsorbents_CH4N2_PSA.csv"
_VSA_CSV = _REPO / "src" / "SuperPSA" / "data" / "Adsorbents_CH4N2_VSA.csv"
_OUTPUT_DIR = _REPO / "results" / "alignn" / "model_ep150" / "psa_optimization"

# Column names for the 15-column data rows
_DATA_COLS = [
    "Var1", "Var2", "Var3", "Var4", "Var5",
    "Var6", "Var7", "Var8", "Var9", "Var10",
    "Obj1", "Obj2",
    "Cons1", "Cons2", "Cons3",
]

# Regex for filename parsing
# Examples:
#   opt_PSA_001_ARC-DB0-m3_o156_o47_f0_fsc.sym.14_repeat_20260328_025407.txt
#   opt_PSA_HR_001_ARC-DB0-m3_o156_o47_f0_fsc.sym.14_repeat_20260329_025143.txt
#   opt_VSA_010_CoRE-2020[Cu][pts]3[ASR]1_20260326_232259.txt
_FNAME_RE = re.compile(
    r"^opt_"
    r"(?P<mode>PSA|VSA)"
    r"(?:_(?P<hr>HR))?"
    r"_(?P<idx>\d{3})"
    r"_(?P<name>.+?)"
    r"_\d{8}_\d{6}"
    r"\.txt$"
)


def parse_result_file(filepath: Path, min_gen: int = 10) -> dict | None:
    """Parse a single NSGA-II result file.

    Extracts the last generation's population data.

    Args:
        filepath: Path to the result .txt file.
        min_gen: Minimum number of generations required (skip file otherwise).

    Returns:
        Dict with metadata and DataFrame of population, or None if skipped.
    """
    fname_match = _FNAME_RE.match(filepath.name)
    if fname_match is None:
        logger.warning("Filename does not match expected pattern: %s", filepath.name)
        return None

    mode = fname_match.group("mode")
    is_hr = fname_match.group("hr") is not None
    mat_idx = int(fname_match.group("idx"))
    mat_name = fname_match.group("name")
    cycle_type = "HR" if is_hr else "Basic"

    with open(filepath, "r") as f:
        content = f.read()

    # Split by generation blocks
    gen_pattern = re.compile(r"#Generation\s+(\d+)\s*/\s*(\d+)")
    gen_matches = list(gen_pattern.finditer(content))

    if not gen_matches:
        logger.warning("No generation blocks found in %s", filepath.name)
        return None

    last_gen_match = gen_matches[-1]
    last_gen_num = int(last_gen_match.group(1))
    max_gen = int(last_gen_match.group(2))

    if last_gen_num < min_gen:
        logger.warning(
            "File %s has only %d generations (min_gen=%d), skipping",
            filepath.name, last_gen_num, min_gen,
        )
        return None

    # Extract data rows after the last generation block
    # Find the position after #end and header line
    last_gen_start = last_gen_match.start()
    remaining = content[last_gen_start:]

    # Find #end marker
    end_idx = remaining.find("#end")
    if end_idx == -1:
        logger.warning("No #end marker after last generation in %s", filepath.name)
        return None

    after_end = remaining[end_idx + 4:]

    # Skip header line (Var1\tVar2\t...)
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
            break  # next generation block (shouldn't happen for last)
        if header_found:
            data_lines.append(line)

    if not data_lines:
        logger.warning("No data rows in last generation of %s", filepath.name)
        return None

    # Parse data rows
    rows = []
    for line in data_lines:
        parts = line.split("\t")
        # Filter out empty trailing fields from trailing tab
        parts = [p for p in parts if p.strip()]
        if len(parts) < 15:
            continue
        try:
            row = [float(x) for x in parts[:15]]
            rows.append(row)
        except ValueError:
            continue

    if not rows:
        logger.warning("No valid data rows parsed from %s", filepath.name)
        return None

    df = pd.DataFrame(rows, columns=_DATA_COLS)

    # Extract generation metadata
    # Parse totalTime and other stats from the block
    meta_text = remaining[: end_idx]
    total_time = _extract_meta_value(meta_text, "totalTime")
    eval_count = _extract_meta_value(meta_text, "evaluateCount")

    return {
        "mode": mode,
        "cycle_type": cycle_type,
        "material_idx": mat_idx,
        "material_name": mat_name,
        "last_gen": last_gen_num,
        "max_gen": max_gen,
        "total_time_s": total_time,
        "evaluate_count": eval_count,
        "population": df,
    }


def _extract_meta_value(text: str, key: str) -> float | None:
    """Extract a numeric value from generation metadata block."""
    match = re.search(rf"^{key}\s+(.+)$", text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).strip())
        except ValueError:
            return None
    return None


def filter_feasible(df: pd.DataFrame) -> pd.DataFrame:
    """Filter population to feasible solutions (all constraints <= 0).

    Args:
        df: Population DataFrame with Cons1, Cons2, Cons3 columns.

    Returns:
        DataFrame of feasible solutions only.
    """
    mask = (df["Cons1"] <= 0) & (df["Cons2"] <= 0) & (df["Cons3"] <= 0)
    return df[mask].copy()


def compute_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Find non-dominated solutions from a set of feasible solutions.

    Objectives: minimize Obj1 (= -productivity, so more negative = better)
                minimize Obj2 (= energy, lower = better)

    A solution a dominates b if:
      a.Obj1 <= b.Obj1 AND a.Obj2 <= b.Obj2 AND at least one strict <.

    Args:
        df: DataFrame with Obj1 and Obj2 columns (feasible solutions only).

    Returns:
        DataFrame of non-dominated solutions.
    """
    if df.empty:
        return df.copy()

    # Deduplicate on objective values (NSGA-II populations can have clones)
    df = df.drop_duplicates(subset=["Obj1", "Obj2"]).reset_index(drop=True)

    obj1 = df["Obj1"].values
    obj2 = df["Obj2"].values
    n = len(obj1)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # Check if j dominates i
            if obj1[j] <= obj1[i] and obj2[j] <= obj2[i]:
                if obj1[j] < obj1[i] or obj2[j] < obj2[i]:
                    is_dominated[i] = True
                    break

    return df[~is_dominated].copy()


def load_material_index(csv_path: Path) -> dict[int, str]:
    """Load material CSV and return 1-based index -> name mapping.

    Args:
        csv_path: Path to Adsorbents_CH4N2_{PSA,VSA}.csv.

    Returns:
        Dict mapping 1-based material index to material_name.
    """
    df = pd.read_csv(csv_path)
    # 1-based indexing to match filename convention (001, 002, ...)
    return {i + 1: name for i, name in enumerate(df["material_name"])}


def cross_validate_material(
    file_name: str,
    file_idx: int,
    csv_index: dict[int, str],
) -> bool:
    """Cross-validate material name from filename against CSV index.

    Args:
        file_name: Material name extracted from filename.
        file_idx: Material index extracted from filename.
        csv_index: Dict from load_material_index().

    Returns:
        True if match is valid, False otherwise.
    """
    if file_idx not in csv_index:
        logger.warning(
            "Material index %d not found in CSV (max %d)",
            file_idx, max(csv_index.keys()),
        )
        return False

    csv_name = csv_index[file_idx]
    if csv_name != file_name:
        logger.warning(
            "Name mismatch for index %d: filename='%s', CSV='%s'",
            file_idx, file_name, csv_name,
        )
        return False

    return True


def discover_result_files(results_dir: Path) -> list[Path]:
    """Find all NSGA-II result files, excluding archives and non-result files.

    Args:
        results_dir: Path to SuperPSA/Results/ directory.

    Returns:
        Sorted list of result file paths.
    """
    files = []
    for f in results_dir.glob("opt_*.txt"):
        # Skip files in archive subdirectories
        if "archive" in f.parent.name.lower():
            continue
        # Skip batch_singlepoint_results.csv and similar non-result files
        if not _FNAME_RE.match(f.name):
            continue
        files.append(f)

    files.sort(key=lambda p: p.name)
    return files


def build_pareto_analysis(
    results_dir: Path,
    psa_csv: Path,
    vsa_csv: Path,
    min_gen: int = 10,
    exclude_materials: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main analysis pipeline: parse, filter, compute Pareto fronts, rank.

    Args:
        results_dir: Path to SuperPSA/Results/.
        psa_csv: Path to Adsorbents_CH4N2_PSA.csv.
        vsa_csv: Path to Adsorbents_CH4N2_VSA.csv.
        min_gen: Minimum generation threshold.
        exclude_materials: Set of material names to exclude from analysis.

    Returns:
        Tuple of (pareto_analysis_df, material_ranking_df).
    """
    # Load material indices for cross-validation
    psa_index = load_material_index(psa_csv)
    vsa_index = load_material_index(vsa_csv)

    logger.info("PSA materials: %d, VSA materials: %d", len(psa_index), len(vsa_index))

    # Discover and parse all result files
    files = discover_result_files(results_dir)
    logger.info("Found %d result files", len(files))

    all_pareto_rows = []
    parse_summary = []

    for fpath in files:
        result = parse_result_file(fpath, min_gen=min_gen)
        if result is None:
            continue

        # Skip excluded materials
        if exclude_materials and result["material_name"] in exclude_materials:
            logger.info("Excluding material: %s", result["material_name"])
            continue

        mode = result["mode"]
        csv_index = psa_index if mode == "PSA" else vsa_index

        # Cross-validate material name
        cross_validate_material(
            result["material_name"],
            result["material_idx"],
            csv_index,
        )

        # Filter feasible solutions
        feasible = filter_feasible(result["population"])
        n_total = len(result["population"])
        n_feasible = len(feasible)

        # Compute per-material Pareto front
        if not feasible.empty:
            pareto = compute_pareto_front(feasible)
        else:
            pareto = feasible

        n_pareto = len(pareto)

        parse_summary.append({
            "file": fpath.name,
            "mode": mode,
            "cycle_type": result["cycle_type"],
            "material_name": result["material_name"],
            "material_idx": result["material_idx"],
            "last_gen": result["last_gen"],
            "n_total": n_total,
            "n_feasible": n_feasible,
            "n_pareto": n_pareto,
        })

        # Convert Pareto solutions to output rows
        for _, row in pareto.iterrows():
            all_pareto_rows.append({
                "mode": mode,
                "cycle_type": result["cycle_type"],
                "material_name": result["material_name"],
                "material_idx": result["material_idx"],
                "productivity": -row["Obj1"],  # negate: Obj1 = -productivity
                "energy": row["Obj2"],
                "Var1": row["Var1"],
                "Var2": row["Var2"],
                "Var3": row["Var3"],
                "Var4": row["Var4"],
                "Var5": row["Var5"],
                "Var6": row["Var6"],
                "Var7": row["Var7"],
                "Var8": row["Var8"],
                "Var9": row["Var9"],
                "Var10": row["Var10"],
                "Cons1": row["Cons1"],
                "Cons2": row["Cons2"],
                "Cons3": row["Cons3"],
                "is_globally_nondominated": False,  # computed later
            })

    # Print parse summary
    _print_parse_summary(parse_summary)

    if not all_pareto_rows:
        logger.error("No Pareto-optimal feasible solutions found across all files")
        return pd.DataFrame(), pd.DataFrame()

    pareto_df = pd.DataFrame(all_pareto_rows)

    # Compute global Pareto fronts per mode (pooling Basic + HR)
    for mode in ["PSA", "VSA"]:
        mode_mask = pareto_df["mode"] == mode
        if not mode_mask.any():
            continue

        mode_df = pareto_df[mode_mask].copy()
        mode_idx = mode_df.index.tolist()

        # Build temporary DataFrame aligned with mode_df for non-dominated sorting
        mode_obj = pd.DataFrame(
            {
                "Obj1": -mode_df["productivity"].values,
                "Obj2": mode_df["energy"].values,
            },
            index=mode_idx,  # preserve original pareto_df indices
        )
        # Deduplicate + find non-dominated set
        global_pareto = compute_pareto_front(mode_obj)

        # Map back: for each surviving (Obj1, Obj2), mark ALL matching rows
        # in pareto_df as globally non-dominated (handles clones correctly)
        gp_set = set(zip(global_pareto["Obj1"], global_pareto["Obj2"]))
        for idx in mode_idx:
            obj_pair = (-pareto_df.at[idx, "productivity"], pareto_df.at[idx, "energy"])
            if obj_pair in gp_set:
                pareto_df.at[idx, "is_globally_nondominated"] = True

    # Build material ranking
    ranking_df = _build_ranking(pareto_df)

    return pareto_df, ranking_df


def _print_parse_summary(summary: list[dict]) -> None:
    """Print a formatted summary table of parsing results."""
    if not summary:
        logger.info("No files parsed.")
        return

    print("\n" + "=" * 100)
    print(f"{'File':<70} {'Mode':<5} {'Cycle':<6} {'Pop':>5} {'Feas':>5} {'Pareto':>6}")
    print("-" * 100)
    for s in summary:
        print(
            f"{s['file']:<70} {s['mode']:<5} {s['cycle_type']:<6} "
            f"{s['n_total']:>5} {s['n_feasible']:>5} {s['n_pareto']:>6}"
        )
    print("=" * 100)

    total_files = len(summary)
    total_feasible = sum(s["n_feasible"] for s in summary)
    total_pareto = sum(s["n_pareto"] for s in summary)
    zero_feasible = sum(1 for s in summary if s["n_feasible"] == 0)
    print(
        f"Total: {total_files} files, {total_feasible} feasible solutions, "
        f"{total_pareto} Pareto-optimal points, {zero_feasible} files with 0 feasible"
    )
    print()


def _build_ranking(pareto_df: pd.DataFrame) -> pd.DataFrame:
    """Build material ranking from Pareto analysis.

    For each (mode, material), aggregate across cycle types:
      - n_pareto_points: total Pareto points across all cycles
      - n_global_contributions: points on the global front
      - best_productivity, best_energy: best values achieved
      - cycle_type_best: which cycle type contributed most global points

    Args:
        pareto_df: DataFrame from build_pareto_analysis with is_globally_nondominated.

    Returns:
        Ranking DataFrame sorted by (mode, global_rank).
    """
    rows = []
    for mode in ["PSA", "VSA"]:
        mode_df = pareto_df[pareto_df["mode"] == mode]
        if mode_df.empty:
            continue

        materials = mode_df["material_name"].unique()
        mat_stats = []

        for mat in materials:
            mat_df = mode_df[mode_df["material_name"] == mat]
            n_pareto = len(mat_df)
            global_df = mat_df[mat_df["is_globally_nondominated"]]
            n_global = len(global_df)

            best_prod = mat_df["productivity"].max()
            best_energy = mat_df["energy"].min()
            prod_range = mat_df["productivity"].max() - mat_df["productivity"].min()
            energy_range = mat_df["energy"].max() - mat_df["energy"].min()

            # Which cycle type contributed most global points
            if n_global > 0:
                cycle_counts = global_df["cycle_type"].value_counts()
                cycle_best = cycle_counts.index[0]
            else:
                # Fall back to cycle type with most Pareto points
                cycle_counts = mat_df["cycle_type"].value_counts()
                cycle_best = cycle_counts.index[0]

            mat_stats.append({
                "mode": mode,
                "material_name": mat,
                "cycle_type_best": cycle_best,
                "n_pareto_points": n_pareto,
                "n_global_contributions": n_global,
                "best_productivity": best_prod,
                "best_energy": best_energy,
                "productivity_range": prod_range,
                "energy_range": energy_range,
            })

        # Rank by: primary = n_global_contributions (desc), secondary = best_energy (asc)
        mat_stats.sort(key=lambda x: (-x["n_global_contributions"], x["best_energy"]))
        for rank, ms in enumerate(mat_stats, 1):
            ms["global_rank"] = rank

        rows.extend(mat_stats)

    return pd.DataFrame(rows)


def print_ranking_summary(ranking_df: pd.DataFrame) -> None:
    """Print a formatted ranking summary to stdout."""
    if ranking_df.empty:
        print("No ranking data available.")
        return

    for mode in ["PSA", "VSA"]:
        mode_df = ranking_df[ranking_df["mode"] == mode]
        if mode_df.empty:
            continue

        print(f"\n{'=' * 110}")
        print(f"  {mode} Material Ranking (by global Pareto contribution)")
        print(f"{'=' * 110}")
        print(
            f"{'Rank':>4}  {'Material':<50} {'Cycle':<6} "
            f"{'Pareto':>6} {'Global':>6} {'BestProd':>10} {'BestEnergy':>10}"
        )
        print("-" * 110)
        for _, row in mode_df.iterrows():
            print(
                f"{row['global_rank']:>4}  {row['material_name']:<50} "
                f"{row['cycle_type_best']:<6} {row['n_pareto_points']:>6} "
                f"{row['n_global_contributions']:>6} "
                f"{row['best_productivity']:>10.2f} {row['best_energy']:>10.1f}"
            )
        print(f"{'=' * 110}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse NSGA-II optimization results and analyze Pareto fronts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python parse_nsga2_results.py\n"
            "  python parse_nsga2_results.py --results-dir /path/to/Results --min-gen 50\n"
        ),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_RESULTS_DIR,
        help=f"Directory containing opt_*.txt result files (default: {_RESULTS_DIR})",
    )
    parser.add_argument(
        "--psa-csv", type=Path, default=_PSA_CSV,
        help=f"PSA material index CSV (default: {_PSA_CSV})",
    )
    parser.add_argument(
        "--vsa-csv", type=Path, default=_VSA_CSV,
        help=f"VSA material index CSV (default: {_VSA_CSV})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_OUTPUT_DIR,
        help=f"Output directory for CSV files (default: {_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-gen", type=int, default=10,
        help="Minimum number of generations required to include a file (default: 10)",
    )
    parser.add_argument(
        "--exclude-materials", type=str, default=None,
        help="Comma-separated list of material names to exclude from analysis",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate inputs
    if not args.results_dir.is_dir():
        parser.error(f"Results directory does not exist: {args.results_dir}")
    if not args.psa_csv.is_file():
        parser.error(f"PSA CSV not found: {args.psa_csv}")
    if not args.vsa_csv.is_file():
        parser.error(f"VSA CSV not found: {args.vsa_csv}")

    # Parse exclusion list
    exclude_set = None
    if args.exclude_materials:
        exclude_set = {s.strip() for s in args.exclude_materials.split(",")}
        logger.info("Excluding materials: %s", exclude_set)

    # Run analysis
    pareto_df, ranking_df = build_pareto_analysis(
        results_dir=args.results_dir,
        psa_csv=args.psa_csv,
        vsa_csv=args.vsa_csv,
        min_gen=args.min_gen,
        exclude_materials=exclude_set,
    )

    if pareto_df.empty:
        logger.error("No results to save.")
        return

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pareto_path = args.output_dir / "pareto_analysis.csv"
    pareto_df.to_csv(pareto_path, index=False)
    logger.info("Saved Pareto analysis (%d rows) to %s", len(pareto_df), pareto_path)

    ranking_path = args.output_dir / "material_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    logger.info("Saved material ranking (%d rows) to %s", len(ranking_df), ranking_path)

    # Print summary
    print_ranking_summary(ranking_df)

    # Print global Pareto front stats
    for mode in ["PSA", "VSA"]:
        gp = pareto_df[(pareto_df["mode"] == mode) & pareto_df["is_globally_nondominated"]]
        if gp.empty:
            continue
        n_mats = gp["material_name"].nunique()
        cycle_counts = gp["cycle_type"].value_counts().to_dict()
        print(f"\n{mode} global Pareto front: {len(gp)} solutions from {n_mats} materials")
        print(f"  Cycle type breakdown: {cycle_counts}")
        print(
            f"  Productivity range: {gp['productivity'].min():.2f} – "
            f"{gp['productivity'].max():.2f} mol/kg/h"
        )
        print(
            f"  Energy range: {gp['energy'].min():.1f} – "
            f"{gp['energy'].max():.1f} kWh/ton"
        )


if __name__ == "__main__":
    main()
