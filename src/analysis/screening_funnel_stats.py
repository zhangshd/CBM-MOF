#!/usr/bin/env python3
"""Count MOFs at each screening stage, broken down by experimental/hypothetical/total.

Experimental MOFs: IDs starting with CoRE-, MOSAEC-, ARC-DB12-, ARC-DB14-
Hypothetical MOFs: all other ARC-* prefixes

The pipeline topology:
  Stage 0-2: linear (Raw → Dedup → Geometric)
  Stage 3:   from 2 (Elemental availability filter)
  Stage 4:   from 3 (MOFSNN stability)
  Stage 5:   from 4 (ALIGNN inference, atom-count limit)
  Stage 6-7: from 5 (Top-100 PSA/VSA selections)
  Stage 8:   union of 6+7 (186 unique candidates)

Loss is computed relative to each stage's parent, not the previous row.

Usage:
    python screening_funnel_stats.py
    python screening_funnel_stats.py --output funnel.csv
    python screening_funnel_stats.py --model-dir /path/to/model_ep150
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

EXP_PREFIXES = ("CoRE-", "MOSAEC-", "ARC-DB12-", "ARC-DB14-")


def classify_mof_ids(ids: pd.Series) -> tuple[int, int, int]:
    """Return (total, exp, hypo) counts for a Series of MOF IDs."""
    total = len(ids)
    exp = ids.str.startswith(EXP_PREFIXES).sum()
    hypo = total - exp
    return int(total), int(exp), int(hypo)


def load_ids_csv(path: Path, id_col: str) -> Optional[pd.Series]:
    """Load MOF IDs from a CSV file, returning None if file is missing."""
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return None
    df = pd.read_csv(path, usecols=[id_col])
    return df[id_col]


def load_ids_txt(path: Path) -> Optional[pd.Series]:
    """Load MOF IDs from a headerless text file (one ID per line).

    Strips .cif suffix if present for consistency with other stages.
    """
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return None
    df = pd.read_csv(path, header=None, names=["name"])
    df["name"] = df["name"].str.replace(r"\.cif$", "", regex=True)
    return df["name"]


def load_union_csv(paths: list[Path], id_col: str) -> Optional[pd.Series]:
    """Load and union MOF IDs from multiple CSV files."""
    parts = []
    for p in paths:
        if not p.exists():
            logger.warning("File not found, skipping: %s", p)
            continue
        df = pd.read_csv(p, usecols=[id_col])
        parts.append(df[id_col])
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def _load_precious_set(model_dir: Path) -> Optional[set]:
    """Load the set of MOF IDs with precious/rare metals from the CIF-detection cache.

    Cache file: {model_dir}/top_candidates/precious_rare_flags_geometric.csv
    Generate with: detect_precious_rare_metals_parallel() from filter_stable_candidates.py
    """
    cache = model_dir / "top_candidates" / "precious_rare_flags_geometric.csv"
    if not cache.exists():
        logger.warning(
            "Precious-metal cache not found: %s\n"
            "  Run detect_precious_rare_metals_parallel() on geometric-screened MOFs "
            "and save to this path.",
            cache,
        )
        return None
    df = pd.read_csv(cache)
    return set(df.loc[df["has_precious_rare"], "mof_id"])


def _load_mofsnn_passing_set() -> Optional[tuple[set, set]]:
    """Load the set of MOF IDs that pass both MOFSNN stability criteria (SSD+WS24)."""
    path = REPO_ROOT / "data" / "processed" / "stabilities" / "infer_results_mofsnn.csv"
    if not path.exists():
        logger.warning("MOFSNN file not found: %s", path)
        return None
    stab = pd.read_csv(path, usecols=["MofName", "SSD_pred", "WS24_water_pred"])
    passes = set(
        stab.loc[(stab["SSD_pred"] == 1) & (stab["WS24_water_pred"] == 1), "MofName"]
    )
    covered = set(stab["MofName"])
    # Uncovered MOFs are retained (conservative: assumed stable)
    return passes, covered


def build_funnel_table(model_dir: Path) -> pd.DataFrame:
    """Build the screening funnel statistics table.

    Pipeline order (reordered for narrative):
      Raw → Dedup → Geometric → Elemental → MOFSNN → Inference → Top-100 → 186

    Each intermediate set is computed via set operations on the cached
    precious-metal flags, MOFSNN predictions, and inference output.

    Args:
        model_dir: Path to the model results directory (e.g. results/alignn/model_ep150/).

    Returns:
        DataFrame with one row per stage and columns for counts and losses.
    """
    data_dir = REPO_ROOT / "data" / "processed"

    # Pre-load shared data once to avoid repeated file reads
    precious_set = _load_precious_set(model_dir)
    mofsnn_result = _load_mofsnn_passing_set()
    geo = load_ids_txt(data_dir / "textural_screened" / "textural_screened_list.txt")

    # Cache intermediate results for the cascading filter pipeline
    _after_elem: Optional[set] = None
    _after_mofsnn: Optional[set] = None

    def _get_after_elemental() -> Optional[set]:
        nonlocal _after_elem
        if _after_elem is not None:
            return _after_elem
        if geo is None or precious_set is None:
            return None
        _after_elem = set(geo[~geo.isin(precious_set)])
        return _after_elem

    def _get_after_mofsnn() -> Optional[set]:
        nonlocal _after_mofsnn
        if _after_mofsnn is not None:
            return _after_mofsnn
        after_elem = _get_after_elemental()
        if after_elem is None or mofsnn_result is None:
            return None
        passes, covered = mofsnn_result
        # Keep MOFs that pass MOFSNN or are not covered (conservative retention)
        _after_mofsnn = {m for m in after_elem if m in passes or m not in covered}
        return _after_mofsnn

    def _compute_after_elemental() -> Optional[pd.Series]:
        result = _get_after_elemental()
        return pd.Series(list(result)) if result is not None else None

    def _compute_after_mofsnn() -> Optional[pd.Series]:
        result = _get_after_mofsnn()
        return pd.Series(list(result)) if result is not None else None

    def _compute_after_inference() -> Optional[pd.Series]:
        after_mofsnn = _get_after_mofsnn()
        infer = load_ids_csv(
            model_dir / "full_library_inference" / "full_library_with_api.csv", "mof_id"
        )
        if after_mofsnn is None or infer is None:
            return None
        after_infer = after_mofsnn & set(infer)
        return pd.Series(list(after_infer))

    # (stage_num, description, parent_stage, loader)
    stages = [
        (0, "Raw features", None,
         lambda: load_ids_csv(data_dir / "RAC_and_zeo_features.csv", "name")),
        (1, "Deduplicated", 0,
         lambda: load_ids_csv(data_dir / "RAC_and_zeo_features_deduplicated.csv", "name")),
        (2, "Geometric screening", 1,
         lambda: load_ids_txt(data_dir / "textural_screened" / "textural_screened_list.txt")),
        (3, "Elemental availability", 2,
         _compute_after_elemental),
        (4, "MOFSNN stability", 3,
         _compute_after_mofsnn),
        (5, "ALIGNN inference", 4,
         _compute_after_inference),
        (6, "Top-100 PSA", 5,
         lambda: load_union_csv([
             model_dir / "top_candidates" / "exp_top50_psa.csv",
             model_dir / "top_candidates" / "hypo_top50_psa.csv",
         ], "mof_id")),
        (7, "Top-100 VSA", 5,
         lambda: load_union_csv([
             model_dir / "top_candidates" / "exp_top50_vsa.csv",
             model_dir / "top_candidates" / "hypo_top50_vsa.csv",
         ], "mof_id")),
        (8, "186 unique candidates", 5,
         lambda: load_ids_csv(model_dir / "process_candidates" / "gcmc_vs_ml_comparison.csv", "mof_id")),
    ]

    # First pass: compute counts
    stage_counts = {}  # stage_num -> (total, exp, hypo)
    stage_meta = []    # list of (stage_num, description, parent_stage)

    for stage_num, description, parent, loader in stages:
        ids = loader()
        if ids is None:
            stage_meta.append((stage_num, description, parent, None))
            continue
        counts = classify_mof_ids(ids)
        stage_counts[stage_num] = counts
        stage_meta.append((stage_num, description, parent, counts))

    # Second pass: compute losses relative to parent
    rows = []
    for stage_num, description, parent, counts in stage_meta:
        if counts is None:
            rows.append({
                "Stage": stage_num, "Description": description, "Parent": "-",
                "Total": None, "Exp": None, "Hypo": None,
                "Loss_Total": None, "Loss_Exp": None, "Loss_Hypo": None,
                "Loss%_Total": None, "Loss%_Exp": None, "Loss%_Hypo": None,
            })
            continue

        total, exp, hypo = counts
        parent_label = str(parent) if parent is not None else "-"

        if parent is not None and parent in stage_counts:
            p_total, p_exp, p_hypo = stage_counts[parent]
            loss_total = p_total - total
            loss_exp = p_exp - exp
            loss_hypo = p_hypo - hypo
            loss_pct_total = (loss_total / p_total * 100) if p_total else 0.0
            loss_pct_exp = (loss_exp / p_exp * 100) if p_exp else 0.0
            loss_pct_hypo = (loss_hypo / p_hypo * 100) if p_hypo else 0.0
        else:
            loss_total = loss_exp = loss_hypo = 0
            loss_pct_total = loss_pct_exp = loss_pct_hypo = 0.0

        rows.append({
            "Stage": stage_num,
            "Description": description,
            "Parent": parent_label,
            "Total": total,
            "Exp": exp,
            "Hypo": hypo,
            "Loss_Total": loss_total,
            "Loss_Exp": loss_exp,
            "Loss_Hypo": loss_hypo,
            "Loss%_Total": round(loss_pct_total, 2),
            "Loss%_Exp": round(loss_pct_exp, 2),
            "Loss%_Hypo": round(loss_pct_hypo, 2),
        })

    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    """Format the funnel table as a human-readable aligned text table."""
    headers = list(df.columns)

    str_rows = []
    for _, row in df.iterrows():
        r = []
        for h in headers:
            val = row[h]
            if val is None:
                r.append("N/A")
            elif h.startswith("Loss%"):
                r.append(f"{val:.2f}%")
            elif isinstance(val, float):
                r.append(f"{val:.0f}")
            else:
                r.append(str(val))
        str_rows.append(r)

    # Compute column widths
    all_rows = [headers] + str_rows
    col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    # Left-align text columns, right-align numeric
    text_cols = {"Stage", "Description", "Parent"}

    def fmt_row(row: list[str]) -> str:
        parts = []
        for i, (val, width) in enumerate(zip(row, col_widths)):
            if headers[i] in text_cols:
                parts.append(val.ljust(width))
            else:
                parts.append(val.rjust(width))
        return " | ".join(parts)

    lines = [fmt_row(headers)]
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in str_rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Count MOFs at each screening stage (exp/hypo/total).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "results" / "alignn" / "model_ep150",
        help="Path to model results directory (default: results/alignn/model_ep150/).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Save the table as CSV to this path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    df = build_funnel_table(args.model_dir)

    print(format_table(df))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        logger.info("Saved CSV to %s", args.output)


if __name__ == "__main__":
    main()
