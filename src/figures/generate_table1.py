#!/usr/bin/env python3
"""Generate Table 1: PSA/VSA process metrics summary for the CBM-MOF paper.

Merges material ranking, IAST selectivity, adsorbent density, and Pareto
re-evaluation recovery into a publication-ready summary table.

Output:
    - CSV:      {out_dir}/table1_summary.csv
    - Markdown: {out_dir}/table1_summary.md
    - stdout:   both tables printed
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(
    "/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150"
)
_SUPERPSA_DATA = Path("/home/zhangsd/repos/CBM-MOF/src/SuperPSA/data")
_SUPERPSA_RESULTS = Path("/home/zhangsd/repos/CBM-MOF/src/SuperPSA/Results_extDSL_HR")

# ── Short name mapping (consistent with fig_psa_performance.py) ──────────────

_SHORT_NAME_MAP = {
    "CoRE-2020[Cu][pts]3[ASR]1": "ATC-Cu",
    "CoRE-2014[Al][nan]3[ASR]4": "CoRE-Al",
    "CoRE-2023[Cu][pts]3[ASR]2": "CoRE-Cu-2023",
    "CoRE-2009[Cd][nuc]3[ASR]1": "CoRE-Cd",
    "CoRE-2013[Mg][dia]3[ASR]1": "CoRE-Mg",
    "CoRE-2010[Co][pts]3[ASR]2": "CoRE-Co",
    "CoRE-2011[Ni][dia]3[ASR]1": "CoRE-Ni",
    "MOSAEC-YOBPOW_full_REPEAT": "YOBPOW",
}

_BENCHMARK_MATERIAL = "CoRE-2020[Cu][pts]3[ASR]1"


def _shorten_material_name(name: str) -> str:
    """Create a short, unique display name for a material."""
    if name in _SHORT_NAME_MAP:
        return _SHORT_NAME_MAP[name]

    # ARC-DB1 pattern: extract formula + No{digits}
    m = re.match(r"ARC-DB1-(\w+)-(\w+)-\w+_\w+_No(\d+)_repeat", name)
    if m:
        return f"{m.group(1)} #{m.group(3)}"

    # ARC-DB0 pattern: extract o{digits} identifiers
    m = re.match(
        r"ARC-DB\d+-m\d+_o(\d+)_o(\d+)_f\d+_fsc(?:\.sym\.(\d+))?_repeat",
        name,
    )
    if m:
        o1 = m.group(1)
        sym = m.group(3)
        suffix = f".{sym}" if sym else ""
        return f"ARC-o{o1}{suffix}"

    return name[:15]


def _build_short_names(materials: list[str]) -> dict[str, str]:
    """Build unique short names; append suffix if collisions exist."""
    raw = {m: _shorten_material_name(m) for m in materials}
    # Check for collisions
    from collections import Counter

    counts = Counter(raw.values())
    result = {}
    seen: dict[str, int] = {}
    for m in materials:
        short = raw[m]
        if counts[short] > 1:
            idx = seen.get(short, 0) + 1
            seen[short] = idx
            result[m] = f"{short}-{idx}"
        else:
            result[m] = short
    return result


def _load_density(mode: str) -> dict[str, float]:
    """Load adsorbent density (ro_s) from the SuperPSA Adsorbents CSV."""
    csv_path = _SUPERPSA_DATA / f"Adsorbents_CH4N2_{mode}.csv"
    if not csv_path.exists():
        logger.warning("Adsorbents CSV not found: %s", csv_path)
        return {}
    df = pd.read_csv(csv_path)
    return dict(zip(df["material_name"], df["ro_s [kg/m^3]"]))


def _load_recovery() -> dict[tuple[str, str], float]:
    """Load max recovery per (mode, material) from pareto_metrics_summary.csv.

    Returns dict mapping (mode, material_name) -> max_recovery (as percentage).
    """
    csv_path = _SUPERPSA_RESULTS / "pareto_metrics_summary.csv"
    if not csv_path.exists():
        logger.info("Pareto metrics summary not found: %s (skipping recovery)", csv_path)
        return {}
    df = pd.read_csv(csv_path)
    if "recovery" not in df.columns:
        logger.warning("No 'recovery' column in %s", csv_path)
        return {}
    result = {}
    for (mode, mat), grp in df.groupby(["mode", "material_name"]):
        result[(mode, mat)] = grp["recovery"].max() * 100.0  # fraction -> %
    return result


def _format_cycle(cycle_type_best: str) -> str:
    """Format cycle type for display."""
    if isinstance(cycle_type_best, str):
        upper = cycle_type_best.upper()
        if upper == "HR":
            return "HR"
        return cycle_type_best.capitalize()
    return str(cycle_type_best)


def build_table(model_dir: Path) -> pd.DataFrame:
    """Build the merged summary table for both PSA and VSA."""
    opt_dir = model_dir / "psa_optimization"

    # 1) Material ranking
    ranking = pd.read_csv(opt_dir / "material_ranking.csv")
    logger.info("Loaded %d ranking rows", len(ranking))

    # 2) IAST selectivity
    iast = pd.read_csv(opt_dir / "iast_dsl_selectivity.csv")
    logger.info("Loaded %d IAST rows", len(iast))

    # 3) Density per mode
    density_psa = _load_density("PSA")
    density_vsa = _load_density("VSA")

    # 4) Recovery
    recovery_map = _load_recovery()

    # Build IAST lookup: material_name -> {PSA: alpha, VSA: alpha}
    iast_lookup: dict[str, dict[str, float]] = {}
    for _, row in iast.iterrows():
        name = row["MofName"]
        iast_lookup[name] = {
            "PSA": row["alpha_IAST_PSA"],
            "VSA": row["alpha_IAST_VSA"],
        }

    rows = []
    for _, r in ranking.iterrows():
        mode = r["mode"]
        mat = r["material_name"]

        # Density
        density_map = density_psa if mode == "PSA" else density_vsa
        rho_s = density_map.get(mat, float("nan"))

        # IAST selectivity
        alpha = float("nan")
        if mat in iast_lookup and mode in iast_lookup[mat]:
            alpha = iast_lookup[mat][mode]

        # Recovery
        rec = recovery_map.get((mode, mat))

        # Short name
        short = _shorten_material_name(mat)

        is_benchmark = mat == _BENCHMARK_MATERIAL

        rows.append(
            {
                "mode": mode,
                "material_name": mat,
                "short_name": short,
                "is_benchmark": is_benchmark,
                "global_rank": int(r["global_rank"]),
                "cycle_type": _format_cycle(r["cycle_type_best"]),
                "rho_s": rho_s,
                "alpha_IAST": alpha,
                "best_productivity": r["best_productivity"],
                "best_energy": r["best_energy"],
                "recovery_pct": rec,
            }
        )

    df = pd.DataFrame(rows)
    # Ensure unique short names within each mode
    for mode in df["mode"].unique():
        mask = df["mode"] == mode
        mats = df.loc[mask, "material_name"].tolist()
        name_map = _build_short_names(mats)
        for mat, short in name_map.items():
            df.loc[(df["mode"] == mode) & (df["material_name"] == mat), "short_name"] = short

    # Mark benchmark with *
    df.loc[df["is_benchmark"], "short_name"] = df.loc[df["is_benchmark"], "short_name"] + "*"

    return df


def _format_value(val, fmt: str = ".1f", na_str: str = "\u2014") -> str:
    """Format a numeric value; return na_str if NaN/None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return na_str
    return f"{val:{fmt}}"


def to_markdown(df: pd.DataFrame) -> str:
    """Convert the summary DataFrame to a Markdown table string."""
    lines: list[str] = []

    for mode in ["PSA", "VSA"]:
        sub = df[df["mode"] == mode].sort_values("global_rank")
        if sub.empty:
            continue

        lines.append(f"\n### {mode} Process\n")
        header = (
            "| Rank | Material | Cycle | "
            "\u03c1_s (kg/m\u00b3) | \u03b1_IAST | "
            "Productivity (mol/kg/h) | Energy (kWh/ton) | Recovery (%) |"
        )
        sep = "|------|----------|-------|" + "-------------|--------|" + "-------------------------|-------------------|--------------|"
        lines.append(header)
        lines.append(sep)

        for _, row in sub.iterrows():
            rank = row["global_rank"]
            name = row["short_name"]
            cycle = row["cycle_type"]
            rho = _format_value(row["rho_s"], ".1f")
            alpha = _format_value(row["alpha_IAST"], ".2f")
            prod = _format_value(row["best_productivity"], ".2f")
            energy = _format_value(row["best_energy"], ".1f")
            rec = _format_value(row["recovery_pct"], ".1f")
            lines.append(
                f"| {rank} | {name} | {cycle} | {rho} | {alpha} | {prod} | {energy} | {rec} |"
            )

        lines.append("")

    return "\n".join(lines)


def save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    """Save CSV and Markdown outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "table1_summary.csv"
    csv_cols = [
        "mode", "global_rank", "short_name", "material_name",
        "cycle_type", "rho_s", "alpha_IAST",
        "best_productivity", "best_energy", "recovery_pct",
    ]
    df[csv_cols].to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("CSV saved: %s", csv_path)

    # Markdown
    md_path = out_dir / "table1_summary.md"
    md_text = to_markdown(df)
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Markdown saved: %s", md_path)

    # Print
    print(md_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Table 1: PSA/VSA process metrics summary."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_DEFAULT_MODEL_DIR,
        help="Model results directory (default: ep150)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = build_table(args.model_dir)
    out_dir = args.model_dir / "psa_optimization"
    save_outputs(df, out_dir)


if __name__ == "__main__":
    main()
