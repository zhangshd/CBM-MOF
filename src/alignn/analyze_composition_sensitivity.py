"""
analyze_composition_sensitivity.py
===================================
Post-GCMC analysis: Compare separation performance at 20:80 vs 50:50
CH4:N2 feed compositions across ALL available MOFs.

Data sources (all under results/alignn/model_ep150/):
  - gcmc_exp_top/       (86 exp MOFs: GCMC 20:80 + 50:50 + Widom)
  - gcmc_hypo_top/      (100 hypo MOFs: GCMC 20:80 + 50:50 + Widom)
  - gcmc_top_candidates/(194 old MOFs: GCMC 20:80 + 50:50 + Widom)
  - gcmc_atccu/         (1 MOF: 50:50 only, no 20:80 or Widom)

Computes:
  1. API at both compositions for all MOFs with both 20:80 and 50:50 data
  2. Spearman rank correlation (rank_2080 vs rank_5050)
  3. Rank-switching MOFs (rank change >= threshold)
  4. Exp vs hypo comparison

Usage:
    python src/alignn/analyze_composition_sensitivity.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path for imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignn.parse_validation_results import parse_gcmc, parse_widom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "composition_sensitivity"

# Feed mole fractions
COMP_2080 = {"y_ch4": 0.20, "y_n2": 0.80}
COMP_5050 = {"y_ch4": 0.50, "y_n2": 0.50}

# Exp/hypo classification prefixes
EXP_PREFIXES = ("CoRE-", "MOSAEC-", "ARC-DB12-", "ARC-DB14-")

# Data source directories: (label, gcmc_2080_dir, gcmc_5050_dir, widom_dir)
DATA_SOURCES = [
    (
        "exp_top",
        MODEL_DIR / "gcmc_exp_top" / "gcmc_DreidingTraPPEJson",
        MODEL_DIR / "gcmc_exp_top" / "gcmc_DreidingTraPPEJson_5050",
        MODEL_DIR / "gcmc_exp_top" / "widom_DREIDING",
    ),
    (
        "hypo_top",
        MODEL_DIR / "gcmc_hypo_top" / "gcmc_DreidingTraPPEJson",
        MODEL_DIR / "gcmc_hypo_top" / "gcmc_DreidingTraPPEJson_5050",
        MODEL_DIR / "gcmc_hypo_top" / "widom_DREIDING",
    ),
    (
        "old_194",
        MODEL_DIR / "gcmc_top_candidates" / "gcmc_DreidingTraPPEJson",
        MODEL_DIR / "gcmc_top_candidates" / "gcmc_DreidingTraPPEJson_5050",
        MODEL_DIR / "gcmc_top_candidates" / "widom_DREIDING",
    ),
]

# ATC-Cu has only 50:50 GCMC, no 20:80 or Widom — excluded from composition
# sensitivity but noted in the summary.


# ---------------------------------------------------------------------------
# Widom loader that handles both naming conventions
# ---------------------------------------------------------------------------

def load_widom(widom_dir: Path) -> pd.DataFrame:
    """
    Load Widom results from widom_dir, handling both old
    (widom_results*.csv) and new (raspa2_widom_results*.csv) naming.
    """
    csv_files = sorted(widom_dir.glob("widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("raspa2_widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("batch_*/widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("batch_*/raspa2_widom_results*.csv"))
    # Deduplicate
    csv_files = sorted(set(csv_files))
    if not csv_files:
        raise FileNotFoundError(
            f"No widom CSV files found in {widom_dir} or its batch_* subdirs"
        )

    dfs = [pd.read_csv(f) for f in csv_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"  Widom raw: {len(df_raw):,} rows from {len(csv_files)} files", flush=True)

    df_raw = df_raw[df_raw["Temperature[K]"].round(1) == 298.0].copy()

    df_ch4 = (
        df_raw[df_raw["GasName"] == "methane"]
        .groupby("MofName")["AdsorptionHeat"]
        .mean()
        .rename("QstCH4_gcmc")
        .reset_index()
        .rename(columns={"MofName": "mof_id"})
    )
    df_n2 = (
        df_raw[df_raw["GasName"] == "N2"]
        .groupby("MofName")["AdsorptionHeat"]
        .mean()
        .rename("QstN2_gcmc")
        .reset_index()
        .rename(columns={"MofName": "mof_id"})
    )

    df_widom = df_ch4.merge(df_n2, on="mof_id", how="outer")
    print(f"  Widom pivoted: {len(df_widom):,} unique MOFs", flush=True)
    return df_widom


# ---------------------------------------------------------------------------
# API calculation
# ---------------------------------------------------------------------------

def compute_api_from_gcmc(
    df: pd.DataFrame,
    y_ch4: float,
    y_n2: float,
    prefix: str = "gcmc_",
) -> pd.DataFrame:
    """
    Compute PSA/VSA API from GCMC columns.

    API = (alpha - 1) * WC / |Qst_CH4|
    where alpha = (q_CH4 / q_N2) * (y_N2 / y_CH4)

    PSA: P_high = 10 bar (1000kPa), P_low = 1 bar (100kPa)
    VSA: P_high = 1 bar (100kPa),  P_low = 0.1 bar (10kPa)
    """
    result = df.copy()
    qst_abs = np.abs(result["QstCH4_gcmc"])

    process_defs = [
        ("PSA", f"{prefix}AdsCH4_1000kPa", f"{prefix}AdsCH4_100kPa",
                f"{prefix}AdsN2_1000kPa",  f"{prefix}AdsN2_100kPa"),
        ("VSA", f"{prefix}AdsCH4_100kPa",  f"{prefix}AdsCH4_10kPa",
                f"{prefix}AdsN2_100kPa",   f"{prefix}AdsN2_10kPa"),
    ]
    for proc, ch4_hi, ch4_lo, n2_hi, n2_lo in process_defs:
        q_ch4_hi = result[ch4_hi]
        q_n2_hi = result[n2_hi]

        wc = q_ch4_hi - result[ch4_lo]
        alpha = np.where(
            q_n2_hi > 1e-10,
            (q_ch4_hi / q_n2_hi) * (y_n2 / y_ch4),
            np.nan,
        )
        valid = (qst_abs > 1e-10) & np.isfinite(alpha) & (alpha > 1e-10)
        api = np.where(valid, (alpha - 1) * wc / qst_abs, np.nan)

        result[f"{proc}_alpha"] = alpha
        result[f"{proc}_WC"] = wc
        result[f"{proc}_API"] = api

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_exp_hypo(mof_id: str) -> bool:
    """Return True if experimental MOF."""
    return any(mof_id.startswith(p) for p in EXP_PREFIXES)


def load_all_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate GCMC (20:80, 50:50) and Widom from all data sources.
    Returns (df_gcmc_2080, df_gcmc_5050, df_widom) with deduplicated mof_id.
    """
    all_2080, all_5050, all_widom = [], [], []

    for label, dir_2080, dir_5050, dir_widom in DATA_SOURCES:
        print(f"\n--- Loading source: {label} ---", flush=True)

        # GCMC 20:80
        if dir_2080.exists():
            print(f"  Parsing GCMC 20:80 from {dir_2080.name} ...", flush=True)
            df = parse_gcmc(dir_2080)
            df["source"] = label
            all_2080.append(df)
        else:
            print(f"  [SKIP] GCMC 20:80 not found: {dir_2080}", flush=True)

        # GCMC 50:50
        if dir_5050.exists():
            print(f"  Parsing GCMC 50:50 from {dir_5050.name} ...", flush=True)
            df = parse_gcmc(dir_5050)
            df["source"] = label
            all_5050.append(df)
        else:
            print(f"  [SKIP] GCMC 50:50 not found: {dir_5050}", flush=True)

        # Widom
        if dir_widom.exists():
            print(f"  Parsing Widom from {dir_widom.name} ...", flush=True)
            df = load_widom(dir_widom)
            df["source"] = label
            all_widom.append(df)
        else:
            print(f"  [SKIP] Widom not found: {dir_widom}", flush=True)

    # Concatenate and deduplicate (keep first occurrence)
    def dedup(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=["mof_id"])
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="mof_id", keep="first")
        return combined

    df_2080 = dedup(all_2080)
    df_5050 = dedup(all_5050)
    df_widom = dedup(all_widom)

    print(f"\n--- Totals after dedup ---", flush=True)
    print(f"  GCMC 20:80: {len(df_2080)} MOFs", flush=True)
    print(f"  GCMC 50:50: {len(df_5050)} MOFs", flush=True)
    print(f"  Widom:      {len(df_widom)} MOFs", flush=True)

    return df_2080, df_5050, df_widom


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(output_dir: Path, rank_threshold: int = 10) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all data sources
    print("=" * 70, flush=True)
    print("Step 1: Loading GCMC + Widom from all data sources", flush=True)
    print("=" * 70, flush=True)
    df_2080, df_5050, df_widom = load_all_sources()

    # Step 2: Find common MOFs with GCMC at BOTH compositions + Widom
    common_mofs = set(df_2080["mof_id"]) & set(df_5050["mof_id"])
    widom_mofs = set(df_widom["mof_id"])
    valid_mofs = common_mofs & widom_mofs
    print(f"\nStep 2: Finding common MOFs", flush=True)
    print(f"  GCMC at both compositions: {len(common_mofs)}", flush=True)
    print(f"  + Widom available:         {len(valid_mofs)}", flush=True)

    # Also count MOFs with only one composition
    only_2080 = set(df_2080["mof_id"]) - set(df_5050["mof_id"])
    only_5050 = set(df_5050["mof_id"]) - set(df_2080["mof_id"])
    print(f"  Only 20:80 (no 50:50): {len(only_2080)}", flush=True)
    print(f"  Only 50:50 (no 20:80): {len(only_5050)}", flush=True)

    # Step 3: Merge and compute API
    print(f"\nStep 3: Computing API at both compositions", flush=True)
    df_m2080 = (
        df_2080[df_2080["mof_id"].isin(valid_mofs)]
        .drop(columns=["source"], errors="ignore")
        .merge(df_widom.drop(columns=["source"], errors="ignore"), on="mof_id", how="inner")
    )
    df_m5050 = (
        df_5050[df_5050["mof_id"].isin(valid_mofs)]
        .drop(columns=["source"], errors="ignore")
        .merge(df_widom.drop(columns=["source"], errors="ignore"), on="mof_id", how="inner")
    )

    df_m2080 = compute_api_from_gcmc(df_m2080, **COMP_2080)
    df_m5050 = compute_api_from_gcmc(df_m5050, **COMP_5050)

    # Slim down and rename for merge
    api_cols = ["PSA_alpha", "PSA_WC", "PSA_API", "VSA_alpha", "VSA_WC", "VSA_API"]
    keep_2080 = ["mof_id"] + api_cols
    keep_5050 = ["mof_id"] + api_cols

    rename_2080 = {c: f"{c}_2080" for c in api_cols}
    rename_5050 = {c: f"{c}_5050" for c in api_cols}

    df_slim_2080 = df_m2080[keep_2080].rename(columns=rename_2080)
    df_slim_5050 = df_m5050[keep_5050].rename(columns=rename_5050)

    df_comp = df_slim_2080.merge(df_slim_5050, on="mof_id", how="inner")

    # Add exp/hypo classification
    df_comp["is_exp"] = df_comp["mof_id"].apply(classify_exp_hypo)
    print(f"  Final comparison table: {len(df_comp)} MOFs", flush=True)
    print(f"  Exp: {df_comp['is_exp'].sum()}, Hypo: {(~df_comp['is_exp']).sum()}", flush=True)

    # Step 4: Spearman rank correlation
    print(f"\nStep 4: Spearman rank correlations (API_2080 vs API_5050)", flush=True)
    spearman_results = {}
    for proc in ["PSA", "VSA"]:
        col_2080 = f"{proc}_API_2080"
        col_5050 = f"{proc}_API_5050"
        mask = df_comp[col_2080].notna() & df_comp[col_5050].notna()
        n = mask.sum()
        if n >= 3:
            rho, pval = spearmanr(
                df_comp.loc[mask, col_2080], df_comp.loc[mask, col_5050]
            )
        else:
            rho, pval = np.nan, np.nan
        spearman_results[proc] = {"rho": rho, "pval": pval, "n": n}
        print(f"  {proc}: rho = {rho:.4f}, p = {pval:.2e}, n = {n}", flush=True)

    # Step 5: Rank switching analysis
    print(f"\nStep 5: Rank-switching MOFs (threshold >= {rank_threshold})", flush=True)
    for proc in ["PSA", "VSA"]:
        col_2080 = f"{proc}_API_2080"
        col_5050 = f"{proc}_API_5050"
        mask = df_comp[col_2080].notna() & df_comp[col_5050].notna()
        subset = df_comp.loc[mask].copy()

        # Rank descending (higher API = better = rank 1)
        subset[f"{proc}_rank_2080"] = subset[col_2080].rank(
            ascending=False, method="min"
        )
        subset[f"{proc}_rank_5050"] = subset[col_5050].rank(
            ascending=False, method="min"
        )
        subset[f"{proc}_rank_change"] = np.abs(
            subset[f"{proc}_rank_2080"] - subset[f"{proc}_rank_5050"]
        )

        for rc in [f"{proc}_rank_2080", f"{proc}_rank_5050", f"{proc}_rank_change"]:
            df_comp.loc[mask, rc] = subset[rc].values

    switching_psa = df_comp[
        df_comp["PSA_rank_change"].notna()
        & (df_comp["PSA_rank_change"] >= rank_threshold)
    ]
    switching_vsa = df_comp[
        df_comp["VSA_rank_change"].notna()
        & (df_comp["VSA_rank_change"] >= rank_threshold)
    ]
    print(f"  PSA switching: {len(switching_psa)} MOFs", flush=True)
    print(f"  VSA switching: {len(switching_vsa)} MOFs", flush=True)

    # Step 6: Exp vs Hypo hit-rate analysis
    print(f"\nStep 6: Exp vs Hypo hit-rate analysis", flush=True)
    hit_rate_rows = []
    for proc in ["PSA", "VSA"]:
        for comp_label, col in [
            ("20:80", f"{proc}_API_2080"),
            ("50:50", f"{proc}_API_5050"),
        ]:
            mask = df_comp[col].notna()
            subset = df_comp[mask]
            n_total = len(subset)
            if n_total == 0:
                continue

            median_api = subset[col].median()
            above_median = subset[subset[col] >= median_api]

            n_exp = int(subset["is_exp"].sum())
            n_hypo = n_total - n_exp
            n_exp_above = int(above_median["is_exp"].sum())
            n_hypo_above = len(above_median) - n_exp_above

            hit_rate_exp = n_exp_above / max(n_exp, 1)
            hit_rate_hypo = n_hypo_above / max(n_hypo, 1)

            hit_rate_rows.append({
                "process": proc,
                "composition": comp_label,
                "n_total": n_total,
                "median_API": round(median_api, 4),
                "n_exp": n_exp,
                "hit_rate_exp": round(hit_rate_exp, 4),
                "n_hypo": n_hypo,
                "hit_rate_hypo": round(hit_rate_hypo, 4),
            })
            print(
                f"  {proc} {comp_label}: exp hit = {hit_rate_exp:.1%} "
                f"({n_exp_above}/{n_exp}), "
                f"hypo hit = {hit_rate_hypo:.1%} ({n_hypo_above}/{n_hypo})",
                flush=True,
            )

    df_hit_rate = pd.DataFrame(hit_rate_rows)

    # Step 7: Summary statistics for API distributions
    print(f"\nStep 7: API distribution statistics", flush=True)
    for proc in ["PSA", "VSA"]:
        for comp, sfx in [("20:80", "2080"), ("50:50", "5050")]:
            col = f"{proc}_API_{sfx}"
            vals = df_comp[col].dropna()
            if len(vals) > 0:
                print(
                    f"  {proc} {comp}: median={vals.median():.4f}, "
                    f"mean={vals.mean():.4f}, std={vals.std():.4f}, "
                    f"min={vals.min():.4f}, max={vals.max():.4f}",
                    flush=True,
                )

    # Step 8: Save CSV
    csv_path = output_dir / "composition_sensitivity_results.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}", flush=True)

    if len(df_hit_rate) > 0:
        hr_path = output_dir / "composition_sensitivity_hit_rates.csv"
        df_hit_rate.to_csv(hr_path, index=False)
        print(f"Saved: {hr_path}", flush=True)

    # Step 9: Write summary markdown
    summary_path = output_dir / "composition_sensitivity_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Composition Sensitivity Analysis: 20:80 vs 50:50 CH4:N2\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**MOFs analysed**: {len(df_comp)} "
                f"(exp: {df_comp['is_exp'].sum()}, "
                f"hypo: {(~df_comp['is_exp']).sum()})\n")
        f.write(f"**Rank switching threshold**: {rank_threshold}\n\n")
        f.write("---\n\n")

        # Data sources
        f.write("## Data Sources\n\n")
        f.write("| Source | GCMC 20:80 | GCMC 50:50 | Widom |\n")
        f.write("|--------|-----------|-----------|-------|\n")
        for label, d2080, d5050, dw in DATA_SOURCES:
            n2080 = len(df_2080[df_2080.get("source", pd.Series()) == label]) if "source" in df_2080.columns else "?"
            n5050 = len(df_5050[df_5050.get("source", pd.Series()) == label]) if "source" in df_5050.columns else "?"
            nw = len(df_widom[df_widom.get("source", pd.Series()) == label]) if "source" in df_widom.columns else "?"
            f.write(f"| {label} | {n2080} | {n5050} | {nw} |\n")
        f.write("\n")

        # Spearman correlation
        f.write("## Spearman Rank Correlation (API_2080 vs API_5050)\n\n")
        f.write("| Process | rho | p-value | n |\n")
        f.write("|---------|-----|---------|---|\n")
        for proc, vals in spearman_results.items():
            rho_s = f"{vals['rho']:.4f}" if np.isfinite(vals["rho"]) else "N/A"
            p_s = f"{vals['pval']:.2e}" if np.isfinite(vals["pval"]) else "N/A"
            f.write(f"| {proc} | {rho_s} | {p_s} | {vals['n']} |\n")
        f.write("\n")

        # Rank switching
        f.write(f"## Rank-Switching MOFs (rank change >= {rank_threshold})\n\n")
        for proc, sw_df in [("PSA", switching_psa), ("VSA", switching_vsa)]:
            f.write(f"### {proc} ({len(sw_df)} MOFs)\n\n")
            if len(sw_df) > 0:
                show_cols = [
                    "mof_id", "is_exp",
                    f"{proc}_API_2080", f"{proc}_API_5050",
                    f"{proc}_rank_2080", f"{proc}_rank_5050",
                    f"{proc}_rank_change",
                ]
                show_cols = [c for c in show_cols if c in sw_df.columns]
                top_sw = sw_df.nlargest(
                    min(20, len(sw_df)), f"{proc}_rank_change"
                )
                f.write(top_sw[show_cols].to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("No switching MOFs detected.\n\n")

        # Hit rate
        f.write("## Exp vs Hypo Hit Rate (above-median API)\n\n")
        if len(df_hit_rate) > 0:
            f.write(df_hit_rate.to_markdown(index=False))
        else:
            f.write("No data available.\n")
        f.write("\n\n")

        # Interpretation
        f.write("---\n\n")
        f.write("## Interpretation Guide\n\n")
        f.write("- **High Spearman rho** (> 0.8): rankings are robust "
                "across compositions.\n")
        f.write("- **Low rho** (< 0.5): composition significantly "
                "affects which MOFs appear best.\n")
        f.write("- **Switching MOFs**: investigate whether structural "
                "features explain sensitivity.\n")
        f.write("- **Hit rate**: if hypo >> exp, experimental MOFs may "
                "not cover the optimal design space.\n")

    print(f"Saved: {summary_path}", flush=True)
    print("\nDone.", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare GCMC separation performance at 20:80 vs 50:50 "
                    "CH4:N2 compositions across all data sources."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--rank-threshold",
        type=int,
        default=10,
        help="Minimum rank change to flag as 'switching' (default: 10).",
    )
    args = parser.parse_args()

    run_analysis(
        output_dir=Path(args.output_dir),
        rank_threshold=args.rank_threshold,
    )


if __name__ == "__main__":
    main()
