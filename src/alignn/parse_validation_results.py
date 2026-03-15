"""
parse_validation_results.py
===========================
Task 2.4b: Parse GCMC + Widom validation results for ranked candidates.

Run AFTER GCMC jobs have completed. Parses:
  - RASPA3 parsed results (GCMC adsorption loadings)
  - RASPA2 Widom results (Qst heats of adsorption)

Then:
1. Pivots adsorption data → AdsCH4_10kPa, AdsCH4_100kPa, AdsCH4_1000kPa,
   AdsN2_10kPa, AdsN2_100kPa, AdsN2_1000kPa [mmol/g]
2. Extracts QstCH4, QstN2 [kJ/mol] from Widom
3. Merges with top_union.csv
4. Recalculates PSA/VSA API using calculate_separation_metrics()
5. Computes R² and MAE vs ML predictions (8 properties)
6. Saves gcmc_vs_ml_comparison.csv and gcmc_validation_summary.md

Usage:
    python src/alignn/parse_validation_results.py
    python src/alignn/parse_validation_results.py --test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[2]
TOP_CAND_DIR = REPO_ROOT / "results" / "alignn" / "top_candidates"
GCMC_BASE    = REPO_ROOT / "results" / "alignn" / "gcmc_top_candidates"
GCMC_DIR     = GCMC_BASE / "gcmc_DreidingTraPPEJson"
WIDOM_DIR    = GCMC_BASE / "widom_DREIDING"
UNION_CSV    = TOP_CAND_DIR / "top_union.csv"
OUTPUT_DIR   = GCMC_BASE
SUMMARY_DIR  = REPO_ROOT / "results" / "summary"


# ---------------------------------------------------------------------------
# Separation metrics
# ---------------------------------------------------------------------------
from src.alignn.screening.metrics import calculate_separation_metrics


def calculate_validation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute PSA/VSA metrics on GCMC columns using the canonical formula."""
    result_df = df.copy()
    gcmc_input = pd.DataFrame(
        {
            "AdsCH4_10kPa": df["gcmc_AdsCH4_10kPa"],
            "AdsCH4_100kPa": df["gcmc_AdsCH4_100kPa"],
            "AdsCH4_1000kPa": df["gcmc_AdsCH4_1000kPa"],
            "AdsN2_10kPa": df["gcmc_AdsN2_10kPa"],
            "AdsN2_100kPa": df["gcmc_AdsN2_100kPa"],
            "AdsN2_1000kPa": df["gcmc_AdsN2_1000kPa"],
            "QstCH4": df["QstCH4_gcmc"],
        }
    )
    calculated = calculate_separation_metrics(gcmc_input)
    rename_back = {
        "PSA_WC_CH4": "gcmc_PSA_WC_CH4",
        "PSA_WC_N2": "gcmc_PSA_WC_N2",
        "PSA_alpha_CH4_N2": "gcmc_PSA_alpha_CH4_N2",
        "PSA_API_CH4": "gcmc_PSA_API_CH4",
        "VSA_WC_CH4": "gcmc_VSA_WC_CH4",
        "VSA_WC_N2": "gcmc_VSA_WC_N2",
        "VSA_alpha_CH4_N2": "gcmc_VSA_alpha_CH4_N2",
        "VSA_API_CH4": "gcmc_VSA_API_CH4",
    }
    calculated = calculated.rename(columns=rename_back)
    for src_col, dst_col in rename_back.items():
        result_df[dst_col] = calculated[dst_col]
    return result_df


# ---------------------------------------------------------------------------
# Parse GCMC results
# ---------------------------------------------------------------------------

def parse_gcmc(gcmc_dir: Path) -> pd.DataFrame:
    """
    Load all raspa3_parsed_results_*.csv from gcmc_dir, pivot to wide format.

    Returns DataFrame with columns:
      mof_id, AdsCH4_10kPa, AdsCH4_100kPa, AdsCH4_1000kPa,
              AdsN2_10kPa,  AdsN2_100kPa,  AdsN2_1000kPa
    Units: mmol/g (mol/kg → mmol/g × 1.0 conversion: 1 mol/kg = 1 mmol/g)
    """
    # Search in both gcmc_dir root and batch_* subdirectories
    csv_files = sorted(gcmc_dir.glob("raspa3_parsed_results*.csv"))
    csv_files += sorted(gcmc_dir.glob("batch_*/raspa3_parsed_results*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raspa3_parsed_results*.csv found in {gcmc_dir} or its batch_* subdirs")

    dfs = [pd.read_csv(f) for f in csv_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"  GCMC raw: {len(df_raw):,} rows from {len(csv_files)} files", flush=True)

    # Filter to 298 K only and CH4/N2 mixture simulations
    df_raw = df_raw[df_raw["Temperature[K]"].round(1) == 298.0].copy()

    # Map pressure (bar) → label
    pressure_map = {0.1: "10kPa", 1.0: "100kPa", 10.0: "1000kPa"}
    df_raw["pressure_label"] = df_raw["Pressure[bar]"].round(2).map(pressure_map)
    df_raw = df_raw.dropna(subset=["pressure_label"])

    # Convert mol/kg → mmol/g (same unit in 1:1 ratio for mol/kg → mmol/g)
    # 1 mol/kg = 1 mmol/g
    df_raw["loading_mmol_g"] = df_raw["AbsLoading"]  # mol/kg == mmol/g

    # Pivot: one row per MOF
    records = {}
    for _, row in df_raw.iterrows():
        mof = row["MofName"]
        gas = row["GasName"]
        plabel = row["pressure_label"]
        col = f"gcmc_Ads{gas.replace('methane','CH4').replace('N2','N2')}_{plabel}"
        if mof not in records:
            records[mof] = {}
        records[mof][col] = row["loading_mmol_g"]

    df_gcmc = pd.DataFrame.from_dict(records, orient="index").reset_index()
    df_gcmc.rename(columns={"index": "mof_id"}, inplace=True)

    # Rename methane columns
    rename_cols = {}
    for col in df_gcmc.columns:
        if "gcmc_Adsmethane_" in col:
            rename_cols[col] = col.replace("gcmc_Adsmethane_", "gcmc_AdsCH4_")
    df_gcmc.rename(columns=rename_cols, inplace=True)

    print(f"  GCMC pivoted: {len(df_gcmc):,} unique MOFs", flush=True)
    return df_gcmc


# ---------------------------------------------------------------------------
# Parse Widom results
# ---------------------------------------------------------------------------

def parse_widom(widom_dir: Path) -> pd.DataFrame:
    """
    Load all widom_results_*.csv from widom_dir.

    Returns DataFrame with columns: mof_id, QstCH4_gcmc, QstN2_gcmc [kJ/mol]
    AdsorptionHeat is already in kJ/mol in the Widom output.
    """
    # Search in both widom_dir root and batch_* subdirectories
    csv_files = sorted(widom_dir.glob("widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("batch_*/widom_results*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No widom_results*.csv found in {widom_dir} or its batch_* subdirs")

    dfs = [pd.read_csv(f) for f in csv_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"  Widom raw: {len(df_raw):,} rows from {len(csv_files)} files", flush=True)

    df_raw = df_raw[df_raw["Temperature[K]"].round(1) == 298.0].copy()

    # Pivot: CH4 and N2 Qst per MOF
    df_ch4 = (df_raw[df_raw["GasName"] == "methane"]
              .groupby("MofName")["AdsorptionHeat"]
              .mean()
              .rename("QstCH4_gcmc")
              .reset_index()
              .rename(columns={"MofName": "mof_id"}))
    df_n2 = (df_raw[df_raw["GasName"] == "N2"]
             .groupby("MofName")["AdsorptionHeat"]
             .mean()
             .rename("QstN2_gcmc")
             .reset_index()
             .rename(columns={"MofName": "mof_id"}))

    df_widom = df_ch4.merge(df_n2, on="mof_id", how="outer")
    print(f"  Widom pivoted: {len(df_widom):,} unique MOFs", flush=True)
    return df_widom


# ---------------------------------------------------------------------------
# Compute R² and MAE (ML vs GCMC)
# ---------------------------------------------------------------------------

def compute_metrics(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compare ML predictions vs GCMC measurements for 8 target properties.
    Returns DataFrame with columns: property, ml_col, gcmc_col, R2, MAE, n
    """
    pairs = [
        ("AdsCH4_10kPa",  "gcmc_AdsCH4_10kPa"),
        ("AdsCH4_100kPa", "gcmc_AdsCH4_100kPa"),
        ("AdsCH4_1000kPa","gcmc_AdsCH4_1000kPa"),
        ("AdsN2_10kPa",   "gcmc_AdsN2_10kPa"),
        ("AdsN2_100kPa",  "gcmc_AdsN2_100kPa"),
        ("AdsN2_1000kPa", "gcmc_AdsN2_1000kPa"),
        ("QstCH4",        "QstCH4_gcmc"),
        ("QstN2",         "QstN2_gcmc"),
    ]
    rows = []
    for ml_col, gcmc_col in pairs:
        if ml_col not in df_merged.columns or gcmc_col not in df_merged.columns:
            continue
        mask = df_merged[ml_col].notna() & df_merged[gcmc_col].notna()
        n = mask.sum()
        if n < 3:
            rows.append({"property": ml_col, "ml_col": ml_col,
                         "gcmc_col": gcmc_col, "R2": np.nan, "MAE": np.nan, "n": n})
            continue
        y_true = df_merged.loc[mask, gcmc_col].values
        y_pred = df_merged.loc[mask, ml_col].values
        rows.append({
            "property": ml_col,
            "ml_col": ml_col,
            "gcmc_col": gcmc_col,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "n": n,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_mode: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Check GCMC output existence
    # ------------------------------------------------------------------
    if test_mode:
        print("[TEST MODE] Checking GCMC directory structure …", flush=True)
        print(f"  GCMC dir  : {GCMC_DIR}  exists={GCMC_DIR.exists()}")
        print(f"  Widom dir : {WIDOM_DIR}  exists={WIDOM_DIR.exists()}")
        gcmc_csvs = list(GCMC_DIR.glob("raspa3_parsed_results*.csv")) if GCMC_DIR.exists() else []
        widom_csvs = list(WIDOM_DIR.glob("widom_results*.csv")) if WIDOM_DIR.exists() else []
        print(f"  GCMC CSVs : {len(gcmc_csvs)} found")
        print(f"  Widom CSVs: {len(widom_csvs)} found")
        if not gcmc_csvs:
            print("  [INFO] GCMC not completed yet. Run this script after jobs finish.")
        return

    # ------------------------------------------------------------------
    # Step 1 — Load union candidates
    # ------------------------------------------------------------------
    print("Step 1: Loading top_union.csv …", flush=True)
    df_union = pd.read_csv(UNION_CSV)
    n_union = len(df_union)
    print(f"  Union candidates: {n_union}", flush=True)

    # ------------------------------------------------------------------
    # Step 2 — Parse GCMC adsorption data
    # ------------------------------------------------------------------
    print("Step 2: Parsing GCMC results …", flush=True)
    df_gcmc = parse_gcmc(GCMC_DIR)

    # ------------------------------------------------------------------
    # Step 3 — Parse Widom Qst data
    # ------------------------------------------------------------------
    print("Step 3: Parsing Widom results …", flush=True)
    df_widom = parse_widom(WIDOM_DIR)

    # ------------------------------------------------------------------
    # Step 4 — Merge all data
    # ------------------------------------------------------------------
    print("Step 4: Merging datasets …", flush=True)
    df_merged = df_union.merge(df_gcmc, on="mof_id", how="left")
    df_merged = df_merged.merge(df_widom, on="mof_id", how="left")

    n_gcmc_found = df_merged["gcmc_AdsCH4_100kPa"].notna().sum()
    n_missing_gcmc = n_union - n_gcmc_found
    failure_rate = 100 * n_missing_gcmc / max(n_union, 1)
    print(f"  GCMC coverage: {n_gcmc_found}/{n_union} "
          f"(failure rate: {failure_rate:.1f}%)", flush=True)

    if failure_rate >= 5.0:
        print(f"  [WARN] Failure rate {failure_rate:.1f}% >= 5% threshold!", flush=True)

    # ------------------------------------------------------------------
    # Step 5 — Recalculate GCMC-based PSA/VSA API
    # ------------------------------------------------------------------
    print("Step 5: Recalculating GCMC-based API metrics …", flush=True)
    df_merged = calculate_validation_metrics(df_merged)

    # ------------------------------------------------------------------
    # Step 6 — ML vs GCMC comparison metrics
    # ------------------------------------------------------------------
    print("Step 6: Computing ML vs GCMC R²/MAE …", flush=True)
    metrics_df = compute_metrics(df_merged)
    print(metrics_df[["property", "R2", "MAE", "n"]].to_string(index=False))

    mean_r2 = metrics_df["R2"].mean()
    print(f"\n  Mean R² across 8 properties: {mean_r2:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 7 — Save outputs
    # ------------------------------------------------------------------
    comparison_out = OUTPUT_DIR / "gcmc_vs_ml_comparison.csv"
    metrics_out    = OUTPUT_DIR / "gcmc_ml_metrics.csv"

    df_merged.to_csv(comparison_out, index=False)
    metrics_df.to_csv(metrics_out, index=False)
    print(f"\nSaved:", flush=True)
    print(f"  Comparison CSV : {comparison_out}", flush=True)
    print(f"  Metrics CSV    : {metrics_out}", flush=True)

    # ------------------------------------------------------------------
    # Step 8 — Write validation summary
    # ------------------------------------------------------------------
    summary_path = SUMMARY_DIR / "gcmc_validation_summary.md"
    top10_psa = (df_merged.dropna(subset=["gcmc_PSA_API_CH4"])
                 .nlargest(10, "gcmc_PSA_API_CH4")
                 [["mof_id", "gcmc_PSA_API_CH4", "PSA_API_CH4", "psa_rank", "vsa_rank"]]
                 .reset_index(drop=True))

    with open(summary_path, "w") as f:
        f.write("# GCMC Validation Summary — Task 2.4b\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Input**: {n_union} Top candidates (PSA ∪ VSA union)\n")
        f.write(f"**GCMC coverage**: {n_gcmc_found}/{n_union} ({100-failure_rate:.1f}%)\n\n")
        f.write("---\n\n")

        f.write("## ML vs GCMC Comparison (R² / MAE)\n\n")
        f.write("| Property | R² | MAE | n |\n")
        f.write("|----------|-----|-----|---|\n")
        for _, row in metrics_df.iterrows():
            r2_str  = f"{row['R2']:.4f}"  if not pd.isna(row['R2'])  else "N/A"
            mae_str = f"{row['MAE']:.4f}" if not pd.isna(row['MAE']) else "N/A"
            f.write(f"| {row['property']} | {r2_str} | {mae_str} | {int(row['n'])} |\n")
        f.write(f"\n**Mean R²**: {mean_r2:.4f}\n\n")

        f.write("---\n\n")
        f.write("## Top-10 GCMC-Based PSA Candidates\n\n")
        f.write("| Rank | mof_id | GCMC PSA_API | ML PSA_API | PSA Rank | VSA Rank |\n")
        f.write("|------|--------|-------------|------------|----------|----------|\n")
        for i, (_, row) in enumerate(top10_psa.iterrows(), 1):
            ml_api = f"{row['PSA_API_CH4']:.4f}" if not pd.isna(row.get("PSA_API_CH4", np.nan)) else "N/A"
            psa_r  = f"{int(row['psa_rank'])}" if not pd.isna(row.get("psa_rank", np.nan)) else "—"
            vsa_r  = f"{int(row['vsa_rank'])}" if not pd.isna(row.get("vsa_rank", np.nan)) else "—"
            f.write(f"| {i} | {row['mof_id']} | {row['gcmc_PSA_API_CH4']:.4f} "
                    f"| {ml_api} | {psa_r} | {vsa_r} |\n")

        f.write("\n---\n\n")
        f.write("## Next Step\n\n")
        f.write("**Task 2.5**: Select final Top-10 from GCMC-ranked candidates for Phase 3 BKT simulation.\n")

    print(f"  Validation summary : {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2.4b: Parse GCMC results")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: check file structure only (no parsing)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model-specific results dir (e.g. results/alignn/model_ep220). "
                             "Overrides TOP_CAND_DIR, GCMC_BASE and derived paths.")
    args = parser.parse_args()

    if args.model_dir:
        _md = Path(args.model_dir)
        if not _md.is_absolute():
            _md = REPO_ROOT / _md
        TOP_CAND_DIR = _md / "top_candidates"
        GCMC_BASE    = _md / "gcmc_top_candidates"
        GCMC_DIR     = GCMC_BASE / "gcmc_DreidingTraPPEJson"
        WIDOM_DIR    = GCMC_BASE / "widom_DREIDING"
        UNION_CSV    = TOP_CAND_DIR / "top_union.csv"
        OUTPUT_DIR   = GCMC_BASE

    main(test_mode=args.test)
