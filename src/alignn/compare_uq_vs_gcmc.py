"""
compare_uq_vs_gcmc.py
======================
Post-GCMC analysis: Compare ML prediction accuracy for high-UQ vs low-UQ MOFs.

Loads the 186 MOFs from all_top_union.csv (with flag_high_uq), then merges
GCMC ground truth from multiple directories (exp_top + hypo_top + old_194)
to maximize coverage. Reports per-group R^2, MAPE, MAE for all 8 targets.

Usage:
    conda run -n mofmthnn --no-banner python src/alignn/compare_uq_vs_gcmc.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path for imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignn.parse_validation_results import parse_gcmc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "composition_sensitivity"

# 8 target properties: 6 adsorption loadings + 2 heats of adsorption
TARGET_COLS = [
    "AdsCH4_10kPa",
    "AdsCH4_100kPa",
    "AdsCH4_1000kPa",
    "AdsN2_10kPa",
    "AdsN2_100kPa",
    "AdsN2_1000kPa",
    "QstCH4",
    "QstN2",
]

# Corresponding GCMC column names
GCMC_COL_MAP = {
    "AdsCH4_10kPa": "gcmc_AdsCH4_10kPa",
    "AdsCH4_100kPa": "gcmc_AdsCH4_100kPa",
    "AdsCH4_1000kPa": "gcmc_AdsCH4_1000kPa",
    "AdsN2_10kPa": "gcmc_AdsN2_10kPa",
    "AdsN2_100kPa": "gcmc_AdsN2_100kPa",
    "AdsN2_1000kPa": "gcmc_AdsN2_1000kPa",
    "QstCH4": "QstCH4_gcmc",
    "QstN2": "QstN2_gcmc",
}

# GCMC directories (20:80 composition) — order matters for dedup (first wins)
GCMC_DIRS = [
    ("exp_top", MODEL_DIR / "gcmc_exp_top" / "gcmc_DreidingTraPPEJson"),
    ("hypo_top", MODEL_DIR / "gcmc_hypo_top" / "gcmc_DreidingTraPPEJson"),
    ("old_194", MODEL_DIR / "gcmc_top_candidates" / "gcmc_DreidingTraPPEJson"),
]

# Widom directories
WIDOM_DIRS = [
    ("exp_top", MODEL_DIR / "gcmc_exp_top" / "widom_DREIDING"),
    ("hypo_top", MODEL_DIR / "gcmc_hypo_top" / "widom_DREIDING"),
    ("old_194", MODEL_DIR / "gcmc_top_candidates" / "widom_DREIDING"),
]


# ---------------------------------------------------------------------------
# Widom loader (handles both naming conventions)
# ---------------------------------------------------------------------------

def load_widom(widom_dir: Path) -> pd.DataFrame:
    """Load Widom CSV from widom_dir, handling both old and new naming."""
    csv_files = sorted(widom_dir.glob("widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("raspa2_widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("batch_*/widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("batch_*/raspa2_widom_results*.csv"))
    csv_files = sorted(set(csv_files))
    if not csv_files:
        raise FileNotFoundError(
            f"No widom CSV found in {widom_dir} or batch_* subdirs"
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
    return df_ch4.merge(df_n2, on="mof_id", how="outer")


def load_all_gcmc() -> pd.DataFrame:
    """Load and concatenate GCMC 20:80 from all available directories."""
    frames = []
    for label, gcmc_dir in GCMC_DIRS:
        if gcmc_dir.exists():
            print(f"  Parsing GCMC from {label} ...", flush=True)
            df = parse_gcmc(gcmc_dir)
            frames.append(df)
        else:
            print(f"  [SKIP] {label}: {gcmc_dir}", flush=True)
    if not frames:
        raise FileNotFoundError("No GCMC directories found")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="mof_id", keep="first")
    print(f"  GCMC total (dedup): {len(combined)} MOFs", flush=True)
    return combined


def load_all_widom() -> pd.DataFrame:
    """Load and concatenate Widom from all available directories."""
    frames = []
    for label, widom_dir in WIDOM_DIRS:
        if widom_dir.exists():
            print(f"  Parsing Widom from {label} ...", flush=True)
            df = load_widom(widom_dir)
            frames.append(df)
        else:
            print(f"  [SKIP] {label}: {widom_dir}", flush=True)
    if not frames:
        raise FileNotFoundError("No Widom directories found")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="mof_id", keep="first")
    print(f"  Widom total (dedup): {len(combined)} MOFs", flush=True)
    return combined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, skipping near-zero true values."""
    mask = np.abs(y_true) > 1e-10
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_group_metrics(df: pd.DataFrame, group_name: str) -> list[dict]:
    """Compute R2, MAPE, MAE for each target within a group."""
    rows = []
    for ml_col in TARGET_COLS:
        gcmc_col = GCMC_COL_MAP[ml_col]
        if ml_col not in df.columns or gcmc_col not in df.columns:
            rows.append({
                "group": group_name, "property": ml_col,
                "R2": np.nan, "MAPE_%": np.nan, "MAE": np.nan, "n": 0,
            })
            continue

        mask = df[ml_col].notna() & df[gcmc_col].notna()
        n = mask.sum()
        if n < 3:
            rows.append({
                "group": group_name, "property": ml_col,
                "R2": np.nan, "MAPE_%": np.nan, "MAE": np.nan, "n": n,
            })
            continue

        y_true = df.loc[mask, gcmc_col].values.astype(float)
        y_pred = df.loc[mask, ml_col].values.astype(float)

        rows.append({
            "group": group_name,
            "property": ml_col,
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": mape(y_true, y_pred),
            "MAE": float(np.mean(np.abs(y_true - y_pred))),
            "n": n,
        })
    return rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    union_csv = MODEL_DIR / "top_candidates" / "all_top_union.csv"

    # Step 1: Load ML predictions + UQ flags
    print("Step 1: Loading all_top_union.csv ...", flush=True)
    df_union = pd.read_csv(union_csv)
    n_total = len(df_union)

    if "flag_high_uq" not in df_union.columns:
        print("[ERROR] Column 'flag_high_uq' not found.", flush=True)
        print("  Available:", list(df_union.columns), flush=True)
        sys.exit(1)

    df_union["flag_high_uq"] = df_union["flag_high_uq"].astype(bool)
    n_high = int(df_union["flag_high_uq"].sum())
    n_low = n_total - n_high
    print(f"  Total: {n_total}, High-UQ: {n_high}, Low-UQ: {n_low}", flush=True)

    # Step 2: Load GCMC ground truth from all sources
    print("\nStep 2: Loading GCMC (20:80) from all sources ...", flush=True)
    df_gcmc = load_all_gcmc()

    print("\nStep 3: Loading Widom from all sources ...", flush=True)
    df_widom = load_all_widom()

    # Step 4: Merge
    print("\nStep 4: Merging ML predictions with GCMC ground truth ...", flush=True)
    df_merged = df_union.merge(df_gcmc, on="mof_id", how="left")
    df_merged = df_merged.merge(df_widom, on="mof_id", how="left")

    n_with_gcmc = df_merged["gcmc_AdsCH4_100kPa"].notna().sum()
    n_with_widom = df_merged["QstCH4_gcmc"].notna().sum()
    print(f"  With GCMC data: {n_with_gcmc}/{n_total}", flush=True)
    print(f"  With Widom data: {n_with_widom}/{n_total}", flush=True)

    # Step 5: Split and compute metrics
    print("\nStep 5: Computing per-group metrics ...", flush=True)
    df_low_uq = df_merged[~df_merged["flag_high_uq"]]
    df_high_uq = df_merged[df_merged["flag_high_uq"]]

    all_rows = []
    all_rows.extend(compute_group_metrics(df_low_uq, "low_UQ"))
    all_rows.extend(compute_group_metrics(df_high_uq, "high_UQ"))
    all_rows.extend(compute_group_metrics(df_merged, "all"))

    df_metrics = pd.DataFrame(all_rows)

    # Step 6: Print comparison table
    print("\n" + "=" * 80, flush=True)
    print("UQ vs GCMC Accuracy Comparison", flush=True)
    print("=" * 80, flush=True)

    for prop in TARGET_COLS:
        prop_rows = df_metrics[df_metrics["property"] == prop]
        print(f"\n  {prop}:", flush=True)
        for _, row in prop_rows.iterrows():
            r2_s = f"{row['R2']:.4f}" if np.isfinite(row["R2"]) else "N/A"
            mape_s = f"{row['MAPE_%']:.1f}%" if np.isfinite(row["MAPE_%"]) else "N/A"
            mae_s = f"{row['MAE']:.4f}" if np.isfinite(row["MAE"]) else "N/A"
            print(f"    {row['group']:>8s}: R2={r2_s}  MAPE={mape_s}  "
                  f"MAE={mae_s}  n={int(row['n'])}", flush=True)

    # Summary
    print("\n" + "-" * 80, flush=True)
    print("Summary (mean across 8 properties):", flush=True)
    for group in ["low_UQ", "high_UQ", "all"]:
        grp = df_metrics[df_metrics["group"] == group]
        mr2 = grp["R2"].mean()
        mmape = grp["MAPE_%"].mean()
        mmae = grp["MAE"].mean()
        r2_s = f"{mr2:.4f}" if np.isfinite(mr2) else "N/A"
        mape_s = f"{mmape:.1f}%" if np.isfinite(mmape) else "N/A"
        mae_s = f"{mmae:.4f}" if np.isfinite(mmae) else "N/A"
        print(f"  {group:>8s}: mean R2={r2_s}  mean MAPE={mape_s}  "
              f"mean MAE={mae_s}", flush=True)

    low_r2 = df_metrics[df_metrics["group"] == "low_UQ"]["R2"].dropna()
    high_r2 = df_metrics[df_metrics["group"] == "high_UQ"]["R2"].dropna()
    if len(low_r2) > 0 and len(high_r2) > 0:
        delta = low_r2.mean() - high_r2.mean()
        print(f"\n  Delta(low_UQ - high_UQ) mean R2 = {delta:+.4f}", flush=True)
        if delta > 0.05:
            print("  >> High-UQ MOFs show SIGNIFICANTLY worse predictions.", flush=True)
        elif delta > 0.01:
            print("  >> High-UQ MOFs show moderately worse predictions.", flush=True)
        else:
            print("  >> High-UQ and low-UQ MOFs have similar prediction accuracy.",
                  flush=True)

    # Step 7: Save outputs
    metrics_csv = output_dir / "uq_vs_gcmc_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\nSaved: {metrics_csv}", flush=True)

    detail_csv = output_dir / "uq_vs_gcmc_detail.csv"
    detail_cols = ["mof_id", "flag_high_uq", "is_exp", "lsv_norm_composite"]
    for ml_col in TARGET_COLS:
        gcmc_col = GCMC_COL_MAP[ml_col]
        if ml_col in df_merged.columns:
            detail_cols.append(ml_col)
        if gcmc_col in df_merged.columns:
            detail_cols.append(gcmc_col)
    detail_cols = [c for c in detail_cols if c in df_merged.columns]
    df_merged[detail_cols].to_csv(detail_csv, index=False)
    print(f"Saved: {detail_csv}", flush=True)

    # Step 8: Write summary markdown
    summary_path = output_dir / "uq_vs_gcmc_summary.md"
    with open(summary_path, "w") as f:
        f.write("# UQ vs GCMC Prediction Accuracy\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Total MOFs**: {n_total} "
                f"(low-UQ: {n_low}, high-UQ: {n_high})\n")
        f.write(f"**GCMC coverage**: {n_with_gcmc}/{n_total}\n")
        f.write(f"**Widom coverage**: {n_with_widom}/{n_total}\n\n")
        f.write("---\n\n")

        f.write("## Per-Property Metrics\n\n")
        f.write("| Property | Group | R2 | MAPE (%) | MAE | n |\n")
        f.write("|----------|-------|----|----------|-----|---|\n")
        for _, row in df_metrics.iterrows():
            r2_s = f"{row['R2']:.4f}" if np.isfinite(row["R2"]) else "N/A"
            mape_s = f"{row['MAPE_%']:.1f}" if np.isfinite(row["MAPE_%"]) else "N/A"
            mae_s = f"{row['MAE']:.4f}" if np.isfinite(row["MAE"]) else "N/A"
            f.write(f"| {row['property']} | {row['group']} | {r2_s} "
                    f"| {mape_s} | {mae_s} | {int(row['n'])} |\n")

        f.write("\n## Summary\n\n")
        f.write("| Group | Mean R2 | Mean MAPE (%) | Mean MAE |\n")
        f.write("|-------|---------|---------------|----------|\n")
        for group in ["low_UQ", "high_UQ", "all"]:
            grp = df_metrics[df_metrics["group"] == group]
            mr2 = grp["R2"].mean()
            mmape = grp["MAPE_%"].mean()
            mmae = grp["MAE"].mean()
            f.write(f"| {group} | {mr2:.4f} | {mmape:.1f} | {mmae:.4f} |\n")

        if len(low_r2) > 0 and len(high_r2) > 0:
            delta = low_r2.mean() - high_r2.mean()
            f.write(f"\n**Delta (low_UQ - high_UQ) mean R2**: {delta:+.4f}\n\n")

        f.write("\n---\n\n")
        f.write("## Key Question\n\n")
        f.write("Does high UQ (OOD flag) correlate with worse GCMC prediction "
                "accuracy?\n\n")
        f.write("- If yes: UQ filtering is justified and protects against "
                "unreliable candidates.\n")
        f.write("- If no: high-UQ MOFs may be novel-but-predictable, and "
                "aggressive UQ filtering could miss valuable candidates "
                "(see ATC-Cu case: high UQ but accurate predictions).\n")

    print(f"Saved: {summary_path}", flush=True)
    print("\nDone.", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare ML prediction accuracy for high-UQ vs low-UQ MOFs "
                    "against GCMC ground truth."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write output files.",
    )
    args = parser.parse_args()

    run_analysis(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
