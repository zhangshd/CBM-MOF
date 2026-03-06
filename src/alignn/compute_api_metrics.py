"""
compute_api_metrics.py — Tasks 2.1 + 2.2

Task 2.1 (Phase A):
    Concatenate 24 batch prediction CSVs, compute PSA/VSA working capacity,
    selectivity, and API scores, then merge UQ flags.
    Output: full_library_with_api.csv  (230,651 rows × ~23 cols)

Task 2.2 (Phase B):
    Apply two-stage pre-screening filter:
      1. UQ filter   : remove flag_high_uq == 1
      2. Uptake floor: remove AdsCH4_1000kPa < 0.01 mmol/g
    Output: full_library_screened.csv  (~209K rows)

Run
---
    # dry-run (first 3 batches only → results/test_run/)
    python src/alignn/compute_api_metrics.py --test

    # full run
    python src/alignn/compute_api_metrics.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = REPO_ROOT / "results" / "alignn" / "full_library_inference" / "batches"
UQ_CSV    = REPO_ROOT / "results" / "alignn" / "full_library_inference" / "full_library_uq.csv"
OUT_DIR   = REPO_ROOT / "results" / "alignn" / "full_library_inference"
TEST_DIR  = REPO_ROOT / "results" / "test_run"

N_BATCHES = 24    # batch_0000 … batch_0023
TEST_BATCHES = 3  # how many batches to load in --test mode

# Gas uptake floor (mmol/g): MOFs below this are treated as non-adsorbers
UPTAKE_FLOOR = 0.01


# ---------------------------------------------------------------------------
# Separation metrics  (inlined from src/experiments/exp07_inference_ml.py)
# ---------------------------------------------------------------------------
def calculate_separation_metrics(
    df: pd.DataFrame,
    y_ch4: float = 0.2,
    y_n2: float = 0.8,
    A: float = 1.0,
    B: float = 1.0,
    C: float = 1.0,
) -> pd.DataFrame:
    """
    Add PSA / VSA working capacity, selectivity, and API columns to *df*.

    PSA: 10 bar adsorption (1000 kPa), 1 bar desorption  (100 kPa)
    VSA:  1 bar adsorption (100 kPa),  0.1 bar desorption  (10 kPa)

    API formula:
        alpha = (q_CH4_ads / q_N2_ads) * (y_N2 / y_CH4)
        API   = ((alpha - 1)^A * WC^B) / |QstCH4|^C
    """
    result_df = df.copy()
    qst_ch4_abs = np.abs(result_df["QstCH4"])

    for process, ads_p, des_p in [("PSA", "1000kPa", "100kPa"), ("VSA", "100kPa", "10kPa")]:
        q_ch4_ads = result_df[f"AdsCH4_{ads_p}"]
        q_n2_ads  = result_df[f"AdsN2_{ads_p}"]

        result_df[f"{process}_WC_CH4"] = result_df[f"AdsCH4_{ads_p}"] - result_df[f"AdsCH4_{des_p}"]
        result_df[f"{process}_WC_N2"]  = result_df[f"AdsN2_{ads_p}"]  - result_df[f"AdsN2_{des_p}"]

        alpha = np.where(
            q_n2_ads > 1e-10,
            (q_ch4_ads / q_n2_ads) * (y_n2 / y_ch4),
            np.nan,
        )
        result_df[f"{process}_alpha_CH4_N2"] = alpha

        valid = (qst_ch4_abs > 1e-10) & (result_df[f"{process}_alpha_CH4_N2"] > 1e-10)
        result_df[f"{process}_API_CH4"] = np.where(
            valid,
            ((result_df[f"{process}_alpha_CH4_N2"] - 1) ** A
             * result_df[f"{process}_WC_CH4"] ** B)
            / (qst_ch4_abs ** C),
            np.nan,
        )

    return result_df


# ---------------------------------------------------------------------------
# Phase A — Task 2.1: load batches + compute API + merge UQ
# ---------------------------------------------------------------------------
def phase_a(n_batches: int, out_dir: Path) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Phase A — Task 2.1: Loading {n_batches} prediction batches")
    print(f"{'='*60}")

    # --- 1. Load and concatenate batch prediction CSVs ---
    dfs = []
    for i in range(n_batches):
        csv_path = BATCH_DIR / f"batch_{i:04d}_predictions.csv"
        if not csv_path.exists():
            print(f"  [WARN] Missing: {csv_path.name} — skipping")
            continue
        df_i = pd.read_csv(csv_path)
        dfs.append(df_i)

    if not dfs:
        raise FileNotFoundError(f"No batch CSVs found in {BATCH_DIR}")

    df_pred = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(dfs)} batches → {len(df_pred):,} rows × {df_pred.shape[1]} cols")

    # --- 2. Compute separation metrics ---
    print("  Computing PSA/VSA working capacity, selectivity, and API ...")
    df_api = calculate_separation_metrics(df_pred)
    n_psa_nan = df_api["PSA_API_CH4"].isna().sum()
    n_vsa_nan = df_api["VSA_API_CH4"].isna().sum()
    print(f"  PSA_API_CH4 NaN: {n_psa_nan:,}  |  VSA_API_CH4 NaN: {n_vsa_nan:,}")

    # --- 3. Merge UQ data ---
    print(f"  Loading UQ CSV: {UQ_CSV.name} ...")
    df_uq = pd.read_csv(UQ_CSV)
    print(f"  UQ rows: {len(df_uq):,}")

    df_merged = df_api.merge(df_uq, on="mof_id", how="left")
    n_missing_uq = df_merged["flag_high_uq"].isna().sum()
    if n_missing_uq > 0:
        print(f"  [WARN] {n_missing_uq:,} MOFs have no UQ data (left join miss)")
    print(f"  Merged DataFrame: {len(df_merged):,} rows × {df_merged.shape[1]} cols")

    # --- 4. Save Task 2.1 output ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_library_with_api.csv"
    df_merged.to_csv(out_path, index=False)
    print(f"\n  [Task 2.1] Saved → {out_path.relative_to(REPO_ROOT)}")
    print(f"  Shape: {df_merged.shape[0]:,} rows × {df_merged.shape[1]} cols")

    # Print API distribution summary
    print("\n  --- API Distribution Summary ---")
    for col in ["PSA_API_CH4", "VSA_API_CH4"]:
        s = df_merged[col].dropna()
        print(f"  {col}: mean={s.mean():.4f}, median={s.median():.4f}, "
              f"p90={s.quantile(0.90):.4f}, max={s.max():.4f}")

    return df_merged


# ---------------------------------------------------------------------------
# Phase B — Task 2.2: UQ pre-screening + gas uptake floor filter
# ---------------------------------------------------------------------------
def phase_b(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Phase B — Task 2.2: UQ pre-screening + uptake floor filter")
    print(f"{'='*60}")

    n_start = len(df)
    print(f"  Starting count: {n_start:,}")

    # --- Filter 1: Remove high-UQ MOFs ---
    mask_low_uq = df["flag_high_uq"] == 0
    n_high_uq = (~mask_low_uq).sum()
    df_f1 = df[mask_low_uq].copy()
    print(f"\n  Filter 1 (UQ flag): removed {n_high_uq:,} high-UQ MOFs "
          f"({n_high_uq/n_start*100:.1f}%)")
    print(f"  Remaining: {len(df_f1):,}")

    # --- Filter 2: Gas uptake floor ---
    mask_uptake = df_f1["AdsCH4_1000kPa"] >= UPTAKE_FLOOR
    n_below_floor = (~mask_uptake).sum()
    df_f2 = df_f1[mask_uptake].copy()
    print(f"\n  Filter 2 (uptake floor < {UPTAKE_FLOOR} mmol/g): "
          f"removed {n_below_floor:,} near-zero adsorbers "
          f"({n_below_floor/n_start*100:.2f}%)")
    print(f"  Remaining: {len(df_f2):,}")

    print(f"\n  Total removed: {n_start - len(df_f2):,} / {n_start:,} "
          f"({(n_start - len(df_f2))/n_start*100:.1f}%)")
    print(f"  Final screened count: {len(df_f2):,}")

    # --- Save Task 2.2 output ---
    out_path = out_dir / "full_library_screened.csv"
    df_f2.to_csv(out_path, index=False)
    print(f"\n  [Task 2.2] Saved → {out_path.relative_to(REPO_ROOT)}")

    # --- Print Top-5 PSA and VSA candidates ---
    for process in ["PSA", "VSA"]:
        api_col = f"{process}_API_CH4"
        top5 = df_f2.nlargest(5, api_col)[["mof_id", api_col]]
        print(f"\n  Top-5 {process}_API_CH4 (screened):")
        print(top5.to_string(index=False))

    return df_f2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute API metrics (Task 2.1) and apply UQ pre-screening (Task 2.2)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Dry-run: load only first {TEST_BATCHES} batches, write to results/test_run/",
    )
    args = parser.parse_args()

    if args.test:
        out_dir = TEST_DIR
        n_batches = TEST_BATCHES
        print(f"[TEST MODE] Loading {n_batches} batches → {out_dir}")
    else:
        out_dir = OUT_DIR
        n_batches = N_BATCHES

    # Phase A: Task 2.1
    df_with_api = phase_a(n_batches=n_batches, out_dir=out_dir)

    # Phase B: Task 2.2
    df_screened = phase_b(df=df_with_api, out_dir=out_dir)

    print(f"\n{'='*60}")
    print("All done.")
    print(f"  Task 2.1 output : {(out_dir / 'full_library_with_api.csv').relative_to(REPO_ROOT)}")
    print(f"  Task 2.2 output : {(out_dir / 'full_library_screened.csv').relative_to(REPO_ROOT)}")
    if args.test:
        print("[TEST MODE] — re-run without --test for full 230,651 MOFs.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
