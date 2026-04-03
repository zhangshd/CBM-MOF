"""
run_new_top10_pipeline.py — Parse validation GCMC, select Top-10, prepare for BKT.

Handles TWO separate candidate groups (exp + hypo) that each have their own
GCMC/Widom results dirs and ML prediction CSVs.

Steps:
  1. Parse GCMC + Widom for exp (86 MOFs) and hypo (100 MOFs)
  2. Compute GCMC-based API and identify benchmark-beating candidates
  3. Cluster-aware Top-10 selection per process (PSA / VSA)
  4. Save combined Top-20 and symlink CIFs for pure-component GCMC

Usage:
    conda run -n mofmthnn python src/alignn/run_new_top10_pipeline.py
    conda run -n mofmthnn python src/alignn/run_new_top10_pipeline.py --test
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------------------------
# Repo root & sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alignn.screening.metrics import calculate_separation_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"

GROUPS = {
    "exp": {
        "union_csv": MODEL_DIR / "top_candidates" / "exp_union.csv",
        "gcmc_dir": MODEL_DIR / "gcmc_exp_top" / "gcmc_DreidingTraPPEJson",
        "widom_dir": MODEL_DIR / "gcmc_exp_top" / "widom_DREIDING",
    },
    "hypo": {
        "union_csv": MODEL_DIR / "top_candidates" / "hypo_union.csv",
        "gcmc_dir": MODEL_DIR / "gcmc_hypo_top" / "gcmc_DreidingTraPPEJson",
        "widom_dir": MODEL_DIR / "gcmc_hypo_top" / "widom_DREIDING",
    },
}

CLUSTER_CSV = (
    REPO_ROOT / "results" / "cbm_screening" / "inference"
    / "umap_coordinates_descriptor_with_metrics_ml.csv"
)

# ATC-Cu benchmark: extracted from validation set (Round-1 GCMC data)
BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"

OUTPUT_DIR = MODEL_DIR / "process_candidates"
CIF_SRC_DIR = MODEL_DIR / "top_candidates" / "cifs_all_top"

# ---------------------------------------------------------------------------
# Step 1: Parse GCMC + Widom
# ---------------------------------------------------------------------------

def parse_gcmc(gcmc_dir: Path) -> pd.DataFrame:
    """Load raspa3_parsed_results*.csv, pivot to wide per-MOF format."""
    csv_files = sorted(gcmc_dir.glob("raspa3_parsed_results*.csv"))
    csv_files += sorted(gcmc_dir.glob("batch_*/raspa3_parsed_results*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raspa3_parsed_results*.csv in {gcmc_dir}")
    dfs = [pd.read_csv(f) for f in csv_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"  GCMC raw: {len(df_raw):,} rows from {len(csv_files)} files")

    df_raw = df_raw[df_raw["Temperature[K]"].round(1) == 298.0].copy()
    pressure_map = {0.1: "10kPa", 1.0: "100kPa", 10.0: "1000kPa"}
    df_raw["pressure_label"] = df_raw["Pressure[bar]"].round(2).map(pressure_map)
    df_raw = df_raw.dropna(subset=["pressure_label"])

    records = {}
    for _, row in df_raw.iterrows():
        mof = row["MofName"]
        gas = row["GasName"].replace("methane", "CH4")
        col = f"gcmc_Ads{gas}_{row['pressure_label']}"
        if mof not in records:
            records[mof] = {}
        records[mof][col] = row["AbsLoading"]  # mol/kg == mmol/g

    df_gcmc = pd.DataFrame.from_dict(records, orient="index").reset_index()
    df_gcmc.rename(columns={"index": "mof_id"}, inplace=True)
    print(f"  GCMC pivoted: {len(df_gcmc):,} unique MOFs")
    return df_gcmc


def parse_widom(widom_dir: Path) -> pd.DataFrame:
    """Load widom_results*.csv or raspa2_widom_results*.csv, pivot to Qst per MOF."""
    # Try both naming conventions
    csv_files = sorted(widom_dir.glob("widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("raspa2_widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("batch_*/widom_results*.csv"))
    csv_files += sorted(widom_dir.glob("batch_*/raspa2_widom_results*.csv"))
    # Deduplicate (in case both symlink and original exist)
    csv_files = list(dict.fromkeys(csv_files))
    if not csv_files:
        raise FileNotFoundError(f"No widom CSV in {widom_dir}")
    dfs = [pd.read_csv(f) for f in csv_files]
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"  Widom raw: {len(df_raw):,} rows from {len(csv_files)} files")

    df_raw = df_raw[df_raw["Temperature[K]"].round(1) == 298.0].copy()
    df_ch4 = (df_raw[df_raw["GasName"] == "methane"]
              .groupby("MofName")["AdsorptionHeat"].mean()
              .rename("QstCH4_gcmc").reset_index()
              .rename(columns={"MofName": "mof_id"}))
    df_n2 = (df_raw[df_raw["GasName"] == "N2"]
             .groupby("MofName")["AdsorptionHeat"].mean()
             .rename("QstN2_gcmc").reset_index()
             .rename(columns={"MofName": "mof_id"}))
    df_widom = df_ch4.merge(df_n2, on="mof_id", how="outer")
    print(f"  Widom pivoted: {len(df_widom):,} unique MOFs")
    return df_widom


# ---------------------------------------------------------------------------
# Step 1b: Extract ATC-Cu benchmark from validation set
# ---------------------------------------------------------------------------

def extract_benchmark_api(df: pd.DataFrame) -> tuple[float, float]:
    """Extract ATC-Cu PSA/VSA API from the GCMC-validated dataset."""
    row = df.loc[df["mof_id"] == BENCHMARK_MOF]
    if row.empty:
        raise ValueError(f"Benchmark MOF {BENCHMARK_MOF} not found in validation set.")
    return float(row.iloc[0]["gcmc_PSA_API_CH4"]), float(row.iloc[0]["gcmc_VSA_API_CH4"])


# ---------------------------------------------------------------------------
# Step 2: Compute GCMC-based API
# ---------------------------------------------------------------------------

def compute_gcmc_api(df: pd.DataFrame) -> pd.DataFrame:
    """Compute GCMC-based PSA/VSA API using calculate_separation_metrics."""
    gcmc_input = pd.DataFrame({
        "AdsCH4_10kPa": df["gcmc_AdsCH4_10kPa"],
        "AdsCH4_100kPa": df["gcmc_AdsCH4_100kPa"],
        "AdsCH4_1000kPa": df["gcmc_AdsCH4_1000kPa"],
        "AdsN2_10kPa": df["gcmc_AdsN2_10kPa"],
        "AdsN2_100kPa": df["gcmc_AdsN2_100kPa"],
        "AdsN2_1000kPa": df["gcmc_AdsN2_1000kPa"],
        "QstCH4": df["QstCH4_gcmc"],
    })
    calculated = calculate_separation_metrics(gcmc_input)
    rename_map = {
        "PSA_WC_CH4": "gcmc_PSA_WC_CH4",
        "PSA_WC_N2": "gcmc_PSA_WC_N2",
        "PSA_alpha_CH4_N2": "gcmc_PSA_alpha_CH4_N2",
        "PSA_API_CH4": "gcmc_PSA_API_CH4",
        "VSA_WC_CH4": "gcmc_VSA_WC_CH4",
        "VSA_WC_N2": "gcmc_VSA_WC_N2",
        "VSA_alpha_CH4_N2": "gcmc_VSA_alpha_CH4_N2",
        "VSA_API_CH4": "gcmc_VSA_API_CH4",
    }
    calculated = calculated.rename(columns=rename_map)
    result = df.copy()
    for _, dst_col in rename_map.items():
        result[dst_col] = calculated[dst_col].values
    return result


# ---------------------------------------------------------------------------
# Step 3: ML vs GCMC metrics
# ---------------------------------------------------------------------------

def compute_ml_gcmc_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """R2 and MAE for 8 adsorption targets."""
    pairs = [
        ("AdsCH4_10kPa", "gcmc_AdsCH4_10kPa"),
        ("AdsCH4_100kPa", "gcmc_AdsCH4_100kPa"),
        ("AdsCH4_1000kPa", "gcmc_AdsCH4_1000kPa"),
        ("AdsN2_10kPa", "gcmc_AdsN2_10kPa"),
        ("AdsN2_100kPa", "gcmc_AdsN2_100kPa"),
        ("AdsN2_1000kPa", "gcmc_AdsN2_1000kPa"),
        ("QstCH4", "QstCH4_gcmc"),
        ("QstN2", "QstN2_gcmc"),
    ]
    rows = []
    for ml_col, gcmc_col in pairs:
        if ml_col not in df.columns or gcmc_col not in df.columns:
            continue
        mask = df[ml_col].notna() & df[gcmc_col].notna()
        n = mask.sum()
        if n < 3:
            rows.append({"property": ml_col, "R2": np.nan, "MAE": np.nan, "n": n})
            continue
        y_true = df.loc[mask, gcmc_col].values
        y_pred = df.loc[mask, ml_col].values
        rows.append({
            "property": ml_col,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "n": n,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: Cluster-aware Top-N selection
# ---------------------------------------------------------------------------

def allocate_slots(cluster_counts: dict, total_slots: int = 10) -> dict:
    """Proportional allocation with minimum 1 per cluster."""
    clusters = sorted(cluster_counts.keys())
    n_clusters = len(clusters)
    if n_clusters > total_slots:
        raise ValueError(
            f"More clusters ({n_clusters}) than total slots ({total_slots})."
        )
    allocation = {c: 1 for c in clusters}
    remaining = total_slots - n_clusters
    if remaining <= 0:
        return allocation
    total_count = sum(cluster_counts.values())
    raw_extra = {c: (cluster_counts[c] / total_count) * remaining for c in clusters}
    floored = {c: int(math.floor(v)) for c, v in raw_extra.items()}
    remainders = {c: raw_extra[c] - floored[c] for c in clusters}
    for c in clusters:
        allocation[c] += floored[c]
    leftover = remaining - sum(floored.values())
    if leftover > 0:
        sorted_by_remainder = sorted(
            remainders.keys(), key=lambda c: remainders[c], reverse=True
        )
        for c in sorted_by_remainder[:leftover]:
            allocation[c] += 1
    assert sum(allocation.values()) == total_slots
    return allocation


def select_top_n(
    df: pd.DataFrame,
    api_col: str,
    benchmark: float,
    cluster_col: str = "cluster",
    n: int = 10,
    label: str = "PSA",
) -> pd.DataFrame:
    """Select top-N benchmark-beating MOFs with cluster-aware diversity."""
    # Filter to benchmark beaters with valid GCMC API
    subset = df[df[api_col].notna() & (df[api_col] >= benchmark)].copy()
    print(f"\n{'='*60}")
    print(f"[{label}] Benchmark threshold: {benchmark:.3f}")
    print(f"[{label}] Benchmark-beating candidates: {len(subset)}")
    print(f"{'='*60}")

    if len(subset) == 0:
        print(f"  [WARN] No benchmark-beating candidates for {label}!")
        return pd.DataFrame()

    if len(subset) <= n:
        print(f"  [INFO] Only {len(subset)} candidates, selecting all.")
        result = subset.sort_values(api_col, ascending=False).copy()
        result[f"{label.lower()}_top10_rank"] = range(1, len(result) + 1)
        return result

    # Cluster distribution among benchmark beaters
    cluster_counts = subset[cluster_col].value_counts().to_dict()
    print(f"  Cluster distribution: {dict(sorted(cluster_counts.items()))}")

    allocation = allocate_slots(cluster_counts, total_slots=n)
    print(f"  Slot allocation:      {dict(sorted(allocation.items()))}")

    selected = []
    for cid, n_slots in sorted(allocation.items()):
        cluster_df = subset[subset[cluster_col] == cid].sort_values(
            api_col, ascending=False
        )
        picked = cluster_df.head(n_slots)
        selected.append(picked)
        print(
            f"  Cluster {cid:>2d}: picked {len(picked)}/{len(cluster_df)} "
            f"(best {api_col}={picked[api_col].iloc[0]:.4f})"
        )

    result = pd.concat(selected).sort_values(api_col, ascending=False)
    result[f"{label.lower()}_top10_rank"] = range(1, len(result) + 1)
    print(f"\n  Selected {len(result)} MOFs for {label} Top-{n}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse validation GCMC, select Top-10, prepare for BKT."
    )
    parser.add_argument("--test", action="store_true", help="Dry run: parse only.")
    parser.add_argument("--n", type=int, default=10, help="Top-N per process.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # STEP 1: Parse GCMC + Widom for both groups
    # ======================================================================
    all_merged = []
    for group_name, paths in GROUPS.items():
        print(f"\n{'#'*60}")
        print(f"# Parsing {group_name.upper()} group")
        print(f"{'#'*60}")

        # Load ML predictions
        union_df = pd.read_csv(paths["union_csv"])
        print(f"  Union candidates: {len(union_df)}")

        # Parse GCMC
        print("  Parsing GCMC ...")
        df_gcmc = parse_gcmc(paths["gcmc_dir"])

        # Parse Widom
        print("  Parsing Widom ...")
        df_widom = parse_widom(paths["widom_dir"])

        # Merge
        merged = union_df.merge(df_gcmc, on="mof_id", how="left")
        merged = merged.merge(df_widom, on="mof_id", how="left")

        n_gcmc = merged["gcmc_AdsCH4_100kPa"].notna().sum()
        n_widom = merged["QstCH4_gcmc"].notna().sum()
        print(f"  GCMC coverage: {n_gcmc}/{len(union_df)}")
        print(f"  Widom coverage: {n_widom}/{len(union_df)}")

        merged["group"] = group_name
        all_merged.append(merged)

    # Combine both groups
    df_all = pd.concat(all_merged, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Combined: {len(df_all)} MOFs ({df_all['group'].value_counts().to_dict()})")
    print(f"{'='*60}")

    # ======================================================================
    # STEP 2: Compute GCMC-based API
    # ======================================================================
    print("\nComputing GCMC-based API ...")
    # Only compute for MOFs with complete GCMC data
    has_gcmc = df_all["gcmc_AdsCH4_100kPa"].notna()
    if has_gcmc.sum() > 0:
        df_all = compute_gcmc_api(df_all)
    else:
        print("  [ERROR] No GCMC data found!")
        return

    # Extract ATC-Cu benchmark API from validation set
    benchmark_psa, benchmark_vsa = extract_benchmark_api(df_all)
    print(f"\nATC-Cu benchmark API: PSA={benchmark_psa:.6f}, VSA={benchmark_vsa:.6f}")

    # Identify benchmark beaters
    psa_beaters = df_all[df_all["gcmc_PSA_API_CH4"] >= benchmark_psa]
    vsa_beaters = df_all[df_all["gcmc_VSA_API_CH4"] >= benchmark_vsa]
    print(f"Benchmark-beating candidates:")
    print(f"  PSA (>={benchmark_psa:.4f}): {len(psa_beaters)} "
          f"(exp: {(psa_beaters['group']=='exp').sum()}, "
          f"hypo: {(psa_beaters['group']=='hypo').sum()})")
    print(f"  VSA (>={benchmark_vsa:.4f}): {len(vsa_beaters)} "
          f"(exp: {(vsa_beaters['group']=='exp').sum()}, "
          f"hypo: {(vsa_beaters['group']=='hypo').sum()})")

    # ======================================================================
    # ML vs GCMC metrics
    # ======================================================================
    print("\nML vs GCMC validation metrics (combined):")
    metrics = compute_ml_gcmc_metrics(df_all)
    print(metrics[["property", "R2", "MAE", "n"]].to_string(index=False))
    mean_r2 = metrics["R2"].mean()
    print(f"\nMean R² across 8 properties: {mean_r2:.4f}")

    # Per-group metrics
    for grp in ["exp", "hypo"]:
        grp_df = df_all[df_all["group"] == grp]
        grp_metrics = compute_ml_gcmc_metrics(grp_df)
        grp_r2 = grp_metrics["R2"].mean()
        print(f"  {grp.upper()} Mean R²: {grp_r2:.4f} (n={len(grp_df)})")

    # Save comparison CSV
    comparison_out = OUTPUT_DIR / "gcmc_vs_ml_comparison.csv"
    df_all.to_csv(comparison_out, index=False)
    metrics_out = OUTPUT_DIR / "gcmc_ml_metrics.csv"
    metrics.to_csv(metrics_out, index=False)
    print(f"\nSaved: {comparison_out}")
    print(f"Saved: {metrics_out}")

    if args.test:
        print("\n[TEST MODE] Stopping before selection.")
        return

    # ======================================================================
    # STEP 3: Cluster-aware Top-10 selection
    # ======================================================================
    print("\nLoading cluster labels ...")
    cluster_df = pd.read_csv(CLUSTER_CSV, usecols=["CifId", "cluster"])
    df_all = df_all.merge(cluster_df, left_on="mof_id", right_on="CifId", how="left")
    n_matched = df_all["cluster"].notna().sum()
    print(f"  Cluster labels matched: {n_matched}/{len(df_all)}")
    if n_matched < len(df_all):
        missing = df_all[df_all["cluster"].isna()]["mof_id"].tolist()
        print(f"  WARNING: {len(missing)} MOFs without cluster: {missing[:10]}")

    # Drop MOFs without cluster labels for selection
    df_clustered = df_all[df_all["cluster"].notna()].copy()
    df_clustered["cluster"] = df_clustered["cluster"].astype(int)

    # Select PSA Top-10
    psa_top = select_top_n(
        df_clustered, api_col="gcmc_PSA_API_CH4",
        benchmark=benchmark_psa, n=args.n, label="PSA"
    )

    # Select VSA Top-10
    vsa_top = select_top_n(
        df_clustered, api_col="gcmc_VSA_API_CH4",
        benchmark=benchmark_vsa, n=args.n, label="VSA"
    )

    # Overlap analysis
    psa_ids = set(psa_top["mof_id"]) if len(psa_top) > 0 else set()
    vsa_ids = set(vsa_top["mof_id"]) if len(vsa_top) > 0 else set()
    overlap = psa_ids & vsa_ids
    combined_ids = psa_ids | vsa_ids

    print(f"\n{'='*60}")
    print(f"PSA Top-{args.n}: {len(psa_ids)} MOFs")
    print(f"VSA Top-{args.n}: {len(vsa_ids)} MOFs")
    print(f"Overlap: {len(overlap)} MOFs")
    print(f"Combined unique: {len(combined_ids)} MOFs")
    if overlap:
        print(f"  Overlapping MOFs: {sorted(overlap)}")
    print(f"{'='*60}")

    # Save selection
    psa_out = OUTPUT_DIR / "top10_psa.csv"
    vsa_out = OUTPUT_DIR / "top10_vsa.csv"
    combined = df_all[df_all["mof_id"].isin(combined_ids)].copy()
    combined["in_psa_top10"] = combined["mof_id"].isin(psa_ids)
    combined["in_vsa_top10"] = combined["mof_id"].isin(vsa_ids)
    combined_out = OUTPUT_DIR / "top20_combined.csv"

    if len(psa_top) > 0:
        psa_top.to_csv(psa_out, index=False)
    if len(vsa_top) > 0:
        vsa_top.to_csv(vsa_out, index=False)
    combined.to_csv(combined_out, index=False)

    print(f"\nSaved:")
    print(f"  PSA Top-{args.n}: {psa_out} ({len(psa_top)} MOFs)")
    print(f"  VSA Top-{args.n}: {vsa_out} ({len(vsa_top)} MOFs)")
    print(f"  Combined:        {combined_out} ({len(combined)} unique MOFs)")

    # ======================================================================
    # STEP 4: Symlink CIFs for pure-component GCMC
    # ======================================================================
    cif_dst_dir = OUTPUT_DIR / "cifs"
    cif_dst_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing
    for existing in cif_dst_dir.glob("*.cif"):
        existing.unlink()

    n_linked = 0
    missing_cifs = []
    for mof_id in sorted(combined_ids):
        src = CIF_SRC_DIR / f"{mof_id}.cif"
        dst = cif_dst_dir / f"{mof_id}.cif"
        if src.exists():
            dst.symlink_to(src.resolve())
            n_linked += 1
        else:
            missing_cifs.append(mof_id)

    if missing_cifs:
        print(f"\n  WARNING: {len(missing_cifs)} CIFs not found in {CIF_SRC_DIR}:")
        for m in missing_cifs:
            print(f"    {m}")
    print(f"  Symlinked {n_linked} CIFs → {cif_dst_dir}")

    # Print Top-10 tables
    print(f"\n{'='*60}")
    print(f"PSA Top-{args.n} MOF IDs:")
    print(f"{'='*60}")
    if len(psa_top) > 0:
        for _, row in psa_top.iterrows():
            exp_tag = "[exp]" if row.get("is_exp", False) else "[hyp]"
            print(f"  {int(row['psa_top10_rank']):>2d}. {row['mof_id']} "
                  f"{exp_tag} GCMC_API={row['gcmc_PSA_API_CH4']:.4f} "
                  f"cluster={int(row['cluster'])}")

    print(f"\n{'='*60}")
    print(f"VSA Top-{args.n} MOF IDs:")
    print(f"{'='*60}")
    if len(vsa_top) > 0:
        for _, row in vsa_top.iterrows():
            exp_tag = "[exp]" if row.get("is_exp", False) else "[hyp]"
            print(f"  {int(row['vsa_top10_rank']):>2d}. {row['mof_id']} "
                  f"{exp_tag} GCMC_API={row['gcmc_VSA_API_CH4']:.4f} "
                  f"cluster={int(row['cluster'])}")

    print(f"\n[DONE] Next: submit pure-component GCMC for {len(combined_ids)} MOFs")
    print(f"  conda run -n mofmthnn python src/alignn/submit_pure_component_gcmc.py \\")
    print(f"    --model-dir results/alignn/model_ep150 --bkt-dir process_candidates")


if __name__ == "__main__":
    main()
