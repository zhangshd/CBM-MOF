"""Prepare all SuperPSA input files for NSGA-II optimization.

End-to-end pipeline:
  1. Cluster-aware Top-N candidate selection (GCMC-based API ranking)
  2. Build Adsorbents_CH4N2_{PSA,VSA}.csv (DSL isotherm + density + Qst)
  3. Generate ProcessConfig_{PSA,VSA}.yaml (CH4/N2 bounds + LDF)

No intermediate files — PSA and VSA candidate lists are kept separate throughout.

Usage:
    cd /home/zhangsd/repos/CBM-MOF
    conda run -n mofmthnn python -m alignn.process.select_optimization_candidates
    conda run -n mofmthnn python -m alignn.process.select_optimization_candidates --top-n 10
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Repo root & sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent   # src/alignn/process/
ALIGNN_DIR = SCRIPT_DIR.parent                 # src/alignn/
SRC_DIR = ALIGNN_DIR.parent                    # src/
REPO_ROOT = SRC_DIR.parent                     # CBM-MOF repo root
SUPERPSA_DATA = SRC_DIR / "SuperPSA" / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.alignn.run_new_top10_pipeline import (
    GROUPS,
    BENCHMARK_PSA,
    BENCHMARK_VSA,
    CLUSTER_CSV,
    parse_gcmc,
    parse_widom,
    compute_gcmc_api,
    select_top_n,
)
from convert_params_for_superpsa import build_adsorbents_table
from generate_process_config import load_template, adapt_for_ch4_n2, write_config

logger = logging.getLogger(__name__)

ATC_CU_ID = "CoRE-2020[Cu][pts]3[ASR]1"


def select_candidates(top_n: int = 10) -> tuple[set[str], set[str]]:
    """Run cluster-aware selection and return separate PSA/VSA MOF ID sets.

    Each set includes ATC-Cu as benchmark.

    Returns:
        (psa_ids, vsa_ids) — sets of mof_id strings.
    """
    # Step 1: Parse GCMC + Widom for both groups
    all_merged = []
    for group_name, paths in GROUPS.items():
        print(f"\n{'#'*60}")
        print(f"# Parsing {group_name.upper()} group")
        print(f"{'#'*60}")

        union_df = pd.read_csv(paths["union_csv"])
        print(f"  Union candidates: {len(union_df)}")

        df_gcmc = parse_gcmc(paths["gcmc_dir"])
        df_widom = parse_widom(paths["widom_dir"])

        merged = union_df.merge(df_gcmc, on="mof_id", how="left")
        merged = merged.merge(df_widom, on="mof_id", how="left")

        n_gcmc = merged["gcmc_AdsCH4_100kPa"].notna().sum()
        n_widom = merged["QstCH4_gcmc"].notna().sum()
        print(f"  GCMC coverage: {n_gcmc}/{len(union_df)}")
        print(f"  Widom coverage: {n_widom}/{len(union_df)}")

        merged["group"] = group_name
        all_merged.append(merged)

    df_all = pd.concat(all_merged, ignore_index=True)
    print(f"\nCombined: {len(df_all)} MOFs")

    # Step 2: Compute GCMC-based API
    print("\nComputing GCMC-based API ...")
    if df_all["gcmc_AdsCH4_100kPa"].notna().sum() == 0:
        raise RuntimeError("No GCMC data found — cannot compute API")
    df_all = compute_gcmc_api(df_all)

    # Step 3: Cluster-aware Top-N selection
    print("\nLoading cluster labels ...")
    cluster_df = pd.read_csv(CLUSTER_CSV, usecols=["CifId", "cluster"])
    df_all = df_all.merge(cluster_df, left_on="mof_id", right_on="CifId", how="left")
    n_matched = df_all["cluster"].notna().sum()
    print(f"  Cluster labels matched: {n_matched}/{len(df_all)}")

    df_clustered = df_all[df_all["cluster"].notna()].copy()
    df_clustered["cluster"] = df_clustered["cluster"].astype(int)

    psa_top = select_top_n(
        df_clustered, api_col="gcmc_PSA_API_CH4",
        benchmark=BENCHMARK_PSA, n=top_n, label="PSA",
    )
    vsa_top = select_top_n(
        df_clustered, api_col="gcmc_VSA_API_CH4",
        benchmark=BENCHMARK_VSA, n=top_n, label="VSA",
    )

    psa_ids = set(psa_top["mof_id"]) if len(psa_top) > 0 else set()
    vsa_ids = set(vsa_top["mof_id"]) if len(vsa_top) > 0 else set()

    # ATC-Cu always runs both processes as benchmark
    psa_ids.add(ATC_CU_ID)
    vsa_ids.add(ATC_CU_ID)

    # Summary
    overlap = psa_ids & vsa_ids
    print(f"\n{'='*60}")
    print(f"  Selection Summary (top-{top_n})")
    print(f"{'='*60}")
    print(f"  PSA candidates : {len(psa_ids)} (incl. ATC-Cu benchmark)")
    print(f"  VSA candidates : {len(vsa_ids)} (incl. ATC-Cu benchmark)")
    print(f"  Overlap        : {len(overlap)}")
    print(f"{'='*60}")
    print(f"\n  PSA: {sorted(psa_ids)}")
    print(f"  VSA: {sorted(vsa_ids)}")

    return psa_ids, vsa_ids


def main() -> None:
    # Default paths
    bkt_dir = REPO_ROOT / "results" / "alignn" / "model_ep150" / "process_candidates"
    default_fits = bkt_dir / "isotherm_fits" / "best_isotherm_fits.csv"
    default_density = REPO_ROOT / "data" / "processed" / "RAC_and_zeo_features.csv"
    default_qst = bkt_dir / "top20_combined.csv"
    default_template = SUPERPSA_DATA / "ProcessConfig.yaml"

    parser = argparse.ArgumentParser(
        description="Prepare SuperPSA inputs: candidate selection + Adsorbents CSVs + ProcessConfig YAMLs",
    )
    parser.add_argument("--top-n", "-n", type=int, default=10,
                        help="Top-N candidates per process (default: 10)")
    parser.add_argument("--output-dir", "-o", type=Path, default=SUPERPSA_DATA,
                        help="Output directory for CSVs and YAMLs")
    parser.add_argument("--fits", type=Path, default=default_fits,
                        help="Path to best_isotherm_fits.csv")
    parser.add_argument("--density", type=Path, default=default_density,
                        help="Path to RAC_and_zeo_features.csv")
    parser.add_argument("--qst", type=Path, default=default_qst,
                        help="Path to top20_combined.csv")
    parser.add_argument("--template", type=Path, default=default_template,
                        help="ProcessConfig.yaml template")
    parser.add_argument("--skip-config", action="store_true",
                        help="Skip ProcessConfig YAML generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ---- Step 1: Select candidates ----
    psa_ids, vsa_ids = select_candidates(args.top_n)

    # ---- Step 2: Build Adsorbents CSVs ----
    print("\nBuilding Adsorbents tables ...")
    df_all = build_adsorbents_table(
        fits_csv=args.fits,
        density_csv=args.density,
        qst_csv=args.qst,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for mode, mof_ids in [("PSA", psa_ids), ("VSA", vsa_ids)]:
        df_mode = df_all[df_all["material_name"].isin(mof_ids)].reset_index(drop=True)
        csv_path = args.output_dir / f"Adsorbents_CH4N2_{mode}.csv"
        df_mode.to_csv(csv_path, index=False)
        print(f"  {mode}: {len(df_mode)} materials -> {csv_path}")
        for _, row in df_mode.iterrows():
            print(f"    {row['material_name']:50s}  rho_s={row['ro_s [kg/m^3]']:7.1f}"
                  f"  deltaU_CH4={row['deltaU_CO2 [J/mol]']:8.1f}"
                  f"  deltaU_N2={row['deltaU_N2 [J/mol]']:8.1f}")

    # ---- Step 3: Generate ProcessConfig YAMLs ----
    if not args.skip_config:
        print("\nGenerating ProcessConfig YAMLs ...")
        template = load_template(args.template)
        for mode in ("PSA", "VSA"):
            config = adapt_for_ch4_n2(template, mode)
            yaml_path = args.output_dir / f"ProcessConfig_{mode}.yaml"
            write_config(config, yaml_path)
            print(f"  {mode}: {yaml_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
