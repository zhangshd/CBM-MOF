"""
select_final_top10.py — Task 2.5: Cluster-aware Top-10 selection for BKT simulation.

Selects Top-10 PSA + Top-10 VSA MOFs from GCMC-validated candidates using
cluster-proportional allocation (each cluster gets at least 1 representative).

Within each cluster, MOFs are ranked by GCMC-computed API (gcmc_PSA_API_CH4
for PSA, gcmc_VSA_API_CH4 for VSA).

Usage:
    python src/alignn/select_final_top10.py
    python src/alignn/select_final_top10.py --model-dir results/alignn/model_ep150
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

GCMC_CSV_DEFAULT = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "gcmc_top_candidates" / "gcmc_vs_ml_comparison.csv"
)
CLUSTER_CSV = (
    REPO_ROOT / "results" / "cbm_screening" / "inference"
    / "umap_coordinates_descriptor_with_metrics_ml.csv"
)
TRAINING_ADS_R1_CSV = (
    REPO_ROOT / "results" / "cbm_screening"
    / "gcmc_round1_DreidingTraPPEJson" / "raspa3_parsed_results_0911.csv"
)
TRAINING_WIDOM_R1_CSV = (
    REPO_ROOT / "results" / "cbm_screening"
    / "widom_round1_DREIDING" / "widom_results_0911.csv"
)
OUTPUT_DIR_DEFAULT = (
    REPO_ROOT / "results" / "alignn" / "model_ep150" / "bkt_candidates"
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def allocate_slots(cluster_counts: dict, total_slots: int = 10) -> dict:
    """
    Allocate *total_slots* across clusters proportionally, ensuring each
    cluster gets at least 1 slot.

    Strategy:
      1. Give every cluster 1 slot (minimum guarantee).
      2. Distribute remaining slots proportionally to cluster size.
      3. Handle rounding by giving extra slots to largest remainders.
    """
    clusters = sorted(cluster_counts.keys())
    n_clusters = len(clusters)

    if n_clusters > total_slots:
        raise ValueError(
            f"More clusters ({n_clusters}) than total slots ({total_slots}). "
            "Cannot guarantee 1 slot per cluster."
        )

    # Step 1: everyone gets 1
    allocation = {c: 1 for c in clusters}
    remaining = total_slots - n_clusters

    if remaining <= 0:
        return allocation

    # Step 2: distribute remaining proportionally
    total_count = sum(cluster_counts.values())
    raw_extra = {
        c: (cluster_counts[c] / total_count) * remaining for c in clusters
    }
    floored = {c: int(math.floor(v)) for c, v in raw_extra.items()}
    remainders = {c: raw_extra[c] - floored[c] for c in clusters}

    for c in clusters:
        allocation[c] += floored[c]

    # Step 3: distribute leftover by largest remainder
    leftover = remaining - sum(floored.values())
    if leftover > 0:
        sorted_by_remainder = sorted(
            remainders.keys(), key=lambda c: remainders[c], reverse=True
        )
        for c in sorted_by_remainder[:leftover]:
            allocation[c] += 1

    assert sum(allocation.values()) == total_slots
    return allocation


def load_benchmark_api_thresholds() -> dict[str, float]:
    """Recompute ATC-Cu API thresholds from the Round-1 benchmark data."""
    ads_df = pd.read_csv(TRAINING_ADS_R1_CSV)
    widom_df = pd.read_csv(TRAINING_WIDOM_R1_CSV)

    ads_piv = ads_df.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"], values="AbsLoading", aggfunc="first"
    )
    ads_piv.columns = [f"Ads{gas}_{int(pressure * 100)}kPa" for gas, pressure in ads_piv.columns]
    ads_piv = ads_piv.reset_index().rename(columns={"MofName": "mof_id"})
    ads_piv.rename(columns={c: c.replace("methane", "CH4") for c in ads_piv.columns if "methane" in c}, inplace=True)

    widom_piv = widom_df.pivot_table(index="MofName", columns="GasName", values="AdsorptionHeat", aggfunc="first")
    widom_piv.columns = [f"Qst{gas}" for gas in widom_piv.columns]
    widom_piv = widom_piv.reset_index().rename(columns={"MofName": "mof_id", "Qstmethane": "QstCH4"})

    merged = ads_piv.merge(widom_piv, on="mof_id", how="outer")
    for process, ads_label, des_label in [("PSA", "1000kPa", "100kPa"), ("VSA", "100kPa", "10kPa")]:
        merged[f"{process}_WC_CH4"] = merged[f"AdsCH4_{ads_label}"] - merged[f"AdsCH4_{des_label}"]
        alpha = (merged[f"AdsCH4_{ads_label}"] / merged[f"AdsN2_{ads_label}"]) * 4.0
        merged[f"{process}_alpha_CH4_N2"] = alpha
        merged[f"{process}_API_CH4"] = ((alpha - 1.0) * merged[f"{process}_WC_CH4"]) / merged["QstCH4"].abs()

    row = merged.loc[merged["mof_id"] == "CoRE-2020[Cu][pts]3[ASR]1"]
    if row.empty:
        raise ValueError("ATC-Cu benchmark not found in Round-1 training data.")
    row = row.iloc[0]
    return {"PSA": float(row["PSA_API_CH4"]), "VSA": float(row["VSA_API_CH4"])}


def select_top_n(
    df: pd.DataFrame,
    rank_col: str,
    api_col: str,
    benchmark_threshold: float,
    cluster_col: str = "cluster",
    n: int = 10,
    label: str = "PSA",
) -> pd.DataFrame:
    """
    Select top-N MOFs from the validated benchmark-beating subset where
    *rank_col* is non-null and *api_col* exceeds the ATC-Cu benchmark.
    Within each cluster, rank by *api_col*.
    """
    subset = df[df[rank_col].notna()].copy()
    subset = subset[subset[api_col] > benchmark_threshold].copy()
    print(f"\n{'='*60}")
    print(f"[{label}] Selecting Top-{n} from {len(subset)} benchmark-beating candidates")
    print(f"{'='*60}")

    # Cluster distribution
    cluster_counts = subset[cluster_col].value_counts().to_dict()
    print(f"  Cluster distribution: {dict(sorted(cluster_counts.items()))}")

    # Allocate slots
    allocation = allocate_slots(cluster_counts, total_slots=n)
    print(f"  Slot allocation:      {dict(sorted(allocation.items()))}")

    # Select within each cluster
    selected = []
    for cid, n_slots in sorted(allocation.items()):
        cluster_df = subset[subset[cluster_col] == cid].sort_values(
            [api_col, rank_col, "mof_id"], ascending=[False, True, True]
        )
        picked = cluster_df.head(n_slots)
        selected.append(picked)
        print(
            f"  Cluster {cid:>2d}: picked {len(picked)}/{len(cluster_df)} "
            f"(best {api_col}={picked[api_col].iloc[0]:.4f})"
        )

    result = pd.concat(selected).sort_values([api_col, rank_col, "mof_id"], ascending=[False, True, True])
    result[f"{label.lower()}_top10_rank"] = range(1, len(result) + 1)

    print(f"\n  Selected {len(result)} MOFs for {label} Top-{n}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 2.5: Cluster-aware Top-10 selection for BKT."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model-specific results dir (e.g. results/alignn/model_ep150)."
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of MOFs to select per process (default: 10)."
    )
    args = parser.parse_args()

    # Resolve paths
    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = REPO_ROOT / md
        gcmc_csv = md / "gcmc_top_candidates" / "gcmc_vs_ml_comparison.csv"
        output_dir = md / "bkt_candidates"
    else:
        gcmc_csv = GCMC_CSV_DEFAULT
        output_dir = OUTPUT_DIR_DEFAULT

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading GCMC data:    {gcmc_csv}")
    print(f"Loading cluster data: {CLUSTER_CSV}")
    gcmc = pd.read_csv(gcmc_csv)
    cluster = pd.read_csv(CLUSTER_CSV, usecols=["CifId", "cluster"])
    benchmark_thresholds = load_benchmark_api_thresholds()

    # Merge cluster labels
    merged = gcmc.merge(cluster, left_on="mof_id", right_on="CifId", how="left")
    n_matched = merged["cluster"].notna().sum()
    print(f"Merged: {n_matched}/{len(merged)} MOFs matched cluster labels")
    if n_matched < len(merged):
        missing = merged[merged["cluster"].isna()]["mof_id"].tolist()
        print(f"  WARNING: {len(missing)} MOFs without cluster: {missing[:5]}...")

    # Select Top-10 PSA
    psa_top = select_top_n(
        merged, rank_col="psa_rank", api_col="gcmc_PSA_API_CH4",
        benchmark_threshold=benchmark_thresholds["PSA"],
        n=args.n, label="PSA"
    )

    # Select Top-10 VSA
    vsa_top = select_top_n(
        merged, rank_col="vsa_rank", api_col="gcmc_VSA_API_CH4",
        benchmark_threshold=benchmark_thresholds["VSA"],
        n=args.n, label="VSA"
    )

    # Check overlap
    psa_ids = set(psa_top["mof_id"])
    vsa_ids = set(vsa_top["mof_id"])
    overlap = psa_ids & vsa_ids
    print(f"\n{'='*60}")
    print(f"Overlap between PSA and VSA Top-{args.n}: {len(overlap)} MOFs")
    if overlap:
        print(f"  Overlapping MOFs: {overlap}")
    print(f"{'='*60}")

    # Combined unique MOFs for GCMC submission
    combined_ids = psa_ids | vsa_ids
    combined = merged[merged["mof_id"].isin(combined_ids)].copy()
    combined["in_psa_top10"] = combined["mof_id"].isin(psa_ids)
    combined["in_vsa_top10"] = combined["mof_id"].isin(vsa_ids)

    # Save outputs
    psa_out = output_dir / "top10_psa.csv"
    vsa_out = output_dir / "top10_vsa.csv"
    combined_out = output_dir / "top20_combined.csv"

    psa_top.to_csv(psa_out, index=False)
    vsa_top.to_csv(vsa_out, index=False)
    combined.to_csv(combined_out, index=False)

    print(f"\nSaved:")
    print(f"  PSA Top-{args.n}: {psa_out} ({len(psa_top)} MOFs)")
    print(f"  VSA Top-{args.n}: {vsa_out} ({len(vsa_top)} MOFs)")
    print(f"  Combined:        {combined_out} ({len(combined)} unique MOFs)")

    # Generate summary markdown
    summary_path = output_dir / "cluster_selection_summary.md"
    _write_summary(summary_path, psa_top, vsa_top, combined, overlap, args.n)
    print(f"  Summary:         {summary_path}")

    # Symlink CIFs for BKT candidates
    cif_src_dir = output_dir.parent / "top_candidates" / "cifs"
    cif_dst_dir = output_dir / "cifs"
    if cif_src_dir.exists():
        cif_dst_dir.mkdir(parents=True, exist_ok=True)
        for existing in cif_dst_dir.glob("*.cif"):
            existing.unlink()
        n_linked = 0
        missing_cifs = []
        for mof_id in sorted(combined_ids):
            src = cif_src_dir / f"{mof_id}.cif"
            dst = cif_dst_dir / f"{mof_id}.cif"
            if src.exists():
                dst.symlink_to(src.resolve())
                n_linked += 1
            else:
                missing_cifs.append(str(src))
        for missing in missing_cifs:
            print(f"  WARNING: CIF not found: {missing}")
        print(f"  Symlinked {n_linked} CIFs → {cif_dst_dir}")
    else:
        print(f"  WARNING: CIF source dir not found: {cif_src_dir}")


def _write_summary(
    path: Path, psa: pd.DataFrame, vsa: pd.DataFrame,
    combined: pd.DataFrame, overlap: set, n: int
) -> None:
    """Write a markdown summary of the selection."""
    lines = [
        f"# Top-{n} BKT Candidate Selection Summary",
        f"",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## PSA Top-{n}",
        f"",
        f"| Rank | MOF ID | Cluster | GCMC PSA API CH4 | PSA Rank (Top-100) |",
        f"|------|--------|---------|------------------|--------------------|",
    ]
    for _, row in psa.iterrows():
        lines.append(
            f"| {int(row['psa_top10_rank'])} | {row['mof_id']} | "
            f"{int(row['cluster'])} | {row['gcmc_PSA_API_CH4']:.4f} | "
            f"{int(row['psa_rank'])} |"
        )

    lines += [
        f"",
        f"## VSA Top-{n}",
        f"",
        f"| Rank | MOF ID | Cluster | GCMC VSA API CH4 | VSA Rank (Top-100) |",
        f"|------|--------|---------|------------------|--------------------|",
    ]
    for _, row in vsa.iterrows():
        lines.append(
            f"| {int(row['vsa_top10_rank'])} | {row['mof_id']} | "
            f"{int(row['cluster'])} | {row['gcmc_VSA_API_CH4']:.4f} | "
            f"{int(row['vsa_rank'])} |"
        )

    lines += [
        f"",
        f"## Summary",
        f"",
        f"- PSA Top-{n}: {len(psa)} MOFs from "
        f"{psa['cluster'].nunique()} clusters",
        f"- VSA Top-{n}: {len(vsa)} MOFs from "
        f"{vsa['cluster'].nunique()} clusters",
        f"- Overlap: {len(overlap)} MOFs",
        f"- Unique MOFs for GCMC: {len(combined)}",
        f"",
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
