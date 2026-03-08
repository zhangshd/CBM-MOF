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


def select_top_n(
    df: pd.DataFrame,
    rank_col: str,
    api_col: str,
    cluster_col: str = "cluster",
    n: int = 10,
    label: str = "PSA",
) -> pd.DataFrame:
    """
    Select top-N MOFs from the subset where *rank_col* is non-null,
    using cluster-proportional allocation and ranking by *api_col*.
    """
    subset = df[df[rank_col].notna()].copy()
    print(f"\n{'='*60}")
    print(f"[{label}] Selecting Top-{n} from {len(subset)} candidates")
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
        n=args.n, label="PSA"
    )

    # Select Top-10 VSA
    vsa_top = select_top_n(
        merged, rank_col="vsa_rank", api_col="gcmc_VSA_API_CH4",
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
        n_linked = 0
        for mof_id in combined_ids:
            src = cif_src_dir / f"{mof_id}.cif"
            dst = cif_dst_dir / f"{mof_id}.cif"
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())
                n_linked += 1
            elif not src.exists():
                print(f"  WARNING: CIF not found: {src}")
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
