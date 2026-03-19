"""
select_exp_top_candidates.py — Select Top-50 experimental + Top-50 hypothetical MOFs per process.

Uses the no-UQ-filter stable library (full_library_stable_no_uq_filter.csv) as input.
Classification: CoRE-, MOSAEC-, ARC-DB12-, ARC-DB14- = experimental; all other = hypothetical.

Outputs per-process and union CSVs + CIF symlink directories.

Usage:
    python src/alignn/select_exp_top_candidates.py
    python src/alignn/select_exp_top_candidates.py --top-n 50  # default
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "results" / "alignn" / "model_ep150" / "top_candidates" / "full_library_stable_no_uq_filter.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150" / "top_candidates"
CIF_SOURCE = REPO_ROOT / "data" / "processed" / "integrated_cifs"

EXP_PREFIXES = ("CoRE-", "MOSAEC-", "ARC-DB12-", "ARC-DB14-")


def is_experimental(mof_id: str) -> bool:
    return any(mof_id.startswith(p) for p in EXP_PREFIXES)


def main():
    parser = argparse.ArgumentParser(description="Select exp/hypo Top-N per process")
    parser.add_argument("--top-n", type=int, default=50, help="Top N per process per category")
    parser.add_argument("--input", type=str, default=None, help="Custom input CSV")
    args = parser.parse_args()

    top_n = args.top_n
    input_csv = Path(args.input) if args.input else INPUT_CSV
    if not input_csv.is_absolute():
        input_csv = REPO_ROOT / input_csv

    print(f"Loading {input_csv} ...", flush=True)
    df = pd.read_csv(input_csv)
    df["is_exp"] = df["mof_id"].apply(is_experimental)
    print(f"  Total: {len(df)}, Exp: {df['is_exp'].sum()}, Hypo: {(~df['is_exp']).sum()}")

    exp = df[df["is_exp"]]
    hypo = df[~df["is_exp"]]

    results = {}
    for cat_name, cat_df in [("exp", exp), ("hypo", hypo)]:
        psa_top = cat_df.nlargest(top_n, "PSA_API_CH4")
        vsa_top = cat_df.nlargest(top_n, "VSA_API_CH4")
        union = pd.concat([psa_top, vsa_top]).drop_duplicates("mof_id")

        # Save CSVs
        psa_path = OUTPUT_DIR / f"{cat_name}_top{top_n}_psa.csv"
        vsa_path = OUTPUT_DIR / f"{cat_name}_top{top_n}_vsa.csv"
        union_path = OUTPUT_DIR / f"{cat_name}_union.csv"
        psa_top.to_csv(psa_path, index=False)
        vsa_top.to_csv(vsa_path, index=False)
        union.to_csv(union_path, index=False)

        results[cat_name] = {
            "psa": psa_top, "vsa": vsa_top, "union": union,
        }

        print(f"\n{cat_name.upper()} Top-{top_n}:")
        print(f"  PSA range: {psa_top['PSA_API_CH4'].min():.4f} - {psa_top['PSA_API_CH4'].max():.4f}")
        print(f"  VSA range: {vsa_top['VSA_API_CH4'].min():.4f} - {vsa_top['VSA_API_CH4'].max():.4f}")
        print(f"  Union: {len(union)} unique MOFs")

        # UQ stats
        if "flag_high_uq" in union.columns:
            n_high = union["flag_high_uq"].sum()
            print(f"  High UQ: {n_high}/{len(union)} ({100*n_high/len(union):.0f}%)")

        # Source breakdown for exp
        if cat_name == "exp":
            for prefix in EXP_PREFIXES:
                n = len(union[union["mof_id"].str.startswith(prefix)])
                if n > 0:
                    print(f"  {prefix}: {n}")

    # Combined union for GCMC
    all_union = pd.concat([results["exp"]["union"], results["hypo"]["union"]]).drop_duplicates("mof_id")
    all_union_path = OUTPUT_DIR / "all_top_union.csv"
    all_union.to_csv(all_union_path, index=False)
    print(f"\nCombined union: {len(all_union)} unique MOFs → {all_union_path}")

    # Create CIF symlinks directory
    cif_dir = OUTPUT_DIR / "cifs_all_top"
    cif_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing symlinks
    for old_link in cif_dir.glob("*.cif"):
        old_link.unlink()

    n_found = 0
    n_missing = 0
    for mof_id in all_union["mof_id"]:
        src = CIF_SOURCE / f"{mof_id}.cif"
        dst = cif_dir / f"{mof_id}.cif"
        if src.exists():
            dst.symlink_to(src)
            n_found += 1
        else:
            print(f"  [WARN] CIF not found: {src}")
            n_missing += 1

    print(f"\nCIF symlinks: {n_found} created, {n_missing} missing → {cif_dir}")

    # Also create separate exp/hypo CIF dirs for targeted GCMC submission
    for cat_name in ["exp", "hypo"]:
        cat_cif_dir = OUTPUT_DIR / f"cifs_{cat_name}_top"
        cat_cif_dir.mkdir(parents=True, exist_ok=True)
        for old_link in cat_cif_dir.glob("*.cif"):
            old_link.unlink()
        cat_union = results[cat_name]["union"]
        n = 0
        for mof_id in cat_union["mof_id"]:
            src = CIF_SOURCE / f"{mof_id}.cif"
            dst = cat_cif_dir / f"{mof_id}.cif"
            if src.exists():
                dst.symlink_to(src)
                n += 1
        print(f"  {cat_name} CIF dir: {n} symlinks → {cat_cif_dir}")

    # Check overlap with old top_union
    old_top_path = OUTPUT_DIR / "top_union.csv"
    if old_top_path.exists():
        old_top = pd.read_csv(old_top_path)
        old_ids = set(old_top["mof_id"])
        new_hypo_ids = set(results["hypo"]["union"]["mof_id"])
        overlap = old_ids & new_hypo_ids
        new_only = new_hypo_ids - old_ids
        old_only = old_ids - new_hypo_ids
        print(f"\nHypo vs old Top-{len(old_ids)} overlap:")
        print(f"  In both: {len(overlap)}")
        print(f"  New only (need 20:80 GCMC): {len(new_only)}")
        print(f"  Old only (already have 20:80 GCMC): {len(old_only)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
