"""
select_top_candidates.py — Task 2.3b: Top-100 PSA/VSA selection + CIF collection.

1. Load full_library_stable.csv
2. Select Top-100 by PSA_API_CH4 (descending)
3. Select Top-100 by VSA_API_CH4 (descending)
4. Deduplicate union (~150-200 unique MOFs)
5. Validate CIF existence and create symlinks (fallback: copy)
6. Print statistics + Top-5 previews

Usage:
    python src/alignn/select_top_candidates.py          # full run
    python src/alignn/select_top_candidates.py --test   # test mode (uses *_test.csv)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[2]
TOP_CAND_DIR = REPO_ROOT / "results" / "alignn" / "top_candidates"
STABLE_CSV  = TOP_CAND_DIR / "full_library_stable.csv"
# NOTE: Use integrated_cifs (with _symmetry_space_group_name_H-M) for RASPA2 compatibility.
# all_graphs_grids uses _space_group_name_H-M_alt which RASPA2 cannot parse.
CIF_DIR     = REPO_ROOT / "data" / "processed" / "integrated_cifs"

OUTPUT_PSA  = TOP_CAND_DIR / "top100_psa.csv"
OUTPUT_VSA  = TOP_CAND_DIR / "top100_vsa.csv"
OUTPUT_UNION = TOP_CAND_DIR / "top_union.csv"
CIF_LINK_DIR = TOP_CAND_DIR / "cifs"

TOP_N = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_cifs(mof_ids: list, cif_dir: Path, link_dir: Path) -> dict:
    """
    Create symlinks from cif_dir/{mof_id}.cif into link_dir/.
    Falls back to copy if symlink fails (cross-filesystem).
    Returns stats: {"symlink": int, "copy": int, "missing": int}
    """
    link_dir.mkdir(parents=True, exist_ok=True)
    stats = {"symlink": 0, "copy": 0, "missing": 0}

    for mof_id in mof_ids:
        src = cif_dir / f"{mof_id}.cif"
        dst = link_dir / f"{mof_id}.cif"

        if not src.exists():
            print(f"  [WARN] CIF not found: {src}", flush=True)
            stats["missing"] += 1
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        try:
            dst.symlink_to(src.resolve())
            stats["symlink"] += 1
        except OSError:
            # Cross-filesystem: fall back to copy
            shutil.copy2(src, dst)
            stats["copy"] += 1

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_mode: bool = False) -> None:
    TOP_CAND_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load stable library
    # ------------------------------------------------------------------
    if test_mode:
        stable_csv = TOP_CAND_DIR / "full_library_stable_test.csv"
        print(f"[TEST MODE] Loading {stable_csv} …", flush=True)
    else:
        stable_csv = STABLE_CSV
        print(f"Step 1: Loading {stable_csv} …", flush=True)

    df = pd.read_csv(stable_csv)
    print(f"  Stable library: {len(df):,} MOFs × {df.shape[1]} columns", flush=True)

    # ------------------------------------------------------------------
    # Step 2 — Top-100 by PSA_API_CH4
    # ------------------------------------------------------------------
    print("Step 2: Selecting Top-100 by PSA_API_CH4 …", flush=True)
    df_psa = (
        df.dropna(subset=["PSA_API_CH4"])
          .nlargest(TOP_N, "PSA_API_CH4")
          .copy()
          .reset_index(drop=True)
    )
    df_psa.insert(0, "psa_rank", range(1, len(df_psa) + 1))
    print(f"  PSA Top-{TOP_N}: max={df_psa['PSA_API_CH4'].max():.4f}, "
          f"min={df_psa['PSA_API_CH4'].min():.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 3 — Top-100 by VSA_API_CH4
    # ------------------------------------------------------------------
    print("Step 3: Selecting Top-100 by VSA_API_CH4 …", flush=True)
    df_vsa = (
        df.dropna(subset=["VSA_API_CH4"])
          .nlargest(TOP_N, "VSA_API_CH4")
          .copy()
          .reset_index(drop=True)
    )
    df_vsa.insert(0, "vsa_rank", range(1, len(df_vsa) + 1))
    print(f"  VSA Top-{TOP_N}: max={df_vsa['VSA_API_CH4'].max():.4f}, "
          f"min={df_vsa['VSA_API_CH4'].min():.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 4 — Deduplicate union
    # ------------------------------------------------------------------
    print("Step 4: Building deduplicated union …", flush=True)
    psa_ids = set(df_psa["mof_id"].tolist())
    vsa_ids = set(df_vsa["mof_id"].tolist())
    both_ids = psa_ids | vsa_ids

    # Build rank lookup maps
    psa_rank_map = dict(zip(df_psa["mof_id"], df_psa["psa_rank"]))
    vsa_rank_map = dict(zip(df_vsa["mof_id"], df_vsa["vsa_rank"]))

    # Use original df as base (contains all columns)
    df_union = df[df["mof_id"].isin(both_ids)].copy()
    df_union["psa_rank"] = df_union["mof_id"].map(psa_rank_map)
    df_union["vsa_rank"] = df_union["mof_id"].map(vsa_rank_map)

    # Sort: PSA-ranked first (by psa_rank ascending, NaN last), then VSA-only
    df_union = df_union.sort_values(
        ["psa_rank", "vsa_rank"],
        na_position="last"
    ).reset_index(drop=True)

    n_union = len(df_union)
    n_psa_only = len(psa_ids - vsa_ids)
    n_vsa_only = len(vsa_ids - psa_ids)
    n_overlap  = len(psa_ids & vsa_ids)
    print(f"  Union: {n_union} unique MOFs "
          f"(PSA-only={n_psa_only}, VSA-only={n_vsa_only}, both={n_overlap})", flush=True)

    # ------------------------------------------------------------------
    # Step 5 — Validate CIF existence
    # ------------------------------------------------------------------
    print("Step 5: Validating CIF paths …", flush=True)
    union_mof_ids = df_union["mof_id"].tolist()
    cif_exists = [( CIF_DIR / f"{mid}.cif").exists() for mid in union_mof_ids]
    n_found = sum(cif_exists)
    n_missing = n_union - n_found
    print(f"  CIF found: {n_found}/{n_union}, missing: {n_missing}", flush=True)

    # ------------------------------------------------------------------
    # Step 6 — Create symlinks (or copies)
    # ------------------------------------------------------------------
    print("Step 6: Collecting CIFs into cifs/ directory …", flush=True)
    cif_stats = collect_cifs(union_mof_ids, CIF_DIR, CIF_LINK_DIR)
    print(f"  Symlinks: {cif_stats['symlink']}, Copies: {cif_stats['copy']}, "
          f"Missing: {cif_stats['missing']}", flush=True)

    # ------------------------------------------------------------------
    # Step 7 — Preview tables
    # ------------------------------------------------------------------
    print("\nTop-5 PSA candidates:")
    print(df_psa[["psa_rank", "mof_id", "PSA_API_CH4", "PSA_WC_CH4", "PSA_alpha_CH4_N2"]]
          .head(5).to_string(index=False))

    print("\nTop-5 VSA candidates:")
    print(df_vsa[["vsa_rank", "mof_id", "VSA_API_CH4", "VSA_WC_CH4", "VSA_alpha_CH4_N2"]]
          .head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # Step 8 — Save outputs
    # ------------------------------------------------------------------
    if test_mode:
        psa_out   = TOP_CAND_DIR / "top100_psa_test.csv"
        vsa_out   = TOP_CAND_DIR / "top100_vsa_test.csv"
        union_out = TOP_CAND_DIR / "top_union_test.csv"
    else:
        psa_out   = OUTPUT_PSA
        vsa_out   = OUTPUT_VSA
        union_out = OUTPUT_UNION

    df_psa.to_csv(psa_out, index=False)
    df_vsa.to_csv(vsa_out, index=False)
    df_union.to_csv(union_out, index=False)

    print(f"\nSaved:")
    print(f"  PSA Top-100 : {psa_out}  ({len(df_psa)} rows)")
    print(f"  VSA Top-100 : {vsa_out}  ({len(df_vsa)} rows)")
    print(f"  Union       : {union_out}  ({len(df_union)} rows)")
    print(f"  CIFs dir    : {CIF_LINK_DIR}  ({cif_stats['symlink']+cif_stats['copy']} files)")

    # Final summary for documentation
    print("\n=== Summary for api_screening_summary.md ===")
    print(f"PSA Top-100: max={df_psa['PSA_API_CH4'].max():.4f}, "
          f"min={df_psa['PSA_API_CH4'].min():.4f}")
    print(f"VSA Top-100: max={df_vsa['VSA_API_CH4'].max():.4f}, "
          f"min={df_vsa['VSA_API_CH4'].min():.4f}")
    print(f"Union total: {n_union} MOFs")
    print(f"CIF collection: {cif_stats['symlink']+cif_stats['copy']}/{n_union} "
          f"({'%.1f' % (100*(cif_stats['symlink']+cif_stats['copy'])/max(n_union,1))}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2.3b: Top-100 candidate selection")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: use *_test.csv files (from filter --test)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model-specific results dir (e.g. results/alignn/model_ep220). "
                             "Overrides TOP_CAND_DIR and derived paths.")
    args = parser.parse_args()

    if args.model_dir:
        _md = Path(args.model_dir)
        if not _md.is_absolute():
            _md = REPO_ROOT / _md
        TOP_CAND_DIR = _md / "top_candidates"
        STABLE_CSV   = TOP_CAND_DIR / "full_library_stable.csv"
        OUTPUT_PSA   = TOP_CAND_DIR / "top100_psa.csv"
        OUTPUT_VSA   = TOP_CAND_DIR / "top100_vsa.csv"
        OUTPUT_UNION = TOP_CAND_DIR / "top_union.csv"
        CIF_LINK_DIR = TOP_CAND_DIR / "cifs"

    main(test_mode=args.test)
