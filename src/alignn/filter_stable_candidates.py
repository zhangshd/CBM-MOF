"""
filter_stable_candidates.py — Task 2.3a: Stability screening of ML-screened library.

Apply four stability filters to full_library_screened.csv:
  Filter 1: NOT IfPreciousOrRare  (precious/rare metal detection from CIF)
  Filter 2: No Al metal nodes     (DREIDING force field artifact; Li et al. 2024 JCTC)
  Filter 3: SSD_pred == 1         (solvent-removal stable; MOFSNN-covered MOFs only)
  Filter 4: WS24_water_pred == 1  (water stable; MOFSNN-covered MOFs only)

Precious/rare metal detection logic is replicated verbatim from
  src/experiments/exp08_screening_ml.py::detect_precious_rare_metals_in_cif()

Usage:
    python src/alignn/filter_stable_candidates.py          # full run
    python src/alignn/filter_stable_candidates.py --test   # first 500 rows only
    python src/alignn/filter_stable_candidates.py --input path/to/input.csv  # custom input
"""

import argparse
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT       = Path(__file__).resolve().parents[2]
SCREENED_CSV    = REPO_ROOT / "results" / "alignn" / "full_library_inference" / "full_library_screened.csv"
STABILITY_CSV   = REPO_ROOT / "data" / "processed" / "stabilities" / "infer_results_mofsnn.csv"
CIF_DIR         = REPO_ROOT / "results" / "cbm_screening" / "all_graphs_grids"
OUTPUT_DIR      = REPO_ROOT / "results" / "alignn" / "top_candidates"
OUTPUT_CSV      = OUTPUT_DIR / "full_library_stable.csv"

# ---------------------------------------------------------------------------
# Precious / rare metal set (46 elements — verbatim from exp08)
# ---------------------------------------------------------------------------
PRECIOUS_RARE_METALS = {
    'Am', 'Au', 'Ag', 'Dy', 'Eu', 'Ga', 'Gd', 'Hf', 'In', 'Ir', 'La', 'Mo', 'Nd',
    'Pd', 'Pr', 'Pt', 'Rh', 'Ru', 'Se', 'Sm', 'Tb', 'Te', 'Tm', 'U',  'Y',
    'Be', 'Bi', 'Cs', 'Er', 'Ho', 'Lu', 'Nb', 'Os', 'Re', 'Sb', 'Ta', 'Th',
    'Tl', 'W',  'Yb', 'Hg',
}

# Force field reliability: Group IIIA metals (Al) — DREIDING gives unreliable
# adsorption for these metal nodes due to missing/poor LJ parameters.
# Li et al. 2024, J. Chem. Theory Comput. — 16x CH4 overestimate for Al-MOFs.
# Ga and In are already in PRECIOUS_RARE_METALS; only Al needs a separate filter.
FF_UNRELIABLE_METALS = {'Al'}


def extract_elements_from_cif(cif_file_path) -> set:
    """Extract element symbols found in a CIF file.

    Parsing logic from exp08_screening_ml.py::detect_precious_rare_metals_in_cif().
    Returns a set of element symbols (e.g. {'Cu', 'C', 'O', 'N', 'H'}).
    """
    cif_path = Path(cif_file_path)
    if not cif_path.exists():
        print(f"  [WARN] CIF not found: {cif_path}", flush=True)
        return set()
    try:
        content = cif_path.read_text(encoding="utf-8", errors="ignore")
        element_patterns = [
            r'_atom_site_type_symbol\s+(\w+)',
            r'_atom_site_label\s+(\w+)',
            r'_chemical_formula_sum\s+[\'"]([^\'"]+)[\'"]',
            r'_chemical_formula_structural\s+[\'"]([^\'"]+)[\'"]',
            r'^(\w+)\d*\s+\w+\s+[\d\.\-\+]+\s+[\d\.\-\+]+\s+[\d\.\-\+]+',
        ]
        found_elements: set = set()
        for pattern in element_patterns:
            for match in re.findall(pattern, content, re.MULTILINE | re.IGNORECASE):
                if ' ' in match:
                    found_elements.update(re.findall(r'([A-Z][a-z]?)', match))
                else:
                    m = re.match(r'^([A-Z][a-z]?)', match)
                    if m:
                        found_elements.add(m.group(1))
        found_elements.update(re.findall(r'\b([A-Z][a-z]?)\b', content))
        return found_elements
    except Exception as e:
        print(f"  [WARN] Error reading {cif_path}: {e}", flush=True)
        return set()


def detect_precious_rare_metals_in_cif(cif_file_path) -> bool:
    """Return True if the CIF contains any precious/rare metals."""
    return bool(extract_elements_from_cif(cif_file_path) & PRECIOUS_RARE_METALS)


def _extract_elements_batch(batch: list) -> list:
    """Module-level wrapper for parallel element extraction."""
    return [extract_elements_from_cif(CIF_DIR / f"{mof_id}.cif") for mof_id in batch]


def extract_elements_parallel(mof_ids: list) -> list:
    """Extract elements from CIFs in parallel; returns list of sets."""
    n_cores = min(cpu_count(), 32)
    batch_size = max(1, len(mof_ids) // (n_cores * 8))
    batches = [mof_ids[i:i + batch_size] for i in range(0, len(mof_ids), batch_size)]

    print(f"  Extracting elements from CIFs: {n_cores} workers, "
          f"{len(batches)} batches × ~{batch_size} MOFs …", flush=True)

    results: list = []
    with Pool(processes=n_cores) as pool:
        for i, batch_result in enumerate(pool.imap(_extract_elements_batch, batches, chunksize=1), 1):
            results.extend(batch_result)
            if i % max(1, len(batches) // 10) == 0:
                print(f"    {i}/{len(batches)} batches done ({len(results)} MOFs processed)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(test_mode: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load screened library
    # ------------------------------------------------------------------
    print("Step 1: Loading full_library_screened.csv …", flush=True)
    df = pd.read_csv(SCREENED_CSV)
    n_start = len(df)
    print(f"  Input: {n_start:,} MOFs × {df.shape[1]} columns", flush=True)

    if test_mode:
        df = df.head(500).copy()
        print(f"  [TEST MODE] Truncated to {len(df)} rows", flush=True)

    # ------------------------------------------------------------------
    # Step 2 — Load MOFSNN stability predictions, left-join
    # ------------------------------------------------------------------
    print("Step 2: Loading MOFSNN stability CSV and left-joining …", flush=True)
    df_stable = pd.read_csv(STABILITY_CSV, usecols=["MofName", "SSD_pred", "WS24_water_pred"])
    n_mofsnn = len(df_stable)
    print(f"  MOFSNN CSV: {n_mofsnn:,} entries", flush=True)

    df = df.merge(df_stable.rename(columns={"MofName": "mof_id"}),
                  on="mof_id", how="left")

    n_covered = df["SSD_pred"].notna().sum()
    n_uncovered = df["SSD_pred"].isna().sum()
    print(f"  MOFSNN coverage: {n_covered:,} covered / {n_uncovered:,} uncovered (kept, not filtered)", flush=True)

    # ------------------------------------------------------------------
    # Step 3 — Element extraction from CIFs (parallel, single pass)
    # ------------------------------------------------------------------
    print("Step 3: Extracting elements from CIFs …", flush=True)
    mof_ids = df["mof_id"].tolist()
    element_sets = extract_elements_parallel(mof_ids)

    # Derive precious/rare metal flags and Al flags from extracted elements
    precious_flags = [bool(elems & PRECIOUS_RARE_METALS) for elems in element_sets]
    al_flags = [bool(elems & FF_UNRELIABLE_METALS) for elems in element_sets]
    df["IfPreciousOrRare"] = precious_flags
    df["IfAlMetal"] = al_flags
    n_precious = sum(precious_flags)
    n_al = sum(al_flags)
    print(f"  Detected {n_precious:,} MOFs with precious/rare metals", flush=True)
    print(f"  Detected {n_al:,} MOFs with Al metal nodes", flush=True)

    # ------------------------------------------------------------------
    # Step 4 — Apply four filters with per-step statistics
    # ------------------------------------------------------------------
    print("Step 4: Applying stability filters …", flush=True)
    n_before = len(df)

    # Filter 1: no precious/rare metals
    mask_precious = ~df["IfPreciousOrRare"]
    removed_f1 = (~mask_precious).sum()
    df = df[mask_precious].copy()
    print(f"  Filter 1 (no precious/rare metals): removed {removed_f1:,}, remaining {len(df):,}", flush=True)

    # Filter 2: Force field reliability — exclude Al metal-node MOFs.
    # DREIDING force field produces unreliable CH4 adsorption for Al-MOFs
    # (up to 16x overestimate; Li et al. 2024, J. Chem. Theory Comput.).
    # Ga and In are already excluded by Filter 1 (precious/rare metals).
    mask_al = ~df["IfAlMetal"]
    removed_f2 = (~mask_al).sum()
    df = df[mask_al].copy()
    print(f"  Filter 2 (no Al metal nodes — FF reliability): removed {removed_f2:,}, remaining {len(df):,}", flush=True)

    # Filter 3: SSD_pred == 1 (only for MOFSNN-covered MOFs)
    mask_ssd = df["SSD_pred"].isna() | (df["SSD_pred"] == 1)
    removed_f3 = (~mask_ssd).sum()
    df = df[mask_ssd].copy()
    print(f"  Filter 3 (SSD_pred == 1, MOFSNN-covered only): removed {removed_f3:,}, remaining {len(df):,}", flush=True)

    # Filter 4: WS24_water_pred == 1 (only for MOFSNN-covered MOFs)
    mask_ws = df["WS24_water_pred"].isna() | (df["WS24_water_pred"] == 1)
    removed_f4 = (~mask_ws).sum()
    df = df[mask_ws].copy()
    print(f"  Filter 4 (WS24_water_pred == 1, MOFSNN-covered only): removed {removed_f4:,}, remaining {len(df):,}", flush=True)

    n_final = len(df)
    total_removed = n_before - n_final
    print(f"\n  Summary: {n_start:,} → {n_final:,} (removed {total_removed:,} total)", flush=True)

    # ------------------------------------------------------------------
    # Step 5 — Print filter statistics table
    # ------------------------------------------------------------------
    print("\n  Filter statistics:")
    print(f"  {'Step':<45} {'Removed':>10} {'Remaining':>12}")
    print(f"  {'-'*45} {'-'*10} {'-'*12}")
    print(f"  {'Input (post-Tasks 2.1+2.2)':<45} {'':>10} {n_before:>12,}")
    print(f"  {'Filter 1: precious/rare metals':<45} {removed_f1:>10,} {n_before-removed_f1:>12,}")
    print(f"  {'Filter 2: Al metal nodes (FF reliability)':<45} {removed_f2:>10,} {n_before-removed_f1-removed_f2:>12,}")
    print(f"  {'Filter 3: SSD_pred != 1':<45} {removed_f3:>10,} {n_before-removed_f1-removed_f2-removed_f3:>12,}")
    print(f"  {'Filter 4: WS24_water_pred != 1':<45} {removed_f4:>10,} {n_final:>12,}")

    # ------------------------------------------------------------------
    # Step 6 — Save output
    # ------------------------------------------------------------------
    if test_mode:
        out_path = OUTPUT_DIR / "full_library_stable_test.csv"
    else:
        out_path = OUTPUT_CSV

    df.to_csv(out_path, index=False)
    print(f"\nStep 6: Saved → {out_path}", flush=True)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns", flush=True)

    # Final MOFSNN coverage in stable set
    n_cov_final = df["SSD_pred"].notna().sum()
    print(f"  MOFSNN coverage in stable set: {n_cov_final:,} / {n_final:,} "
          f"({100*n_cov_final/max(n_final,1):.1f}%)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2.3a: Stability screening")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process first 500 rows only")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model-specific results dir (e.g. results/alignn/model_ep220). "
                             "Overrides SCREENED_CSV and OUTPUT_DIR.")
    parser.add_argument("--input", type=str, default=None,
                        help="Custom input CSV (overrides default SCREENED_CSV). "
                             "Use to bypass UQ pre-screening, e.g. full_library_with_api.csv.")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output CSV path (overrides default OUTPUT_CSV).")
    args = parser.parse_args()

    if args.model_dir:
        _md = Path(args.model_dir)
        if not _md.is_absolute():
            _md = REPO_ROOT / _md
        SCREENED_CSV = _md / "full_library_inference" / "full_library_screened.csv"
        OUTPUT_DIR   = _md / "top_candidates"
        OUTPUT_CSV   = OUTPUT_DIR / "full_library_stable.csv"

    if args.input:
        _ip = Path(args.input)
        if not _ip.is_absolute():
            _ip = REPO_ROOT / _ip
        SCREENED_CSV = _ip

    if args.output:
        _op = Path(args.output)
        if not _op.is_absolute():
            _op = REPO_ROOT / _op
        OUTPUT_CSV = _op
        OUTPUT_DIR = _op.parent

    main(test_mode=args.test)
