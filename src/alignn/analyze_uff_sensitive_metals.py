#!/usr/bin/env python
"""
Identify UFF-sensitive MOFs in top-50 candidate lists.

Group IIA (Mg, Ca, Sr, Ba) and Group IIIA (Al, Ga, In) metals are known to
have inflated GCMC adsorption due to UFF force field artifacts
(Li et al. 2024 JCTC).

This script:
  1. Reads the top-50 candidate CSV files (hypo/exp × PSA/VSA) + union
  2. Extracts metal nodes from CIF files or mof_id naming conventions
  3. Flags UFF-sensitive MOFs and reports summary statistics
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# UFF-sensitive metals (Li et al. 2024 JCTC)
UFF_SENSITIVE_METALS = {"Al", "Mg", "Ca", "Sr", "Ba", "Ga", "In"}

# Known non-metal elements (for filtering out from CIF)
NON_METALS = {
    "H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar",
    "Se", "Br", "Kr", "I", "Xe", "At", "Rn",
    "B", "Si", "Ge", "As", "Sb", "Te",  # metalloids (not MOF metals)
}


def extract_metals_from_cif(cif_path: Path) -> set[str]:
    """Extract unique element symbols from a CIF file's _atom_site_type_symbol column."""
    elements = set()
    in_atom_loop = False
    type_symbol_col = None

    with open(cif_path, "r") as f:
        lines = f.readlines()

    # Parse the loop_ block containing _atom_site_type_symbol
    col_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "loop_":
            in_atom_loop = False
            type_symbol_col = None
            col_idx = 0
            # Check if next lines contain _atom_site_type_symbol
            for j in range(i + 1, min(i + 20, len(lines))):
                next_line = lines[j].strip()
                if next_line.startswith("_atom_site_type_symbol"):
                    type_symbol_col = col_idx
                    in_atom_loop = True
                if next_line.startswith("_atom_site") or next_line.startswith("_atom_type"):
                    col_idx += 1
                elif not next_line.startswith("_") and next_line and not next_line.startswith("loop_"):
                    break
            continue

        if in_atom_loop and type_symbol_col is not None:
            if not stripped.startswith("_") and not stripped.startswith("loop_") and stripped:
                parts = stripped.split()
                if len(parts) > type_symbol_col:
                    elem = parts[type_symbol_col]
                    # Validate it looks like an element symbol
                    if re.match(r"^[A-Z][a-z]?$", elem):
                        elements.add(elem)

    return elements


def extract_metals_from_name(mof_id: str) -> set[str]:
    """Try to extract metals from MOF name patterns.

    Handles:
      - CoRE-YYYY[Metal]... → bracket-enclosed metal
      - ARC-DB1-Al2O6-...   → formula segment with metals
      - ARC-DB12/14-...     → experimental, may have metal in name
    """
    metals = set()

    # CoRE pattern: CoRE-2020[Cu][pts]3[ASR]1
    bracket_matches = re.findall(r"\[([A-Z][a-z]?)\]", mof_id)
    for m in bracket_matches:
        if re.match(r"^[A-Z][a-z]?$", m) and m not in {"A", "B"}:
            # Check if it's a known element (simple heuristic: 1-2 chars, uppercase start)
            metals.add(m)

    # ARC-DB1 pattern: ARC-DB1-Al2O6-... (formula in the name)
    if mof_id.startswith("ARC-DB1-"):
        # Extract the formula segment (e.g., Al2O6)
        parts = mof_id.split("-")
        if len(parts) >= 3:
            formula_seg = parts[2]  # e.g., "Al2O6"
            elem_matches = re.findall(r"([A-Z][a-z]?)\d*", formula_seg)
            for elem in elem_matches:
                metals.add(elem)

    return metals


def identify_metal_nodes(mof_id: str, cif_dir: Path) -> tuple[set[str], str]:
    """Identify the metal node(s) of a MOF.

    Returns:
        (set of metal element symbols, source method)
    """
    all_metals = set()
    source = "unknown"

    # Step 1: Try name-based extraction
    name_metals = extract_metals_from_name(mof_id)

    # Step 2: Try CIF-based extraction
    cif_name = mof_id + ".cif"
    cif_path = cif_dir / cif_name
    cif_elements = set()
    if cif_path.exists():
        cif_elements = extract_metals_from_cif(cif_path)
        cif_metals = cif_elements - NON_METALS
        all_metals = cif_metals
        source = "cif"
    elif name_metals:
        # Filter to likely metals
        all_metals = name_metals - NON_METALS
        source = "name"
    else:
        source = "no_cif"

    return all_metals, source


def analyze_list(df: pd.DataFrame, list_name: str, cif_dir: Path) -> pd.DataFrame:
    """Analyze a single top-50 list for UFF-sensitive metals."""
    results = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        mof_id = row["mof_id"]
        metals, source = identify_metal_nodes(mof_id, cif_dir)
        uff_metals = metals & UFF_SENSITIVE_METALS
        is_uff_sensitive = len(uff_metals) > 0

        results.append({
            "rank": rank,
            "mof_id": mof_id,
            "metals": ",".join(sorted(metals)) if metals else "unknown",
            "uff_sensitive_metals": ",".join(sorted(uff_metals)) if uff_metals else "",
            "is_uff_sensitive": is_uff_sensitive,
            "source": source,
            "PSA_API": row.get("PSA_API_CH4", None),
            "VSA_API": row.get("VSA_API_CH4", None),
            "PSA_WC_CH4": row.get("PSA_WC_CH4", None),
            "PSA_alpha": row.get("PSA_alpha_CH4_N2", None),
            "list": list_name,
        })

    return pd.DataFrame(results)


def print_separator(char: str = "=", width: int = 100):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(description="Identify UFF-sensitive MOFs in top-50 candidates")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150"),
        help="Model results directory",
    )
    args = parser.parse_args()

    top_dir = args.model_dir / "top_candidates"
    cif_dir = top_dir / "cifs_all_top"

    # Define files to analyze
    files = {
        "hypo_top50_psa": top_dir / "hypo_top50_psa.csv",
        "hypo_top50_vsa": top_dir / "hypo_top50_vsa.csv",
        "exp_top50_psa": top_dir / "exp_top50_psa.csv",
        "exp_top50_vsa": top_dir / "exp_top50_vsa.csv",
        "all_top_union": top_dir / "all_top_union.csv",
    }

    # Verify files exist
    for name, path in files.items():
        if not path.exists():
            logger.error("File not found: %s", path)
            return

    if not cif_dir.exists():
        logger.error("CIF directory not found: %s", cif_dir)
        return

    # Analyze each list
    all_results = []
    for name, path in files.items():
        df = pd.read_csv(path)
        result_df = analyze_list(df, name, cif_dir)
        all_results.append(result_df)

    combined = pd.concat(all_results, ignore_index=True)

    # ==========================================
    # REPORT
    # ==========================================
    print_separator("=")
    print("UFF-SENSITIVE METAL ANALYSIS — TOP-50 MOF CANDIDATES")
    print(f"Target metals: {', '.join(sorted(UFF_SENSITIVE_METALS))}")
    print(f"Rationale: Li et al. 2024 JCTC — UFF force field artifacts for Group IIA/IIIA")
    print_separator("=")

    # Per-list summary
    print("\n[1] SUMMARY BY LIST")
    print_separator("-")
    print(f"{'List':<25} {'Total':>6} {'UFF-Sens':>9} {'Pct':>6} {'Metals Found':<40}")
    print_separator("-")

    for name in files.keys():
        subset = combined[combined["list"] == name]
        n_total = len(subset)
        n_uff = subset["is_uff_sensitive"].sum()
        pct = 100.0 * n_uff / n_total if n_total > 0 else 0.0
        # Unique UFF metals found
        uff_metals_found = set()
        for m in subset[subset["is_uff_sensitive"]]["uff_sensitive_metals"]:
            if m:
                uff_metals_found.update(m.split(","))
        metals_str = ", ".join(sorted(uff_metals_found)) if uff_metals_found else "none"
        print(f"{name:<25} {n_total:>6} {n_uff:>9} {pct:>5.1f}% {metals_str:<40}")

    print_separator("-")

    # Detailed UFF-sensitive MOFs for each top-50 list
    for name in files.keys():
        if name == "all_top_union":
            continue  # Handle union separately
        subset = combined[(combined["list"] == name) & (combined["is_uff_sensitive"])]
        if len(subset) == 0:
            print(f"\n[{name}] No UFF-sensitive MOFs found.")
            continue

        print(f"\n[2] UFF-SENSITIVE MOFs IN: {name}")
        print_separator("-")
        print(f"{'Rank':>5} {'MOF ID':<60} {'UFF Metal':>10} {'PSA_API':>9} {'VSA_API':>9} {'PSA_WC':>9} {'PSA_alpha':>10}")
        print_separator("-")
        for _, row in subset.iterrows():
            psa_api = f"{row['PSA_API']:.4f}" if pd.notna(row["PSA_API"]) else "N/A"
            vsa_api = f"{row['VSA_API']:.4f}" if pd.notna(row["VSA_API"]) else "N/A"
            psa_wc = f"{row['PSA_WC_CH4']:.4f}" if pd.notna(row["PSA_WC_CH4"]) else "N/A"
            psa_alpha = f"{row['PSA_alpha']:.4f}" if pd.notna(row["PSA_alpha"]) else "N/A"
            print(
                f"{row['rank']:>5} {row['mof_id']:<60} {row['uff_sensitive_metals']:>10} "
                f"{psa_api:>9} {vsa_api:>9} {psa_wc:>9} {psa_alpha:>10}"
            )
        print_separator("-")

    # Union analysis
    union_subset = combined[combined["list"] == "all_top_union"]
    n_union = len(union_subset)
    n_uff_union = union_subset["is_uff_sensitive"].sum()
    uff_union = union_subset[union_subset["is_uff_sensitive"]]

    print(f"\n\n[3] ALL-TOP UNION ANALYSIS ({n_union} MOFs)")
    print_separator("=")
    print(f"UFF-sensitive: {n_uff_union}/{n_union} ({100*n_uff_union/n_union:.1f}%)")

    # Metal distribution in union
    metal_counts = {}
    for _, row in union_subset.iterrows():
        for m in row["metals"].split(","):
            if m and m != "unknown":
                metal_counts[m] = metal_counts.get(m, 0) + 1

    print(f"\nMetal node distribution (all {n_union} MOFs):")
    print_separator("-")
    print(f"{'Metal':>8} {'Count':>7} {'Pct':>7} {'UFF-Sensitive':>14}")
    print_separator("-")
    for metal, count in sorted(metal_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / n_union
        uff_flag = " *** YES ***" if metal in UFF_SENSITIVE_METALS else ""
        print(f"{metal:>8} {count:>7} {pct:>6.1f}% {uff_flag:>14}")
    print_separator("-")

    # List UFF-sensitive MOFs in union
    if len(uff_union) > 0:
        print(f"\nUFF-sensitive MOFs in union ({len(uff_union)}):")
        print_separator("-")
        print(f"{'#':>4} {'MOF ID':<60} {'Metals':>15} {'UFF Metal':>10} {'PSA_API':>9} {'VSA_API':>9}")
        print_separator("-")
        for i, (_, row) in enumerate(uff_union.iterrows(), start=1):
            psa_api = f"{row['PSA_API']:.4f}" if pd.notna(row["PSA_API"]) else "N/A"
            vsa_api = f"{row['VSA_API']:.4f}" if pd.notna(row["VSA_API"]) else "N/A"
            print(
                f"{i:>4} {row['mof_id']:<60} {row['metals']:>15} {row['uff_sensitive_metals']:>10} "
                f"{psa_api:>9} {vsa_api:>9}"
            )
        print_separator("-")

    # Summary of unknown metals (CIF not found)
    unknown = combined[(combined["list"] == "all_top_union") & (combined["source"] == "no_cif")]
    if len(unknown) > 0:
        print(f"\nWARNING: {len(unknown)} MOFs with no CIF file (metals unknown):")
        for _, row in unknown.iterrows():
            print(f"  - {row['mof_id']}")

    # Final verdict
    print("\n")
    print_separator("=")
    print("CONCLUSION")
    print_separator("=")
    uff_pct_union = 100.0 * n_uff_union / n_union if n_union > 0 else 0.0
    print(f"  {n_uff_union}/{n_union} ({uff_pct_union:.1f}%) of unique top candidates contain UFF-sensitive metals.")
    if n_uff_union > 0:
        uff_metal_list = set()
        for m in uff_union["uff_sensitive_metals"]:
            if m:
                uff_metal_list.update(m.split(","))
        print(f"  UFF-sensitive metals found: {', '.join(sorted(uff_metal_list))}")
        print("  These MOFs may have inflated GCMC adsorption values due to UFF force field artifacts.")
        print("  Consider: (1) noting in manuscript, (2) excluding from top-candidate discussion,")
        print("            or (3) re-running GCMC with UFF4MOF/DREIDING for validation.")
    else:
        print("  No UFF-sensitive metals detected. Pipeline results are not affected by this artifact.")


if __name__ == "__main__":
    main()
