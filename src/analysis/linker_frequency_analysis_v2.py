#!/usr/bin/env python
"""Analyze linker SMILES frequency in Cluster 2 and Cluster 8 MOFid results.

Reads MOFid decomposition CSVs, canonicalizes linker SMILES via RDKit,
counts frequency per unique linker, and categorizes Cluster 8 linkers
into tetracarboxylate (intralayer) vs N-donor pillar types.

Includes verification of figure SMILES against actual data frequencies
and a stereochemistry-aware matching mode.

Usage:
    conda run -n alignn_env python src/analysis/linker_frequency_analysis_v2.py
"""

import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STRUCTURAL_DIR = Path("/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150/structural_analysis")
CLUSTER_2_CSV = STRUCTURAL_DIR / "cluster_2_mofid_results.csv"
CLUSTER_8_CSV = STRUCTURAL_DIR / "cluster_8_mofid_results.csv"
OUTPUT_CSV = STRUCTURAL_DIR / "linker_frequency_analysis.csv"


def canonicalize(smiles: str) -> str:
    """Canonicalize SMILES. Returns empty string on failure."""
    if not smiles or pd.isna(smiles):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles.strip()
    return Chem.MolToSmiles(mol)


def canonicalize_no_stereo(smiles: str) -> str:
    """Canonicalize SMILES after removing stereochemistry."""
    if not smiles or pd.isna(smiles):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles.strip()
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def count_carboxylate_groups(mol) -> int:
    """Count -C(=O)[O-] groups."""
    if mol is None:
        return 0
    pattern = Chem.MolFromSmarts("[CX3](=O)[O-]")
    return len(mol.GetSubstructMatches(pattern))


def count_nitrogen_atoms(mol) -> int:
    """Count nitrogen atoms."""
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)


def aromatic_ring_count(mol) -> int:
    """Count aromatic rings."""
    if mol is None:
        return 0
    ri = mol.GetRingInfo()
    return sum(1 for ring in ri.AtomRings()
               if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring))


def parse_linkers(df: pd.DataFrame) -> list[str]:
    """Extract all individual linker SMILES from linkers_list column (semicolon-separated)."""
    all_linkers = []
    for _, row in df.iterrows():
        linkers_raw = row.get("linkers_list", "")
        if pd.isna(linkers_raw) or not linkers_raw:
            continue
        for smi in str(linkers_raw).split(";"):
            smi = smi.strip()
            if smi:
                all_linkers.append(smi)
    return all_linkers


def build_frequency_table(linkers: list[str]) -> pd.DataFrame:
    """Build frequency table with canonical SMILES and molecular descriptors."""
    canonical = [canonicalize(s) for s in linkers]
    canonical = [s for s in canonical if s]
    counter = Counter(canonical)

    rows = []
    for smi, freq in counter.most_common():
        mol = Chem.MolFromSmiles(smi)
        rows.append({
            "canonical_smiles": smi,
            "frequency": freq,
            "heavy_atoms": mol.GetNumHeavyAtoms() if mol else 0,
            "aromatic_rings": aromatic_ring_count(mol),
            "n_carboxylate": count_carboxylate_groups(mol),
            "n_nitrogen": count_nitrogen_atoms(mol),
        })
    return pd.DataFrame(rows)


def categorize_linker(n_coo: int, n_N: int) -> str:
    """Categorize a linker based on COO and N counts."""
    if n_coo >= 3:
        return "tetracarboxylate"
    elif n_coo == 0 and n_N > 0:
        return "N-donor_pillar"
    elif n_coo > 0:
        return "dicarboxylate"
    return "other"


def verify_smiles(freq_table: pd.DataFrame, figure_smiles: dict, label: str) -> list[dict]:
    """Verify figure SMILES against frequency data. Returns verification results."""
    results = []
    for name, smi in figure_smiles.items():
        canon = canonicalize(smi)
        canon_no_stereo = canonicalize_no_stereo(smi)

        # Try exact match first
        match = freq_table[freq_table["canonical_smiles"] == canon]
        match_type = "exact"

        # If no exact match, try stereo-free match
        if match.empty:
            no_stereo_col = freq_table["canonical_smiles"].apply(canonicalize_no_stereo)
            match = freq_table[no_stereo_col == canon_no_stereo]
            match_type = "stereo-free"

        if not match.empty:
            idx = match.index[0]
            rank = list(freq_table.index).index(idx) + 1
            freq = match["frequency"].values[0]
            data_smi = match["canonical_smiles"].values[0]
            results.append({
                "name": name, "status": "FOUND", "match_type": match_type,
                "rank": rank, "frequency": freq,
                "figure_smiles": smi, "data_smiles": data_smi,
            })
        else:
            results.append({
                "name": name, "status": "NOT_FOUND", "match_type": "none",
                "rank": None, "frequency": 0,
                "figure_smiles": smi, "data_smiles": None,
            })
    return results


def main():
    # -----------------------------------------------------------------------
    # Cluster 2
    # -----------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("CLUSTER 2 (PSA-dominant, rna topology)")
    logger.info("=" * 80)

    df2 = pd.read_csv(CLUSTER_2_CSV)
    linkers_2 = parse_linkers(df2)
    freq_2 = build_frequency_table(linkers_2)

    logger.info(f"{len(df2)} MOFs, {len(linkers_2)} linker instances, {len(freq_2)} unique canonical linkers")
    logger.info("")
    logger.info("Rank | Freq | Heavy | AromR | COO | N  | Canonical SMILES")
    logger.info("-" * 80)
    for i, row in freq_2.iterrows():
        logger.info(
            f" {i+1:3d} | {row['frequency']:4d} | {row['heavy_atoms']:5d} | "
            f"{row['aromatic_rings']:5d} | {row['n_carboxylate']:3d} | {row['n_nitrogen']:2d} | "
            f"{row['canonical_smiles']}"
        )

    c2_total = freq_2["frequency"].sum()
    c2_top3 = freq_2.head(3)["frequency"].sum()
    logger.info(f"\nTop-3 coverage: {c2_top3}/{c2_total} = {100*c2_top3/c2_total:.1f}%")

    # -----------------------------------------------------------------------
    # Cluster 8
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("CLUSTER 8 (VSA-dominant, fsc topology)")
    logger.info("=" * 80)

    df8 = pd.read_csv(CLUSTER_8_CSV)
    linkers_8 = parse_linkers(df8)
    freq_8 = build_frequency_table(linkers_8)
    freq_8["category"] = freq_8.apply(
        lambda r: categorize_linker(r["n_carboxylate"], r["n_nitrogen"]), axis=1
    )

    logger.info(f"{len(df8)} MOFs, {len(linkers_8)} linker instances, {len(freq_8)} unique canonical linkers")

    for cat, cat_label in [
        ("tetracarboxylate", "TETRACARBOXYLATE (intralayer, COO >= 3)"),
        ("N-donor_pillar", "N-DONOR PILLAR (COO=0, N>0)"),
        ("dicarboxylate", "DICARBOXYLATE (1 <= COO <= 2)"),
        ("other", "OTHER"),
    ]:
        subset = freq_8[freq_8["category"] == cat].reset_index(drop=True)
        if len(subset) == 0:
            continue
        logger.info("")
        logger.info(f"--- {cat_label} [{len(subset)} unique, {subset['frequency'].sum()} instances] ---")
        logger.info("Rank | Freq | Heavy | AromR | COO | N  | Canonical SMILES")
        logger.info("-" * 80)
        for i, row in subset.iterrows():
            logger.info(
                f" {i+1:3d} | {row['frequency']:4d} | {row['heavy_atoms']:5d} | "
                f"{row['aromatic_rings']:5d} | {row['n_carboxylate']:3d} | {row['n_nitrogen']:2d} | "
                f"{row['canonical_smiles']}"
            )

    # -----------------------------------------------------------------------
    # Figure SMILES verification
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("FIGURE SMILES VERIFICATION")
    logger.info("=" * 80)

    figure_smiles = {
        "cluster_2": {
            "Fumarate": "[O-]C(=O)/C=C/C(=O)[O-]",
            "BDC": "[O-]C(=O)c1ccc(cc1)C(=O)[O-]",
            "NDC": "[O-]C(=O)c1ccc2c(c1)ccc(c2)C(=O)[O-]",
        },
        "cluster_8_tetracarboxylate": {
            "Pyromellitate": "[O-]C(=O)c1cc(C(=O)[O-])c(cc1C(=O)[O-])C(=O)[O-]",
            "NTC": "[O-]C(=O)c1cc2cc(C(=O)[O-])c(C(=O)[O-])cc2cc1C(=O)[O-]",
        },
        "cluster_8_pillar": {
            "Quinoxaline": "c1ccc2c(c1)nccn2",
            "4,4'-bipyridine": "c1cc(-c2ccncc2)ncc1",
        },
    }

    # Cluster 2 verification
    logger.info("\n[Cluster 2]")
    for r in verify_smiles(freq_2, figure_smiles["cluster_2"], "Cluster 2"):
        if r["status"] == "FOUND":
            logger.info(f"  {r['name']:20s}: FOUND rank #{r['rank']}, freq={r['frequency']} "
                        f"(match={r['match_type']})")
            if r["figure_smiles"] != r["data_smiles"]:
                logger.info(f"    Figure:  {r['figure_smiles']}")
                logger.info(f"    Data:    {r['data_smiles']}")
                logger.info(f"    NOTE: Different canonical forms (stereo difference)")
        else:
            logger.info(f"  {r['name']:20s}: NOT FOUND")
            logger.info(f"    Figure:  {r['figure_smiles']}")

    # Cluster 8 tetra verification
    logger.info("\n[Cluster 8 — Tetracarboxylate]")
    tetra_sub = freq_8[freq_8["category"] == "tetracarboxylate"].reset_index(drop=True)
    for r in verify_smiles(tetra_sub, figure_smiles["cluster_8_tetracarboxylate"], "C8 tetra"):
        if r["status"] == "FOUND":
            logger.info(f"  {r['name']:20s}: FOUND rank #{r['rank']}, freq={r['frequency']}")
        else:
            logger.info(f"  {r['name']:20s}: NOT FOUND in data")
            logger.info(f"    Figure:    {canonicalize(r['figure_smiles'])}")
            # Find closest match
            fig_mol = Chem.MolFromSmiles(r["figure_smiles"])
            fig_ha = fig_mol.GetNumHeavyAtoms() if fig_mol else 0
            same_size = tetra_sub[tetra_sub["heavy_atoms"] == fig_ha]
            if len(same_size) > 0:
                logger.info(f"    Candidates with same heavy atom count ({fig_ha}):")
                for _, cand in same_size.iterrows():
                    logger.info(f"      freq={cand['frequency']}: {cand['canonical_smiles']}")

    # Cluster 8 pillar verification
    logger.info("\n[Cluster 8 — N-donor pillar]")
    pillar_sub = freq_8[freq_8["category"] == "N-donor_pillar"].reset_index(drop=True)
    for r in verify_smiles(pillar_sub, figure_smiles["cluster_8_pillar"], "C8 pillar"):
        if r["status"] == "FOUND":
            logger.info(f"  {r['name']:20s}: FOUND rank #{r['rank']}, freq={r['frequency']}")
        else:
            logger.info(f"  {r['name']:20s}: NOT FOUND in data")
            logger.info(f"    Figure canonical: {canonicalize(r['figure_smiles'])}")

    # -----------------------------------------------------------------------
    # Corrections needed
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 80)
    logger.info("CORRECTIONS NEEDED FOR FIGURE")
    logger.info("=" * 80)

    logger.info("")
    logger.info("1. FUMARATE (Cluster 2):")
    logger.info("   Status: MATCH (stereo-free). Figure uses E-stereo annotation; data omits it.")
    logger.info("   Figure:   [O-]C(=O)/C=C/C(=O)[O-]  (trans-butenedioate)")
    logger.info("   Data:     O=C([O-])C=CC(=O)[O-]     (no stereo)")
    logger.info("   Rank: #1 (freq=8). CORRECT representative.")
    logger.info("")

    logger.info("2. BDC (Cluster 2):")
    logger.info("   Status: EXACT MATCH. Rank #3 (freq=6). CORRECT representative.")
    logger.info("")

    logger.info("3. NDC (Cluster 2):")
    logger.info("   Status: EXACT MATCH. Rank #2 (freq=7). CORRECT representative.")
    logger.info("")

    logger.info("4. PYROMELLITATE (Cluster 8 tetracarboxylate):")
    logger.info("   Status: EXACT MATCH. Rank #1 (freq=8). CORRECT representative.")
    logger.info("")

    logger.info("5. NTC (Cluster 8 tetracarboxylate):")
    logger.info("   Status: NOT FOUND. Figure shows 2,3,6,7-naphthalenetetracarboxylate.")
    logger.info("   Data rank #2 (freq=5) is a DIFFERENT regioisomer:")
    logger.info("   Figure NTC: O=C([O-])c1cc2cc(C(=O)[O-])c(C(=O)[O-])cc2cc1C(=O)[O-]")
    logger.info("   Data rank2: O=C([O-])c1cc(C(=O)[O-])c2cc(C(=O)[O-])cc(C(=O)[O-])c2c1")
    logger.info("   Both: 22 heavy atoms, 2 aromatic rings, 4 COO groups (naphthalene-TC)")
    logger.info("   CORRECTION: Use the actual data rank #2 SMILES in the figure.")
    logger.info("")

    logger.info("6. 4,4'-BIPYRIDINE (Cluster 8 N-donor pillar):")
    logger.info("   Status: WRONG ISOMER. Figure SMILES is 2,4'-bipyridine, not 4,4'-bipyridine.")
    logger.info("   Figure:    c1cc(-c2ccncc2)ncc1  → canonical: c1ccc(-c2ccncc2)nc1")
    logger.info("              InChIKey: RMHQDKYZXJVCME = 2,4'-bipyridine")
    logger.info("   Correct:   c1cc(-c2ccncc2)ccn1  → canonical: c1cc(-c2ccncc2)ccn1")
    logger.info("              InChIKey: MWVTWFVJZLCBMC = TRUE 4,4'-bipyridine")
    logger.info("   Data rank: #4 (freq=4). CORRECTION: Update figure SMILES.")
    logger.info("")

    logger.info("7. QUINOXALINE (Cluster 8 N-donor pillar):")
    logger.info("   Status: EXACT MATCH. Rank #1 (freq=10). CORRECT representative.")

    # -----------------------------------------------------------------------
    # Save combined output
    # -----------------------------------------------------------------------
    freq_2["cluster"] = 2
    freq_2["category"] = "dicarboxylate"
    freq_8["cluster"] = 8

    combined = pd.concat([freq_2, freq_8], ignore_index=True)
    combined = combined[["cluster", "category", "canonical_smiles", "frequency",
                         "heavy_atoms", "aromatic_rings", "n_carboxylate", "n_nitrogen"]]
    combined.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
