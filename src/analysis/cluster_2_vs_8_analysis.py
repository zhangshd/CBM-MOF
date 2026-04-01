#!/usr/bin/env python
"""Structural feature analysis comparing Cluster 2 (PSA-dominant) vs Cluster 8 (VSA-dominant).

This script performs a comprehensive comparison of MOFs from two structurally distinct
clusters to understand why they favor different separation processes:
  - Cluster 2 (display label; internal idx=1): PSA-dominant (73% GCMC PSA benchmark-beaters)
  - Cluster 8 (display label; internal idx=7): VSA-dominant (74% GCMC VSA benchmark-beaters)

Steps:
  1. Identify and list MOFs in each cluster
  2. Compare geometric features (Zeo++ descriptors)
  3. Run MOFid decomposition for chemical analysis
  4. Analyze linker chemistry with RDKit
  5. Summarize metal node distributions
  6. Generate a comprehensive report

Usage:
    conda run -n alignn_env python src/analysis/cluster_2_vs_8_analysis.py
"""

import argparse
import logging
import os
import sys
import tempfile
import warnings
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress MOFid / openbabel warnings
warnings.filterwarnings("ignore")
os.environ["BABEL_DATADIR"] = ""  # suppress openbabel warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
REPO_ROOT = Path("/home/zhangsd/repos/CBM-MOF")
CLUSTER_CSV = REPO_ROOT / "results/cbm_screening/inference/umap_coordinates_descriptor_with_metrics_ml.csv"
GCMC_CSV = REPO_ROOT / "results/alignn/model_ep150/process_candidates/gcmc_vs_ml_comparison.csv"
ZEO_CSV = REPO_ROOT / "data/processed/RAC_and_zeo_features_deduplicated.csv"
CIF_DIR_PRIMARY = REPO_ROOT / "data/processed/integrated_cifs"
CIF_DIR_FALLBACK = REPO_ROOT / "results/cbm_screening/all_graphs_grids"
OUTPUT_DIR = REPO_ROOT / "results/alignn/model_ep150/structural_analysis"

ATC_CU_PSA_API = 0.457  # ATC-Cu GCMC benchmark
ATC_CU_VSA_API = 0.173

# Cluster mapping: internal index -> display label
CLUSTER_MAP = {1: 2, 7: 8}
CLUSTER_2_IDX = 1
CLUSTER_8_IDX = 7

# Key Zeo++ geometric columns
GEO_COLS = ["Di", "Df", "Dif", "rho", "VSA", "GSA", "VPOV", "GPOV", "POAV_vol_frac"]


def load_and_merge_data() -> pd.DataFrame:
    """Load cluster assignments, GCMC validation, and merge them."""
    cluster_df = pd.read_csv(CLUSTER_CSV)
    gcmc_df = pd.read_csv(GCMC_CSV)

    merged = cluster_df[["CifId", "cluster"]].merge(
        gcmc_df, left_on="CifId", right_on="mof_id", how="inner"
    )
    logger.info(f"Merged {len(merged)} validated MOFs with cluster assignments")
    return merged


def identify_cluster_mofs(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract MOFs belonging to Cluster 2 and Cluster 8."""
    c2 = merged[merged["cluster"] == CLUSTER_2_IDX].copy()
    c8 = merged[merged["cluster"] == CLUSTER_8_IDX].copy()

    # Add benchmark-beating flags
    c2["beats_psa"] = c2["gcmc_PSA_API_CH4"] > ATC_CU_PSA_API
    c2["beats_vsa"] = c2["gcmc_VSA_API_CH4"] > ATC_CU_VSA_API
    c8["beats_psa"] = c8["gcmc_PSA_API_CH4"] > ATC_CU_PSA_API
    c8["beats_vsa"] = c8["gcmc_VSA_API_CH4"] > ATC_CU_VSA_API

    logger.info(
        f"Cluster 2 (idx={CLUSTER_2_IDX}): {len(c2)} MOFs, "
        f"{c2['beats_psa'].sum()} PSA beaters ({c2['beats_psa'].mean()*100:.0f}%), "
        f"{c2['beats_vsa'].sum()} VSA beaters ({c2['beats_vsa'].mean()*100:.0f}%)"
    )
    logger.info(
        f"Cluster 8 (idx={CLUSTER_8_IDX}): {len(c8)} MOFs, "
        f"{c8['beats_psa'].sum()} PSA beaters ({c8['beats_psa'].mean()*100:.0f}%), "
        f"{c8['beats_vsa'].sum()} VSA beaters ({c8['beats_vsa'].mean()*100:.0f}%)"
    )
    return c2, c8


def geometric_feature_comparison(c2: pd.DataFrame, c8: pd.DataFrame) -> pd.DataFrame:
    """Compare Zeo++ geometric features between the two clusters."""
    zeo_df = pd.read_csv(ZEO_CSV, usecols=["name"] + GEO_COLS)
    zeo_df.rename(columns={"name": "CifId"}, inplace=True)

    c2_geo = c2[["CifId"]].merge(zeo_df, on="CifId", how="left")
    c8_geo = c8[["CifId"]].merge(zeo_df, on="CifId", how="left")

    logger.info(f"Geometric data matched: Cluster 2 = {c2_geo[GEO_COLS[0]].notna().sum()}/{len(c2)}, "
                f"Cluster 8 = {c8_geo[GEO_COLS[0]].notna().sum()}/{len(c8)}")

    # Compute summary statistics
    geo_labels = {
        "Di": "LCD (A)",
        "Df": "PLD (A)",
        "Dif": "LCD_free (A)",
        "rho": "Density (g/cm3)",
        "VSA": "Vol_SA (m2/cm3)",
        "GSA": "Grav_SA (m2/g)",
        "VPOV": "Pore_Vol (cm3/cm3)",
        "GPOV": "Grav_Pore_Vol (cm3/g)",
        "POAV_vol_frac": "Void_Fraction",
    }

    rows = []
    for col in GEO_COLS:
        label = geo_labels.get(col, col)
        c2_vals = c2_geo[col].dropna()
        c8_vals = c8_geo[col].dropna()
        rows.append({
            "Feature": label,
            "Cluster_2_mean": c2_vals.mean(),
            "Cluster_2_std": c2_vals.std(),
            "Cluster_2_min": c2_vals.min(),
            "Cluster_2_max": c2_vals.max(),
            "Cluster_2_median": c2_vals.median(),
            "Cluster_8_mean": c8_vals.mean(),
            "Cluster_8_std": c8_vals.std(),
            "Cluster_8_min": c8_vals.min(),
            "Cluster_8_max": c8_vals.max(),
            "Cluster_8_median": c8_vals.median(),
            "Cluster_2_n": len(c2_vals),
            "Cluster_8_n": len(c8_vals),
        })

    geo_summary = pd.DataFrame(rows)
    return geo_summary


def find_cif_path(mof_id: str) -> Path | None:
    """Locate the CIF file for a given MOF ID."""
    for cif_dir in [CIF_DIR_PRIMARY, CIF_DIR_FALLBACK]:
        cif_path = cif_dir / f"{mof_id}.cif"
        if cif_path.exists():
            return cif_path
    return None


def run_mofid_analysis(mof_ids: list[str], cluster_label: str) -> pd.DataFrame:
    """Run MOFid decomposition on a list of MOFs.

    Args:
        mof_ids: List of MOF identifiers.
        cluster_label: Label for logging (e.g. "Cluster 2").

    Returns:
        DataFrame with columns: mof_id, mofid, mofkey, smiles_nodes, smiles_linkers,
        topology, cat, metals, linkers_list, status.
    """
    from mofid.run_mofid import cif2mofid

    results = []
    total = len(mof_ids)
    for i, mof_id in enumerate(mof_ids):
        cif_path = find_cif_path(mof_id)
        if cif_path is None:
            logger.warning(f"[{cluster_label}] CIF not found: {mof_id}")
            results.append({"mof_id": mof_id, "status": "cif_not_found"})
            continue

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = cif2mofid(str(cif_path), output_path=tmpdir)

            mofid_str = result.get("mofid", "")
            topology = result.get("topology", "")
            cat = result.get("cat", "")

            # smiles_nodes and smiles_linkers are returned as lists by MOFid
            raw_nodes = result.get("smiles_nodes", [])
            raw_linkers = result.get("smiles_linkers", [])
            if isinstance(raw_nodes, str):
                raw_nodes = [s.strip() for s in raw_nodes.split(".") if s.strip()]
            if isinstance(raw_linkers, str):
                raw_linkers = [s.strip() for s in raw_linkers.split(".") if s.strip()]

            smiles_nodes_str = ".".join(raw_nodes) if raw_nodes else ""
            smiles_linkers_str = ".".join(raw_linkers) if raw_linkers else ""

            # Extract metal elements from all node SMILES
            metals = set()
            for node_smi in raw_nodes:
                metals.update(extract_metals_from_nodes(node_smi))
            metals = sorted(metals)

            # Linker SMILES list (already a list)
            linkers_list = [s.strip() for s in raw_linkers if s.strip()]

            results.append({
                "mof_id": mof_id,
                "mofid": mofid_str,
                "mofkey": result.get("mofkey", ""),
                "smiles_nodes": smiles_nodes_str,
                "smiles_linkers": smiles_linkers_str,
                "topology": topology,
                "cat": cat,
                "metals": ",".join(metals) if metals else "",
                "linkers_list": ";".join(linkers_list) if linkers_list else "",
                "status": "success",
            })
            if (i + 1) % 10 == 0 or i == total - 1:
                logger.info(f"[{cluster_label}] MOFid progress: {i+1}/{total}")
        except Exception as e:
            logger.warning(f"[{cluster_label}] MOFid failed for {mof_id}: {e}")
            results.append({"mof_id": mof_id, "status": f"error: {str(e)[:100]}"})

    return pd.DataFrame(results)


def extract_metals_from_nodes(smiles_nodes: str) -> list[str]:
    """Extract metal element symbols from MOFid node SMILES string."""
    if not smiles_nodes:
        return []
    # Match element symbols in square brackets (typical for metals in SMILES)
    # Also handle bare metal symbols
    metal_elements = set()
    # Common metals in MOFs
    known_metals = {
        "Li", "Na", "K", "Rb", "Cs",
        "Be", "Mg", "Ca", "Sr", "Ba",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
        "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
        "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi",
    }
    # Find bracketed atoms
    brackets = re.findall(r"\[([A-Z][a-z]?)", smiles_nodes)
    for atom in brackets:
        if atom in known_metals:
            metal_elements.add(atom)
    return sorted(metal_elements)


def analyze_linkers_rdkit(mofid_df: pd.DataFrame, cluster_label: str) -> pd.DataFrame:
    """Analyze linker chemistry using RDKit.

    For each unique linker SMILES, compute:
      - Heavy atom count
      - Number of aromatic rings
      - Functional groups (carboxylate, amine, hydroxyl, halogen, nitro, etc.)
      - Molecular weight
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    successful = mofid_df[mofid_df["status"] == "success"]
    all_linkers = []
    for _, row in successful.iterrows():
        if row["linkers_list"]:
            for smi in row["linkers_list"].split(";"):
                smi = smi.strip()
                if smi:
                    all_linkers.append({"mof_id": row["mof_id"], "smiles": smi})

    if not all_linkers:
        logger.warning(f"[{cluster_label}] No linkers found for RDKit analysis")
        return pd.DataFrame()

    linker_df = pd.DataFrame(all_linkers)
    logger.info(f"[{cluster_label}] Analyzing {len(linker_df)} linker instances "
                f"({linker_df['smiles'].nunique()} unique)")

    # Pre-compile functional group SMARTS patterns (static, no need to rebuild per molecule)
    fg_smarts = {
        "n_carboxylate": "[CX3](=O)[OX1,OX2H1]",  # -COOH or -COO-
        "n_amine": "[NX3;H2,H1;!$(NC=O)]",  # primary/secondary amine
        "n_hydroxyl": "[OX2H1;!$(OC=O)]",  # -OH (not carboxylic)
        "n_halogen": "[F,Cl,Br,I]",
        "n_nitro": "[NX3](=O)=O",
        "n_sulfonate": "[SX4](=O)(=O)[OX1,OX2H1]",
    }
    fg_compiled = {name: Chem.MolFromSmarts(smarts) for name, smarts in fg_smarts.items()}

    results = []
    for smi in linker_df["smiles"].unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Try with sanitization disabled
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    mol = None

        if mol is None:
            results.append({
                "smiles": smi,
                "heavy_atoms": None,
                "aromatic_rings": None,
                "mol_weight": None,
                "n_carboxylate": None,
                "n_amine": None,
                "n_hydroxyl": None,
                "n_halogen": None,
                "n_nitro": None,
                "n_sulfonate": None,
                "n_rings": None,
                "parseable": False,
            })
            continue

        heavy_atoms = mol.GetNumHeavyAtoms()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        mol_weight = Descriptors.MolWt(mol)

        fg_counts = {}
        for fg_name, pat in fg_compiled.items():
            if pat is not None:
                fg_counts[fg_name] = len(mol.GetSubstructMatches(pat))
            else:
                fg_counts[fg_name] = 0

        results.append({
            "smiles": smi,
            "heavy_atoms": heavy_atoms,
            "aromatic_rings": aromatic_rings,
            "mol_weight": mol_weight,
            "n_rings": n_rings,
            "parseable": True,
            **fg_counts,
        })

    linker_props = pd.DataFrame(results)

    # Merge back to get per-MOF linker properties
    linker_full = linker_df.merge(linker_props, on="smiles", how="left")
    return linker_full


def summarize_linker_chemistry(
    c2_linkers: pd.DataFrame, c8_linkers: pd.DataFrame
) -> pd.DataFrame:
    """Create a summary comparison of linker chemistry between clusters."""
    rows = []
    numeric_cols = [
        "heavy_atoms", "aromatic_rings", "mol_weight", "n_rings",
        "n_carboxylate", "n_amine", "n_hydroxyl", "n_halogen", "n_nitro", "n_sulfonate",
    ]

    for col in numeric_cols:
        c2_vals = c2_linkers[col].dropna() if col in c2_linkers.columns else pd.Series(dtype=float)
        c8_vals = c8_linkers[col].dropna() if col in c8_linkers.columns else pd.Series(dtype=float)
        if len(c2_vals) == 0 and len(c8_vals) == 0:
            continue
        rows.append({
            "Property": col,
            "Cluster_2_mean": c2_vals.mean() if len(c2_vals) else None,
            "Cluster_2_std": c2_vals.std() if len(c2_vals) else None,
            "Cluster_2_median": c2_vals.median() if len(c2_vals) else None,
            "Cluster_8_mean": c8_vals.mean() if len(c8_vals) else None,
            "Cluster_8_std": c8_vals.std() if len(c8_vals) else None,
            "Cluster_8_median": c8_vals.median() if len(c8_vals) else None,
        })

    return pd.DataFrame(rows)


def summarize_metals(mofid_df: pd.DataFrame) -> Counter:
    """Count metal occurrences from MOFid results."""
    metal_counter: Counter = Counter()
    successful = mofid_df[mofid_df["status"] == "success"]
    for _, row in successful.iterrows():
        if row["metals"]:
            for m in row["metals"].split(","):
                m = m.strip()
                if m:
                    metal_counter[m] += 1
    return metal_counter


def summarize_topologies(mofid_df: pd.DataFrame) -> Counter:
    """Count topology occurrences from MOFid results."""
    topo_counter: Counter = Counter()
    successful = mofid_df[mofid_df["status"] == "success"]
    for _, row in successful.iterrows():
        topo = row.get("topology", "")
        if topo and topo.strip():
            topo_counter[topo.strip()] += 1
    return topo_counter


def generate_report(
    c2: pd.DataFrame,
    c8: pd.DataFrame,
    geo_summary: pd.DataFrame,
    c2_mofid: pd.DataFrame,
    c8_mofid: pd.DataFrame,
    c2_linkers: pd.DataFrame,
    c8_linkers: pd.DataFrame,
    linker_summary: pd.DataFrame,
) -> str:
    """Generate a comprehensive Markdown report."""
    lines = []
    lines.append("# Structural Feature Analysis: Cluster 2 (PSA-dominant) vs Cluster 8 (VSA-dominant)")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Property | Cluster 2 (PSA) | Cluster 8 (VSA) |")
    lines.append("|----------|----------------|-----------------|")
    lines.append(f"| Display label | 2 | 8 |")
    lines.append(f"| Internal index | {CLUSTER_2_IDX} | {CLUSTER_8_IDX} |")
    lines.append(f"| Total MOFs | {len(c2)} | {len(c8)} |")
    lines.append(f"| PSA benchmark beaters | {c2['beats_psa'].sum()} ({c2['beats_psa'].mean()*100:.0f}%) "
                 f"| {c8['beats_psa'].sum()} ({c8['beats_psa'].mean()*100:.0f}%) |")
    lines.append(f"| VSA benchmark beaters | {c2['beats_vsa'].sum()} ({c2['beats_vsa'].mean()*100:.0f}%) "
                 f"| {c8['beats_vsa'].sum()} ({c8['beats_vsa'].mean()*100:.0f}%) |")
    lines.append(f"| Benchmark | ATC-Cu PSA API = {ATC_CU_PSA_API} | ATC-Cu VSA API = {ATC_CU_VSA_API} |")
    lines.append("")

    # Performance metrics
    lines.append("### Process Performance (GCMC-validated)")
    lines.append("")
    lines.append("| Metric | Cluster 2 mean (std) | Cluster 8 mean (std) |")
    lines.append("|--------|---------------------|---------------------|")
    for col, label in [
        ("gcmc_PSA_API_CH4", "PSA API"),
        ("gcmc_VSA_API_CH4", "VSA API"),
        ("gcmc_PSA_WC_CH4", "PSA WC CH4 (mol/kg)"),
        ("gcmc_VSA_WC_CH4", "VSA WC CH4 (mol/kg)"),
        ("gcmc_PSA_alpha_CH4_N2", "PSA alpha"),
        ("gcmc_VSA_alpha_CH4_N2", "VSA alpha"),
        ("QstCH4_gcmc", "Qst CH4 (kJ/mol)"),
        ("QstN2_gcmc", "Qst N2 (kJ/mol)"),
    ]:
        if col in c2.columns:
            c2v = c2[col].dropna()
            c8v = c8[col].dropna()
            lines.append(f"| {label} | {c2v.mean():.3f} ({c2v.std():.3f}) | {c8v.mean():.3f} ({c8v.std():.3f}) |")
    lines.append("")

    # Geometric features
    lines.append("## 2. Geometric Feature Comparison (Zeo++)")
    lines.append("")
    lines.append("| Feature | Cluster 2 mean (std) | Cluster 8 mean (std) | Delta (C2 - C8) |")
    lines.append("|---------|---------------------|---------------------|-----------------|")
    for _, row in geo_summary.iterrows():
        c2m = row["Cluster_2_mean"]
        c8m = row["Cluster_8_mean"]
        delta = c2m - c8m if pd.notna(c2m) and pd.notna(c8m) else None
        c2_str = f"{c2m:.3f} ({row['Cluster_2_std']:.3f})" if pd.notna(c2m) else "N/A"
        c8_str = f"{c8m:.3f} ({row['Cluster_8_std']:.3f})" if pd.notna(c8m) else "N/A"
        delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
        lines.append(f"| {row['Feature']} | {c2_str} | {c8_str} | {delta_str} |")
    lines.append("")

    # Key geometric insights
    lines.append("### Key Geometric Insights")
    lines.append("")
    if len(geo_summary) > 0:
        lcd_c2 = geo_summary.loc[geo_summary["Feature"] == "LCD (A)", "Cluster_2_mean"].values
        lcd_c8 = geo_summary.loc[geo_summary["Feature"] == "LCD (A)", "Cluster_8_mean"].values
        pld_c2 = geo_summary.loc[geo_summary["Feature"] == "PLD (A)", "Cluster_2_mean"].values
        pld_c8 = geo_summary.loc[geo_summary["Feature"] == "PLD (A)", "Cluster_8_mean"].values
        rho_c2 = geo_summary.loc[geo_summary["Feature"] == "Density (g/cm3)", "Cluster_2_mean"].values
        rho_c8 = geo_summary.loc[geo_summary["Feature"] == "Density (g/cm3)", "Cluster_8_mean"].values
        vsa_c2 = geo_summary.loc[geo_summary["Feature"] == "Vol_SA (m2/cm3)", "Cluster_2_mean"].values
        vsa_c8 = geo_summary.loc[geo_summary["Feature"] == "Vol_SA (m2/cm3)", "Cluster_8_mean"].values
        vf_c2 = geo_summary.loc[geo_summary["Feature"] == "Void_Fraction", "Cluster_2_mean"].values
        vf_c8 = geo_summary.loc[geo_summary["Feature"] == "Void_Fraction", "Cluster_8_mean"].values

        if len(lcd_c2) and len(lcd_c8):
            lines.append(f"- **LCD**: Cluster 2 = {lcd_c2[0]:.2f} A vs Cluster 8 = {lcd_c8[0]:.2f} A")
        if len(pld_c2) and len(pld_c8):
            lines.append(f"- **PLD**: Cluster 2 = {pld_c2[0]:.2f} A vs Cluster 8 = {pld_c8[0]:.2f} A")
        if len(rho_c2) and len(rho_c8):
            lines.append(f"- **Density**: Cluster 2 = {rho_c2[0]:.3f} g/cm3 vs Cluster 8 = {rho_c8[0]:.3f} g/cm3")
        if len(vsa_c2) and len(vsa_c8):
            lines.append(f"- **Vol SA**: Cluster 2 = {vsa_c2[0]:.1f} m2/cm3 vs Cluster 8 = {vsa_c8[0]:.1f} m2/cm3")
        if len(vf_c2) and len(vf_c8):
            lines.append(f"- **Void Fraction**: Cluster 2 = {vf_c2[0]:.3f} vs Cluster 8 = {vf_c8[0]:.3f}")
    lines.append("")

    # Metal node analysis
    lines.append("## 3. Metal Node Analysis (MOFid)")
    lines.append("")
    c2_metals = summarize_metals(c2_mofid)
    c8_metals = summarize_metals(c8_mofid)
    all_metals = sorted(set(c2_metals.keys()) | set(c8_metals.keys()))

    c2_success = (c2_mofid["status"] == "success").sum()
    c8_success = (c8_mofid["status"] == "success").sum()
    lines.append(f"MOFid success rate: Cluster 2 = {c2_success}/{len(c2_mofid)}, "
                 f"Cluster 8 = {c8_success}/{len(c8_mofid)}")
    lines.append("")
    lines.append("| Metal | Cluster 2 (count) | Cluster 8 (count) |")
    lines.append("|-------|-------------------|-------------------|")
    for metal in all_metals:
        c2_count = c2_metals.get(metal, 0)
        c8_count = c8_metals.get(metal, 0)
        lines.append(f"| {metal} | {c2_count} | {c8_count} |")
    lines.append("")

    # Topology analysis
    lines.append("## 4. Topology Analysis (MOFid)")
    lines.append("")
    c2_topos = summarize_topologies(c2_mofid)
    c8_topos = summarize_topologies(c8_mofid)
    all_topos = sorted(set(c2_topos.keys()) | set(c8_topos.keys()), key=lambda t: -(c2_topos.get(t, 0) + c8_topos.get(t, 0)))

    lines.append("| Topology | Cluster 2 (count) | Cluster 8 (count) |")
    lines.append("|----------|-------------------|-------------------|")
    for topo in all_topos[:15]:  # Top 15
        lines.append(f"| {topo} | {c2_topos.get(topo, 0)} | {c8_topos.get(topo, 0)} |")
    lines.append("")

    # Linker chemistry
    lines.append("## 5. Linker Chemistry Analysis (RDKit)")
    lines.append("")
    if len(linker_summary) > 0:
        lines.append("| Property | Cluster 2 mean (std) | Cluster 8 mean (std) |")
        lines.append("|----------|---------------------|---------------------|")
        for _, row in linker_summary.iterrows():
            c2_str = f"{row['Cluster_2_mean']:.2f} ({row['Cluster_2_std']:.2f})" if pd.notna(row["Cluster_2_mean"]) else "N/A"
            c8_str = f"{row['Cluster_8_mean']:.2f} ({row['Cluster_8_std']:.2f})" if pd.notna(row["Cluster_8_mean"]) else "N/A"
            lines.append(f"| {row['Property']} | {c2_str} | {c8_str} |")
        lines.append("")

    # Top linkers by frequency
    lines.append("### Most Common Linkers")
    lines.append("")
    for label, linker_df_part in [("Cluster 2", c2_linkers), ("Cluster 8", c8_linkers)]:
        if len(linker_df_part) == 0:
            continue
        parseable = linker_df_part[linker_df_part["parseable"] == True]
        if len(parseable) == 0:
            continue
        freq = parseable.groupby("smiles").agg(
            count=("mof_id", "count"),
            heavy_atoms=("heavy_atoms", "first"),
            aromatic_rings=("aromatic_rings", "first"),
            n_carboxylate=("n_carboxylate", "first"),
        ).sort_values("count", ascending=False).head(10)

        lines.append(f"**{label}** top linkers:")
        lines.append("")
        lines.append("| SMILES | Count | Heavy Atoms | Aromatic Rings | Carboxylates |")
        lines.append("|--------|-------|-------------|----------------|-------------|")
        for smi, row in freq.iterrows():
            smi_display = smi if len(smi) <= 60 else smi[:57] + "..."
            lines.append(f"| `{smi_display}` | {row['count']} | {row['heavy_atoms']:.0f} | "
                         f"{row['aromatic_rings']:.0f} | {row['n_carboxylate']:.0f} |")
        lines.append("")

    # Node SMILES summary
    lines.append("### Node SMILES Patterns")
    lines.append("")
    for label, mofid_part in [("Cluster 2", c2_mofid), ("Cluster 8", c8_mofid)]:
        successful = mofid_part[mofid_part["status"] == "success"]
        if len(successful) == 0:
            continue
        node_counter = Counter(successful["smiles_nodes"].dropna().tolist())
        lines.append(f"**{label}** top node SMILES (count):")
        lines.append("")
        for node_smi, cnt in node_counter.most_common(5):
            node_display = node_smi if len(str(node_smi)) <= 80 else str(node_smi)[:77] + "..."
            lines.append(f"- `{node_display}` ({cnt})")
        lines.append("")

    # Synthesis interpretation
    lines.append("## 6. Physical Interpretation: Why PSA vs VSA?")
    lines.append("")
    lines.append("### PSA Process (Cluster 2)")
    lines.append("")
    lines.append("PSA operates between high pressure (P_H = 10 bar) and atmospheric (P_L = 1 bar). "
                 "High working capacity requires:")
    lines.append("- **Large pore volumes and high void fractions** to accommodate more gas at elevated pressures")
    lines.append("- **Moderate Qst** (not too high) so that adsorbed gas can be released at P_L")
    lines.append("- **Sufficient surface area** for gas-framework interactions but not so strong that desorption is hindered")
    lines.append("")
    lines.append("### VSA Process (Cluster 8)")
    lines.append("")
    lines.append("VSA operates between atmospheric (P_H = 1 bar) and vacuum (P_L = 0.1 bar). "
                 "High selectivity at low pressures requires:")
    lines.append("- **Smaller, more confined pores** that enhance selectivity through size/shape effects")
    lines.append("- **Higher density** frameworks with stronger confinement effects")
    lines.append("- **Higher Qst** for preferential CH4 binding at low pressures")
    lines.append("- **Functional groups** that create specific binding sites for CH4 vs N2 discrimination")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Cluster 2 vs Cluster 8 structural analysis")
    parser.add_argument("--skip-mofid", action="store_true", help="Skip MOFid analysis (use cached results)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data and identify cluster MOFs
    logger.info("=" * 60)
    logger.info("Step 1: Identifying cluster MOFs")
    merged = load_and_merge_data()
    c2, c8 = identify_cluster_mofs(merged)

    # Step 2: Geometric feature comparison
    logger.info("=" * 60)
    logger.info("Step 2: Geometric feature comparison")
    geo_summary = geometric_feature_comparison(c2, c8)
    geo_path = OUTPUT_DIR / "cluster_2_vs_8_geometric_comparison.csv"
    geo_summary.to_csv(geo_path, index=False)
    logger.info(f"Saved geometric comparison to {geo_path}")
    print("\n--- Geometric Feature Summary ---")
    print(geo_summary.to_string(index=False))

    # Step 3: MOFid decomposition
    logger.info("=" * 60)
    logger.info("Step 3: MOFid decomposition")

    c2_mofid_path = OUTPUT_DIR / "cluster_2_mofid_results.csv"
    c8_mofid_path = OUTPUT_DIR / "cluster_8_mofid_results.csv"

    if args.skip_mofid and c2_mofid_path.exists() and c8_mofid_path.exists():
        logger.info("Loading cached MOFid results")
        c2_mofid = pd.read_csv(c2_mofid_path)
        c8_mofid = pd.read_csv(c8_mofid_path)
    else:
        c2_mofid = run_mofid_analysis(c2["CifId"].tolist(), "Cluster 2")
        c8_mofid = run_mofid_analysis(c8["CifId"].tolist(), "Cluster 8")
        c2_mofid.to_csv(c2_mofid_path, index=False)
        c8_mofid.to_csv(c8_mofid_path, index=False)
        logger.info(f"Saved MOFid results: {c2_mofid_path}, {c8_mofid_path}")

    c2_success = (c2_mofid["status"] == "success").sum()
    c8_success = (c8_mofid["status"] == "success").sum()
    logger.info(f"MOFid success: Cluster 2 = {c2_success}/{len(c2_mofid)}, "
                f"Cluster 8 = {c8_success}/{len(c8_mofid)}")

    # Step 4: Linker chemistry analysis
    logger.info("=" * 60)
    logger.info("Step 4: Linker chemistry analysis (RDKit)")
    c2_linkers = analyze_linkers_rdkit(c2_mofid, "Cluster 2")
    c8_linkers = analyze_linkers_rdkit(c8_mofid, "Cluster 8")

    linker_summary = summarize_linker_chemistry(c2_linkers, c8_linkers)
    linker_path = OUTPUT_DIR / "cluster_2_vs_8_linker_analysis.csv"
    linker_summary.to_csv(linker_path, index=False)
    logger.info(f"Saved linker analysis to {linker_path}")
    print("\n--- Linker Chemistry Summary ---")
    print(linker_summary.to_string(index=False))

    # Also save per-linker detail
    if len(c2_linkers) > 0:
        c2_linkers.to_csv(OUTPUT_DIR / "cluster_2_linkers_detail.csv", index=False)
    if len(c8_linkers) > 0:
        c8_linkers.to_csv(OUTPUT_DIR / "cluster_8_linkers_detail.csv", index=False)

    # Step 5: Metal node summary
    logger.info("=" * 60)
    logger.info("Step 5: Metal node analysis")
    c2_metals = summarize_metals(c2_mofid)
    c8_metals = summarize_metals(c8_mofid)
    print("\n--- Metal Distribution ---")
    print(f"Cluster 2: {dict(c2_metals.most_common())}")
    print(f"Cluster 8: {dict(c8_metals.most_common())}")

    # Step 6: Generate report
    logger.info("=" * 60)
    logger.info("Step 6: Generating comprehensive report")
    report = generate_report(c2, c8, geo_summary, c2_mofid, c8_mofid, c2_linkers, c8_linkers, linker_summary)
    report_path = OUTPUT_DIR / "cluster_2_vs_8_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Saved report to {report_path}")

    # Summary counts
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("cluster_2_vs_8_*")):
        logger.info(f"  {f.name}")
    for f in sorted(OUTPUT_DIR.glob("cluster_*_mofid_results*")):
        logger.info(f"  {f.name}")
    for f in sorted(OUTPUT_DIR.glob("cluster_*_linkers_detail*")):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
