#!/usr/bin/env python
"""Chemical feature analysis: PSA-only vs VSA-only vs Both benchmark-beating MOFs.

Compare structural and chemical characteristics of MOFs that beat the ATC-Cu GCMC
benchmark in PSA-only, VSA-only, or both separation scenarios.

Steps:
  1. Load GCMC validation data and classify MOFs into PSA-only / VSA-only / Both
  2. Parse chemical features from MOF naming convention (metal, topology, linker codes)
  3. Compare Zeo++ geometric features (density, LCD, PLD, pore volume, etc.)
  4. Run MOFid decomposition for detailed chemical analysis (if available)
  5. Analyze linker chemistry with RDKit (functional groups, aromaticity, etc.)
  6. Compare adsorption metrics (Qst, selectivity, uptake)
  7. Generate summary statistics and export results

Usage:
    conda run -n alignn_env python src/analysis/psa_vs_vsa_beaters_analysis.py
"""

import argparse
import logging
import os
import re
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["BABEL_DATADIR"] = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
REPO_ROOT = Path("/home/zhangsd/repos/CBM-MOF")
GCMC_CSV = REPO_ROOT / "results/alignn/model_ep150/process_candidates/gcmc_vs_ml_comparison.csv"
ZEO_CSV = REPO_ROOT / "data/processed/RAC_and_zeo_features_deduplicated.csv"
CIF_DIR_PRIMARY = REPO_ROOT / "data/processed/integrated_cifs"
CIF_DIR_FALLBACK = REPO_ROOT / "results/cbm_screening/all_graphs_grids"
OUTPUT_DIR = REPO_ROOT / "results/alignn/model_ep150/structural_analysis/psa_vs_vsa_beaters"

# Existing MOFid results that may cover some of our MOFs
EXISTING_MOFID_C2 = REPO_ROOT / "results/alignn/model_ep150/structural_analysis/cluster_2_mofid_results.csv"
EXISTING_MOFID_C8 = REPO_ROOT / "results/alignn/model_ep150/structural_analysis/cluster_8_mofid_results.csv"

# ATC-Cu GCMC benchmarks
ATC_CU_PSA_API = 0.4574
ATC_CU_VSA_API = 0.1729

# Key Zeo++ geometric columns
GEO_COLS = ["Di", "Df", "Dif", "rho", "VSA", "GSA", "VPOV", "GPOV", "POAV_vol_frac"]
GEO_LABELS = {
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

# Known metals in MOFs
KNOWN_METALS = {
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi",
}

# ARC naming convention: metal code -> element mapping
# From ARC-MOF database documentation
ARC_METAL_MAP = {
    "m1": "Zn",    # Zn-based
    "m2": "Cu",    # Cu-based
    "m3": "Zn",    # Zn paddlewheel
    "m4": "Cu",    # Cu paddlewheel
    "m5": "Zr",    # Zr-based
    "m6": "Al",    # Al-based
    "m7": "Fe",    # Fe-based
    "m8": "Co",    # Co-based
    "m9": "Ni",    # Ni-based
    "m10": "Mn",   # Mn-based
    "m11": "Cd",   # Cd-based
    "m12": "In",   # In-based
    "m13": "V",    # V-based
    "m14": "Cr",   # Cr-based
    "m15": "Mg",   # Mg-based
    "m16": "Ti",   # Ti-based
    "m17": "Zr",   # Zr cluster
    "m18": "Ca",   # Ca-based
}


def classify_mof_source(mof_id: str) -> str:
    """Classify MOF as experimental or hypothetical based on naming convention."""
    if mof_id.startswith("CoRE-") or mof_id.startswith("MOSAEC-"):
        return "exp"
    if mof_id.startswith("ARC-DB12-") or mof_id.startswith("ARC-DB14-"):
        return "exp"
    return "hypo"


def parse_arc_name(mof_id: str) -> dict:
    """Parse ARC-DB0 naming convention to extract metal, linker, functional group, and topology codes.

    Format: ARC-DB0-m{metal}_o{linker1}_o{linker2}_f{func}_{topo}[.sym.N]_repeat
    """
    info = {"metal_code": None, "metal_element": None, "linker_codes": [], "func_code": None, "topology": None}
    if not mof_id.startswith("ARC-DB0-"):
        return info

    # Strip prefix and suffix
    core = mof_id.replace("ARC-DB0-", "")
    core = re.sub(r"_repeat$", "", core)
    # Remove symmetry suffix like .sym.1, .sym.2
    core = re.sub(r"\.sym\.\d+$", "", core)

    parts = core.split("_")
    linker_codes = []
    for part in parts:
        if part.startswith("m") and part[1:].isdigit():
            info["metal_code"] = part
            info["metal_element"] = ARC_METAL_MAP.get(part, f"?({part})")
        elif part.startswith("o") and part[1:].isdigit():
            linker_codes.append(part)
        elif part.startswith("f") and part[1:].replace("_", "").isdigit():
            info["func_code"] = part
        else:
            # Likely topology code (e.g. fsc, dia, pcu, bcu, etc.)
            if info["topology"] is None:
                info["topology"] = part

    info["linker_codes"] = linker_codes
    return info


def parse_core_name(mof_id: str) -> dict:
    """Parse CoRE-YYYY[Metal][topo]D[ASR]N naming."""
    info = {"metal_element": None, "topology": None}
    m = re.match(r"CoRE-\d{4}\[(\w+)\]\[(\w+)\]", mof_id)
    if m:
        info["metal_element"] = m.group(1)
        info["topology"] = m.group(2)
    return info


def parse_mosaec_name(mof_id: str) -> dict:
    """Extract refcode from MOSAEC naming."""
    return {"refcode": mof_id.replace("MOSAEC-", "").replace("_full_REPEAT", "").replace("_repeat", "")}


def parse_arc_db1_name(mof_id: str) -> dict:
    """Parse ARC-DB1 naming: ARC-DB1-{formula}-{linker_info}_repeat."""
    info = {"metal_element": None, "formula": None}
    m = re.match(r"ARC-DB1-(\w+?)-(.*?)_repeat", mof_id)
    if m:
        info["formula"] = m.group(1)
        # Extract metal from formula (e.g., Al2O6 -> Al)
        metals = re.findall(r"([A-Z][a-z]?)\d*", m.group(1))
        for metal in metals:
            if metal in KNOWN_METALS:
                info["metal_element"] = metal
                break
    return info


def parse_arc_db12_name(mof_id: str) -> dict:
    """Parse ARC-DB12 naming: ARC-DB12-REFCODE_SL_repeat."""
    info = {"refcode": None}
    m = re.match(r"ARC-DB12-(.+?)(?:_SL)?_repeat", mof_id)
    if m:
        info["refcode"] = m.group(1)
    return info


def extract_metals_from_smiles(smiles: str) -> list[str]:
    """Extract metal element symbols from MOFid node SMILES string."""
    if not smiles:
        return []
    brackets = re.findall(r"\[([A-Z][a-z]?)", smiles)
    return sorted({atom for atom in brackets if atom in KNOWN_METALS})


# =============================================================================
# Step 1: Data loading and classification
# =============================================================================
def load_and_classify() -> pd.DataFrame:
    """Load GCMC validation data and classify MOFs into PSA/VSA beater groups."""
    df = pd.read_csv(GCMC_CSV)
    logger.info(f"Loaded {len(df)} MOFs from GCMC validation")

    # Classify benchmark-beating
    df["beats_psa"] = df["gcmc_PSA_API_CH4"] >= ATC_CU_PSA_API
    df["beats_vsa"] = df["gcmc_VSA_API_CH4"] >= ATC_CU_VSA_API

    # Determine group
    df["beater_group"] = "Neither"
    df.loc[df["beats_psa"] & ~df["beats_vsa"], "beater_group"] = "PSA-only"
    df.loc[~df["beats_psa"] & df["beats_vsa"], "beater_group"] = "VSA-only"
    df.loc[df["beats_psa"] & df["beats_vsa"], "beater_group"] = "Both"

    # Source classification
    df["source"] = df["mof_id"].apply(classify_mof_source)

    # Summary
    for grp in ["PSA-only", "VSA-only", "Both", "Neither"]:
        subset = df[df["beater_group"] == grp]
        n_exp = (subset["source"] == "exp").sum()
        n_hypo = (subset["source"] == "hypo").sum()
        logger.info(f"  {grp}: {len(subset)} MOFs ({n_exp} exp, {n_hypo} hypo)")

    return df


# =============================================================================
# Step 2: Parse chemical features from naming convention
# =============================================================================
def parse_naming_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse metal, topology, and linker codes from MOF naming conventions."""
    records = []
    for _, row in df.iterrows():
        mof_id = row["mof_id"]
        rec = {"mof_id": mof_id, "beater_group": row["beater_group"]}

        if mof_id.startswith("ARC-DB0-"):
            info = parse_arc_name(mof_id)
            rec["metal_code"] = info["metal_code"]
            rec["metal_from_name"] = info["metal_element"]
            rec["topology_from_name"] = info["topology"]
            rec["linker_codes"] = ";".join(info["linker_codes"]) if info["linker_codes"] else ""
            rec["func_code"] = info["func_code"]
            rec["db_source"] = "ARC-DB0"
        elif mof_id.startswith("ARC-DB1-"):
            info = parse_arc_db1_name(mof_id)
            rec["metal_from_name"] = info["metal_element"]
            rec["db_source"] = "ARC-DB1"
        elif mof_id.startswith("ARC-DB12-"):
            info = parse_arc_db12_name(mof_id)
            rec["db_source"] = "ARC-DB12"
        elif mof_id.startswith("CoRE-"):
            info = parse_core_name(mof_id)
            rec["metal_from_name"] = info["metal_element"]
            rec["topology_from_name"] = info["topology"]
            rec["db_source"] = "CoRE"
        elif mof_id.startswith("MOSAEC-"):
            info = parse_mosaec_name(mof_id)
            rec["db_source"] = "MOSAEC"
        else:
            rec["db_source"] = "other"

        records.append(rec)

    naming_df = pd.DataFrame(records)
    return naming_df


# =============================================================================
# Step 3: Zeo++ geometric feature comparison
# =============================================================================
def compare_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compare Zeo++ geometric features across beater groups."""
    zeo_df = pd.read_csv(ZEO_CSV, usecols=["name"] + GEO_COLS)
    zeo_df.rename(columns={"name": "mof_id"}, inplace=True)

    merged = df[["mof_id", "beater_group"]].merge(zeo_df, on="mof_id", how="left")

    matched = merged[GEO_COLS[0]].notna().sum()
    logger.info(f"Zeo++ geometric data matched: {matched}/{len(df)} MOFs")

    rows = []
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = merged[merged["beater_group"] == grp]
        for col in GEO_COLS:
            vals = subset[col].dropna()
            rows.append({
                "Group": grp,
                "Feature": GEO_LABELS.get(col, col),
                "Feature_key": col,
                "n": len(vals),
                "mean": vals.mean() if len(vals) > 0 else np.nan,
                "std": vals.std() if len(vals) > 0 else np.nan,
                "median": vals.median() if len(vals) > 0 else np.nan,
                "min": vals.min() if len(vals) > 0 else np.nan,
                "max": vals.max() if len(vals) > 0 else np.nan,
                "q25": vals.quantile(0.25) if len(vals) > 0 else np.nan,
                "q75": vals.quantile(0.75) if len(vals) > 0 else np.nan,
            })

    geo_summary = pd.DataFrame(rows)
    return geo_summary, merged


# =============================================================================
# Step 4: MOFid decomposition (try existing results first, then run new)
# =============================================================================
def load_existing_mofid() -> pd.DataFrame:
    """Load existing MOFid results from cluster 2 and 8 analyses."""
    frames = []
    for path in [EXISTING_MOFID_C2, EXISTING_MOFID_C8]:
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
            logger.info(f"Loaded {len(df)} existing MOFid results from {path.name}")
    if frames:
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["mof_id"])
        return combined
    return pd.DataFrame()


def find_cif_path(mof_id: str) -> Path | None:
    """Locate the CIF file for a given MOF ID."""
    for cif_dir in [CIF_DIR_PRIMARY, CIF_DIR_FALLBACK]:
        cif_path = cif_dir / f"{mof_id}.cif"
        if cif_path.exists():
            return cif_path
    return None


def run_mofid_for_missing(mof_ids: list[str], existing_df: pd.DataFrame) -> pd.DataFrame:
    """Run MOFid only for MOFs not already in existing results."""
    if len(existing_df) > 0:
        already_done = set(existing_df["mof_id"].tolist())
        missing = [m for m in mof_ids if m not in already_done]
    else:
        missing = list(mof_ids)

    if not missing:
        logger.info("All MOFs already have MOFid results")
        return existing_df[existing_df["mof_id"].isin(mof_ids)]

    logger.info(f"Running MOFid for {len(missing)} missing MOFs (out of {len(mof_ids)} total)")

    try:
        from mofid.run_mofid import cif2mofid
    except ImportError:
        logger.warning("MOFid not available. Using existing results + name-based parsing only.")
        return existing_df[existing_df["mof_id"].isin(mof_ids)]

    results = []
    for i, mof_id in enumerate(missing):
        cif_path = find_cif_path(mof_id)
        if cif_path is None:
            logger.warning(f"CIF not found: {mof_id}")
            results.append({"mof_id": mof_id, "status": "cif_not_found"})
            continue

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = cif2mofid(str(cif_path), output_path=tmpdir)

            raw_nodes = result.get("smiles_nodes", [])
            raw_linkers = result.get("smiles_linkers", [])
            if isinstance(raw_nodes, str):
                raw_nodes = [s.strip() for s in raw_nodes.split(".") if s.strip()]
            if isinstance(raw_linkers, str):
                raw_linkers = [s.strip() for s in raw_linkers.split(".") if s.strip()]

            metals = set()
            for node_smi in raw_nodes:
                metals.update(extract_metals_from_smiles(node_smi))
            metals = sorted(metals)

            linkers_list = [s.strip() for s in raw_linkers if s.strip()]

            results.append({
                "mof_id": mof_id,
                "mofid": result.get("mofid", ""),
                "mofkey": result.get("mofkey", ""),
                "smiles_nodes": ".".join(raw_nodes) if raw_nodes else "",
                "smiles_linkers": ".".join(raw_linkers) if raw_linkers else "",
                "topology": result.get("topology", ""),
                "cat": result.get("cat", ""),
                "metals": ",".join(metals),
                "linkers_list": ";".join(linkers_list),
                "status": "success",
            })
            if (i + 1) % 10 == 0 or i == len(missing) - 1:
                logger.info(f"MOFid progress: {i+1}/{len(missing)}")
        except Exception as e:
            logger.warning(f"MOFid failed for {mof_id}: {e}")
            results.append({"mof_id": mof_id, "status": f"error: {str(e)[:100]}"})

    new_df = pd.DataFrame(results)

    # Combine with existing results for the target MOFs
    existing_subset = existing_df[existing_df["mof_id"].isin(mof_ids)]
    combined = pd.concat([existing_subset, new_df], ignore_index=True).drop_duplicates(subset=["mof_id"])
    return combined


# =============================================================================
# Step 5: RDKit linker analysis
# =============================================================================
def analyze_linkers_rdkit(mofid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze linker chemistry using RDKit.

    Returns:
        linker_props: Per-unique-linker properties (heavy atoms, aromatic rings, functional groups)
        mof_linker_map: Per-MOF linker usage
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        logger.warning("RDKit not available, skipping linker analysis")
        return pd.DataFrame(), pd.DataFrame()

    successful = mofid_df[mofid_df.get("status", "unknown") == "success"]
    if len(successful) == 0:
        logger.warning("No successful MOFid results for RDKit analysis")
        return pd.DataFrame(), pd.DataFrame()

    # Collect all linker instances
    mof_linker_records = []
    for _, row in successful.iterrows():
        linkers_str = row.get("linkers_list", "")
        if not linkers_str or pd.isna(linkers_str):
            continue
        for smi in str(linkers_str).split(";"):
            smi = smi.strip()
            if smi:
                mof_linker_records.append({"mof_id": row["mof_id"], "smiles": smi})

    if not mof_linker_records:
        return pd.DataFrame(), pd.DataFrame()

    mof_linker_df = pd.DataFrame(mof_linker_records)
    logger.info(f"Analyzing {len(mof_linker_df)} linker instances ({mof_linker_df['smiles'].nunique()} unique)")

    # Functional group SMARTS
    fg_smarts = {
        "n_carboxylate": "[CX3](=O)[OX1,OX2H1]",
        "n_amine": "[NX3;H2,H1;!$(NC=O)]",
        "n_hydroxyl": "[OX2H1;!$(OC=O)]",
        "n_halogen": "[F,Cl,Br,I]",
        "n_nitro": "[NX3](=O)=O",
        "n_sulfonate": "[SX4](=O)(=O)[OX1,OX2H1]",
        "n_nitrogen_any": "[#7]",
        "n_sulfur_any": "[#16]",
    }
    fg_compiled = {name: Chem.MolFromSmarts(smarts) for name, smarts in fg_smarts.items()}

    # Analyze unique linkers
    linker_results = []
    for smi in mof_linker_df["smiles"].unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    mol = None

        if mol is None:
            linker_results.append({"smiles": smi, "parseable": False})
            continue

        props = {
            "smiles": smi,
            "parseable": True,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "mol_weight": Descriptors.MolWt(mol),
            "n_rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        }
        for fg_name, fg_pat in fg_compiled.items():
            if fg_pat is not None:
                props[fg_name] = len(mol.GetSubstructMatches(fg_pat))
            else:
                props[fg_name] = 0

        # Classify linker type
        n_carb = props["n_carboxylate"]
        has_nitrogen = props["n_nitrogen_any"] > 0
        if n_carb >= 3:
            props["linker_class"] = "tricarboxylate+"
        elif n_carb == 2:
            props["linker_class"] = "dicarboxylate"
        elif n_carb == 1:
            props["linker_class"] = "monocarboxylate"
        elif has_nitrogen and n_carb == 0:
            props["linker_class"] = "N-donor"
        else:
            props["linker_class"] = "other"

        linker_results.append(props)

    linker_props = pd.DataFrame(linker_results)
    return linker_props, mof_linker_df


# =============================================================================
# Step 6: Adsorption metrics comparison
# =============================================================================
def compare_adsorption_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compare GCMC adsorption metrics across beater groups."""
    metrics_cols = [
        "gcmc_AdsCH4_10kPa", "gcmc_AdsCH4_100kPa", "gcmc_AdsCH4_1000kPa",
        "gcmc_AdsN2_10kPa", "gcmc_AdsN2_100kPa", "gcmc_AdsN2_1000kPa",
        "QstCH4_gcmc", "QstN2_gcmc",
        "gcmc_PSA_WC_CH4", "gcmc_PSA_WC_N2", "gcmc_PSA_alpha_CH4_N2", "gcmc_PSA_API_CH4",
        "gcmc_VSA_WC_CH4", "gcmc_VSA_WC_N2", "gcmc_VSA_alpha_CH4_N2", "gcmc_VSA_API_CH4",
    ]
    # Derived metrics
    df = df.copy()
    df["QstCH4_minus_QstN2"] = df["QstCH4_gcmc"] - df["QstN2_gcmc"]
    df["alpha_PSA_over_VSA"] = df["gcmc_PSA_alpha_CH4_N2"] / df["gcmc_VSA_alpha_CH4_N2"].replace(0, np.nan)
    metrics_cols.extend(["QstCH4_minus_QstN2", "alpha_PSA_over_VSA"])

    metric_labels = {
        "gcmc_AdsCH4_10kPa": "CH4@10kPa (mmol/g)",
        "gcmc_AdsCH4_100kPa": "CH4@100kPa (mmol/g)",
        "gcmc_AdsCH4_1000kPa": "CH4@1000kPa (mmol/g)",
        "gcmc_AdsN2_10kPa": "N2@10kPa (mmol/g)",
        "gcmc_AdsN2_100kPa": "N2@100kPa (mmol/g)",
        "gcmc_AdsN2_1000kPa": "N2@1000kPa (mmol/g)",
        "QstCH4_gcmc": "Qst_CH4 (kJ/mol)",
        "QstN2_gcmc": "Qst_N2 (kJ/mol)",
        "QstCH4_minus_QstN2": "delta_Qst (kJ/mol)",
        "gcmc_PSA_WC_CH4": "PSA_WC_CH4 (mmol/g)",
        "gcmc_PSA_WC_N2": "PSA_WC_N2 (mmol/g)",
        "gcmc_PSA_alpha_CH4_N2": "PSA_alpha",
        "gcmc_PSA_API_CH4": "PSA_API",
        "gcmc_VSA_WC_CH4": "VSA_WC_CH4 (mmol/g)",
        "gcmc_VSA_WC_N2": "VSA_WC_N2 (mmol/g)",
        "gcmc_VSA_alpha_CH4_N2": "VSA_alpha",
        "gcmc_VSA_API_CH4": "VSA_API",
        "alpha_PSA_over_VSA": "PSA_alpha / VSA_alpha",
    }

    rows = []
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = df[df["beater_group"] == grp]
        for col in metrics_cols:
            if col not in df.columns:
                continue
            vals = subset[col].dropna()
            rows.append({
                "Group": grp,
                "Metric": metric_labels.get(col, col),
                "Metric_key": col,
                "n": len(vals),
                "mean": vals.mean() if len(vals) > 0 else np.nan,
                "std": vals.std() if len(vals) > 0 else np.nan,
                "median": vals.median() if len(vals) > 0 else np.nan,
                "min": vals.min() if len(vals) > 0 else np.nan,
                "max": vals.max() if len(vals) > 0 else np.nan,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Step 7: Aggregate and report
# =============================================================================
def build_metal_distribution(naming_df: pd.DataFrame, mofid_df: pd.DataFrame) -> pd.DataFrame:
    """Build metal distribution table by combining naming + MOFid info."""
    records = []
    for _, row in naming_df.iterrows():
        mof_id = row["mof_id"]
        grp = row["beater_group"]

        # Prefer MOFid metals, fall back to name-parsed metal
        metal = None
        if len(mofid_df) > 0:
            mofid_row = mofid_df[mofid_df["mof_id"] == mof_id]
            if len(mofid_row) > 0:
                metals_str = mofid_row.iloc[0].get("metals", "")
                if metals_str and not pd.isna(metals_str):
                    metal = str(metals_str)

        if not metal:
            metal = row.get("metal_from_name")
            if metal and not pd.isna(metal):
                metal = str(metal)
            else:
                metal = "Unknown"

        records.append({"mof_id": mof_id, "beater_group": grp, "metal": metal})

    metal_df = pd.DataFrame(records)

    # Distribution table
    dist_rows = []
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = metal_df[metal_df["beater_group"] == grp]
        total = len(subset)
        for metal, count in Counter(subset["metal"]).most_common():
            dist_rows.append({
                "Group": grp,
                "Metal": metal,
                "Count": count,
                "Fraction": count / total if total > 0 else 0,
            })

    return pd.DataFrame(dist_rows)


def build_topology_distribution(naming_df: pd.DataFrame, mofid_df: pd.DataFrame) -> pd.DataFrame:
    """Build topology distribution table."""
    records = []
    for _, row in naming_df.iterrows():
        mof_id = row["mof_id"]
        grp = row["beater_group"]

        topo = None
        # Prefer MOFid topology
        if len(mofid_df) > 0:
            mofid_row = mofid_df[mofid_df["mof_id"] == mof_id]
            if len(mofid_row) > 0:
                topo_str = mofid_row.iloc[0].get("topology", "")
                if topo_str and not pd.isna(topo_str):
                    topo = str(topo_str)

        if not topo:
            topo = row.get("topology_from_name")
            if topo and not pd.isna(topo):
                topo = str(topo)
            else:
                topo = "Unknown"

        records.append({"mof_id": mof_id, "beater_group": grp, "topology": topo})

    topo_df = pd.DataFrame(records)

    dist_rows = []
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = topo_df[topo_df["beater_group"] == grp]
        total = len(subset)
        for topo, count in Counter(subset["topology"]).most_common():
            dist_rows.append({
                "Group": grp,
                "Topology": topo,
                "Count": count,
                "Fraction": count / total if total > 0 else 0,
            })

    return pd.DataFrame(dist_rows)


def build_linker_distribution_by_group(mof_linker_df: pd.DataFrame, linker_props: pd.DataFrame,
                                        beater_df: pd.DataFrame) -> pd.DataFrame:
    """Build linker frequency distribution per beater group."""
    if len(mof_linker_df) == 0:
        return pd.DataFrame()

    # Merge beater group info
    merged = mof_linker_df.merge(beater_df[["mof_id", "beater_group"]], on="mof_id", how="left")

    rows = []
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = merged[merged["beater_group"] == grp]
        total_mofs = subset["mof_id"].nunique()
        for smi, count in Counter(subset["smiles"]).most_common(15):
            # Get linker properties
            props = linker_props[linker_props["smiles"] == smi]
            linker_class = props.iloc[0]["linker_class"] if len(props) > 0 and "linker_class" in props.columns else ""
            n_carb = props.iloc[0].get("n_carboxylate", 0) if len(props) > 0 else 0
            aromatic = props.iloc[0].get("aromatic_rings", 0) if len(props) > 0 else 0
            heavy = props.iloc[0].get("heavy_atoms", 0) if len(props) > 0 else 0
            rows.append({
                "Group": grp,
                "SMILES": smi,
                "Count": count,
                "Fraction_of_group": count / total_mofs if total_mofs > 0 else 0,
                "Linker_class": linker_class,
                "Carboxylates": n_carb,
                "Aromatic_rings": aromatic,
                "Heavy_atoms": heavy,
            })

    return pd.DataFrame(rows)


def build_linker_class_summary(mof_linker_df: pd.DataFrame, linker_props: pd.DataFrame,
                                beater_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize linker classes per beater group."""
    if len(mof_linker_df) == 0 or len(linker_props) == 0:
        return pd.DataFrame()

    # Merge linker classes into mof_linker_df
    merged = mof_linker_df.merge(linker_props[["smiles", "linker_class", "aromatic_rings", "heavy_atoms"]],
                                  on="smiles", how="left")
    merged = merged.merge(beater_df[["mof_id", "beater_group"]], on="mof_id", how="left")

    rows = []
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = merged[merged["beater_group"] == grp]
        total = len(subset)
        for cls, count in Counter(subset["linker_class"]).most_common():
            cls_data = subset[subset["linker_class"] == cls]
            rows.append({
                "Group": grp,
                "Linker_class": cls,
                "Count": count,
                "Fraction": count / total if total > 0 else 0,
                "Mean_aromatic_rings": cls_data["aromatic_rings"].mean(),
                "Mean_heavy_atoms": cls_data["heavy_atoms"].mean(),
            })

    return pd.DataFrame(rows)


def generate_report(
    df: pd.DataFrame,
    naming_df: pd.DataFrame,
    geo_summary: pd.DataFrame,
    ads_summary: pd.DataFrame,
    metal_dist: pd.DataFrame,
    topo_dist: pd.DataFrame,
    linker_dist: pd.DataFrame,
    linker_class_summary: pd.DataFrame,
) -> str:
    """Generate a human-readable text report of key findings."""
    lines = []
    lines.append("=" * 80)
    lines.append("PSA vs VSA Benchmark-Beating MOFs: Chemical Feature Analysis")
    lines.append("=" * 80)
    lines.append("")

    # Group sizes
    for grp in ["PSA-only", "VSA-only", "Both", "Neither"]:
        n = len(df[df["beater_group"] == grp])
        lines.append(f"  {grp}: {n} MOFs")
    lines.append("")

    # Metal distribution
    lines.append("-" * 40)
    lines.append("METAL NODE DISTRIBUTION")
    lines.append("-" * 40)
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = metal_dist[metal_dist["Group"] == grp].head(8)
        lines.append(f"\n  {grp}:")
        for _, row in subset.iterrows():
            lines.append(f"    {row['Metal']:>8s}: {row['Count']:3d} ({row['Fraction']:.1%})")

    # Topology distribution
    lines.append("")
    lines.append("-" * 40)
    lines.append("TOPOLOGY DISTRIBUTION")
    lines.append("-" * 40)
    for grp in ["PSA-only", "VSA-only", "Both"]:
        subset = topo_dist[topo_dist["Group"] == grp].head(8)
        lines.append(f"\n  {grp}:")
        for _, row in subset.iterrows():
            lines.append(f"    {row['Topology']:>12s}: {row['Count']:3d} ({row['Fraction']:.1%})")

    # Geometric features (key ones only)
    lines.append("")
    lines.append("-" * 40)
    lines.append("GEOMETRIC FEATURE COMPARISON (mean +/- std)")
    lines.append("-" * 40)
    key_geo = ["LCD (A)", "PLD (A)", "Density (g/cm3)", "Void_Fraction", "Grav_SA (m2/g)", "Pore_Vol (cm3/cm3)"]
    for feat in key_geo:
        feat_data = geo_summary[geo_summary["Feature"] == feat]
        if len(feat_data) == 0:
            continue
        lines.append(f"\n  {feat}:")
        for _, row in feat_data.iterrows():
            if row["n"] > 0:
                lines.append(f"    {row['Group']:>10s}: {row['mean']:.3f} +/- {row['std']:.3f}  "
                             f"(median={row['median']:.3f}, n={int(row['n'])})")

    # Adsorption metrics (key ones)
    lines.append("")
    lines.append("-" * 40)
    lines.append("ADSORPTION METRIC COMPARISON (mean +/- std)")
    lines.append("-" * 40)
    key_ads = ["Qst_CH4 (kJ/mol)", "Qst_N2 (kJ/mol)", "delta_Qst (kJ/mol)",
               "PSA_alpha", "VSA_alpha", "PSA_API", "VSA_API"]
    for met in key_ads:
        met_data = ads_summary[ads_summary["Metric"] == met]
        if len(met_data) == 0:
            continue
        lines.append(f"\n  {met}:")
        for _, row in met_data.iterrows():
            if row["n"] > 0:
                lines.append(f"    {row['Group']:>10s}: {row['mean']:.4f} +/- {row['std']:.4f}  "
                             f"(median={row['median']:.4f}, n={int(row['n'])})")

    # Linker class summary
    if len(linker_class_summary) > 0:
        lines.append("")
        lines.append("-" * 40)
        lines.append("LINKER CLASS DISTRIBUTION")
        lines.append("-" * 40)
        for grp in ["PSA-only", "VSA-only", "Both"]:
            subset = linker_class_summary[linker_class_summary["Group"] == grp]
            lines.append(f"\n  {grp}:")
            for _, row in subset.iterrows():
                lines.append(f"    {row['Linker_class']:>20s}: {row['Count']:3d} ({row['Fraction']:.1%}) "
                             f"  avg aromatic={row['Mean_aromatic_rings']:.1f}, "
                             f"avg heavy_atoms={row['Mean_heavy_atoms']:.1f}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PSA vs VSA beaters chemical feature analysis")
    parser.add_argument("--skip-mofid", action="store_true", help="Skip MOFid decomposition (use existing only)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and classify
    logger.info("Step 1: Loading and classifying MOFs...")
    df = load_and_classify()

    # Filter to beater groups only for analysis
    beaters = df[df["beater_group"].isin(["PSA-only", "VSA-only", "Both"])].copy()
    logger.info(f"Total beaters for analysis: {len(beaters)}")

    # Step 2: Parse naming features
    logger.info("Step 2: Parsing naming convention features...")
    naming_df = parse_naming_features(beaters)

    # Step 3: Geometric features
    logger.info("Step 3: Comparing geometric features...")
    geo_summary, geo_merged = compare_geometric_features(beaters)

    # Step 4: MOFid decomposition
    logger.info("Step 4: MOFid decomposition...")
    existing_mofid = load_existing_mofid()
    if args.skip_mofid:
        mofid_df = existing_mofid[existing_mofid["mof_id"].isin(beaters["mof_id"])]
        logger.info(f"Using existing MOFid results only: {len(mofid_df)} MOFs")
    else:
        mofid_df = run_mofid_for_missing(beaters["mof_id"].tolist(), existing_mofid)

    mofid_success = mofid_df[mofid_df.get("status", "unknown") == "success"] if len(mofid_df) > 0 else pd.DataFrame()
    logger.info(f"MOFid successful: {len(mofid_success)}/{len(beaters)} beaters")

    # Step 5: RDKit linker analysis
    logger.info("Step 5: RDKit linker analysis...")
    linker_props, mof_linker_df = analyze_linkers_rdkit(mofid_df)

    # Step 6: Adsorption metrics
    logger.info("Step 6: Comparing adsorption metrics...")
    ads_summary = compare_adsorption_metrics(beaters)

    # Step 7: Build distributions
    logger.info("Step 7: Building distributions and report...")
    metal_dist = build_metal_distribution(naming_df, mofid_df)
    topo_dist = build_topology_distribution(naming_df, mofid_df)
    linker_dist = build_linker_distribution_by_group(mof_linker_df, linker_props, beaters)
    linker_class_summary = build_linker_class_summary(mof_linker_df, linker_props, beaters)

    # Generate report
    report = generate_report(df, naming_df, geo_summary, ads_summary,
                              metal_dist, topo_dist, linker_dist, linker_class_summary)
    print(report)

    # Save all results
    beaters.to_csv(OUTPUT_DIR / "beaters_classified.csv", index=False)
    naming_df.to_csv(OUTPUT_DIR / "naming_features.csv", index=False)
    geo_summary.to_csv(OUTPUT_DIR / "geometric_comparison.csv", index=False)
    ads_summary.to_csv(OUTPUT_DIR / "adsorption_comparison.csv", index=False)
    metal_dist.to_csv(OUTPUT_DIR / "metal_distribution.csv", index=False)
    topo_dist.to_csv(OUTPUT_DIR / "topology_distribution.csv", index=False)
    if len(linker_dist) > 0:
        linker_dist.to_csv(OUTPUT_DIR / "linker_frequency_by_group.csv", index=False)
    if len(linker_class_summary) > 0:
        linker_class_summary.to_csv(OUTPUT_DIR / "linker_class_summary.csv", index=False)
    if len(mofid_df) > 0:
        mofid_df.to_csv(OUTPUT_DIR / "mofid_results.csv", index=False)
    if len(linker_props) > 0:
        linker_props.to_csv(OUTPUT_DIR / "linker_properties.csv", index=False)

    # Save report
    with open(OUTPUT_DIR / "analysis_report.txt", "w") as f:
        f.write(report)

    # Save geo_merged for downstream plotting
    geo_merged.to_csv(OUTPUT_DIR / "geo_merged_per_mof.csv", index=False)

    logger.info(f"All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
