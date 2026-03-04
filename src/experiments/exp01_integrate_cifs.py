"""
Exp01 – Integrate CIF files from ARC-MOF, CoREMOF2024, and MOSAEC-DB.

Source: src/jupyter/1_integrat_cifs.ipynb

Output
------
data/processed/integrated_cifs/    (normal mode)
results/test_run/data/processed/integrated_cifs/    (--test mode)
data/processed/file_code_map.json

Run
---
python src/experiments/exp01_integrate_cifs.py
python src/experiments/exp01_integrate_cifs.py --test
"""
import json
import os
import random
import shutil
import sys
import tarfile
from pathlib import Path

# Allow imports from this package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import REPO_ROOT, add_test_arg, resolve_data_dir

import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration – paths relative to REPO_ROOT
# ---------------------------------------------------------------------------
RAW_DIR = REPO_ROOT / "data" / "raw"
ARC_TAR = RAW_DIR / "ARC-MOF" / "ARCMOF_20241004.tar.gz"
ARC_CIF_DIR = RAW_DIR / "ARC-MOF" / "ARCMOF_20241004"
CORE_RAW_DIR = RAW_DIR / "CoREMOF2024"
MOSAEC_RAW_DIR = RAW_DIR / "MOSAEC-DB" / "database_REPEAT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_files(path: Path) -> dict:
    """Return a dict mapping file extension → count for *path* tree."""
    file_types: dict = {}
    for root, _dirs, files in os.walk(path):
        for file in files:
            ext = file.split(".")[-1]
            file_types[ext] = file_types.get(ext, 0) + 1
    return file_types


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def extract_arc_mof() -> Path:
    """Extract ARC-MOF tar.gz if not already done; return the CIF sub-directory."""
    if not ARC_CIF_DIR.exists():
        print(f"Extracting {ARC_TAR} …")
        with tarfile.open(ARC_TAR) as tar:
            tar.extractall(path=ARC_TAR.parent)
        print("Extraction complete.")
    else:
        print(f"ARC-MOF already extracted: {ARC_CIF_DIR}")
    file_types = count_files(ARC_CIF_DIR)
    print(f"ARC-MOF file types: {file_types}")
    return ARC_CIF_DIR


def integrate_arc_mof(processed_dir: Path, file2code: dict) -> int:
    """Copy all ARC-MOF CIFs → processed_dir with 'ARC-' prefix."""
    arc_cif_dir = extract_arc_mof()
    n = 0
    for cif in arc_cif_dir.glob("*.cif"):
        dst = processed_dir / ("ARC-" + cif.name)
        if not dst.exists():
            shutil.copy(cif, dst)
        n += 1

    # Build file → CSD-refcode map for DB12/DB14 (experimental) entries
    exp_list = [f for f in os.listdir(arc_cif_dir) if f.startswith(("DB12", "DB14"))]
    for fname in exp_list:
        code = "_".join(fname.replace("DB12-", "").replace("DB14-", "").split("_")[:-2])
        key = "ARC-" + fname.replace(".cif", "")
        file2code[key] = code

    print(f"ARC-MOF: integrated {n} CIFs.")
    return n


def integrate_coremof(processed_dir: Path, file2code: dict) -> int:
    """Copy CoREMOF2024 ASR CIFs → processed_dir with 'CoRE-' prefix."""
    cif_dict: dict = {}
    for root, _dirs, files in os.walk(CORE_RAW_DIR):
        if not root.endswith("ASR"):
            continue
        for fname in files:
            if fname.endswith(".cif") and "ASR" in fname:
                cif_dict[fname.replace(".cif", "")] = Path(root) / fname

    n = 0
    for k, src in cif_dict.items():
        dst = processed_dir / ("CoRE-" + k + ".cif")
        if not dst.exists():
            shutil.copy(src, dst)
        n += 1

    # Update file→refcode map from CSV metadata
    for csv_path in [
        CORE_RAW_DIR / "CSD-modified" / "CR_data_CSD_modified_20250227.csv",
        CORE_RAW_DIR / "ASR_data_SI_20250204.csv",
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                file2code["CoRE-" + row["coreid"]] = "_".join(
                    str(row["refcode"]).split("_")[:-2]
                )

    print(f"CoREMOF2024: integrated {n} CIFs.")
    return n


def integrate_mosaec(processed_dir: Path, file2code: dict) -> int:
    """Copy MOSAEC-DB 'full' CIFs → processed_dir with 'MOSAEC-' prefix."""
    n = 0
    for cif in MOSAEC_RAW_DIR.glob("*.cif"):
        if "full" in cif.name:
            dst = processed_dir / ("MOSAEC-" + cif.name)
            if not dst.exists():
                shutil.copy(cif, dst)
            n += 1

    # Update file→refcode map from XLSX metadata
    xlsx = RAW_DIR / "MOSAEC-DB" / "mosaec-db.xlsx"
    if xlsx.exists():
        info_df_dic = pd.read_excel(xlsx, sheet_name=None)
        for key, df in info_df_dic.items():
            if "full" not in key:
                continue
            for _, row in df.iterrows():
                file2code["MOSAEC-" + str(row["cif"]) + "_REPEAT"] = row["refcode"]

    print(f"MOSAEC-DB: integrated {n} CIFs.")
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp01: Integrate CIF files from three MOF databases.")
    add_test_arg(parser)
    args = parser.parse_args()

    # Resolve output directories (test-safe)
    processed_dir = resolve_data_dir(args.test, "processed/integrated_cifs")
    map_dir = resolve_data_dir(args.test, "processed")

    file2code: dict = {}

    total = 0
    total += integrate_arc_mof(processed_dir, file2code)
    total += integrate_coremof(processed_dir, file2code)
    total += integrate_mosaec(processed_dir, file2code)

    map_path = map_dir / "file_code_map.json"
    with open(map_path, "w") as f:
        json.dump(file2code, f, indent=4)

    print(f"\nTotal CIFs in {processed_dir}: {len(list(processed_dir.glob('*.cif')))}")
    print(f"file_code_map.json written → {map_path}  ({len(file2code)} entries)")
    if args.test:
        print("[TEST MODE] All outputs in results/test_run/")


if __name__ == "__main__":
    main()
