"""
Prepare MOF dataset for CrystalFramer training.

Reads the CBM-MOF stratified split CSVs and label file, parses CIF structures
via pymatgen, applies symlog transform to adsorption targets, and writes
raw_data.pkl files expected by CrystalFramer's RegressionDatasetMP.

Output layout (relative to CBM-MOF repo root):
    data/cbm_mof/{train,val,test}/raw/raw_data.pkl

Each pkl is a list[dict] with keys:
    structure    : pymatgen.core.structure.Structure (original unit cell)
    material_id  : str
    AdsCH4_10kPa / AdsCH4_100kPa / AdsCH4_1000kPa   : float (symlog)
    AdsN2_10kPa  / AdsN2_100kPa  / AdsN2_1000kPa    : float (symlog)
    QstCH4 / QstN2                                    : float (no transform)

Usage:
    cd /home/zhangsd/repos/CBM-MOF
    python src/crystalframer/prepare_data_cf.py [--cif-dir PATH] [--output-dir PATH]
"""

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # CBM-MOF/

LABEL_FILE = REPO_ROOT / "src/ml/data/round2/RAC_and_zeo_features_with_id_prop.csv"
SPLITS_DIR = REPO_ROOT / "data/processed/stratified_datasets"
CIF_DIR    = SPLITS_DIR / "cifs"
OUTPUT_DIR = REPO_ROOT / "data/cbm_mof"

TARGET_COLS = [
    "AdsCH4_10kPa",
    "AdsCH4_100kPa",
    "AdsCH4_1000kPa",
    "AdsN2_10kPa",
    "AdsN2_100kPa",
    "AdsN2_1000kPa",
    "QstCH4",
    "QstN2",
]
# Apply symlog transform to adsorption uptakes; leave Qst columns raw
SYMLOG_COLS = TARGET_COLS[:6]

SYMLOG_THRESHOLD = 1e-4  # same as ALIGNN pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────

def symlog(x: float, threshold: float = SYMLOG_THRESHOLD) -> float:
    """Signed log transform: sign(x) * log10(1 + |x| / threshold)."""
    return float(np.sign(x) * np.log10(1.0 + abs(x) / threshold))


def build_split(
    split_name: str,
    split_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    cif_dir: Path,
    output_dir: Path,
    skip_on_error: bool = True,
    max_atoms: int = 0,
) -> None:
    """Build and save raw_data.pkl for one split."""
    # Merge split names with labels
    merged = split_df[["name"]].merge(
        labels_df[["MofName"] + TARGET_COLS],
        left_on="name",
        right_on="MofName",
        how="inner",
    )
    missing = len(split_df) - len(merged)
    if missing > 0:
        warnings.warn(
            f"[{split_name}] {missing} entries had no label match and were dropped."
        )

    out_dir = output_dir / split_name / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_data.pkl"

    data_list = []
    atom_counts = []
    n_failed = 0

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=split_name):
        name = row["name"]
        cif_path = cif_dir / f"{name}.cif"

        if not cif_path.exists():
            if skip_on_error:
                warnings.warn(f"CIF not found: {cif_path} — skipped.")
                n_failed += 1
                continue
            else:
                raise FileNotFoundError(cif_path)

        try:
            structure = Structure.from_file(str(cif_path))
        except Exception as exc:
            if skip_on_error:
                warnings.warn(f"Failed to parse {cif_path}: {exc} — skipped.")
                n_failed += 1
                continue
            else:
                raise

        n_atoms = len(structure)
        if max_atoms > 0 and n_atoms > max_atoms:
            n_failed += 1
            continue
        atom_counts.append(n_atoms)

        entry = {
            "structure": structure,
            "material_id": name,
        }
        for col in TARGET_COLS:
            val = row[col]
            if pd.isna(val):
                if skip_on_error:
                    warnings.warn(
                        f"NaN in {col} for {name} — filling with 0.0."
                    )
                    val = 0.0
                else:
                    raise ValueError(f"NaN in {col} for {name}")
            entry[col] = symlog(val) if col in SYMLOG_COLS else float(val)

        data_list.append(entry)

    with open(out_path, "wb") as f:
        pickle.dump(data_list, f, protocol=4)

    counts = np.array(atom_counts)
    thresh_info = f", max_atoms_filter={max_atoms}" if max_atoms > 0 else ""
    print(
        f"[{split_name}] saved {len(data_list)} samples "
        f"(failed/skipped: {n_failed}{thresh_info}) → {out_path}\n"
        f"  atom counts — min={counts.min()}, max={counts.max()}, "
        f"mean={counts.mean():.0f}, median={np.median(counts):.0f}, "
        f"pct_over_512={100*(counts > 512).mean():.1f}%"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare CrystalFramer datasets for CBM-MOF")
    parser.add_argument("--cif-dir",    type=Path, default=CIF_DIR,
                        help="Directory containing CIF files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Root output directory (data/cbm_mof by default)")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                        help="Directory containing train/val/test_set.csv")
    parser.add_argument("--label-file", type=Path, default=LABEL_FILE,
                        help="CSV with MofName + 8 target columns")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to build (default: all three)")
    parser.add_argument("--max-atoms", type=int, default=512,
                        help="Filter out structures with more than this many atoms "
                             "(0 = no filter; default: 512 = KernelManager.MAX_SYSTEM_SIZE)")
    args = parser.parse_args()

    # Validate paths
    for p in [args.cif_dir, args.splits_dir, args.label_file]:
        if not p.exists():
            sys.exit(f"ERROR: path not found: {p}")

    # Load labels
    print(f"Loading labels from {args.label_file} …")
    labels_df = pd.read_csv(args.label_file, usecols=["MofName"] + TARGET_COLS)
    print(f"  Labels: {len(labels_df)} entries")

    # Check that all target columns are present
    missing_cols = [c for c in TARGET_COLS if c not in labels_df.columns]
    if missing_cols:
        sys.exit(f"ERROR: label file missing columns: {missing_cols}")

    # Process each split
    for split_name in args.splits:
        csv_path = args.splits_dir / f"{split_name}_set.csv"
        if not csv_path.exists():
            warnings.warn(f"Split file not found: {csv_path} — skipping.")
            continue
        split_df = pd.read_csv(csv_path)
        build_split(
            split_name=split_name,
            split_df=split_df,
            labels_df=labels_df,
            cif_dir=args.cif_dir,
            output_dir=args.output_dir,
            max_atoms=args.max_atoms,
        )

    print("\nAll splits done.")


if __name__ == "__main__":
    main()
