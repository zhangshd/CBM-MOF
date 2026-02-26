"""
prepare_data.py
===============
Data preparation script for ALIGNN training on CBM-MOF dataset.

Steps:
1. Load labels from RAC_and_zeo_features_with_id_prop.csv
2. Apply symlog transform to 6 uptake targets; keep 2 Qst targets as-is
3. Merge with train/val/test split
4. Output ALIGNN-format id_prop.csv files for each split
5. Create symlink to CIF directory
6. (Optional) Check atom count distribution from CIF files

Usage:
    # Full preparation
    python prepare_data.py

    # Only verify transforms and atom counts (no file generation)
    python prepare_data.py --check-atoms --check-symlog
"""

import argparse
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
REPO_ROOT = Path("/home/zhangsd/repos/CBM-MOF")
LABEL_CSV  = REPO_ROOT / "src/ml/data/round2/RAC_and_zeo_features_with_id_prop.csv"
SPLIT_DIR  = REPO_ROOT / "data/processed/stratified_datasets"
CIF_DIR    = REPO_ROOT / "data/processed/stratified_datasets/cifs"
ALIGNN_DIR = REPO_ROOT / "data/alignn"

# 8 prediction targets
UPTAKE_TARGETS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
]
QST_TARGETS = ["QstCH4", "QstN2"]
ALL_TARGETS  = UPTAKE_TARGETS + QST_TARGETS

SYMLOG_THRESH = 1e-4   # τ in step1_plan


# ──────────────────────────────────────────────────────────────
# Symlog / inverse-symlog transforms
# ──────────────────────────────────────────────────────────────
def symlog(x: np.ndarray, threshold: float = SYMLOG_THRESH) -> np.ndarray:
    """sign(x) * log10(1 + |x| / threshold)"""
    return np.sign(x) * np.log10(1.0 + np.abs(x) / threshold)


def inv_symlog(y: np.ndarray, threshold: float = SYMLOG_THRESH) -> np.ndarray:
    """Inverse of symlog."""
    return np.sign(y) * threshold * (10.0 ** np.abs(y) - 1.0)


def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Return new DataFrame with transformed target columns."""
    out = df.copy()
    for col in UPTAKE_TARGETS:
        out[col] = symlog(df[col].values)
    # QstCH4 and QstN2 kept as-is
    return out


# ──────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────
def check_symlog(df_raw: pd.DataFrame):
    """Verify symlog round-trip and print distribution stats."""
    print("\n=== Symlog transform verification ===")
    for col in UPTAKE_TARGETS:
        raw = df_raw[col].values
        transformed = symlog(raw)
        recovered   = inv_symlog(transformed)
        max_err = np.max(np.abs(recovered - raw))
        print(f"  {col:25s}  raw=[{raw.min():.4f}, {raw.max():.4f}]  "
              f"sym=[{transformed.min():.2f}, {transformed.max():.2f}]  "
              f"round-trip max_err={max_err:.2e}")
    print("\n  QstCH4 range: [{:.2f}, {:.2f}]".format(
        df_raw['QstCH4'].min(), df_raw['QstCH4'].max()))
    print("  QstN2  range: [{:.2f}, {:.2f}]".format(
        df_raw['QstN2'].min(), df_raw['QstN2'].max()))

    # Plot raw vs symlog distributions
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i, col in enumerate(UPTAKE_TARGETS):
        axes[0, i].hist(df_raw[col].values, bins=100, color='steelblue', alpha=0.7)
        axes[0, i].set_title(f"raw {col}", fontsize=7)
        axes[1, i].hist(symlog(df_raw[col].values), bins=100, color='tomato', alpha=0.7)
        axes[1, i].set_title(f"symlog {col}", fontsize=7)
    plt.tight_layout()
    out_path = ALIGNN_DIR / "symlog_distributions.png"
    plt.savefig(out_path, dpi=100)
    print(f"\n  Distribution plot saved to: {out_path}")
    plt.close()


def check_atom_counts(sample_size: int = 500):
    """Sample CIF files and plot atom count distribution."""
    print("\n=== Atom count distribution check ===")
    from pymatgen.core import Structure

    cif_files = list(CIF_DIR.glob("*.cif"))
    np.random.seed(42)
    sampled = np.random.choice(cif_files, min(sample_size, len(cif_files)), replace=False)

    counts = []
    failed = 0
    for cif in sampled:
        try:
            s = Structure.from_file(str(cif))
            counts.append(len(s))
        except Exception:
            failed += 1

    counts = np.array(counts)
    print(f"  Sampled {len(counts)} CIFs (failed: {failed})")
    print(f"  Atom count: min={counts.min()}  median={int(np.median(counts))}  "
          f"mean={counts.mean():.1f}  max={counts.max()}")
    print(f"  > 320 atoms: {(counts > 320).sum()} ({100*(counts>320).mean():.1f}%)")
    print(f"  > 500 atoms: {(counts > 500).sum()} ({100*(counts>500).mean():.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts, bins=50, color='steelblue', alpha=0.8)
    ax.axvline(320, color='red', linestyle='--', label='320 (CrystalFramer limit)')
    ax.set_xlabel("Atom count per unit cell")
    ax.set_ylabel("Count")
    ax.set_title("Atom count distribution (sampled 500 CIFs)")
    ax.legend()
    plt.tight_layout()
    out_path = ALIGNN_DIR / "atom_count_distribution.png"
    plt.savefig(out_path, dpi=100)
    print(f"  Atom count plot saved to: {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────
def prepare_datasets(df_transformed: pd.DataFrame):
    """
    Write ALIGNN id_prop.csv for train / val / test splits.
    Format: mol_id (no .cif), target1, target2, ...
    """
    for split in ["train", "val", "test"]:
        split_csv = SPLIT_DIR / f"{split}_set.csv"
        split_df  = pd.read_csv(split_csv)
        names     = split_df["name"].tolist()

        # Merge with labels
        merged = df_transformed[df_transformed["MofName"].isin(names)].copy()
        missing = set(names) - set(merged["MofName"].tolist())
        if missing:
            print(f"  WARNING: {len(missing)} {split} names not found in label file → skipped")

        # Build id_prop rows: mol_id (filename stem) + 8 targets
        out = merged[["MofName"] + ALL_TARGETS].copy()
        out.columns = ["mol_id"] + ALL_TARGETS

        out_path = ALIGNN_DIR / split / "id_prop.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"  {split:5s}: {len(out)} rows → {out_path}")

    # Write header with target names (for reference)
    with open(ALIGNN_DIR / "targets.txt", "w") as f:
        f.write("\n".join(ALL_TARGETS) + "\n")
        f.write(f"# symlog_threshold={SYMLOG_THRESH}\n")
        f.write("# symlog applied to: " + ", ".join(UPTAKE_TARGETS) + "\n")
        f.write("# raw (no transform): " + ", ".join(QST_TARGETS) + "\n")

    print(f"\n  Targets file: {ALIGNN_DIR}/targets.txt")


def setup_cif_symlink():
    """Create a symlink to the CIF directory under ALIGNN_DIR."""
    link = ALIGNN_DIR / "cifs"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(CIF_DIR)
    print(f"  CIF symlink: {link} -> {CIF_DIR}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-atoms",  action="store_true",
                        help="Check atom count distribution from CIF files")
    parser.add_argument("--check-symlog", action="store_true",
                        help="Verify symlog transform and plot distributions")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip writing id_prop.csv files")
    args = parser.parse_args()

    ALIGNN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels from: {LABEL_CSV}")
    df_raw = pd.read_csv(LABEL_CSV)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    if args.check_symlog:
        check_symlog(df_raw)

    if args.check_atoms:
        check_atom_counts()

    if not args.skip_prepare:
        print("\n=== Applying transforms ===")
        df_transformed = apply_transforms(df_raw)
        print("  Transformed uptake targets with symlog")

        print("\n=== Writing ALIGNN data splits ===")
        prepare_datasets(df_transformed)
        setup_cif_symlink()

        print("\n✓ Data preparation complete!")
        print(f"  Output directory: {ALIGNN_DIR}")
        print(f"  Run: python train_alignn.py --data_dir {ALIGNN_DIR}/train ...")


if __name__ == "__main__":
    main()
