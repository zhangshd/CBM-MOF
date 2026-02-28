"""
prepare_data.py
===============
Data preparation script for ALIGNN training on CBM-MOF dataset.

Steps:
1. Load labels from RAC_and_zeo_features_with_id_prop.csv
2. Apply per-column symlog transform (τ* from CBM-MOF-symlog v2 Brent search)
3. Merge with train/val/test split
4. Output ALIGNN-format id_prop.csv files for each split
5. Export transform_config.json for use by evaluate_alignn.py
6. Create symlink to CIF directory
7. (Optional) Check atom count distribution from CIF files

Transform strategy (updated 2026-02-28):
  - Replaced global τ=1e-4 with per-column τ* to avoid distribution distortion.
  - Root cause: τ=1e-4 worsened AdsN2_1000kPa skewness from -0.245 → -2.78.
  - AdsN2_1000kPa is kept as raw (near-symmetric, skew=-0.245).
  - Qst columns remain raw (kJ/mol) as before.

Usage:
    # Full preparation
    python prepare_data.py

    # Only verify transforms and atom counts (no file generation)
    python prepare_data.py --check-atoms --check-symlog
"""

import argparse
import json
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

TRANSFORM_CONFIG: dict = {
    "AdsCH4_10kPa":   {"type": "symlog", "tau": 1e-6},
    "AdsCH4_100kPa":  {"type": "symlog", "tau": 1e-6},
    "AdsCH4_1000kPa": {"type": "symlog", "tau": 0.177},
    "AdsN2_10kPa":    {"type": "symlog", "tau": 1e-6},
    "AdsN2_100kPa":   {"type": "symlog", "tau": 0.013},
    "AdsN2_1000kPa":  {"type": "raw"},          # near-symmetric: skew=-0.245; τ=1e-4 → -2.78
    "QstCH4":         {"type": "raw"},           # kJ/mol, kept as-is
    "QstN2":          {"type": "raw"},           # kJ/mol, kept as-is
}


# ──────────────────────────────────────────────────────────────
# Symlog / inverse-symlog transforms
# ──────────────────────────────────────────────────────────────
def symlog(x: np.ndarray, tau: float) -> np.ndarray:
    """sign(x) * log10(1 + |x| / tau)"""
    return np.sign(x) * np.log10(1.0 + np.abs(x) / tau)


def inv_symlog(y: np.ndarray, tau: float) -> np.ndarray:
    """Inverse of symlog(x, tau): sign(y) * tau * (10^|y| - 1)."""
    return np.sign(y) * tau * (10.0 ** np.abs(y) - 1.0)


def apply_transforms(df: pd.DataFrame,
                     config: dict = TRANSFORM_CONFIG) -> pd.DataFrame:
    """Return new DataFrame with per-column symlog transforms applied."""
    out = df.copy()
    for col in ALL_TARGETS:
        cfg = config[col]
        if cfg["type"] == "symlog":
            out[col] = symlog(df[col].values, cfg["tau"])
        # else: raw → keep as-is
    return out


# ──────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────
def check_symlog(df_raw: pd.DataFrame):
    """Verify per-column symlog round-trip and print distribution stats."""
    from scipy.stats import skew as _skew
    print("\n=== Per-column symlog transform verification ===")
    print(f"  {'Column':25s}  {'raw_range':>22}  {'τ*':>8}  "
          f"{'old_skew(1e-4)':>14}  {'new_skew':>9}  {'err':>10}")
    print("  " + "─" * 94)
    OLD_THRESH = 1e-4
    for col in ALL_TARGETS:
        raw = df_raw[col].values
        cfg = TRANSFORM_CONFIG[col]
        if cfg["type"] == "symlog":
            tau = cfg["tau"]
            transformed = symlog(raw, tau)
            recovered   = inv_symlog(transformed, tau)
            max_err = float(np.max(np.abs(recovered - raw)))
            old_xf  = np.sign(raw) * np.log10(1.0 + np.abs(raw) / OLD_THRESH)
            old_sk  = float(_skew(old_xf))
            new_sk  = float(_skew(transformed))
            tau_str = str(tau)
        else:
            transformed = raw
            max_err = 0.0
            old_sk  = float(_skew(raw))
            new_sk  = float(_skew(raw))
            tau_str = "raw"
        raw_range = f"[{raw.min():.3f}, {raw.max():.3f}]"
        print(f"  {col:25s}  {raw_range:>22}  {tau_str:>8}  "
              f"{old_sk:>14.3f}  {new_sk:>9.3f}  {max_err:>10.2e}")

    # Plot raw vs new symlog distributions (2 rows: raw / transformed)
    fig, axes = plt.subplots(2, len(ALL_TARGETS), figsize=(len(ALL_TARGETS) * 3, 6))
    for i, col in enumerate(ALL_TARGETS):
        raw = df_raw[col].values
        cfg = TRANSFORM_CONFIG[col]
        xf  = symlog(raw, cfg["tau"]) if cfg["type"] == "symlog" else raw
        axes[0, i].hist(raw, bins=80, color="steelblue", alpha=0.7)
        axes[0, i].set_title(f"raw\n{col}", fontsize=6)
        axes[1, i].hist(xf, bins=80, color="tomato", alpha=0.7)
        label = f"symlog τ={cfg['tau']}" if cfg["type"] == "symlog" else "raw (no xform)"
        axes[1, i].set_title(label, fontsize=6)
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
        f.write("# per-column transform config (from CBM-MOF-symlog v2 Brent search):\n")
        for col, cfg in TRANSFORM_CONFIG.items():
            if cfg["type"] == "symlog":
                f.write(f"#   {col}: symlog τ={cfg['tau']}\n")
            else:
                f.write(f"#   {col}: raw\n")

    # Export transform config as JSON (consumed by evaluate_alignn.py)
    config_path = ALIGNN_DIR / "transform_config.json"
    with open(config_path, "w") as f:
        json.dump(TRANSFORM_CONFIG, f, indent=2)
    print(f"  Transform config: {config_path}")

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
        print("\n=== Applying per-column transforms ===")
        df_transformed = apply_transforms(df_raw)
        symlog_cols = [c for c, v in TRANSFORM_CONFIG.items() if v["type"] == "symlog"]
        raw_cols    = [c for c, v in TRANSFORM_CONFIG.items() if v["type"] == "raw"]
        print(f"  Symlog applied ({len(symlog_cols)}): {', '.join(symlog_cols)}")
        print(f"  Raw kept ({len(raw_cols)}):         {', '.join(raw_cols)}")

        print("\n=== Writing ALIGNN data splits ===")
        prepare_datasets(df_transformed)
        setup_cif_symlink()

        print("\n✓ Data preparation complete!")
        print(f"  Output directory: {ALIGNN_DIR}")
        print(f"  Run: python train_alignn.py --data_dir {ALIGNN_DIR}/train ...")


if __name__ == "__main__":
    main()
