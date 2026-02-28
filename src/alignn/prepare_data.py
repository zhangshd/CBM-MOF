"""
prepare_data.py
===============
Data preparation script for ALIGNN training on CBM-MOF dataset.

Steps:
1. Load labels from RAC_and_zeo_features_with_id_prop.csv
2. Apply configurable transform (symlog / log10 / none) to uptake columns
3. Merge with train/val/test split
4. Output ALIGNN-format id_prop.csv files for each split
5. Export transform_config.json for use by evaluate_alignn.py
6. Create symlink to CIF directory
7. (Optional) Check atom count distribution from CIF files

Two orthogonal parameters control the transform:
  --transform  symlog | log10 | none     (transform type for 6 uptake columns)
  --tau        opt | <float>             (τ for symlog; ignored for log10/none)

Qst columns (QstCH4, QstN2) ALWAYS stay as raw kJ/mol regardless of --transform/--tau.

Output directories:
  data/alignn_log10/         log10 transform
  data/alignn_symlog_1e-2/   symlog with uniform τ=1e-2
  data/alignn_symlog_1e-3/   symlog with uniform τ=1e-3
  data/alignn_symlog_opt/    symlog with per-column optimal τ* (i.e. data/alignn/)
  data/alignn_none/          no transform (raw uptake)

Usage:
    # Per-column optimal τ (default, same as current data/alignn/)
    python prepare_data.py --transform symlog --tau opt

    # log10 transform
    python prepare_data.py --transform log10

    # Uniform symlog τ=1e-2
    python prepare_data.py --transform symlog --tau 1e-2

    # Uniform symlog τ=1e-3
    python prepare_data.py --transform symlog --tau 1e-3

    # No transform (raw uptake values)
    python prepare_data.py --transform none

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

# Default output directory (for --transform symlog --tau opt, backward compat)
DEFAULT_ALIGNN_DIR = REPO_ROOT / "data/alignn"

# 8 prediction targets
UPTAKE_TARGETS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
]
QST_TARGETS = ["QstCH4", "QstN2"]
ALL_TARGETS  = UPTAKE_TARGETS + QST_TARGETS

# Per-column optimal τ* from CBM-MOF-symlog v2 Brent search
OPTAU_CONFIG: dict = {
    "AdsCH4_10kPa":   {"type": "symlog", "tau": 1e-6},
    "AdsCH4_100kPa":  {"type": "symlog", "tau": 1e-6},
    "AdsCH4_1000kPa": {"type": "symlog", "tau": 0.177},
    "AdsN2_10kPa":    {"type": "symlog", "tau": 1e-6},
    "AdsN2_100kPa":   {"type": "symlog", "tau": 0.013},
    "AdsN2_1000kPa":  {"type": "raw"},          # near-symmetric: skew=-0.245
    "QstCH4":         {"type": "raw"},           # kJ/mol, kept as-is
    "QstN2":          {"type": "raw"},           # kJ/mol, kept as-is
}


# ──────────────────────────────────────────────────────────────
# Transform functions
# ──────────────────────────────────────────────────────────────
def symlog(x: np.ndarray, tau: float) -> np.ndarray:
    """sign(x) * log10(1 + |x| / tau)"""
    return np.sign(x) * np.log10(1.0 + np.abs(x) / tau)


def inv_symlog(y: np.ndarray, tau: float) -> np.ndarray:
    """Inverse of symlog(x, tau): sign(y) * tau * (10^|y| - 1)."""
    return np.sign(y) * tau * (10.0 ** np.abs(y) - 1.0)


def log10_transform(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """log10(x + eps) for strictly positive uptake values."""
    return np.log10(np.abs(x) + eps)


def inv_log10(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Inverse of log10_transform."""
    return 10.0 ** y - eps


# ──────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────
def build_transform_config(transform: str, tau: str) -> dict:
    """Build per-column transform config from CLI args.

    Args:
        transform: 'symlog', 'log10', or 'none'
        tau: 'opt' (per-column optimal) or a float string like '1e-2'

    Returns:
        dict mapping column name -> {"type": ..., "tau": ...}
    """
    config = {}

    for col in ALL_TARGETS:
        if col in QST_TARGETS:
            # Qst always raw
            config[col] = {"type": "raw"}
        elif transform == "none":
            config[col] = {"type": "raw"}
        elif transform == "log10":
            config[col] = {"type": "log10", "eps": 1e-8}
        elif transform == "symlog":
            if tau == "opt":
                # Use per-column optimal τ* from OPTAU_CONFIG
                config[col] = OPTAU_CONFIG[col].copy()
            else:
                tau_val = float(tau)
                config[col] = {"type": "symlog", "tau": tau_val}
        else:
            raise ValueError(f"Unknown transform: {transform}")

    return config


def build_output_dir(transform: str, tau: str) -> Path:
    """Determine output directory path from transform parameters.

    Convention:
        data/alignn_log10/
        data/alignn_symlog_1e-2/
        data/alignn_symlog_1e-3/
        data/alignn_symlog_opt/  (== data/alignn/ for backward compat)
        data/alignn_none/
    """
    if transform == "log10":
        return REPO_ROOT / "data" / "alignn_log10"
    elif transform == "none":
        return REPO_ROOT / "data" / "alignn_none"
    elif transform == "symlog":
        if tau == "opt":
            # Backward compatible: same as data/alignn/
            return DEFAULT_ALIGNN_DIR
        else:
            tau_tag = tau.replace(".", "p").replace("-", "m")
            return REPO_ROOT / "data" / f"alignn_symlog_{tau}"
    else:
        raise ValueError(f"Unknown transform: {transform}")


def apply_transforms(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return new DataFrame with per-column transforms applied."""
    out = df.copy()
    for col in ALL_TARGETS:
        cfg = config[col]
        if cfg["type"] == "symlog":
            out[col] = symlog(df[col].values, cfg["tau"])
        elif cfg["type"] == "log10":
            out[col] = log10_transform(df[col].values, cfg.get("eps", 1e-8))
        # else: raw → keep as-is
    return out


# ──────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────
def check_symlog(df_raw: pd.DataFrame, config: dict = None):
    """Verify per-column transforms and print distribution stats."""
    if config is None:
        config = OPTAU_CONFIG
    from scipy.stats import skew as _skew
    print("\n=== Per-column transform verification ===")
    print(f"  {'Column':25s}  {'raw_range':>22}  {'type':>8}  "
          f"{'param':>10}  {'raw_skew':>9}  {'xf_skew':>9}  {'err':>10}")
    print("  " + "─" * 100)
    for col in ALL_TARGETS:
        raw = df_raw[col].values
        cfg = config[col]
        if cfg["type"] == "symlog":
            tau = cfg["tau"]
            transformed = symlog(raw, tau)
            recovered = inv_symlog(transformed, tau)
            max_err = float(np.max(np.abs(recovered - raw)))
            param_str = f"τ={tau}"
        elif cfg["type"] == "log10":
            eps = cfg.get("eps", 1e-8)
            transformed = log10_transform(raw, eps)
            recovered = inv_log10(transformed, eps)
            max_err = float(np.max(np.abs(recovered - raw)))
            param_str = f"eps={eps}"
        else:
            transformed = raw
            max_err = 0.0
            param_str = "-"
        raw_sk = float(_skew(raw))
        xf_sk = float(_skew(transformed))
        raw_range = f"[{raw.min():.3f}, {raw.max():.3f}]"
        print(f"  {col:25s}  {raw_range:>22}  {cfg['type']:>8}  "
              f"{param_str:>10}  {raw_sk:>9.3f}  {xf_sk:>9.3f}  {max_err:>10.2e}")


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


# ──────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────
def prepare_datasets(df_transformed: pd.DataFrame, output_dir: Path, config: dict):
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

        out_path = output_dir / split / "id_prop.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"  {split:5s}: {len(out)} rows → {out_path}")

    # Write header with target names (for reference)
    with open(output_dir / "targets.txt", "w") as f:
        f.write("\n".join(ALL_TARGETS) + "\n")
        f.write("# per-column transform config:\n")
        for col, cfg in config.items():
            if cfg["type"] == "symlog":
                f.write(f"#   {col}: symlog τ={cfg['tau']}\n")
            elif cfg["type"] == "log10":
                f.write(f"#   {col}: log10 eps={cfg.get('eps', 1e-8)}\n")
            else:
                f.write(f"#   {col}: raw\n")

    # Export transform config as JSON (consumed by evaluate_alignn.py)
    config_path = output_dir / "transform_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Transform config: {config_path}")
    print(f"\n  Targets file: {output_dir}/targets.txt")


def setup_cif_symlink(output_dir: Path):
    """Create a symlink to the CIF directory under output_dir."""
    link = output_dir / "cifs"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(CIF_DIR)
    print(f"  CIF symlink: {link} -> {CIF_DIR}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Prepare ALIGNN data with configurable target transforms.")
    parser.add_argument("--transform", choices=["symlog", "log10", "none"],
                        default="symlog",
                        help="Transform type for 6 uptake columns (default: symlog)")
    parser.add_argument("--tau", default="opt",
                        help="Symlog τ: 'opt' for per-column optimal, or a float "
                             "like '1e-2' for uniform; ignored for log10/none")
    parser.add_argument("--check-atoms",  action="store_true",
                        help="Check atom count distribution from CIF files")
    parser.add_argument("--check-symlog", action="store_true",
                        help="Verify transform and plot distributions")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip writing id_prop.csv files")
    args = parser.parse_args()

    # Determine output directory
    output_dir = build_output_dir(args.transform, args.tau)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build transform config
    config = build_transform_config(args.transform, args.tau)

    print(f"=== ALIGNN Data Preparation ===")
    print(f"  Transform : {args.transform}")
    print(f"  Tau       : {args.tau}")
    print(f"  Output    : {output_dir}")

    print(f"\nLoading labels from: {LABEL_CSV}")
    df_raw = pd.read_csv(LABEL_CSV)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    if args.check_symlog:
        check_symlog(df_raw, config)

    if args.check_atoms:
        check_atom_counts()

    if not args.skip_prepare:
        print("\n=== Applying transforms ===")
        df_transformed = apply_transforms(df_raw, config)

        # Print summary
        for col in ALL_TARGETS:
            cfg = config[col]
            if cfg["type"] == "symlog":
                print(f"  {col:25s}  symlog τ={cfg['tau']}")
            elif cfg["type"] == "log10":
                print(f"  {col:25s}  log10")
            else:
                print(f"  {col:25s}  raw")

        print("\n=== Writing ALIGNN data splits ===")
        prepare_datasets(df_transformed, output_dir, config)
        setup_cif_symlink(output_dir)

        print(f"\n✓ Data preparation complete!")
        print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
