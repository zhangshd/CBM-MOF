"""
Add symlog (tau=1e-3) and log10 transform columns to CGCNN and MOFTransformer CSVs.

Appends 12 new columns to each of train/val/test.csv in:
  - src/cgcnn/data/round2/
  - src/moftransformer/data/round2/

New columns (6 symlog_1e-3 + 6 log10, only for uptake tasks; Qst unchanged):
  symlogAdsCH4_10kPa_1e3,  symlogAdsCH4_100kPa_1e3,  symlogAdsCH4_1000kPa_1e3
  symlogAdsN2_10kPa_1e3,   symlogAdsN2_100kPa_1e3,   symlogAdsN2_1000kPa_1e3
  logAdsCH4_10kPa,         logAdsCH4_100kPa,          logAdsCH4_1000kPa
  logAdsN2_10kPa,          logAdsN2_100kPa,           logAdsN2_1000kPa

Usage:
    python src/data/add_transform_columns.py
    python src/data/add_transform_columns.py --dry-run
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ── Transform functions ────────────────────────────────────────────────────────

def symlog(x: np.ndarray, tau: float = 1e-3) -> np.ndarray:
    """Signed log10 transform: sign(x) * log10(1 + |x| / tau)."""
    return np.sign(x) * np.log10(1 + np.abs(x) / tau)


def log10_transform(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """log10 transform (clamp at eps to avoid -inf for near-zero values)."""
    return np.log10(np.maximum(x, eps))


# ── Source columns to transform ───────────────────────────────────────────────

UPTAKE_COLS = [
    "AdsCH4_10kPa",
    "AdsCH4_100kPa",
    "AdsCH4_1000kPa",
    "AdsN2_10kPa",
    "AdsN2_100kPa",
    "AdsN2_1000kPa",
]


def build_new_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and return a DataFrame of new transform columns."""
    new_cols: dict[str, np.ndarray] = {}

    for col in UPTAKE_COLS:
        if col not in df.columns:
            raise KeyError(f"Source column '{col}' not found in DataFrame. "
                           f"Available: {list(df.columns)}")
        vals = df[col].to_numpy(dtype=float)

        # symlog tau=1e-3
        sym_name = col.replace("Ads", "symlogAds").replace("kPa", "kPa_1e3")
        new_cols[sym_name] = symlog(vals, tau=1e-3)

        # log10
        log_name = "log" + col  # e.g. logAdsCH4_10kPa
        new_cols[log_name] = log10_transform(vals)

    return pd.DataFrame(new_cols, index=df.index)


def validate_columns(df: pd.DataFrame, new_cols_df: pd.DataFrame) -> None:
    """Run sanity checks on the newly computed columns."""
    errors: list[str] = []

    for col in new_cols_df.columns:
        vals = new_cols_df[col]
        if vals.isna().any():
            errors.append(f"NaN values in '{col}' (count={vals.isna().sum()})")

    # Check symlog range for uptake columns (physical range: ~0.001–300 mmol/g)
    for col in new_cols_df.columns:
        if "symlog" in col and "_1e3" in col:
            lo, hi = new_cols_df[col].min(), new_cols_df[col].max()
            if lo < -1 or hi > 5:
                errors.append(f"symlog column '{col}' has unexpected range [{lo:.3f}, {hi:.3f}]")

    # Check log10 range (uptake values can span ~0.001–300 mmol/g → log10 ≈ [-3, 2.5])
    for col in new_cols_df.columns:
        if col.startswith("log") and "symlog" not in col:
            lo, hi = new_cols_df[col].min(), new_cols_df[col].max()
            if lo < -6 or hi > 4:
                errors.append(f"log10 column '{col}' has unexpected range [{lo:.3f}, {hi:.3f}]")

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def process_csv(csv_path: Path, dry_run: bool = False) -> None:
    """Append new transform columns to a single CSV file."""
    df = pd.read_csv(csv_path)
    original_cols = list(df.columns)
    n_original = len(original_cols)

    new_cols_df = build_new_columns(df)

    # Check for existing columns to avoid duplicates (overwrite if present)
    for col in new_cols_df.columns:
        if col in df.columns:
            print(f"  [overwrite] '{col}' already exists, will be overwritten")
            df = df.drop(columns=[col])

    validate_columns(df, new_cols_df)

    # Report column stats
    for col in new_cols_df.columns:
        lo, hi = new_cols_df[col].min(), new_cols_df[col].max()
        print(f"  + {col:45s}  range=[{lo:.4f}, {hi:.4f}]")

    if dry_run:
        print(f"  [dry-run] Would write {csv_path} "
              f"({n_original} → {len(df.columns) + len(new_cols_df.columns)} columns)")
        return

    result = pd.concat([df, new_cols_df], axis=1)
    result.to_csv(csv_path, index=False)
    print(f"  Written: {csv_path} ({n_original} → {len(result.columns)} columns)")


def process_data_dir(data_dir: Path, dry_run: bool = False) -> None:
    """Process all split CSVs in a data directory."""
    print(f"\n{'='*60}")
    print(f"Processing: {data_dir}")
    print(f"{'='*60}")

    for split in ("train", "val", "test"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        print(f"\n  [{split}] {csv_path}")
        process_csv(csv_path, dry_run=dry_run)


def self_test() -> None:
    """Quick round-trip check for transform correctness."""
    x = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    tau = 1e-3

    y = symlog(x, tau)
    # Inverse: x = tau * (10^|y| - 1) * sign(y)
    x_rec = tau * (10 ** np.abs(y) - 1) * np.sign(y)
    assert np.allclose(x, x_rec, rtol=1e-6), f"Round-trip failed: max err={np.abs(x-x_rec).max():.2e}"

    print("Self-test PASSED: symlog tau=1e-3 round-trip error < 1e-6")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing files")
    args = parser.parse_args()

    # Locate project root (3 levels up from src/data/)
    root = Path(__file__).resolve().parent.parent.parent

    data_dirs = [
        root / "src" / "cgcnn" / "data" / "round2",
        root / "src" / "moftransformer" / "data" / "round2",
    ]

    self_test()

    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"[skip] {data_dir} does not exist")
            continue
        process_data_dir(data_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run complete] No files were written.")
    else:
        print("\nDone. All CSVs updated.")


if __name__ == "__main__":
    main()
