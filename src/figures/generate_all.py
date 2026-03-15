"""
One-click generation of the active publication figures.

Usage:
    python src/figures/generate_all.py [--output_dir DIR] [--table_csv PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(
        description="Generate all publication figures for CBM-MOF paper.")
    parser.add_argument(
        "--output_dir", type=str, default="manuscript/figures",
        help="Output directory for figures (default: manuscript/figures)")
    parser.add_argument(
        "--table_csv", type=str, default="results/summary/Table_S3_model_metrics.csv",
        help="CSV path for Table S3 metrics export.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_csv = Path(args.table_csv)
    table_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}\n")

    # ── Figure 4 / Figure 5 / Table S3 ──
    print("[1/3] Figure 4, Figure 5, and Table S3...")
    from src.figures.fig_model_comparison import generate_assets
    generate_assets(output_dir, table_csv)

    # ── Figure 9 ──
    print("[2/2] Combined Top-100 GCMC validation figure (Fig 9)...")
    from src.figures.fig_top100_validation import plot_figure9
    plot_figure9(output_dir)

    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
