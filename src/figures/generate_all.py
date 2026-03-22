"""
One-click generation of the active publication figures.

Usage:
    python src/figures/generate_all.py [--output-dir DIR] [--table-csv PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    default_output = REPO_ROOT / "results" / "alignn" / "model_ep150" / "figures"
    default_table = default_output / "Table_S3_model_metrics.csv"

    parser = argparse.ArgumentParser(
        description="Generate all publication figures for CBM-MOF paper.")
    parser.add_argument(
        "--output-dir", type=str, default=str(default_output),
        help="Output directory for figures (default: %(default)s)")
    parser.add_argument(
        "--table-csv", type=str, default=str(default_table),
        help="CSV path for Table S3 metrics export (default: %(default)s).")
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
    print("[2/3] Combined Top-100 GCMC validation figure (Fig 9)...")
    from src.figures.fig_top100_validation import plot_figure9
    plot_figure9(output_dir)

    # ── Figure 10 / Figure 11 ──
    print("[3/3] Validated screening-result figures (Fig 10 and Fig 11)...")
    from src.figures.fig_screening_results import plot_figure10, plot_figure11
    plot_figure10(output_dir)
    plot_figure11(output_dir)

    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
