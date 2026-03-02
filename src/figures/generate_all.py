"""
One-click generation of all publication figures.

Usage:
    python src/figures/generate_all.py [--output_dir DIR]
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
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}\n")

    # ── Step 3: Model comparison heatmap ──
    print("[1/5] Model comparison heatmap...")
    from src.figures.fig_model_comparison import plot_heatmap
    plot_heatmap(output_dir)

    # ── Step 4: Multi-model parity plots ──
    print("[2/5] Multi-model parity plots (Fig 4)...")
    from src.figures.fig4_model_parity import plot_parity
    plot_parity(output_dir)

    # ── Step 5a: Top-100 PSA parity ──
    print("[3/5] Top-100 PSA parity plots (Fig 8a)...")
    from src.figures.fig8_top100_parity import plot_top100_psa, plot_top100_vsa
    plot_top100_psa(output_dir)

    # ── Step 5b: Top-100 VSA parity ──
    print("[4/5] Top-100 VSA parity plots (Fig 8b)...")
    plot_top100_vsa(output_dir)

    # ── Step 6: Synthesizability distribution ──
    print("[5/5] Synthesizability distribution...")
    from src.figures.fig_synthesizability import plot_synthesizability
    plot_synthesizability(output_dir)

    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
