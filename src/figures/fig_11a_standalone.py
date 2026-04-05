"""
Generate standalone Panel (a) for Figure 11: PLD vs Qst_CH4 scatter plot.

Outputs a single high-DPI PNG for manual composition in PPT.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from figures.style import (  # noqa: E402
    NATURE_COLORS,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    DPI,
    save_figure,
    set_publication_style,
)

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
ZEO_CSV = REPO_ROOT / "data" / "processed" / "RAC_and_zeo_features_deduplicated.csv"
GCMC_CSV = MODEL_DIR / "process_candidates" / "gcmc_vs_ml_comparison.csv"
OUTPUT_DIR = MODEL_DIR / "figures"

# ── Constants ────────────────────────────────────────────────────────────────
BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"
PSA_API_THRESHOLD = 0.4573247894732109
VSA_API_THRESHOLD = 0.17293197972618327

COLOR_PSA = NATURE_COLORS["blue"]
COLOR_VSA = NATURE_COLORS["orange"]
COLOR_BENCHMARK = "black"


def load_scatter_data() -> pd.DataFrame:
    """Load and merge geometry + GCMC data for the scatter plot."""
    gcmc = pd.read_csv(GCMC_CSV)
    zeo = pd.read_csv(ZEO_CSV, usecols=["name", "Di"])[["name", "Di"]]
    zeo = zeo.rename(columns={"name": "mof_id"})
    merged = gcmc.merge(zeo, on="mof_id", how="left")

    merged["is_psa_beater"] = (merged["in_psa100"] == True) & (merged["gcmc_PSA_API_CH4"] >= PSA_API_THRESHOLD)
    merged["is_vsa_beater"] = (merged["in_vsa100"] == True) & (merged["gcmc_VSA_API_CH4"] >= VSA_API_THRESHOLD)
    merged["is_benchmark"] = merged["mof_id"] == BENCHMARK_MOF

    logger.info(
        "Scatter data: %d total, %d with PLD, %d PSA beaters, %d VSA beaters",
        len(merged), merged["Di"].notna().sum(),
        merged["is_psa_beater"].sum(), merged["is_vsa_beater"].sum(),
    )
    return merged


def plot_panel_a_standalone(data: pd.DataFrame, output_dir: Path) -> None:
    """Generate a standalone Panel (a) as a high-DPI PNG."""
    set_publication_style()

    # Bump fonts by +2 pt for standalone PPT asset (larger than inline journal figs)
    label_fs = LABEL_FONT_SIZE + 2
    tick_fs = LABEL_FONT_SIZE + 2  # match labels for readability at this asset size
    legend_fs = LEGEND_FONT_SIZE + 2

    plot_df = data.dropna(subset=["Di", "QstCH4_gcmc"]).copy()

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    # PSA beaters (excluding benchmark)
    psa = plot_df[plot_df["is_psa_beater"] & ~plot_df["is_benchmark"]]
    ax.scatter(psa["Di"], psa["QstCH4_gcmc"], s=28, c=COLOR_PSA, alpha=0.8,
               edgecolors="white", linewidths=0.3, zorder=3,
               label=f"PSA beaters ($n$={len(psa) + 1})")

    # VSA beaters (excluding benchmark)
    vsa = plot_df[plot_df["is_vsa_beater"] & ~plot_df["is_benchmark"]]
    ax.scatter(vsa["Di"], vsa["QstCH4_gcmc"], s=28, c=COLOR_VSA, alpha=0.8,
               edgecolors="white", linewidths=0.3, zorder=3,
               label=f"VSA beaters ($n$={len(vsa) + 1})")

    # ATC-Cu benchmark
    bm = plot_df[plot_df["is_benchmark"]]
    if not bm.empty:
        ax.scatter(bm["Di"], bm["QstCH4_gcmc"], s=100, c=COLOR_BENCHMARK,
                   marker="*", zorder=5, label="ATC-Cu (benchmark)")

    ax.set_xlabel(r"PLD ($\AA$)", fontsize=label_fs)
    ax.set_ylabel(r"$Q_{\mathrm{st,CH_4}}$ (kJ/mol)", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs, loc="upper right",
              handletextpad=0.3, borderpad=0.3, labelspacing=0.3,
              markerscale=1.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "Figure11a_PLD_vs_Qst.png"
    fig.savefig(out_path, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    logger.info("Saved: %s", out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone Figure 11a (PLD vs Qst scatter).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args()

    data = load_scatter_data()
    plot_panel_a_standalone(data, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
