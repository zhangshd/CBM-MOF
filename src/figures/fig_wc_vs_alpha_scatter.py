"""Figure 8': Full-library Working Capacity vs Selectivity scatter (PSA/VSA).

Two-panel scatter showing all ~122k stable MOFs in WC–alpha space,
coloured by predicted API.  ATC-Cu is highlighted with a star marker.
Each panel highlights only the top-50 candidates for that specific process
(PSA or VSA), using per-process exp/hypo top-50 files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "top_candidates"
FULL_CSV = RESULTS_DIR / "full_library_stable_no_uq_filter.csv"
EXP_PSA_CSV = RESULTS_DIR / "exp_top50_psa.csv"
EXP_VSA_CSV = RESULTS_DIR / "exp_top50_vsa.csv"
HYPO_PSA_CSV = RESULTS_DIR / "hypo_top50_psa.csv"
HYPO_VSA_CSV = RESULTS_DIR / "hypo_top50_vsa.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures"
ATC_CU_ID = "CoRE-2020[Cu][pts]3[ASR]1"


def load_data():
    """Load the full library and per-process top-50 candidate lists."""
    df = pd.read_csv(FULL_CSV)
    exp_psa_ids = set(pd.read_csv(EXP_PSA_CSV)["mof_id"].tolist())
    exp_vsa_ids = set(pd.read_csv(EXP_VSA_CSV)["mof_id"].tolist())
    hypo_psa_ids = set(pd.read_csv(HYPO_PSA_CSV)["mof_id"].tolist())
    hypo_vsa_ids = set(pd.read_csv(HYPO_VSA_CSV)["mof_id"].tolist())
    return df, exp_psa_ids, exp_vsa_ids, hypo_psa_ids, hypo_vsa_ids


def make_figure(
    df: pd.DataFrame,
    exp_psa_ids: set, exp_vsa_ids: set,
    hypo_psa_ids: set, hypo_vsa_ids: set,
):
    """Create the 2-panel WC vs alpha scatter figure with per-panel highlights."""
    set_publication_style()

    layout = compute_panel_grid_layout(
        nrows=1, ncols=2,
        figure_width_inch=DOUBLE_COL_INCH,
        right_margin_inch=0.55,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left, right=layout.right,
        bottom=layout.bottom, top=layout.top,
        wspace=layout.wspace + 0.08,
    )

    # Panel config: (ax, wc_col, alpha_col, api_col, label, exp_ids, hypo_ids)
    panels = [
        (axes[0], "PSA_WC_CH4", "PSA_alpha_CH4_N2", "PSA_API_CH4", "(a) PSA Case", exp_psa_ids, hypo_psa_ids),
        (axes[1], "VSA_WC_CH4", "VSA_alpha_CH4_N2", "VSA_API_CH4", "(b) VSA Case", exp_vsa_ids, hypo_vsa_ids),
    ]

    is_atccu = df["mof_id"] == ATC_CU_ID
    cmap = plt.cm.viridis
    scatter_mappables = []  # one per panel, for per-panel colorbars

    for panel_idx, (ax, wc_col, alpha_col, api_col, label, panel_exp_ids, panel_hypo_ids) in enumerate(panels):
        # Per-panel masks
        is_exp_top = df["mof_id"].isin(panel_exp_ids) & ~is_atccu
        is_hypo_top = df["mof_id"].isin(panel_hypo_ids)
        is_background = ~is_atccu & ~is_exp_top & ~is_hypo_top

        # Per-panel API range for maximum color contrast
        api_vals = df[api_col].dropna()
        vmin, vmax = 0.0, np.percentile(api_vals, 99)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        bg = df[is_background]
        # Sort by API so high-API dots are drawn on top
        bg = bg.sort_values(api_col, ascending=True)
        ax.scatter(
            bg[wc_col], bg[alpha_col],
            c=bg[api_col], cmap=cmap, norm=norm,
            s=1.0, alpha=0.25, edgecolors="none", rasterized=True,
            zorder=1,
        )

        # Exp Top candidates (blue triangles with edge)
        exp = df[is_exp_top]
        ax.scatter(
            exp[wc_col], exp[alpha_col],
            c=exp[api_col], cmap=cmap, norm=norm,
            s=18, marker="^", edgecolors=NATURE_COLORS["blue"],
            linewidths=0.5, alpha=0.9, zorder=3,
            label="Exp Top-50",
        )

        # Hypo Top candidates (orange circles with edge)
        hypo = df[is_hypo_top]
        ax.scatter(
            hypo[wc_col], hypo[alpha_col],
            c=hypo[api_col], cmap=cmap, norm=norm,
            s=18, marker="o", edgecolors=NATURE_COLORS["orange"],
            linewidths=0.5, alpha=0.9, zorder=3,
            label="Hypo Top-50",
        )

        # ATC-Cu star
        atccu = df[is_atccu]
        if not atccu.empty:
            ax.scatter(
                atccu[wc_col].values, atccu[alpha_col].values,
                s=100, marker="*", c="red", edgecolors="black",
                linewidths=0.5, zorder=5, label="ATC-Cu",
            )

        # Axes
        ax.set_xlabel(r"Working Capacity (mol/kg)", fontsize=layout.body_font)
        if panel_idx == 0:
            ax.set_ylabel(r"Selectivity $\alpha$(CH$_4$/N$_2$)", fontsize=layout.body_font)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=layout.tick_font)
        ax.set_title(label, fontsize=layout.title_font, fontweight="bold")

        # Show ALL data: use data max + 5% padding instead of percentile clipping
        wc_max = df[wc_col].dropna().max()
        alpha_max = df[alpha_col].dropna().max()
        ax.set_xlim(0, wc_max * 1.05)
        ax.set_ylim(0, alpha_max * 1.05)

        # Per-panel legend
        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels_,
            loc="upper left",
            fontsize=layout.tick_font,
            markerscale=1.0,
            handletextpad=0.3,
            borderpad=0.3,
        )

        scatter_mappables.append((ax, norm))

    # Per-panel colorbars (one for each panel)
    for i, (ax, panel_norm) in enumerate(scatter_mappables):
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=panel_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        # Only show colorbar label on the last (rightmost) panel
        if i == len(scatter_mappables) - 1:
            cbar.set_label(r"Predicted API (mol$^2$ kg$^{-1}$ kJ$^{-1}$)", fontsize=layout.body_font)
        cbar.ax.tick_params(labelsize=layout.tick_font)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 8: WC vs alpha scatter for PSA/VSA."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the figure (default: results/alignn/model_ep150/figures).",
    )
    args = parser.parse_args()

    df, exp_psa_ids, exp_vsa_ids, hypo_psa_ids, hypo_vsa_ids = load_data()
    print(f"Loaded {len(df):,} MOFs")
    print(f"  PSA highlights: {len(exp_psa_ids)} exp + {len(hypo_psa_ids)} hypo")
    print(f"  VSA highlights: {len(exp_vsa_ids)} exp + {len(hypo_vsa_ids)} hypo")
    print(f"ATC-Cu present: {ATC_CU_ID in df['mof_id'].values}")

    fig = make_figure(df, exp_psa_ids, exp_vsa_ids, hypo_psa_ids, hypo_vsa_ids)
    save_figure(fig, "Figure08_wc_vs_alpha", args.output_dir, formats=("png",), tight_layout=False)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
