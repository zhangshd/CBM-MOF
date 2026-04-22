"""
SI Figure: Full-library WC vs Selectivity landscape colored by LSV_norm.

Two-panel scatter showing all ~122k stability-filtered MOFs in WC–α space,
colored by lsv_norm_composite (ensemble uncertainty metric).
Uses the SAME data source and candidate sets as Figure 4 (API-colored version),
only changing the color variable from API to LSV_norm.

Usage:
    conda run -n mofmthnn python fig_si_lsv_norm_landscape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project paths ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]  # CBM-MOF repo root
sys.path.insert(0, str(REPO_ROOT))

from src.figures.style import (
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)

# ── Data paths (SAME as fig_wc_vs_alpha_scatter.py / Figure 4) ──────────────
RESULTS_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150" / "top_candidates"
FULL_CSV = RESULTS_DIR / "full_library_stable_no_uq_filter.csv"
EXP_PSA_CSV = RESULTS_DIR / "exp_top50_psa.csv"
EXP_VSA_CSV = RESULTS_DIR / "exp_top50_vsa.csv"
HYPO_PSA_CSV = RESULTS_DIR / "hypo_top50_psa.csv"
HYPO_VSA_CSV = RESULTS_DIR / "hypo_top50_vsa.csv"

# ── Output paths ─────────────────────────────────────────────────────────────
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
FIGURES_DIR = MODEL_DIR / "figures"
PAPER_REPO = REPO_ROOT.parent / "CBM-MOF-paper"
SI_IMAGES_DIR = PAPER_REPO / "manuscript" / "SuppInfo_CBM" / "images"

ATC_CU_ID = "CoRE-2020[Cu][pts]3[ASR]1"


def load_data():
    """Load the full library and per-process top-50 candidate lists.

    Mirrors fig_wc_vs_alpha_scatter.py exactly.
    """
    print("Loading data...")
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
) -> plt.Figure:
    """Create 2-panel WC vs α scatter colored by LSV_norm.

    Layout, markers, axes, and legend match fig_wc_vs_alpha_scatter.py (Figure 4)
    exactly — only the color variable changes from API to LSV_norm.
    """
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
        wspace=layout.wspace + 0.15,
    )

    # Panel config mirrors Figure 4 exactly: separate exp/hypo highlights per panel
    panels = [
        (axes[0], "PSA_WC_CH4", "PSA_alpha_CH4_N2", "(a) PSA Case",
         exp_psa_ids, hypo_psa_ids),
        (axes[1], "VSA_WC_CH4", "VSA_alpha_CH4_N2", "(b) VSA Case",
         exp_vsa_ids, hypo_vsa_ids),
    ]

    is_atccu = df["mof_id"] == ATC_CU_ID
    cmap = plt.cm.viridis  # same colormap as Figure 4
    scatter_mappables = []

    for panel_idx, (ax, wc_col, alpha_col, label, panel_exp_ids, panel_hypo_ids) in enumerate(panels):
        is_exp_top = df["mof_id"].isin(panel_exp_ids) & ~is_atccu
        is_hypo_top = df["mof_id"].isin(panel_hypo_ids)
        is_background = ~is_atccu & ~is_exp_top & ~is_hypo_top

        # Per-panel LSV_norm range: 99.9th percentile to cover top-candidate range
        lsv_vals = df["lsv_norm_composite"].dropna()
        vmin, vmax = 0.0, np.percentile(lsv_vals, 99.9)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        bg = df[is_background]
        bg = bg.sort_values("lsv_norm_composite", ascending=True)
        ax.scatter(
            bg[wc_col], bg[alpha_col],
            c=bg["lsv_norm_composite"], cmap=cmap, norm=norm,
            s=1.0, alpha=0.25, edgecolors="none", rasterized=True,
            zorder=1,
        )

        # Exp Top-50 (blue triangles — same as Figure 4)
        exp = df[is_exp_top]
        ax.scatter(
            exp[wc_col], exp[alpha_col],
            c=exp["lsv_norm_composite"], cmap=cmap, norm=norm,
            s=18, marker="^", edgecolors=NATURE_COLORS["blue"],
            linewidths=0.5, alpha=0.9, zorder=3,
            label="Exp Top-50",
        )

        # Hypo Top-50 (orange circles — same as Figure 4)
        hypo = df[is_hypo_top]
        ax.scatter(
            hypo[wc_col], hypo[alpha_col],
            c=hypo["lsv_norm_composite"], cmap=cmap, norm=norm,
            s=18, marker="o", edgecolors=NATURE_COLORS["orange"],
            linewidths=0.5, alpha=0.9, zorder=3,
            label="Hypo Top-50",
        )

        # ATC-Cu star (same as Figure 4)
        atccu = df[is_atccu]
        if not atccu.empty:
            ax.scatter(
                atccu[wc_col].values, atccu[alpha_col].values,
                s=100, marker="*", c="red", edgecolors="black",
                linewidths=0.5, zorder=5, label="ATC-Cu",
            )

        # Axes (same labels and limits as Figure 4)
        ax.set_xlabel(r"Working Capacity (mol/kg)", fontsize=layout.body_font)
        if panel_idx == 0:
            ax.set_ylabel(r"Selectivity $\alpha$(CH$_4$/N$_2$)", fontsize=layout.body_font)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=layout.tick_font)
        ax.set_title(label, fontsize=layout.title_font, fontweight="bold")

        # Show ALL data: data max + 5% padding (no percentile clipping)
        wc_max = df[wc_col].dropna().max()
        alpha_max = df[alpha_col].dropna().max()
        ax.set_xlim(0, wc_max * 1.05)
        ax.set_ylim(0, alpha_max * 1.05)

        # Per-panel legend (same as Figure 4)
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

    # Per-panel colorbars (same layout as Figure 4, different label)
    for i, (ax, panel_norm) in enumerate(scatter_mappables):
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=panel_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        if i == len(scatter_mappables) - 1:
            cbar.set_label(
                r"LSV$_{\mathrm{norm}}$ (composite)", fontsize=layout.body_font
            )
        cbar.ax.tick_params(labelsize=layout.tick_font)

    return fig


def main() -> None:
    df, exp_psa_ids, exp_vsa_ids, hypo_psa_ids, hypo_vsa_ids = load_data()
    print(f"Loaded {len(df):,} MOFs")
    print(f"  PSA highlights: {len(exp_psa_ids)} exp + {len(hypo_psa_ids)} hypo")
    print(f"  VSA highlights: {len(exp_vsa_ids)} exp + {len(hypo_vsa_ids)} hypo")
    print(f"  ATC-Cu present: {ATC_CU_ID in df['mof_id'].values}")
    print(f"  lsv_norm_composite: {df['lsv_norm_composite'].notna().sum():,} non-null")

    fig = make_figure(df, exp_psa_ids, exp_vsa_ids, hypo_psa_ids, hypo_vsa_ids)

    # Save to figures dir in results
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig, "fig_si_lsv_norm_landscape", FIGURES_DIR,
        formats=("png",), tight_layout=False,
    )

    # Copy to SI images dir
    SI_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig, "fig_lsv_norm_landscape", SI_IMAGES_DIR,
        formats=("png",), tight_layout=False,
    )

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
