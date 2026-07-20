"""
SI Figure S6: Multi-temperature isotherm fits (Extended DSL model).

Generates two composite figures:
  (a) CH4 isotherms — GCMC data points + ext-DSL fit curves at 273/298/323 K
  (b) N2  isotherms — same layout

Each figure contains one subplot per MOF arranged in a grid.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (
    DPI,
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    compute_panel_grid_layout,
    set_publication_style,
)

logger = logging.getLogger(__name__)

# ATC-Cu duplicates confirmed by StructureMatcher — excluded from process validation
EXCLUDE_MOFS = {
    "ARC-DB12-BIMDIL_freeONLY_repeat",
    "MOSAEC-IMAZAA_full_REPEAT",
}

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "alignn" / "model_ep150" / "process_candidates"
GCMC_298K = RESULTS / "isotherm_input" / "top20_pure_component.csv"
GCMC_MULTITEMP = RESULTS / "isotherm_input" / "top20_pure_component_multitemp.csv"
FIT_PARAMS = RESULTS / "isotherm_fits" / "ext_dsl_fits.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2].parent / "CBM-MOF-paper" / "manuscript" / "SuppInfo_CBM" / "images"
CH4_FIG_NAME = "fig_isotherm_fits_ch4.png"
N2_FIG_NAME = "fig_isotherm_fits_n2.png"

# ── Temperature visual identity ──────────────────────────────────────────────
TEMP_STYLE = {
    273.0: {"color": NATURE_COLORS["blue"], "marker": "o", "label": "273 K"},
    298.0: {"color": NATURE_COLORS["green"], "marker": "s", "label": "298 K"},
    323.0: {"color": NATURE_COLORS["orange"], "marker": "^", "label": "323 K"},
}

SHORT_NAME_MAP = {
    "CoRE-2020[Cu][pts]3[ASR]1": "ATC-Cu",
}

# Gas display names
GAS_DISPLAY = {"methane": r"CH$_4$", "N2": r"N$_2$"}

R_GAS = 8.314  # J/(mol*K)
DEFAULT_LAYOUT_WIDTH = DOUBLE_COL_INCH


def ext_dsl(P: np.ndarray, T: float, params: dict) -> np.ndarray:
    """Extended DSL isotherm: q = qs_b * b_b*P / (1+b_b*P) + qs_d * b_d*P / (1+b_d*P)."""
    b_b = params["b0_b"] * np.exp(-params["deltaU_b"] / (R_GAS * T))
    b_d = params["b0_d"] * np.exp(-params["deltaU_d"] / (R_GAS * T))
    q = params["qs_b"] * b_b * P / (1 + b_b * P)
    q += params["qs_d"] * b_d * P / (1 + b_d * P)
    return q


def simplify_mof_name(name: str) -> str:
    """Shorten MOF ID for subplot titles using the Figure 11 naming rules."""
    if name in SHORT_NAME_MAP:
        return SHORT_NAME_MAP[name]

    cleaned = name
    cleaned = re.sub(r"_(full_REPEAT|clean_repeat|repeat)$", "", cleaned)
    cleaned = cleaned.replace("_full", "")
    cleaned = cleaned.replace("_clean", "")
    cleaned = cleaned.replace("_freeONLY", "")

    for prefix in ("CoRE-", "MOSAEC-"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    cleaned = re.sub(r"^ARC-DB\d+-", "", cleaned)
    cleaned = cleaned.replace("_f0_fsc", "")
    cleaned = cleaned.replace(".sym.", ".")

    return cleaned


def load_gcmc_data() -> pd.DataFrame:
    """Load and merge 298 K and 273/323 K GCMC data."""
    df_298 = pd.read_csv(GCMC_298K)
    df_mt = pd.read_csv(GCMC_MULTITEMP)
    df = pd.concat([df_298, df_mt], ignore_index=True)
    # Standardise columns
    df = df.rename(columns={"Temperature[K]": "T", "Pressure[bar]": "P"})
    df = df[~df["MofName"].isin(EXCLUDE_MOFS)]
    return df


def load_fit_params() -> pd.DataFrame:
    """Load ext-DSL fit parameters."""
    df = pd.read_csv(FIT_PARAMS)
    return df[~df["MofName"].isin(EXCLUDE_MOFS)]


def choose_grid(n: int) -> tuple[int, int]:
    """Choose (nrows, ncols) to accommodate n subplots, preferring larger panels."""
    if n <= 4:
        return 1, n
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 3, 4
    if n <= 16:
        return 4, 4
    if n <= 20:
        return 5, 4
    if n <= 25:
        return 5, 5
    return 6, 5


def build_figure_layout(nrows: int, ncols: int, layout_width: float = DEFAULT_LAYOUT_WIDTH):
    """Build a page-friendly grid layout for the multi-temperature fit figure."""
    return compute_panel_grid_layout(
        nrows,
        ncols,
        layout_width,
        panel_aspect=0.70,
        gap_ratio_x=0.18,
        gap_ratio_y=0.31,
        top_margin_inch=0.26,
        bottom_margin_inch=0.46,
        left_margin_inch=0.48,
        right_margin_inch=0.08,
    )


def plot_gas_figure(
    gas: str,
    gcmc: pd.DataFrame,
    fits: pd.DataFrame,
    output_path: Path,
    layout_width: float = DEFAULT_LAYOUT_WIDTH,
) -> None:
    """Generate one composite figure for a single gas species."""
    # Filter data
    gcmc_gas = gcmc[gcmc["GasName"] == gas].copy()
    fits_gas = fits[fits["GasName"] == gas].copy()

    mof_ids = sorted(fits_gas["MofName"].unique())
    n_mofs = len(mof_ids)
    nrows, ncols = choose_grid(n_mofs)

    logger.info("Plotting %s: %d MOFs in %d x %d grid", gas, n_mofs, nrows, ncols)

    gl = build_figure_layout(nrows, ncols, layout_width)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(gl.figure_width, gl.figure_height),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=gl.left, right=gl.right,
        bottom=gl.bottom, top=gl.top,
        wspace=gl.wspace * 1.05,
        hspace=gl.hspace * 1.10,
    )

    # Pressure grid for smooth fit curves
    P_fit = np.logspace(np.log10(0.008), np.log10(12), 200)

    for idx, mof_id in enumerate(mof_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Get fit params for this MOF+gas
        row_fit = fits_gas[fits_gas["MofName"] == mof_id]
        if row_fit.empty:
            logger.warning("No fit params for %s / %s", mof_id, gas)
            ax.set_visible(False)
            continue

        params = row_fit.iloc[0]
        p_dict = {
            "qs_b": params["qs_b"],
            "qs_d": params["qs_d"],
            "b0_b": params["b0_b"],
            "b0_d": params["b0_d"],
            "deltaU_b": params["deltaU_b"],
            "deltaU_d": params["deltaU_d"],
        }

        r2_global = params["R2_global"]
        model_type = params["model_used"]

        # Plot each temperature
        for T, style in TEMP_STYLE.items():
            mask = (gcmc_gas["MofName"] == mof_id) & (np.isclose(gcmc_gas["T"], T, atol=1.0))
            df_t = gcmc_gas[mask].sort_values("P")

            if not df_t.empty:
                ax.scatter(
                    df_t["P"], df_t["AbsLoading"],
                    c=style["color"], marker=style["marker"],
                    s=gl.marker_area * 2.0, zorder=5,
                    edgecolors="white", linewidths=0.25, alpha=0.95,
                    rasterized=True,
                )

            # Fit curve
            q_fit = ext_dsl(P_fit, T, p_dict)
            ax.plot(P_fit, q_fit, color=style["color"], lw=1.0, zorder=3, solid_capstyle="round")

        # Axes formatting
        ax.set_xscale("log")
        ax.set_xlim(0.008, 12)
        ax.set_ylim(bottom=0.0)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=4))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Title
        short_name = simplify_mof_name(mof_id)
        ax.set_title(
            short_name,
            fontsize=gl.annotation_font + 1.0,
            fontweight="bold",
            loc="center",
            pad=2,
        )

        # R2 annotation
        r2_text = f"$R^2$ = {r2_global:.4f}"
        if model_type == "single_site":
            r2_text += "\n1-site"
        ax.text(
            0.04,
            0.96,
            r2_text,
            transform=ax.transAxes,
            fontsize=max(7.0, gl.annotation_font + 0.5),
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CFCFCF", lw=0.35, alpha=0.88),
        )

        # Tick formatting
        ax.tick_params(axis="both", which="major", labelsize=gl.tick_font - 0.25, pad=1.2)

    # Hide unused axes
    for idx in range(n_mofs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.0,
            markersize=max(4.2, gl.tick_font - 2.0),
            markeredgewidth=0.0,
            solid_capstyle="round",
            label=style["label"],
        )
        for style in TEMP_STYLE.values()
    ]
    legend_idx = n_mofs
    if legend_idx < nrows * ncols:
        row_leg, col_leg = divmod(legend_idx, ncols)
        ax_leg = axes[row_leg, col_leg]
        ax_leg.set_visible(True)
        ax_leg.axis("off")
        ax_leg.legend(
            handles=legend_handles,
            loc="center",
            ncol=1,
            frameon=False,
            fontsize=gl.body_font + 1.0,
            handlelength=1.6,
            handletextpad=0.55,
            labelspacing=0.7,
            borderaxespad=0.0,
        )
    else:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=gl.tick_font + 1.0,
            columnspacing=1.1,
            handletextpad=0.45,
            borderaxespad=0.0,
        )

    # Shared axis labels
    gas_label = GAS_DISPLAY.get(gas, gas)
    fig.text(
        0.5, 0.012, "Pressure (bar)",
        ha="center", fontsize=gl.body_font + 1.0,
    )
    fig.text(
        0.012, 0.5, f"{gas_label} uptake (mol/kg)",
        va="center", rotation=90, fontsize=gl.body_font + 1.0,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    logger.info("Saved: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SI Figure S6: multi-temperature isotherm fits",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for output images (default: manuscript images)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    set_publication_style()

    gcmc = load_gcmc_data()
    fits = load_fit_params()

    logger.info("Loaded %d GCMC data points, %d fit entries", len(gcmc), len(fits))
    logger.info("Temperatures in GCMC data: %s", sorted(gcmc["T"].unique()))
    logger.info("MOFs in fits: %d", fits["MofName"].nunique())

    # CH4 figure
    plot_gas_figure("methane", gcmc, fits, args.output_dir / CH4_FIG_NAME)

    # N2 figure
    plot_gas_figure("N2", gcmc, fits, args.output_dir / N2_FIG_NAME)

    logger.info("Done. Both figures generated.")


if __name__ == "__main__":
    main()
