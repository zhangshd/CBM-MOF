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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (
    DPI,
    DOUBLE_COL_INCH,
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

# ── Temperature visual identity ──────────────────────────────────────────────
TEMP_STYLE = {
    273.0: {"color": "#0173B2", "marker": "o", "label": "273 K"},
    298.0: {"color": "#029E73", "marker": "s", "label": "298 K"},
    323.0: {"color": "#D55E00", "marker": "^", "label": "323 K"},
}

# Gas display names
GAS_DISPLAY = {"methane": r"CH$_4$", "N2": r"N$_2$"}

R_GAS = 8.314  # J/(mol*K)


def ext_dsl(P: np.ndarray, T: float, params: dict) -> np.ndarray:
    """Extended DSL isotherm: q = qs_b * b_b*P / (1+b_b*P) + qs_d * b_d*P / (1+b_d*P)."""
    b_b = params["b0_b"] * np.exp(-params["deltaU_b"] / (R_GAS * T))
    b_d = params["b0_d"] * np.exp(-params["deltaU_d"] / (R_GAS * T))
    q = params["qs_b"] * b_b * P / (1 + b_b * P)
    q += params["qs_d"] * b_d * P / (1 + b_d * P)
    return q


def simplify_mof_name(name: str) -> str:
    """Shorten MOF ID for subplot titles."""
    name = name.replace("_repeat", "").replace("_freeONLY", "")
    name = name.replace("_full_REPEAT", "").replace("_full", "")
    name = name.replace("_clean", "")
    # Shorten prefixes
    if name.startswith("ARC-DB0-"):
        name = name.replace("ARC-DB0-", "")
    elif name.startswith("ARC-DB12-"):
        name = name.replace("ARC-DB12-", "DB12-")
    # Collapse CoRE-20YY[Metal][topo]3[ASR]N → CoRE-YY-Metal-topo
    m = re.match(r"CoRE-(\d{4})\[(\w+)\]\[(\w+)\]3\[ASR\](\d+)", name)
    if m:
        name = f"CoRE-{m.group(1)[-2:]}-{m.group(2)}-{m.group(3)}"
        if m.group(4) != "1":
            name += f"-{m.group(4)}"
    # Collapse MOSAEC-XXX → XXX
    name = name.replace("MOSAEC-", "")
    # Strip _f0_fsc and collapse .sym.XX → -sXX
    name = name.replace("_f0_fsc", "")
    name = re.sub(r"\.sym\.(\d+)", r"-s\g<1>", name)
    return name


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
    """Choose (nrows, ncols) to accommodate n subplots, preferring wider layouts."""
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
        return 4, 5
    if n <= 20:
        return 4, 5
    if n <= 25:
        return 5, 5
    return 6, 5


def plot_gas_figure(
    gas: str,
    gcmc: pd.DataFrame,
    fits: pd.DataFrame,
    output_path: Path,
    layout_width: float = DOUBLE_COL_INCH * 2.3,
) -> None:
    """Generate one composite figure for a single gas species."""
    # Filter data
    gcmc_gas = gcmc[gcmc["GasName"] == gas].copy()
    fits_gas = fits[fits["GasName"] == gas].copy()

    mof_ids = sorted(fits_gas["MofName"].unique())
    n_mofs = len(mof_ids)
    nrows, ncols = choose_grid(n_mofs)

    logger.info("Plotting %s: %d MOFs in %d x %d grid", gas, n_mofs, nrows, ncols)

    # Layout — use generous spacing for subplot titles
    gl = compute_panel_grid_layout(
        nrows, ncols, layout_width,
        panel_aspect=0.82,
        top_margin_inch=0.15,
        bottom_margin_inch=0.50,
        left_margin_inch=0.55,
        right_margin_inch=0.08,
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(gl.figure_width, gl.figure_height),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=gl.left, right=gl.right,
        bottom=gl.bottom, top=gl.top,
        wspace=gl.wspace * 2.0,
        hspace=gl.hspace * 5.5,
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
                    s=gl.marker_area * 1.5, zorder=5,
                    edgecolors="none", alpha=0.85,
                )

            # Fit curve
            q_fit = ext_dsl(P_fit, T, p_dict)
            ax.plot(P_fit, q_fit, color=style["color"], lw=0.7, zorder=3)

        # Axes formatting
        ax.set_xscale("log")
        ax.set_xlim(0.008, 12)

        # Title
        short_name = simplify_mof_name(mof_id)
        ax.set_title(short_name, fontsize=gl.annotation_font - 0.5, fontweight="normal",
                      loc="center", pad=3)

        # R2 annotation
        r2_text = f"$R^2$={r2_global:.4f}"
        if model_type == "single_site":
            r2_text += "\n(single-site)"
        ax.text(
            0.97, 0.05, r2_text,
            transform=ax.transAxes, fontsize=gl.annotation_font - 1.0,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

        # Tick formatting
        ax.tick_params(axis="both", which="both", labelsize=gl.tick_font - 0.5)
        ax.minorticks_on()

    # Hide unused axes
    for idx in range(n_mofs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Place legend in the last empty subplot (or first hidden one)
    legend_idx = n_mofs  # first hidden subplot
    if legend_idx < nrows * ncols:
        row_leg, col_leg = divmod(legend_idx, ncols)
        ax_leg = axes[row_leg, col_leg]
        ax_leg.set_visible(True)
        ax_leg.axis("off")
        handles = []
        for T, style in TEMP_STYLE.items():
            h = ax_leg.scatter([], [], c=style["color"], marker=style["marker"],
                               s=gl.marker_area * 3, label=f"{style['label']} (GCMC)")
            handles.append(h)
        for T, style in TEMP_STYLE.items():
            h, = ax_leg.plot([], [], color=style["color"], lw=1.0,
                             label=f"{style['label']} (ext-DSL)")
            handles.append(h)
        ax_leg.legend(
            handles=handles,
            loc="center",
            fontsize=gl.body_font,
            ncol=1,
            frameon=False,
        )

    # Shared axis labels
    gas_label = GAS_DISPLAY.get(gas, gas)
    fig.text(
        0.5, 0.005, "Pressure (bar)",
        ha="center", fontsize=gl.body_font,
    )
    fig.text(
        0.002, 0.5, f"{gas_label} uptake (mol/kg)",
        va="center", rotation=90, fontsize=gl.body_font,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
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

    # CH4 figure → img_007a.png
    plot_gas_figure("methane", gcmc, fits, args.output_dir / "img_007a.png")

    # N2 figure → img_007b.png
    plot_gas_figure("N2", gcmc, fits, args.output_dir / "img_007b.png")

    logger.info("Done. Both figures generated.")


if __name__ == "__main__":
    main()
