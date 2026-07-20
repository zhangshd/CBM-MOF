"""Figure S3: ATC-Cu GCMC validation — experimental vs simulated isotherms.

Generates a single-panel figure comparing GCMC-simulated adsorption isotherms
(CH₄ and N₂ at 298 K) against published experimental data for the ATC-Cu
benchmark MOF, validating the simulation methodology used in the screening.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)

logger = logging.getLogger(__name__)

# ── Path constants ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_RESULTS = PROJECT_ROOT / "results" / "cbm_screening" / "gcmc_ATC-Cu_DreidingTraPPEJson" / "raspa3_parsed_results_1111.csv"
EXP_CH4_CSV = PROJECT_ROOT / "src" / "gcmc" / "examples" / "dup_demo_ATC-Cu" / "CH4_298.csv"
EXP_N2_CSV = PROJECT_ROOT / "src" / "gcmc" / "examples" / "dup_demo_ATC-Cu" / "N2_298.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures"

# ── Data constants ───────────────────────────────────────────────────────────
ATC_CU_NAME = "CoRE-2020[Cu][pts]3[ASR]1"
P_MAX = 1.1  # bar — clip simulation data to experimental pressure range
LITERATURE_UPTAKE_1BAR = {"CH4": 2.90, "N2": 0.75}  # mol/kg, Niu et al. (2019)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_experimental(ch4_path: Path, n2_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load experimental isotherm CSVs and convert pressure from kPa to bar.

    Args:
        ch4_path: Path to the CH₄ experimental CSV (no header, kPa / mol·kg⁻¹).
        n2_path: Path to the N₂ experimental CSV (no header, kPa / mol·kg⁻¹).

    Returns:
        Tuple of (exp_ch4, exp_n2) DataFrames with ``Pressure_bar`` and
        ``Loading_mol_kg`` columns.
    """
    exp_ch4 = pd.read_csv(ch4_path, header=None, names=["Pressure_kPa", "Loading_mol_kg"])
    exp_n2 = pd.read_csv(n2_path, header=None, names=["Pressure_kPa", "Loading_mol_kg"])
    exp_ch4["Pressure_bar"] = exp_ch4["Pressure_kPa"] / 100.0
    exp_n2["Pressure_bar"] = exp_n2["Pressure_kPa"] / 100.0
    return exp_ch4, exp_n2


def load_simulation(sim_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GCMC simulation results for ATC-Cu and split by gas.

    Args:
        sim_path: Path to the RASPA parsed results CSV.

    Returns:
        Tuple of (sim_ch4, sim_n2) DataFrames with ``Pressure_bar`` and
        ``Loading_mean`` columns, filtered to pressures ≤ ``P_MAX`` and sorted.
    """
    sim_data = pd.read_csv(sim_path)
    atc_sim = sim_data[
        (sim_data["MofName"] == ATC_CU_NAME)
        & (sim_data["Temperature[K]"] == 298.0)
        & (sim_data["MoleculeFraction"] == 1.0)
    ]

    sim_ch4 = (
        atc_sim[atc_sim["GasName"] == "methane"]
        .groupby("Pressure[bar]")
        .agg(Loading_mean=("AbsLoading", "mean"))
        .reset_index()
        .rename(columns={"Pressure[bar]": "Pressure_bar"})
    )
    sim_n2 = (
        atc_sim[atc_sim["GasName"] == "N2"]
        .groupby("Pressure[bar]")
        .agg(Loading_mean=("AbsLoading", "mean"))
        .reset_index()
        .rename(columns={"Pressure[bar]": "Pressure_bar"})
    )

    sim_ch4 = sim_ch4[sim_ch4["Pressure_bar"] <= P_MAX].sort_values("Pressure_bar")
    sim_n2 = sim_n2[sim_n2["Pressure_bar"] <= P_MAX].sort_values("Pressure_bar")
    return sim_ch4, sim_n2


def relative_deviation_at_1bar(simulation: pd.DataFrame, literature: float) -> float:
    """Return signed relative deviation from the published 1-bar uptake."""
    at_1bar = simulation.loc[
        simulation["Pressure_bar"].round(6).eq(1.0), "Loading_mean"
    ]
    if len(at_1bar) != 1:
        raise ValueError(f"Expected one simulated 1-bar point, found {len(at_1bar)}")
    return 100.0 * (float(at_1bar.iloc[0]) - literature) / literature


# ── Figure construction ──────────────────────────────────────────────────────


def make_figure(
    exp_ch4: pd.DataFrame,
    exp_n2: pd.DataFrame,
    sim_ch4: pd.DataFrame,
    sim_n2: pd.DataFrame,
) -> plt.Figure:
    """Build the single-panel GCMC validation figure.

    Args:
        exp_ch4: Experimental CH₄ isotherm (``Pressure_bar``, ``Loading_mol_kg``).
        exp_n2: Experimental N₂ isotherm (``Pressure_bar``, ``Loading_mol_kg``).
        sim_ch4: Simulated CH₄ isotherm (``Pressure_bar``, ``Loading_mean``).
        sim_n2: Simulated N₂ isotherm (``Pressure_bar``, ``Loading_mean``).

    Returns:
        Matplotlib Figure ready for saving.
    """
    set_publication_style()
    layout = compute_panel_grid_layout(nrows=1, ncols=1, figure_width_inch=DOUBLE_COL_INCH)

    # 14:8 aspect ratio, 0.85× double-column width
    fig_w = DOUBLE_COL_INCH * 0.85
    fig_h = fig_w * 8.0 / 14.0

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
    )

    # ── Experimental data (scatter) ──────────────────────────────────────────
    ax.scatter(
        exp_ch4["Pressure_bar"],
        exp_ch4["Loading_mol_kg"],
        color=NATURE_COLORS["orange"],
        s=50,
        marker="o",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.85,
        zorder=5,
        label=r"CH$_4$ Experimental",
    )
    ax.scatter(
        exp_n2["Pressure_bar"],
        exp_n2["Loading_mol_kg"],
        color=NATURE_COLORS["red"],
        s=50,
        marker="^",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.85,
        zorder=5,
        label=r"N$_2$ Experimental",
    )

    # ── Simulation data (line + marker) ──────────────────────────────────────
    if not sim_ch4.empty:
        ax.plot(
            sim_ch4["Pressure_bar"],
            sim_ch4["Loading_mean"],
            color=NATURE_COLORS["blue"],
            marker="s",
            markersize=5,
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
            label=r"CH$_4$ Simulation",
        )
    if not sim_n2.empty:
        ax.plot(
            sim_n2["Pressure_bar"],
            sim_n2["Loading_mean"],
            color=NATURE_COLORS["cyan"],
            marker="D",
            markersize=5,
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
            label=r"N$_2$ Simulation",
        )

    # ── Axis labels (no title — caption goes in manuscript) ──────────────────
    ax.set_xlabel("Pressure (bar)", fontsize=layout.body_font)
    ax.set_ylabel("Uptake (mol/kg)", fontsize=layout.body_font)
    ax.tick_params(axis="both", which="major", labelsize=layout.tick_font)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(
        loc="upper left",
        frameon=True,
        edgecolor="black",
        framealpha=0.7,
        fontsize=layout.tick_font,
    )

    # ── Spine styling ────────────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    return fig


# ── CLI entry ────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, load data, build figure, and save."""
    parser = argparse.ArgumentParser(
        description="Generate Figure S3: ATC-Cu GCMC validation (experiment vs simulation).",
    )
    parser.add_argument(
        "--sim-csv",
        type=Path,
        default=SIM_RESULTS,
        help="Path to RASPA parsed results CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--exp-ch4",
        type=Path,
        default=EXP_CH4_CSV,
        help="Path to experimental CH₄ isotherm CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--exp-n2",
        type=Path,
        default=EXP_N2_CSV,
        help="Path to experimental N₂ isotherm CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved figure (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    # ── Validate input files exist ───────────────────────────────────────────
    for label, path in [("Simulation", args.sim_csv), ("Exp CH₄", args.exp_ch4), ("Exp N₂", args.exp_n2)]:
        if not path.exists():
            logger.error("Missing input file (%s): %s", label, path)
            sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading experimental data …")
    exp_ch4, exp_n2 = load_experimental(args.exp_ch4, args.exp_n2)
    logger.info("  CH₄: %d points (%.3f–%.3f bar)", len(exp_ch4), exp_ch4["Pressure_bar"].min(), exp_ch4["Pressure_bar"].max())
    logger.info("  N₂:  %d points (%.3f–%.3f bar)", len(exp_n2), exp_n2["Pressure_bar"].min(), exp_n2["Pressure_bar"].max())

    logger.info("Loading simulation data from %s", args.sim_csv)
    sim_ch4, sim_n2 = load_simulation(args.sim_csv)
    logger.info("  CH₄ sim: %d points ≤ %.1f bar", len(sim_ch4), P_MAX)
    logger.info("  N₂  sim: %d points ≤ %.1f bar", len(sim_n2), P_MAX)
    logger.info(
        "  1-bar relative deviations vs Niu et al.: CH₄ %.2f%%; N₂ %.2f%%",
        relative_deviation_at_1bar(sim_ch4, LITERATURE_UPTAKE_1BAR["CH4"]),
        relative_deviation_at_1bar(sim_n2, LITERATURE_UPTAKE_1BAR["N2"]),
    )

    # ── Build and save ───────────────────────────────────────────────────────
    fig = make_figure(exp_ch4, exp_n2, sim_ch4, sim_n2)
    save_figure(fig, "FigureS3_gcmc_validation", args.output_dir, tight_layout=True)
    logger.info("Done.")


if __name__ == "__main__":
    main()
