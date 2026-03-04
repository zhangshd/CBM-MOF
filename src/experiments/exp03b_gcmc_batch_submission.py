"""
Exp03b – GCMC (RASPA3) and Widom insertion (RASPA2) batch SLURM submission,
          plus ATC-Cu experimental-vs-simulation validation plot.

Source: src/jupyter/3_gcmc_batch_submission_cbm.ipynb

Steps
-----
1. Submit RASPA3 GCMC jobs for benchmark ATC-Cu structure (validation).
2. Submit RASPA2 Widom insertion jobs for ATC-Cu (validation).
3. (Optional) Parse simulation results and plot vs experimental data.
4. Submit RASPA3 GCMC jobs for full screening library (CH4/N2 mixture).
5. Submit RASPA2 Widom insertion jobs for full screening library.

Outputs (normal mode)
----------------------
results/cbm_screening/gcmc_ATC-Cu_DreidingTraPPEJson/   (SLURM scripts)
results/cbm_screening/widom_ATC-Cu_DREIDING/            (SLURM scripts)
results/cbm_screening/gcmc_round2_DreidingTraPPEJson/   (SLURM scripts)
results/cbm_screening/widom_round2_DREIDING/            (SLURM scripts)
results/figures/exp03b_atc_cu_gcmc_validation.png

Run
---
python src/experiments/exp03b_gcmc_batch_submission.py
python src/experiments/exp03b_gcmc_batch_submission.py --test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    REPO_ROOT,
    NATURE_COLORS,
    add_test_arg,
    apply_nature_axes,
    resolve_output_dir,
    savefig,
    setup_matplotlib,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOF_HTS_SRC = Path("/home/zhangsd/repos/MOF-HTS/src")
GCMC_SCRIPT_DIR = MOF_HTS_SRC / "gcmc"
FORCE_FIELD_DIR = GCMC_SCRIPT_DIR / "DreidingTraPPEJson"
CIF_DIR = "/home/zhangsd/repos/MOF-HTS/examples/dup_demo_ATC-Cu"

BATCH_SIZE = 50
TEMPERATURES = [298.0]
PRESSURES_GCMC = [1.0e5 * n for n in [1.1, 1.2, 1.3, 1.4, 1.5]]
PRESSURES_WIDOM = [0.0]
ADSORBATE_COMBINATIONS_GCMC = [
    {"molecules": ["methane"], "mol_fractions": [1.0]},
    {"molecules": ["N2"],      "mol_fractions": [1.0]},
]
WIDOM_MOLECULES = ["methane", "N2"]
N_CPUS_GCMC   = 128
N_CPUS_WIDOM  = 16
PARTITION = "C9654"

SIMULATION_PARAMS_FILE = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_simulation.json"
FORCE_FIELD_PARAMS_FILE = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_force_field.json"
SIMULATION_WIDOM_PARAMS = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_widom_simulation.json"
COMPONENT_WIDOM_PARAMS  = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_widom_component.json"
WIDOM_FORCE_FIELD_DIR = GCMC_SCRIPT_DIR / "DREIDING"

# ---------------------------------------------------------------------------
# Full-library screening configuration (mirrors notebook Cell 11 / Cell 13)
# ---------------------------------------------------------------------------
SCREENING_CIF_DIR = str(REPO_ROOT / "data" / "processed" / "stratified_datasets" / "cifs")
SCREENING_PRESSURES_GCMC = [1.0e4, 1.0e5, 1.0e6]   # Pa  (0.1, 1, 10 bar)
SCREENING_ADSORBATE_COMBINATIONS = [
    {"molecules": ["methane", "N2"], "mol_fractions": [0.2, 0.8]},
]
SCREENING_PRESSURES_WIDOM = [0.0]                    # Pa  (Widom insertion)
SCREENING_WIDOM_MOLECULES = ["methane", "N2"]
N_CPUS_SCREENING_GCMC  = 16
N_CPUS_SCREENING_WIDOM = 16

# Validation data
EXP_CH4_CSV = "/home/zhangsd/repos/MOF-HTS/examples/dup_demo_ATC-Cu/CH4_298.csv"
EXP_N2_CSV  = "/home/zhangsd/repos/MOF-HTS/examples/dup_demo_ATC-Cu/N2_298.csv"
SIM_RESULTS = "/home/zhangsd/repos/MOF-HTS/results/cbm_screening/gcmc_ATC-Cu_DreidingTraPPEJson/raspa3_parsed_results_1111.csv"
ATC_CU_ALIASES = ["CoRE-2020[Cu][pts]3[ASR]1"]


# ---------------------------------------------------------------------------
# GCMC / Widom submission
# ---------------------------------------------------------------------------

def submit_gcmc(output_dir: str, dry_run: bool) -> None:
    """Submit RASPA3 GCMC batch jobs for ATC-Cu benchmark."""
    sys.path.insert(0, str(MOF_HTS_SRC))
    try:
        from gcmc.raspa3_batch_slurm_submitter import main as raspa3_batch_slurm_submitter
    except ImportError:
        print("[WARN] MOF-HTS raspa3_batch_slurm_submitter not available; skipping GCMC submission.")
        return

    raspa3_batch_slurm_submitter(
        CIF_DIR, output_dir, BATCH_SIZE,
        TEMPERATURES, PRESSURES_GCMC, ADSORBATE_COMBINATIONS_GCMC,
        FORCE_FIELD_DIR, SIMULATION_PARAMS_FILE, FORCE_FIELD_PARAMS_FILE,
        N_CPUS_GCMC, PARTITION, dry_run,
    )


def submit_screening_gcmc(output_dir: str, dry_run: bool) -> None:
    """Submit RASPA3 GCMC batch jobs for the full screening library (CH4/N2 mixture)."""
    sys.path.insert(0, str(MOF_HTS_SRC))
    try:
        from gcmc.raspa3_batch_slurm_submitter import main as raspa3_batch_slurm_submitter
    except ImportError:
        print("[WARN] MOF-HTS raspa3_batch_slurm_submitter not available; skipping screening GCMC submission.")
        return

    raspa3_batch_slurm_submitter(
        SCREENING_CIF_DIR, output_dir, BATCH_SIZE,
        TEMPERATURES, SCREENING_PRESSURES_GCMC, SCREENING_ADSORBATE_COMBINATIONS,
        FORCE_FIELD_DIR, SIMULATION_PARAMS_FILE, FORCE_FIELD_PARAMS_FILE,
        N_CPUS_SCREENING_GCMC, PARTITION, dry_run,
    )


def submit_screening_widom(output_dir: str, dry_run: bool) -> None:
    """Submit RASPA2 Widom insertion batch jobs for the full screening library."""
    sys.path.insert(0, str(MOF_HTS_SRC))
    try:
        from gcmc.raspa2_widom_batch_slurm_submitter import main as raspa2_widom_batch_slurm_submitter
    except ImportError:
        print("[WARN] MOF-HTS raspa2_widom_batch_slurm_submitter not available; skipping screening Widom submission.")
        return

    force_field_params = {
        "shifted_vs_truncated": "truncated",
        "tailcorrections": "yes",
    }
    raspa2_widom_batch_slurm_submitter(
        SCREENING_CIF_DIR, output_dir, BATCH_SIZE,
        TEMPERATURES, SCREENING_PRESSURES_WIDOM, SCREENING_WIDOM_MOLECULES,
        WIDOM_FORCE_FIELD_DIR, force_field_params,
        SIMULATION_WIDOM_PARAMS, COMPONENT_WIDOM_PARAMS,
        N_CPUS_SCREENING_WIDOM, PARTITION, dry_run,
    )


def submit_widom(output_dir: str, dry_run: bool) -> None:
    """Submit RASPA2 Widom insertion batch jobs for ATC-Cu benchmark."""
    sys.path.insert(0, str(MOF_HTS_SRC))
    try:
        from gcmc.raspa2_widom_batch_slurm_submitter import main as raspa2_widom_batch_slurm_submitter
    except ImportError:
        print("[WARN] MOF-HTS raspa2_widom_batch_slurm_submitter not available; skipping Widom submission.")
        return

    force_field_params = {
        "shifted_vs_truncated": "truncated",
        "tailcorrections": "yes",
    }
    raspa2_widom_batch_slurm_submitter(
        CIF_DIR, output_dir, BATCH_SIZE,
        TEMPERATURES, PRESSURES_WIDOM, WIDOM_MOLECULES,
        WIDOM_FORCE_FIELD_DIR, force_field_params,
        SIMULATION_WIDOM_PARAMS, COMPONENT_WIDOM_PARAMS,
        N_CPUS_WIDOM, PARTITION, dry_run,
    )


# ---------------------------------------------------------------------------
# Validation plot
# ---------------------------------------------------------------------------

def plot_validation(fig_dir: Path) -> None:
    """Plot ATC-Cu sim vs experimental adsorption if data files exist."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    for p in [EXP_CH4_CSV, EXP_N2_CSV, SIM_RESULTS]:
        if not Path(p).exists():
            print(f"[SKIP] Validation plot – missing file: {p}")
            return

    exp_ch4 = pd.read_csv(EXP_CH4_CSV, header=None, names=["Pressure_kPa", "Loading_mol_kg"])
    exp_n2  = pd.read_csv(EXP_N2_CSV,  header=None, names=["Pressure_kPa", "Loading_mol_kg"])
    exp_ch4["Pressure_bar"] = exp_ch4["Pressure_kPa"] / 100.0
    exp_n2["Pressure_bar"]  = exp_n2["Pressure_kPa"]  / 100.0

    sim_data = pd.read_csv(SIM_RESULTS)
    atc_sim = sim_data[
        sim_data["MofName"].isin(ATC_CU_ALIASES) &
        (sim_data["Temperature[K]"] == 298.0) &
        (sim_data["MoleculeFraction"] == 1.0)
    ]
    sim_ch4 = atc_sim[atc_sim["GasName"] == "methane"].groupby("Pressure[bar]").agg(
        Loading_mean=("AbsLoading", "mean")).reset_index().rename(columns={"Pressure[bar]": "Pressure_bar"})
    sim_n2  = atc_sim[atc_sim["GasName"] == "N2"].groupby("Pressure[bar]").agg(
        Loading_mean=("AbsLoading", "mean")).reset_index().rename(columns={"Pressure[bar]": "Pressure_bar"})

    # Filter to experimental range
    p_max = 1.1
    sim_ch4_v = sim_ch4[sim_ch4["Pressure_bar"] <= p_max]
    sim_n2_v  = sim_n2[sim_n2["Pressure_bar"]   <= p_max]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(exp_ch4["Pressure_bar"], exp_ch4["Loading_mol_kg"],
               color=NATURE_COLORS["orange"], s=80, label="CH₄ Experimental",
               marker="o", edgecolors="black", linewidth=0.8, alpha=0.8, zorder=5)
    ax.scatter(exp_n2["Pressure_bar"],  exp_n2["Loading_mol_kg"],
               color=NATURE_COLORS["red"],    s=80, label="N₂ Experimental",
               marker="^", edgecolors="black", linewidth=0.8, alpha=0.8, zorder=5)
    if not sim_ch4_v.empty:
        ax.plot(sim_ch4_v["Pressure_bar"], sim_ch4_v["Loading_mean"],
                color=NATURE_COLORS["blue"], marker="s", markersize=8,
                label="CH₄ Simulation", linewidth=1.5, alpha=0.85, zorder=4)
    if not sim_n2_v.empty:
        ax.plot(sim_n2_v["Pressure_bar"], sim_n2_v["Loading_mean"],
                color=NATURE_COLORS["cyan"], marker="D", markersize=8,
                label="N₂ Simulation", linewidth=1.5, alpha=0.85, zorder=4)

    ax.set_xlabel("Pressure (bar)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Uptake (mol/kg)", fontsize=12, fontweight="bold")
    ax.set_title("ATC-Cu MOF: Experimental vs GCMC Simulation at 298 K",
                 fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=True, edgecolor="black", loc="lower right", framealpha=0.9)
    apply_nature_axes(ax)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp03b_atc_cu_gcmc_validation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp03b: GCMC/Widom batch SLURM submission + ATC-Cu validation plot."
    )
    add_test_arg(parser)
    args = parser.parse_args()

    setup_matplotlib()
    fig_dir = resolve_output_dir(args.test, "figures")

    dry_run = args.test

    # ATC-Cu benchmark output dirs (routed to test_run/ when --test)
    gcmc_output_dir   = str(resolve_output_dir(args.test, f"cbm_screening/gcmc_ATC-Cu_{FORCE_FIELD_DIR.name}"))
    widom_output_dir  = str(resolve_output_dir(args.test, f"cbm_screening/widom_ATC-Cu_{WIDOM_FORCE_FIELD_DIR.name}"))

    # Full screening library output dirs
    screening_gcmc_dir  = str(resolve_output_dir(args.test, f"cbm_screening/gcmc_round2_{FORCE_FIELD_DIR.name}"))
    screening_widom_dir = str(resolve_output_dir(args.test, f"cbm_screening/widom_round2_{WIDOM_FORCE_FIELD_DIR.name}"))

    print("=== [Step 1] Submitting ATC-Cu GCMC benchmark jobs ===")
    submit_gcmc(gcmc_output_dir, dry_run)

    print("\n=== [Step 2] Submitting ATC-Cu Widom benchmark jobs ===")
    submit_widom(widom_output_dir, dry_run)

    print("\n=== [Step 3] Generating ATC-Cu validation plot ===")
    plot_validation(fig_dir)

    print("\n=== [Step 4] Submitting full-library screening GCMC jobs ===")
    submit_screening_gcmc(screening_gcmc_dir, dry_run)

    print("\n=== [Step 5] Submitting full-library screening Widom jobs ===")
    submit_screening_widom(screening_widom_dir, dry_run)

    if args.test:
        print("\n[TEST MODE] SLURM calls were dry-run; outputs in results/test_run/")


if __name__ == "__main__":
    main()
