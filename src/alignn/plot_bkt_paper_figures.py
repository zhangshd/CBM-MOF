"""
plot_bkt_paper_figures.py — Task 3.3: Generate BKT paper figures and tables.

Generates:
  Step 1: 2-panel breakthrough curve overlay (PSA + VSA, with ATC-Cu baseline)
  Step 2: Performance comparison table (Top-10 vs ATC-Cu, Markdown + CSV)
  Step 3: 2-panel selectivity comparison (GCMC thermodynamic vs BKT dynamic)
  Step 4: Multi-panel isotherm fit figure (SI)

Usage:
    python src/alignn/plot_bkt_paper_figures.py [--step 1|2|3|4|all]
    python src/alignn/plot_bkt_paper_figures.py --model-dir results/alignn/model_ep150
"""

import argparse
import collections
import glob
import os
import signal
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pymatgen.core import Structure

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Publication style
sys.path.insert(0, str(REPO_ROOT / "src" / "figures"))
from style import (
    set_publication_style, save_figure,
    SINGLE_COL_INCH, DOUBLE_COL_INCH, MAX_HEIGHT_INCH, DPI,
)

# BKT modules
import scipy.integrate
import bkt.src.model as model
import bkt.src.params as params
from bkt.src.util import calculate_ki_Dax
from bkt.src.plot import data_to_state

# Suppress BKT/pymatgen/scipy verbose output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Constants from run_bkt_top_candidates.py
# ---------------------------------------------------------------------------
ATC_CU_CIF = Path(
    "/home/zhangsd/repos/MOF-HTS/src/gcmc/examples/dup_demo_ATC-Cu/"
    "CoRE-2020[Cu][pts]3[ASR]1.cif"
)
ATC_CU_NAME = "CoRE-2020[Cu][pts]3[ASR]1"

# Clustered CSV for looking up ATC-Cu cluster ID
CLUSTERED_CSV = (
    REPO_ROOT / "data" / "processed" / "textural_screened"
    / "textural_screened_clustered_with_umap.csv"
)

STANDARD_BED = {
    "D": 9e-3, "L": 0.15, "epsilon": 0.4, "rp": 2e-4,
    "r_pore": 25e-9, "tor": 3, "epsilon_p": 0.35,
    "tN": 500, "Ta": 298, "R": 8.314,
}

PROCESS_CONFIG = {
    "PSA": {"feed_pressure": 10, "tstop_init": 25000, "tstop_extend": 5000, "N": 30},
    "VSA": {"feed_pressure": 1,  "tstop_init": 12000, "tstop_extend": 3000, "N": 30},
}

# ATC-Cu GCMC data paths (for thermodynamic selectivity)
MOF_HTS_REPO = Path("/home/zhangsd/repos/MOF-HTS")
TRAINING_ADS_R1_CSV = (
    MOF_HTS_REPO / "results" / "cbm_screening"
    / "gcmc_round1_DreidingTraPPEJson" / "raspa3_parsed_results_0911.csv"
)
TRAINING_WIDOM_R1_CSV = (
    MOF_HTS_REPO / "results" / "cbm_screening"
    / "widom_round1_DREIDING" / "widom_results_0911.csv"
)
BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"

# Color palette for 10 candidates + ATC-Cu benchmark
CANDIDATE_COLORS = plt.cm.tab10.colors[:10]
BENCHMARK_COLOR = "black"
BENCHMARK_LINESTYLE = "--"

# ---------------------------------------------------------------------------
# Helper: MOF ID simplification
# ---------------------------------------------------------------------------

def simplify_mof_id(mof_id: str) -> str:
    """Shorten MOF ID for figure labels."""
    s = mof_id
    # Remove common prefixes/suffixes
    for prefix in ["ARC-DB0-", "ARC-DB1-"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.endswith("_repeat"):
        s = s[:-7]
    # Shorten CoRE-2020 benchmark
    if "CoRE-2020" in mof_id:
        return "ATC-Cu"
    return s


# ---------------------------------------------------------------------------
# Helper: BKT parameter building (from run_bkt_top_candidates.py)
# ---------------------------------------------------------------------------

def get_rho_s(cif_path: Path) -> float:
    """Calculate adsorbent density from CIF (g/cm³ → kg/m³)."""
    struct = Structure.from_file(str(cif_path))
    return struct.density * 1000


def build_mods(mof_name, process, iso_params, rho_s):
    """Build BKT parameter dict for one simulation."""
    cfg = PROCESS_CONFIG[process]
    bed = STANDARD_BED

    mods = collections.defaultdict()
    mods["nocomponents"] = 2
    mods["feed_yi"] = [0.2, 0.8]
    mods["ini_yi"] = [1e-10, 1e-10]
    mods["isomodel"] = "Langmuir-Freundlich"
    mods["component_names"] = ["CH4", "N2"]
    mods["bi"] = [iso_params["K_CH4"], iso_params["K_N2"]]
    mods["qsi"] = [iso_params["n_m_CH4"], iso_params["n_m_N2"]]
    mods["ni"] = [1, 1]
    mods["Hi"] = [0, 0]
    mods["R"] = bed["R"]
    mods["D"] = bed["D"]
    mods["A"] = np.pi * (bed["D"] / 2) ** 2
    mods["L"] = bed["L"]
    mods["epsilon"] = bed["epsilon"]
    mods["rp"] = bed["rp"]
    mods["Ta"] = bed["Ta"]
    mods["feed_pressure"] = cfg["feed_pressure"]

    flow_rate_ml_min = 10 / cfg["feed_pressure"]
    flow_rate_m3_s = flow_rate_ml_min * 1e-6 / 60
    mods["vfeed"] = flow_rate_m3_s / mods["A"] / mods["epsilon"]
    mods["rho_s"] = rho_s

    k1, k2, Dax = calculate_ki_Dax(
        "CH4", "N2", bed["Ta"], cfg["feed_pressure"],
        bed["rp"], bed["r_pore"], bed["epsilon_p"], bed["tor"],
        mods["vfeed"],
    )
    mods["ki"] = [k1, k2]
    mods["DL"] = Dax

    mods["bed"] = "Breakthrough"
    mods["tstart"] = 0
    mods["tstop"] = cfg["tstop_init"]
    mods["tbreak"] = 0
    mods["tN"] = bed["tN"]
    mods["N"] = cfg["N"]
    return mods


class _SolverTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _SolverTimeout("Solver timed out")


def solve_breakthrough_robust(localparam, min_points=10, timeout_per_strategy=120):
    """Robust BKT solver with fallback strategies and per-strategy timeout."""
    x0_all = model.init(localparam)
    x0 = np.hstack([x0_all[item] for item in localparam.state_names])
    breakthroughmodel = model.oadesmodel
    ev = np.linspace(0, localparam.norm_tbreak, localparam.tN + 1, endpoint=True)
    t_span = (0, localparam.norm_tbreak)

    strategies = [
        ("BDF",   1e-6, 1e-9, "BDF tight"),
        ("Radau", 1e-6, 1e-9, "Radau tight"),
        ("BDF",   1e-4, 1e-7, "BDF default"),
        ("Radau", 1e-4, 1e-7, "Radau default"),
        ("Radau", 1e-3, 1e-6, "Radau relaxed"),
        ("LSODA", 1e-3, 1e-6, "LSODA relaxed"),
    ]
    outcome = None
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        for method, rtol, atol, label in strategies:
            try:
                signal.alarm(timeout_per_strategy)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    outcome = scipy.integrate.solve_ivp(
                        breakthroughmodel, t_span, x0,
                        vectorized=False, t_eval=ev, method=method,
                        rtol=rtol, atol=atol, args=(localparam,),
                    )
                signal.alarm(0)
                if outcome.success and len(outcome.t) >= min_points:
                    print(f"    Solver OK: {label} ({len(outcome.t)} points)")
                    return outcome
            except _SolverTimeout:
                print(f"    Solver {label}: TIMEOUT ({timeout_per_strategy}s)")
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                print(f"    Solver {label} exception: {e}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return outcome


# ---------------------------------------------------------------------------
# Helper: ATC-Cu thermodynamic selectivity
# ---------------------------------------------------------------------------

def get_atc_cu_thermo_selectivity():
    """Get ATC-Cu GCMC thermodynamic selectivity from Round 1 data."""
    ads_r1 = pd.read_csv(TRAINING_ADS_R1_CSV)
    widom_r1 = pd.read_csv(TRAINING_WIDOM_R1_CSV)

    # Pivot adsorption data
    ads_piv = ads_r1.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"],
        values="AbsLoading", aggfunc="first",
    )
    ads_piv.columns = [f"Ads{g}_{int(p*100)}kPa"
                       for g, p in ads_piv.columns]
    # Simpler: reconstruct column names to match expected format
    ads_piv2 = ads_r1.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"],
        values="AbsLoading", aggfunc="first",
    )

    # Extract ATC-Cu adsorption values at key pressures
    atc = ads_r1[ads_r1["MofName"] == BENCHMARK_MOF]
    result = {}

    for gas, gas_label in [("methane", "CH4"), ("N2", "N2")]:
        gas_data = atc[atc["GasName"] == gas]
        for p_bar in [0.1, 1.0, 10.0]:
            p_key = f"{int(p_bar*100)}kPa"
            # Map pressure to kPa label
            p_kpa = p_bar * 100
            if p_kpa == 100:
                key = f"Ads{gas_label}_100kPa"
            elif p_kpa == 1000:
                key = f"Ads{gas_label}_1000kPa"
            elif p_kpa == 10:
                key = f"Ads{gas_label}_10kPa"
            else:
                continue
            row = gas_data[np.isclose(gas_data["Pressure[bar]"], p_bar)]
            if not row.empty:
                result[key] = float(row["AbsLoading"].iloc[0])

    # Compute selectivities: α = (q_CH4 / q_N2) × (y_N2 / y_CH4) = (q_CH4 / q_N2) × 4
    if "AdsCH4_1000kPa" in result and "AdsN2_1000kPa" in result:
        result["PSA_alpha"] = (result["AdsCH4_1000kPa"] / result["AdsN2_1000kPa"]) * 4
    # VSA: at 1 bar (key "100kPa" = p_bar=1.0)
    if "AdsCH4_100kPa" in result and "AdsN2_100kPa" in result:
        result["VSA_alpha"] = (result["AdsCH4_100kPa"] / result["AdsN2_100kPa"]) * 4

    return result


# ---------------------------------------------------------------------------
# Helper: Load BKT summary data
# ---------------------------------------------------------------------------

def load_bkt_summaries(bkt_dir: Path) -> pd.DataFrame:
    """Load and concatenate all bkt_summary_job*.csv files."""
    files = sorted(bkt_dir.glob("bkt_summary_job*.csv"))
    if not files:
        raise FileNotFoundError(f"No bkt_summary_job*.csv found in {bkt_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Drop duplicated header rows (from cat concat)
    df = df[df["mof"] != "mof"].copy()
    # Convert numeric columns
    for col in ["q_CH4_mol_per_kg", "q_N2_mol_per_kg", "rho_s", "feed_pressure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ===================================================================
# STEP 1: Breakthrough Curve Overlay (2-panel)
# ===================================================================

def step1_breakthrough_overlay(bkt_dir: Path, fig_dir: Path):
    """Re-run 22 BKT simulations to extract C/C0 curves, generate 2-panel figure."""
    print("\n" + "=" * 70)
    print("STEP 1: Breakthrough Curve Overlay")
    print("=" * 70)

    # Load data
    fit_csv = bkt_dir / "isotherm_fits" / "best_isotherm_fits.csv"
    fits = pd.read_csv(fit_csv)
    iso_lookup = {}
    for mof in fits["MofName"].unique():
        mof_fits = fits[fits["MofName"] == mof]
        ch4 = mof_fits[mof_fits["GasName"] == "methane"].iloc[0]
        n2 = mof_fits[mof_fits["GasName"] == "N2"].iloc[0]
        iso_lookup[mof] = {
            "K_CH4": ch4["K"], "n_m_CH4": ch4["n_m"],
            "K_N2": n2["K"], "n_m_N2": n2["n_m"],
        }

    psa_mofs = pd.read_csv(bkt_dir / "top10_psa.csv")["mof_id"].tolist()
    vsa_mofs = pd.read_csv(bkt_dir / "top10_vsa.csv")["mof_id"].tolist()
    cif_dir = bkt_dir / "cifs"

    # Build simulation queue
    queue = []
    for mof_id in psa_mofs:
        queue.append((mof_id, "PSA", cif_dir / f"{mof_id}.cif"))
    queue.append((ATC_CU_NAME, "PSA", ATC_CU_CIF))
    for mof_id in vsa_mofs:
        queue.append((mof_id, "VSA", cif_dir / f"{mof_id}.cif"))
    queue.append((ATC_CU_NAME, "VSA", ATC_CU_CIF))

    # Run simulations and extract C/C0
    curves = {"PSA": [], "VSA": []}
    all_cc0_data = []

    for i, (mof_name, process, cif_path) in enumerate(queue):
        print(f"\n[{i+1}/{len(queue)}] {simplify_mof_id(mof_name)} — {process}")
        rho_s = get_rho_s(cif_path)
        mods = build_mods(mof_name, process, iso_lookup[mof_name], rho_s)
        localparam = params.create_param(mods)
        outcome = solve_breakthrough_robust(localparam)

        if outcome is None or not outcome.success:
            print(f"  WARNING: Solver failed for {mof_name} [{process}]")
            continue

        data1 = data_to_state(outcome.y, localparam)
        time_min = outcome.t * localparam.norm_t0 / 60  # seconds → minutes

        # Extract CH4 C/C0 (component A = CH4)
        y_inlet = data1.yA[0]    # inlet mole fraction over time
        y_outlet = data1.yA[-1]  # outlet mole fraction over time (z=L)
        cc0_ch4 = y_outlet / y_inlet

        # Store
        curves[process].append({
            "mof": mof_name,
            "label": simplify_mof_id(mof_name),
            "time_min": time_min,
            "cc0_ch4": cc0_ch4,
            "is_benchmark": mof_name == ATC_CU_NAME,
        })

        # CSV data for reproducibility
        for t, c in zip(time_min, cc0_ch4):
            all_cc0_data.append({
                "mof": mof_name, "process": process,
                "time_min": t, "CC0_CH4": c,
            })
        print(f"  OK: {len(time_min)} time points, C/C0 range [{cc0_ch4.min():.4f}, {cc0_ch4.max():.4f}]")

    # Save C/C0 data CSV
    cc0_df = pd.DataFrame(all_cc0_data)
    cc0_csv = bkt_dir / "breakthrough_curves_data.csv"
    cc0_df.to_csv(cc0_csv, index=False)
    print(f"\nC/C0 data saved: {cc0_csv}")

    # Generate 2-panel figure
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_INCH, 0.45 * DOUBLE_COL_INCH))

    for ax, process, title in zip(
        axes, ["PSA", "VSA"],
        [r"(a) PSA Case (10 bar)", r"(b) VSA Case (1 bar)"]
    ):
        proc_curves = curves[process]
        # Sort: candidates first (by label), benchmark last
        candidates = [c for c in proc_curves if not c["is_benchmark"]]
        benchmarks = [c for c in proc_curves if c["is_benchmark"]]

        # Sort: Benchmark first, then candidates
        for curve in benchmarks:
            ax.plot(
                curve["time_min"][1:], curve["cc0_ch4"][1:],
                color=BENCHMARK_COLOR, linestyle=BENCHMARK_LINESTYLE,
                linewidth=1.2, label="ATC-Cu", zorder=3,
            )

        for idx, curve in enumerate(candidates):
            color = CANDIDATE_COLORS[idx % len(CANDIDATE_COLORS)]
            # Skip first point (t=0, C/C0 undefined)
            ax.plot(
                curve["time_min"][1:], curve["cc0_ch4"][1:],
                color=color, linewidth=0.8,
                label=curve["label"], zorder=2,
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"$C_{\mathrm{CH_4}}/C_0$")
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlim(left=0)

        ax.legend(fontsize=4.5, loc="upper left", ncol=2,
                  framealpha=0.8, edgecolor="none",
                  handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout(w_pad=0.5, h_pad=0.2)
    save_figure(fig, "FigX_breakthrough_overlay", fig_dir, formats=("png",))
    plt.close(fig)
    print(f"\nStep 1 complete: breakthrough overlay figure saved.")


# ===================================================================
# STEP 2: Performance Comparison Table
# ===================================================================

def _lookup_atc_cu_cluster() -> int:
    """Look up ATC-Cu cluster ID from the clustered CSV."""
    if not CLUSTERED_CSV.exists():
        print(f"  WARNING: Clustered CSV not found: {CLUSTERED_CSV}")
        return -1
    df = pd.read_csv(CLUSTERED_CSV, usecols=["CifId", "Cluster"])
    match = df[df["CifId"] == ATC_CU_NAME]
    if match.empty:
        print(f"  WARNING: ATC-Cu ({ATC_CU_NAME}) not found in clustered CSV")
        return -1
    cluster_id = int(match["Cluster"].iloc[0])
    print(f"  ATC-Cu cluster ID: {cluster_id} (from {CLUSTERED_CSV.name})")
    return cluster_id


def step2_performance_table(bkt_dir: Path, fig_dir: Path):
    """Generate Top-10 vs ATC-Cu performance comparison tables.

    Includes α_thermo (GCMC thermodynamic selectivity), α_IAST (IAST from
    pure-component Langmuir fits), α_dyn (BKT dynamic selectivity), and
    derived ratios.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Performance Comparison Table")
    print("=" * 70)

    bkt_df = load_bkt_summaries(bkt_dir)
    psa_top10 = pd.read_csv(bkt_dir / "top10_psa.csv")
    vsa_top10 = pd.read_csv(bkt_dir / "top10_vsa.csv")

    # ATC-Cu thermodynamic selectivity from Round 1 GCMC
    atc_thermo = get_atc_cu_thermo_selectivity()
    atc_alpha_thermo = {
        "PSA": atc_thermo.get("PSA_alpha", np.nan),
        "VSA": atc_thermo.get("VSA_alpha", np.nan),
    }
    print(f"  ATC-Cu α_thermo: PSA={atc_alpha_thermo['PSA']:.2f}, "
          f"VSA={atc_alpha_thermo['VSA']:.2f}")

    # Load IAST selectivity
    iast_csv = bkt_dir / "iast_selectivity.csv"
    if iast_csv.exists():
        iast_df = pd.read_csv(iast_csv)
        print(f"  Loaded IAST data: {len(iast_df)} MOFs")
    else:
        print(f"  WARNING: {iast_csv} not found — run compute_iast_selectivity.py first")
        iast_df = None

    # ATC-Cu cluster ID
    atc_cluster = _lookup_atc_cu_cluster()

    # GCMC alpha column name mapping
    gcmc_alpha_col = {
        "PSA": "gcmc_PSA_alpha_CH4_N2",
        "VSA": "gcmc_VSA_alpha_CH4_N2",
    }

    # Build tables for each process
    md_lines = ["# BKT Performance Comparison: Top-10 vs ATC-Cu\n"]

    for process, top10_df, rank_col in [
        ("PSA", psa_top10, "psa_top10_rank"),
        ("VSA", vsa_top10, "vsa_top10_rank"),
    ]:
        alpha_col = gcmc_alpha_col[process]
        proc_bkt = bkt_df[bkt_df["process"] == process].copy()
        atc_row = proc_bkt[proc_bkt["mof"] == ATC_CU_NAME]
        cand_rows = proc_bkt[proc_bkt["mof"] != ATC_CU_NAME]

        # Merge with rank/cluster/alpha_thermo info
        merge_cols = ["mof_id", "cluster", rank_col]
        if alpha_col in top10_df.columns:
            merge_cols.append(alpha_col)
        merged = cand_rows.merge(
            top10_df[merge_cols],
            left_on="mof", right_on="mof_id", how="left",
        )
        merged = merged.sort_values(rank_col)

        # Compute dynamic selectivity
        merged["alpha_dyn"] = (merged["q_CH4_mol_per_kg"] / merged["q_N2_mol_per_kg"]) * 4

        # Thermodynamic selectivity from GCMC
        if alpha_col in merged.columns:
            merged["alpha_thermo"] = merged[alpha_col]
        else:
            merged["alpha_thermo"] = np.nan

        # IAST selectivity
        iast_col = f"alpha_IAST_{process}"
        if iast_df is not None and iast_col in iast_df.columns:
            merged = merged.merge(
                iast_df[["MofName", iast_col]],
                left_on="mof", right_on="MofName", how="left",
            )
            merged["alpha_IAST"] = merged[iast_col]
            merged.drop(columns=["MofName"], inplace=True, errors="ignore")
        else:
            merged["alpha_IAST"] = np.nan

        # Ratios
        merged["ratio_dyn_thermo"] = merged["alpha_dyn"] / merged["alpha_thermo"]
        merged["ratio_iast_thermo"] = merged["alpha_IAST"] / merged["alpha_thermo"]

        # ATC-Cu values
        if not atc_row.empty:
            atc = atc_row.iloc[0]
            atc_q_ch4 = atc["q_CH4_mol_per_kg"]
            atc_q_n2 = atc["q_N2_mol_per_kg"]
            atc_alpha_dyn = (atc_q_ch4 / atc_q_n2) * 4
            atc_alpha_th = atc_alpha_thermo[process]
            atc_ratio_dt = atc_alpha_dyn / atc_alpha_th if atc_alpha_th else np.nan
            # ATC-Cu IAST
            if iast_df is not None:
                atc_iast_row = iast_df[iast_df["MofName"] == ATC_CU_NAME]
                atc_alpha_iast = float(atc_iast_row[iast_col].iloc[0]) if not atc_iast_row.empty else np.nan
            else:
                atc_alpha_iast = np.nan
            atc_ratio_it = atc_alpha_iast / atc_alpha_th if atc_alpha_th and not np.isnan(atc_alpha_iast) else np.nan
        else:
            atc_q_ch4, atc_q_n2, atc_alpha_dyn = np.nan, np.nan, np.nan
            atc_alpha_th, atc_ratio_dt = np.nan, np.nan
            atc_alpha_iast, atc_ratio_it = np.nan, np.nan

        merged["vs_ATC_Cu"] = merged["q_CH4_mol_per_kg"] / atc_q_ch4

        # Build Markdown table
        pressure_label = "10 bar" if process == "PSA" else "1 bar"
        md_lines.append(f"\n## {process} Process ({pressure_label})\n")
        md_lines.append(
            "| Rank | MOF ID | Cluster | ρ_s (kg/m³) | "
            "q_CH₄ (mol/kg) | q_N₂ (mol/kg) | α_thermo | α_IAST | α_dyn | "
            "α_dyn/α_thermo | α_IAST/α_thermo | vs ATC-Cu |"
        )
        md_lines.append(
            "|------|--------|---------|-------------|"
            "----------------|----------------|----------|--------|-------"
            "|----------------|-----------------|-----------|"
        )

        for _, row in merged.iterrows():
            rank = int(row[rank_col]) if pd.notna(row.get(rank_col)) else "—"
            cluster = int(row["cluster"]) if pd.notna(row.get("cluster")) else "—"
            a_iast = f"{row['alpha_IAST']:.2f}" if pd.notna(row.get("alpha_IAST")) else "—"
            r_it = f"{row['ratio_iast_thermo']:.2f}" if pd.notna(row.get("ratio_iast_thermo")) else "—"
            md_lines.append(
                f"| {rank} | {simplify_mof_id(row['mof'])} | {cluster} | "
                f"{row['rho_s']:.0f} | "
                f"{row['q_CH4_mol_per_kg']:.3f} | {row['q_N2_mol_per_kg']:.3f} | "
                f"{row['alpha_thermo']:.2f} | "
                f"{a_iast} | "
                f"{row['alpha_dyn']:.2f} | {row['ratio_dyn_thermo']:.2f} | "
                f"{r_it} | "
                f"{row['vs_ATC_Cu']:.2f}× |"
            )

        # ATC-Cu baseline row
        a_iast_atc = f"{atc_alpha_iast:.2f}" if not np.isnan(atc_alpha_iast) else "—"
        r_it_atc = f"{atc_ratio_it:.2f}" if not np.isnan(atc_ratio_it) else "—"
        md_lines.append(
            f"| — | **ATC-Cu** | {atc_cluster} | "
            f"{atc['rho_s']:.0f} | "
            f"{atc_q_ch4:.3f} | {atc_q_n2:.3f} | "
            f"{atc_alpha_th:.2f} | "
            f"{a_iast_atc} | "
            f"{atc_alpha_dyn:.2f} | {atc_ratio_dt:.2f} | "
            f"{r_it_atc} | 1.00× |"
        )

        # Summary note
        ratios_dt = merged["ratio_dyn_thermo"].dropna()
        ratios_it = merged["ratio_iast_thermo"].dropna()
        md_lines.append(
            f"\nα_thermo: GCMC mixed-component thermodynamic selectivity. "
            f"α_IAST: IAST from pure-component Langmuir. "
            f"α_dyn: BKT dynamic selectivity."
        )
        if not ratios_dt.empty:
            md_lines.append(
                f"\n**{process} summary**: α_dyn/α_thermo = {ratios_dt.min():.2f}–{ratios_dt.max():.2f} "
                f"(mean {ratios_dt.mean():.2f}). "
                f"α_IAST/α_thermo = {ratios_it.min():.2f}–{ratios_it.max():.2f} "
                f"(mean {ratios_it.mean():.2f})."
            )

        # Save per-process CSV
        out_cols = ["mof", "cluster", "rho_s", "q_CH4_mol_per_kg",
                    "q_N2_mol_per_kg", "alpha_thermo", "alpha_IAST", "alpha_dyn",
                    "ratio_dyn_thermo", "ratio_iast_thermo", "vs_ATC_Cu"]
        csv_path = bkt_dir / f"performance_table_{process.lower()}.csv"
        merged[out_cols].to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Save combined Markdown
    md_path = bkt_dir / "performance_table_combined.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")
    print(f"\nStep 2 complete: performance tables generated.")


# ===================================================================
# STEP 3: Selectivity Comparison (GCMC Thermodynamic vs BKT Dynamic)
# ===================================================================

def step3_selectivity_comparison(bkt_dir: Path, fig_dir: Path):
    """Generate 4-panel selectivity comparison: α_dyn vs α_thermo + α_IAST vs α_dyn."""
    print("\n" + "=" * 70)
    print("STEP 3: Selectivity Comparison (3-way: GCMC / IAST / BKT)")
    print("=" * 70)

    bkt_df = load_bkt_summaries(bkt_dir)
    psa_top10 = pd.read_csv(bkt_dir / "top10_psa.csv")
    vsa_top10 = pd.read_csv(bkt_dir / "top10_vsa.csv")

    # ATC-Cu thermodynamic selectivity
    atc_thermo = get_atc_cu_thermo_selectivity()
    print(f"  ATC-Cu PSA α_thermo = {atc_thermo.get('PSA_alpha', 'N/A'):.2f}")
    print(f"  ATC-Cu VSA α_thermo = {atc_thermo.get('VSA_alpha', 'N/A'):.2f}")

    # Load IAST selectivity
    iast_csv = bkt_dir / "iast_selectivity.csv"
    if iast_csv.exists():
        iast_df = pd.read_csv(iast_csv)
        print(f"  Loaded IAST data: {len(iast_df)} MOFs")
    else:
        print(f"  WARNING: {iast_csv} not found — run compute_iast_selectivity.py first")
        iast_df = None

    # ATC-Cu BKT dynamic selectivity
    atc_bkt_psa = bkt_df[(bkt_df["mof"] == ATC_CU_NAME) & (bkt_df["process"] == "PSA")]
    atc_bkt_vsa = bkt_df[(bkt_df["mof"] == ATC_CU_NAME) & (bkt_df["process"] == "VSA")]

    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_INCH, DOUBLE_COL_INCH * 0.7))

    panel_configs = [
        # row 0: α_dyn vs α_thermo
        (axes[0, 0], "PSA", psa_top10, "psa_top10_rank",
         "(a) PSA Case (10 bar)", "thermo_vs_dyn"),
        (axes[0, 1], "VSA", vsa_top10, "vsa_top10_rank",
         "(b) VSA Case (1 bar)", "thermo_vs_dyn"),
        # row 1: α_IAST vs α_dyn
        (axes[1, 0], "PSA", psa_top10, "psa_top10_rank",
         "(c) PSA Case (10 bar)", "iast_vs_dyn"),
        (axes[1, 1], "VSA", vsa_top10, "vsa_top10_rank",
         "(d) VSA Case (1 bar)", "iast_vs_dyn"),
    ]

    for ax, process, top10_df, rank_col, title, panel_type in panel_configs:
        proc_bkt = bkt_df[(bkt_df["process"] == process) & (bkt_df["mof"] != ATC_CU_NAME)]

        # Merge BKT with GCMC thermodynamic selectivity
        gcmc_alpha_col = f"gcmc_{process}_alpha_CH4_N2"
        merged = proc_bkt.merge(
            top10_df[["mof_id", gcmc_alpha_col, rank_col]],
            left_on="mof", right_on="mof_id", how="inner",
        )
        merged["alpha_dyn"] = (merged["q_CH4_mol_per_kg"] / merged["q_N2_mol_per_kg"]) * 4
        merged["alpha_thermo"] = merged[gcmc_alpha_col]

        # Merge IAST
        iast_col = f"alpha_IAST_{process}"
        if iast_df is not None and iast_col in iast_df.columns:
            merged = merged.merge(
                iast_df[["MofName", iast_col]],
                left_on="mof", right_on="MofName", how="left",
            )
            merged["alpha_IAST"] = merged[iast_col]
        else:
            merged["alpha_IAST"] = np.nan

        merged = merged.sort_values(rank_col)

        # Determine x/y columns and marker style
        if panel_type == "thermo_vs_dyn":
            x_col, y_col = "alpha_thermo", "alpha_dyn"
            marker = "o"
        else:  # iast_vs_dyn
            x_col, y_col = "alpha_IAST", "alpha_dyn"
            marker = "^"

        # Compute ATC-Cu coordinates first (needed for legend-first plotting)
        if process == "PSA":
            atc_thermo_val = atc_thermo.get("PSA_alpha", np.nan)
            atc_bkt_row = atc_bkt_psa
        else:
            atc_thermo_val = atc_thermo.get("VSA_alpha", np.nan)
            atc_bkt_row = atc_bkt_vsa

        atc_dyn = np.nan
        atc_iast_val = np.nan
        if not atc_bkt_row.empty:
            atc_dyn = (float(atc_bkt_row["q_CH4_mol_per_kg"].iloc[0]) /
                       float(atc_bkt_row["q_N2_mol_per_kg"].iloc[0])) * 4
            if iast_df is not None:
                atc_iast_row = iast_df[iast_df["MofName"] == ATC_CU_NAME]
                if not atc_iast_row.empty:
                    atc_iast_val = float(atc_iast_row[iast_col].iloc[0])

        if panel_type == "thermo_vs_dyn":
            atc_x, atc_y = atc_thermo_val, atc_dyn
        else:
            atc_x, atc_y = atc_iast_val, atc_dyn

        # Plot ATC-Cu first for legend order
        if not np.isnan(atc_x) and not np.isnan(atc_y):
            ax.scatter(
                atc_x, atc_y,
                marker="*", s=72, color=BENCHMARK_COLOR,
                zorder=4, label="ATC-Cu",
            )

        # Plot candidates
        for idx, (_, row) in enumerate(merged.iterrows()):
            if pd.isna(row.get(x_col)) or pd.isna(row.get(y_col)):
                continue
            color = CANDIDATE_COLORS[idx % len(CANDIDATE_COLORS)]
            label = simplify_mof_id(row["mof"])
            ax.scatter(
                row[x_col], row[y_col],
                marker=marker, color=color, s=30, zorder=3,
                label=label, edgecolors="none",
            )

        # y=x diagonal
        x_vals = list(merged[x_col].dropna())
        y_vals = list(merged[y_col].dropna())
        all_vals = x_vals + y_vals
        if not np.isnan(atc_x):
            all_vals.append(atc_x)
        if not np.isnan(atc_y):
            all_vals.append(atc_y)
        if all_vals:
            vmin = min(all_vals) * 0.85
            vmax = max(all_vals) * 1.1
            ax.plot([vmin, vmax], [vmin, vmax], ":", color="gray",
                    linewidth=0.6, zorder=1)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)

        # Axis labels
        if panel_type == "thermo_vs_dyn":
            ax.set_xlabel(r"$\alpha_{\mathrm{thermo}}$ (GCMC)")
            ax.set_ylabel(r"$\alpha_{\mathrm{dyn}}$ (BKT)")
        else:
            ax.set_xlabel(r"$\alpha_{\mathrm{IAST}}$")
            ax.set_ylabel(r"$\alpha_{\mathrm{dyn}}$ (BKT)")

        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        ax.set_aspect("equal", adjustable="box")

        ax.legend(fontsize=4, loc="upper left", ncol=2,
                  framealpha=0.8, edgecolor="none",
                  handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout(w_pad=0.2, h_pad=0.2)
    save_figure(fig, "FigX_selectivity_comparison", fig_dir, formats=("png",))
    plt.close(fig)
    print(f"\nStep 3 complete: 4-panel selectivity comparison figure saved.")


# ===================================================================
# STEP 4: Isotherm Fit Multi-panel Figure (SI)
# ===================================================================

def langmuir(P, K, n_m):
    """Langmuir isotherm: q = n_m * K * P / (1 + K * P)."""
    return n_m * K * P / (1 + K * P)


def step4_isotherm_multipanel(bkt_dir: Path, fig_dir: Path):
    """Generate multi-panel isotherm fit figure for SI."""
    print("\n" + "=" * 70)
    print("STEP 4: Isotherm Fit Multi-panel (SI)")
    print("=" * 70)

    iso_dir = bkt_dir / "isotherm_fits"
    raw_data = pd.read_csv(iso_dir / "pure_component_data_merged.csv")
    fit_params = pd.read_csv(iso_dir / "best_isotherm_fits.csv")

    # Unique MOFs
    mof_list = fit_params["MofName"].unique()
    n_mofs = len(mof_list)
    print(f"  {n_mofs} MOFs to plot")

    # Layout: 7 rows × 3 columns for 21 MOFs
    ncols = 3
    nrows = int(np.ceil(n_mofs / ncols))

    set_publication_style()
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(DOUBLE_COL_INCH, nrows * 1.2),
        squeeze=False,
    )

    # Colors
    ch4_color = "#D62728"   # red
    n2_color = "#1F77B4"    # blue

    # Sort ATC-Cu first
    mof_list_sorted = sorted(mof_list, key=lambda x: 0 if "CoRE-2020" in x else 1)
    for idx, mof_name in enumerate(mof_list_sorted):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        mof_raw = raw_data[raw_data["MofName"] == mof_name]
        mof_fits = fit_params[fit_params["MofName"] == mof_name]

        for gas_name, gas_label, color, marker in [
            ("methane", r"CH$_4$", ch4_color, "o"),
            ("N2", r"N$_2$", n2_color, "s"),
        ]:
            # GCMC data points
            gas_data = mof_raw[mof_raw["GasName"] == gas_name]
            if gas_data.empty:
                continue
            P_data = gas_data["Pressure[bar]"].values
            q_data = gas_data["AbsLoading"].values

            # Fit curve (compute R² for legend label)
            gas_fit = mof_fits[mof_fits["GasName"] == gas_name]
            r2_str = ""
            if not gas_fit.empty:
                K = gas_fit["K"].iloc[0]
                n_m = gas_fit["n_m"].iloc[0]
                r2 = gas_fit["R2"].iloc[0]
                r2_str = f" ($R^2$={r2:.3f})"
                P_fit = np.logspace(np.log10(max(P_data.min(), 0.001)),
                                    np.log10(P_data.max()), 100)
                q_fit = langmuir(P_fit, K, n_m)
                ax.plot(P_fit, q_fit, color=color, linewidth=0.8, zorder=2)

            ax.scatter(
                P_data, q_data,
                color=color, marker=marker, s=12, zorder=3,
                label=f"{gas_label}{r2_str}", edgecolors="none", alpha=0.8,
            )

        ax.set_xscale("log")
        ax.set_title(simplify_mof_id(mof_name), fontsize=7.5, pad=2)
        ax.tick_params(axis="both", labelsize=7.0)

        if col == 0:
            ax.set_ylabel("q (mol/kg)", fontsize=8.0)
        if row == nrows - 1:
            ax.set_xlabel("P (bar)", fontsize=8.0)

        ax.legend(fontsize=6.0, loc="lower right", framealpha=0.8, ncol=1)

    # Hide empty subplots
    for idx in range(n_mofs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    save_figure(fig, "FigSX_isotherm_fits", fig_dir, formats=("png",))
    plt.close(fig)
    print(f"\nStep 4 complete: isotherm multi-panel figure saved.")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Task 3.3: Generate BKT paper figures and tables."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model results dir (default: results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which step to run (default: all).",
    )
    parser.add_argument(
        "--fig-dir", type=str, default=None,
        help="Output directory for figures (default: manuscript/figures/).",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = REPO_ROOT / md
    else:
        md = REPO_ROOT / "results" / "alignn" / "model_ep150"

    bkt_dir = md / "bkt_candidates"

    if args.fig_dir:
        fig_dir = Path(args.fig_dir)
    else:
        # Default: paper repo figures dir
        paper_repo = Path("/home/zhangsd/repos/CBM-MOF-paper")
        fig_dir = paper_repo / "manuscript" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"BKT data: {bkt_dir}")
    print(f"Figures:  {fig_dir}")

    steps = ["1", "2", "3", "4"] if args.step == "all" else [args.step]

    if "1" in steps:
        step1_breakthrough_overlay(bkt_dir, fig_dir)
    if "2" in steps:
        step2_performance_table(bkt_dir, fig_dir)
    if "3" in steps:
        step3_selectivity_comparison(bkt_dir, fig_dir)
    if "4" in steps:
        step4_isotherm_multipanel(bkt_dir, fig_dir)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
