"""
run_bkt_top_candidates.py — Task 3.2: Batch BKT breakthrough simulation
for Top-10 PSA + Top-10 VSA + ATC-Cu benchmark.

Runs 22 simulations total:
  - PSA Top-10: feed_pressure=10 bar, CH4:N2=20:80
  - VSA Top-10: feed_pressure=1 bar,  CH4:N2=20:80
  - ATC-Cu:     both PSA (10 bar) and VSA (1 bar)

Isotherm parameters from best_isotherm_fits.csv (all Langmuir →
BKT isomodel="Langmuir-Freundlich" with ni=[1,1]).

Adsorbent density (rho_s) from pymatgen Structure.density (g/cm³ × 1000 → kg/m³).

Usage:
    python src/alignn/run_bkt_top_candidates.py
    python src/alignn/run_bkt_top_candidates.py --model-dir results/alignn/model_ep150
    python src/alignn/run_bkt_top_candidates.py --dry-run   # print params only
    python src/alignn/run_bkt_top_candidates.py --rebuild-curve-cache
"""

import argparse
import collections
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BKT_ROOT = REPO_ROOT / "src" / "bkt"

# Add BKT to path
sys.path.insert(0, str(REPO_ROOT / "src"))
from alignn.bkt_curve_cache import build_curve_dataframe, rebuild_curve_cache


def solve_breakthrough_robust(localparam, min_points=10):
    """Wrapper around BKT solver with fallback methods for stiff systems.

    Tries BDF first (default), then Radau, then LSODA, with progressively
    relaxed tolerances. Returns the first successful outcome.
    """
    import scipy.integrate
    import bkt.src.model as model

    param = localparam
    x0_all = model.init(param)
    x0 = np.hstack([x0_all[item] for item in param.state_names])
    breakthroughmodel = model.oadesmodel
    ev = np.linspace(0, param.norm_tbreak, param.tN + 1, endpoint=True)
    t_span = (0, param.norm_tbreak)

    # Solver strategies: (method, rtol, atol, label)
    # Tight tolerances first for better mass balance; relax only as fallback
    strategies = [
        ("BDF",   1e-6, 1e-9, "BDF tight"),
        ("Radau", 1e-6, 1e-9, "Radau tight"),
        ("BDF",   1e-4, 1e-7, "BDF default"),
        ("Radau", 1e-4, 1e-7, "Radau default"),
        ("Radau", 1e-3, 1e-6, "Radau relaxed"),
        ("LSODA", 1e-3, 1e-6, "LSODA relaxed"),
    ]

    outcome = None
    for method, rtol, atol, label in strategies:
        try:
            outcome = scipy.integrate.solve_ivp(
                breakthroughmodel, t_span, x0,
                vectorized=False, t_eval=ev, method=method,
                rtol=rtol, atol=atol, args=(param,),
            )
            if outcome.success and len(outcome.t) >= min_points:
                print(f"    Solver OK: {label} ({len(outcome.t)} points)")
                return outcome
        except Exception as e:
            print(f"    Solver {label} exception: {e}")

    # Return best failed attempt (last one), or None if all raised
    if outcome is None:
        print(f"    ERROR: All solver strategies raised exceptions")
    else:
        print(f"    WARNING: All solver strategies failed, returning last attempt")
    return outcome

# ATC-Cu CIF (benchmark)
ATC_CU_CIF = Path(
    "/home/zhangsd/repos/MOF-HTS/src/gcmc/examples/dup_demo_ATC-Cu/"
    "CoRE-2020[Cu][pts]3[ASR]1.cif"
)
ATC_CU_NAME = "CoRE-2020[Cu][pts]3[ASR]1"

# ---------------------------------------------------------------------------
# Standard bed parameters (same for all MOFs)
# ---------------------------------------------------------------------------
STANDARD_BED = {
    "D": 9e-3,          # column diameter [m]
    "L": 0.15,          # column length [m]
    "epsilon": 0.4,     # void fraction
    "rp": 2e-4,         # particle radius [m]
    "r_pore": 25e-9,    # pore radius [m]
    "tor": 3,           # tortuosity
    "epsilon_p": 0.35,  # particle porosity
    "tN": 500,          # time points (PSA: 50s/pt, VSA: 24s/pt)
    "Ta": 298,          # temperature [K]
    "R": 8.314,         # gas constant
}

# Process configurations (N is per-process: PSA@10bar needs coarser grid to avoid stiffness)
PROCESS_CONFIG = {
    "PSA": {
        "feed_pressure": 10,   # bar (adsorption at 10 bar)
        "tstop_init": 25000,   # initial simulation time [s] (max t_95%≈18.4ks, 1.36× margin)
        "tstop_extend": 5000,  # time extension per iteration [s]
        "N": 30,               # spatial discretisation (Radau handles stiffness)
    },
    "VSA": {
        "feed_pressure": 1,    # bar (adsorption at 1 bar)
        "tstop_init": 12000,   # initial simulation time [s] (max t_95%=8.4ks, 1.43× margin)
        "tstop_extend": 3000,  # time extension per iteration [s]
        "N": 30,               # spatial discretisation
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_rho_s(cif_path: Path) -> float:
    """Calculate adsorbent density from CIF (pymatgen Structure.density g/cm³ → kg/m³)."""
    from pymatgen.core import Structure

    struct = Structure.from_file(str(cif_path))
    rho_s = struct.density * 1000  # g/cm³ → kg/m³
    return rho_s


def build_mods(
    mof_name: str,
    process: str,
    iso_params: dict,
    rho_s: float,
) -> dict:
    """Build BKT parameter dict for one simulation."""
    from bkt.src.util import calculate_ki_Dax

    cfg = PROCESS_CONFIG[process]
    bed = STANDARD_BED

    mods = collections.defaultdict()
    mods["nocomponents"] = 2
    mods["feed_yi"] = [0.2, 0.8]  # CH4:N2 = 20:80
    mods["ini_yi"] = [1e-10, 1e-10]
    mods["isomodel"] = "Langmuir-Freundlich"
    mods["component_names"] = ["CH4", "N2"]

    # Isotherm: Langmuir params → BKT Langmuir-Freundlich with ni=1
    # pyGAPS Langmuir: q = n_m * K * P / (1 + K * P)
    # BKT L-F: uses bi (affinity, bar⁻¹) and qsi (saturation, mol/kg)
    mods["bi"] = [iso_params["K_CH4"], iso_params["K_N2"]]
    mods["qsi"] = [iso_params["n_m_CH4"], iso_params["n_m_N2"]]
    mods["ni"] = [1, 1]  # Langmuir = L-F with n=1

    # No heat effects (isothermal)
    mods["Hi"] = [0, 0]

    # Bed geometry
    mods["R"] = bed["R"]
    mods["D"] = bed["D"]
    mods["A"] = np.pi * (bed["D"] / 2) ** 2
    mods["L"] = bed["L"]
    mods["epsilon"] = bed["epsilon"]
    mods["rp"] = bed["rp"]
    mods["Ta"] = bed["Ta"]

    # Process pressures
    mods["feed_pressure"] = cfg["feed_pressure"]

    # Flow rate: 10 mL/min at 1 bar equivalent
    flow_rate_ml_min = 10 / cfg["feed_pressure"]
    flow_rate_m3_s = flow_rate_ml_min * 1e-6 / 60
    mods["vfeed"] = flow_rate_m3_s / mods["A"] / mods["epsilon"]

    # Adsorbent density
    mods["rho_s"] = rho_s

    # Mass transfer and dispersion
    k1, k2, Dax = calculate_ki_Dax(
        "CH4", "N2", bed["Ta"], cfg["feed_pressure"],
        bed["rp"], bed["r_pore"], bed["epsilon_p"], bed["tor"],
        mods["vfeed"],
    )
    mods["ki"] = [k1, k2]
    mods["DL"] = Dax

    # Simulation time
    mods["bed"] = "Breakthrough"
    mods["tstart"] = 0
    mods["tstop"] = cfg["tstop_init"]
    mods["tbreak"] = 0
    mods["tN"] = bed["tN"]
    mods["N"] = cfg["N"]

    return mods


def run_single_simulation(
    mof_name: str,
    process: str,
    mods: dict,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Run one BKT simulation with iterative time extension."""
    import bkt.src.analysis as analysis
    import bkt.src.params as params
    import bkt.src.plot as plot
    from bkt.src.plot import data_to_state

    label = f"{mof_name} [{process}]"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  rho_s={mods['rho_s']:.1f} kg/m³  P_feed={mods['feed_pressure']} bar")
    print(f"  bi={mods['bi']}  qsi={mods['qsi']}")
    print(f"  vfeed={mods['vfeed']:.6f} m/s  ki={mods['ki']}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping simulation.")
        return {"mof": mof_name, "process": process, "status": "dry_run"}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterative solve until complete breakthrough
    breakthrough_threshold = 0.99
    min_time_points = 10  # Minimum valid solver output
    max_iterations = 20
    iteration = 0
    all_breakthrough = False
    solver_failed_count = 0  # Track consecutive solver failures

    while not all_breakthrough and iteration < max_iterations:
        iteration += 1
        print(f"  Iteration {iteration}: tstop = {mods['tstop']} s  N = {mods['N']}")

        localparam = params.create_param(mods)
        outcome = solve_breakthrough_robust(localparam, min_points=min_time_points)

        # --- Solver validation ---
        n_tpoints = len(outcome.t)
        solver_ok = outcome.success and n_tpoints >= min_time_points
        if not solver_ok:
            solver_failed_count += 1
            print(f"    ALL SOLVERS FAILED: success={outcome.success}, "
                  f"t_points={n_tpoints}/{mods.get('tN', 100)+1}")

            # Adaptive N reduction: if solver fails twice in a row, reduce N
            if solver_failed_count >= 2 and mods["N"] > 10:
                old_N = mods["N"]
                mods["N"] = max(10, mods["N"] - 5)
                print(f"    Reducing N: {old_N} → {mods['N']} (adaptive fallback)")
                solver_failed_count = 0

            # Extend time and retry
            extend = PROCESS_CONFIG[process].get("tstop_extend", 10000)
            mods["tstop"] += extend
            continue

        solver_failed_count = 0  # Reset on success
        data1 = data_to_state(outcome.y, localparam)

        # Check breakthrough for each component
        breakthrough_status = []
        for i in range(localparam.nocomponents):
            attr_name = f"y{chr(65+i)}"
            if hasattr(data1, attr_name):
                y_comp = getattr(data1, attr_name)
                C0 = y_comp[0][-1]
                C1 = y_comp[-1][-1]
                ratio = C1 / C0 if C0 > 0 else 0
                is_bt = ratio >= breakthrough_threshold
                breakthrough_status.append(is_bt)
                status = "OK" if is_bt else f"C/C0={ratio:.4f}"
                print(f"    {mods['component_names'][i]}: {status}")
            else:
                breakthrough_status.append(False)

        all_breakthrough = all(breakthrough_status)
        if not all_breakthrough:
            extend = PROCESS_CONFIG[process].get("tstop_extend", 10000)
            mods["tstop"] += extend

    if iteration >= max_iterations:
        print(f"  WARNING: Max iterations reached!")

    print(f"  Final tstop = {mods['tstop']} s ({iteration} iteration(s))")

    # Final validation: ensure outcome has meaningful time extent
    sim_time_real = outcome.t[-1] * localparam.norm_t0 if len(outcome.t) > 0 else 0
    print(f"  Solver time extent: {sim_time_real:.1f} s ({len(outcome.t)} points)")
    if sim_time_real < 1.0:
        print(f"  ERROR: Solver produced near-zero time extent ({sim_time_real:.4f} s)")
        return {
            "mof": mof_name,
            "process": process,
            "status": "solver_failed",
            "error": f"near-zero time extent: {sim_time_real:.4f} s",
            "iterations": iteration,
            "N_final": mods["N"],
        }

    # Analysis (separate data extraction from plotting for robustness)
    try:
        curve_state = data_to_state(outcome.y, localparam)
        time_min = outcome.t * localparam.norm_t0 / 60
        y_inlet = curve_state.yA[0]
        y_outlet = curve_state.yA[-1]
        curve_df = build_curve_dataframe(
            mof_name=mof_name,
            process=process,
            time_min=time_min,
            cc0_ch4=(y_outlet / y_inlet),
        )
        curve_csv_path = output_dir / "breakthrough_curve_data.csv"
        curve_df.to_csv(curve_csv_path, index=False)

        adsorption_results = analysis.calculate_cumulative_adsorption(
            outcome, localparam, validate=True
        )
        analysis.print_adsorption_summary(adsorption_results, localparam)

        selectivity_results = analysis.calculate_dynamic_selectivity(
            adsorption_results, localparam, reference_component="CH4"
        )
        analysis.print_selectivity_summary(selectivity_results)

        # Save analysis CSV (critical — must succeed)
        csv_path = analysis.save_adsorption_and_selectivity_to_csv(
            adsorption_results, selectivity_results, output_dir=str(output_dir)
        )

        # Extract key metrics (before plotting, in case plots fail)
        result = {
            "mof": mof_name,
            "process": process,
            "status": "success",
            "curve_csv": str(curve_csv_path),
            "tstop_final": mods["tstop"],
            "iterations": iteration,
            "N_final": mods["N"],
            "sim_time_s": sim_time_real,
            "rho_s": mods["rho_s"],
            "feed_pressure": mods["feed_pressure"],
        }

        for comp in mods["component_names"]:
            key = f"q_{comp}_mol_per_kg"
            if comp in adsorption_results.get("cumulative_adsorption", {}):
                result[key] = adsorption_results["cumulative_adsorption"][comp]

        if "selectivity_timeseries" in selectivity_results:
            ts = selectivity_results["selectivity_timeseries"]
            if len(ts) > 0:
                result["alpha_final"] = ts[-1]

        if "validation" in adsorption_results:
            val = adsorption_results["validation"]
            errors = [v.get("relative_error", 0) for v in val.values() if isinstance(v, dict)]
            if errors:
                result["max_mass_balance_error"] = max(abs(e) for e in errors)

        # Plotting (non-critical — failures won't affect data)
        try:
            analysis.plot_selectivity_and_adsorption(
                selectivity_results, adsorption_results, output_dir=str(output_dir)
            )
            analysis.plot_mass_balance_validation(
                adsorption_results, localparam, output_dir=str(output_dir)
            )
            plot.BreakthroughCurvePlot(
                outcome, localparam, output_dir=str(output_dir), time_unit="min"
            )
        except Exception as e_plot:
            print(f"  WARNING: Plotting failed (data saved OK): {e_plot}")

        return result

    except Exception as e:
        print(f"  ERROR in analysis: {e}")
        traceback.print_exc()
        return {
            "mof": mof_name,
            "process": process,
            "status": "error",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.2: Batch BKT simulation for Top candidates."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model results dir (default: results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print parameters only, do not run simulations.",
    )
    parser.add_argument(
        "--mof", type=str, default=None,
        help="Run only this MOF (for debugging).",
    )
    parser.add_argument(
        "--process", type=str, default=None, choices=["PSA", "VSA"],
        help="Run only this process (for debugging).",
    )
    parser.add_argument(
        "--job-index", type=int, default=None,
        help="SLURM array index: run only the i-th simulation from the queue.",
    )
    parser.add_argument(
        "--rebuild-curve-cache", action="store_true",
        help="Rebuild breakthrough_curves_data.csv from per-run curve CSV files.",
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
    cif_dir = bkt_dir / "cifs"
    fit_csv = bkt_dir / "isotherm_fits" / "best_isotherm_fits.csv"
    psa_csv = bkt_dir / "top10_psa.csv"
    vsa_csv = bkt_dir / "top10_vsa.csv"

    if args.rebuild_curve_cache:
        output_csv = bkt_dir / "breakthrough_curves_data.csv"
        merged = rebuild_curve_cache(bkt_dir, output_csv=output_csv)
        print(f"Rebuilt curve cache: {output_csv}")
        print(f"  Rows: {len(merged)}")
        print(
            "  Groups: "
            f"{merged[['process', 'mof']].drop_duplicates().shape[0]}"
        )
        return

    # Load isotherm fits
    print(f"Loading isotherm fits: {fit_csv}")
    fits = pd.read_csv(fit_csv)

    # Build per-MOF isotherm param lookup
    iso_lookup = {}
    for mof in fits["MofName"].unique():
        mof_fits = fits[fits["MofName"] == mof]
        ch4 = mof_fits[mof_fits["GasName"] == "methane"].iloc[0]
        n2 = mof_fits[mof_fits["GasName"] == "N2"].iloc[0]
        iso_lookup[mof] = {
            "K_CH4": ch4["K"],
            "n_m_CH4": ch4["n_m"],
            "K_N2": n2["K"],
            "n_m_N2": n2["n_m"],
            "model": ch4["selected_model"],
        }

    # Load Top-10 lists
    psa_mofs = pd.read_csv(psa_csv)["mof_id"].tolist()
    vsa_mofs = pd.read_csv(vsa_csv)["mof_id"].tolist()

    # Build simulation queue: (mof_name, process, cif_path)
    queue = []

    if args.mof:
        # Debug: single MOF
        processes = [args.process] if args.process else ["PSA", "VSA"]
        for proc in processes:
            cif = cif_dir / f"{args.mof}.cif"
            if not cif.exists():
                cif = ATC_CU_CIF
            queue.append((args.mof, proc, cif))
    else:
        # PSA Top-10
        if not args.process or args.process == "PSA":
            for mof_id in psa_mofs:
                cif = cif_dir / f"{mof_id}.cif"
                queue.append((mof_id, "PSA", cif))

        # VSA Top-10
        if not args.process or args.process == "VSA":
            for mof_id in vsa_mofs:
                cif = cif_dir / f"{mof_id}.cif"
                queue.append((mof_id, "VSA", cif))

        # ATC-Cu benchmark (both processes)
        if not args.process:
            queue.append((ATC_CU_NAME, "PSA", ATC_CU_CIF))
            queue.append((ATC_CU_NAME, "VSA", ATC_CU_CIF))
        elif args.process:
            queue.append((ATC_CU_NAME, args.process, ATC_CU_CIF))

    print(f"\nFull simulation queue: {len(queue)} runs")
    print(f"  PSA: {sum(1 for _, p, _ in queue if p == 'PSA')}")
    print(f"  VSA: {sum(1 for _, p, _ in queue if p == 'VSA')}")

    # SLURM array mode: run only the job-index-th simulation
    if args.job_index is not None:
        if args.job_index < 0 or args.job_index >= len(queue):
            print(f"ERROR: --job-index {args.job_index} out of range [0, {len(queue)-1}]")
            sys.exit(1)
        queue = [queue[args.job_index]]
        print(f"  SLURM array mode: running job index {args.job_index}")

    # Run simulations
    results = []
    for i, (mof_name, process, cif_path) in enumerate(queue):
        print(f"\n[{i+1}/{len(queue)}] {mof_name} — {process}")

        # Check CIF
        if not cif_path.exists():
            print(f"  ERROR: CIF not found: {cif_path}")
            results.append({
                "mof": mof_name, "process": process, "status": "cif_missing"
            })
            continue

        # Check isotherm params
        if mof_name not in iso_lookup:
            print(f"  ERROR: No isotherm params for {mof_name}")
            results.append({
                "mof": mof_name, "process": process, "status": "no_iso_params"
            })
            continue

        # Calculate density
        rho_s = get_rho_s(cif_path)
        print(f"  rho_s = {rho_s:.1f} kg/m³ (from CIF)")

        # Build params
        mods = build_mods(mof_name, process, iso_lookup[mof_name], rho_s)

        # Output directory
        safe_name = mof_name.replace("[", "_").replace("]", "_")
        out_dir = bkt_dir / f"bkt_{process.lower()}" / f"{safe_name}_{process}"

        # Run
        result = run_single_simulation(
            mof_name, process, mods, out_dir, dry_run=args.dry_run
        )
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(results)} simulations")
    print(f"{'='*70}")

    summary_df = pd.DataFrame(results)
    success = summary_df[summary_df["status"] == "success"]
    failed = summary_df[summary_df["status"] != "success"]

    if not args.dry_run:
        print(f"  Success: {len(success)}")
        print(f"  Failed:  {len(failed)}")
        if len(failed) > 0:
            print(f"  Failed MOFs: {failed[['mof', 'process', 'status']].to_string()}")

    summaries_dir = bkt_dir / "summaries"
    job_summaries_dir = summaries_dir / "jobs"
    job_summaries_dir.mkdir(parents=True, exist_ok=True)

    # Save summary (per-job for array mode, combined otherwise)
    if args.job_index is not None:
        summary_path = job_summaries_dir / f"bkt_summary_job{args.job_index:03d}.csv"
    else:
        summary_path = summaries_dir / "bkt_simulation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
