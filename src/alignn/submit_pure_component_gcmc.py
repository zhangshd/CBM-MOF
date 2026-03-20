"""
submit_pure_component_gcmc.py — Task 3.0: Submit pure-component GCMC SLURM jobs
for BKT Top-10 candidates.

Runs pure CH4 and pure N2 GCMC simulations at 10 log-spaced pressures
(0.01–10 bar) for the 20 MOFs selected by select_final_top10.py.

Uses the same MOF-HTS submitter interface as submit_gcmc_validation.py.

Pure-component GCMC parameters:
  Temperature: 298 K
  Pressures: 10 log-spaced from 0.01–10 bar (Pa: 1000–1000000)
  Adsorbates: Pure CH4 and pure N2 (separate runs)
  Force field: DreidingTraPPEJson
  CPUs: 64 per job
  Partition: C9654

Usage:
    python src/alignn/submit_pure_component_gcmc.py                   # submit real jobs
    python src/alignn/submit_pure_component_gcmc.py --test            # dry run
    python src/alignn/submit_pure_component_gcmc.py --gas methane     # CH4 only
    python src/alignn/submit_pure_component_gcmc.py --gas N2          # N2 only
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
GCMC_SRC = REPO_ROOT / "src" / "gcmc"
FORCE_FIELD_DIR = GCMC_SRC / "DreidingTraPPEJson"

# Custom simulation param files (same as used in full-library screening)
SIMULATION_PARAMS_FILE = str(REPO_ROOT / "configs" / "custom_simulation.json")
FORCE_FIELD_PARAMS_FILE = str(REPO_ROOT / "configs" / "custom_force_field.json")

# ---------------------------------------------------------------------------
# GCMC parameters
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 50
TEMPERATURES = [298.0]  # K

# 10 log-spaced pressures from 0.01 to 10 bar, in Pa
PRESSURES_BAR = np.logspace(np.log10(0.01), np.log10(10), 10)
PRESSURES_PA = [round(p * 1e5) for p in PRESSURES_BAR]
# = [1000, 2154, 4642, 10000, 21544, 46416, 100000, 215443, 464159, 1000000]

# Pure component adsorbate definitions
ADSORBATE_CH4 = [{"molecules": ["methane"], "mol_fractions": [1.0]}]
ADSORBATE_N2 = [{"molecules": ["N2"], "mol_fractions": [1.0]}]

DEFAULT_N_CPUS = 200
DEFAULT_PARTITION = "C9654"


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_pure_component(
    gas_name: str,
    cif_dir: str,
    output_base: str,
    dry_run: bool,
    batch_size: int,
    n_cpus: int,
    partition: str,
) -> None:
    """Submit pure-component GCMC jobs for a single gas."""
    sys.path.insert(0, str(GCMC_SRC))
    try:
        from raspa3_batch_slurm_submitter import main as raspa3_batch_slurm_submitter
    except ImportError:
        print("[ERROR] MOF-HTS raspa3_batch_slurm_submitter not available.")
        return

    if gas_name == "methane":
        adsorbate = ADSORBATE_CH4
    elif gas_name == "N2":
        adsorbate = ADSORBATE_N2
    else:
        raise ValueError(f"Unknown gas: {gas_name}")

    output_dir = str(Path(output_base) / gas_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Pure-Component GCMC] Gas: {gas_name}")
    print(f"{'='*60}")
    print(f"  CIF dir     : {cif_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Pressures   : {len(PRESSURES_PA)} points, "
          f"{PRESSURES_PA[0]}–{PRESSURES_PA[-1]} Pa")
    print(f"  Pressures (bar): {[f'{p:.4f}' for p in PRESSURES_BAR]}")
    print(f"  CPUs/job    : {n_cpus}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Dry run     : {dry_run}")

    raspa3_batch_slurm_submitter(
        cif_dir, output_dir, batch_size,
        TEMPERATURES, PRESSURES_PA, adsorbate,
        FORCE_FIELD_DIR, SIMULATION_PARAMS_FILE, FORCE_FIELD_PARAMS_FILE,
        n_cpus, partition, dry_run,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.0: Submit pure-component GCMC jobs for BKT candidates."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Dry run: generate SLURM scripts but do not submit."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model-specific results dir (e.g. results/alignn/model_ep150)."
    )
    parser.add_argument(
        "--gas", type=str, default=None, choices=["methane", "N2"],
        help="Submit only this gas (default: both)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Number of CIFs per batch (default: 50)."
    )
    parser.add_argument(
        "--n-cpus", type=int, default=DEFAULT_N_CPUS,
        help="Maximum parallel workers per submitted job (default: 200)."
    )
    parser.add_argument(
        "--partition", type=str, default=DEFAULT_PARTITION,
        help="SLURM partition name (default: C9654)."
    )
    parser.add_argument(
        "--bkt-dir", type=str, default="bkt_candidates",
        help="BKT candidates subdir name under model-dir (default: bkt_candidates)."
    )
    args = parser.parse_args()
    dry_run = args.test

    # Resolve paths
    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = REPO_ROOT / md
    else:
        md = REPO_ROOT / "results" / "alignn" / "model_ep150"

    cif_dir = str(md / args.bkt_dir / "cifs")
    output_base = str(md / args.bkt_dir / "gcmc_pure_component")

    # Verify CIFs
    cif_path = Path(cif_dir)
    cif_count = len(list(cif_path.glob("*.cif"))) if cif_path.exists() else 0
    print(f"CIF directory: {cif_dir}")
    print(f"CIF count    : {cif_count}")
    if cif_count == 0:
        print("[ERROR] No CIF files found. Run select_final_top10.py first.")
        return

    # Submit
    gases = [args.gas] if args.gas else ["methane", "N2"]
    for gas in gases:
        submit_pure_component(
            gas,
            cif_dir,
            output_base,
            dry_run,
            batch_size=args.batch_size,
            n_cpus=args.n_cpus,
            partition=args.partition,
        )

    if not dry_run:
        print(f"\n{'='*60}")
        print("[IMPORTANT] Record SLURM Job IDs in HANDOFF.md!")
        print(f"{'='*60}")
    else:
        print(f"\n[DRY RUN] SLURM scripts generated. No jobs submitted.")
        print(f"  Output base: {output_base}")


if __name__ == "__main__":
    main()
