"""
submit_gcmc_validation.py — Task 2.4a: Submit GCMC + Widom SLURM jobs for Top candidates.

Submits RASPA3 GCMC and RASPA2 Widom insertion jobs for the MOFs in
  results/alignn/top_candidates/cifs/

Uses the same MOF-HTS submitter interface as exp03b_gcmc_batch_submission.py.

GCMC parameters (mirror full-library screening):
  Temperatures: 298 K
  Pressures: 0.1, 1, 10 bar (1e4, 1e5, 1e6 Pa)
  Adsorbate: CH4/N2 mixture (20/80 mol%)
  Force field: DreidingTraPPEJson
  CPUs: 64 per job (cluster has ample CPU resources)
  Partition: C9654

Usage:
    python src/alignn/submit_gcmc_validation.py          # submit real jobs
    python src/alignn/submit_gcmc_validation.py --test   # dry run (generate scripts, no submit)
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[2]
MOF_HTS_SRC  = Path("/home/zhangsd/repos/MOF-HTS/src")
GCMC_SCRIPT_DIR  = MOF_HTS_SRC / "gcmc"
FORCE_FIELD_DIR  = GCMC_SCRIPT_DIR / "DreidingTraPPEJson"
WIDOM_FORCE_FIELD_DIR = GCMC_SCRIPT_DIR / "DREIDING"

# NOTE: cifs/ symlinks point to MOF-HTS/data/processed/integrated_cifs (with space group)
# NOT to all_graphs_grids/ (which lacks _symmetry_space_group → RASPA2 fails)
CIF_DIR     = str(REPO_ROOT / "results" / "alignn" / "top_candidates" / "cifs")
OUT_DIR     = str(REPO_ROOT / "results" / "alignn" / "gcmc_top_candidates")

# Custom simulation param files (same as used in exp03b full-library screening)
SIMULATION_PARAMS_FILE  = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_simulation.json"
FORCE_FIELD_PARAMS_FILE = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_force_field.json"
SIMULATION_WIDOM_PARAMS = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_widom_simulation.json"
COMPONENT_WIDOM_PARAMS  = "/home/zhangsd/repos/MOF-HTS/examples/custom_params/custom_widom_component.json"

# ---------------------------------------------------------------------------
# GCMC / Widom parameters
# ---------------------------------------------------------------------------
BATCH_SIZE       = 50
TEMPERATURES     = [298.0]                      # K
PRESSURES_GCMC   = [1e4, 1e5, 1e6]             # Pa → 0.1, 1, 10 bar
PRESSURES_WIDOM  = [0.0]                        # Pa (Widom insertion dummy)
ADSORBATE_GCMC   = [{"molecules": ["methane", "N2"], "mol_fractions": [0.2, 0.8]}]
WIDOM_MOLECULES  = ["methane", "N2"]
N_CPUS_GCMC      = 64
N_CPUS_WIDOM     = 64
PARTITION        = "C9654"


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def submit_gcmc(output_dir: str, dry_run: bool) -> None:
    """Submit RASPA3 GCMC jobs for Top candidates."""
    sys.path.insert(0, str(MOF_HTS_SRC))
    try:
        from gcmc.raspa3_batch_slurm_submitter import main as raspa3_batch_slurm_submitter
    except ImportError:
        print("[WARN] MOF-HTS raspa3_batch_slurm_submitter not available; skipping GCMC submission.")
        return

    print(f"\n[GCMC] Submitting RASPA3 jobs …", flush=True)
    print(f"  CIF dir    : {CIF_DIR}", flush=True)
    print(f"  Output dir : {output_dir}", flush=True)
    print(f"  Pressures  : {PRESSURES_GCMC} Pa", flush=True)
    print(f"  CPUs/job   : {N_CPUS_GCMC}", flush=True)
    print(f"  Dry run    : {dry_run}", flush=True)

    raspa3_batch_slurm_submitter(
        CIF_DIR, output_dir, BATCH_SIZE,
        TEMPERATURES, PRESSURES_GCMC, ADSORBATE_GCMC,
        FORCE_FIELD_DIR, SIMULATION_PARAMS_FILE, FORCE_FIELD_PARAMS_FILE,
        N_CPUS_GCMC, PARTITION, dry_run,
    )


def submit_widom(output_dir: str, dry_run: bool) -> None:
    """Submit RASPA2 Widom insertion jobs for Top candidates."""
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

    print(f"\n[Widom] Submitting RASPA2 Widom jobs …", flush=True)
    print(f"  CIF dir    : {CIF_DIR}", flush=True)
    print(f"  Output dir : {output_dir}", flush=True)
    print(f"  Molecules  : {WIDOM_MOLECULES}", flush=True)
    print(f"  CPUs/job   : {N_CPUS_WIDOM}", flush=True)
    print(f"  Dry run    : {dry_run}", flush=True)

    raspa2_widom_batch_slurm_submitter(
        CIF_DIR, output_dir, BATCH_SIZE,
        TEMPERATURES, PRESSURES_WIDOM, WIDOM_MOLECULES,
        WIDOM_FORCE_FIELD_DIR, force_field_params,
        SIMULATION_WIDOM_PARAMS, COMPONENT_WIDOM_PARAMS,
        N_CPUS_WIDOM, PARTITION, dry_run,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 2.4a: Submit GCMC + Widom SLURM jobs for Top candidates."
    )
    parser.add_argument("--test", action="store_true",
                        help="Dry run: generate SLURM scripts but do not submit.")
    args = parser.parse_args()
    dry_run = args.test

    gcmc_out  = str(Path(OUT_DIR) / "gcmc_DreidingTraPPEJson")
    widom_out = str(Path(OUT_DIR) / "widom_DREIDING")

    Path(gcmc_out).mkdir(parents=True, exist_ok=True)
    Path(widom_out).mkdir(parents=True, exist_ok=True)

    # Verify CIF directory has files
    cif_path = Path(CIF_DIR)
    cif_count = len(list(cif_path.glob("*.cif"))) if cif_path.exists() else 0
    print(f"CIF directory: {CIF_DIR}")
    print(f"CIF count    : {cif_count}")
    if cif_count == 0:
        print("[ERROR] No CIF files found. Run select_top_candidates.py first.", flush=True)
        return

    submit_gcmc(gcmc_out, dry_run)
    submit_widom(widom_out, dry_run)

    if not dry_run:
        print("\n[IMPORTANT] Record SLURM Job IDs in HANDOFF.md!", flush=True)
    else:
        print("\n[DRY RUN] SLURM scripts generated. No jobs submitted.", flush=True)
        print(f"  GCMC scripts  : {gcmc_out}", flush=True)
        print(f"  Widom scripts : {widom_out}", flush=True)


if __name__ == "__main__":
    main()
