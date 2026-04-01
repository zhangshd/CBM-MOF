"""
submit_gcmc_validation.py — Submit GCMC + Widom SLURM jobs for Top candidates.

Submits RASPA3 GCMC and RASPA2 Widom insertion jobs.

GCMC parameters (mirror full-library screening):
  Temperatures: 298 K
  Pressures: 0.1, 1, 10 bar (1e4, 1e5, 1e6 Pa)
  Adsorbate: CH4/N2 mixture (configurable via --composition)
  Force field: DreidingTraPPEJson
  CPUs: 190 per job
  Partition: C9654

Usage:
    python src/alignn/submit_gcmc_validation.py                          # default 20:80
    python src/alignn/submit_gcmc_validation.py --composition 50:50      # equimolar
    python src/alignn/submit_gcmc_validation.py --cif-dir path/to/cifs   # custom CIF dir
    python src/alignn/submit_gcmc_validation.py --gcmc-only              # skip Widom
    python src/alignn/submit_gcmc_validation.py --test                   # dry run
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parents[2]
GCMC_SRC  = REPO_ROOT / "src" / "gcmc"
GCMC_SCRIPT_DIR  = GCMC_SRC
FORCE_FIELD_DIR  = GCMC_SCRIPT_DIR / "DreidingTraPPEJson"
WIDOM_FORCE_FIELD_DIR = GCMC_SCRIPT_DIR / "DREIDING"

# NOTE: cifs/ symlinks point to MOF-HTS/data/processed/integrated_cifs (with space group)
# NOT to all_graphs_grids/ (which lacks _symmetry_space_group → RASPA2 fails)
CIF_DIR     = str(REPO_ROOT / "results" / "alignn" / "model_ep150" / "top_candidates" / "cifs_all_top")
OUT_DIR     = str(REPO_ROOT / "results" / "alignn" / "model_ep150" / "process_candidates")

# Custom simulation param files (same as used in exp03b full-library screening)
SIMULATION_PARAMS_FILE  = str(REPO_ROOT / "configs" / "custom_simulation.json")
FORCE_FIELD_PARAMS_FILE = str(REPO_ROOT / "configs" / "custom_force_field.json")
SIMULATION_WIDOM_PARAMS = str(REPO_ROOT / "configs" / "custom_widom_simulation.json")
COMPONENT_WIDOM_PARAMS  = str(REPO_ROOT / "configs" / "custom_widom_component.json")

# ---------------------------------------------------------------------------
# GCMC / Widom parameters
# ---------------------------------------------------------------------------
BATCH_SIZE       = 100                           # larger batches → fewer jobs
TEMPERATURES     = [298.0]                      # K
PRESSURES_GCMC   = [1e4, 1e5, 1e6]             # Pa → 0.1, 1, 10 bar
PRESSURES_WIDOM  = [0.0]                        # Pa (Widom insertion dummy)
ADSORBATE_GCMC_2080 = [{"molecules": ["methane", "N2"], "mol_fractions": [0.2, 0.8]}]
ADSORBATE_GCMC_5050 = [{"molecules": ["methane", "N2"], "mol_fractions": [0.5, 0.5]}]
ADSORBATE_GCMC   = ADSORBATE_GCMC_2080  # default, overridden by --composition
WIDOM_MOLECULES  = ["methane", "N2"]
N_CPUS_GCMC      = 100                           # scaled for ~15 concurrent jobs on 1520 cores
N_CPUS_WIDOM     = 100
PARTITION        = "C9654"


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def submit_gcmc(output_dir: str, dry_run: bool) -> None:
    """Submit RASPA3 GCMC jobs for Top candidates."""
    sys.path.insert(0, str(GCMC_SRC))
    try:
        from raspa3_batch_slurm_submitter import main as raspa3_batch_slurm_submitter
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
    sys.path.insert(0, str(GCMC_SRC))
    try:
        from raspa2_widom_batch_slurm_submitter import main as raspa2_widom_batch_slurm_submitter
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
        REWRITE_CHARGE_COLUMN_FOR_RASPA2=False,
        N_CPUS=N_CPUS_WIDOM,
        PARTITION=PARTITION,
        DRY_RUN=dry_run,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global CIF_DIR, OUT_DIR, ADSORBATE_GCMC

    parser = argparse.ArgumentParser(
        description="Submit GCMC + Widom SLURM jobs for Top candidates."
    )
    parser.add_argument("--test", action="store_true",
                        help="Dry run: generate SLURM scripts but do not submit.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model-specific results dir (e.g. results/alignn/model_ep220). "
                             "Overrides CIF_DIR and OUT_DIR.")
    parser.add_argument("--cif-dir", type=str, default=None,
                        help="Custom CIF directory (overrides default).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Custom output directory (overrides default).")
    parser.add_argument("--composition", type=str, default="20:80",
                        choices=["20:80", "50:50"],
                        help="CH4:N2 mole fraction (default: 20:80).")
    parser.add_argument("--gcmc-only", action="store_true",
                        help="Submit only GCMC jobs, skip Widom.")
    parser.add_argument("--widom-only", action="store_true",
                        help="Submit only Widom jobs, skip GCMC.")
    args = parser.parse_args()
    dry_run = args.test

    # Composition
    if args.composition == "50:50":
        ADSORBATE_GCMC = ADSORBATE_GCMC_5050
        comp_tag = "5050"
    else:
        ADSORBATE_GCMC = ADSORBATE_GCMC_2080
        comp_tag = "2080"

    if args.model_dir:
        _md = Path(args.model_dir)
        if not _md.is_absolute():
            _md = REPO_ROOT / _md
        CIF_DIR = str(_md / "top_candidates" / "cifs_all_top")
        OUT_DIR = str(_md / "process_candidates")

    if args.cif_dir:
        _cd = Path(args.cif_dir)
        if not _cd.is_absolute():
            _cd = REPO_ROOT / _cd
        CIF_DIR = str(_cd)

    if args.out_dir:
        _od = Path(args.out_dir)
        if not _od.is_absolute():
            _od = REPO_ROOT / _od
        OUT_DIR = str(_od)

    # Output subdirs include composition tag for 50:50
    if comp_tag == "5050":
        gcmc_subdir = f"gcmc_DreidingTraPPEJson_{comp_tag}"
    else:
        gcmc_subdir = "gcmc_DreidingTraPPEJson"
    gcmc_out  = str(Path(OUT_DIR) / gcmc_subdir)
    widom_out = str(Path(OUT_DIR) / "widom_DREIDING")

    Path(gcmc_out).mkdir(parents=True, exist_ok=True)
    if not args.gcmc_only:
        Path(widom_out).mkdir(parents=True, exist_ok=True)

    # Verify CIF directory has files
    cif_path = Path(CIF_DIR)
    cif_count = len(list(cif_path.glob("*.cif"))) if cif_path.exists() else 0
    print(f"CIF directory : {CIF_DIR}")
    print(f"CIF count     : {cif_count}")
    print(f"Composition   : CH4:N2 = {args.composition}")
    print(f"Mol fractions : {ADSORBATE_GCMC[0]['mol_fractions']}")
    if cif_count == 0:
        print("[ERROR] No CIF files found.", flush=True)
        return

    if not args.widom_only:
        submit_gcmc(gcmc_out, dry_run)
    if not args.gcmc_only:
        submit_widom(widom_out, dry_run)

    if not dry_run:
        print("\n[IMPORTANT] Record SLURM Job IDs in HANDOFF.md!", flush=True)
    else:
        print("\n[DRY RUN] SLURM scripts generated. No jobs submitted.", flush=True)
        if not args.widom_only:
            print(f"  GCMC scripts  : {gcmc_out}", flush=True)
        if not args.gcmc_only:
            print(f"  Widom scripts : {widom_out}", flush=True)


if __name__ == "__main__":
    main()
