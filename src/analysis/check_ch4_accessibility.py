#!/usr/bin/env python
"""Check CH4 pore accessibility for Top-100 MOF candidates using Zeo++.

Runs Zeo++ volpo analysis with a CH4-sized probe (radius = 1.9 A, diameter = 3.8 A)
on Top-100 candidates that have PLD < 3.8 A. Also runs with He probe (1.32 A) for
comparison. Reports POAV (Probe-Occupiable Accessible Volume) to determine if
CH4 can actually access the pore network.

If POAV_CH4 = 0, the MOF's pores are inaccessible to CH4 despite having nonzero
PLD (which was computed with a smaller probe). This means GCMC adsorption results
for such MOFs may be artifacts.
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
ZEOPP_BIN = Path("/home/zhangsd/repos/zeo++-0.3/network")
CIF_DIR = REPO / "data" / "processed" / "integrated_cifs"
FEATURES_CSV = REPO / "data" / "processed" / "RAC_and_zeo_features_deduplicated.csv"
TOP_DIR = REPO / "results" / "alignn" / "model_ep150" / "top_candidates"
OUTPUT_DIR = REPO / "results" / "alignn" / "model_ep150" / "top_candidates" / "ch4_accessibility"

# Probe radii (half of kinetic diameter)
CH4_PROBE_RADIUS = 1.9   # CH4 kinetic diameter = 3.8 A
HE_PROBE_RADIUS = 1.32   # He kinetic diameter = 2.64 A (standard Zeo++ probe)
N2_PROBE_RADIUS = 1.82   # N2 kinetic diameter = 3.64 A (for reference)
ZEOPP_NPOINTS = 50000    # Monte Carlo sampling points


def load_top100_ids() -> set:
    """Load unique MOF IDs from Top-100 candidate files (union of exp/hypo top50 PSA/VSA)."""
    files = ["exp_top50_psa.csv", "exp_top50_vsa.csv", "hypo_top50_psa.csv", "hypo_top50_vsa.csv"]
    all_ids = set()
    for f in files:
        path = TOP_DIR / f
        if path.exists():
            df = pd.read_csv(path)
            all_ids.update(df["mof_id"].values)
            logger.info("Loaded %d IDs from %s", len(df), f)
        else:
            logger.warning("File not found: %s", path)
    logger.info("Total unique Top-100 MOF IDs: %d", len(all_ids))
    return all_ids


def filter_narrow_pld(all_ids: set, pld_threshold: float = 3.8) -> pd.DataFrame:
    """Filter Top-100 MOFs for those with PLD (Df) < threshold."""
    feat = pd.read_csv(FEATURES_CSV, usecols=["name", "Df", "Di", "POAV", "PONAV"])
    top_feat = feat[feat["name"].isin(all_ids)].copy()
    logger.info("Top MOFs found in feature data: %d / %d", len(top_feat), len(all_ids))

    narrow = top_feat[top_feat["Df"] < pld_threshold].sort_values("Df").reset_index(drop=True)
    logger.info("MOFs with PLD < %.1f A: %d", pld_threshold, len(narrow))
    return narrow


def run_zeopp_volpo(cif_path: Path, probe_radius: float, n_points: int = ZEOPP_NPOINTS) -> dict:
    """Run Zeo++ volpo analysis and parse results.

    Args:
        cif_path: Path to CIF file.
        probe_radius: Probe radius in Angstroms.
        n_points: Number of Monte Carlo sampling points.

    Returns:
        Dictionary with parsed volume results, or dict with NaN values on failure.
    """
    result_keys = [
        "POAV_A3", "POAV_vol_frac", "POAV_cm3_per_g",
        "PONAV_A3", "PONAV_vol_frac", "PONAV_cm3_per_g",
        "unitcell_volume", "density",
    ]
    nan_result = {k: float("nan") for k in result_keys}
    nan_result["success"] = False

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as tmp:
        tmp_path = tmp.name

    cmd = [
        str(ZEOPP_BIN), "-ha",
        "-volpo", str(probe_radius), str(probe_radius), str(n_points),
        tmp_path, str(cif_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.error("Zeo++ timed out for %s (probe=%.2f)", cif_path.name, probe_radius)
        return nan_result
    except Exception as e:
        logger.error("Zeo++ failed for %s: %s", cif_path.name, e)
        return nan_result

    # Parse the output file
    out_path = Path(tmp_path)
    if not out_path.exists():
        logger.error("Zeo++ output file not created for %s", cif_path.name)
        return nan_result

    try:
        content = out_path.read_text()
        # Parse first line: @ filename Unitcell_volume: X Density: X POAV_A^3: X ...
        first_line = content.split("\n")[0]
        parsed = {
            "unitcell_volume": float(first_line.split("Unitcell_volume:")[1].split()[0]),
            "density": float(first_line.split("Density:")[1].split()[0]),
            "POAV_A3": float(first_line.split("POAV_A^3:")[1].split()[0]),
            "POAV_vol_frac": float(first_line.split("POAV_Volume_fraction:")[1].split()[0]),
            "POAV_cm3_per_g": float(first_line.split("POAV_cm^3/g:")[1].split()[0]),
            "PONAV_A3": float(first_line.split("PONAV_A^3:")[1].split()[0]),
            "PONAV_vol_frac": float(first_line.split("PONAV_Volume_fraction:")[1].split()[0]),
            "PONAV_cm3_per_g": float(first_line.split("PONAV_cm^3/g:")[1].split()[0]),
            "success": True,
        }
    except (IndexError, ValueError) as e:
        logger.error("Failed to parse Zeo++ output for %s: %s", cif_path.name, e)
        parsed = nan_result
    finally:
        out_path.unlink(missing_ok=True)

    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Check CH4 pore accessibility for Top-100 MOF candidates using Zeo++.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pld-threshold", type=float, default=3.8,
        help="PLD threshold in Angstroms (default: 3.8, CH4 kinetic diameter)",
    )
    parser.add_argument(
        "--ch4-probe", type=float, default=CH4_PROBE_RADIUS,
        help="CH4 probe radius in Angstroms (default: 1.9)",
    )
    parser.add_argument(
        "--he-probe", type=float, default=HE_PROBE_RADIUS,
        help="He probe radius in Angstroms (default: 1.32)",
    )
    parser.add_argument(
        "--n-points", type=int, default=ZEOPP_NPOINTS,
        help="Monte Carlo sampling points for Zeo++ (default: 50000)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--all-top100", action="store_true",
        help="Run on ALL Top-100 MOFs, not just those with PLD < threshold",
    )
    args = parser.parse_args()

    # Validate Zeo++ binary
    if not ZEOPP_BIN.exists():
        logger.error("Zeo++ binary not found at %s", ZEOPP_BIN)
        return

    # Load and filter MOFs
    all_ids = load_top100_ids()
    if args.all_top100:
        feat = pd.read_csv(FEATURES_CSV, usecols=["name", "Df", "Di", "POAV", "PONAV"])
        narrow = feat[feat["name"].isin(all_ids)].sort_values("Df").reset_index(drop=True)
        logger.info("Running on ALL %d Top-100 MOFs", len(narrow))
    else:
        narrow = filter_narrow_pld(all_ids, args.pld_threshold)

    if narrow.empty:
        logger.info("No MOFs found below PLD threshold. Nothing to do.")
        return

    # Run Zeo++ for each MOF
    results = []
    n_total = len(narrow)
    for i, row in narrow.iterrows():
        mof_id = row["name"]
        pld = row["Df"]
        cif_path = CIF_DIR / f"{mof_id}.cif"

        if not cif_path.exists():
            logger.warning("CIF not found: %s", cif_path)
            continue

        logger.info("[%d/%d] Processing %s (PLD=%.3f A)", i + 1, n_total, mof_id, pld)

        # Run with CH4 probe
        ch4_result = run_zeopp_volpo(cif_path, args.ch4_probe, args.n_points)

        # Run with He probe
        he_result = run_zeopp_volpo(cif_path, args.he_probe, args.n_points)

        results.append({
            "mof_id": mof_id,
            "PLD_Df": pld,
            "Di": row["Di"],
            "original_POAV": row["POAV"],
            "original_PONAV": row["PONAV"],
            # CH4 probe results
            "CH4_POAV_A3": ch4_result["POAV_A3"],
            "CH4_POAV_vol_frac": ch4_result["POAV_vol_frac"],
            "CH4_POAV_cm3_per_g": ch4_result["POAV_cm3_per_g"],
            "CH4_PONAV_A3": ch4_result["PONAV_A3"],
            "CH4_PONAV_vol_frac": ch4_result["PONAV_vol_frac"],
            "CH4_accessible": ch4_result["POAV_A3"] > 0 if ch4_result["success"] else None,
            # He probe results
            "He_POAV_A3": he_result["POAV_A3"],
            "He_POAV_vol_frac": he_result["POAV_vol_frac"],
            "He_POAV_cm3_per_g": he_result["POAV_cm3_per_g"],
            "He_PONAV_A3": he_result["PONAV_A3"],
            "He_PONAV_vol_frac": he_result["PONAV_vol_frac"],
            "He_accessible": he_result["POAV_A3"] > 0 if he_result["success"] else None,
        })

    # Save results
    df_results = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "ch4_accessibility_results.csv"
    df_results.to_csv(output_csv, index=False)
    logger.info("Results saved to %s", output_csv)

    # Print summary
    n_total_analyzed = len(df_results)
    n_ch4_accessible = df_results["CH4_accessible"].sum()
    n_ch4_inaccessible = n_total_analyzed - n_ch4_accessible
    n_he_accessible = df_results["He_accessible"].sum()

    logger.info("=" * 70)
    logger.info("SUMMARY: CH4 Pore Accessibility for Top-100 Candidates (PLD < %.1f A)", args.pld_threshold)
    logger.info("=" * 70)
    logger.info("Total MOFs analyzed: %d", n_total_analyzed)
    logger.info("CH4-accessible (POAV > 0 with %.2f A probe): %d", args.ch4_probe, n_ch4_accessible)
    logger.info("CH4-inaccessible (POAV = 0): %d", n_ch4_inaccessible)
    logger.info("He-accessible (POAV > 0 with %.2f A probe): %d", args.he_probe, n_he_accessible)
    logger.info("-" * 70)

    # Detailed breakdown
    if n_ch4_inaccessible > 0:
        inacc = df_results[~df_results["CH4_accessible"]]
        logger.info("CH4-INACCESSIBLE MOFs (GCMC results may be artifacts):")
        for _, r in inacc.iterrows():
            logger.info("  %s | PLD=%.3f | CH4_POAV=%.1f | He_POAV=%.1f",
                        r["mof_id"], r["PLD_Df"], r["CH4_POAV_A3"], r["He_POAV_A3"])

    if n_ch4_accessible > 0:
        acc = df_results[df_results["CH4_accessible"]]
        logger.info("CH4-ACCESSIBLE MOFs (pores accessible despite narrow PLD):")
        for _, r in acc.iterrows():
            logger.info("  %s | PLD=%.3f | CH4_POAV=%.1f | He_POAV=%.1f",
                        r["mof_id"], r["PLD_Df"], r["CH4_POAV_A3"], r["He_POAV_A3"])


if __name__ == "__main__":
    main()
