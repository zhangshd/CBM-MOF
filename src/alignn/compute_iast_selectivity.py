"""
compute_iast_selectivity.py — Task 3.4: IAST selectivity from pure-component Langmuir fits.

Reads Langmuir parameters (K, n_m) from best_isotherm_fits.csv, constructs
pyGAPS ModelIsotherms for each MOF's CH4 and N2, then calls IAST to compute
mixed-component selectivity at CBM conditions (CH4:N2 = 20:80).

Output: bkt_candidates/iast_selectivity.csv

Usage:
    conda run -n mofmthnn python src/alignn/compute_iast_selectivity.py
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pygaps
try:
    from pygaps.prediction import iast_point_fraction
except ImportError:  # pragma: no cover - backward compatibility
    from pygaps.iast import iast_point_fraction

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]

# CBM feed composition
Y_CH4 = 0.2
Y_N2 = 0.8

# Process conditions
PROCESS_CONDITIONS = {
    "PSA": {"total_pressure": 10.0},  # bar
    "VSA": {"total_pressure": 1.0},   # bar
}


def langmuir_points(K, n_m, p_max=15.0, n_points=100):
    """Generate Langmuir isotherm data points for pyGAPS fitting."""
    P = np.linspace(0.001, p_max, n_points)
    q = n_m * K * P / (1 + K * P)
    return P, q


def build_model_isotherm(K, n_m, material, adsorbate, temperature=298):
    """Create a pyGAPS ModelIsotherm from Langmuir parameters."""
    P, q = langmuir_points(K, n_m)
    iso = pygaps.ModelIsotherm(
        pressure=P.tolist(),
        loading=q.tolist(),
        model="Langmuir",
        material=material,
        adsorbate=adsorbate,
        temperature=temperature,
        pressure_unit="bar",
        loading_basis="molar",
        loading_unit="mmol",
        material_basis="mass",
        material_unit="g",
    )
    return iso


def compute_iast_selectivity(fits_csv: Path, output_csv: Path):
    """Compute IAST selectivity for all MOFs in best_isotherm_fits.csv."""
    df = pd.read_csv(fits_csv)

    # Group by MOF — need CH4 and N2 rows
    mof_names = df["MofName"].unique()
    print(f"Found {len(mof_names)} MOFs in {fits_csv.name}")

    results = []

    for mof in mof_names:
        mof_data = df[df["MofName"] == mof]

        # Extract Langmuir params for CH4 and N2
        ch4_row = mof_data[mof_data["gas_key"].str.contains("methane")]
        n2_row = mof_data[mof_data["gas_key"].str.contains("N2")]

        if ch4_row.empty or n2_row.empty:
            print(f"  WARNING: {mof} missing CH4 or N2 data, skipping")
            continue

        ch4_row = ch4_row.iloc[0]
        n2_row = n2_row.iloc[0]

        # Only use Langmuir model fits
        if ch4_row["selected_model"] != "Langmuir" or n2_row["selected_model"] != "Langmuir":
            print(f"  WARNING: {mof} not using Langmuir model, skipping")
            continue

        K_ch4, nm_ch4 = ch4_row["K"], ch4_row["n_m"]
        K_n2, nm_n2 = n2_row["K"], n2_row["n_m"]

        # Build pyGAPS isotherms
        iso_ch4 = build_model_isotherm(K_ch4, nm_ch4, mof, "methane")
        iso_n2 = build_model_isotherm(K_n2, nm_n2, mof, "nitrogen")

        row_result = {"MofName": mof}

        for process, cond in PROCESS_CONDITIONS.items():
            total_p = cond["total_pressure"]
            try:
                loadings = iast_point_fraction(
                    [iso_ch4, iso_n2],
                    gas_mole_fraction=[Y_CH4, Y_N2],
                    total_pressure=total_p,
                    warningoff=True,
                )
                q_ch4 = loadings[0]  # mmol/g
                q_n2 = loadings[1]   # mmol/g
                alpha_iast = (q_ch4 / q_n2) * (Y_N2 / Y_CH4) if q_n2 > 0 else np.nan
            except Exception as e:
                print(f"  WARNING: IAST failed for {mof} {process}: {e}")
                q_ch4, q_n2, alpha_iast = np.nan, np.nan, np.nan

            row_result[f"alpha_IAST_{process}"] = alpha_iast
            row_result[f"q_CH4_IAST_{process}"] = q_ch4
            row_result[f"q_N2_IAST_{process}"] = q_n2

        results.append(row_result)
        print(f"  {mof}: PSA α_IAST={row_result.get('alpha_IAST_PSA', 'N/A'):.4f}, "
              f"VSA α_IAST={row_result.get('alpha_IAST_VSA', 'N/A'):.4f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"Total: {len(result_df)} MOFs processed")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute IAST selectivity from pure-component Langmuir fits."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model results dir (default: results/alignn/model_ep150).",
    )
    args = parser.parse_args()

    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = REPO_ROOT / md
    else:
        md = REPO_ROOT / "results" / "alignn" / "model_ep150"

    bkt_dir = md / "bkt_candidates"
    fits_csv = bkt_dir / "isotherm_fits" / "best_isotherm_fits.csv"
    output_csv = bkt_dir / "iast_selectivity.csv"

    if not fits_csv.exists():
        print(f"ERROR: {fits_csv} not found")
        sys.exit(1)

    print("=" * 60)
    print("IAST Selectivity Computation")
    print("=" * 60)
    print(f"Input:  {fits_csv}")
    print(f"Output: {output_csv}")
    print(f"Feed:   CH4:N2 = {Y_CH4*100:.0f}:{Y_N2*100:.0f}")
    print(f"PSA:    {PROCESS_CONDITIONS['PSA']['total_pressure']} bar")
    print(f"VSA:    {PROCESS_CONDITIONS['VSA']['total_pressure']} bar")
    print()

    compute_iast_selectivity(fits_csv, output_csv)


if __name__ == "__main__":
    main()
