"""
compute_iast_selectivity.py — IAST selectivity from pure-component isotherm fits.

Reads isotherm parameters from best_isotherm_fits.csv (supports DSL, DSLF,
Langmuir, Langmuir-Freundlich, and extDSL models), then solves binary IAST via
spreading pressure equality using the generic solver from bkt.src.isotherms.

For extDSL: reads ext_dsl_fits.csv and converts Arrhenius parameters to 298 K
DSL equivalents before IAST solving.

Output: process_candidates/iast_selectivity.csv

Usage:
    conda run -n alignn_env python src/alignn/compute_iast_selectivity.py --model DSL
    conda run -n alignn_env python src/alignn/compute_iast_selectivity.py --model extDSL
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bkt.src.isotherms import _IAST_MODELS, _iast_binary

# Physical constants
R_GAS = 8.314       # J/(mol·K)
T_REF = 298.15      # K — reference temperature for extDSL → DSL conversion

# CBM feed composition
Y_CH4 = 0.2
Y_N2 = 0.8

# Process conditions
PROCESS_CONDITIONS = {
    "PSA": {"total_pressure": 10.0},  # bar
    "VSA": {"total_pressure": 1.0},   # bar
}


# ---------------------------------------------------------------------------
# Parameter extraction from CSV row
# ---------------------------------------------------------------------------

def _extdsl_b_at_T(b0, delta_u, T=T_REF):
    """Compute affinity constant at temperature T from Arrhenius parameters.

    b(T) = b0 * exp(-ΔU / (R * T))
    where ΔU is in J/mol (negative for exothermic adsorption).
    """
    return b0 * np.exp(-delta_u / (R_GAS * T))


def _row_to_params(row, model_type):
    """Convert a CSV row to an IAST parameter dict for the given model.

    For extDSL, converts Arrhenius form (b0, ΔU) to 298 K DSL equivalents,
    then returns standard DSL params for IAST solving.
    """
    if model_type == 'Langmuir':
        return {'qs': row['qs1'], 'b': row['b1']}
    elif model_type == 'Langmuir-Freundlich':
        return {'qs': row['qs1'], 'b': row['b1'], 'n': row.get('n1', 1.0)}
    elif model_type == 'DSL':
        return {'qs1': row['qs1'], 'b1': row['b1'],
                'qs2': row['qs2'], 'b2': row['b2']}
    elif model_type == 'DSLF':
        return {'qs1': row['qs1'], 'b1': row['b1'],
                'n1': row.get('n1', 1.0),
                'qs2': row['qs2'], 'b2': row['b2'],
                'n2': row.get('n2', 1.0)}
    elif model_type == 'extDSL':
        # Extended DSL: convert Arrhenius b0/ΔU to b(298 K)
        b1 = _extdsl_b_at_T(row['b0_b'], row['deltaU_b'])
        b2 = _extdsl_b_at_T(row['b0_d'], row['deltaU_d'])
        qs1 = row['qs_b']
        qs2 = row['qs_d']
        return {'qs1': qs1, 'b1': b1, 'qs2': qs2, 'b2': b2}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# IAST solver wrapper
# ---------------------------------------------------------------------------

def iast_binary(
    params_1: dict,
    params_2: dict,
    y: tuple,
    P_total: float,
    model_type: str = 'DSL',
) -> tuple:
    """Solve binary IAST for two components.

    Returns (alpha, q1, q2) where alpha = (q1/q2)*(y2/y1).
    """
    yi = np.array([y[0], y[1]])
    result = _iast_binary(yi, P_total, [params_1, params_2], model_type=model_type)

    if result is None:
        return np.nan, np.nan, np.nan

    q1, q2 = result[0], result[1]
    alpha = (q1 / q2) * (y[1] / y[0]) if q2 > 0 else np.nan
    return alpha, q1, q2


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _get_gas_column(df):
    """Detect the gas identifier column name (gas_key or GasName)."""
    if 'gas_key' in df.columns:
        return 'gas_key'
    elif 'GasName' in df.columns:
        return 'GasName'
    else:
        raise ValueError("CSV must contain 'gas_key' or 'GasName' column")


def compute_iast_selectivity(fits_csv: Path, output_csv: Path,
                             model_override: str = None):
    """Compute IAST selectivity for all MOFs from isotherm fit CSV.

    For extDSL model, Arrhenius parameters are converted to 298 K DSL
    equivalents and IAST is solved using the standard DSL functions.
    """
    df = pd.read_csv(fits_csv)
    gas_col = _get_gas_column(df)

    mof_names = df["MofName"].unique()
    print(f"Found {len(mof_names)} MOFs in {fits_csv.name}")

    results = []

    for mof in mof_names:
        mof_data = df[df["MofName"] == mof]

        ch4_row = mof_data[mof_data[gas_col].str.contains("methane")]
        n2_row = mof_data[mof_data[gas_col].str.contains("N2")]

        if ch4_row.empty or n2_row.empty:
            print(f"  WARNING: {mof} missing CH4 or N2 data, skipping")
            continue

        ch4_row = ch4_row.iloc[0]
        n2_row = n2_row.iloc[0]

        # Determine model: CLI override > CSV selected_model/model_used
        if model_override:
            model_type = model_override
        elif 'selected_model' in ch4_row.index:
            model_type = ch4_row['selected_model']
        elif 'model_used' in ch4_row.index:
            model_type = ch4_row['model_used']
        else:
            model_type = 'DSL'

        # Extract parameters based on model type
        params_ch4 = _row_to_params(ch4_row, model_type)
        params_n2 = _row_to_params(n2_row, model_type)

        # extDSL converts to DSL params, so IAST solver uses DSL model
        iast_model = 'DSL' if model_type == 'extDSL' else model_type

        row_result = {"MofName": mof, "model": model_type}

        for process, cond in PROCESS_CONDITIONS.items():
            total_p = cond["total_pressure"]
            alpha, q_ch4, q_n2 = iast_binary(
                params_ch4, params_n2, (Y_CH4, Y_N2), total_p,
                model_type=iast_model,
            )

            row_result[f"alpha_IAST_{process}"] = alpha
            row_result[f"q_CH4_IAST_{process}"] = q_ch4
            row_result[f"q_N2_IAST_{process}"] = q_n2

        results.append(row_result)

        psa_a = row_result.get('alpha_IAST_PSA', float('nan'))
        vsa_a = row_result.get('alpha_IAST_VSA', float('nan'))
        psa_str = f"{psa_a:.4f}" if not np.isnan(psa_a) else "N/A"
        vsa_str = f"{vsa_a:.4f}" if not np.isnan(vsa_a) else "N/A"
        print(f"  {mof} [{model_type}]: PSA α={psa_str}, VSA α={vsa_str}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"Total: {len(result_df)} MOFs processed")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute IAST selectivity from pure-component isotherm fits."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model results dir (default: results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["Langmuir", "Langmuir-Freundlich", "DSL", "DSLF", "extDSL"],
        help="Override isotherm model (default: use selected_model from CSV).",
    )
    parser.add_argument(
        "--fits-csv", type=str, default=None,
        help="Override path to isotherm fits CSV (auto-detected for extDSL).",
    )
    parser.add_argument(
        "--bkt-dir", type=str, default=None,
        help="Override process_candidates directory path.",
    )
    args = parser.parse_args()

    if args.bkt_dir:
        bkt_dir = Path(args.bkt_dir)
        if not bkt_dir.is_absolute():
            bkt_dir = REPO_ROOT / bkt_dir
    else:
        if args.model_dir:
            md = Path(args.model_dir)
            if not md.is_absolute():
                md = REPO_ROOT / md
        else:
            md = REPO_ROOT / "results" / "alignn" / "model_ep150"
        bkt_dir = md / "process_candidates"

    # Determine input CSV: explicit --fits-csv > auto-detect by model
    if args.fits_csv:
        fits_csv = Path(args.fits_csv)
        if not fits_csv.is_absolute():
            fits_csv = REPO_ROOT / fits_csv
    elif args.model == "extDSL":
        fits_csv = bkt_dir / "isotherm_fits" / "ext_dsl_fits.csv"
    else:
        fits_csv = bkt_dir / "isotherm_fits" / "best_isotherm_fits.csv"
    output_csv = bkt_dir / "iast_selectivity.csv"

    if not fits_csv.exists():
        print(f"ERROR: {fits_csv} not found")
        sys.exit(1)

    model_label = args.model or "auto (from CSV)"
    print("=" * 60)
    print(f"IAST Selectivity Computation ({model_label})")
    print("=" * 60)
    print(f"Input:  {fits_csv}")
    print(f"Output: {output_csv}")
    print(f"Feed:   CH4:N2 = {Y_CH4*100:.0f}:{Y_N2*100:.0f}")
    print(f"PSA:    {PROCESS_CONDITIONS['PSA']['total_pressure']} bar")
    print(f"VSA:    {PROCESS_CONDITIONS['VSA']['total_pressure']} bar")
    print()

    compute_iast_selectivity(fits_csv, output_csv, model_override=args.model)


if __name__ == "__main__":
    main()
