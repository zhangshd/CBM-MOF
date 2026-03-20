"""
compute_iast_selectivity.py — IAST selectivity from pure-component isotherm fits.

Reads isotherm parameters from best_isotherm_fits.csv (supports DSL, DSLF,
Langmuir, and Langmuir-Freundlich models), then solves binary IAST via
spreading pressure equality using the generic solver from bkt.src.isotherms.

Output: bkt_candidates/iast_selectivity.csv

Usage:
    conda run -n mofmthnn python src/alignn/compute_iast_selectivity.py
    conda run -n mofmthnn python src/alignn/compute_iast_selectivity.py --model DSL
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

def _row_to_params(row, model_type):
    """Convert a CSV row to an IAST parameter dict for the given model."""
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

def compute_iast_selectivity(fits_csv: Path, output_csv: Path,
                             model_override: str = None):
    """Compute IAST selectivity for all MOFs in best_isotherm_fits.csv."""
    df = pd.read_csv(fits_csv)

    mof_names = df["MofName"].unique()
    print(f"Found {len(mof_names)} MOFs in {fits_csv.name}")

    results = []

    for mof in mof_names:
        mof_data = df[df["MofName"] == mof]

        ch4_row = mof_data[mof_data["gas_key"].str.contains("methane")]
        n2_row = mof_data[mof_data["gas_key"].str.contains("N2")]

        if ch4_row.empty or n2_row.empty:
            print(f"  WARNING: {mof} missing CH4 or N2 data, skipping")
            continue

        ch4_row = ch4_row.iloc[0]
        n2_row = n2_row.iloc[0]

        # Determine model: CLI override > CSV selected_model
        model_type = model_override or ch4_row.get("selected_model", "DSL")

        # Extract parameters based on model type
        params_ch4 = _row_to_params(ch4_row, model_type)
        params_n2 = _row_to_params(n2_row, model_type)

        row_result = {"MofName": mof, "model": model_type}

        for process, cond in PROCESS_CONDITIONS.items():
            total_p = cond["total_pressure"]
            alpha, q_ch4, q_n2 = iast_binary(
                params_ch4, params_n2, (Y_CH4, Y_N2), total_p,
                model_type=model_type,
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
        choices=["Langmuir", "Langmuir-Freundlich", "DSL", "DSLF"],
        help="Override isotherm model (default: use selected_model from CSV).",
    )
    parser.add_argument(
        "--bkt-dir", type=str, default=None,
        help="Override bkt_candidates directory path.",
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
        bkt_dir = md / "bkt_candidates"
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
