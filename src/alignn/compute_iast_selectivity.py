"""
compute_iast_selectivity.py — IAST selectivity from pure-component DSLF fits.

Reads DSLF parameters (qs1, b1, n1, qs2, b2, n2) from best_isotherm_fits.csv,
constructs loading and spreading pressure functions, then solves binary IAST
via spreading pressure equality (brentq) to compute mixed-component selectivity
at CBM conditions (CH4:N2 = 20:80).

No pyGAPS dependency — uses custom IAST solver.

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
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]

# CBM feed composition
Y_CH4 = 0.2
Y_N2 = 0.8

# Process conditions
PROCESS_CONDITIONS = {
    "PSA": {"total_pressure": 10.0},  # bar
    "VSA": {"total_pressure": 1.0},   # bar
}


# ---------------------------------------------------------------------------
# DSLF isotherm functions
# ---------------------------------------------------------------------------

def dslf_loading(P, qs1, b1, n1, qs2, b2, n2):
    """DSLF loading at pressure P [bar]. Returns q [mol/kg]."""
    Pn1 = np.power(np.maximum(P, 1e-30), n1)
    Pn2 = np.power(np.maximum(P, 1e-30), n2)
    return qs1 * b1 * Pn1 / (1.0 + b1 * Pn1) + qs2 * b2 * Pn2 / (1.0 + b2 * Pn2)


def dslf_spreading_pressure(P, qs1, b1, n1, qs2, b2, n2):
    """DSLF spreading pressure integral: (qs1/n1)*ln(1+b1*P^n1) + (qs2/n2)*ln(1+b2*P^n2)."""
    Pn1 = np.power(np.maximum(P, 1e-30), n1)
    Pn2 = np.power(np.maximum(P, 1e-30), n2)
    return (qs1 / n1) * np.log(1.0 + b1 * Pn1) + (qs2 / n2) * np.log(1.0 + b2 * Pn2)


# ---------------------------------------------------------------------------
# Custom IAST solver
# ---------------------------------------------------------------------------

def iast_binary(
    params_1: dict,
    params_2: dict,
    y: tuple,
    P_total: float,
) -> tuple:
    """
    Solve binary IAST for two components with DSLF isotherms.

    params_1, params_2: dicts with keys {qs1, b1, n1, qs2, b2, n2}
    y: (y1, y2) gas-phase mole fractions
    P_total: total pressure [bar]

    Returns (alpha, q1, q2) where alpha = (q1/q2)*(y2/y1).
    """
    y1, y2 = y

    def sp1(P):
        return dslf_spreading_pressure(P, **params_1)

    def sp2(P):
        return dslf_spreading_pressure(P, **params_2)

    def q1_fn(P):
        return dslf_loading(P, **params_1)

    def q2_fn(P):
        return dslf_loading(P, **params_2)

    def objective(x1):
        if x1 <= 0 or x1 >= 1:
            return 1e10
        P1_0 = P_total * y1 / x1
        P2_0 = P_total * y2 / (1.0 - x1)
        return sp1(P1_0) - sp2(P2_0)

    eps = 1e-10
    try:
        f_lo = objective(eps)
        f_hi = objective(1.0 - eps)
        if f_lo * f_hi > 0:
            # Sweep to find bracket
            xx = np.linspace(eps, 1.0 - eps, 200)
            ff = np.array([objective(x) for x in xx])
            sign_changes = np.where(np.diff(np.sign(ff)))[0]
            if len(sign_changes) == 0:
                return np.nan, np.nan, np.nan
            idx = sign_changes[0]
            x1_sol = brentq(objective, xx[idx], xx[idx + 1], xtol=1e-12)
        else:
            x1_sol = brentq(objective, eps, 1.0 - eps, xtol=1e-12)
    except Exception:
        return np.nan, np.nan, np.nan

    x2_sol = 1.0 - x1_sol
    P1_0 = P_total * y1 / x1_sol
    P2_0 = P_total * y2 / x2_sol

    q1_pure = q1_fn(P1_0)
    q2_pure = q2_fn(P2_0)

    if q1_pure <= 0 or q2_pure <= 0:
        return np.nan, np.nan, np.nan

    q_total = 1.0 / (x1_sol / q1_pure + x2_sol / q2_pure)
    q1 = x1_sol * q_total
    q2 = x2_sol * q_total

    alpha = (q1 / q2) * (y2 / y1) if q2 > 0 else np.nan

    return alpha, q1, q2


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_iast_selectivity(fits_csv: Path, output_csv: Path):
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

        # Extract DSLF parameters
        params_ch4 = {
            "qs1": ch4_row["qs1"], "b1": ch4_row["b1"], "n1": ch4_row["n1"],
            "qs2": ch4_row["qs2"], "b2": ch4_row["b2"], "n2": ch4_row["n2"],
        }
        params_n2 = {
            "qs1": n2_row["qs1"], "b1": n2_row["b1"], "n1": n2_row["n1"],
            "qs2": n2_row["qs2"], "b2": n2_row["b2"], "n2": n2_row["n2"],
        }

        row_result = {"MofName": mof}

        for process, cond in PROCESS_CONDITIONS.items():
            total_p = cond["total_pressure"]
            alpha, q_ch4, q_n2 = iast_binary(
                params_ch4, params_n2, (Y_CH4, Y_N2), total_p,
            )

            row_result[f"alpha_IAST_{process}"] = alpha
            row_result[f"q_CH4_IAST_{process}"] = q_ch4
            row_result[f"q_N2_IAST_{process}"] = q_n2

        results.append(row_result)

        psa_a = row_result.get('alpha_IAST_PSA', float('nan'))
        vsa_a = row_result.get('alpha_IAST_VSA', float('nan'))
        psa_str = f"{psa_a:.4f}" if not np.isnan(psa_a) else "N/A"
        vsa_str = f"{vsa_a:.4f}" if not np.isnan(vsa_a) else "N/A"
        print(f"  {mof}: PSA α_IAST={psa_str}, VSA α_IAST={vsa_str}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"Total: {len(result_df)} MOFs processed")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute IAST selectivity from pure-component DSLF fits."
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
    print("IAST Selectivity Computation (DSLF)")
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
