"""
fit_pure_component_isotherms.py — Fit pure-component isotherms using DSLF model.

For each MOF × gas, fits a Dual-Site Langmuir-Freundlich (DSLF) isotherm:
  q = qs1*b1*P^n1/(1+b1*P^n1) + qs2*b2*P^n2/(1+b2*P^n2)

No model selection needed — DSLF subsumes Langmuir/DSL/LF as special cases.

Uses L-BFGS-B optimizer with light L2 regularization on (n-1) to prevent
overfitting of exponents while preserving physical n values.

BKT mapping:
  DSLF → isomodel="DSLF"
    b1 → bi[i]   (site 1 affinity)
    qs1 → qsbi[i] (site 1 saturation)
    n1 → n1i[i]  (site 1 exponent)
    b2 → di[i]   (site 2 affinity)
    qs2 → qsdi[i] (site 2 saturation)
    n2 → n2i[i]  (site 2 exponent)

Usage:
    python src/alignn/fit_pure_component_isotherms.py \\
        --input-csv .../atc_cu_pure_component.csv \\
        --input-csv .../top20_pure_component.csv \\
        --output-dir .../isotherm_fits
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_CSVS = [
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "bkt_candidates" / "isotherm_input" / "atc_cu_pure_component.csv",
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "bkt_candidates" / "isotherm_input" / "top20_pure_component.csv",
]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "bkt_candidates" / "isotherm_fits"
)

STANDARD_COLUMNS = [
    "MofName", "GasName", "Temperature[K]", "Pressure[bar]",
    "AllComponents", "MoleculeFraction", "LoadingUnit",
    "AbsLoading", "ExcessLoading", "SimuDuration[h]", "FilePath", "Notes",
]


# ---------------------------------------------------------------------------
# DSLF model
# ---------------------------------------------------------------------------

def dslf(P, params):
    """Dual-Site Langmuir-Freundlich. P [bar], params=[qs1,b1,n1,qs2,b2,n2].
    Returns q [mol/kg]."""
    qs1, b1, n1, qs2, b2, n2 = params
    Pn1 = np.power(np.maximum(P, 1e-30), n1)
    Pn2 = np.power(np.maximum(P, 1e-30), n2)
    return qs1 * b1 * Pn1 / (1.0 + b1 * Pn1) + qs2 * b2 * Pn2 / (1.0 + b2 * Pn2)


# Wide physical bounds — no artificial n restriction
DSLF_BOUNDS = [
    (0.01, 200.0),   # qs1
    (1e-8, 1e6),     # b1
    (0.3, 3.0),      # n1
    (0.01, 200.0),   # qs2
    (1e-8, 1e6),     # b2
    (0.3, 3.0),      # n2
]

# L2 regularization strength on (n - 1).
# λ=0.0001 is optimal: eliminates overfitting-driven n drift to boundaries
# while preserving physically meaningful n values (max_n ≈ 1.4).
# Validated: IAST MAPE 1.21% (best across λ sweep), R² loss < 2e-6.
REGULARIZATION_LAMBDA = 1e-4

DSLF_P0_LIST = [
    [3.0, 1.0, 1.0, 2.0, 0.05, 1.0],
    [5.0, 0.5, 0.9, 3.0, 0.01, 0.9],
    [2.0, 2.0, 0.8, 5.0, 0.1, 1.2],
    [1.0, 5.0, 1.1, 8.0, 0.02, 0.8],
    [4.0, 0.1, 1.0, 1.0, 1.0, 1.0],
    [3.0, 3.0, 0.7, 2.0, 0.2, 1.3],
    [6.0, 0.3, 1.2, 4.0, 0.005, 0.7],
    [2.0, 0.5, 0.9, 1.0, 0.05, 1.1],
    [5.0, 1.0, 1.0, 3.0, 0.1, 1.0],
    [2.0, 0.5, 1.0, 8.0, 0.05, 1.0],
]

DSLF_PARAM_NAMES = ["qs1", "b1", "n1", "qs2", "b2", "n2"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# DSLF fitting with L-BFGS-B + regularization
# ---------------------------------------------------------------------------

def _fit_dslf(
    pressures: np.ndarray,
    loadings: np.ndarray,
    reg_lambda: float = REGULARIZATION_LAMBDA,
) -> Optional[Dict]:
    """Fit DSLF with L-BFGS-B optimizer and L2 regularization on n exponents.

    Objective: SSE_normalized + λ * [(n1-1)² + (n2-1)²]

    SSE is normalized by (N * Var(q)) to make λ scale-independent across
    different gases and MOFs with varying loading magnitudes.
    """
    n_data = len(pressures)
    q_var = np.var(loadings)
    if q_var == 0:
        q_var = 1.0

    def objective(params):
        pred = dslf(pressures, params)
        sse_norm = np.sum((loadings - pred) ** 2) / (n_data * q_var)
        n1, n2 = params[2], params[5]
        reg = reg_lambda * ((n1 - 1.0) ** 2 + (n2 - 1.0) ** 2)
        return sse_norm + reg

    best_result = None
    best_obj = np.inf

    for p0 in DSLF_P0_LIST:
        try:
            result = minimize(
                objective, p0, bounds=DSLF_BOUNDS, method="L-BFGS-B",
                options={"maxiter": 10000, "ftol": 1e-15},
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None

    popt = best_result.x
    q_pred = dslf(pressures, popt)
    r2 = _r_squared(loadings, q_pred)
    mae = _mae(loadings, q_pred)
    rmse = _rmse(loadings, q_pred)

    params = dict(zip(DSLF_PARAM_NAMES, popt.tolist()))

    return {
        "parameters": params,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "n_params": 6,
        "experimental_pressures": pressures.tolist(),
        "experimental_loadings": loadings.tolist(),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_and_merge(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")
        df = pd.read_csv(p)
        missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {p}: {missing}")
        frames.append(df[STANDARD_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def fit_all(merged: pd.DataFrame, reg_lambda: float = REGULARIZATION_LAMBDA) -> Tuple[pd.DataFrame, Dict]:
    """Fit DSLF to all MOFs. Returns (fit_df, summary_dict)."""

    rows = []
    summary = {}

    mof_names = sorted(merged["MofName"].unique())
    print(f"\nFitting {len(mof_names)} MOFs × DSLF (L-BFGS-B, λ={reg_lambda}) ...")

    for mof in mof_names:
        mof_df = merged[merged["MofName"] == mof]
        gases = sorted(mof_df["GasName"].unique())
        temps = mof_df["Temperature[K]"].unique()
        temp = float(temps[0])

        print(f"\n  {mof}  (gases={gases}, T={temp} K)")

        mof_r2s = []

        for gas in gases:
            sub = mof_df[mof_df["GasName"] == gas].sort_values("Pressure[bar]")
            pressures = sub["Pressure[bar]"].values.astype(float)
            loadings = sub["AbsLoading"].values.astype(float)

            result = _fit_dslf(pressures, loadings, reg_lambda=reg_lambda)
            if result is None:
                print(f"    {gas:>8s}  DSLF  FAILED")
                continue

            p = result["parameters"]
            print(f"    {gas:>8s}  DSLF  R²={result['R2']:.6f}  "
                  f"MAE={result['MAE']:.4f}  n1={p['n1']:.3f}  n2={p['n2']:.3f}")
            mof_r2s.append(result["R2"])

            gas_key = f"{gas}_{temp}K"
            params = result["parameters"]
            row = {
                "MofName": mof,
                "gas_key": gas_key,
                "GasName": gas,
                "Temperature[K]": temp,
                "selected_model": "DSLF",
                "bkt_isomodel": "DSLF",
                "R2": result["R2"],
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "n_points": len(result["experimental_pressures"]),
                "pressure_min_bar": min(result["experimental_pressures"]),
                "pressure_max_bar": max(result["experimental_pressures"]),
                "qs1": params["qs1"],
                "b1": params["b1"],
                "n1": params["n1"],
                "qs2": params["qs2"],
                "b2": params["b2"],
                "n2": params["n2"],
            }
            rows.append(row)

        if mof_r2s:
            summary[mof] = {
                "selected_model": "DSLF",
                "mean_r2": float(np.mean(mof_r2s)),
                "n_gases_fit": len(mof_r2s),
            }
            print(f"    → DSLF  mean R²={np.mean(mof_r2s):.6f}")

    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit pure-component isotherms with DSLF model."
    )
    parser.add_argument(
        "--input-csv", dest="input_csvs", action="append", default=None,
        help="Input pure-component CSV (repeat for multiple files).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for fit results.",
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=REGULARIZATION_LAMBDA,
        help=f"L2 regularization strength on (n-1) (default: {REGULARIZATION_LAMBDA}).",
    )
    args = parser.parse_args()

    reg_lambda = args.reg_lambda

    csv_paths = (
        [Path(p) for p in args.input_csvs]
        if args.input_csvs
        else DEFAULT_INPUT_CSVS
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load & merge
    merged = load_and_merge(csv_paths)
    merged_csv = output_dir / "pure_component_data_merged.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"Merged input: {merged_csv}  ({len(merged)} rows)")

    # Fit (pass reg_lambda to override default if specified)
    if reg_lambda != REGULARIZATION_LAMBDA:
        print(f"Using custom λ={reg_lambda}")
    best_df, sel_summary = fit_all(merged, reg_lambda=reg_lambda)

    # Save
    best_csv = output_dir / "best_isotherm_fits.csv"
    summary_json = output_dir / "model_selection_summary.json"

    best_df.to_csv(best_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(sel_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Best-fit CSV  : {best_csv}  ({len(best_df)} rows)")
    print(f"Summary JSON  : {summary_json}")
    print(f"{'='*60}")

    # Quick summary stats
    n_mofs = best_df["MofName"].nunique()
    mean_r2 = best_df["R2"].mean()
    min_r2 = best_df["R2"].min()
    max_n = max(best_df["n1"].max(), best_df["n2"].max())
    print(f"  MOFs fitted : {n_mofs}")
    print(f"  Model       : DSLF (L-BFGS-B, λ={reg_lambda})")
    print(f"  R² mean/min : {mean_r2:.6f} / {min_r2:.6f}")
    print(f"  max n       : {max_n:.3f}")


if __name__ == "__main__":
    main()
