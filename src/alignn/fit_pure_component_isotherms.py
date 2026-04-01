"""
fit_pure_component_isotherms.py — Fit pure-component isotherms (DSL or DSLF).

For each MOF × gas, fits either:
  DSL:  q = qs1*b1*P/(1+b1*P) + qs2*b2*P/(1+b2*P)
  DSLF: q = qs1*b1*P^n1/(1+b1*P^n1) + qs2*b2*P^n2/(1+b2*P^n2)

Uses reparameterized L-BFGS-B optimization with physical constraints:
  - Site ordering: b1 >= b2 (site 1 = strong, site 2 = weak)
  - No phantom sites: each site >= 5% of total capacity
  - Reparameterization: theta = [q_total, alpha, log_b1, delta_log_b, (n1, n2)]

BKT mapping (isomodel="DSL"):
  b1 → bi[i],  qs1 → qsbi[i]  (site 1)
  b2 → di[i],  qs2 → qsdi[i]  (site 2)

BKT mapping (isomodel="DSLF"):
  Same as DSL, plus n1 → n1i[i], n2 → n2i[i]

Usage:
    python src/alignn/fit_pure_component_isotherms.py
    python src/alignn/fit_pure_component_isotherms.py --model DSLF
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
    / "process_candidates" / "isotherm_input" / "atc_cu_pure_component.csv",
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "process_candidates" / "isotherm_input" / "top20_pure_component.csv",
]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "process_candidates" / "isotherm_fits"
)

STANDARD_COLUMNS = [
    "MofName", "GasName", "Temperature[K]", "Pressure[bar]",
    "AllComponents", "MoleculeFraction", "LoadingUnit",
    "AbsLoading", "ExcessLoading", "SimuDuration[h]", "FilePath", "Notes",
]


# ---------------------------------------------------------------------------
# DSL model
# ---------------------------------------------------------------------------

def dsl(P, params):
    """Dual-Site Langmuir. P [bar], params=[qs1, b1, qs2, b2].
    Returns q [mol/kg]."""
    qs1, b1, qs2, b2 = params
    return qs1 * b1 * P / (1.0 + b1 * P) + qs2 * b2 * P / (1.0 + b2 * P)


def dsl_sp(P, params):
    """DSL spreading pressure: qs1*ln(1+b1*P) + qs2*ln(1+b2*P)."""
    qs1, b1, qs2, b2 = params
    return qs1 * np.log(1.0 + b1 * P) + qs2 * np.log(1.0 + b2 * P)


# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------

def _theta_to_params(theta):
    """Convert reparameterized theta to physical DSL params [qs1, b1, qs2, b2]."""
    q_total, alpha, log_b1, delta = theta
    qs1 = q_total * alpha
    qs2 = q_total * (1.0 - alpha)
    b1 = np.exp(log_b1)
    b2 = np.exp(log_b1 - delta)  # delta >= 0 → b1 >= b2
    return np.array([qs1, b1, qs2, b2])


# Bounds in theta space
THETA_BOUNDS = [
    (0.1, 400.0),    # q_total [mol/kg]
    (0.05, 0.95),    # alpha — each site >= 5% capacity (no phantom sites)
    (-18.0, 14.0),   # log_b1 (b1 ∈ [~1e-8, ~1e6])
    (0.0, 15.0),     # delta_log_b >= 0 → b1 >= b2
]

# Multi-start initial guesses in theta space
THETA_P0_LIST = [
    [5.0, 0.4, 0.0, 3.0],
    [8.0, 0.3, -0.7, 2.0],
    [3.0, 0.6, 1.6, 4.0],
    [10.0, 0.2, 0.0, 2.0],
    [6.0, 0.5, -2.3, 1.0],
    [4.0, 0.3, 1.0, 5.0],
    [15.0, 0.15, -1.0, 3.0],
    [2.0, 0.7, 2.0, 2.0],
    [5.0, 0.5, 0.0, 1.0],
    [12.0, 0.1, -0.5, 4.0],
]

DSL_PARAM_NAMES = ["qs1", "b1", "qs2", "b2"]


# ---------------------------------------------------------------------------
# DSLF model
# ---------------------------------------------------------------------------

def dslf(P, params):
    """Dual-Site Langmuir-Freundlich. P [bar], params=[qs1,b1,n1,qs2,b2,n2].
    Returns q [mol/kg]."""
    qs1, b1, n1, qs2, b2, n2 = params
    P_safe = np.maximum(P, 0.0)
    Pn1 = np.power(P_safe, n1)
    Pn2 = np.power(P_safe, n2)
    return qs1 * b1 * Pn1 / (1.0 + b1 * Pn1) + qs2 * b2 * Pn2 / (1.0 + b2 * Pn2)


def dslf_sp(P, params):
    """DSLF spreading pressure: (qs1/n1)*ln(1+b1*P^n1) + (qs2/n2)*ln(1+b2*P^n2)."""
    qs1, b1, n1, qs2, b2, n2 = params
    P_safe = np.maximum(P, 0.0)
    Pn1 = np.power(P_safe, n1)
    Pn2 = np.power(P_safe, n2)
    return (qs1 / n1) * np.log(1.0 + b1 * Pn1) + (qs2 / n2) * np.log(1.0 + b2 * Pn2)


# DSLF reparameterization: theta = [q_total, alpha, log_b1, delta_log_b, n1, n2]
def _theta_to_dslf_params(theta):
    """Convert reparameterized theta to physical DSLF params [qs1,b1,n1,qs2,b2,n2]."""
    q_total, alpha, log_b1, delta, n1, n2 = theta
    qs1 = q_total * alpha
    qs2 = q_total * (1.0 - alpha)
    b1 = np.exp(log_b1)
    b2 = np.exp(log_b1 - delta)
    return np.array([qs1, b1, n1, qs2, b2, n2])


DSLF_THETA_BOUNDS = [
    (0.1, 400.0),    # q_total [mol/kg]
    (0.05, 0.95),    # alpha
    (-18.0, 14.0),   # log_b1
    (0.0, 15.0),     # delta_log_b
    (0.3, 3.0),      # n1
    (0.3, 3.0),      # n2
]

DSLF_THETA_P0_LIST = [
    [5.0, 0.4, 0.0, 3.0, 1.0, 1.0],
    [8.0, 0.3, -0.7, 2.0, 0.8, 0.8],
    [3.0, 0.6, 1.6, 4.0, 1.2, 1.2],
    [10.0, 0.2, 0.0, 2.0, 0.9, 0.9],
    [6.0, 0.5, -2.3, 1.0, 1.1, 1.1],
    [4.0, 0.3, 1.0, 5.0, 0.7, 0.7],
    [15.0, 0.15, -1.0, 3.0, 1.3, 1.3],
    [2.0, 0.7, 2.0, 2.0, 0.6, 0.6],
    [5.0, 0.5, 0.0, 1.0, 1.0, 0.8],
    [12.0, 0.1, -0.5, 4.0, 0.9, 1.1],
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
# DSL fitting
# ---------------------------------------------------------------------------

def _fit_dsl(
    pressures: np.ndarray,
    loadings: np.ndarray,
) -> Optional[Dict]:
    """Fit DSL with reparameterized L-BFGS-B."""
    n_data = len(pressures)
    q_var = np.var(loadings)
    if q_var == 0:
        q_var = 1.0

    def objective(theta):
        params = _theta_to_params(theta)
        pred = dsl(pressures, params)
        return np.sum((loadings - pred) ** 2) / (n_data * q_var)

    best_result = None
    best_obj = np.inf

    for p0 in THETA_P0_LIST:
        try:
            result = minimize(
                objective, p0, bounds=THETA_BOUNDS, method="L-BFGS-B",
                options={"maxiter": 10000, "ftol": 1e-15},
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None

    popt = _theta_to_params(best_result.x)
    q_pred = dsl(pressures, popt)
    r2 = _r_squared(loadings, q_pred)
    mae = _mae(loadings, q_pred)
    rmse = _rmse(loadings, q_pred)

    params = dict(zip(DSL_PARAM_NAMES, popt.tolist()))

    return {
        "parameters": params,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "n_params": 4,
        "experimental_pressures": pressures.tolist(),
        "experimental_loadings": loadings.tolist(),
    }


# ---------------------------------------------------------------------------
# DSLF fitting
# ---------------------------------------------------------------------------

def _fit_dslf(
    pressures: np.ndarray,
    loadings: np.ndarray,
) -> Optional[Dict]:
    """Fit DSLF with reparameterized L-BFGS-B."""
    n_data = len(pressures)
    q_var = np.var(loadings)
    if q_var == 0:
        q_var = 1.0

    def objective(theta):
        params = _theta_to_dslf_params(theta)
        pred = dslf(pressures, params)
        return np.sum((loadings - pred) ** 2) / (n_data * q_var)

    best_result = None
    best_obj = np.inf

    for p0 in DSLF_THETA_P0_LIST:
        try:
            result = minimize(
                objective, p0, bounds=DSLF_THETA_BOUNDS, method="L-BFGS-B",
                options={"maxiter": 10000, "ftol": 1e-15},
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None

    popt = _theta_to_dslf_params(best_result.x)
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


def fit_all(merged: pd.DataFrame, model: str = "DSL") -> Tuple[pd.DataFrame, Dict]:
    """Fit isotherms to all MOFs. Returns (fit_df, summary_dict).

    Parameters
    ----------
    model : str
        'DSL' or 'DSLF'.
    """
    fit_fn = _fit_dslf if model == "DSLF" else _fit_dsl

    rows = []
    summary = {}

    mof_names = sorted(merged["MofName"].unique())
    print(f"\nFitting {len(mof_names)} MOFs × {model} (reparam L-BFGS-B) ...")

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

            result = fit_fn(pressures, loadings)
            if result is None:
                print(f"    {gas:>8s}  {model}  FAILED")
                continue

            p = result["parameters"]
            b1_b2 = p["b1"] / p["b2"] if p["b2"] > 0 else float("inf")
            extra = ""
            if model == "DSLF":
                extra = f"  n1={p['n1']:.3f} n2={p['n2']:.3f}"
            print(f"    {gas:>8s}  {model}  R²={result['R2']:.6f}  "
                  f"qs1={p['qs1']:.2f} b1={p['b1']:.3f}  "
                  f"qs2={p['qs2']:.2f} b2={p['b2']:.3f}  "
                  f"b1/b2={b1_b2:.1f}{extra}")
            mof_r2s.append(result["R2"])

            gas_key = f"{gas}_{temp}K"
            row = {
                "MofName": mof,
                "gas_key": gas_key,
                "GasName": gas,
                "Temperature[K]": temp,
                "selected_model": model,
                "bkt_isomodel": model,
                "R2": result["R2"],
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "n_points": len(result["experimental_pressures"]),
                "pressure_min_bar": min(result["experimental_pressures"]),
                "pressure_max_bar": max(result["experimental_pressures"]),
                "qs1": p["qs1"],
                "b1": p["b1"],
                "qs2": p["qs2"],
                "b2": p["b2"],
                "n1": p.get("n1", 1.0),
                "n2": p.get("n2", 1.0),
            }
            rows.append(row)

        if mof_r2s:
            summary[mof] = {
                "selected_model": model,
                "mean_r2": float(np.mean(mof_r2s)),
                "n_gases_fit": len(mof_r2s),
            }
            print(f"    → {model}  mean R²={np.mean(mof_r2s):.6f}")

    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit pure-component isotherms with reparameterized DSL/DSLF model."
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
        "--model", type=str, default="DSLF", choices=["DSL", "DSLF"],
        help="Isotherm model to fit (default: DSLF).",
    )
    args = parser.parse_args()

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

    # Fit
    best_df, sel_summary = fit_all(merged, model=args.model)

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

    # Summary stats
    n_mofs = best_df["MofName"].nunique()
    mean_r2 = best_df["R2"].mean()
    min_r2 = best_df["R2"].min()
    max_b1 = best_df["b1"].max()
    min_qs = min(best_df["qs1"].min(), best_df["qs2"].min())
    print(f"  MOFs fitted : {n_mofs}")
    print(f"  Model       : {args.model} (all)")
    print(f"  R² mean/min : {mean_r2:.6f} / {min_r2:.6f}")
    print(f"  max b1      : {max_b1:.3f}")
    print(f"  min qs      : {min_qs:.3f}")


if __name__ == "__main__":
    main()
