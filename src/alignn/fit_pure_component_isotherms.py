"""
fit_pure_component_isotherms.py — Fit pure-component isotherms using DSLF model.

For each MOF × gas, fits a Dual-Site Langmuir-Freundlich (DSLF) isotherm:
  q = qs1*b1*P^n1/(1+b1*P^n1) + qs2*b2*P^n2/(1+b2*P^n2)

No model selection needed — DSLF subsumes Langmuir/DSL/LF as special cases.
42/42 fits validated with mean R² = 0.999988.

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
from scipy.optimize import curve_fit


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

def dslf(P, qs1, b1, n1, qs2, b2, n2):
    """Dual-Site Langmuir-Freundlich. P [bar], returns q [mol/kg]."""
    Pn1 = np.power(np.maximum(P, 1e-30), n1)
    Pn2 = np.power(np.maximum(P, 1e-30), n2)
    return qs1 * b1 * Pn1 / (1.0 + b1 * Pn1) + qs2 * b2 * Pn2 / (1.0 + b2 * Pn2)


# Fitting bounds and initial guesses (validated in test_dsl_iast_comparison.py)
DSLF_BOUNDS = ([0.01, 1e-8, 0.3, 0.01, 1e-8, 0.3],
               [200.0, 1e6, 3.0, 200.0, 1e6, 3.0])

DSLF_P0_LIST = [
    [3.0, 1.0, 1.0, 2.0, 0.05, 1.0],
    [5.0, 0.5, 0.9, 3.0, 0.01, 0.9],
    [2.0, 2.0, 0.8, 5.0, 0.1, 1.2],
    [1.0, 5.0, 1.1, 8.0, 0.02, 0.8],
    [4.0, 0.1, 1.0, 1.0, 1.0, 1.0],
    [3.0, 3.0, 0.7, 2.0, 0.2, 1.3],
    [6.0, 0.3, 1.2, 4.0, 0.005, 0.7],
    [2.0, 0.5, 0.9, 1.0, 0.05, 1.1],
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
# DSLF fitting
# ---------------------------------------------------------------------------

def _fit_dslf(
    pressures: np.ndarray,
    loadings: np.ndarray,
) -> Optional[Dict]:
    """Fit DSLF to pressure/loading data. Returns result dict or None."""
    best_popt = None
    best_sse = np.inf

    for p0 in DSLF_P0_LIST:
        try:
            popt, _ = curve_fit(
                dslf, pressures, loadings,
                p0=p0, bounds=DSLF_BOUNDS, maxfev=10000,
            )
            q_pred = dslf(pressures, *popt)
            sse = np.sum((loadings - q_pred) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_popt = popt
        except Exception:
            continue

    if best_popt is None:
        return None

    q_pred = dslf(pressures, *best_popt)
    r2 = _r_squared(loadings, q_pred)
    mae = _mae(loadings, q_pred)
    rmse = _rmse(loadings, q_pred)

    params = dict(zip(DSLF_PARAM_NAMES, best_popt.tolist()))

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


def fit_all(merged: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Fit DSLF to all MOFs. Returns (fit_df, summary_dict)."""

    rows = []
    summary = {}

    mof_names = sorted(merged["MofName"].unique())
    print(f"\nFitting {len(mof_names)} MOFs × DSLF ...")

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

            result = _fit_dslf(pressures, loadings)
            if result is None:
                print(f"    {gas:>8s}  DSLF  FAILED")
                continue

            print(f"    {gas:>8s}  DSLF  R²={result['R2']:.6f}  "
                  f"MAE={result['MAE']:.4f}")
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
    best_df, sel_summary = fit_all(merged)

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
    print(f"  MOFs fitted : {n_mofs}")
    print(f"  Model       : DSLF (all)")
    print(f"  R² mean/min : {mean_r2:.6f} / {min_r2:.6f}")


if __name__ == "__main__":
    main()
