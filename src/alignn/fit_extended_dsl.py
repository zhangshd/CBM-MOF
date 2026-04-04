"""
fit_extended_dsl.py — Fit Extended DSL isotherms with temperature dependence.

Primary method: **Global fit** — all 3 temperatures fitted simultaneously with
  Arrhenius temperature dependence built into the model (6 params per MOF/gas).
Fallback initialization: Two-step method (independent DSL → Arrhenius extraction)
  provides starting points for the global optimizer.

Extended DSL model:
  q(P, T) = qs_b * b(T)*P / (1 + b(T)*P) + qs_d * d(T)*P / (1 + d(T)*P)
  b(T) = b0 * exp(-deltaU / R / T)    [deltaU < 0 for exothermic adsorption]

SuperPSA convention: "CO2" columns = CH4, "N2" columns = N2.
  b0/d0 units in CSV are labeled "kPa^-1" but are actually Pa^-1.

Usage:
    python src/alignn/fit_extended_dsl.py
    python src/alignn/fit_extended_dsl.py --skip-plots
    python src/alignn/fit_extended_dsl.py --input-298 path/to/298K.csv --input-multitemp path/to/multitemp.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Import DSL fitter from sibling module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pure_component_isotherms import _fit_dsl, dsl, _r_squared, _mae, _rmse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_GAS = 8.314          # J/(mol*K)
BAR_TO_PA = 1e5        # 1 bar = 1e5 Pa
T_REF = 298.0          # Reference temperature [K]
TEMPERATURES = [273.0, 298.0, 323.0]

# ---------------------------------------------------------------------------
# Extended DSL model (global fit)
# ---------------------------------------------------------------------------

def ext_dsl(T: np.ndarray, P: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Extended DSL with Arrhenius temperature dependence.

    Args:
        T: Temperature array [K].
        P: Pressure array [bar].
        params: [qs_b, qs_d, b0_b, b0_d, deltaU_b, deltaU_d].
            b0 in bar^-1, deltaU in J/mol (negative for exothermic).

    Returns:
        Loading q [mol/kg].
    """
    qs_b, qs_d, b0_b, b0_d, deltaU_b, deltaU_d = params
    B = b0_b * np.exp(-deltaU_b / R_GAS / T)
    D = b0_d * np.exp(-deltaU_d / R_GAS / T)
    return qs_b * B * P / (1.0 + B * P) + qs_d * D * P / (1.0 + D * P)


def _theta_to_ext_dsl_params(theta: np.ndarray) -> np.ndarray:
    """Convert reparameterized theta to physical Extended DSL params.

    theta = [q_total, alpha, ln_b0_b, delta_ln_b0, deltaU_b, deltaU_d]
    Returns [qs_b, qs_d, b0_b, b0_d, deltaU_b, deltaU_d].
    """
    q_total, alpha, ln_b0_b, delta_ln_b0, deltaU_b, deltaU_d = theta
    qs_b = q_total * alpha
    qs_d = q_total * (1.0 - alpha)
    b0_b = np.exp(ln_b0_b)
    b0_d = np.exp(ln_b0_b - delta_ln_b0)   # delta >= 0 → b0_b >= b0_d
    return np.array([qs_b, qs_d, b0_b, b0_d, deltaU_b, deltaU_d])


# Bounds in theta space for global Extended DSL fit
EXTDSL_THETA_BOUNDS = [
    (0.1, 400.0),       # q_total [mol/kg]
    (0.05, 0.95),       # alpha (each site >= 5%)
    (-40.0, 5.0),       # ln_b0_b (pre-exponential, very wide)
    (0.0, 30.0),        # delta_ln_b0 >= 0 (b0_b >= b0_d)
    (-100000.0, -500.0),  # deltaU_b [J/mol] (must be negative, exothermic)
    (-100000.0, -500.0),  # deltaU_d [J/mol] (must be negative)
]

# Standard multi-start initial guesses in theta space
# Columns: [q_total, alpha, ln_b0_b, delta_ln_b0, deltaU_b, deltaU_d]
EXTDSL_THETA_P0_LIST = [
    [5.0,  0.4,  -4.0,  3.0, -20000.0, -15000.0],
    [8.0,  0.3,  -6.0,  2.0, -25000.0, -18000.0],
    [3.0,  0.6,  -2.0,  4.0, -15000.0, -12000.0],
    [10.0, 0.2,  -8.0,  2.0, -30000.0, -20000.0],
    [6.0,  0.5,  -5.0,  1.0, -18000.0, -16000.0],
    [4.0,  0.3,  -3.0,  5.0, -22000.0, -14000.0],
    [15.0, 0.15, -7.0,  3.0, -28000.0, -22000.0],
    [2.0,  0.7,  -1.0,  2.0, -12000.0, -10000.0],
    [5.0,  0.5,  -4.5,  1.5, -20000.0, -20000.0],
    [12.0, 0.1,  -9.0,  4.0, -35000.0, -25000.0],
]


def _fit_ext_dsl_global(
    T_all: np.ndarray,
    P_all: np.ndarray,
    q_all: np.ndarray,
    twostep_params: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Global fit of Extended DSL to multi-temperature data.

    Args:
        T_all: Temperature array [K], shape (N,).
        P_all: Pressure array [bar], shape (N,).
        q_all: Loading array [mol/kg], shape (N,).
        twostep_params: Optional [qs_b, qs_d, b0_b, b0_d, deltaU_b, deltaU_d]
            from the two-step method, used as an additional starting point.

    Returns:
        Dict with keys: parameters, R2_global, R2_per_temp, or None on failure.
    """
    n_data = len(T_all)
    q_var = np.var(q_all)
    if q_var == 0:
        q_var = 1.0

    def objective(theta):
        params = _theta_to_ext_dsl_params(theta)
        q_pred = ext_dsl(T_all, P_all, params)
        return np.sum((q_all - q_pred) ** 2) / (n_data * q_var)

    best_result = None
    best_obj = np.inf

    # Standard multi-start
    for p0 in EXTDSL_THETA_P0_LIST:
        try:
            result = minimize(
                objective, p0, bounds=EXTDSL_THETA_BOUNDS, method="L-BFGS-B",
                options={"maxiter": 20000, "ftol": 1e-15},
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue

    # Two-step initialization (if available and reasonable)
    if twostep_params is not None:
        qs_b, qs_d, b0_b, b0_d, deltaU_b, deltaU_d = twostep_params
        # Convert physical params to theta space
        if qs_b > 0 and qs_d > 0 and b0_b > 0 and b0_d > 0:
            q_total = qs_b + qs_d
            alpha = qs_b / q_total
            ln_b0_b = np.log(b0_b)
            # Ensure b0_b >= b0_d for the reparameterization
            if b0_b >= b0_d:
                delta_ln_b0 = ln_b0_b - np.log(b0_d)
            else:
                # Swap sites so b0_b >= b0_d
                ln_b0_b = np.log(b0_d)
                delta_ln_b0 = np.log(b0_d) - np.log(b0_b)
                alpha = 1.0 - alpha
                deltaU_b, deltaU_d = deltaU_d, deltaU_b

            # Clip to bounds
            theta_ts = np.array([q_total, alpha, ln_b0_b, delta_ln_b0,
                                 deltaU_b, deltaU_d])
            for i, (lo, hi) in enumerate(EXTDSL_THETA_BOUNDS):
                theta_ts[i] = np.clip(theta_ts[i], lo, hi)

            try:
                result = minimize(
                    objective, theta_ts, bounds=EXTDSL_THETA_BOUNDS, method="L-BFGS-B",
                    options={"maxiter": 20000, "ftol": 1e-15},
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
            except Exception:
                pass

    if best_result is None:
        return None

    popt = _theta_to_ext_dsl_params(best_result.x)
    q_pred = ext_dsl(T_all, P_all, popt)
    r2_global = _r_squared(q_all, q_pred)

    # Per-temperature R2
    r2_per_temp = {}
    for temp in TEMPERATURES:
        mask = np.isclose(T_all, temp)
        if np.sum(mask) > 0:
            q_pred_t = ext_dsl(T_all[mask], P_all[mask], popt)
            r2_per_temp[f"R2_{int(temp)}"] = _r_squared(q_all[mask], q_pred_t)

    return {
        "parameters": {
            "qs_b": float(popt[0]),
            "qs_d": float(popt[1]),
            "b0_b": float(popt[2]),   # bar^-1
            "b0_d": float(popt[3]),   # bar^-1
            "deltaU_b": float(popt[4]),  # J/mol
            "deltaU_d": float(popt[5]),  # J/mol
        },
        "R2_global": r2_global,
        "R2_per_temp": r2_per_temp,
        "objective": float(best_obj),
    }


def fit_global(
    merged: pd.DataFrame,
    twostep_fits: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run global Extended DSL fit for all (MOF, gas) pairs.

    Args:
        merged: Multi-temperature GCMC data (all 3 temps).
        twostep_fits: Optional two-step Arrhenius results (for initialization).

    Returns:
        DataFrame with global fit parameters and R2 values.
    """
    rows = []
    mofs = sorted(merged["MofName"].unique())
    gases = sorted(merged["GasName"].unique())

    n_total = len(mofs) * len(gases)
    n_success = 0

    logger.info("=" * 60)
    logger.info("Global Extended DSL fit (%d MOF x %d gas = %d fits)",
                len(mofs), len(gases), n_total)
    logger.info("=" * 60)

    for mof in mofs:
        for gas in gases:
            sub = merged[
                (merged["MofName"] == mof) & (merged["GasName"] == gas)
            ].sort_values(["Temperature[K]", "Pressure[bar]"])

            T_all = sub["Temperature[K]"].values.astype(float)
            P_all = sub["Pressure[bar]"].values.astype(float)
            q_all = sub["AbsLoading"].values.astype(float)

            if len(T_all) < 15:
                logger.warning("  Insufficient data for %s / %s: %d pts (need >= 15)",
                               mof, gas, len(T_all))
                continue

            # Get two-step params if available
            twostep_params = None
            if twostep_fits is not None:
                ts_row = twostep_fits[
                    (twostep_fits["MofName"] == mof) & (twostep_fits["GasName"] == gas)
                ]
                if len(ts_row) > 0:
                    ts = ts_row.iloc[0]
                    # Use two-step only if Arrhenius R2 > 0.80 and deltaU < 0
                    if (ts["R2_arrhenius_b"] > 0.80 and ts["R2_arrhenius_d"] > 0.80
                            and ts["deltaU_b"] < 0 and ts["deltaU_d"] < 0):
                        twostep_params = np.array([
                            ts["qs_b"], ts["qs_d"],
                            ts["b0_b"], ts["b0_d"],
                            ts["deltaU_b"], ts["deltaU_d"],
                        ])

            result = _fit_ext_dsl_global(T_all, P_all, q_all, twostep_params)
            if result is None:
                logger.warning("  FAILED: %s / %s (global fit)", mof, gas)
                continue

            p = result["parameters"]
            row = {
                "MofName": mof,
                "GasName": gas,
                "qs_b": p["qs_b"],
                "qs_d": p["qs_d"],
                "b0_b": p["b0_b"],
                "b0_d": p["b0_d"],
                "deltaU_b": p["deltaU_b"],
                "deltaU_d": p["deltaU_d"],
                "R2_global": result["R2_global"],
                **result["R2_per_temp"],
            }
            rows.append(row)
            n_success += 1

            init_tag = "TS-init" if twostep_params is not None else "std-init"
            logger.info("  %s / %s [%s]: R2_global=%.6f  "
                         "qs_b=%.3f qs_d=%.3f  b0_b=%.4e b0_d=%.4e  "
                         "dU_b=%.0f dU_d=%.0f",
                         mof, gas, init_tag, result["R2_global"],
                         p["qs_b"], p["qs_d"], p["b0_b"], p["b0_d"],
                         p["deltaU_b"], p["deltaU_d"])

    logger.info("Global fit complete: %d/%d success", n_success, n_total)
    return pd.DataFrame(rows)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_298 = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "process_candidates" / "isotherm_input" / "top20_pure_component.csv"
)
DEFAULT_INPUT_MULTITEMP = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "process_candidates" / "isotherm_input" / "top20_pure_component_multitemp.csv"
)
DEFAULT_CURRENT_DSL = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "process_candidates" / "isotherm_fits" / "best_isotherm_fits.csv"
)
DEFAULT_TEMPLATE_PSA = REPO_ROOT / "src" / "SuperPSA" / "data" / "Adsorbents_CH4N2_PSA.csv"
DEFAULT_TEMPLATE_VSA = REPO_ROOT / "src" / "SuperPSA" / "data" / "Adsorbents_CH4N2_VSA.csv"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "process_candidates" / "isotherm_fits"
)
DEFAULT_SUPERPSA_DIR = REPO_ROOT / "src" / "SuperPSA" / "data"

# New independent data sources (remove circular dependency on SuperPSA templates)
_MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
DEFAULT_TOP10_PSA = _MODEL_DIR / "process_candidates" / "top10_psa.csv"
DEFAULT_TOP10_VSA = _MODEL_DIR / "process_candidates" / "top10_vsa.csv"
DEFAULT_CIF_DIR = _MODEL_DIR / "process_candidates" / "cifs"
DEFAULT_CIF_DIR_FALLBACK = _MODEL_DIR / "top_candidates" / "cifs_all_top"
DEFAULT_QST_CSV = _MODEL_DIR / "process_candidates" / "gcmc_vs_ml_comparison.csv"
DEFAULT_BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"

SUPERPSA_COLUMNS = [
    "material_name",
    "q_s_b_CO2 [mol/kg]", "q_s_d_CO2 [mol/kg]",
    "b0_CO2 [kPa^-1]", "d0_CO2 [kPa^-1]",
    "deltaU_b_CO2 [J/mol]", "deltaU_d_CO2 [J/mol]",
    "q_s_b_N2 [mol/kg]", "q_s_d_N2 [mol/kg]",
    "b0_N2 [kPa^-1]", "d0_N2 [kPa^-1]",
    "deltaU_b_N2 [J/mol]", "deltaU_d_N2 [J/mol]",
    "isotherm_type", "ro_s [kg/m^3]",
    "deltaU_CO2 [J/mol]", "deltaU_N2 [J/mol]",
]


# ---------------------------------------------------------------------------
# Step 0: Load & merge multi-temperature GCMC data
# ---------------------------------------------------------------------------

def load_and_merge_temperatures(
    csv_298: Path,
    csv_multitemp: Path,
) -> pd.DataFrame:
    """Load 298K and multi-temp (273K, 323K) CSVs and merge.

    Returns DataFrame with standard columns, validated for completeness.
    """
    logger.info("Loading 298K data from %s", csv_298)
    df298 = pd.read_csv(csv_298)
    logger.info("  %d rows, %d MOFs", len(df298), df298["MofName"].nunique())

    logger.info("Loading multi-temp data from %s", csv_multitemp)
    df_mt = pd.read_csv(csv_multitemp)
    logger.info("  %d rows, %d MOFs, temps=%s",
                len(df_mt), df_mt["MofName"].nunique(),
                sorted(df_mt["Temperature[K]"].unique()))

    merged = pd.concat([df298, df_mt], ignore_index=True)
    logger.info("Merged: %d rows", len(merged))

    # Validate completeness: each MOF × gas should have all 3 temperatures
    for mof in sorted(merged["MofName"].unique()):
        for gas in sorted(merged["GasName"].unique()):
            sub = merged[(merged["MofName"] == mof) & (merged["GasName"] == gas)]
            temps = sorted(sub["Temperature[K]"].unique())
            if len(temps) != 3:
                logger.warning("Incomplete data for %s/%s: found temps=%s (expected 3)",
                               mof, gas, temps)

    return merged


# ---------------------------------------------------------------------------
# Step 1: Independent DSL fit at each temperature
# ---------------------------------------------------------------------------

def fit_dsl_per_temperature(merged: pd.DataFrame) -> pd.DataFrame:
    """Fit standard DSL at each (MOF, gas, T) triple.

    Returns DataFrame with columns:
        MofName, GasName, Temperature, qs1, b1, qs2, b2, R2, MAE, RMSE
    """
    rows = []
    groups = merged.groupby(["MofName", "GasName", "Temperature[K]"])

    n_total = len(groups)
    n_success = 0
    n_fail = 0

    logger.info("Step 1: Fitting DSL at each temperature (%d groups)", n_total)

    for (mof, gas, temp), sub in sorted(groups):
        sub = sub.sort_values("Pressure[bar]")
        pressures = sub["Pressure[bar]"].values.astype(float)
        loadings = sub["AbsLoading"].values.astype(float)

        result = _fit_dsl(pressures, loadings)
        if result is None:
            logger.warning("  FAILED: %s / %s / %.0fK", mof, gas, temp)
            n_fail += 1
            continue

        p = result["parameters"]
        rows.append({
            "MofName": mof,
            "GasName": gas,
            "Temperature": temp,
            "qs1": p["qs1"],
            "b1": p["b1"],
            "qs2": p["qs2"],
            "b2": p["b2"],
            "R2": result["R2"],
            "MAE": result["MAE"],
            "RMSE": result["RMSE"],
        })
        n_success += 1

        logger.debug("  %s / %s / %.0fK: R2=%.6f  qs1=%.3f b1=%.4f  qs2=%.3f b2=%.4f",
                      mof, gas, temp, result["R2"],
                      p["qs1"], p["b1"], p["qs2"], p["b2"])

    logger.info("Step 1 complete: %d/%d success, %d failed", n_success, n_total, n_fail)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1.5: Site consistency check across temperatures
# ---------------------------------------------------------------------------

def check_and_fix_site_consistency(per_temp_fits: pd.DataFrame) -> pd.DataFrame:
    """Detect and fix site swaps across temperatures.

    The reparameterization enforces b1 >= b2 at each temperature, but sites
    might swap identity across temperatures. Use 298K as reference and
    compare qs values to detect swaps.

    Returns corrected DataFrame.
    """
    df = per_temp_fits.copy()
    n_swaps = 0

    for mof in sorted(df["MofName"].unique()):
        for gas in sorted(df["GasName"].unique()):
            mask = (df["MofName"] == mof) & (df["GasName"] == gas)
            sub = df.loc[mask].copy()

            if len(sub) < 2:
                continue

            # Use 298K as reference
            ref_row = sub[sub["Temperature"] == T_REF]
            if len(ref_row) == 0:
                logger.warning("No 298K reference for %s/%s, skipping consistency check", mof, gas)
                continue
            ref_qs1 = ref_row["qs1"].values[0]
            ref_qs2 = ref_row["qs2"].values[0]

            for idx, row in sub.iterrows():
                if row["Temperature"] == T_REF:
                    continue

                qs1_val = row["qs1"]
                qs2_val = row["qs2"]

                # Check if this temperature's site 1 is closer to reference site 2
                # (indicating a swap)
                dist_same = abs(qs1_val - ref_qs1) + abs(qs2_val - ref_qs2)
                dist_swap = abs(qs1_val - ref_qs2) + abs(qs2_val - ref_qs1)

                if dist_swap < dist_same:
                    logger.info("Site swap detected: %s / %s / %.0fK "
                                "(qs1=%.3f↔qs2=%.3f, ref qs1=%.3f qs2=%.3f)",
                                mof, gas, row["Temperature"],
                                qs1_val, qs2_val, ref_qs1, ref_qs2)
                    # Swap sites
                    df.loc[idx, "qs1"] = qs2_val
                    df.loc[idx, "qs2"] = qs1_val
                    df.loc[idx, "b1"] = row["b2"]
                    df.loc[idx, "b2"] = row["b1"]
                    n_swaps += 1

    logger.info("Site consistency: %d swaps corrected", n_swaps)
    return df


# ---------------------------------------------------------------------------
# Step 2: Arrhenius extraction
# ---------------------------------------------------------------------------

def fit_arrhenius(per_temp_fits: pd.DataFrame) -> pd.DataFrame:
    """For each (MOF, gas), fit Arrhenius parameters from DSL fits at 3 temperatures.

    ln(b(T)) = ln(b0) - deltaU / (R * T)
      slope = -deltaU / R  =>  deltaU = -slope * R
      intercept = ln(b0)   =>  b0 = exp(intercept)

    Returns DataFrame with Extended DSL parameters per MOF/gas.
    """
    rows = []

    for mof in sorted(per_temp_fits["MofName"].unique()):
        for gas in sorted(per_temp_fits["GasName"].unique()):
            sub = per_temp_fits[
                (per_temp_fits["MofName"] == mof) & (per_temp_fits["GasName"] == gas)
            ].sort_values("Temperature")

            if len(sub) < 3:
                logger.warning("Insufficient temperatures for %s/%s: %d (need 3)",
                               mof, gas, len(sub))
                continue

            T_arr = sub["Temperature"].values
            inv_T = 1.0 / T_arr

            # --- Site b (site 1 = strong affinity) ---
            b1_arr = sub["b1"].values
            qs1_arr = sub["qs1"].values

            ln_b1 = np.log(b1_arr)
            slope_b, intercept_b = np.polyfit(inv_T, ln_b1, 1)
            deltaU_b = -slope_b * R_GAS    # J/mol
            b0_b = np.exp(intercept_b)     # bar^-1

            # Arrhenius R2 for site b
            ln_b1_pred = slope_b * inv_T + intercept_b
            r2_arr_b = _r_squared(ln_b1, ln_b1_pred)

            # qs stability
            qs_b_mean = float(np.mean(qs1_arr))
            qs_b_cv = float(np.std(qs1_arr) / qs_b_mean) if qs_b_mean > 0 else 0.0

            # --- Site d (site 2 = weak affinity) ---
            b2_arr = sub["b2"].values
            qs2_arr = sub["qs2"].values

            ln_b2 = np.log(b2_arr)
            slope_d, intercept_d = np.polyfit(inv_T, ln_b2, 1)
            deltaU_d = -slope_d * R_GAS
            b0_d = np.exp(intercept_d)

            ln_b2_pred = slope_d * inv_T + intercept_d
            r2_arr_d = _r_squared(ln_b2, ln_b2_pred)

            qs_d_mean = float(np.mean(qs2_arr))
            qs_d_cv = float(np.std(qs2_arr) / qs_d_mean) if qs_d_mean > 0 else 0.0

            # Collect per-temperature R2 values
            r2_by_temp = {}
            for _, row in sub.iterrows():
                r2_by_temp[f"R2_{int(row['Temperature'])}"] = row["R2"]

            rows.append({
                "MofName": mof,
                "GasName": gas,
                "qs_b": qs_b_mean,
                "qs_d": qs_d_mean,
                "b0_b": b0_b,          # bar^-1
                "b0_d": b0_d,          # bar^-1
                "deltaU_b": deltaU_b,  # J/mol (should be < 0)
                "deltaU_d": deltaU_d,  # J/mol (should be < 0)
                "R2_arrhenius_b": r2_arr_b,
                "R2_arrhenius_d": r2_arr_d,
                "qs_cv_b": qs_b_cv,
                "qs_cv_d": qs_d_cv,
                **r2_by_temp,
            })

            logger.info("  %s / %s: b0_b=%.4e deltaU_b=%.0f  b0_d=%.4e deltaU_d=%.0f  "
                         "R2_arr_b=%.4f R2_arr_d=%.4f  qs_cv_b=%.3f qs_cv_d=%.3f",
                         mof, gas, b0_b, deltaU_b, b0_d, deltaU_d,
                         r2_arr_b, r2_arr_d, qs_b_cv, qs_d_cv)

    logger.info("Step 2 complete: %d Extended DSL parameter sets", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_parameters(
    ext_fits: pd.DataFrame,
    current_dsl: Optional[pd.DataFrame],
    twostep_fits: Optional[pd.DataFrame] = None,
) -> Dict:
    """Validate Extended DSL parameters against physical constraints.

    Handles both global fit results (with R2_global) and two-step results
    (with R2_arrhenius_b/d, qs_cv_b/d as diagnostics only).

    Returns summary dict with validation results.
    """
    logger.info("=" * 60)
    logger.info("Validation")
    logger.info("=" * 60)

    issues = []
    has_global = "R2_global" in ext_fits.columns

    # 1. All deltaU < 0 (exothermic)
    for site in ["b", "d"]:
        col = f"deltaU_{site}"
        positive = ext_fits[ext_fits[col] >= 0]
        if len(positive) > 0:
            for _, row in positive.iterrows():
                msg = (f"deltaU_{site} >= 0 for {row['MofName']}/{row['GasName']}: "
                       f"{row[col]:.1f} J/mol (expected < 0)")
                logger.warning("  WARNING: %s", msg)
                issues.append(msg)
        else:
            logger.info("  PASS: all deltaU_%s < 0", site)

    # 2. All b0 > 0 (guaranteed by reparameterization, but verify)
    for site in ["b", "d"]:
        col = f"b0_{site}"
        neg = ext_fits[ext_fits[col] <= 0]
        if len(neg) > 0:
            for _, row in neg.iterrows():
                msg = f"b0_{site} <= 0 for {row['MofName']}/{row['GasName']}: {row[col]:.4e}"
                logger.warning("  WARNING: %s", msg)
                issues.append(msg)
        else:
            logger.info("  PASS: all b0_%s > 0", site)

    # 3. Global R2 > 0.999
    if has_global:
        low_r2 = ext_fits[ext_fits["R2_global"] < 0.999]
        if len(low_r2) > 0:
            for _, row in low_r2.iterrows():
                msg = (f"R2_global < 0.999 for {row['MofName']}/{row['GasName']}: "
                       f"{row['R2_global']:.6f}")
                logger.warning("  WARNING: %s", msg)
                issues.append(msg)
        else:
            logger.info("  PASS: all R2_global > 0.999")

    # 4. Two-step diagnostics (if available)
    if twostep_fits is not None:
        n_ts_issues = 0
        for _, ts_row in twostep_fits.iterrows():
            for site in ["b", "d"]:
                arr_col = f"R2_arrhenius_{site}"
                cv_col = f"qs_cv_{site}"
                if ts_row[arr_col] < 0.90 or ts_row[cv_col] > 0.10:
                    n_ts_issues += 1
        if n_ts_issues > 0:
            logger.info("  INFO: %d two-step diagnostic issues "
                         "(superseded by global fit)", n_ts_issues)
        else:
            logger.info("  PASS: two-step diagnostics all clean")

    # 5. 298K reproduction: compare b_298_extDSL vs b_298_DSL
    deviations = []
    if current_dsl is not None:
        logger.info("  Comparing 298K reproduction (ExtDSL vs current DSL)...")
        for _, erow in ext_fits.iterrows():
            mof = erow["MofName"]
            gas = erow["GasName"]

            dsl_row = current_dsl[
                (current_dsl["MofName"] == mof) & (current_dsl["GasName"] == gas)
            ]
            if len(dsl_row) == 0:
                continue
            dsl_row = dsl_row.iloc[0]

            # Reconstruct b at 298K from Extended DSL
            b_298_ext_b = erow["b0_b"] * np.exp(-erow["deltaU_b"] / R_GAS / T_REF)
            b_298_ext_d = erow["b0_d"] * np.exp(-erow["deltaU_d"] / R_GAS / T_REF)

            # Current DSL b values at 298K
            b_298_dsl_b = dsl_row["b1"]
            b_298_dsl_d = dsl_row["b2"]

            for site_label, b_ext, b_dsl in [
                ("b", b_298_ext_b, b_298_dsl_b),
                ("d", b_298_ext_d, b_298_dsl_d),
            ]:
                if b_dsl > 0:
                    pct_dev = abs(b_ext - b_dsl) / b_dsl * 100
                    deviations.append(pct_dev)
                    if pct_dev > 5.0:
                        msg = (f"298K b_{site_label} deviation > 5% for {mof}/{gas}: "
                               f"ExtDSL={b_ext:.6f} vs DSL={b_dsl:.6f} ({pct_dev:.1f}%)")
                        logger.warning("  WARNING: %s", msg)
                        issues.append(msg)

        if deviations:
            logger.info("  298K deviation: mean=%.2f%%, max=%.2f%%, "
                         "within 5%%: %d/%d",
                         np.mean(deviations), np.max(deviations),
                         sum(1 for d in deviations if d <= 5.0), len(deviations))

    # deltaU range check (Hu 2023: -90,000 to -5,800 J/mol)
    all_deltaU = pd.concat([ext_fits["deltaU_b"], ext_fits["deltaU_d"]])
    logger.info("  deltaU range: [%.0f, %.0f] J/mol (Hu 2023 ref: [-90000, -5800])",
                all_deltaU.min(), all_deltaU.max())

    summary = {
        "n_issues": len(issues),
        "issues": issues,
        "deltaU_range": [float(all_deltaU.min()), float(all_deltaU.max())],
    }
    if has_global:
        summary["R2_global_range"] = [
            float(ext_fits["R2_global"].min()),
            float(ext_fits["R2_global"].max()),
        ]
    if current_dsl is not None and deviations:
        summary["b298_deviation_mean_pct"] = float(np.mean(deviations))
        summary["b298_deviation_max_pct"] = float(np.max(deviations))

    return summary


# ---------------------------------------------------------------------------
# Material metadata helpers (independent data sources)
# ---------------------------------------------------------------------------

def compute_framework_density(
    cif_path: Path,
) -> float:
    """Compute framework density [kg/m^3] from CIF file using pymatgen.

    Args:
        cif_path: Path to the CIF file.

    Returns:
        Framework density in kg/m^3.
    """
    from pymatgen.core import Structure  # lazy import — only needed for this function
    s = Structure.from_file(str(cif_path))
    return s.density * 1000.0  # g/cm^3 -> kg/m^3


def qst_to_deltaU(qst_kJ: float) -> float:
    """Convert isosteric heat of adsorption [kJ/mol] to LDF deltaU [J/mol].

    deltaU = R*T_ref - Qst*1000
    """
    return R_GAS * T_REF - qst_kJ * 1000.0


def _find_cif(mof_id: str, cif_dirs: List[Path]) -> Optional[Path]:
    """Search for a CIF file across multiple directories.

    Args:
        mof_id: MOF identifier (filename stem).
        cif_dirs: Ordered list of directories to search.

    Returns:
        Path to the CIF file, or None if not found.
    """
    for d in cif_dirs:
        # Use glob to handle bracket characters in filenames
        candidates = list(d.glob(f"{mof_id}.cif"))
        if not candidates:
            # Fallback: iterate directory listing (handles special chars)
            for f in d.iterdir():
                if f.name == f"{mof_id}.cif":
                    return f
        else:
            return candidates[0]
    return None


def load_material_metadata(
    top10_psa_csv: Path,
    top10_vsa_csv: Path,
    cif_dirs: List[Path],
    qst_csv: Path,
    benchmark_mof: str,
) -> Dict[str, Dict]:
    """Load material lists and compute metadata from independent sources.

    Returns dict: process_type -> {material_name -> {ro_s, deltaU_CO2, deltaU_N2}}.
    """
    metadata = {}
    for suffix, csv_path in [("PSA", top10_psa_csv), ("VSA", top10_vsa_csv)]:
        if not csv_path.exists():
            logger.warning("Top-10 CSV not found: %s, skipping %s", csv_path, suffix)
            continue

        top10 = pd.read_csv(csv_path)
        mof_ids = list(top10["mof_id"])
        if benchmark_mof and benchmark_mof not in mof_ids:
            mof_ids.append(benchmark_mof)
            logger.info("Added benchmark MOF %s to %s material list", benchmark_mof, suffix)

        logger.info("Loading metadata for %s: %d materials", suffix, len(mof_ids))

        # Load Qst data
        qst_df = pd.read_csv(qst_csv)
        qst_map_ch4 = qst_df.set_index("mof_id")["QstCH4_gcmc"].to_dict()
        qst_map_n2 = qst_df.set_index("mof_id")["QstN2_gcmc"].to_dict()

        mat_meta = {}
        for mof in mof_ids:
            # Framework density from CIF
            cif_path = _find_cif(mof, cif_dirs)
            if cif_path is None:
                logger.warning("CIF not found for %s in %s, skipping", mof,
                               [str(d) for d in cif_dirs])
                continue

            try:
                ro_s = compute_framework_density(cif_path)
            except Exception as e:
                logger.warning("Failed to compute density for %s: %s", mof, e)
                continue

            # LDF deltaU from Qst
            if mof not in qst_map_ch4 or mof not in qst_map_n2:
                logger.warning("Missing Qst data for %s, skipping", mof)
                continue

            deltaU_ch4 = qst_to_deltaU(qst_map_ch4[mof])
            deltaU_n2 = qst_to_deltaU(qst_map_n2[mof])

            mat_meta[mof] = {
                "ro_s": ro_s,
                "deltaU_CO2": deltaU_ch4,
                "deltaU_N2": deltaU_n2,
            }
            logger.debug("  %s: ro_s=%.2f, dU_CH4=%.1f, dU_N2=%.1f",
                         mof, ro_s, deltaU_ch4, deltaU_n2)

        metadata[suffix] = mat_meta
        logger.info("  Loaded metadata for %d/%d materials", len(mat_meta), len(mof_ids))

    return metadata


# ---------------------------------------------------------------------------
# SuperPSA CSV generation
# ---------------------------------------------------------------------------

def build_superpsa_csv(
    ext_fits: pd.DataFrame,
    template_csv: Optional[Path] = None,
    material_metadata: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """Build SuperPSA adsorbent CSV from Extended DSL parameters.

    Two modes:
      1. Independent mode (preferred): uses material_metadata dict for material list,
         ro_s, and deltaU (LDF). No dependency on SuperPSA template CSVs.
      2. Template mode (deprecated): reads a SuperPSA template CSV for metadata.

    At least one of template_csv or material_metadata must be provided.

    Unit conversions:
      b0 [bar^-1] -> b0 [Pa^-1] = b0_bar / BAR_TO_PA
    """
    if material_metadata is not None:
        material_list = list(material_metadata.keys())
        logger.info("Building SuperPSA CSV from independent metadata (%d materials)",
                     len(material_list))
    elif template_csv is not None:
        template = pd.read_csv(template_csv)
        material_list = list(template["material_name"])
        logger.info("Building SuperPSA CSV from template %s (%d materials) [deprecated]",
                     template_csv.name, len(material_list))
        # Convert template to metadata dict for unified code path
        material_metadata = {}
        for _, tmpl_row in template.iterrows():
            mat = tmpl_row["material_name"]
            material_metadata[mat] = {
                "ro_s": tmpl_row["ro_s [kg/m^3]"],
                "deltaU_CO2": tmpl_row["deltaU_CO2 [J/mol]"],
                "deltaU_N2": tmpl_row["deltaU_N2 [J/mol]"],
            }
    else:
        raise ValueError("Either template_csv or material_metadata must be provided")

    out_rows = []
    for mat in material_list:
        meta = material_metadata[mat]

        # Get CH4 (= "CO2" in SuperPSA convention) parameters
        ch4 = ext_fits[
            (ext_fits["MofName"] == mat) & (ext_fits["GasName"] == "methane")
        ]
        n2 = ext_fits[
            (ext_fits["MofName"] == mat) & (ext_fits["GasName"] == "N2")
        ]

        if len(ch4) == 0 or len(n2) == 0:
            logger.warning("Missing ExtDSL data for %s, skipping", mat)
            continue

        ch4 = ch4.iloc[0]
        n2 = n2.iloc[0]

        row = {
            "material_name": mat,
            # CH4 -> "CO2" columns
            "q_s_b_CO2 [mol/kg]": ch4["qs_b"],
            "q_s_d_CO2 [mol/kg]": ch4["qs_d"],
            "b0_CO2 [kPa^-1]": ch4["b0_b"] / BAR_TO_PA,   # bar^-1 -> Pa^-1
            "d0_CO2 [kPa^-1]": ch4["b0_d"] / BAR_TO_PA,
            "deltaU_b_CO2 [J/mol]": ch4["deltaU_b"],
            "deltaU_d_CO2 [J/mol]": ch4["deltaU_d"],
            # N2 columns
            "q_s_b_N2 [mol/kg]": n2["qs_b"],
            "q_s_d_N2 [mol/kg]": n2["qs_d"],
            "b0_N2 [kPa^-1]": n2["b0_b"] / BAR_TO_PA,
            "d0_N2 [kPa^-1]": n2["b0_d"] / BAR_TO_PA,
            "deltaU_b_N2 [J/mol]": n2["deltaU_b"],
            "deltaU_d_N2 [J/mol]": n2["deltaU_d"],
            # Material properties
            "isotherm_type": 0,  # partial pressure basis [Pa]
            "ro_s [kg/m^3]": meta["ro_s"],
            "deltaU_CO2 [J/mol]": meta["deltaU_CO2"],
            "deltaU_N2 [J/mol]": meta["deltaU_N2"],
        }
        out_rows.append(row)

    df = pd.DataFrame(out_rows, columns=SUPERPSA_COLUMNS)
    logger.info("  Generated %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _try_import_style():
    """Try to import publication style; fall back to basic matplotlib."""
    try:
        style_dir = Path(__file__).resolve().parents[1] / "figures"
        sys.path.insert(0, str(style_dir))
        import style as pub_style
        pub_style.apply_style()
        logger.info("Using publication style from src/figures/style.py")
        return pub_style
    except Exception:
        logger.info("Publication style not available, using default matplotlib")
        return None


def generate_diagnostics(
    per_temp_fits: pd.DataFrame,
    ext_fits: pd.DataFrame,
    merged_data: pd.DataFrame,
    output_dir: Path,
    global_fits: Optional[pd.DataFrame] = None,
) -> None:
    """Generate diagnostic plots for Extended DSL fitting.

    Plot 1: Per-MOF isotherm fit quality (GCMC points + DSL curves + global fit)
    Plot 2: Arrhenius plots (1/T vs ln(b), both sites)
    Plot 3: 298K comparison (Extended DSL vs independent DSL)
    Plot 4: qs stability summary (bar chart of CV)
    """
    diag_dir = output_dir / "ext_dsl_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating diagnostic plots in %s", diag_dir)

    _try_import_style()

    temp_colors = {273.0: "#2166ac", 298.0: "#4dac26", 323.0: "#d6604d"}
    temp_markers = {273.0: "o", 298.0: "s", 323.0: "^"}

    mofs = sorted(ext_fits["MofName"].unique())
    gases = sorted(ext_fits["GasName"].unique())

    # --- Plot 1: Isotherm fit quality per MOF ---
    for gas in gases:
        n_mofs = len(mofs)
        ncols = 4
        nrows = int(np.ceil(n_mofs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                                 squeeze=False)
        fig.suptitle(f"DSL fits — {gas}", fontsize=12, y=1.02)

        for i, mof in enumerate(mofs):
            ax = axes[i // ncols, i % ncols]
            short_name = mof[:30] + "..." if len(mof) > 30 else mof
            ax.set_title(short_name, fontsize=7)

            for temp in TEMPERATURES:
                gcmc = merged_data[
                    (merged_data["MofName"] == mof) &
                    (merged_data["GasName"] == gas) &
                    (merged_data["Temperature[K]"] == temp)
                ].sort_values("Pressure[bar]")

                if len(gcmc) == 0:
                    continue

                P_data = gcmc["Pressure[bar]"].values
                q_data = gcmc["AbsLoading"].values

                ax.scatter(P_data, q_data,
                           color=temp_colors[temp], marker=temp_markers[temp],
                           s=20, label=f"{int(temp)}K GCMC", zorder=3)

                # Per-temperature DSL curve (thin, dashed)
                fit_row = per_temp_fits[
                    (per_temp_fits["MofName"] == mof) &
                    (per_temp_fits["GasName"] == gas) &
                    (per_temp_fits["Temperature"] == temp)
                ]
                if len(fit_row) > 0:
                    fr = fit_row.iloc[0]
                    P_smooth = np.linspace(0, P_data.max() * 1.05, 200)
                    q_fit = dsl(P_smooth, [fr["qs1"], fr["b1"], fr["qs2"], fr["b2"]])
                    ax.plot(P_smooth, q_fit, color=temp_colors[temp],
                            linewidth=0.8, alpha=0.5, linestyle="--")

                # Global fit curve (solid, thicker)
                if global_fits is not None:
                    gf_row = global_fits[
                        (global_fits["MofName"] == mof) &
                        (global_fits["GasName"] == gas)
                    ]
                    if len(gf_row) > 0:
                        gf = gf_row.iloc[0]
                        P_smooth = np.linspace(0, P_data.max() * 1.05, 200)
                        T_smooth = np.full_like(P_smooth, temp)
                        gf_params = np.array([
                            gf["qs_b"], gf["qs_d"], gf["b0_b"], gf["b0_d"],
                            gf["deltaU_b"], gf["deltaU_d"],
                        ])
                        q_gf = ext_dsl(T_smooth, P_smooth, gf_params)
                        ax.plot(P_smooth, q_gf, color=temp_colors[temp],
                                linewidth=1.2, alpha=0.9)

            ax.set_xlabel("P [bar]", fontsize=8)
            ax.set_ylabel("q [mol/kg]", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=6, loc="lower right")

        # Hide unused axes
        for j in range(n_mofs, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.tight_layout()
        out_path = diag_dir / f"isotherm_fits_{gas}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", out_path.name)

    # --- Plot 2: Arrhenius plots ---
    for gas in gases:
        n_mofs = len(mofs)
        ncols = 4
        nrows = int(np.ceil(n_mofs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows),
                                 squeeze=False)
        fig.suptitle(f"Arrhenius plots — {gas}", fontsize=12, y=1.02)

        for i, mof in enumerate(mofs):
            ax = axes[i // ncols, i % ncols]
            short_name = mof[:30] + "..." if len(mof) > 30 else mof
            ax.set_title(short_name, fontsize=7)

            sub = per_temp_fits[
                (per_temp_fits["MofName"] == mof) & (per_temp_fits["GasName"] == gas)
            ].sort_values("Temperature")

            if len(sub) < 2:
                continue

            T_arr = sub["Temperature"].values
            inv_T = 1.0 / T_arr

            ext_row = ext_fits[
                (ext_fits["MofName"] == mof) & (ext_fits["GasName"] == gas)
            ]
            if len(ext_row) == 0:
                continue
            er = ext_row.iloc[0]

            # Site b
            ln_b1 = np.log(sub["b1"].values)
            ax.scatter(inv_T * 1000, ln_b1, color="#d62728", marker="o", s=30,
                       label=f"site b (R2={er['R2_arrhenius_b']:.3f})", zorder=3)
            inv_T_line = np.linspace(inv_T.min() * 0.98, inv_T.max() * 1.02, 50)
            ln_b0_b = np.log(er["b0_b"])
            ax.plot(inv_T_line * 1000,
                    ln_b0_b + (-er["deltaU_b"] / R_GAS) * inv_T_line,
                    color="#d62728", linewidth=1, linestyle="--")

            # Site d
            ln_b2 = np.log(sub["b2"].values)
            ax.scatter(inv_T * 1000, ln_b2, color="#1f77b4", marker="s", s=30,
                       label=f"site d (R2={er['R2_arrhenius_d']:.3f})", zorder=3)
            ln_b0_d = np.log(er["b0_d"])
            ax.plot(inv_T_line * 1000,
                    ln_b0_d + (-er["deltaU_d"] / R_GAS) * inv_T_line,
                    color="#1f77b4", linewidth=1, linestyle="--")

            ax.set_xlabel("1000/T [K$^{-1}$]", fontsize=8)
            ax.set_ylabel("ln(b) [ln(bar$^{-1}$)]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="best")

        for j in range(n_mofs, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.tight_layout()
        out_path = diag_dir / f"arrhenius_{gas}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", out_path.name)

    # --- Plot 3: 298K b comparison (Extended DSL vs independent DSL) ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax_idx, site in enumerate(["b", "d"]):
        ax = axes[ax_idx]
        b_ext_list = []
        b_ind_list = []
        labels = []

        for _, row in ext_fits.iterrows():
            mof = row["MofName"]
            gas = row["GasName"]

            ind_row = per_temp_fits[
                (per_temp_fits["MofName"] == mof) &
                (per_temp_fits["GasName"] == gas) &
                (per_temp_fits["Temperature"] == T_REF)
            ]
            if len(ind_row) == 0:
                continue

            b_col = "b1" if site == "b" else "b2"
            b_ind = ind_row.iloc[0][b_col]
            b0_col = f"b0_{site}"
            du_col = f"deltaU_{site}"
            b_ext = row[b0_col] * np.exp(-row[du_col] / R_GAS / T_REF)

            b_ext_list.append(b_ext)
            b_ind_list.append(b_ind)
            labels.append(f"{mof[:15]}_{gas[:3]}")

        b_ext_arr = np.array(b_ext_list)
        b_ind_arr = np.array(b_ind_list)

        ax.scatter(b_ind_arr, b_ext_arr, s=20, alpha=0.7)

        # Parity line
        lims = [min(b_ind_arr.min(), b_ext_arr.min()) * 0.8,
                max(b_ind_arr.max(), b_ext_arr.max()) * 1.2]
        if lims[0] > 0:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        r2_parity = _r_squared(b_ind_arr, b_ext_arr)
        ax.set_xlabel(f"b_{site} independent DSL [bar$^{{-1}}$]", fontsize=9)
        ax.set_ylabel(f"b_{site} Extended DSL [bar$^{{-1}}$]", fontsize=9)
        ax.set_title(f"Site {site}: 298K parity (R2={r2_parity:.4f})", fontsize=10)

    fig.tight_layout()
    out_path = diag_dir / "b298_parity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path.name)

    # --- Plot 4: qs CV summary ---
    fig, ax = plt.subplots(figsize=(10, 4))

    x_labels = []
    cv_b_vals = []
    cv_d_vals = []
    for _, row in ext_fits.iterrows():
        x_labels.append(f"{row['MofName'][:18]}\n{row['GasName']}")
        cv_b_vals.append(row["qs_cv_b"])
        cv_d_vals.append(row["qs_cv_d"])

    x = np.arange(len(x_labels))
    width = 0.35
    ax.bar(x - width / 2, cv_b_vals, width, label="site b", color="#d62728", alpha=0.7)
    ax.bar(x + width / 2, cv_d_vals, width, label="site d", color="#1f77b4", alpha=0.7)
    ax.axhline(y=0.10, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="CV = 0.10 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
    ax.set_ylabel("CV(qs) across 3 temperatures")
    ax.set_title("Saturation capacity stability")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = diag_dir / "qs_cv_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Extended DSL isotherms with temperature dependence "
        "(global fit + two-step diagnostics)."
    )
    parser.add_argument(
        "--input-298", type=str, default=str(DEFAULT_INPUT_298),
        help="Pure-component GCMC data at 298K (CSV).",
    )
    parser.add_argument(
        "--input-multitemp", type=str, default=str(DEFAULT_INPUT_MULTITEMP),
        help="Pure-component GCMC data at 273K and 323K (CSV).",
    )
    parser.add_argument(
        "--current-dsl", type=str, default=str(DEFAULT_CURRENT_DSL),
        help="Current DSL fits at 298K for comparison.",
    )
    parser.add_argument(
        "--template-csv", type=str, default=None,
        help="(Deprecated) SuperPSA adsorbent CSV template. "
        "Use --top10-psa/--top10-vsa + --cif-dir + --qst-csv instead.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for fit results.",
    )
    parser.add_argument(
        "--superpsa-dir", type=str, default=str(DEFAULT_SUPERPSA_DIR),
        help="SuperPSA data directory for adsorbent CSVs.",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip generating diagnostic plots.",
    )
    # New independent data source args
    parser.add_argument(
        "--top10-psa", type=str, default=str(DEFAULT_TOP10_PSA),
        help="Top-10 PSA candidates CSV (mof_id column).",
    )
    parser.add_argument(
        "--top10-vsa", type=str, default=str(DEFAULT_TOP10_VSA),
        help="Top-10 VSA candidates CSV (mof_id column).",
    )
    parser.add_argument(
        "--cif-dir", type=str, default=str(DEFAULT_CIF_DIR),
        help="Primary CIF directory for framework density computation.",
    )
    parser.add_argument(
        "--qst-csv", type=str, default=str(DEFAULT_QST_CSV),
        help="GCMC comparison CSV with QstCH4_gcmc, QstN2_gcmc columns.",
    )
    parser.add_argument(
        "--benchmark-mof", type=str, default=DEFAULT_BENCHMARK_MOF,
        help="Benchmark MOF ID to always include in both PSA and VSA.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    input_298 = Path(args.input_298)
    input_multitemp = Path(args.input_multitemp)
    current_dsl_path = Path(args.current_dsl)
    output_dir = Path(args.output_dir)
    superpsa_dir = Path(args.superpsa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Load and merge multi-temperature data
    # ------------------------------------------------------------------
    merged = load_and_merge_temperatures(input_298, input_multitemp)

    # ------------------------------------------------------------------
    # Step 1: Independent DSL fit at each temperature (diagnostics + init)
    # ------------------------------------------------------------------
    per_temp_fits = fit_dsl_per_temperature(merged)

    # Step 1.5: Site consistency check
    per_temp_fits = check_and_fix_site_consistency(per_temp_fits)

    per_temp_csv = output_dir / "ext_dsl_per_temp_fits.csv"
    per_temp_fits.to_csv(per_temp_csv, index=False)
    logger.info("Per-temperature fits: %s (%d rows)", per_temp_csv, len(per_temp_fits))

    # ------------------------------------------------------------------
    # Step 2: Arrhenius extraction (two-step, for diagnostics + init)
    # ------------------------------------------------------------------
    logger.info("Step 2: Arrhenius parameter extraction (two-step, for diagnostics)")
    twostep_fits = fit_arrhenius(per_temp_fits)

    twostep_csv = output_dir / "ext_dsl_twostep_fits.csv"
    twostep_fits.to_csv(twostep_csv, index=False)
    logger.info("Two-step fits: %s (%d rows)", twostep_csv, len(twostep_fits))

    # ------------------------------------------------------------------
    # Step 3: GLOBAL FIT — primary method
    # ------------------------------------------------------------------
    global_fits = fit_global(merged, twostep_fits)

    # Merge two-step diagnostics into global_fits for the output CSV
    ext_fits = global_fits.copy()
    for _, ts_row in twostep_fits.iterrows():
        mof, gas = ts_row["MofName"], ts_row["GasName"]
        mask = (ext_fits["MofName"] == mof) & (ext_fits["GasName"] == gas)
        if mask.any():
            ext_fits.loc[mask, "R2_arrhenius_b"] = ts_row["R2_arrhenius_b"]
            ext_fits.loc[mask, "R2_arrhenius_d"] = ts_row["R2_arrhenius_d"]
            ext_fits.loc[mask, "qs_cv_b"] = ts_row["qs_cv_b"]
            ext_fits.loc[mask, "qs_cv_d"] = ts_row["qs_cv_d"]

    # Fill missing diagnostic columns (if two-step failed for some)
    for col in ["R2_arrhenius_b", "R2_arrhenius_d", "qs_cv_b", "qs_cv_d"]:
        if col not in ext_fits.columns:
            ext_fits[col] = np.nan

    ext_csv = output_dir / "ext_dsl_fits.csv"
    ext_fits.to_csv(ext_csv, index=False)
    logger.info("Extended DSL fits (global): %s (%d rows)", ext_csv, len(ext_fits))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    current_dsl = None
    if current_dsl_path.exists():
        current_dsl = pd.read_csv(current_dsl_path)
        logger.info("Loaded current DSL fits from %s", current_dsl_path)

    val_summary = validate_parameters(ext_fits, current_dsl, twostep_fits)

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    summary = {
        "n_mofs": int(ext_fits["MofName"].nunique()),
        "n_gases": int(ext_fits["GasName"].nunique()),
        "n_parameter_sets": len(ext_fits),
        "fit_method": "global",
        "temperatures_K": TEMPERATURES,
        "per_temp_fits_csv": str(per_temp_csv),
        "twostep_fits_csv": str(twostep_csv),
        "ext_dsl_fits_csv": str(ext_csv),
        "R2_global_range": [
            float(ext_fits["R2_global"].min()),
            float(ext_fits["R2_global"].max()),
        ],
        "R2_global_mean": float(ext_fits["R2_global"].mean()),
        "deltaU_b_range_J_mol": [
            float(ext_fits["deltaU_b"].min()), float(ext_fits["deltaU_b"].max()),
        ],
        "deltaU_d_range_J_mol": [
            float(ext_fits["deltaU_d"].min()), float(ext_fits["deltaU_d"].max()),
        ],
        "twostep_diagnostics": {
            "R2_arrhenius_b_range": [
                float(twostep_fits["R2_arrhenius_b"].min()),
                float(twostep_fits["R2_arrhenius_b"].max()),
            ],
            "R2_arrhenius_d_range": [
                float(twostep_fits["R2_arrhenius_d"].min()),
                float(twostep_fits["R2_arrhenius_d"].max()),
            ],
            "qs_cv_b_range": [
                float(twostep_fits["qs_cv_b"].min()),
                float(twostep_fits["qs_cv_b"].max()),
            ],
            "qs_cv_d_range": [
                float(twostep_fits["qs_cv_d"].min()),
                float(twostep_fits["qs_cv_d"].max()),
            ],
        },
        "R2_isotherm_mean": float(
            per_temp_fits["R2"].mean() if len(per_temp_fits) > 0 else 0.0
        ),
        "R2_isotherm_min": float(
            per_temp_fits["R2"].min() if len(per_temp_fits) > 0 else 0.0
        ),
        "validation": val_summary,
    }

    summary_json = output_dir / "ext_dsl_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary JSON: %s", summary_json)

    # ------------------------------------------------------------------
    # SuperPSA CSV generation (uses global fit results)
    # ------------------------------------------------------------------
    use_template = args.template_csv is not None
    if use_template:
        logger.info("Using deprecated --template-csv mode for SuperPSA CSV generation")
        psa_template = Path(args.template_csv)
        vsa_template = superpsa_dir / "Adsorbents_CH4N2_VSA.csv"

        for template_path, suffix in [(psa_template, "PSA"), (vsa_template, "VSA")]:
            if not template_path.exists():
                logger.warning("Template not found: %s, skipping %s",
                               template_path, suffix)
                continue
            spsa_df = build_superpsa_csv(ext_fits, template_csv=template_path)
            out_path = superpsa_dir / f"Adsorbents_CH4N2_{suffix}_extDSL.csv"
            spsa_df.to_csv(out_path, index=False)
            logger.info("SuperPSA %s CSV: %s (%d rows)", suffix, out_path, len(spsa_df))
    else:
        # New independent data source mode
        cif_dir_primary = Path(args.cif_dir)
        cif_dirs = [cif_dir_primary]
        if DEFAULT_CIF_DIR_FALLBACK.exists():
            cif_dirs.append(DEFAULT_CIF_DIR_FALLBACK)
        logger.info("CIF search dirs: %s", [str(d) for d in cif_dirs])

        metadata = load_material_metadata(
            top10_psa_csv=Path(args.top10_psa),
            top10_vsa_csv=Path(args.top10_vsa),
            cif_dirs=cif_dirs,
            qst_csv=Path(args.qst_csv),
            benchmark_mof=args.benchmark_mof,
        )

        for suffix in ["PSA", "VSA"]:
            if suffix not in metadata:
                logger.warning("No metadata for %s, skipping", suffix)
                continue
            spsa_df = build_superpsa_csv(
                ext_fits, material_metadata=metadata[suffix],
            )
            out_path = superpsa_dir / f"Adsorbents_CH4N2_{suffix}_extDSL.csv"
            spsa_df.to_csv(out_path, index=False)
            logger.info("SuperPSA %s CSV: %s (%d rows)", suffix, out_path, len(spsa_df))

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    if not args.skip_plots:
        generate_diagnostics(per_temp_fits, twostep_fits, merged, output_dir,
                             global_fits=global_fits)
    else:
        logger.info("Skipping diagnostic plots (--skip-plots)")

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Extended DSL fitting complete (global fit)")
    logger.info("=" * 60)
    logger.info("  Per-temp fits    : %s (%d rows)", per_temp_csv, len(per_temp_fits))
    logger.info("  Two-step fits    : %s (%d rows)", twostep_csv, len(twostep_fits))
    logger.info("  ExtDSL fits      : %s (%d rows)", ext_csv, len(ext_fits))
    logger.info("  Summary JSON     : %s", summary_json)
    logger.info("  Validation issues: %d", val_summary["n_issues"])
    logger.info("  R2_global range  : [%.6f, %.6f]",
                ext_fits["R2_global"].min(), ext_fits["R2_global"].max())
    logger.info("  R2_global mean   : %.6f", ext_fits["R2_global"].mean())
    logger.info("  deltaU_b range   : [%.0f, %.0f] J/mol",
                ext_fits["deltaU_b"].min(), ext_fits["deltaU_b"].max())
    logger.info("  deltaU_d range   : [%.0f, %.0f] J/mol",
                ext_fits["deltaU_d"].min(), ext_fits["deltaU_d"].max())
    # Two-step diagnostics for comparison
    logger.info("  --- Two-step diagnostics (for comparison) ---")
    logger.info("  R2_arr_b range   : [%.4f, %.4f]",
                twostep_fits["R2_arrhenius_b"].min(),
                twostep_fits["R2_arrhenius_b"].max())
    logger.info("  R2_arr_d range   : [%.4f, %.4f]",
                twostep_fits["R2_arrhenius_d"].min(),
                twostep_fits["R2_arrhenius_d"].max())
    logger.info("  qs_cv_b range    : [%.4f, %.4f]",
                twostep_fits["qs_cv_b"].min(), twostep_fits["qs_cv_b"].max())
    logger.info("  qs_cv_d range    : [%.4f, %.4f]",
                twostep_fits["qs_cv_d"].min(), twostep_fits["qs_cv_d"].max())

    # Compare: how many MOFs pass all criteria (global vs two-step)
    logger.info("  --- Method comparison ---")
    # Global: deltaU < 0 AND R2_global > 0.999
    global_pass = ext_fits[
        (ext_fits["deltaU_b"] < 0) & (ext_fits["deltaU_d"] < 0) &
        (ext_fits["R2_global"] > 0.999)
    ]
    logger.info("  Global fit pass (dU<0, R2>0.999): %d/%d", len(global_pass), len(ext_fits))
    # Two-step: deltaU < 0 AND R2_arr > 0.90 AND qs_cv < 0.10
    ts_pass = twostep_fits[
        (twostep_fits["deltaU_b"] < 0) & (twostep_fits["deltaU_d"] < 0) &
        (twostep_fits["R2_arrhenius_b"] > 0.90) & (twostep_fits["R2_arrhenius_d"] > 0.90) &
        (twostep_fits["qs_cv_b"] < 0.10) & (twostep_fits["qs_cv_d"] < 0.10)
    ]
    logger.info("  Two-step pass (dU<0, R2_arr>0.90, cv<0.10): %d/%d",
                len(ts_pass), len(twostep_fits))


if __name__ == "__main__":
    main()
