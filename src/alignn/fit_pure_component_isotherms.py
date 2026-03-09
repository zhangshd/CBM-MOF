"""
fit_pure_component_isotherms.py — Task 3.1c: fit pure-component isotherms
with a unified model type per MOF for downstream breakthrough simulation.

For each MOF, fits both Langmuir and DSLangmuir using pyGAPS, then selects
a single model family that applies to BOTH CH4 and N2.

Selection rule (per MOF):
  1. Compare mean ranking_score across CH4/N2 for Langmuir vs DSLangmuir
  2. If tied, compare mean R²
  3. If still tied, pick the simpler model (Langmuir)

BKT mapping:
  Langmuir   → isomodel="Langmuir-Freundlich", ni=[1,1]
  DSLangmuir → isomodel="DSL"

Usage:
    python src/alignn/fit_pure_component_isotherms.py \\
        --input-csv .../atc_cu_pure_component.csv \\
        --input-csv .../top20_pure_component.csv \\
        --output-dir .../isotherm_fits

Reference: MOF-HTS/src/adsorption_analysis/isotherm_fitting.py (logic adapted, no import)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pygaps
import pygaps.modelling
import pygaps.parsing as pgp


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

CANDIDATE_MODELS = ["Langmuir", "DSLangmuir"]
MODEL_PARAM_COUNTS = {"Langmuir": 2, "DSLangmuir": 4}

STANDARD_COLUMNS = [
    "MofName", "GasName", "Temperature[K]", "Pressure[bar]",
    "AllComponents", "MoleculeFraction", "LoadingUnit",
    "AbsLoading", "ExcessLoading", "SimuDuration[h]", "FilePath", "Notes",
]


# ---------------------------------------------------------------------------
# Metric helpers (adapted from MOF-HTS/src/adsorption_analysis/utils.py)
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


def _ranking_score(r2: float, mae: float, n_params: int) -> float:
    """AIC-like ranking: higher = better. Penalises complexity."""
    complexity_penalty = 0.1 * n_params
    mae_score = 1.0 / (1.0 + mae) if mae > 0 else 1.0
    return max(0.0, 0.7 * r2 + 0.3 * mae_score - complexity_penalty)


# ---------------------------------------------------------------------------
# pyGAPS fitting
# ---------------------------------------------------------------------------

def _create_point_isotherm(
    pressures: np.ndarray,
    loadings: np.ndarray,
    mof_name: str,
    gas_name: str,
    temperature: float,
) -> Any:
    """Create a pyGAPS PointIsotherm (pressure bar, loading mmol/g = mol/kg)."""
    return pygaps.PointIsotherm(
        pressure=pressures.tolist(),
        loading=loadings.tolist(),
        material=mof_name,
        adsorbate=gas_name,
        temperature=temperature,
        pressure_mode="absolute",
        pressure_unit="bar",
        loading_basis="molar",
        loading_unit="mmol",
        material_basis="mass",
        material_unit="g",
        temperature_unit="K",
    )


def _fit_single_model(
    isotherm: Any,
    model_name: str,
) -> Optional[Dict]:
    """Fit *model_name* to *isotherm*; return result dict or None on failure."""
    try:
        fitted = pygaps.modelling.model_iso(isotherm, model=model_name, verbose=False)
        pressures = np.array(isotherm.pressure())
        exp_load = np.array(isotherm.loading())
        pred_load = np.array([fitted.loading_at(p) for p in pressures])

        r2 = _r_squared(exp_load, pred_load)
        mae = _mae(exp_load, pred_load)
        rmse = _rmse(exp_load, pred_load)
        n_params = MODEL_PARAM_COUNTS[model_name]
        rs = _ranking_score(r2, mae, n_params)

        params = fitted.model.to_dict()["parameters"]

        # Serialise fitted isotherm for later reloading
        fitted_json = json.loads(pgp.isotherm_to_json(fitted))

        return {
            "model_name": model_name,
            "parameters": params,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "ranking_score": rs,
            "n_params": n_params,
            "fitted_isotherm_dict": fitted_json,
            "experimental_pressures": pressures.tolist(),
            "experimental_loadings": exp_load.tolist(),
        }
    except Exception as e:
        print(f"    [WARN] {model_name} fit failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-MOF unified model selection
# ---------------------------------------------------------------------------

def _select_unified_model(
    per_model_fits: Dict[str, Dict[str, Dict]],
) -> Tuple[str, Dict]:
    """Pick one model family for a MOF across all gases.

    per_model_fits: {model_name: {gas_key: fit_result_dict, ...}, ...}
    Returns (selected_model, summary_dict).
    """
    candidates = []
    for model_name, gas_fits in per_model_fits.items():
        if not gas_fits:
            continue
        rs_vals = [f["ranking_score"] for f in gas_fits.values()]
        r2_vals = [f["R2"] for f in gas_fits.values()]
        candidates.append((
            np.mean(rs_vals),                # primary: mean ranking_score
            np.mean(r2_vals),                # secondary: mean R²
            -MODEL_PARAM_COUNTS[model_name], # tertiary: simpler is better
            model_name,
        ))

    if not candidates:
        raise ValueError("No successful fits for any model")

    _, _, _, best_model = max(candidates)
    # Reconstruct summary for the chosen model
    gas_fits = per_model_fits[best_model]
    rs_vals = [f["ranking_score"] for f in gas_fits.values()]
    r2_vals = [f["R2"] for f in gas_fits.values()]
    summary = {
        "selected_model": best_model,
        "mean_ranking_score": float(np.mean(rs_vals)),
        "mean_r2": float(np.mean(r2_vals)),
        "n_gases_fit": len(gas_fits),
    }
    return best_model, summary


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
    """Fit all MOFs. Returns (best_fit_df, selection_summary_dict)."""

    rows = []
    summary = {}

    mof_names = sorted(merged["MofName"].unique())
    print(f"\nFitting {len(mof_names)} MOFs × {len(CANDIDATE_MODELS)} models ...")

    for mof in mof_names:
        mof_df = merged[merged["MofName"] == mof]
        gases = sorted(mof_df["GasName"].unique())
        temps = mof_df["Temperature[K]"].unique()
        temp = float(temps[0])

        print(f"\n  {mof}  (gases={gases}, T={temp} K)")

        per_model_fits: Dict[str, Dict[str, Dict]] = {m: {} for m in CANDIDATE_MODELS}

        for gas in gases:
            sub = mof_df[mof_df["GasName"] == gas].sort_values("Pressure[bar]")
            pressures = sub["Pressure[bar]"].values.astype(float)
            # Use AbsLoading (mol/kg = mmol/g)
            loadings = sub["AbsLoading"].values.astype(float)

            iso = _create_point_isotherm(pressures, loadings, mof, gas, temp)

            for model_name in CANDIDATE_MODELS:
                result = _fit_single_model(iso, model_name)
                if result is not None:
                    gas_key = f"{gas}_{temp}K"
                    per_model_fits[model_name][gas_key] = result
                    print(f"    {gas:>8s} {model_name:>12s}  R²={result['R2']:.4f}  "
                          f"RS={result['ranking_score']:.4f}")

        # Select unified model
        try:
            best_model, sel_summary = _select_unified_model(per_model_fits)
        except ValueError:
            print(f"    [ERROR] No fits succeeded for {mof}")
            continue

        summary[mof] = sel_summary
        bkt_iso = "Langmuir-Freundlich" if best_model == "Langmuir" else "DSL"
        print(f"    → selected: {best_model}  (bkt_isomodel={bkt_iso})")

        # Flatten selected fits into rows
        for gas_key, fit in sorted(per_model_fits[best_model].items()):
            gas_name = gas_key.rsplit("_", 1)[0]
            params = fit["parameters"]
            row = {
                "MofName": mof,
                "gas_key": gas_key,
                "GasName": gas_name,
                "Temperature[K]": temp,
                "selected_model": best_model,
                "bkt_isomodel": bkt_iso,
                "R2": fit["R2"],
                "MAE": fit["MAE"],
                "RMSE": fit["RMSE"],
                "ranking_score": fit["ranking_score"],
                "sel_mean_ranking_score": sel_summary["mean_ranking_score"],
                "sel_mean_r2": sel_summary["mean_r2"],
                "n_points": len(fit["experimental_pressures"]),
                "pressure_min_bar": min(fit["experimental_pressures"]),
                "pressure_max_bar": max(fit["experimental_pressures"]),
                # Langmuir params (pyGAPS names: K, n_m)
                "K": params.get("K"),
                "n_m": params.get("n_m"),
                # DSLangmuir params (pyGAPS names: K1, n_m1, K2, n_m2)
                "K1": params.get("K1"),
                "n_m1": params.get("n_m1"),
                "K2": params.get("K2"),
                "n_m2": params.get("n_m2"),
            }
            rows.append(row)

    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.1c: Fit pure-component isotherms with unified model per MOF."
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
    model_counts = best_df.groupby("selected_model")["MofName"].nunique()
    mean_r2 = best_df["R2"].mean()
    min_r2 = best_df["R2"].min()
    print(f"  MOFs fitted : {n_mofs}")
    print(f"  Model usage : {model_counts.to_dict()}")
    print(f"  R² mean/min : {mean_r2:.4f} / {min_r2:.4f}")


if __name__ == "__main__":
    main()
