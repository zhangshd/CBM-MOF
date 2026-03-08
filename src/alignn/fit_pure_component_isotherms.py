"""
fit_pure_component_isotherms.py — Task 3.1c: fit pure-component isotherms
with a unified model type per MOF for downstream breakthrough simulation.

This script reuses the MOF-HTS isotherm fitting implementation but enforces
that CH4 and N2 for the same MOF must share the same final model family
(Langmuir or DSLangmuir).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MOF_HTS_SRC = Path("/home/zhangsd/repos/MOF-HTS/src")

DEFAULT_INPUT_COLUMNS = [
    "MofName",
    "GasName",
    "Temperature[K]",
    "Pressure[bar]",
    "AllComponents",
    "MoleculeFraction",
    "LoadingUnit",
    "AbsLoading",
    "ExcessLoading",
    "SimuDuration[h]",
    "FilePath",
    "Notes",
]

DEFAULT_INPUT_CSVS = [
    REPO_ROOT
    / "results"
    / "alignn"
    / "model_ep150"
    / "bkt_candidates"
    / "isotherm_input"
    / "atc_cu_pure_component.csv"
]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "alignn"
    / "model_ep150"
    / "bkt_candidates"
    / "isotherm_fits"
)

MODEL_PARAMETER_COUNTS = {
    "Langmuir": 2,
    "DSLangmuir": 4,
}


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_input_csvs(csv_paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate standard pure-component CSV inputs."""
    frames = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        _require_columns(frame, DEFAULT_INPUT_COLUMNS)
        frames.append(frame[DEFAULT_INPUT_COLUMNS].copy())

    merged = pd.concat(frames, ignore_index=True)
    return merged


def fit_candidate_models(
    data_csv: Path,
    models: Iterable[str] = ("Langmuir", "DSLangmuir"),
) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """Fit each candidate model separately and keep results grouped by MOF."""
    if str(MOF_HTS_SRC) not in sys.path:
        sys.path.insert(0, str(MOF_HTS_SRC))

    from adsorption_analysis import IsothermFitter

    all_results: Dict[str, Dict[str, Dict[str, Dict]]] = {}

    for model_name in models:
        fitter = IsothermFitter(
            data_file=data_csv,
            output_dir="/tmp/isotherm_fit_probe",
            models_to_fit=[model_name],
        )
        model_results = fitter.fit_all_isotherms()
        for mof_name, gas_results in model_results.items():
            all_results.setdefault(mof_name, {})
            all_results[mof_name].setdefault(model_name, {})
            all_results[mof_name][model_name].update(gas_results)

    return all_results


def select_unified_model_for_mof(
    model_results: Dict[str, Dict[str, Dict]],
) -> Tuple[str, Dict[str, float]]:
    """Select one common model family for all gases of a single MOF."""
    ranked = []
    for model_name, gas_results in model_results.items():
        if not gas_results:
            continue
        ranking_scores = [
            float(result["fit_quality"]["ranking_score"])
            for result in gas_results.values()
        ]
        r2_scores = [
            float(result["fit_quality"]["R2"])
            for result in gas_results.values()
        ]
        ranked.append(
            (
                sum(ranking_scores) / len(ranking_scores),
                sum(r2_scores) / len(r2_scores),
                -MODEL_PARAMETER_COUNTS[model_name],
                model_name,
                {
                    "selected_model": model_name,
                    "mean_ranking_score": sum(ranking_scores) / len(ranking_scores),
                    "mean_r2": sum(r2_scores) / len(r2_scores),
                    "n_gases_fit": len(gas_results),
                },
            )
        )

    if not ranked:
        raise ValueError("No successful model fits available for MOF")

    _, _, _, selected_model, summary = max(ranked)
    return selected_model, summary


def flatten_selected_fit(
    mof_name: str,
    gas_key: str,
    fit_result: Dict,
    selection_summary: Dict[str, float],
) -> Dict:
    """Flatten MOF-HTS fit result JSON into one CSV row."""
    fit_iso = fit_result["fitted_isotherm"]
    model_info = fit_iso["isotherm_model"]
    params = model_info["parameters"]
    pressures = fit_result["experimental_data"]["pressures"]

    gas_name = gas_key.rsplit("_", 1)[0]
    row = {
        "MofName": mof_name,
        "gas_key": gas_key,
        "GasName": gas_name,
        "Temperature[K]": fit_iso["temperature"],
        "selected_model": selection_summary["selected_model"],
        "fit_model": model_info["name"],
        "bkt_isomodel": (
            "Langmuir-Freundlich"
            if selection_summary["selected_model"] == "Langmuir"
            else "DSL"
        ),
        "R2": fit_result["fit_quality"]["R2"],
        "MAE": fit_result["fit_quality"]["MAE"],
        "RMSE": fit_result["fit_quality"]["RMSE"],
        "ranking_score": fit_result["fit_quality"]["ranking_score"],
        "selection_mean_ranking_score": selection_summary["mean_ranking_score"],
        "selection_mean_r2": selection_summary["mean_r2"],
        "selection_n_gases_fit": selection_summary["n_gases_fit"],
        "pressure_unit": fit_iso["pressure_unit"],
        "loading_unit": fit_iso["loading_unit"],
        "n_points": len(pressures),
        "pressure_min_bar": min(pressures),
        "pressure_max_bar": max(pressures),
        "K": params.get("K"),
        "n_m": params.get("n_m"),
        "K1": params.get("K1"),
        "n_m1": params.get("n_m1"),
        "K2": params.get("K2"),
        "n_m2": params.get("n_m2"),
    }
    return row


def build_best_fit_table(
    fitted_results: Dict[str, Dict[str, Dict[str, Dict]]]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Select one model per MOF and flatten selected gas fits into a table."""
    rows = []
    summary = {}

    for mof_name, model_results in sorted(fitted_results.items()):
        selected_model, selection_summary = select_unified_model_for_mof(model_results)
        summary[mof_name] = selection_summary

        for gas_key, fit_result in sorted(model_results[selected_model].items()):
            rows.append(
                flatten_selected_fit(
                    mof_name=mof_name,
                    gas_key=gas_key,
                    fit_result=fit_result,
                    selection_summary=selection_summary,
                )
            )

    return pd.DataFrame(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.1c: Fit pure-component isotherms with a unified model per MOF."
    )
    parser.add_argument(
        "--input-csv",
        dest="input_csvs",
        action="append",
        default=None,
        help="Input pure-component CSV. Repeat to supply multiple files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for merged input and best-fit outputs.",
    )
    args = parser.parse_args()

    input_csvs = [Path(p) for p in (args.input_csvs or [str(p) for p in DEFAULT_INPUT_CSVS])]
    output_dir = Path(args.output_dir)

    merged_input = load_input_csvs(input_csvs)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = output_dir / "pure_component_data_merged.csv"
    merged_input.to_csv(merged_csv, index=False)

    fitted_results = fit_candidate_models(merged_csv)
    best_fit_df, selection_summary = build_best_fit_table(fitted_results)

    best_fit_csv = output_dir / "best_isotherm_fits.csv"
    summary_json = output_dir / "model_selection_summary.json"

    best_fit_df.to_csv(best_fit_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2)

    print(f"Merged input : {merged_csv}")
    print(f"Best-fit CSV : {best_fit_csv}")
    print(f"Summary JSON : {summary_json}")
    print(f"Rows written : {len(best_fit_df)}")


if __name__ == "__main__":
    main()
