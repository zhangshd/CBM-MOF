"""
compute_iast_multicomp.py — Multi-composition IAST selectivity & working capacity.

Computes DSLF+IAST selectivity and working capacity for Top-20 candidates + ATC-Cu
across multiple CH4/N2 feed compositions.

For each MOF × composition × process (PSA/VSA):
  - α_IAST = (x_CH4/x_N2) / (y_CH4/y_N2)
  - WC_CH4 = q_CH4(P_high) - q_CH4(P_low)
  - API = (α - 1) × WC / |Qst_CH4|

Output: results/alignn/model_ep150/process_candidates/iast_multicomp/iast_multicomp.csv

Usage:
    conda run -n alignn_env --no-banner env PYTHONPATH=src python src/alignn/compute_iast_multicomp.py
    conda run -n alignn_env --no-banner env PYTHONPATH=src python src/alignn/compute_iast_multicomp.py --compositions 0.05 0.10 0.20 0.35 0.50
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

# Default feed compositions (y_CH4)
DEFAULT_COMPOSITIONS = [0.05, 0.10, 0.20, 0.35, 0.50]

# Process conditions: (P_high [bar], P_low [bar])
PROCESS_CONDITIONS = {
    "PSA": {"P_high": 10.0, "P_low": 1.0},
    "VSA": {"P_high": 1.0, "P_low": 0.1},
}


# ---------------------------------------------------------------------------
# Parameter extraction from CSV row
# ---------------------------------------------------------------------------

def _row_to_params(row, model_type):
    """Convert a CSV row to an IAST parameter dict for the given model."""
    if model_type == "DSLF":
        return {
            "qs1": row["qs1"], "b1": row["b1"], "n1": row.get("n1", 1.0),
            "qs2": row["qs2"], "b2": row["b2"], "n2": row.get("n2", 1.0),
        }
    elif model_type == "DSL":
        return {"qs1": row["qs1"], "b1": row["b1"],
                "qs2": row["qs2"], "b2": row["b2"]}
    elif model_type == "Langmuir-Freundlich":
        return {"qs": row["qs1"], "b": row["b1"], "n": row.get("n1", 1.0)}
    elif model_type == "Langmuir":
        return {"qs": row["qs1"], "b": row["b1"]}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# IAST solver wrapper
# ---------------------------------------------------------------------------

def iast_binary(params_1, params_2, y_ch4, P_total, model_type="DSLF"):
    """Solve binary IAST for CH4(1)/N2(2).

    Returns (alpha, q_ch4, q_n2) or (NaN, NaN, NaN) on failure.
    """
    y_n2 = 1.0 - y_ch4
    yi = np.array([y_ch4, y_n2])
    result = _iast_binary(yi, P_total, [params_1, params_2], model_type=model_type)

    if result is None:
        return np.nan, np.nan, np.nan

    q_ch4, q_n2 = result[0], result[1]
    if q_n2 > 0 and y_ch4 > 0:
        alpha = (q_ch4 / q_n2) * (y_n2 / y_ch4)
    else:
        alpha = np.nan
    return alpha, q_ch4, q_n2


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_multicomp(fits_csv, qst_csv, output_dir, compositions,
                      model_override=None):
    """Compute IAST at multiple compositions for all MOFs."""
    df_fits = pd.read_csv(fits_csv)
    df_qst = pd.read_csv(qst_csv)

    # Build Qst lookup: mof_id -> QstCH4 (use GCMC Qst when available)
    qst_lookup = {}
    for _, row in df_qst.iterrows():
        mof = row["mof_id"]
        qst = row.get("QstCH4_gcmc", np.nan)
        if pd.isna(qst):
            qst = row.get("QstCH4", np.nan)
        qst_lookup[mof] = qst

    # ATC-Cu Qst from literature (Cessford 2012 / Niu 2019)
    ATC_CU_ID = "CoRE-2020[Cu][pts]3[ASR]1"
    ATC_CU_QST = 29.0  # kJ/mol (OMS nano-trap, Niu 2019)
    if ATC_CU_ID not in qst_lookup:
        qst_lookup[ATC_CU_ID] = ATC_CU_QST

    mof_names = df_fits["MofName"].unique()
    print(f"Found {len(mof_names)} MOFs in {fits_csv.name}")
    print(f"Compositions (y_CH4): {compositions}")
    print(f"Processes: {list(PROCESS_CONDITIONS.keys())}")
    print()

    results = []
    n_failures = 0

    for mof in mof_names:
        mof_data = df_fits[df_fits["MofName"] == mof]
        ch4_row = mof_data[mof_data["gas_key"].str.contains("methane")]
        n2_row = mof_data[mof_data["gas_key"].str.contains("N2")]

        if ch4_row.empty or n2_row.empty:
            print(f"  WARNING: {mof} missing CH4 or N2 data, skipping")
            continue

        ch4_row = ch4_row.iloc[0]
        n2_row = n2_row.iloc[0]
        model_type = model_override or ch4_row.get("selected_model", "DSLF")

        params_ch4 = _row_to_params(ch4_row, model_type)
        params_n2 = _row_to_params(n2_row, model_type)

        qst_ch4 = qst_lookup.get(mof, np.nan)

        for y_ch4 in compositions:
            for process, cond in PROCESS_CONDITIONS.items():
                P_high = cond["P_high"]
                P_low = cond["P_low"]

                # Adsorption at P_high
                alpha, q_ch4_high, q_n2_high = iast_binary(
                    params_ch4, params_n2, y_ch4, P_high, model_type)

                # Desorption at P_low
                _, q_ch4_low, q_n2_low = iast_binary(
                    params_ch4, params_n2, y_ch4, P_low, model_type)

                # Working capacity
                wc_ch4 = q_ch4_high - q_ch4_low if not (
                    np.isnan(q_ch4_high) or np.isnan(q_ch4_low)) else np.nan

                # API
                if not np.isnan(alpha) and not np.isnan(wc_ch4) and not np.isnan(qst_ch4) and abs(qst_ch4) > 0:
                    api = (alpha - 1.0) * wc_ch4 / abs(qst_ch4)
                else:
                    api = np.nan

                if np.isnan(alpha):
                    n_failures += 1

                results.append({
                    "mof_id": mof,
                    "y_CH4": y_ch4,
                    "process": process,
                    "P_total_bar": P_high,
                    "P_low_bar": P_low,
                    "alpha_IAST": alpha,
                    "q_CH4": q_ch4_high,
                    "q_N2": q_n2_high,
                    "WC_CH4": wc_ch4,
                    "Qst_CH4": qst_ch4,
                    "API": api,
                    "model": model_type,
                })

    result_df = pd.DataFrame(results)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "iast_multicomp.csv"
    result_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Saved: {out_csv}")
    print(f"Total rows: {len(result_df)} ({len(mof_names)} MOFs × "
          f"{len(compositions)} compositions × {len(PROCESS_CONDITIONS)} processes)")
    if n_failures > 0:
        print(f"IAST failures: {n_failures}")

    return result_df


# ---------------------------------------------------------------------------
# Analysis & summary
# ---------------------------------------------------------------------------

def print_summary(df):
    """Print summary statistics and ranking reversal analysis."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Reference composition (y=0.20, standard CBM)
    ref_y = 0.20

    for process in ["PSA", "VSA"]:
        print(f"\n--- {process} ---")
        proc_df = df[df["process"] == process]

        # Print alpha range per composition
        for y in sorted(proc_df["y_CH4"].unique()):
            sub = proc_df[proc_df["y_CH4"] == y]
            valid = sub.dropna(subset=["alpha_IAST"])
            print(f"  y_CH4={y:.2f}: α range [{valid['alpha_IAST'].min():.3f}, "
                  f"{valid['alpha_IAST'].max():.3f}], "
                  f"WC range [{valid['WC_CH4'].min():.3f}, {valid['WC_CH4'].max():.3f}] mol/kg, "
                  f"n={len(valid)} MOFs")

        # Ranking reversal analysis: compare rankings at y=0.05 vs y=0.20 vs y=0.50
        if ref_y in proc_df["y_CH4"].values:
            ref = proc_df[proc_df["y_CH4"] == ref_y][["mof_id", "alpha_IAST", "API"]].copy()
            ref = ref.dropna(subset=["alpha_IAST"]).sort_values("alpha_IAST", ascending=False)
            ref["rank_ref"] = range(1, len(ref) + 1)

            for y_comp in sorted(proc_df["y_CH4"].unique()):
                if y_comp == ref_y:
                    continue
                comp = proc_df[proc_df["y_CH4"] == y_comp][["mof_id", "alpha_IAST"]].copy()
                comp = comp.dropna(subset=["alpha_IAST"]).sort_values("alpha_IAST", ascending=False)
                comp["rank_comp"] = range(1, len(comp) + 1)

                merged = ref[["mof_id", "rank_ref"]].merge(
                    comp[["mof_id", "rank_comp"]], on="mof_id", how="inner")
                if len(merged) > 0:
                    rank_diff = (merged["rank_ref"] - merged["rank_comp"]).abs()
                    n_reversals = (rank_diff >= 3).sum()
                    max_shift = rank_diff.max()
                    tau = merged["rank_ref"].corr(merged["rank_comp"], method="kendall")
                    print(f"  Ranking y={ref_y:.2f} vs y={y_comp:.2f}: "
                          f"tau={tau:.3f}, reversals(>=3)={n_reversals}, max_shift={max_shift}")

    # ATC-Cu comparison
    atc = df[df["mof_id"] == "CoRE-2020[Cu][pts]3[ASR]1"]
    if not atc.empty:
        print(f"\n--- ATC-Cu (reference) ---")
        for _, row in atc.iterrows():
            print(f"  y={row['y_CH4']:.2f} {row['process']}: "
                  f"α={row['alpha_IAST']:.3f}, WC={row['WC_CH4']:.3f}, "
                  f"API={row['API']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-composition IAST selectivity & working capacity."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model results dir (default: results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["Langmuir", "Langmuir-Freundlich", "DSL", "DSLF"],
        help="Override isotherm model (default: from CSV).",
    )
    parser.add_argument(
        "--compositions", type=float, nargs="+", default=DEFAULT_COMPOSITIONS,
        help=f"CH4 mole fractions to evaluate (default: {DEFAULT_COMPOSITIONS}).",
    )
    args = parser.parse_args()

    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = REPO_ROOT / md
    else:
        md = REPO_ROOT / "results" / "alignn" / "model_ep150"

    bkt_dir = md / "process_candidates"
    fits_csv = bkt_dir / "isotherm_fits" / "best_isotherm_fits.csv"
    qst_csv = bkt_dir / "top20_combined.csv"
    output_dir = bkt_dir / "iast_multicomp"

    if not fits_csv.exists():
        print(f"ERROR: {fits_csv} not found")
        sys.exit(1)
    if not qst_csv.exists():
        print(f"ERROR: {qst_csv} not found")
        sys.exit(1)

    model_label = args.model or "auto (from CSV)"
    print("=" * 70)
    print("Multi-Composition IAST Selectivity & Working Capacity")
    print("=" * 70)
    print(f"Input fits:  {fits_csv}")
    print(f"Input Qst:   {qst_csv}")
    print(f"Output dir:  {output_dir}")
    print(f"Model:       {model_label}")
    print(f"Compositions: {args.compositions}")
    for proc, cond in PROCESS_CONDITIONS.items():
        print(f"  {proc}: P_high={cond['P_high']} bar, P_low={cond['P_low']} bar")
    print()

    result_df = compute_multicomp(
        fits_csv, qst_csv, output_dir, args.compositions,
        model_override=args.model,
    )
    print_summary(result_df)


if __name__ == "__main__":
    main()
