"""
plot_bkt_paper_figures.py — Task 3.3: Generate BKT paper figures and tables.

Generates:
  Step 1: 2-panel breakthrough curve overlay (PSA + VSA, with ATC-Cu baseline)
  Step 2: Performance comparison table (Top-10 vs ATC-Cu, Markdown + CSV)
  Step 3: 2-panel selectivity comparison (GCMC thermodynamic vs BKT dynamic)
  Step 4: Multi-panel isotherm fit figure (SI)

Usage:
    python src/alignn/plot_bkt_paper_figures.py [--step 1|2|3|4|all]
    python src/alignn/plot_bkt_paper_figures.py --model-dir results/alignn/model_ep150
"""

import argparse
import sys
from typing import Optional
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Publication style
sys.path.insert(0, str(REPO_ROOT / "src" / "figures"))
from style import (
    set_publication_style, save_figure,
    SINGLE_COL_INCH, DOUBLE_COL_INCH, MAX_HEIGHT_INCH, DPI,
)
from alignn.bkt_curve_cache import CURVE_CACHE_COLUMNS

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Constants from run_bkt_top_candidates.py
# ---------------------------------------------------------------------------
ATC_CU_NAME = "CoRE-2020[Cu][pts]3[ASR]1"

# Clustered CSV for looking up ATC-Cu cluster ID
CLUSTERED_CSV = (
    REPO_ROOT / "data" / "processed" / "textural_screened"
    / "textural_screened_clustered_with_umap.csv"
)

# ATC-Cu GCMC data paths (for thermodynamic selectivity)
TRAINING_ADS_R1_CSV = (
    REPO_ROOT / "results" / "cbm_screening"
    / "gcmc_round1_DreidingTraPPEJson" / "raspa3_parsed_results_0911.csv"
)
TRAINING_WIDOM_R1_CSV = (
    REPO_ROOT / "results" / "cbm_screening"
    / "widom_round1_DREIDING" / "widom_results_0911.csv"
)
BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"

# Color palette for 10 candidates + ATC-Cu benchmark
CANDIDATE_COLORS = plt.cm.tab10.colors[:10]
BENCHMARK_COLOR = "black"
BENCHMARK_LINESTYLE = "--"


def get_default_fig_dir(model_dir: Path) -> Path:
    """Return the default figure directory for a given model results dir."""
    return model_dir / "figures"

# ---------------------------------------------------------------------------
# Helper: MOF ID simplification
# ---------------------------------------------------------------------------

def simplify_mof_id(mof_id: str) -> str:
    """Shorten MOF ID for figure labels."""
    s = mof_id
    # Remove common prefixes/suffixes
    for prefix in ["ARC-DB0-", "ARC-DB1-"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.endswith("_repeat"):
        s = s[:-7]
    # Shorten CoRE-2020 benchmark
    if "CoRE-2020" in mof_id:
        return "ATC-Cu"
    return s


# ---------------------------------------------------------------------------
# Helper: ATC-Cu thermodynamic selectivity
# ---------------------------------------------------------------------------

def get_atc_cu_thermo_selectivity():
    """Get ATC-Cu GCMC thermodynamic selectivity from Round 1 data."""
    ads_r1 = pd.read_csv(TRAINING_ADS_R1_CSV)
    widom_r1 = pd.read_csv(TRAINING_WIDOM_R1_CSV)

    # Pivot adsorption data
    ads_piv = ads_r1.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"],
        values="AbsLoading", aggfunc="first",
    )
    ads_piv.columns = [f"Ads{g}_{int(p*100)}kPa"
                       for g, p in ads_piv.columns]
    # Simpler: reconstruct column names to match expected format
    ads_piv2 = ads_r1.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"],
        values="AbsLoading", aggfunc="first",
    )

    # Extract ATC-Cu adsorption values at key pressures
    atc = ads_r1[ads_r1["MofName"] == BENCHMARK_MOF]
    result = {}

    for gas, gas_label in [("methane", "CH4"), ("N2", "N2")]:
        gas_data = atc[atc["GasName"] == gas]
        for p_bar in [0.1, 1.0, 10.0]:
            p_key = f"{int(p_bar*100)}kPa"
            # Map pressure to kPa label
            p_kpa = p_bar * 100
            if p_kpa == 100:
                key = f"Ads{gas_label}_100kPa"
            elif p_kpa == 1000:
                key = f"Ads{gas_label}_1000kPa"
            elif p_kpa == 10:
                key = f"Ads{gas_label}_10kPa"
            else:
                continue
            row = gas_data[np.isclose(gas_data["Pressure[bar]"], p_bar)]
            if not row.empty:
                result[key] = float(row["AbsLoading"].iloc[0])

    # Compute selectivities: α = (q_CH4 / q_N2) × (y_N2 / y_CH4) = (q_CH4 / q_N2) × 4
    if "AdsCH4_1000kPa" in result and "AdsN2_1000kPa" in result:
        result["PSA_alpha"] = (result["AdsCH4_1000kPa"] / result["AdsN2_1000kPa"]) * 4
    # VSA: at 1 bar (key "100kPa" = p_bar=1.0)
    if "AdsCH4_100kPa" in result and "AdsN2_100kPa" in result:
        result["VSA_alpha"] = (result["AdsCH4_100kPa"] / result["AdsN2_100kPa"]) * 4

    return result


# ---------------------------------------------------------------------------
# Helper: Load BKT summary data
# ---------------------------------------------------------------------------

def load_bkt_summaries(bkt_dir: Path) -> pd.DataFrame:
    """Load and concatenate all bkt_summary_job*.csv files."""
    files = sorted((bkt_dir / "summaries" / "jobs").glob("bkt_summary_job*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No bkt_summary_job*.csv found in {bkt_dir / 'summaries' / 'jobs'}"
        )
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Drop duplicated header rows (from cat concat)
    df = df[df["mof"] != "mof"].copy()
    # Convert numeric columns
    for col in ["q_CH4_mol_per_kg", "q_N2_mol_per_kg", "rho_s", "feed_pressure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_breakthrough_curve_cache(bkt_dir: Path) -> pd.DataFrame:
    """Load cached breakthrough curves generated by run_bkt_top_candidates.py."""
    cache_csv = bkt_dir / "breakthrough_curves_data.csv"
    if not cache_csv.exists():
        raise FileNotFoundError(
            "Missing breakthrough curve cache: "
            f"{cache_csv}. Run "
            "`python src/alignn/run_bkt_top_candidates.py --rebuild-curve-cache` "
            "after generating per-run BKT outputs."
        )
    df = pd.read_csv(cache_csv)
    missing = [col for col in CURVE_CACHE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{cache_csv} is missing required columns: {missing}")
    return df[CURVE_CACHE_COLUMNS].copy()


# ===================================================================
# STEP 1: Breakthrough Curve Overlay (2-panel)
# ===================================================================

def step1_breakthrough_overlay(bkt_dir: Path, fig_dir: Path):
    """Load cached C/C0 curves and generate the 2-panel breakthrough figure."""
    print("\n" + "=" * 70)
    print("STEP 1: Breakthrough Curve Overlay")
    print("=" * 70)

    curve_df = load_breakthrough_curve_cache(bkt_dir)
    psa_mofs = pd.read_csv(bkt_dir / "top10_psa.csv")["mof_id"].tolist()
    vsa_mofs = pd.read_csv(bkt_dir / "top10_vsa.csv")["mof_id"].tolist()

    curves = {"PSA": [], "VSA": []}
    process_order = {
        "PSA": psa_mofs + [ATC_CU_NAME],
        "VSA": vsa_mofs + [ATC_CU_NAME],
    }

    for process, ordered_mofs in process_order.items():
        for mof_name in ordered_mofs:
            mof_curve = curve_df[
                (curve_df["process"] == process) & (curve_df["mof"] == mof_name)
            ].sort_values("time_min")
            if mof_curve.empty:
                print(f"  WARNING: Missing cached curve for {mof_name} [{process}]")
                continue
            curves[process].append({
                "mof": mof_name,
                "label": simplify_mof_id(mof_name),
                "time_min": mof_curve["time_min"].to_numpy(),
                "cc0_ch4": mof_curve["CC0_CH4"].to_numpy(),
                "is_benchmark": mof_name == ATC_CU_NAME,
            })

    # Generate 2-panel figure
    set_publication_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(1.08 * DOUBLE_COL_INCH, 0.60 * DOUBLE_COL_INCH)
    )

    for ax, process, title in zip(
        axes, ["PSA", "VSA"],
        [r"(a) PSA Case (10 bar)", r"(b) VSA Case (1 bar)"]
    ):
        proc_curves = curves[process]
        # Sort: candidates first (by label), benchmark last
        candidates = [c for c in proc_curves if not c["is_benchmark"]]
        benchmarks = [c for c in proc_curves if c["is_benchmark"]]

        # Sort: Benchmark first, then candidates
        for curve in benchmarks:
            ax.plot(
                curve["time_min"][1:], curve["cc0_ch4"][1:],
                color=BENCHMARK_COLOR, linestyle=BENCHMARK_LINESTYLE,
                linewidth=1.2, label="ATC-Cu", zorder=3,
            )

        for idx, curve in enumerate(candidates):
            color = CANDIDATE_COLORS[idx % len(CANDIDATE_COLORS)]
            # Skip first point (t=0, C/C0 undefined)
            ax.plot(
                curve["time_min"][1:], curve["cc0_ch4"][1:],
                color=color, linewidth=0.8,
                label=curve["label"], zorder=2,
            )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel(r"$C_{\mathrm{CH_4}}/C_0$")
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlim(left=0)

        ax.legend(fontsize=5.5, loc="upper left", ncol=1,
                  framealpha=0.8, edgecolor="none",
                  handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout(w_pad=0.5, h_pad=0.2)
    save_figure(fig, "FigX_breakthrough_overlay", fig_dir, formats=("png",))
    plt.close(fig)
    print(f"\nStep 1 complete: breakthrough overlay figure saved.")


# ===================================================================
# STEP 2: Performance Comparison Table
# ===================================================================

def _lookup_atc_cu_cluster() -> int:
    """Look up ATC-Cu cluster ID from the clustered CSV."""
    if not CLUSTERED_CSV.exists():
        print(f"  WARNING: Clustered CSV not found: {CLUSTERED_CSV}")
        return -1
    df = pd.read_csv(CLUSTERED_CSV, usecols=["CifId", "Cluster"])
    match = df[df["CifId"] == ATC_CU_NAME]
    if match.empty:
        print(f"  WARNING: ATC-Cu ({ATC_CU_NAME}) not found in clustered CSV")
        return -1
    cluster_id = int(match["Cluster"].iloc[0])
    print(f"  ATC-Cu cluster ID: {cluster_id} (from {CLUSTERED_CSV.name})")
    return cluster_id


def step2_performance_table(bkt_dir: Path, fig_dir: Path):
    """Generate Top-10 vs ATC-Cu performance comparison tables.

    Includes α_thermo (GCMC thermodynamic selectivity), α_IAST (IAST from
    pure-component Langmuir fits), α_dyn (BKT dynamic selectivity), and
    derived ratios.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Performance Comparison Table")
    print("=" * 70)

    bkt_df = load_bkt_summaries(bkt_dir)
    psa_top10 = pd.read_csv(bkt_dir / "top10_psa.csv")
    vsa_top10 = pd.read_csv(bkt_dir / "top10_vsa.csv")

    # ATC-Cu thermodynamic selectivity from Round 1 GCMC
    atc_thermo = get_atc_cu_thermo_selectivity()
    atc_alpha_thermo = {
        "PSA": atc_thermo.get("PSA_alpha", np.nan),
        "VSA": atc_thermo.get("VSA_alpha", np.nan),
    }
    print(f"  ATC-Cu α_thermo: PSA={atc_alpha_thermo['PSA']:.2f}, "
          f"VSA={atc_alpha_thermo['VSA']:.2f}")

    # Load IAST selectivity
    iast_csv = bkt_dir / "iast_selectivity.csv"
    if iast_csv.exists():
        iast_df = pd.read_csv(iast_csv)
        print(f"  Loaded IAST data: {len(iast_df)} MOFs")
    else:
        print(f"  WARNING: {iast_csv} not found — run compute_iast_selectivity.py first")
        iast_df = None

    # ATC-Cu cluster ID
    atc_cluster = _lookup_atc_cu_cluster()

    # GCMC alpha column name mapping
    gcmc_alpha_col = {
        "PSA": "gcmc_PSA_alpha_CH4_N2",
        "VSA": "gcmc_VSA_alpha_CH4_N2",
    }

    # Build tables for each process
    md_lines = ["# BKT Performance Comparison: Top-10 vs ATC-Cu\n"]

    for process, top10_df, rank_col in [
        ("PSA", psa_top10, "psa_top10_rank"),
        ("VSA", vsa_top10, "vsa_top10_rank"),
    ]:
        alpha_col = gcmc_alpha_col[process]
        proc_bkt = bkt_df[bkt_df["process"] == process].copy()
        atc_row = proc_bkt[proc_bkt["mof"] == ATC_CU_NAME]
        cand_rows = proc_bkt[proc_bkt["mof"] != ATC_CU_NAME]

        # Merge with rank/cluster/alpha_thermo info
        merge_cols = ["mof_id", "cluster", rank_col]
        if alpha_col in top10_df.columns:
            merge_cols.append(alpha_col)
        merged = cand_rows.merge(
            top10_df[merge_cols],
            left_on="mof", right_on="mof_id", how="left",
        )
        merged = merged.sort_values(rank_col)

        # Compute dynamic selectivity
        merged["alpha_dyn"] = (merged["q_CH4_mol_per_kg"] / merged["q_N2_mol_per_kg"]) * 4

        # Thermodynamic selectivity from GCMC
        if alpha_col in merged.columns:
            merged["alpha_thermo"] = merged[alpha_col]
        else:
            merged["alpha_thermo"] = np.nan

        # IAST selectivity
        iast_col = f"alpha_IAST_{process}"
        if iast_df is not None and iast_col in iast_df.columns:
            merged = merged.merge(
                iast_df[["MofName", iast_col]],
                left_on="mof", right_on="MofName", how="left",
            )
            merged["alpha_IAST"] = merged[iast_col]
            merged.drop(columns=["MofName"], inplace=True, errors="ignore")
        else:
            merged["alpha_IAST"] = np.nan

        # Ratios
        merged["ratio_dyn_thermo"] = merged["alpha_dyn"] / merged["alpha_thermo"]
        merged["ratio_iast_thermo"] = merged["alpha_IAST"] / merged["alpha_thermo"]

        # ATC-Cu values
        if not atc_row.empty:
            atc = atc_row.iloc[0]
            atc_q_ch4 = atc["q_CH4_mol_per_kg"]
            atc_q_n2 = atc["q_N2_mol_per_kg"]
            atc_alpha_dyn = (atc_q_ch4 / atc_q_n2) * 4
            atc_alpha_th = atc_alpha_thermo[process]
            atc_ratio_dt = atc_alpha_dyn / atc_alpha_th if atc_alpha_th else np.nan
            # ATC-Cu IAST
            if iast_df is not None:
                atc_iast_row = iast_df[iast_df["MofName"] == ATC_CU_NAME]
                atc_alpha_iast = float(atc_iast_row[iast_col].iloc[0]) if not atc_iast_row.empty else np.nan
            else:
                atc_alpha_iast = np.nan
            atc_ratio_it = atc_alpha_iast / atc_alpha_th if atc_alpha_th and not np.isnan(atc_alpha_iast) else np.nan
        else:
            atc_q_ch4, atc_q_n2, atc_alpha_dyn = np.nan, np.nan, np.nan
            atc_alpha_th, atc_ratio_dt = np.nan, np.nan
            atc_alpha_iast, atc_ratio_it = np.nan, np.nan

        merged["vs_ATC_Cu"] = merged["q_CH4_mol_per_kg"] / atc_q_ch4

        # Build Markdown table
        pressure_label = "10 bar" if process == "PSA" else "1 bar"
        md_lines.append(f"\n## {process} Process ({pressure_label})\n")
        md_lines.append(
            "| Rank | MOF ID | Cluster | ρ_s (kg/m³) | "
            "q_CH₄ (mol/kg) | q_N₂ (mol/kg) | α_thermo | α_IAST | α_dyn | "
            "α_dyn/α_thermo | α_IAST/α_thermo | vs ATC-Cu |"
        )
        md_lines.append(
            "|------|--------|---------|-------------|"
            "----------------|----------------|----------|--------|-------"
            "|----------------|-----------------|-----------|"
        )

        for _, row in merged.iterrows():
            rank = int(row[rank_col]) if pd.notna(row.get(rank_col)) else "—"
            cluster = int(row["cluster"]) if pd.notna(row.get("cluster")) else "—"
            a_iast = f"{row['alpha_IAST']:.2f}" if pd.notna(row.get("alpha_IAST")) else "—"
            r_it = f"{row['ratio_iast_thermo']:.2f}" if pd.notna(row.get("ratio_iast_thermo")) else "—"
            md_lines.append(
                f"| {rank} | {simplify_mof_id(row['mof'])} | {cluster} | "
                f"{row['rho_s']:.0f} | "
                f"{row['q_CH4_mol_per_kg']:.3f} | {row['q_N2_mol_per_kg']:.3f} | "
                f"{row['alpha_thermo']:.2f} | "
                f"{a_iast} | "
                f"{row['alpha_dyn']:.2f} | {row['ratio_dyn_thermo']:.2f} | "
                f"{r_it} | "
                f"{row['vs_ATC_Cu']:.2f}× |"
            )

        # ATC-Cu baseline row
        a_iast_atc = f"{atc_alpha_iast:.2f}" if not np.isnan(atc_alpha_iast) else "—"
        r_it_atc = f"{atc_ratio_it:.2f}" if not np.isnan(atc_ratio_it) else "—"
        md_lines.append(
            f"| — | **ATC-Cu** | {atc_cluster} | "
            f"{atc['rho_s']:.0f} | "
            f"{atc_q_ch4:.3f} | {atc_q_n2:.3f} | "
            f"{atc_alpha_th:.2f} | "
            f"{a_iast_atc} | "
            f"{atc_alpha_dyn:.2f} | {atc_ratio_dt:.2f} | "
            f"{r_it_atc} | 1.00× |"
        )

        # Summary note
        ratios_dt = merged["ratio_dyn_thermo"].dropna()
        ratios_it = merged["ratio_iast_thermo"].dropna()
        md_lines.append(
            f"\nα_thermo: GCMC mixed-component thermodynamic selectivity. "
            f"α_IAST: IAST from pure-component Langmuir. "
            f"α_dyn: BKT dynamic selectivity."
        )
        if not ratios_dt.empty:
            md_lines.append(
                f"\n**{process} summary**: α_dyn/α_thermo = {ratios_dt.min():.2f}–{ratios_dt.max():.2f} "
                f"(mean {ratios_dt.mean():.2f}). "
                f"α_IAST/α_thermo = {ratios_it.min():.2f}–{ratios_it.max():.2f} "
                f"(mean {ratios_it.mean():.2f})."
            )

        # Save per-process CSV
        out_cols = ["mof", "cluster", "rho_s", "q_CH4_mol_per_kg",
                    "q_N2_mol_per_kg", "alpha_thermo", "alpha_IAST", "alpha_dyn",
                    "ratio_dyn_thermo", "ratio_iast_thermo", "vs_ATC_Cu"]
        csv_path = bkt_dir / f"performance_table_{process.lower()}.csv"
        merged[out_cols].to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Save combined Markdown
    md_path = bkt_dir / "performance_table_combined.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")
    print(f"\nStep 2 complete: performance tables generated.")


# ===================================================================
# STEP 3: Selectivity Comparison (GCMC Thermodynamic vs BKT Dynamic)
# ===================================================================

def _prepare_selectivity_process_df(
    bkt_df: pd.DataFrame,
    top10_df: pd.DataFrame,
    iast_df: Optional[pd.DataFrame],
    atc_thermo: dict,
    process: str,
) -> pd.DataFrame:
    """Build a plotting table for one process."""
    proc_bkt = bkt_df[bkt_df["process"] == process].copy()
    proc_bkt["alpha_BKT"] = (
        proc_bkt["q_CH4_mol_per_kg"] / proc_bkt["q_N2_mol_per_kg"]
    ) * 4

    gcmc_col = f"gcmc_{process}_alpha_CH4_N2"
    rank_col = f"{process.lower()}_top10_rank"
    merged = proc_bkt.merge(
        top10_df[["mof_id", gcmc_col, rank_col]],
        left_on="mof",
        right_on="mof_id",
        how="left",
    )
    merged["alpha_GCMC"] = merged[gcmc_col]

    iast_col = f"alpha_IAST_{process}"
    if iast_df is not None and iast_col in iast_df.columns:
        merged = merged.merge(
            iast_df[["MofName", iast_col]],
            left_on="mof",
            right_on="MofName",
            how="left",
        )
        merged["alpha_IAST"] = merged[iast_col]
    else:
        merged["alpha_IAST"] = np.nan

    atc_mask = merged["mof"] == ATC_CU_NAME
    if atc_mask.any():
        merged.loc[atc_mask, "alpha_GCMC"] = atc_thermo.get(f"{process}_alpha", np.nan)

    merged["label"] = merged["mof"].map(simplify_mof_id)
    merged["is_benchmark"] = merged["mof"] == ATC_CU_NAME
    merged = merged.sort_values(
        ["alpha_BKT", "is_benchmark"], ascending=[False, False]
    ).reset_index(drop=True)
    merged["ypos"] = np.arange(len(merged))[::-1]
    return merged

def step3_selectivity_comparison(bkt_dir: Path, fig_dir: Path):
    """Generate 2-panel dumbbell plot comparing GCMC, IAST, and BKT selectivities."""
    print("\n" + "=" * 70)
    print("STEP 3: Selectivity Comparison (3-way: GCMC / IAST / BKT)")
    print("=" * 70)

    bkt_df = load_bkt_summaries(bkt_dir)
    psa_top10 = pd.read_csv(bkt_dir / "top10_psa.csv")
    vsa_top10 = pd.read_csv(bkt_dir / "top10_vsa.csv")

    # ATC-Cu thermodynamic selectivity
    atc_thermo = get_atc_cu_thermo_selectivity()
    print(f"  ATC-Cu PSA α_thermo = {atc_thermo.get('PSA_alpha', 'N/A'):.2f}")
    print(f"  ATC-Cu VSA α_thermo = {atc_thermo.get('VSA_alpha', 'N/A'):.2f}")

    # Load IAST selectivity
    iast_csv = bkt_dir / "iast_selectivity.csv"
    if iast_csv.exists():
        iast_df = pd.read_csv(iast_csv)
        print(f"  Loaded IAST data: {len(iast_df)} MOFs")
    else:
        print(f"  WARNING: {iast_csv} not found — run compute_iast_selectivity.py first")
        iast_df = None

    psa_df = _prepare_selectivity_process_df(
        bkt_df=bkt_df,
        top10_df=psa_top10,
        iast_df=iast_df,
        atc_thermo=atc_thermo,
        process="PSA",
    )
    vsa_df = _prepare_selectivity_process_df(
        bkt_df=bkt_df,
        top10_df=vsa_top10,
        iast_df=iast_df,
        atc_thermo=atc_thermo,
        process="VSA",
    )

    set_publication_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(1.16 * DOUBLE_COL_INCH, 0.78 * DOUBLE_COL_INCH), sharey=False
    )

    style_map = {
        "alpha_GCMC": dict(color="#1f77b4", marker="o", label=r"$\alpha_{\mathrm{GCMC}}$"),
        "alpha_IAST": dict(color="#d55e00", marker="^", label=r"$\alpha_{\mathrm{IAST}}$"),
        "alpha_BKT": dict(color="black", marker="s", label=r"$\alpha_{\mathrm{BKT}}$"),
    }

    for ax, df, title in zip(
        axes,
        [psa_df, vsa_df],
        ["(a) PSA Case (10 bar)", "(b) VSA Case (1 bar)"],
    ):
        for _, row in df.iterrows():
            vals = [row["alpha_GCMC"], row["alpha_IAST"], row["alpha_BKT"]]
            vals = [v for v in vals if pd.notna(v)]
            if len(vals) < 2:
                continue
            ax.plot(
                [min(vals), max(vals)],
                [row["ypos"], row["ypos"]],
                color="0.85",
                linewidth=1.0,
                zorder=1,
            )

        for key, style in style_map.items():
            sizes = np.where(df["is_benchmark"], 56, 34)
            ax.scatter(
                df[key],
                df["ypos"],
                s=sizes,
                zorder=3,
                edgecolors="none",
                **style,
            )

        ax.set_yticks(df["ypos"])
        ax.set_yticklabels(df["label"], fontsize=9)
        for tick, is_benchmark in zip(ax.get_yticklabels(), df["is_benchmark"]):
            if is_benchmark:
                tick.set_fontweight("bold")
        ax.set_xlabel(r"Selectivity, $\alpha$", fontsize=11)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", color="0.92", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.13, top=0.92, wspace=0.12)
    save_figure(fig, "FigX_selectivity_comparison", fig_dir, formats=("png",))
    plt.close(fig)
    print(f"\nStep 3 complete: dumbbell selectivity comparison figure saved.")


# ===================================================================
# STEP 4: Isotherm Fit Multi-panel Figure (SI)
# ===================================================================

def langmuir(P, K, n_m):
    """Langmuir isotherm: q = n_m * K * P / (1 + K * P)."""
    return n_m * K * P / (1 + K * P)


def step4_isotherm_multipanel(bkt_dir: Path, fig_dir: Path):
    """Generate multi-panel isotherm fit figure for SI."""
    print("\n" + "=" * 70)
    print("STEP 4: Isotherm Fit Multi-panel (SI)")
    print("=" * 70)

    iso_dir = bkt_dir / "isotherm_fits"
    raw_data = pd.read_csv(iso_dir / "pure_component_data_merged.csv")
    fit_params = pd.read_csv(iso_dir / "best_isotherm_fits.csv")

    # Unique MOFs
    mof_list = fit_params["MofName"].unique()
    n_mofs = len(mof_list)
    print(f"  {n_mofs} MOFs to plot")

    # Layout: 7 rows × 3 columns for 21 MOFs
    ncols = 3
    nrows = int(np.ceil(n_mofs / ncols))

    set_publication_style()
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(DOUBLE_COL_INCH, nrows * 1.2),
        squeeze=False,
    )

    # Colors
    ch4_color = "#D62728"   # red
    n2_color = "#1F77B4"    # blue

    # Sort ATC-Cu first
    mof_list_sorted = sorted(mof_list, key=lambda x: 0 if "CoRE-2020" in x else 1)
    for idx, mof_name in enumerate(mof_list_sorted):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        mof_raw = raw_data[raw_data["MofName"] == mof_name]
        mof_fits = fit_params[fit_params["MofName"] == mof_name]

        for gas_name, gas_label, color, marker in [
            ("methane", r"CH$_4$", ch4_color, "o"),
            ("N2", r"N$_2$", n2_color, "s"),
        ]:
            # GCMC data points
            gas_data = mof_raw[mof_raw["GasName"] == gas_name]
            if gas_data.empty:
                continue
            P_data = gas_data["Pressure[bar]"].values
            q_data = gas_data["AbsLoading"].values

            # Fit curve (compute R² for legend label)
            gas_fit = mof_fits[mof_fits["GasName"] == gas_name]
            r2_str = ""
            if not gas_fit.empty:
                K = gas_fit["K"].iloc[0]
                n_m = gas_fit["n_m"].iloc[0]
                r2 = gas_fit["R2"].iloc[0]
                r2_str = f" ($R^2$={r2:.3f})"
                P_fit = np.logspace(np.log10(max(P_data.min(), 0.001)),
                                    np.log10(P_data.max()), 100)
                q_fit = langmuir(P_fit, K, n_m)
                ax.plot(P_fit, q_fit, color=color, linewidth=0.8, zorder=2)

            ax.scatter(
                P_data, q_data,
                color=color, marker=marker, s=12, zorder=3,
                label=f"{gas_label}{r2_str}", edgecolors="none", alpha=0.8,
            )

        ax.set_xscale("log")
        ax.set_title(
            simplify_mof_id(mof_name),
            fontsize=7.5,
            pad=2,
            fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=7.0)

        if col == 0:
            ax.set_ylabel("q (mol/kg)", fontsize=8.0)
        if row == nrows - 1:
            ax.set_xlabel("P (bar)", fontsize=8.0)

        ax.legend(fontsize=6.0, loc="upper left", framealpha=0.8, ncol=1)

    # Hide empty subplots
    for idx in range(n_mofs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    save_figure(fig, "FigSX_isotherm_fits", fig_dir, formats=("png",))
    plt.close(fig)
    print(f"\nStep 4 complete: isotherm multi-panel figure saved.")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Task 3.3: Generate BKT paper figures and tables."
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model results dir (default: results/alignn/model_ep150).",
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which step to run (default: all).",
    )
    parser.add_argument(
        "--fig-dir", type=str, default=None,
        help="Output directory for figures (default: manuscript/figures/).",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.model_dir:
        md = Path(args.model_dir)
        if not md.is_absolute():
            md = REPO_ROOT / md
    else:
        md = REPO_ROOT / "results" / "alignn" / "model_ep150"

    bkt_dir = md / "bkt_candidates"

    if args.fig_dir:
        fig_dir = Path(args.fig_dir)
    else:
        fig_dir = get_default_fig_dir(md)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"BKT data: {bkt_dir}")
    print(f"Figures:  {fig_dir}")

    steps = ["1", "2", "3", "4"] if args.step == "all" else [args.step]

    if "1" in steps:
        step1_breakthrough_overlay(bkt_dir, fig_dir)
    if "2" in steps:
        step2_performance_table(bkt_dir, fig_dir)
    if "3" in steps:
        step3_selectivity_comparison(bkt_dir, fig_dir)
    if "4" in steps:
        step4_isotherm_multipanel(bkt_dir, fig_dir)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
