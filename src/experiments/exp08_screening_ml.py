"""
Exp08 – Screen top performers, benchmark filter, cluster-level selection,
        deduplication, and GCMC validation submission.

Source: src/jupyter/8_screening_ml.ipynb

Workflow (Phase A → G)
-----------------------
A  Load stability CSV, detect precious/rare metals, filter stable cheap MOFs,
   merge UMAP/metrics CSV, apply 6-condition hard filter → df_screened
B  nlargest(100) per process (PSA/VSA), keep 8 columns → df_psa, df_vsa
C  Save top-100 CSVs, flat CIF copy, submit GCMC+Widom SLURM jobs
D  Parse GCMC results → calculate PSA/VSA metrics for top-100 MOFs;
   load training GCMC data for enrichment reference
E  Merge cluster labels (0-indexed → 1-indexed), filter by benchmark ATC-Cu API
F  Build cluster_sum_df, plot cluster distribution
G  Select best-per-cluster, deduplicate PSA∪VSA, copy final CIFs, submit final
   GCMC, plot API enrichment and WC-vs-alpha performance scatter

Outputs (normal mode)
----------------------
results/cbm_screening/inference/top_100_{psa,vsa}_performers_cifs_ml_org/
results/cbm_screening/inference/top_100_{psa,vsa}_performers_cifs_ml_org_filtered/cluster_{id}/
results/cbm_screening/inference/top_100_{psa,vsa}_performers_ml.csv
results/cbm_screening/inference/top_100_{psa,vsa}_performers_ml_org_with_cluster_filtered.csv
results/cbm_screening/inference/selected_best_mofs_per_cluster.csv
results/cbm_screening/inference/final_selected_cifs_ml_org/
results/figures/exp08_cluster_distribution_top100.png
results/figures/exp08_api_enrichment.png
results/figures/exp08_performance_scatter.png

Run
---
python src/experiments/exp08_screening_ml.py
python src/experiments/exp08_screening_ml.py --test
"""
import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    REPO_ROOT,
    NATURE_COLORS,
    add_test_arg,
    resolve_output_dir,
    savefig,
    sbatch_submit,
    setup_matplotlib,
)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOF_HTS_REPO = Path("/home/zhangsd/repos/MOF-HTS")
CIFS_ROOT    = MOF_HTS_REPO / "data" / "processed" / "integrated_cifs"

# Phase A inputs
STABILITY_CSV = MOF_HTS_REPO / "data" / "processed" / "stabilities" / "infer_results_mofsnn.csv"
UMAP_CSV      = (REPO_ROOT / "results" / "cbm_screening" / "inference"
                 / "umap_coordinates_descriptor_with_metrics_ml.csv")

# Phase D inputs – GCMC results for top-100 PSA/VSA
GCMC_PSA_DIR   = REPO_ROOT / "results" / "cbm_screening" / "inference" / "gcmc_top_100_psa_ml_org"
WIDOM_PSA_DIR  = REPO_ROOT / "results" / "cbm_screening" / "inference" / "widom_top_100_psa_ml_org"
GCMC_VSA_DIR   = REPO_ROOT / "results" / "cbm_screening" / "inference" / "gcmc_top_100_vsa_ml_org"
WIDOM_VSA_DIR  = REPO_ROOT / "results" / "cbm_screening" / "inference" / "widom_top_100_vsa_ml_org"

# Phase D inputs – Training GCMC (used as enrichment reference + benchmark source)
TRAINING_ADS_CSV     = REPO_ROOT / "results" / "cbm_screening" / "raspa3_parsed_results_round2_0917.csv"
TRAINING_WIDOM_CSV   = REPO_ROOT / "results" / "cbm_screening" / "widom_results_round2_0917.csv"
TRAINING_ADS_R1_CSV  = (MOF_HTS_REPO / "results" / "cbm_screening"
                         / "gcmc_round1_DreidingTraPPEJson" / "raspa3_parsed_results_0911.csv")
TRAINING_WIDOM_R1_CSV = (MOF_HTS_REPO / "results" / "cbm_screening"
                          / "widom_round1_DREIDING" / "widom_results_0911.csv")

BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"
TOP_N         = 100

# Full ML predictions (used for GCMC vs ML parity plots)
ML_PRED_CSV = (REPO_ROOT / "results" / "cbm_screening" / "inference"
               / "all_batches_predictions_with_separation_metrics_ml.csv")


# ---------------------------------------------------------------------------
# Precious / rare metal detection  (src: notebook Cell 1, lines 6-92)
# ---------------------------------------------------------------------------

def detect_precious_rare_metals_in_cif(cif_file_path) -> bool:
    """Return True if the CIF contains any precious/rare metals."""
    precious_rare_metals = {
        'Am', 'Au', 'Ag', 'Dy', 'Eu', 'Ga', 'Gd', 'Hf', 'In', 'Ir', 'La', 'Mo', 'Nd',
        'Pd', 'Pr', 'Pt', 'Rh', 'Ru', 'Se', 'Sm', 'Tb', 'Te', 'Tm', 'U', 'Y',
        'Be', 'Bi', 'Cs', 'Er', 'Ho', 'Lu', 'Nb', 'Os', 'Re', 'Sb', 'Ta', 'Th',
        'Tl', 'W', 'Yb', 'Hg',
    }
    cif_path = Path(cif_file_path)
    if not cif_path.exists():
        print(f"  [WARN] CIF not found: {cif_path}")
        return False
    try:
        content = cif_path.read_text(encoding="utf-8", errors="ignore")
        element_patterns = [
            r'_atom_site_type_symbol\s+(\w+)',
            r'_atom_site_label\s+(\w+)',
            r'_chemical_formula_sum\s+[\'"]([^\'"]+)[\'"]',
            r'_chemical_formula_structural\s+[\'"]([^\'"]+)[\'"]',
            r'^(\w+)\d*\s+\w+\s+[\d\.\-\+]+\s+[\d\.\-\+]+\s+[\d\.\-\+]+',
        ]
        found_elements: set = set()
        for pattern in element_patterns:
            for match in re.findall(pattern, content, re.MULTILINE | re.IGNORECASE):
                if ' ' in match:
                    found_elements.update(re.findall(r'([A-Z][a-z]?)', match))
                else:
                    m = re.match(r'^([A-Z][a-z]?)', match)
                    if m:
                        found_elements.add(m.group(1))
        found_elements.update(re.findall(r'\b([A-Z][a-z]?)\b', content))
        return bool(found_elements & precious_rare_metals)
    except Exception as e:
        print(f"  [WARN] Error reading {cif_path}: {e}")
        return False


def name2flag(mof_name: str) -> bool:
    """Return True if the MOF (looked up by CIF file) contains precious/rare metals."""
    return detect_precious_rare_metals_in_cif(CIFS_ROOT / f"{mof_name}.cif")


def _name2flag_batch(batch: list) -> list:
    """Module-level wrapper so multiprocessing.Pool can pickle it."""
    return [name2flag(n) for n in batch]


# ---------------------------------------------------------------------------
# Phase A – Load & pre-filter
# ---------------------------------------------------------------------------

def load_and_prefilter() -> pd.DataFrame:
    """
    Phase A: stability filter + 6-condition screening filter.

    Returns df_screened (93543 × 14) ready for top-N selection.
    """
    # A1 – Load stability predictions
    df_stable = pd.read_csv(STABILITY_CSV)
    print(f"  Stability CSV shape: {df_stable.shape}")

    # A2 – Detect precious/rare metals (parallel-safe serial loop)
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    mof_names = df_stable["MofName"].tolist()
    # I/O-bound: cap at 32 workers to avoid NFS storm
    n_cores = min(cpu_count(), 32)
    batch_size = max(1, len(mof_names) // (n_cores * 8))
    batches = [mof_names[i:i + batch_size] for i in range(0, len(mof_names), batch_size)]

    print(f"  Detecting precious/rare metals using {n_cores} cores "
          f"({len(batches)} batches × ~{batch_size} MOFs) …", flush=True)
    with Pool(processes=n_cores) as pool:
        results = []
        with tqdm(total=len(batches), desc="  Precious-metal scan",
                  miniters=1, file=sys.stderr) as pbar:
            for batch_result in pool.imap(_name2flag_batch, batches, chunksize=1):
                results.extend(batch_result)
                pbar.update(1)

    df_stable.insert(1, "IfPreciousOrRare", results)

    # Filter: no precious/rare metals, solvent-removal stable, water stable
    df_stable_cheap = df_stable.loc[
        (~df_stable["IfPreciousOrRare"])
        & (df_stable["SSD_pred"] == 1)
        & (df_stable["WS24_water_pred"] == 1)
    ].copy()
    df_stable_cheap.rename(columns={"MofName": "CifId"}, inplace=True)
    print(f"  df_stable_cheap shape: {df_stable_cheap.shape}")

    # A3 – Load UMAP / metrics CSV
    df_ads = pd.read_csv(UMAP_CSV)
    print(f"  UMAP CSV shape: {df_ads.shape}")

    # A4 – Merge & 6-condition hard filter
    df_screened = df_ads.merge(
        df_stable_cheap[["CifId", "SSD_pred", "WS24_water_pred"]],
        on="CifId", how="inner"
    ).copy()
    df_screened = df_screened[
        (df_screened["QstCH4"]           > 10) &
        (df_screened["QstN2"]            >  0) &
        (df_screened["PSA_alpha_CH4_N2"] >  0) &
        (df_screened["PSA_WC_CH4"]       >  0) &
        (df_screened["PSA_API_CH4"]      >  0) &
        (df_screened["VSA_alpha_CH4_N2"] >  0) &
        (df_screened["VSA_WC_CH4"]       >  0) &
        (df_screened["VSA_API_CH4"]      >  0)
    ]
    df_screened.rename(columns={
        "SSD_pred":       "SolventRemovalStability",
        "WS24_water_pred":"WaterStability",
    }, inplace=True)
    print(f"  df_screened shape: {df_screened.shape}")

    return df_screened


# ---------------------------------------------------------------------------
# Phase B – Top-N selection
# ---------------------------------------------------------------------------

def select_top_performers(df: pd.DataFrame, process: str, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Return top *top_n* MOFs by descending API for *process*, keeping 8 columns
    (mirroring notebook cell #VSC-34551818).
    """
    api_col  = f"{process}_API_CH4"
    wc_col   = f"{process}_WC_CH4"
    alpha_col = f"{process}_alpha_CH4_N2"
    keep = ["CifId", "UMAP1", "UMAP2", wc_col, alpha_col, api_col, "QstCH4", "QstN2"]
    return (
        df.nlargest(top_n, api_col)[keep]
          .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CIF copy helpers
# ---------------------------------------------------------------------------

def copy_cifs_to_dir(mof_ids: list, src_root: Path, dst_dir: Path) -> int:
    """Copy CIF files flat into *dst_dir*. Returns copy count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for mof_id in mof_ids:
        src = src_root / f"{mof_id}.cif"
        if not src.exists():
            print(f"  [WARN] CIF not found: {src}")
            continue
        shutil.copy(src, dst_dir / f"{mof_id}.cif")
        n_copied += 1
    return n_copied


def copy_cifs_by_cluster(df: pd.DataFrame, src_root: Path, out_base: Path) -> None:
    """
    Copy CIFs into per-cluster subdirectories.
    Expects df to have 'CifId' and 'cluster' columns (1-indexed).
    """
    for cid, grp in df.groupby("cluster"):
        cluster_dir = out_base / f"cluster_{int(cid)}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for mof_name in grp["CifId"]:
            src = src_root / f"{mof_name}.cif"
            if src.exists():
                shutil.copy(src, cluster_dir / f"{mof_name}.cif")
            else:
                print(f"  [WARN] CIF not found: {src}")


# ---------------------------------------------------------------------------
# Phase D helpers – GCMC integration & metrics
# ---------------------------------------------------------------------------

def create_integrated_dataset(ads_df: pd.DataFrame, widom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate adsorption (RASPA3) and Widom (RASPA2) data.
    Returns DataFrame with 8 label columns per MOF.
    (src: notebook cell #VSC-bdb3cd66)
    """
    # Adsorption pivot (mixture data)
    ads_pivot = ads_df.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"],
        values="AbsLoading", aggfunc="first",
    )
    ads_pivot.columns = [
        f"Ads{gas}_{pressure * 100:.0f}kPa"
        for gas, pressure in ads_pivot.columns
    ]
    ads_pivot = ads_pivot.reset_index()

    # Widom pivot (Qst)
    widom_pivot = widom_df.pivot_table(
        index="MofName", columns="GasName",
        values="AdsorptionHeat", aggfunc="first",
    )
    widom_pivot.columns = [f"Qst{gas}" for gas in widom_pivot.columns]
    widom_pivot = widom_pivot.reset_index()

    integrated = pd.merge(ads_pivot, widom_pivot, on="MofName", how="outer")
    integrated.rename(columns={"MofName": "CifId"}, inplace=True)
    integrated.rename(
        columns={c: c.replace("methane", "CH4") for c in integrated.columns if "methane" in c},
        inplace=True,
    )
    return integrated


def calculate_separation_metrics(
    df: pd.DataFrame,
    y_ch4: float = 0.2,
    y_n2:  float = 0.8,
    A: float = 1,
    B: float = 1,
    C: float = 1,
) -> pd.DataFrame:
    """
    Compute PSA and VSA working capacity, selectivity, and API.

    PSA : 10 bar adsorption (1000 kPa) ↔ 1 bar desorption (100 kPa)
    VSA :  1 bar adsorption (100 kPa)  ↔ 0.1 bar desorption (10 kPa)

    API = ((alpha - 1)^A * WC^B) / |Qst|^C
    (src: notebook cell #VSC-ccc2507e)
    """
    result = df.copy()

    for process, ads_p, des_p in [("PSA", "1000kPa", "100kPa"),
                                   ("VSA",  "100kPa",  "10kPa")]:
        result[f"{process}_WC_CH4"] = (result[f"AdsCH4_{ads_p}"]
                                        - result[f"AdsCH4_{des_p}"])
        result[f"{process}_WC_N2"]  = (result[f"AdsN2_{ads_p}"]
                                        - result[f"AdsN2_{des_p}"])

        q_ch4 = result[f"AdsCH4_{ads_p}"]
        q_n2  = result[f"AdsN2_{ads_p}"]
        result[f"{process}_alpha_CH4_N2"] = np.where(
            q_n2 > 1e-10,
            (q_ch4 / q_n2) * (y_n2 / y_ch4),
            np.nan,
        )

        qst_abs = np.abs(result["QstCH4"])
        alpha   = result[f"{process}_alpha_CH4_N2"]
        result[f"{process}_API_CH4"] = np.where(
            (qst_abs > 1e-10) & (alpha > 1e-10),
            ((alpha - 1) ** A * result[f"{process}_WC_CH4"] ** B) / (qst_abs ** C),
            np.nan,
        )

    print(f"PSA Process Metrics Summary:")
    print(f"  Average WC_CH4:       {result['PSA_WC_CH4'].mean():.4f}")
    print(f"  Average alpha_CH4/N2: {result['PSA_alpha_CH4_N2'].mean():.4f}")
    print(f"  Average API_CH4:      {result['PSA_API_CH4'].mean():.4f}")
    print(f"VSA Process Metrics Summary:")
    print(f"  Average WC_CH4:       {result['VSA_WC_CH4'].mean():.4f}")
    print(f"  Average alpha_CH4/N2: {result['VSA_alpha_CH4_N2'].mean():.4f}")
    print(f"  Average API_CH4:      {result['VSA_API_CH4'].mean():.4f}")
    return result


def load_gcmc_results(process: str) -> pd.DataFrame:
    """
    Phase D: parse GCMC + Widom results for the top-100 *process* MOFs.
    Returns enhanced DataFrame (100 × 15) with PSA/VSA metrics.
    """
    gcmc_dir  = GCMC_PSA_DIR  if process.upper() == "PSA" else GCMC_VSA_DIR
    widom_dir = WIDOM_PSA_DIR if process.upper() == "PSA" else WIDOM_VSA_DIR

    ads_csv   = gcmc_dir  / "raspa3_parsed_results.csv"
    widom_csv = widom_dir / "raspa2_parsed_results.csv"

    for p in (ads_csv, widom_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"[Phase D] GCMC result file not found: {p}\n"
                f"  → Run Phase C (GCMC/Widom SLURM jobs) first, "
                f"then re-run this script."
            )

    ads_df   = pd.read_csv(ads_csv)
    widom_df = pd.read_csv(widom_csv)
    integrated = create_integrated_dataset(ads_df, widom_df)
    enhanced   = calculate_separation_metrics(integrated)
    return enhanced


def load_training_gcmc() -> tuple:
    """
    Phase D: load training-set GCMC data (two rounds).
    Returns (enhanced_df_training, enhanced_df_training_r1).
    """
    ads_r2    = pd.read_csv(TRAINING_ADS_CSV)
    widom_r2  = pd.read_csv(TRAINING_WIDOM_CSV)
    ads_r1    = pd.read_csv(TRAINING_ADS_R1_CSV)
    widom_r1  = pd.read_csv(TRAINING_WIDOM_R1_CSV)

    int_r2 = create_integrated_dataset(ads_r2, widom_r2)
    int_r1 = create_integrated_dataset(ads_r1, widom_r1)

    enh_r2 = calculate_separation_metrics(int_r2)
    enh_r1 = calculate_separation_metrics(int_r1)

    return enh_r2, enh_r1


# ---------------------------------------------------------------------------
# Phase E helpers – benchmark filter
# ---------------------------------------------------------------------------

def get_benchmark_api(enhanced_df_training_r1: pd.DataFrame) -> tuple:
    """Return (psa_api, vsa_api) for the benchmark ATC-Cu MOF."""
    bdf = enhanced_df_training_r1[enhanced_df_training_r1["CifId"] == BENCHMARK_MOF]
    if bdf.empty:
        raise ValueError(f"Benchmark MOF '{BENCHMARK_MOF}' not found in training round-1 data.")
    psa_api = float(bdf["PSA_API_CH4"].values[0])
    vsa_api = float(bdf["VSA_API_CH4"].values[0])
    print(f"  Benchmark MOF : {BENCHMARK_MOF}")
    print(f"  Benchmark PSA API : {psa_api:.4f}")
    print(f"  Benchmark VSA API : {vsa_api:.4f}")
    return psa_api, vsa_api


def filter_by_benchmark(
    enhanced_df: pd.DataFrame,
    process: str,
    benchmark_api: float,
    df_ads_cluster: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge cluster labels (from UMAP CSV) onto *enhanced_df*, shift cluster
    0-indexed → 1-indexed, then keep rows with API > benchmark_api.
    """
    api_col = f"{process}_API_CH4"
    merged = enhanced_df.merge(
        df_ads_cluster[["CifId", "cluster", "UMAP1", "UMAP2"]],
        on="CifId", how="inner",
    )
    merged["cluster"] = merged["cluster"] + 1
    filtered = merged[merged[api_col] > benchmark_api].copy()
    return filtered


# ---------------------------------------------------------------------------
# Phase G helpers – cluster selection & deduplication
# ---------------------------------------------------------------------------

def select_best_from_clusters(df: pd.DataFrame, process: str) -> pd.DataFrame:
    """
    Select one MOF per cluster with highest API.
    (src: notebook cell #VSC-adb270da)
    """
    api_col = f"{process}_API_CH4"
    best = df.loc[df.groupby("cluster")[api_col].idxmax()]
    return best.sort_values("cluster").reset_index(drop=True)


def build_final_mofs(best_psa: pd.DataFrame, best_vsa: pd.DataFrame) -> pd.DataFrame:
    """
    Combine best-per-cluster PSA and VSA, deduplicate by CifId keeping the
    row with higher API across processes.
    (src: notebook cell #VSC-14cecadf)
    """
    best_psa_viz = best_psa.copy()
    best_psa_viz["process_type"] = "PSA"
    best_psa_viz.sort_values("PSA_API_CH4", ascending=False, inplace=True)

    best_vsa_viz = best_vsa.copy()
    best_vsa_viz["process_type"] = "VSA"
    best_vsa_viz.sort_values("VSA_API_CH4", ascending=False, inplace=True)

    all_best = pd.concat([best_psa_viz, best_vsa_viz], ignore_index=True)
    duplicates = all_best[all_best.duplicated(subset=["CifId"], keep=False)]

    if len(duplicates) > 0:
        print(f"  Found {len(duplicates) // 2} MOF(s) appearing in both PSA and VSA.")
        unique_mofs = []
        seen: set = set()
        for _, row in all_best.iterrows():
            cif_id = row["CifId"]
            if cif_id in seen:
                continue
            seen.add(cif_id)
            cif_rows = all_best[all_best["CifId"] == cif_id]
            if len(cif_rows) > 1:
                psa_row = cif_rows[cif_rows["process_type"] == "PSA"]
                vsa_row = cif_rows[cif_rows["process_type"] == "VSA"]
                if len(psa_row) > 0 and len(vsa_row) > 0:
                    psa_api = float(psa_row["PSA_API_CH4"].values[0])
                    vsa_api = float(vsa_row["VSA_API_CH4"].values[0])
                    kept = psa_row.iloc[0] if psa_api >= vsa_api else vsa_row.iloc[0]
                    print(f"    {cif_id}: keeping {'PSA' if psa_api >= vsa_api else 'VSA'} "
                          f"(PSA={psa_api:.4f}, VSA={vsa_api:.4f})")
                else:
                    kept = cif_rows.iloc[0]
                unique_mofs.append(kept)
            else:
                unique_mofs.append(row)
        return pd.DataFrame(unique_mofs).reset_index(drop=True)
    else:
        print("  No duplicate MOFs between PSA and VSA.")
        return all_best.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_cluster_distribution(
    filtered_psa: pd.DataFrame,
    filtered_vsa: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    Grouped bar chart: number of top-100 candidates above benchmark per cluster.
    (src: notebook cell #VSC-b34d5ed2)
    """
    import matplotlib.pyplot as plt

    cnt_psa = filtered_psa.groupby("cluster")["CifId"].count()
    cnt_vsa = filtered_vsa.groupby("cluster")["CifId"].count()

    all_clusters = sorted(set(cnt_psa.index.tolist() + cnt_vsa.index.tolist()))
    cluster_df = pd.DataFrame({"Cluster Label": all_clusters})
    cluster_df["PSA Count"] = cluster_df["Cluster Label"].map(cnt_psa).fillna(0).astype(int)
    cluster_df["VSA Count"] = cluster_df["Cluster Label"].map(cnt_vsa).fillna(0).astype(int)
    cluster_df.sort_values("Cluster Label", inplace=True)

    x     = np.arange(len(cluster_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, cluster_df["PSA Count"], width,
                   label="Top Candidates for PSA",
                   color=NATURE_COLORS["blue"], edgecolor="black",
                   linewidth=1.0, alpha=0.8)
    bars2 = ax.bar(x + width / 2, cluster_df["VSA Count"], width,
                   label="Top Candidates for VSA",
                   color=NATURE_COLORS["orange"], edgecolor="black",
                   linewidth=1.0, alpha=0.8)

    def _add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, h,
                        f"{int(h)}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

    _add_labels(bars1)
    _add_labels(bars2)

    ax.set_xlabel("Cluster Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of MOFs", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_df["Cluster Label"].astype(int), fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=13, frameon=True,
              edgecolor="black", fancybox=False, shadow=False)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp08_cluster_distribution_top100.png")


def plot_api_enrichment(
    enhanced_df_training: pd.DataFrame,
    enhanced_df_psa: Optional[pd.DataFrame],
    enhanced_df_vsa: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    KDE: training samples vs top-100 GCMC-computed API enrichment.
    Supports PSA-only, VSA-only, or combined mode.
    (src: notebook cell #VSC-64e9857f)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    panels = []
    if enhanced_df_psa is not None:
        panels.append((enhanced_df_psa, "PSA", "(a) PSA_API_CH4 Distribution"))
    panels.append((
        enhanced_df_vsa,
        "VSA",
        f"({'b' if enhanced_df_psa is not None else 'a'}) VSA_API_CH4 Distribution",
    ))
    n_panels = len(panels)
    fig = plt.figure(figsize=(7 * n_panels, 6))

    for idx, (df_top100, process, title) in enumerate(panels, 1):
        ax = plt.subplot(1, n_panels, idx)
        api_col       = f"{process}_API_CH4"
        training_data = enhanced_df_training[api_col].dropna()
        top100_data   = df_top100[api_col].dropna()

        sns.kdeplot(training_data,
                    label=f"Training Samples (n={len(training_data):,})",
                    fill=True, alpha=0.4, color=NATURE_COLORS["blue"],
                    linewidth=2, ax=ax)
        sns.kdeplot(top100_data,
                    label=f"Top 100 ML-Predicted (n={len(top100_data)})",
                    fill=True, alpha=0.4, color=NATURE_COLORS["orange"],
                    linewidth=2, ax=ax)

        ax.axvline(training_data.mean(), color=NATURE_COLORS["blue"],
                   linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Training Mean: {training_data.mean():.2f}")
        ax.axvline(top100_data.mean(), color=NATURE_COLORS["orange"],
                   linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Top 100 Mean: {top100_data.mean():.2f}")

        ax.set_xlabel(
            rf"CH$_\mathbf{{4}}$ {process} API (mol$^\mathbf{{2}}$kg$^\mathbf{{-1}}$kJ$^\mathbf{{-1}}$)",
            fontsize=13, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(loc="upper right", fontsize=13, frameon=True,
                  edgecolor="black", fancybox=False, shadow=False)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp08_api_enrichment.png")


def plot_performance_scatter(
    enhanced_df_psa: Optional[pd.DataFrame],
    enhanced_df_vsa: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """
    WC vs selectivity scatter coloured by API, with ATC-Cu benchmark annotation.
    Supports PSA-only, VSA-only, or combined mode.
    (src: notebook cell #VSC-04de563c)
    """
    import matplotlib.pyplot as plt

    datasets = []
    if enhanced_df_psa is not None:
        datasets.append((
            pd.concat([enhanced_df_psa, benchmark_df], ignore_index=True),
            "PSA", "(a) PSA Process (10 bar ↔ 1 bar)",
        ))
    datasets.append((
        pd.concat([enhanced_df_vsa, benchmark_df], ignore_index=True),
        "VSA",
        f"({'b' if enhanced_df_psa is not None else 'a'}) VSA Process (1 bar ↔ 0.1 bar)",
    ))
    n_panels = len(datasets)
    fig = plt.figure(figsize=(7 * n_panels, 6))

    for idx, (df_data, process_type, title) in enumerate(datasets, 1):
        ax = plt.subplot(1, n_panels, idx)

        x_data = df_data[f"{process_type}_WC_CH4"]
        y_data = df_data[f"{process_type}_alpha_CH4_N2"]
        c_data = df_data[f"{process_type}_API_CH4"]

        valid = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(c_data)
        scatter = ax.scatter(
            x_data[valid], y_data[valid], c=c_data[valid],
            cmap="YlOrRd", s=50, alpha=0.6,
            edgecolors="black", linewidths=0.5,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(
            rf"{process_type} API (mol$^\mathbf{{2}}$kg$^\mathbf{{-1}}$kJ$^\mathbf{{-1}}$)",
            fontsize=13, fontweight="bold")
        cbar.ax.tick_params(labelsize=10)

        if not benchmark_df.empty:
            bx = float(benchmark_df[f"{process_type}_WC_CH4"].values[0])
            by = float(benchmark_df[f"{process_type}_alpha_CH4_N2"].values[0])
            bc = float(benchmark_df[f"{process_type}_API_CH4"].values[0])
            xytext = (-110, -60) if idx == 1 else (-80, 50)
            ax.annotate(
                f"ATC-Cu\nWC={bx:.2f}\nα={by:.1f}\nAPI={bc:.2f}",
                xy=(bx, by), xytext=xytext, textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5",
                          fc=NATURE_COLORS["yellow"],
                          ec="black", alpha=0.5, linewidth=1.5),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=0.3",
                                color="black", lw=1.5),
                fontsize=13, fontweight="bold",
            )

        ax.set_xlabel(
            rf"CH$_{{\mathbf{{4}}}}$ {process_type} Working Capacity (mol/kg)",
            fontsize=13, fontweight="bold")
        if process_type == "PSA":
            ylabel = rf"(CH$_{{\mathbf{{4}}}}$/N$_{{\mathbf{{2}}}}$) {process_type} Selectivity at 10 bar"
        else:
            ylabel = rf"(CH$_{{\mathbf{{4}}}}$/N$_{{\mathbf{{2}}}}$) {process_type} Selectivity at 1 bar"
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left")

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=12)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp08_performance_scatter.png")


def plot_gcmc_vs_ml_comparison(
    enhanced_df: pd.DataFrame,
    df_ml_all: pd.DataFrame,
    process: str,
    fig_dir: Path,
) -> None:
    """
    Parity scatter: GCMC simulation vs ML-predicted raw adsorption properties.
    2×4 grid, Nature-journal style exactly matching notebook plot_gcmc_vs_predictions_comparison.
    Target order: AdsCH4_1000/100/10kPa, QstCH4, AdsN2_1000/100/10kPa, QstN2.
    figsize=(16, 7) — proportional to notebook's (16,10) for 3 rows → 2 rows.
    (src: notebook 8_screening_ml.ipynb cell 25, plot_scatter_custom)
    """
    import matplotlib.pyplot as plt
    from sklearn import metrics as skm

    # Apply the same rcParams as notebook cell 25 (sans-serif/Arial, size=10 baseline)
    plt.rcParams["font.family"]       = "sans-serif"
    plt.rcParams["font.sans-serif"]   = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["font.size"]         = 10
    plt.rcParams["axes.labelsize"]    = 11
    plt.rcParams["axes.titlesize"]    = 12
    plt.rcParams["xtick.labelsize"]   = 10
    plt.rcParams["ytick.labelsize"]   = 10
    plt.rcParams["legend.fontsize"]   = 10
    plt.rcParams["figure.titlesize"]  = 12
    plt.rcParams["axes.linewidth"]    = 1.0
    plt.rcParams["grid.linewidth"]    = 0.5
    plt.rcParams["lines.linewidth"]   = 1.5
    plt.rcParams["patch.linewidth"]   = 0.5
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.major.size"]  = 4
    plt.rcParams["ytick.major.size"]  = 4

    proc = process.upper()
    # Match notebook target order exactly (AdsCH4 descending pressure, then AdsN2)
    targets = [
        "AdsCH4_1000kPa", "AdsCH4_100kPa", "AdsCH4_10kPa", "QstCH4",
        "AdsN2_1000kPa",  "AdsN2_100kPa",  "AdsN2_10kPa",  "QstN2",
    ]
    subplot_labels = [f"({chr(ord('a') + i)})" for i in range(len(targets))]

    # Filter ML to top-100 CifIds then merge with GCMC results
    top_ids  = set(enhanced_df["CifId"])
    ml_sub   = df_ml_all[df_ml_all["CifId"].isin(top_ids)][["CifId"] + targets].copy()
    gcmc_sub = enhanced_df[["CifId"] + targets].copy()
    merged   = gcmc_sub.merge(ml_sub, on="CifId", suffixes=("_GCMC", "_ML"))

    # 2 rows × 4 cols — notebook uses (16,10) for 3 rows; scale height to 2/3
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat = axes.flatten()

    fig.suptitle(
        f"GCMC Simulation vs Model Predictions - Top 100 {proc} Performers",
        fontsize=14, fontweight="bold", y=0.995,
    )

    for i, target in enumerate(targets):
        ax = axes_flat[i]
        gcmc_col = f"{target}_GCMC"
        ml_col   = f"{target}_ML"

        if gcmc_col not in merged.columns or ml_col not in merged.columns:
            ax.text(0.5, 0.5, f"Column not found:\n{target}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, fontweight="bold")
            ax.set_title(f"{subplot_labels[i]} {target.replace('_', ' ')}",
                         fontsize=12, fontweight="bold", loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        mask   = merged[gcmc_col].notna() & merged[ml_col].notna()
        y_true = merged.loc[mask, gcmc_col].values
        y_pred = merged.loc[mask, ml_col].values

        if len(y_true) == 0:
            ax.text(0.5, 0.5, f"No valid data for\n{target}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, fontweight="bold")
            ax.set_title(f"{subplot_labels[i]} {target.replace('_', ' ')}",
                         fontsize=12, fontweight="bold", loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        # Scatter — notebook plot_scatter_custom: alpha=0.6, s=30, linewidth=0.3
        ax.scatter(y_true, y_pred, alpha=0.6, s=30,
                   color=NATURE_COLORS["blue"], edgecolors="black", linewidth=0.3)

        # 1:1 reference line — notebook: 'r--', linewidth=1.5, alpha=0.8
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                "r--", linewidth=1.5, alpha=0.8, label="1:1 Line")

        # Title — notebook: fontsize=12, fontweight='bold', loc='left'
        ax.set_title(f"{subplot_labels[i]} {target.replace('_', ' ')}",
                     fontsize=12, fontweight="bold", loc="left")

        # Axis labels — notebook: fontsize=11, fontweight='bold'
        ax.set_xlabel("Ground Truth", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted",    fontsize=11, fontweight="bold")

        # Metrics text box — notebook: R², MAE, MAPE, n; fontsize=9, monospace
        r2   = skm.r2_score(y_true, y_pred)
        mae  = skm.mean_absolute_error(y_true, y_pred)
        mape = skm.mean_absolute_percentage_error(y_true, y_pred)
        textstr = (f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\n"
                   f"MAPE = {mape:.3f}\nn = {len(y_true)}")
        props = dict(boxstyle="round", facecolor="white", alpha=0.8,
                     edgecolor="black", linewidth=1.0)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", bbox=props, family="monospace")

        # Spines — notebook: top/right hidden, left/bottom linewidth=1.0
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        # Grid & ticks — notebook: '--', alpha=0.3, lw=0.5; labelsize=10, w=1.0, l=4
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.0, length=4)

    plt.tight_layout()
    savefig(fig, fig_dir / f"exp08_gcmc_vs_predictions_{proc.lower()}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp08: full CBM-MOF screening pipeline (Phase A → G)."
    )
    add_test_arg(parser)
    args = parser.parse_args()

    setup_matplotlib()
    fig_dir       = resolve_output_dir(args.test, "figures")
    screening_dir = resolve_output_dir(args.test, "cbm_screening/inference")

    # ===== Phase A: Load & Pre-filter =====
    print("\n=== Phase A: Load & Pre-filter ===")
    df_screened = load_and_prefilter()
    print(f"  df_screened : {df_screened.shape}")

    # ===== Phase B: Top-100 Selection =====
    print("\n=== Phase B: Top-100 Selection ===")
    df_psa = select_top_performers(df_screened, "PSA")
    df_vsa = select_top_performers(df_screened, "VSA")
    overlap = len(set(df_psa["CifId"]) & set(df_vsa["CifId"]))
    print(f"  PSA top-{TOP_N}: {len(df_psa)}  VSA top-{TOP_N}: {len(df_vsa)}  overlap: {overlap}")

    # Save top-100 CSVs (to results/ always; screening_dir for test output)
    infer_dir = REPO_ROOT / "results" / "cbm_screening" / "inference"
    infer_dir.mkdir(parents=True, exist_ok=True)
    df_psa.to_csv(infer_dir / "top_100_psa_performers_ml_org.csv", index=False)
    df_vsa.to_csv(infer_dir / "top_100_vsa_performers_ml_org.csv", index=False)
    print(f"  Saved top-100 CSVs to {infer_dir}")

    # ===== Phase C: Flat CIF copy + GCMC/Widom submission =====
    print("\n=== Phase C: CIF Copy & GCMC Submission ===")
    for process, df_top in [("psa", df_psa), ("vsa", df_vsa)]:
        cif_dest = infer_dir / f"top_100_{process}_performers_cifs_ml_org"
        if cif_dest.exists():
            shutil.rmtree(cif_dest)
        n = copy_cifs_to_dir(df_top["CifId"].tolist(), CIFS_ROOT, cif_dest)
        print(f"  Flat CIF copy ({process.upper()}): {n}/{len(df_top)} CIFs → {cif_dest.name}/")

    if args.test:
        # In test mode, skip GCMC/Widom submission entirely to avoid
        # creating new batch subdirectories that would corrupt production
        # GCMC result directories (which contain raspa3_parsed_results.csv).
        print("  [TEST MODE] GCMC/Widom submission skipped — using production results in Phase D.")
    else:
        try:
            from gcmc.raspa3_batch_slurm_submitter import main as raspa3_submit
            from gcmc.raspa2_widom_batch_slurm_submitter import main as raspa2_submit

            script_dir             = MOF_HTS_REPO / "src" / "gcmc"
            FORCE_FIELD_DIR        = script_dir / "DreidingTraPPEJson"
            SIMULATION_PARAMS_FILE = str(MOF_HTS_REPO / "examples" / "custom_params"
                                         / "custom_simulation.json")
            FORCE_FIELD_PARAMS_FILE = str(MOF_HTS_REPO / "examples" / "custom_params"
                                          / "custom_force_field.json")
            WIDOM_SIM_PARAMS  = str(MOF_HTS_REPO / "examples" / "custom_params"
                                    / "custom_widom_simulation.json")
            WIDOM_COMP_PARAMS = str(MOF_HTS_REPO / "examples" / "custom_params"
                                    / "custom_widom_component.json")

            gcmc_kwargs = dict(
                TEMPERATURES=[298.0],
                PRESSURES=[1.0e4, 1.0e5, 1.0e6],
                ADSORBATE_COMBINATIONS=[{"molecules": ["methane", "N2"],
                                         "mol_fractions": [0.2, 0.8]}],
                FORCE_FIELD_DIR=FORCE_FIELD_DIR,
                SIMULATION_PARAMS_FILE=SIMULATION_PARAMS_FILE,
                FORCE_FIELD_PARAMS_FILE=FORCE_FIELD_PARAMS_FILE,
                N_CPUS=360, PARTITION="C9654",
                DRY_RUN=False,
            )
            widom_kwargs = dict(
                TEMPERATURES=[298.0], PRESSURES=[0.0],
                WIDOM_MOLECULES=["methane", "N2"],
                FORCE_FIELD_DIR=script_dir / "DREIDING",
                FORCE_FIELD_PARAMS={"shifted_vs_truncated": "truncated",
                                     "tailcorrections": "yes"},
                SIMULATION_PARAMS_FILE=WIDOM_SIM_PARAMS,
                COMPONENT_PARAMS_FILE=WIDOM_COMP_PARAMS,
                N_CPUS=128, PARTITION="C9654",
                DRY_RUN=False,
            )

            for process in ["psa", "vsa"]:
                cif_dir   = infer_dir / f"top_100_{process}_performers_cifs_ml_org"
                gcmc_out  = str(infer_dir / f"gcmc_top_100_{process}_ml_org")
                widom_out = str(infer_dir / f"widom_top_100_{process}_ml_org")
                for out in [gcmc_out, widom_out]:
                    if Path(out).exists():
                        shutil.rmtree(out)
                raspa3_submit(str(cif_dir), gcmc_out,  100, **gcmc_kwargs)
                raspa2_submit(cif_dir,      widom_out, 100, **widom_kwargs)
                print(f"  GCMC/Widom submitted for {process.upper()}")
        except ImportError as e:
            print(f"  [WARN] Could not import GCMC submitters: {e}")
            print("  → Skipping GCMC/Widom submission.")

    # ===== Phase D: Parse GCMC Results =====
    print("\n=== Phase D: Load GCMC Results ===")
    print("  Loading PSA GCMC results …")
    enhanced_df_psa: Optional[pd.DataFrame] = None
    try:
        enhanced_df_psa = load_gcmc_results("PSA")
        print(f"  PSA integrated: {len(enhanced_df_psa)} MOFs")
    except FileNotFoundError as e:
        if args.test:
            print(f"  [TEST MODE] PSA GCMC data unavailable — skipping PSA path.\n    ({e})")
        else:
            raise

    print("  Loading VSA GCMC results …")
    enhanced_df_vsa = load_gcmc_results("VSA")
    print(f"  VSA integrated: {len(enhanced_df_vsa)} MOFs")

    print("  Loading training GCMC data …")
    enhanced_df_training, enhanced_df_training_r1 = load_training_gcmc()

    # ===== Phase E: Merge Cluster Labels + Benchmark Filter =====
    print("\n=== Phase E: Benchmark Filter ===")
    df_ads_cluster = pd.read_csv(UMAP_CSV)[["CifId", "cluster", "UMAP1", "UMAP2"]]

    benchmark_psa_api, benchmark_vsa_api = get_benchmark_api(enhanced_df_training_r1)
    benchmark_df = enhanced_df_training_r1[
        enhanced_df_training_r1["CifId"] == BENCHMARK_MOF
    ].copy()

    filtered_psa: Optional[pd.DataFrame] = None
    if enhanced_df_psa is not None:
        filtered_psa = filter_by_benchmark(enhanced_df_psa, "PSA", benchmark_psa_api, df_ads_cluster)
        print(f"  PSA before: {len(enhanced_df_psa)}, after benchmark filter: {len(filtered_psa)}")
    else:
        print("  [SKIP] PSA benchmark filter — PSA GCMC data unavailable.")

    filtered_vsa = filter_by_benchmark(enhanced_df_vsa, "VSA", benchmark_vsa_api, df_ads_cluster)
    print(f"  VSA before: {len(enhanced_df_vsa)}, after benchmark filter: {len(filtered_vsa)}")

    # Save filtered CSVs
    if filtered_psa is not None:
        filtered_psa.to_csv(
            infer_dir / "top_100_psa_performers_ml_org_with_cluster_filtered.csv", index=False)
    filtered_vsa.to_csv(
        infer_dir / "top_100_vsa_performers_ml_org_with_cluster_filtered.csv", index=False)

    # Per-cluster CIF copy (filtered set)
    active_pairs = [("vsa", filtered_vsa)]
    if filtered_psa is not None:
        active_pairs.insert(0, ("psa", filtered_psa))
    for process, fdf in active_pairs:
        cluster_cif_dest = infer_dir / f"top_100_{process}_performers_cifs_ml_org_filtered"
        if cluster_cif_dest.exists():
            shutil.rmtree(cluster_cif_dest)
        copy_cifs_by_cluster(fdf, CIFS_ROOT, cluster_cif_dest)
        n_total = sum(
            len(list((cluster_cif_dest / f"cluster_{int(c)}").glob("*.cif")))
            for c in fdf["cluster"].unique()
        )
        print(f"  Per-cluster CIF copy ({process.upper()}): {n_total} CIFs → "
              f"{cluster_cif_dest.name}/")

    # ===== Phase F: Cluster Distribution Figure =====
    print("\n=== Phase F: Cluster Distribution Figure ===")
    _psa_plot_df = filtered_psa if filtered_psa is not None else pd.DataFrame(columns=["cluster", "CifId"])
    plot_cluster_distribution(_psa_plot_df, filtered_vsa, fig_dir)

    # ===== Phase G: Best per Cluster, Final Selection & Figures =====
    print("\n=== Phase G: Best per Cluster & Final Selection ===")
    best_psa: Optional[pd.DataFrame] = None
    if filtered_psa is not None:
        best_psa = select_best_from_clusters(filtered_psa, "PSA")
        print(f"  Best PSA: {best_psa['cluster'].nunique()} clusters, {len(best_psa)} MOFs")
    else:
        print("  [SKIP] PSA cluster selection — PSA GCMC data unavailable.")
    best_vsa = select_best_from_clusters(filtered_vsa, "VSA")
    print(f"  Best VSA: {best_vsa['cluster'].nunique()} clusters, {len(best_vsa)} MOFs")

    if best_psa is None:
        # VSA-only mode: skip dedup and use VSA best only
        final_mofs = best_vsa.copy()
        final_mofs["process_type"] = "VSA"
        print("  [SKIP] PSA/VSA dedup — PSA not available, using VSA only.")
    else:
        final_mofs = build_final_mofs(best_psa, best_vsa)
    print(f"  Final selected MOFs: {len(final_mofs)}")

    # Save per-cluster selection summary
    final_mofs.to_csv(infer_dir / "selected_best_mofs_per_cluster.csv", index=False)

    # Copy final CIFs
    final_cif_dir = infer_dir / "final_selected_cifs_ml_org"
    final_cif_dir.mkdir(parents=True, exist_ok=True)
    n_final = copy_cifs_to_dir(final_mofs["CifId"].tolist(), CIFS_ROOT, final_cif_dir)
    print(f"  Final CIFs copied: {n_final}")

    # Submit final GCMC
    if args.test:
        print("  [TEST MODE] Final GCMC submission skipped.")
    else:
        try:
            from gcmc.raspa3_batch_slurm_submitter import main as raspa3_submit

            script_dir             = MOF_HTS_REPO / "src" / "gcmc"
            FORCE_FIELD_DIR        = script_dir / "DreidingTraPPEJson"
            SIMULATION_PARAMS_FILE = str(MOF_HTS_REPO / "examples" / "custom_params"
                                         / "custom_simulation_analysis.json")
            FORCE_FIELD_PARAMS_FILE = str(MOF_HTS_REPO / "examples" / "custom_params"
                                          / "custom_force_field.json")
            gcmc_out = str(infer_dir / "gcmc_final_selected_ml_org")
            if Path(gcmc_out).exists():
                shutil.rmtree(gcmc_out)
            raspa3_submit(
                str(final_cif_dir), gcmc_out, 100,
                TEMPERATURES=[298.0],
                PRESSURES=[1.0e4, 1.0e5, 1.0e6],
                ADSORBATE_COMBINATIONS=[{"molecules": ["methane", "N2"],
                                         "mol_fractions": [0.2, 0.8]}],
                FORCE_FIELD_DIR=FORCE_FIELD_DIR,
                SIMULATION_PARAMS_FILE=SIMULATION_PARAMS_FILE,
                FORCE_FIELD_PARAMS_FILE=FORCE_FIELD_PARAMS_FILE,
                N_CPUS=128, PARTITION="C9654",
                DRY_RUN=False,
            )
            print("  Final GCMC submitted.")
        except (ImportError, Exception) as e:
            print(f"  [WARN] Final GCMC submission skipped: {e}")

    # Figures — functions handle None PSA gracefully (single-panel VSA mode)
    plot_api_enrichment(enhanced_df_training, enhanced_df_psa, enhanced_df_vsa, fig_dir)
    plot_performance_scatter(enhanced_df_psa, enhanced_df_vsa, benchmark_df, fig_dir)

    # GCMC vs ML predictions parity plot (per available process)
    if not ML_PRED_CSV.exists():
        print(f"  [SKIP] gcmc_vs_predictions — ML predictions CSV not found: {ML_PRED_CSV.name}")
    else:
        _df_ml_all = pd.read_csv(ML_PRED_CSV)
        for _proc, _enh_df in [("VSA", enhanced_df_vsa), ("PSA", enhanced_df_psa)]:
            if _enh_df is None:
                print(f"  [SKIP] gcmc_vs_predictions {_proc} — GCMC data unavailable.")
                continue
            plot_gcmc_vs_ml_comparison(_enh_df, _df_ml_all, _proc, fig_dir)

    if args.test:
        print("\n[TEST MODE] All outputs in results/test_run/")


if __name__ == "__main__":
    main()
