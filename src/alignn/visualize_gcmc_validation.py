"""
visualize_gcmc_validation.py
============================
Publication-quality figures for GCMC validation (Task 2.4b enhanced).

Outputs
-------
results/figures/gcmc_validation/gcmc_parity_4x4.png
    4×4 parity grid: rows 1-2 = PSA Top-100, rows 3-4 = VSA Top-100.
results/figures/gcmc_validation/gcmc_parity_metrics.csv
    Per-property, per-group (PSA/VSA) R², MAE, MAPE, n.
results/figures/gcmc_validation/gcmc_performance_scatter.png
    WC vs selectivity scatter (GCMC data) with ATC-Cu benchmark annotation.
results/figures/gcmc_validation/gcmc_api_kde.png
    KDE: training GCMC distribution vs GCMC top-100, per process (PSA & VSA).
results/figures/gcmc_validation/gcmc_cluster_distribution.png
    Grouped bar chart: MOFs above ATC-Cu benchmark per cluster, PSA & VSA.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as skm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[2]
DATA_FILE   = REPO_ROOT / "results" / "alignn" / "gcmc_top_candidates" / "gcmc_vs_ml_comparison.csv"
UMAP_CSV    = REPO_ROOT / "results" / "cbm_screening" / "inference" / "umap_coordinates_descriptor_with_metrics_ml.csv"
TRAIN_ADS   = REPO_ROOT / "results" / "cbm_screening" / "raspa3_parsed_results_round2_0917.csv"
TRAIN_WIDOM = REPO_ROOT / "results" / "cbm_screening" / "widom_results_round2_0917.csv"
# Round 1 GCMC data — contains ATC-Cu benchmark (not in Round 2)
MOF_HTS_REPO        = Path("/home/zhangsd/repos/MOF-HTS")
TRAINING_ADS_R1_CSV  = (MOF_HTS_REPO / "results" / "cbm_screening"
                         / "gcmc_round1_DreidingTraPPEJson" / "raspa3_parsed_results_0911.csv")
TRAINING_WIDOM_R1_CSV = (MOF_HTS_REPO / "results" / "cbm_screening"
                          / "widom_round1_DREIDING" / "widom_results_0911.csv")
FIG_DIR     = REPO_ROOT / "results" / "figures" / "gcmc_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
NATURE_COLORS = {
    "blue":    "#0173B2",
    "orange":  "#DE8F05",
    "green":   "#029E73",
    "red":     "#CC78BC",
    "cyan":    "#56B4E9",
    "magenta": "#CA9161",
    "yellow":  "#ECE133",
    "purple":  "#949494",
}


def setup_matplotlib() -> None:
    """Configure matplotlib for headless publication-quality output."""
    plt.rcParams["font.family"]        = "sans-serif"
    plt.rcParams["font.sans-serif"]    = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["font.size"]          = 10
    plt.rcParams["axes.labelsize"]     = 11
    plt.rcParams["axes.titlesize"]     = 12
    plt.rcParams["xtick.labelsize"]    = 10
    plt.rcParams["ytick.labelsize"]    = 10
    plt.rcParams["legend.fontsize"]    = 10
    plt.rcParams["figure.titlesize"]   = 12
    plt.rcParams["axes.linewidth"]     = 1.0
    plt.rcParams["grid.linewidth"]     = 0.5
    plt.rcParams["lines.linewidth"]    = 1.5
    plt.rcParams["patch.linewidth"]    = 0.5
    plt.rcParams["xtick.major.width"]  = 1.0
    plt.rcParams["ytick.major.width"]  = 1.0
    plt.rcParams["xtick.major.size"]   = 4
    plt.rcParams["ytick.major.size"]   = 4
    plt.rcParams["savefig.dpi"]        = 300
    plt.rcParams["savefig.bbox"]       = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1


def apply_nature_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", width=1.0, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)


def savefig(fig, path: Path, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")
    if close:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load 199 candidates and add cluster labels from UMAP CSV.
    Returns (df_all, df_psa, df_vsa) with `cluster` column (1-indexed).
    """
    df   = pd.read_csv(DATA_FILE)
    umap = pd.read_csv(UMAP_CSV, usecols=["CifId", "cluster", "UMAP1", "UMAP2"])

    df = df.merge(umap, left_on="mof_id", right_on="CifId", how="left")
    df["cluster"] = df["cluster"] + 1  # 0-indexed → 1-indexed

    df_psa = df[df["psa_rank"].notna()].copy()
    df_vsa = df[df["vsa_rank"].notna()].copy()
    return df, df_psa, df_vsa


def load_training_gcmc_api() -> pd.DataFrame:
    """
    Load R2 training GCMC data and compute PSA/VSA API.
    Returns DataFrame with columns [CifId, PSA_API_CH4, VSA_API_CH4].
    ~21,976 MOFs.
    """
    ads   = pd.read_csv(TRAIN_ADS)
    widom = pd.read_csv(TRAIN_WIDOM)
    df    = _create_integrated_dataset(ads, widom)
    df    = _calculate_separation_metrics(df)
    return df[["CifId", "PSA_API_CH4", "VSA_API_CH4"]].dropna(how="all")


def _create_integrated_dataset(ads_df: pd.DataFrame, widom_df: pd.DataFrame) -> pd.DataFrame:
    """Integrate adsorption (RASPA3) + Widom (RASPA2) → wide format per MOF."""
    ads_piv = ads_df.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"],
        values="AbsLoading", aggfunc="first",
    )
    ads_piv.columns = [f"Ads{g}_{p * 100:.0f}kPa" for g, p in ads_piv.columns]
    ads_piv = ads_piv.reset_index()
    ads_piv.rename(
        columns={c: c.replace("methane", "CH4") for c in ads_piv.columns if "methane" in c},
        inplace=True,
    )

    widom_piv = widom_df.pivot_table(
        index="MofName", columns="GasName",
        values="AdsorptionHeat", aggfunc="first",
    )
    widom_piv.columns = [f"Qst{g}" for g in widom_piv.columns]
    widom_piv.rename(columns={"Qstmethane": "QstCH4"}, inplace=True)
    widom_piv = widom_piv.reset_index()

    merged = pd.merge(ads_piv, widom_piv, on="MofName", how="outer")
    merged.rename(columns={"MofName": "CifId"}, inplace=True)
    return merged


def _calculate_separation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PSA/VSA WC, alpha, API. Mirrors exp08 calculate_separation_metrics."""
    result = df.copy()
    for process, ads_p, des_p in [("PSA", "1000kPa", "100kPa"),
                                   ("VSA", "100kPa",  "10kPa")]:
        result[f"{process}_WC_CH4"] = result[f"AdsCH4_{ads_p}"] - result[f"AdsCH4_{des_p}"]
        q_ch4 = result[f"AdsCH4_{ads_p}"]
        q_n2  = result[f"AdsN2_{ads_p}"]
        result[f"{process}_alpha_CH4_N2"] = np.where(
            q_n2 > 1e-10, (q_ch4 / q_n2) * (0.8 / 0.2), np.nan,
        )
        qst_abs = np.abs(result["QstCH4"])
        alpha   = result[f"{process}_alpha_CH4_N2"]
        result[f"{process}_API_CH4"] = np.where(
            (qst_abs > 1e-10) & (alpha > 1e-10),
            ((alpha - 1) * result[f"{process}_WC_CH4"]) / qst_abs,
            np.nan,
        )
    return result


def get_benchmark_row() -> pd.Series:
    """
    Return ATC-Cu benchmark from Round 1 GCMC simulation data.
    This matches exp08's get_benchmark_api() — actual GCMC values, NOT ML predictions.
    """
    ads_r1   = pd.read_csv(TRAINING_ADS_R1_CSV)
    widom_r1 = pd.read_csv(TRAINING_WIDOM_R1_CSV)
    int_r1   = _create_integrated_dataset(ads_r1, widom_r1)
    enh_r1   = _calculate_separation_metrics(int_r1)
    brow     = enh_r1[enh_r1["CifId"] == BENCHMARK_MOF]
    if brow.empty:
        raise ValueError(f"Benchmark MOF '{BENCHMARK_MOF}' not found in training R1 GCMC data.")
    return brow.iloc[0]


# ---------------------------------------------------------------------------
# Figure 1: Parity scatter 4×4 (PSA rows 0-1, VSA rows 2-3)
# ---------------------------------------------------------------------------

# Column pairs: (ML col, GCMC col, short label)
PARITY_TARGETS = [
    ("AdsCH4_1000kPa", "gcmc_AdsCH4_1000kPa", r"$n_{CH_4}$@1000 kPa"),
    ("AdsCH4_100kPa",  "gcmc_AdsCH4_100kPa",  r"$n_{CH_4}$@100 kPa"),
    ("AdsCH4_10kPa",   "gcmc_AdsCH4_10kPa",   r"$n_{CH_4}$@10 kPa"),
    ("QstCH4",         "QstCH4_gcmc",          r"$Q_{st,CH_4}$"),
    ("AdsN2_1000kPa",  "gcmc_AdsN2_1000kPa",  r"$n_{N_2}$@1000 kPa"),
    ("AdsN2_100kPa",   "gcmc_AdsN2_100kPa",   r"$n_{N_2}$@100 kPa"),
    ("AdsN2_10kPa",    "gcmc_AdsN2_10kPa",    r"$n_{N_2}$@10 kPa"),
    ("QstN2",          "QstN2_gcmc",           r"$Q_{st,N_2}$"),
]


def _draw_parity_panel(
    ax,
    df_sub: pd.DataFrame,
    ml_col: str,
    gcmc_col: str,
    subplot_label: str,
    prop_label: str,
    color: str,
) -> dict:
    """Draw one parity subplot; return metrics dict."""
    mask   = df_sub[gcmc_col].notna() & df_sub[ml_col].notna()
    y_true = df_sub.loc[mask, gcmc_col].values
    y_pred = df_sub.loc[mask, ml_col].values

    ax.scatter(
        y_true, y_pred,
        alpha=0.6, s=30, color=color,
        edgecolors="black", linewidth=0.3,
    )
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, alpha=0.8)

    ax.set_title(f"{subplot_label} {prop_label}", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("GCMC (ground truth)", fontsize=10, fontweight="bold")
    ax.set_ylabel("ML (predicted)",       fontsize=10, fontweight="bold")

    r2   = skm.r2_score(y_true, y_pred)
    mae  = skm.mean_absolute_error(y_true, y_pred)
    mape = skm.mean_absolute_percentage_error(y_true, y_pred)
    textstr = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nMAPE = {mape:.3f}\nn = {len(y_true)}"
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                  edgecolor="black", linewidth=1.0),
        family="monospace",
    )
    apply_nature_axes(ax)
    return {"R2": r2, "MAE": mae, "MAPE": mape, "n": len(y_true)}


def plot_parity_4x4(df_psa: pd.DataFrame, df_vsa: pd.DataFrame) -> pd.DataFrame:
    """
    4×4 parity figure:
      rows 0-1 = PSA Top-100 (8 targets)
      rows 2-3 = VSA Top-100 (8 targets)
    Also returns a metrics DataFrame (16 rows).
    """
    subplot_labels = [f"({chr(ord('a') + i)})" for i in range(16)]
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    fig.suptitle(
        "GCMC Simulation vs ML Predictions — PSA Top-100 (rows a–h) & VSA Top-100 (rows i–p)",
        fontsize=13, fontweight="bold", y=1.00,
    )

    records = []
    for row_off, (df_sub, process, color) in enumerate([
        (df_psa, "PSA", NATURE_COLORS["blue"]),
        (df_vsa, "VSA", NATURE_COLORS["orange"]),
    ]):
        # 8 targets across 2 rows × 4 cols  (row_off*2, row_off*2+1)
        for col_idx, (ml_col, gcmc_col, prop_label) in enumerate(PARITY_TARGETS):
            row = row_off * 2 + col_idx // 4
            col = col_idx  % 4
            flat_idx = row_off * 8 + col_idx
            ax = axes[row, col]
            metrics = _draw_parity_panel(
                ax, df_sub,
                ml_col, gcmc_col,
                subplot_labels[flat_idx],
                prop_label,
                color,
            )
            records.append({"Property": ml_col, "Group": process, **metrics})

        # Row labels on left spine
        label_row = row_off * 2
        axes[label_row, 0].set_ylabel(
            f"{process} Top-100\nML (predicted)",
            fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    savefig(fig, FIG_DIR / "gcmc_parity_4x4.png")

    metrics_df = pd.DataFrame(records)
    csv_path   = FIG_DIR / "gcmc_parity_metrics.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[SAVED] {csv_path}")
    return metrics_df


# ---------------------------------------------------------------------------
# Figure 2: Performance scatter (PSA + VSA, GCMC data, ATC-Cu annotated)
# ---------------------------------------------------------------------------

def plot_performance_scatter(
    df_psa: pd.DataFrame,
    df_vsa: pd.DataFrame,
    benchmark: pd.Series,
) -> None:
    """
    1×2 WC vs selectivity scatter coloured by GCMC API, with ATC-Cu annotation.
    Uses ML-predicted metrics from UMAP CSV for ATC-Cu (no GCMC available).
    """
    datasets = [
        (df_psa, "PSA", "(a) PSA Process (10 bar ↔ 1 bar)",   "PSA", (-110, -60)),
        (df_vsa, "VSA", "(b) VSA Process (1 bar ↔ 0.1 bar)",  "VSA", (-80,  +50)),
    ]

    fig = plt.figure(figsize=(14, 6))

    for idx, (df_data, process_type, title, bm_prefix, text_offset) in enumerate(datasets, 1):
        ax = plt.subplot(1, 2, idx)

        x_col = f"gcmc_{process_type}_WC_CH4"
        y_col = f"gcmc_{process_type}_alpha_CH4_N2"
        c_col = f"gcmc_{process_type}_API_CH4"

        valid = df_data[x_col].notna() & df_data[y_col].notna() & df_data[c_col].notna()
        x = df_data.loc[valid, x_col].values
        y = df_data.loc[valid, y_col].values
        c = df_data.loc[valid, c_col].values

        sc = ax.scatter(
            x, y, c=c,
            cmap="YlOrRd", s=50, alpha=0.6,
            edgecolors="black", linewidths=0.5,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(
            rf"{process_type} API$_{{CH_4}}$ (mol²·kg⁻¹·kJ⁻¹)",
            fontsize=11, fontweight="bold",
        )
        cbar.ax.tick_params(labelsize=10)

        # ATC-Cu benchmark annotation (ML-predicted values from UMAP CSV)
        bx = float(benchmark[f"{bm_prefix}_WC_CH4"])
        by = float(benchmark[f"{bm_prefix}_alpha_CH4_N2"])
        bc = float(benchmark[f"{bm_prefix}_API_CH4"])
        ax.scatter(bx, by, marker="*", s=200, color="black", zorder=6)
        ax.annotate(
            f"ATC-Cu\nWC={bx:.2f}\nα={by:.1f}\nAPI={bc:.3f}",
            xy=(bx, by), xytext=text_offset, textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5",
                      fc=NATURE_COLORS["yellow"],
                      ec="black", alpha=0.8, linewidth=1.5),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.3",
                            color="black", lw=1.5),
            fontsize=11, fontweight="bold",
        )

        if process_type == "PSA":
            xlabel = r"CH$_4$ PSA Working Capacity (mol/kg)"
            ylabel = r"(CH$_4$/N$_2$) PSA Selectivity @ 10 bar"
        else:
            xlabel = r"CH$_4$ VSA Working Capacity (mol/kg)"
            ylabel = r"(CH$_4$/N$_2$) VSA Selectivity @ 1 bar"

        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        ax.tick_params(axis="both", which="major", labelsize=11)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    savefig(fig, FIG_DIR / "gcmc_performance_scatter.png")


# ---------------------------------------------------------------------------
# Figure 3: API KDE — training GCMC vs GCMC top-100 candidates
# ---------------------------------------------------------------------------

def plot_api_kde(
    df_train: pd.DataFrame,
    df_psa: pd.DataFrame,
    df_vsa: pd.DataFrame,
    benchmark: pd.Series,
) -> None:
    """
    1×2 KDE: training GCMC API distribution vs GCMC top-100 PSA/VSA.
    ATC-Cu API indicated by vertical dashed line.
    """
    panels = [
        (df_psa, "PSA", df_train["PSA_API_CH4"], "(a) PSA API$_{CH_4}$ Distribution"),
        (df_vsa, "VSA", df_train["VSA_API_CH4"], "(b) VSA API$_{CH_4}$ Distribution"),
    ]
    fig = plt.figure(figsize=(14, 6))

    for idx, (df_top, process, train_api, title) in enumerate(panels, 1):
        ax = plt.subplot(1, 2, idx)

        gcmc_api_col = f"gcmc_{process}_API_CH4"
        top100_data  = df_top[gcmc_api_col].dropna()
        train_data   = train_api.dropna()

        sns.kdeplot(
            train_data,
            label=f"Training GCMC (n={len(train_data):,})",
            fill=True, alpha=0.4, color=NATURE_COLORS["blue"],
            linewidth=2, ax=ax,
        )
        sns.kdeplot(
            top100_data,
            label=f"GCMC Top-100 {process} (n={len(top100_data)})",
            fill=True, alpha=0.4, color=NATURE_COLORS["orange"],
            linewidth=2, ax=ax,
        )

        ax.axvline(
            train_data.mean(), color=NATURE_COLORS["blue"],
            linestyle="--", linewidth=1.5, alpha=0.8,
            label=f"Training Mean: {train_data.mean():.3f}",
        )
        ax.axvline(
            top100_data.mean(), color=NATURE_COLORS["orange"],
            linestyle="--", linewidth=1.5, alpha=0.8,
            label=f"Top-100 Mean: {top100_data.mean():.3f}",
        )

        # ATC-Cu benchmark vertical line (ML-predicted API)
        bm_api = float(benchmark[f"{process}_API_CH4"])
        ax.axvline(
            bm_api, color="black",
            linestyle=":", linewidth=2.0,
            label=f"ATC-Cu: {bm_api:.3f}",
        )

        ax.set_xlabel(
            rf"CH$_4$ {process} API (mol²·kg⁻¹·kJ⁻¹)",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        ax.tick_params(axis="both", which="major", labelsize=11)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5, axis="y")
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=11, frameon=True,
                  edgecolor="black", fancybox=False, shadow=False)

    fig.tight_layout()
    savefig(fig, FIG_DIR / "gcmc_api_kde.png")


# ---------------------------------------------------------------------------
# Figure 4: Cluster distribution — MOFs above ATC-Cu per cluster
# ---------------------------------------------------------------------------

def plot_cluster_distribution(
    df_psa: pd.DataFrame,
    df_vsa: pd.DataFrame,
    benchmark: pd.Series,
) -> None:
    """
    Grouped bar chart: count of top-PSA/VSA MOFs whose GCMC API exceeds
    the ATC-Cu benchmark (ML-predicted), grouped by structural cluster.
    """
    bm_psa_api = float(benchmark["PSA_API_CH4"])
    bm_vsa_api = float(benchmark["VSA_API_CH4"])

    filtered_psa = df_psa[df_psa["gcmc_PSA_API_CH4"] > bm_psa_api].copy()
    filtered_vsa = df_vsa[df_vsa["gcmc_VSA_API_CH4"] > bm_vsa_api].copy()

    print(f"[INFO] PSA MOFs > ATC-Cu API ({bm_psa_api:.3f}): "
          f"{len(filtered_psa)} / {len(df_psa)}")
    print(f"[INFO] VSA MOFs > ATC-Cu API ({bm_vsa_api:.3f}): "
          f"{len(filtered_vsa)} / {len(df_vsa)}")

    cnt_psa = filtered_psa.groupby("cluster")["mof_id"].count()
    cnt_vsa = filtered_vsa.groupby("cluster")["mof_id"].count()

    all_clusters = sorted(set(cnt_psa.index.tolist()) | set(cnt_vsa.index.tolist()))
    cluster_df   = pd.DataFrame({"Cluster": all_clusters})
    cluster_df["PSA Count"] = cluster_df["Cluster"].map(cnt_psa).fillna(0).astype(int)
    cluster_df["VSA Count"] = cluster_df["Cluster"].map(cnt_vsa).fillna(0).astype(int)

    x     = np.arange(len(cluster_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2, cluster_df["PSA Count"], width,
        label=f"PSA Top-100 (GCMC API > {bm_psa_api:.3f})",
        color=NATURE_COLORS["blue"], edgecolor="black", linewidth=1.0, alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, cluster_df["VSA Count"], width,
        label=f"VSA Top-100 (GCMC API > {bm_vsa_api:.3f})",
        color=NATURE_COLORS["orange"], edgecolor="black", linewidth=1.0, alpha=0.8,
    )

    def _add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, h,
                    f"{int(h)}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                )

    _add_labels(bars1)
    _add_labels(bars2)

    ax.set_xlabel("Structural Cluster", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of MOFs", fontsize=13, fontweight="bold")
    ax.set_title(
        "Cluster Distributions of Top PSA/VSA MOFs Surpassing ATC-Cu Benchmark",
        fontsize=13, fontweight="bold", loc="left",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_df["Cluster"].astype(int), fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=12, frameon=True,
              edgecolor="black", fancybox=False, shadow=False)

    fig.tight_layout()
    savefig(fig, FIG_DIR / "gcmc_cluster_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Loading candidate data …")
    df, df_psa, df_vsa = load_candidates()
    print(f"  Total={len(df)} | PSA={len(df_psa)} | VSA={len(df_vsa)}")

    print(f"[INFO] Loading training GCMC data (R2) …")
    df_train = load_training_gcmc_api()
    print(f"  Training set: {len(df_train)} MOFs | "
          f"PSA_API valid: {df_train['PSA_API_CH4'].notna().sum()} | "
          f"VSA_API valid: {df_train['VSA_API_CH4'].notna().sum()}")

    print(f"[INFO] Loading ATC-Cu benchmark …")
    benchmark = get_benchmark_row()
    print(f"  ATC-Cu PSA_API (ML): {benchmark['PSA_API_CH4']:.4f}  "
          f"VSA_API (ML): {benchmark['VSA_API_CH4']:.4f}")

    setup_matplotlib()

    print("\n[PLOT 1] Parity 4×4 (PSA + VSA) …")
    metrics_df = plot_parity_4x4(df_psa, df_vsa)
    print(metrics_df.to_string(index=False))

    print("\n[PLOT 2] Performance scatter with ATC-Cu …")
    plot_performance_scatter(df_psa, df_vsa, benchmark)

    print("\n[PLOT 3] API KDE (training vs top-100) …")
    plot_api_kde(df_train, df_psa, df_vsa, benchmark)

    print("\n[PLOT 4] Cluster distribution …")
    plot_cluster_distribution(df_psa, df_vsa, benchmark)

    print("\n[DONE] All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
