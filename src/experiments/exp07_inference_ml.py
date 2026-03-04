"""
Exp07 – Full-library ML inference, API/selectivity/working-capacity calculation,
        and cluster-stratified violin / KDE plots.

Source: src/jupyter/7_inference_ml.ipynb

Outputs (normal mode)
----------------------
results/cbm_screening/inference/all_batches_predictions_with_separation_metrics_ml.csv
results/figures/exp07_api_violin.png
results/figures/exp07_feature_violin.png
results/figures/exp07_target_violin.png
results/figures/exp07_feature_kde.png

Run
---
python src/experiments/exp07_inference_ml.py
python src/experiments/exp07_inference_ml.py --test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    REPO_ROOT,
    NATURE_COLORS,
    add_test_arg,
    apply_nature_axes,
    resolve_output_dir,
    savefig,
    setup_matplotlib,
)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ML_PRED_CSV = (
    REPO_ROOT / "results" / "cbm_screening" / "inference"
    / "all_batches_predictions_api_psa_vsa_ads_qst_ml.csv"
)
CLUSTERED_CSV = (
    REPO_ROOT / "data" / "processed" / "textural_screened"
    / "textural_screened_clustered_with_umap.csv"
)
FEAT_CSV = REPO_ROOT / "data" / "processed" / "RAC_and_zeo_features.csv"

BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"   # ATC-Cu reference

# Feature descriptor columns to visualise (matches notebook)
DESCRIPTOR_COLS = [
    "Di", "Dif", "rho", "VPOV", "GPOV", "POAV_vol_frac", "GPOAV",
    "lc-I-2-all", "lc-alpha-1-all", "lc-Z-0-all", "f-lig-S-0",
]

# Target property columns for distribution violin
TARGET_COLS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
    "QstCH4", "QstN2",
    "PSA_WC_CH4", "PSA_alpha_CH4_N2",
    "VSA_WC_CH4", "VSA_alpha_CH4_N2",
]

TOP_N_KDE = 1000   # number of top performers for KDE density plot


# ---------------------------------------------------------------------------
# Separation metrics
# ---------------------------------------------------------------------------

def calculate_separation_metrics(
    df: pd.DataFrame,
    y_ch4: float = 0.2,
    y_n2: float = 0.8,
    A: float = 1.0,
    B: float = 1.0,
    C: float = 1.0,
) -> pd.DataFrame:
    """
    Add PSA / VSA working capacity, selectivity, and API columns to *df*.

    PSA: 10 bar adsorption (1000 kPa), 1 bar desorption (100 kPa)
    VSA:  1 bar adsorption (100 kPa),  0.1 bar desorption  (10 kPa)

    API formula (mirrors notebook exactly):
        alpha = (q_CH4_ads / q_N2_ads) * (y_N2 / y_CH4)
        API   = ((alpha - 1)^A * WC^B) / |QstCH4|^C
    """
    result_df = df.copy()
    qst_ch4_abs = np.abs(result_df["QstCH4"])

    for process, ads_p, des_p in [("PSA", "1000kPa", "100kPa"), ("VSA", "100kPa", "10kPa")]:
        q_ch4_ads = result_df[f"AdsCH4_{ads_p}"]
        q_n2_ads  = result_df[f"AdsN2_{ads_p}"]

        result_df[f"{process}_WC_CH4"] = result_df[f"AdsCH4_{ads_p}"] - result_df[f"AdsCH4_{des_p}"]
        result_df[f"{process}_WC_N2"]  = result_df[f"AdsN2_{ads_p}"]  - result_df[f"AdsN2_{des_p}"]

        # Selectivity at adsorption pressure; NaN when q_N2 ≤ 1e-10
        alpha = np.where(
            q_n2_ads > 1e-10,
            (q_ch4_ads / q_n2_ads) * (y_n2 / y_ch4),
            np.nan,
        )
        result_df[f"{process}_alpha_CH4_N2"] = alpha

        # API = ((alpha - 1)^A * WC^B) / |QstCH4|^C
        # NaN when |Qst| ≤ 1e-10 or alpha ≤ 1e-10 (no clip, matches notebook)
        valid = (qst_ch4_abs > 1e-10) & (result_df[f"{process}_alpha_CH4_N2"] > 1e-10)
        result_df[f"{process}_API_CH4"] = np.where(
            valid,
            ((result_df[f"{process}_alpha_CH4_N2"] - 1) ** A
             * result_df[f"{process}_WC_CH4"] ** B)
            / (qst_ch4_abs ** C),
            np.nan,
        )

    return result_df


# ---------------------------------------------------------------------------
# Figures – helpers
# ---------------------------------------------------------------------------

def _make_colormap(n_colors: int):
    """Return a colormap covering exactly *n_colors* distinct hues."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if n_colors <= 10:
        return plt.cm.tab10
    if n_colors <= 20:
        return plt.cm.tab20
    colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    if n_colors > 20:
        colors += list(plt.cm.Set3(np.linspace(0, 1, min(12, n_colors - 20))))
    if n_colors > 32:
        colors += list(plt.cm.Paired(np.linspace(0, 1, min(12, n_colors - 32))))
    return mcolors.ListedColormap(colors[:n_colors])


def _violin_palette(unique_clusters, cmap):
    """Build an integer-indexed list palette matching the notebook (cmap(i) style)."""
    return [cmap(i) for i in range(len(unique_clusters))]


def _prep_cluster_df(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Return df with 1-based 'Cluster' column for display, keeping *cols*."""
    out = df[["Cluster"] + list(cols)].copy()
    out["Cluster"] = (out["Cluster"] + 1).astype(int)
    return out


# ---------------------------------------------------------------------------
# Figure 1: API violin (PSA + VSA, 2-row layout, with ATC-Cu benchmark)
# ---------------------------------------------------------------------------

def plot_api_violin(enhanced_df: pd.DataFrame, fig_dir: Path) -> None:
    """PSA + VSA API violin plots as a 2-row figure with ATC-Cu reference line.

    Mirrors notebook exactly: figsize=(12,10), 2 rows, integer-indexed cmap,
    alpha=0.6, linewidth=1.5, hue='Cluster', y-axis LaTeX labels.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    unique_clusters = sorted(enhanced_df["Cluster"].dropna().unique())
    n = len(unique_clusters)
    cmap = _make_colormap(n)
    palette = _violin_palette(unique_clusters, cmap)

    # Benchmark row (may be missing if ATC-Cu not in dataset)
    bench_row = enhanced_df[enhanced_df["CifId"] == BENCHMARK_MOF]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for ax, process, label in [
        (ax1, "PSA", "(a) PSA API Distribution across Clusters"),
        (ax2, "VSA", "(b) VSA API Distribution across Clusters"),
    ]:
        api_col = f"{process}_API_CH4"
        plot_df = _prep_cluster_df(enhanced_df, [api_col]).dropna(subset=[api_col])

        sns.violinplot(
            data=plot_df,
            x="Cluster",
            y=api_col,
            hue="Cluster",
            palette=palette,
            alpha=0.6,
            linewidth=1.5,
            inner="box",
            legend=False,
            ax=ax,
        )

        # ATC-Cu benchmark reference line
        if not bench_row.empty and api_col in bench_row.columns:
            bench_val = bench_row[api_col].values[0]
            if not np.isnan(bench_val):
                ax.axhline(
                    y=bench_val,
                    color="red",
                    linestyle="--",
                    linewidth=2.0,
                    label=f"ATC-Cu: {bench_val:.3f} (Predicted)",
                )
                ax.legend(fontsize=11)

        ax.set_xlabel("Cluster", fontweight="bold")
        ax.set_ylabel(
            rf"{process} API (mol$^\mathbf{{2}}$kg$^\mathbf{{-1}}$kJ$^\mathbf{{-1}}$)",
            fontweight="bold",
        )
        ax.set_title(label, fontsize=13, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
        apply_nature_axes(ax)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp07_api_violin.png")


# ---------------------------------------------------------------------------
# Figure 2: Feature descriptor violin (11 descriptors, 2-column grid)
# ---------------------------------------------------------------------------

def plot_feature_violin(
    enhanced_df: pd.DataFrame,
    feat_csv: Path,
    fig_dir: Path,
) -> None:
    """Violin plots for 11 structural descriptors across clusters.

    Loads descriptor columns from *feat_csv* and merges with *enhanced_df*
    on CifId.  Mirrors notebook layout: n_rows x 2 columns, figsize=(16, 4*n_rows).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not feat_csv.exists():
        print(f"[SKIP] Feature CSV not found: {feat_csv}")
        return

    df_raw = pd.read_csv(feat_csv)
    if "name" in df_raw.columns:
        df_raw.rename(columns={"name": "CifId"}, inplace=True)
    avail_cols = [c for c in DESCRIPTOR_COLS if c in df_raw.columns]
    if not avail_cols:
        print("[SKIP] No descriptor columns found in feature CSV.")
        return

    df_feat = df_raw[["CifId"] + avail_cols].merge(
        enhanced_df[["CifId", "Cluster"]], on="CifId", how="inner"
    )

    unique_clusters = sorted(df_feat["Cluster"].dropna().unique())
    n = len(unique_clusters)
    cmap = _make_colormap(n)
    palette = _violin_palette(unique_clusters, cmap)

    n_cols = 2
    n_rows = int(np.ceil(len(avail_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = axes.flatten()

    plot_df = _prep_cluster_df(df_feat, avail_cols).dropna()
    for i, col in enumerate(avail_cols):
        ax = axes_flat[i]
        sns.violinplot(
            data=plot_df,
            x="Cluster",
            y=col,
            hue="Cluster",
            palette=palette,
            alpha=0.6,
            linewidth=1.5,
            inner="box",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Cluster", fontweight="bold")
        ax.set_ylabel(col, fontweight="bold")
        ax.set_title(col, fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
        apply_nature_axes(ax)

    # Hide unused axes
    for j in range(len(avail_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp07_feature_violin.png")


# ---------------------------------------------------------------------------
# Figure 3: Target properties violin (12 properties, 2-column grid)
# ---------------------------------------------------------------------------

def plot_target_violin(enhanced_df: pd.DataFrame, fig_dir: Path) -> None:
    """Violin plots for 12 target properties across clusters.

    Filters to rows where PSA_WC_CH4 > 0, VSA_WC_CH4 > 0,
    PSA_alpha_CH4_N2 > 0, VSA_alpha_CH4_N2 > 0  (mirrors notebook).
    Layout: n_rows x 2 columns, figsize=(16, 4*n_rows).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    avail_cols = [c for c in TARGET_COLS if c in enhanced_df.columns]
    if not avail_cols:
        print("[SKIP] Target property columns not found in enhanced_df.")
        return

    # Filter rows with valid WC + alpha values
    mask = (
        (enhanced_df.get("PSA_WC_CH4", pd.Series(1, index=enhanced_df.index)) > 0)
        & (enhanced_df.get("VSA_WC_CH4", pd.Series(1, index=enhanced_df.index)) > 0)
        & (enhanced_df.get("PSA_alpha_CH4_N2", pd.Series(1, index=enhanced_df.index)) > 0)
        & (enhanced_df.get("VSA_alpha_CH4_N2", pd.Series(1, index=enhanced_df.index)) > 0)
    )
    df_filt = enhanced_df[mask]

    unique_clusters = sorted(df_filt["Cluster"].dropna().unique())
    n = len(unique_clusters)
    cmap = _make_colormap(n)
    palette = _violin_palette(unique_clusters, cmap)

    n_cols = 2
    n_rows = int(np.ceil(len(avail_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes_flat = axes.flatten()

    plot_df = _prep_cluster_df(df_filt, avail_cols).dropna()
    for i, col in enumerate(avail_cols):
        ax = axes_flat[i]
        sns.violinplot(
            data=plot_df,
            x="Cluster",
            y=col,
            hue="Cluster",
            palette=palette,
            alpha=0.6,
            linewidth=1.5,
            inner="box",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Cluster", fontweight="bold")
        ax.set_ylabel(col, fontweight="bold")
        ax.set_title(col, fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
        apply_nature_axes(ax)

    for j in range(len(avail_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp07_target_violin.png")


# ---------------------------------------------------------------------------
# Figure 4: Feature KDE density (All vs Top-1000 PSA vs Top-1000 VSA)
# ---------------------------------------------------------------------------

def plot_feature_kde(
    enhanced_df: pd.DataFrame,
    feat_csv: Path,
    fig_dir: Path,
) -> None:
    """KDE density plots comparing all MOFs vs top-1000 PSA vs top-1000 VSA performers.

    Layout: n_rows x 3 columns, figsize=(12, 3*n_rows).  Colors:
      All Samples   = #D3D3D3
      Top 1000 PSA  = #0173B2
      Top 1000 VSA  = #DE8F05
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not feat_csv.exists():
        print(f"[SKIP] Feature CSV not found: {feat_csv}")
        return

    df_raw = pd.read_csv(feat_csv)
    if "name" in df_raw.columns:
        df_raw.rename(columns={"name": "CifId"}, inplace=True)
    avail_cols = [c for c in DESCRIPTOR_COLS if c in df_raw.columns]
    if not avail_cols:
        print("[SKIP] No descriptor columns found in feature CSV.")
        return

    density_df = enhanced_df.merge(
        df_raw[["CifId"] + avail_cols], on="CifId", how="inner"
    )

    top_psa = density_df.nlargest(TOP_N_KDE, "PSA_API_CH4")
    top_vsa = density_df.nlargest(TOP_N_KDE, "VSA_API_CH4")

    color_all = "#D3D3D3"
    color_psa = "#0173B2"
    color_vsa = "#DE8F05"

    # Full descriptive names for x-axis labels (mirrors notebook)
    feature_names = {
        "Di":            "Di: Largest Included Sphere Diameter (Å)",
        "Dif":           "Dif: Largest Free Sphere Diameter (Å)",
        "rho":           "ρ: Density (g/cm³)",
        "VPOV":          "VPOV: Volumetric Pore Volume (cm³/cm³)",
        "GPOV":          "GPOV: Gravimetric Pore Volume (cm³/g)",
        "POAV_vol_frac": "POAV_vol_frac: Pore Accessible Volume Fraction",
        "GPOAV":         "GPOAV: Gravimetric Pore Accessible Volume (cm³/g)",
        "lc-I-2-all":    "Linker-I-2 (all)",
        "lc-alpha-1-all":"Linker-alpha-1 (all)",
        "lc-Z-0-all":    "Linker-Z-0 (all)",
        "f-lig-S-0":     "f-lig-S-0",
    }

    n_cols = 3
    n_rows = int(np.ceil(len(avail_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes_flat = axes.flatten()

    for i, col in enumerate(avail_cols):
        ax = axes_flat[i]

        all_data = density_df[col].dropna()
        psa_data = top_psa[col].dropna()
        vsa_data = top_vsa[col].dropna()

        # Plot order: All → VSA → PSA (matches notebook)
        if len(all_data) > 1:
            sns.kdeplot(all_data, ax=ax, color=color_all,
                        linewidth=1.5, fill=True, alpha=0.3, label="All Samples")
        if len(vsa_data) > 1:
            sns.kdeplot(vsa_data, ax=ax, color=color_vsa,
                        linewidth=1.5, fill=True, alpha=0.3, label=f"Top {TOP_N_KDE} VSA")
        if len(psa_data) > 1:
            sns.kdeplot(psa_data, ax=ax, color=color_psa,
                        linewidth=1.5, fill=True, alpha=0.3, label=f"Top {TOP_N_KDE} PSA")

        # Axis labels
        feature_label = feature_names.get(col, col)
        ax.set_title(f"({chr(97 + i)}) {col}", fontsize=13, fontweight="bold", loc="left")
        ax.set_xlabel(feature_label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")

        # x-axis range: [data_min*0.95, min(q3_all*2, data_max)]
        data_min = max(0, all_data.min())
        data_max = all_data.max()
        q3_all = all_data.quantile(0.9)
        ax.set_xlim(left=data_min * 0.95, right=min(q3_all * 2, data_max))

        # 10th–90th percentile annotation (lower right corner)
        q1_all = all_data.quantile(0.1)
        q1_psa, q3_psa = psa_data.quantile(0.1), psa_data.quantile(0.9)
        q1_vsa, q3_vsa = vsa_data.quantile(0.1), vsa_data.quantile(0.9)
        annotation_text = (
            f"10th-90th Percentile:\n"
            f"All: [{q1_all:.2f}, {q3_all:.2f}]\n"
            f"PSA: [{q1_psa:.2f}, {q3_psa:.2f}]\n"
            f"VSA: [{q1_vsa:.2f}, {q3_vsa:.2f}]"
        )
        ax.text(
            0.98, 0.98, annotation_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5,
                      edgecolor="gray", linewidth=0.5),
        )

        # Legend: only first subplot, loc='lower right'; others hidden
        if i == 0:
            ax.legend(loc="lower right", frameon=True, fancybox=False,
                      edgecolor="black", framealpha=0.9, fontsize=11)
        else:
            ax.legend().set_visible(False)

        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)
        apply_nature_axes(ax)

    for j in range(len(avail_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp07_feature_kde.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp07: ML inference + API calculation.")
    add_test_arg(parser)
    args = parser.parse_args()

    setup_matplotlib()
    fig_dir = resolve_output_dir(args.test, "figures")

    # Load data
    if not ML_PRED_CSV.exists():
        print(f"[WARN] Prediction CSV not found: {ML_PRED_CSV}")
        return
    if not CLUSTERED_CSV.exists():
        print(f"[WARN] Clustered CSV not found: {CLUSTERED_CSV}")
        return

    merged_df = pd.read_csv(ML_PRED_CSV)
    # Drop pre-computed API columns if present (will recompute)
    drop_cols = [c for c in ["PSA_API_CH4", "VSA_API_CH4"] if c in merged_df.columns]
    merged_df.drop(columns=drop_cols, inplace=True)
    if "MofName" in merged_df.columns:
        merged_df.rename(columns={"MofName": "CifId"}, inplace=True)

    textural_df = pd.read_csv(CLUSTERED_CSV)
    merged_df = pd.merge(merged_df, textural_df, on="CifId", how="inner")
    print(f"Merged DataFrame: {merged_df.shape}")

    # Calculate separation metrics
    enhanced_df = calculate_separation_metrics(merged_df)
    print("Separation metrics computed.")

    # Print top performers
    for process in ["PSA", "VSA"]:
        api_col = f"{process}_API_CH4"
        if api_col in enhanced_df.columns:
            top5 = enhanced_df.nlargest(5, api_col)[["CifId", api_col]]
            print(f"\nTop 5 {process} performers:")
            print(top5.to_string(index=False))

    # Save enriched CSV
    if not args.test:
        out_dir = ML_PRED_CSV.parent
    else:
        out_dir = resolve_output_dir(args.test, "cbm_screening/inference")
    out_csv = out_dir / "all_batches_predictions_with_separation_metrics_ml.csv"
    enhanced_df.to_csv(out_csv, index=False)
    print(f"Saved enriched predictions → {out_csv}")

    # Figures
    plot_api_violin(enhanced_df, fig_dir)
    plot_feature_violin(enhanced_df, FEAT_CSV, fig_dir)
    plot_target_violin(enhanced_df, fig_dir)
    plot_feature_kde(enhanced_df, FEAT_CSV, fig_dir)

    if args.test:
        print("[TEST MODE] All outputs in results/test_run/")


if __name__ == "__main__":
    main()
