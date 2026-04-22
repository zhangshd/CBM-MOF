"""
Generate validated screening-result figures for the manuscript.

Figure 10:
  - PSA validated top-100 selectivity vs working-capacity map
  - VSA validated top-100 selectivity vs working-capacity map
  - PSA API enrichment (training set vs validated top-100)
  - VSA API enrichment (training set vs validated top-100)

Figure 11:
  - PSA benchmark-beating cluster enrichment
  - VSA benchmark-beating cluster enrichment
  - PSA recurring ARC building-block identifiers
  - VSA recurring ARC building-block identifiers
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from figures.style import (  # noqa: E402
    compute_panel_grid_layout,
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    save_figure,
    set_publication_style,
)

BENCHMARK_MOF = "CoRE-2020[Cu][pts]3[ASR]1"
# New 186-MOF dual-track validation data
GCMC_VALIDATION_CSV = REPO_ROOT / "results" / "alignn" / "model_ep150" / "process_candidates" / "gcmc_vs_ml_comparison.csv"
CLUSTER_CSV = REPO_ROOT / "results" / "cbm_screening" / "inference" / "umap_coordinates_descriptor_with_metrics_ml.csv"
# PSA/VSA splits from dual-track selection
_MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
PSA_SPLIT_CSVS = [
    _MODEL_DIR / "top_candidates" / "exp_top50_psa.csv",
    _MODEL_DIR / "top_candidates" / "hypo_top50_psa.csv",
]
VSA_SPLIT_CSVS = [
    _MODEL_DIR / "top_candidates" / "exp_top50_vsa.csv",
    _MODEL_DIR / "top_candidates" / "hypo_top50_vsa.csv",
]
TRAINING_ADS_R1_CSV = REPO_ROOT / "results" / "cbm_screening" / "gcmc_round1_DreidingTraPPEJson" / "raspa3_parsed_results_0911.csv"
TRAINING_WIDOM_R1_CSV = REPO_ROOT / "results" / "cbm_screening" / "widom_round1_DREIDING" / "widom_results_0911.csv"
TRAINING_ADS_R2_CSV = REPO_ROOT / "results" / "cbm_screening" / "raspa3_parsed_results_round2_0917.csv"
TRAINING_WIDOM_R2_CSV = REPO_ROOT / "results" / "cbm_screening" / "widom_results_round2_0917.csv"
FIGURE10_HEIGHT_RATIO = 0.88
FIGURE10_W_PAD = 0.45


def _create_integrated_dataset(ads_df: pd.DataFrame, widom_df: pd.DataFrame) -> pd.DataFrame:
    ads_piv = ads_df.pivot_table(
        index="MofName", columns=["GasName", "Pressure[bar]"], values="AbsLoading", aggfunc="first"
    )
    ads_piv.columns = [f"Ads{gas}_{int(pressure * 100)}kPa" for gas, pressure in ads_piv.columns]
    ads_piv = ads_piv.reset_index().rename(columns={"MofName": "mof_id"})
    ads_piv.rename(columns={c: c.replace("methane", "CH4") for c in ads_piv.columns if "methane" in c}, inplace=True)

    widom_piv = widom_df.pivot_table(index="MofName", columns="GasName", values="AdsorptionHeat", aggfunc="first")
    widom_piv.columns = [f"Qst{gas}" for gas in widom_piv.columns]
    widom_piv = widom_piv.reset_index().rename(columns={"MofName": "mof_id", "Qstmethane": "QstCH4"})
    return ads_piv.merge(widom_piv, on="mof_id", how="outer")


def _calculate_process_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for process, ads_label, des_label in [("PSA", "1000kPa", "100kPa"), ("VSA", "100kPa", "10kPa")]:
        out[f"{process}_WC_CH4"] = out[f"AdsCH4_{ads_label}"] - out[f"AdsCH4_{des_label}"]
        out[f"{process}_WC_N2"] = out[f"AdsN2_{ads_label}"] - out[f"AdsN2_{des_label}"]
        alpha = (out[f"AdsCH4_{ads_label}"] / out[f"AdsN2_{ads_label}"]) * 4.0
        out[f"{process}_alpha_CH4_N2"] = alpha
        out[f"{process}_API_CH4"] = ((alpha - 1.0) * out[f"{process}_WC_CH4"]) / out["QstCH4"].abs()
    return out

def load_benchmark_reference() -> tuple[pd.Series, pd.Series]:
    """Return benchmark rows from the training and validated datasets.

    The training-set row is used for the lower-panel API-distribution reference,
    while the validated-set row is the canonical source for benchmark-beating
    counts and validated scatter coordinates. This keeps the figure aligned with
    the manuscript and the Top-10 selection pipeline.
    """
    ads = pd.read_csv(TRAINING_ADS_R1_CSV)
    widom = pd.read_csv(TRAINING_WIDOM_R1_CSV)
    training = _calculate_process_metrics(_create_integrated_dataset(ads, widom))
    training_row = training.loc[training["mof_id"] == BENCHMARK_MOF]
    if training_row.empty:
        raise ValueError(f"Benchmark MOF {BENCHMARK_MOF} not found in Round-1 training data.")

    validated = pd.read_csv(GCMC_VALIDATION_CSV)
    validated_row = validated.loc[validated["mof_id"] == BENCHMARK_MOF]
    if validated_row.empty:
        raise ValueError(f"Benchmark MOF {BENCHMARK_MOF} not found in validated GCMC data.")

    return training_row.iloc[0], validated_row.iloc[0]


def load_training_api_distribution() -> pd.DataFrame:
    ads = pd.read_csv(TRAINING_ADS_R2_CSV)
    widom = pd.read_csv(TRAINING_WIDOM_R2_CSV)
    merged = _calculate_process_metrics(_create_integrated_dataset(ads, widom))
    return merged[["mof_id", "PSA_API_CH4", "VSA_API_CH4"]].copy()


def load_validated_top100() -> pd.DataFrame:
    validated = pd.read_csv(GCMC_VALIDATION_CSV)
    cluster = pd.read_csv(CLUSTER_CSV, usecols=["CifId", "cluster"])
    validated = validated.merge(cluster, left_on="mof_id", right_on="CifId", how="left")

    # Add psa_rank / vsa_rank flags from dual-track selection files
    psa_ids = set()
    for p in PSA_SPLIT_CSVS:
        psa_ids.update(pd.read_csv(p, usecols=["mof_id"])["mof_id"].astype(str))
    vsa_ids = set()
    for p in VSA_SPLIT_CSVS:
        vsa_ids.update(pd.read_csv(p, usecols=["mof_id"])["mof_id"].astype(str))

    validated["mof_id"] = validated["mof_id"].astype(str)
    validated["psa_rank"] = validated["mof_id"].apply(lambda x: 1.0 if x in psa_ids else float("nan"))
    validated["vsa_rank"] = validated["mof_id"].apply(lambda x: 1.0 if x in vsa_ids else float("nan"))
    return validated


def _display_cluster_label(cluster_id: float | int) -> str:
    return str(int(cluster_id) + 1)


def _extract_arc_block_tokens(mof_ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    metal_tokens = []
    organic_tokens = []
    for mof_id in mof_ids.dropna():
        metal_tokens.extend(re.findall(r"m(\d+)", str(mof_id)))
        organic_tokens.extend(re.findall(r"o(\d+)", str(mof_id)))
    return pd.Series(metal_tokens, dtype="object"), pd.Series(organic_tokens, dtype="object")


def _apply_axis_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="major", width=0.5, length=3)
    ax.grid(True, linestyle="--", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)


def _format_api_unit() -> str:
    return r"mol$^2$ kg$^{-1}$ kJ$^{-1}$"


def plot_figure10(output_dir: Path) -> dict[str, float]:
    benchmark_train, benchmark_validated = load_benchmark_reference()
    train = load_training_api_distribution()
    validated = load_validated_top100()
    psa = validated[validated["psa_rank"].notna()].copy()
    vsa = validated[validated["vsa_rank"].notna()].copy()

    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_INCH, FIGURE10_HEIGHT_RATIO * DOUBLE_COL_INCH))

    scatter_specs = [
        (axes[0, 0], psa, "PSA", NATURE_COLORS["blue"], "(a) PSA Elites (n=100)", "", r"CH$_4$/N$_2$ selectivity"),
        (axes[0, 1], vsa, "VSA", NATURE_COLORS["orange"], "(b) VSA Elites (n=100)", rf"VSA API ({_format_api_unit()})", ""),
    ]
    enrichment_specs = [
        (axes[1, 0], "PSA", train["PSA_API_CH4"].dropna(), psa["gcmc_PSA_API_CH4"].dropna(), NATURE_COLORS["blue"], "(c) PSA API enrichment", "Density"),
        (axes[1, 1], "VSA", train["VSA_API_CH4"].dropna(), vsa["gcmc_VSA_API_CH4"].dropna(), NATURE_COLORS["orange"], "(d) VSA API enrichment", ""),
    ]

    summary = {}
    for ax, df_sub, process, color, title, cbar_label, ylabel in scatter_specs:
        x_col = f"gcmc_{process}_WC_CH4"
        y_col = f"gcmc_{process}_alpha_CH4_N2"
        c_col = f"gcmc_{process}_API_CH4"
        sc = ax.scatter(
            df_sub[x_col], df_sub[y_col], c=df_sub[c_col], cmap="YlGnBu", s=18,
            edgecolors="black", linewidths=0.25, alpha=0.8,
        )
        b_wc = float(benchmark_validated[f"gcmc_{process}_WC_CH4"])
        b_alpha = float(benchmark_validated[f"gcmc_{process}_alpha_CH4_N2"])
        ax.scatter([b_wc], [b_alpha], marker="*", s=90, color="black", zorder=4)
        ax.annotate("ATC-Cu", (b_wc, b_alpha), xytext=(5, 5), textcoords="offset points", fontsize=7.0)
        ax.set_xlabel(r"CH$_4$ working capacity (mol/kg)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _apply_axis_style(ax)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=7.5)
        summary[f"{process.lower()}_benchmark_api"] = float(benchmark_validated[f"gcmc_{process}_API_CH4"])
        summary[f"{process.lower()}_validated_mean_api"] = float(df_sub[c_col].mean())

    for ax, process, training_api, validated_api, color, title, ylabel in enrichment_specs:
        sns.kdeplot(training_api, ax=ax, color=NATURE_COLORS["purple"], fill=True, alpha=0.25, linewidth=1.0, label=f"Training set (n={len(training_api):,})")
        sns.kdeplot(validated_api, ax=ax, color=color, fill=True, alpha=0.35, linewidth=1.0, label=f"{process} Elites(n={len(validated_api)})")
        ax.axvline(training_api.mean(), color=NATURE_COLORS["purple"], linestyle="--", linewidth=0.8)
        ax.axvline(validated_api.mean(), color=color, linestyle="--", linewidth=0.8)
        benchmark_api = float(benchmark_validated[f"gcmc_{process}_API_CH4"])
        ax.axvline(benchmark_api, color="black", linestyle=":", linewidth=1.0, label=f"ATC-Cu = {benchmark_api:.3f}")
        ax.set_xlabel(rf"API ({_format_api_unit()})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _apply_axis_style(ax)
        # Extend x-axis to avoid legend-line overlap
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] * 1.45)
        ax.legend(loc="upper right", fontsize=6.5, frameon=False)

    fig.tight_layout(w_pad=FIGURE10_W_PAD, h_pad=0.9)
    save_figure(fig, "Figure10_screening_scatter", output_dir, formats=("png",))

    pd.DataFrame([
        {"metric": key, "value": value} for key, value in summary.items()
    ]).to_csv(output_dir / "Figure10_screening_scatter_summary.csv", index=False)
    return summary


def plot_figure11(output_dir: Path) -> pd.DataFrame:
    _benchmark_train, benchmark_validated = load_benchmark_reference()
    validated = load_validated_top100()
    legend_font = compute_panel_grid_layout(1, 2, DOUBLE_COL_INCH).tick_font

    process_specs = [
        ("PSA", "gcmc_PSA_API_CH4", "psa_rank", float(benchmark_validated["gcmc_PSA_API_CH4"]), NATURE_COLORS["blue"]),
        ("VSA", "gcmc_VSA_API_CH4", "vsa_rank", float(benchmark_validated["gcmc_VSA_API_CH4"]), NATURE_COLORS["orange"]),
    ]

    rows = []
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_INCH, 0.45 * DOUBLE_COL_INCH))

    exp_color = NATURE_COLORS["green"]
    hypo_color = NATURE_COLORS["purple"]

    panel_labels = ["(a)", "(b)"]
    for col_idx, (process, api_col, rank_col, threshold, color) in enumerate(process_specs):
        ax = axes[col_idx]
        top = validated[validated[rank_col].notna()].copy()
        top["beat"] = top[api_col] >= threshold
        beaters = top[top["beat"]].copy()

        # Aggregate by cluster and group (exp/hypo)
        summary = (
            beaters.groupby(["cluster", "group"])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(columns=["exp", "hypo"], fill_value=0)
                   .reset_index()
        )
        summary["total"] = summary["exp"] + summary["hypo"]
        summary = summary[summary["total"] > 0].sort_values("total", ascending=False)

        rows.extend(
            {
                "process": process,
                "cluster": int(r.cluster),
                "cluster_display": int(r.cluster) + 1,
                "exp": int(r.exp),
                "hypo": int(r.hypo),
                "total": int(r.total),
                "benchmark_api": threshold,
            }
            for r in summary.itertuples(index=False)
        )

        x = np.arange(len(summary))
        bar_width = 0.7
        # Stacked bars: exp on bottom, hypo on top
        ax.bar(x, summary["exp"], bar_width, color=exp_color, edgecolor="black",
               linewidth=0.4, alpha=0.85, label="Experimental")
        ax.bar(x, summary["hypo"], bar_width, bottom=summary["exp"], color=hypo_color,
               edgecolor="black", linewidth=0.4, alpha=0.85, label="Hypothetical")
        # Count labels on top
        for xi, total in zip(x, summary["total"]):
            ax.text(xi, total, f"{total}", ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels([_display_cluster_label(c) for c in summary["cluster"]])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Benchmark-beating count" if col_idx == 0 else "")
        n_hits = int(summary["total"].sum())
        ax.set_title(f"{panel_labels[col_idx]} {process} (n = {n_hits}, API ≥ {threshold:.3f})")
        _apply_axis_style(ax)
        if col_idx == 0:
            ax.legend(fontsize=legend_font, frameon=False, loc="upper right")

    fig.tight_layout(w_pad=0.45)
    save_figure(fig, "Figure11_cluster_analysis", output_dir, formats=("png",))
    summary_df = pd.DataFrame(rows).sort_values(["process", "total"], ascending=[True, False])
    summary_df.to_csv(output_dir / "Figure11_cluster_analysis_summary.csv", index=False)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "alignn" / "model_ep150" / "figures",
        help="Directory for Figure10/Figure11 outputs.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_figure10(args.output_dir)
    plot_figure11(args.output_dir)
    print(f"Saved Figure 10/11 assets under {args.output_dir}")


if __name__ == "__main__":
    main()
