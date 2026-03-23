"""Database-analysis figures for the revised ALIGNN manuscript workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_LIBRARY_API_CSV = (
    PROJECT_ROOT
    / "results"
    / "alignn"
    / "model_ep150"
    / "full_library_inference"
    / "full_library_with_api.csv"
)
CLUSTER_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "textural_screened"
    / "textural_screened_clustered_with_umap.csv"
)
FEATURE_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "RAC_and_zeo_features_deduplicated.csv"
)
ATC_CU_ID = "CoRE-2020[Cu][pts]3[ASR]1"
TOP_N = 100

# Top candidate CSVs (exp top-50 + hypo top-50 per process)
TOP_CANDIDATES_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "top_candidates"
EXP_PSA_CSV = TOP_CANDIDATES_DIR / "exp_top50_psa.csv"
HYPO_PSA_CSV = TOP_CANDIDATES_DIR / "hypo_top50_psa.csv"
EXP_VSA_CSV = TOP_CANDIDATES_DIR / "exp_top50_vsa.csv"
HYPO_VSA_CSV = TOP_CANDIDATES_DIR / "hypo_top50_vsa.csv"


def load_top_candidate_ids() -> tuple[set[str], set[str]]:
    """Load the dual-track top-50 exp + top-50 hypo candidate IDs per process."""
    psa_ids = set(pd.read_csv(EXP_PSA_CSV)["mof_id"]) | set(pd.read_csv(HYPO_PSA_CSV)["mof_id"])
    vsa_ids = set(pd.read_csv(EXP_VSA_CSV)["mof_id"]) | set(pd.read_csv(HYPO_VSA_CSV)["mof_id"])
    return psa_ids, vsa_ids

ZEO_FEATURE_COLUMNS = [
    "Di",
    "Df",
    "Dif",
    "rho",
    "VSA",
    "GSA",
    "VPOV",
    "GPOV",
    "POAV_vol_frac",
    "PONAV_vol_frac",
    "GPOAV",
    "GPONAV",
    "POAV",
    "PONAV",
]

FEATURE_LABELS = {
    "Di": "Largest Included Sphere Diameter",
    "Df": "PLD",
    "Dif": "LCD",
    "rho": "Density",
    "VSA": "Volumetric Surface Area",
    "GSA": "Gravimetric Surface Area",
    "VPOV": "Volumetric Pore Volume",
    "GPOV": "Gravimetric Pore Volume",
    "POAV_vol_frac": "Void Fraction",
    "PONAV_vol_frac": "Non-Accessible Void Fraction",
    "GPOAV": "Gravimetric Accessible Pore Volume",
    "GPONAV": "Gravimetric Non-Accessible Pore Volume",
    "POAV": "Pore Accessible Volume",
    "PONAV": "Pore Non-Accessible Volume",
}

FEATURE_TITLES = {
    "rho": "Density",
    "POAV_vol_frac": "VF",
    "VPOV": "VPOV",
    "GSA": "GSA",
    "VSA": "VSA",
    "GPOAV": "GPOAV",
    "GPOV": "GPOV",
    "Di": "LISD",
    "Df": "PLD",
    "Dif": "LCD",
}

FEATURE_UNITS = {
    "rho": r"g/cm$^3$",
    "POAV_vol_frac": "",
    "VPOV": r"cm$^3$/cm$^3$",
    "GSA": r"m$^2$/g",
    "VSA": r"m$^2$/cm$^3$",
    "GPOAV": r"cm$^3$/g",
    "GPOV": r"cm$^3$/g",
    "Di": r"$\AA$",
    "Df": r"$\AA$",
    "Dif": r"$\AA$",
}

FIGURE7_FEATURES = ["rho", "POAV_vol_frac", "GSA", "VSA", "Dif", "Df"]

FIGURE6_API_UNIT = r"mol$^2$ kg$^{-1}$ kJ$^{-1}$"
FIGURE6_PANEL_TITLES = {
    "PSA_API_CH4": f"Predicted PSA API ({FIGURE6_API_UNIT})",
    "VSA_API_CH4": f"Predicted VSA API ({FIGURE6_API_UNIT})",
}

GROUP_LABELS = ["All samples", f"Top {TOP_N} PSA", f"Top {TOP_N} VSA"]
GROUP_COLORS = {
    "All samples": NATURE_COLORS["purple"],
    f"Top {TOP_N} PSA": NATURE_COLORS["blue"],
    f"Top {TOP_N} VSA": NATURE_COLORS["orange"],
}

CLUSTER_PROPERTY_COLUMNS = [
    "PSA_API_CH4",
    "VSA_API_CH4",
    "PSA_WC_CH4",
    "PSA_alpha_CH4_N2",
    "VSA_WC_CH4",
    "VSA_alpha_CH4_N2",
    "QstCH4",
]

CLUSTER_PROPERTY_LABELS = {
    "PSA_API_CH4": r"PSA API",
    "VSA_API_CH4": r"VSA API",
    "PSA_WC_CH4": r"PSA $q_{WC,CH_4}$",
    "PSA_alpha_CH4_N2": r"PSA $\alpha_{CH_4/N_2}$",
    "VSA_WC_CH4": r"VSA $q_{WC,CH_4}$",
    "VSA_alpha_CH4_N2": r"VSA $\alpha_{CH_4/N_2}$",
    "QstCH4": r"$Q_{st,CH_4}$",
}

CLUSTER_PROPERTY_TITLES = {
    "PSA_API_CH4": r"PSA API (mol$^2$ kg$^{-1}$ kJ$^{-1}$)",
    "VSA_API_CH4": r"VSA API (mol$^2$ kg$^{-1}$ kJ$^{-1}$)",
    "PSA_WC_CH4": r"PSA CH$_4$ Working Capacity (mol/kg)",
    "PSA_alpha_CH4_N2": r"PSA CH$_4$/N$_2$ Selectivity",
    "VSA_WC_CH4": r"VSA CH$_4$ Working Capacity (mol/kg)",
    "VSA_alpha_CH4_N2": r"VSA CH$_4$/N$_2$ Selectivity",
    "QstCH4": r"CH$_4$ Heat of Adsorption (kJ/mol)",
}


def build_database_analysis_frame(
    api_csv: str | Path = FULL_LIBRARY_API_CSV,
    cluster_csv: str | Path = CLUSTER_CSV,
    feature_csv: str | Path = FEATURE_CSV,
) -> pd.DataFrame:
    """Merge API predictions, cluster labels, and Zeo++ features."""
    api_df = pd.read_csv(
        api_csv,
        usecols=list(dict.fromkeys(["mof_id", "PSA_API_CH4", "VSA_API_CH4"] + CLUSTER_PROPERTY_COLUMNS)),
    )
    cluster_df = pd.read_csv(cluster_csv, usecols=["CifId", "Cluster"]).rename(
        columns={"CifId": "mof_id"}
    )
    feature_df = pd.read_csv(feature_csv, usecols=["name"] + ZEO_FEATURE_COLUMNS).rename(
        columns={"name": "mof_id"}
    )

    merged = api_df.merge(cluster_df, on="mof_id", how="inner").merge(
        feature_df, on="mof_id", how="inner"
    )
    merged["Cluster"] = merged["Cluster"].astype(int) + 1
    return merged


def compute_cluster_property_summary(
    df: pd.DataFrame,
    metric_columns: list[str] = CLUSTER_PROPERTY_COLUMNS,
) -> pd.DataFrame:
    """Summarize key process metrics by cluster with interval statistics."""
    rows: list[dict[str, float | int | str]] = []
    for metric in metric_columns:
        metric_df = df[["Cluster", metric]].dropna().copy()
        for cluster, sub_df in metric_df.groupby("Cluster"):
            values = sub_df[metric]
            rows.append(
                {
                    "metric": metric,
                    "Cluster": int(cluster),
                    "q10": float(values.quantile(0.10)),
                    "median": float(values.median()),
                    "q90": float(values.quantile(0.90)),
                    "count": int(len(values)),
                }
            )
    summary = pd.DataFrame(rows)
    summary["rank_desc"] = summary.groupby("metric")["median"].rank(
        ascending=False, method="first"
    )
    return summary


def _safe_iqr(values: pd.Series) -> float:
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return max(q3 - q1, 1e-9)


def compute_feature_shift_summary(
    df: pd.DataFrame,
    feature_columns: list[str] = ZEO_FEATURE_COLUMNS,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Rank Zeo++ features by how strongly top candidates shift from the full library."""
    psa_ids, vsa_ids = load_top_candidate_ids()
    top_psa = df[df["mof_id"].isin(psa_ids)]
    top_vsa = df[df["mof_id"].isin(vsa_ids)]

    rows: list[dict[str, float | str]] = []
    for feature in feature_columns:
        all_vals = df[feature].dropna()
        psa_vals = top_psa[feature].dropna()
        vsa_vals = top_vsa[feature].dropna()
        if len(all_vals) < 3 or len(psa_vals) < 3 or len(vsa_vals) < 3:
            continue

        scale = _safe_iqr(all_vals)
        all_median = float(all_vals.median())
        psa_median = float(psa_vals.median())
        vsa_median = float(vsa_vals.median())
        psa_shift = abs(psa_median - all_median) / scale
        vsa_shift = abs(vsa_median - all_median) / scale

        rows.append(
            {
                "feature": feature,
                "display_name": FEATURE_LABELS.get(feature, feature),
                "shift_score": max(psa_shift, vsa_shift),
                "psa_shift_score": psa_shift,
                "vsa_shift_score": vsa_shift,
                "all_q10": float(all_vals.quantile(0.10)),
                "all_median": all_median,
                "all_q90": float(all_vals.quantile(0.90)),
                "all_q5": float(all_vals.quantile(0.05)),
                "all_q95": float(all_vals.quantile(0.95)),
                "all_min": float(all_vals.min()),
                "all_max": float(all_vals.max()),
                "psa_q10": float(psa_vals.quantile(0.10)),
                "psa_median": psa_median,
                "psa_q90": float(psa_vals.quantile(0.90)),
                "psa_min": float(psa_vals.min()),
                "psa_max": float(psa_vals.max()),
                "vsa_q10": float(vsa_vals.quantile(0.10)),
                "vsa_median": vsa_median,
                "vsa_q90": float(vsa_vals.quantile(0.90)),
                "vsa_min": float(vsa_vals.min()),
                "vsa_max": float(vsa_vals.max()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["shift_score", "psa_shift_score", "vsa_shift_score"],
        ascending=False,
    )


def select_shift_features(
    summary_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    min_features: int = 4,
    max_features: int = 6,
    threshold: float = 0.35,
    redundancy_corr_threshold: float = 0.98,
) -> list[str]:
    """Select 4-6 shifted features while skipping near-redundant descriptors."""
    ordered = summary_df.sort_values("shift_score", ascending=False)["feature"].tolist()
    strong = set(summary_df.loc[summary_df["shift_score"] >= threshold, "feature"].tolist())

    def _redundant(candidate: str, chosen: list[str]) -> bool:
        for existing in chosen:
            corr = feature_df[[candidate, existing]].corr().iloc[0, 1]
            if np.isfinite(corr) and abs(float(corr)) >= redundancy_corr_threshold:
                return True
        return False

    selected: list[str] = []
    for feature in ordered:
        if feature not in strong and len(selected) >= min_features:
            break
        if _redundant(feature, selected):
            continue
        selected.append(feature)
        if len(selected) >= max_features:
            break

    if len(selected) < min_features:
        for feature in ordered:
            if feature in selected or _redundant(feature, selected):
                continue
            selected.append(feature)
            if len(selected) >= min_features:
                break

    return selected


def format_range_value(value: float) -> str:
    """Format percentile-range numbers with scale-aware precision."""
    abs_value = abs(float(value))
    if abs_value >= 1000:
        return f"{value:.0f}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def clip_feature_for_kde(
    feature: str,
    all_data: pd.Series,
    psa_data: pd.Series,
    vsa_data: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Clip long-tailed features for cleaner KDE visualization."""
    upper_bound = float(all_data.quantile(0.99))
    return (
        all_data[all_data <= upper_bound],
        psa_data[psa_data <= upper_bound],
        vsa_data[vsa_data <= upper_bound],
    )


def compose_panel_title(panel_label: str, title: str, unit: str = "") -> str:
    """Compose a panel title while keeping units in compact LaTeX form."""
    body = title if not unit else f"{title} ({unit})"
    return f"{panel_label} {body}".strip()


def plot_cluster_api_landscape(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    benchmark_id: str = ATC_CU_ID,
) -> None:
    """Plot cluster-wise API landscapes for PSA and VSA."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_publication_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL_INCH, 0.48 * DOUBLE_COL_INCH),
        sharex=False,
        sharey=False,
    )

    benchmark = df.loc[df["mof_id"] == benchmark_id, ["PSA_API_CH4", "VSA_API_CH4"]]
    benchmark_psa = float(benchmark["PSA_API_CH4"].iloc[0]) if not benchmark.empty else None
    benchmark_vsa = float(benchmark["VSA_API_CH4"].iloc[0]) if not benchmark.empty else None

    for ax, api_col, panel_label, color, benchmark_value in [
        (axes[0], "PSA_API_CH4", "(a)", NATURE_COLORS["blue"], benchmark_psa),
        (axes[1], "VSA_API_CH4", "(b)", NATURE_COLORS["orange"], benchmark_vsa),
    ]:
        plot_df = df[["Cluster", api_col]].dropna().copy()
        plot_df["Cluster"] = plot_df["Cluster"].astype(int)
        order = sorted(plot_df["Cluster"].unique())

        sns.boxenplot(
            data=plot_df,
            x="Cluster",
            y=api_col,
            order=order,
            color=color,
            linewidth=0.6,
            k_depth="trustworthy",
            width=0.7,
            saturation=0.75,
            ax=ax,
        )

        if benchmark_value is not None and np.isfinite(benchmark_value):
            ax.axhline(
                benchmark_value,
                color=NATURE_COLORS["red"],
                linestyle="--",
                linewidth=1.0,
            )

        ax.set_title(
            compose_panel_title(panel_label, FIGURE6_PANEL_TITLES[api_col]),
            loc="left",
            fontweight="bold",
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_figure(fig, "Figure06_database_distribution", output_dir)
    plt.close(fig)


def plot_feature_shift_kde(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_features: list[str],
    output_dir: str | Path,
) -> None:
    """Plot KDE comparisons for selected Zeo++ features with percentile annotations."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_publication_style()
    n_features = len(selected_features)
    ncols = 2 if n_features > 3 else 1
    nrows = int(np.ceil(n_features / ncols))
    layout = compute_panel_grid_layout(
        nrows=nrows,
        ncols=ncols,
        figure_width_inch=DOUBLE_COL_INCH,
        panel_aspect=0.64,
        gap_ratio_x=0.16,
        gap_ratio_y=0.12,
        left_margin_inch=0.48,
        right_margin_inch=0.10,
        bottom_margin_inch=0.34,
        top_margin_inch=0.26,
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(layout.figure_width, layout.figure_height),
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols).flatten()

    psa_ids, vsa_ids = load_top_candidate_ids()
    top_psa = df[df["mof_id"].isin(psa_ids)]
    top_vsa = df[df["mof_id"].isin(vsa_ids)]

    for i, feature in enumerate(selected_features):
        ax = axes[i]
        row = summary_df.loc[summary_df["feature"] == feature].iloc[0]
        title = FEATURE_TITLES.get(feature, row["display_name"])
        unit = FEATURE_UNITS.get(feature, "")
        title_with_unit = f"{title} ({unit})" if unit else title

        all_data = df[feature].dropna()
        psa_data = top_psa[feature].dropna()
        vsa_data = top_vsa[feature].dropna()
        all_data, psa_data, vsa_data = clip_feature_for_kde(
            feature,
            all_data,
            psa_data,
            vsa_data,
        )

        if len(all_data) > 1:
            sns.kdeplot(
                all_data,
                ax=ax,
                color=GROUP_COLORS["All samples"],
                linewidth=1.2,
                fill=True,
                alpha=0.18,
                label="All samples",
            )
        if len(vsa_data) > 1:
            sns.kdeplot(
                vsa_data,
                ax=ax,
                color=GROUP_COLORS[f"Top {TOP_N} VSA"],
                linewidth=1.4,
                fill=True,
                alpha=0.22,
                label=f"Top {TOP_N} VSA",
            )
        if len(psa_data) > 1:
            sns.kdeplot(
                psa_data,
                ax=ax,
                color=GROUP_COLORS[f"Top {TOP_N} PSA"],
                linewidth=1.4,
                fill=True,
                alpha=0.22,
                label=f"Top {TOP_N} PSA",
            )

        annotation_text = (
            f"All: [{format_range_value(row['all_min'])}, "
            f"{format_range_value(row['all_max'])}]\n"
            f"PSA Top-{TOP_N}: [{format_range_value(row['psa_min'])}, "
            f"{format_range_value(row['psa_max'])}]\n"
            f"VSA Top-{TOP_N}: [{format_range_value(row['vsa_min'])}, "
            f"{format_range_value(row['vsa_max'])}]"
        )
        ax.text(
            0.98,
            0.98,
            annotation_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=layout.annotation_font,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#BBBBBB",
                linewidth=0.4,
                alpha=0.90,
            ),
        )

        ax.set_title(
            compose_panel_title(f"({chr(97 + i)})", title, unit),
            loc="left",
            fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("Density" if i % ncols == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.4)

        if i == 0:
            ax.legend(
                loc="best",
                frameon=True,
                fancybox=False,
                edgecolor="#999999",
                framealpha=0.9,
                fontsize=layout.tick_font,
            )
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(False)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    save_figure(fig, "Figure07_feature_shift", output_dir)
    plt.close(fig)


def plot_cluster_property_intervals(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    file_name: str = "FigureS04_cluster_property_intervals",
    benchmark_api: dict[str, float] | None = None,
) -> None:
    """Plot ranked interval summaries for key process metrics by cluster.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`compute_cluster_property_summary`.
    output_dir : str | Path
        Directory to write the figure into.
    file_name : str
        Stem of the output file (no extension).
    benchmark_api : dict[str, float] | None
        Optional mapping of API column name to ATC-Cu benchmark value.
        When provided, a vertical dashed line is drawn on the corresponding
        API panel(s).
    """
    import matplotlib.pyplot as plt

    set_publication_style()
    layout = compute_panel_grid_layout(
        nrows=4,
        ncols=2,
        figure_width_inch=DOUBLE_COL_INCH,
        panel_aspect=0.70,
        gap_ratio_x=0.18,
        gap_ratio_y=0.15,
        bottom_margin_inch=0.34,
        top_margin_inch=0.18,
    )
    fig = plt.figure(figsize=(layout.figure_width, layout.figure_height))
    gs = fig.add_gridspec(4, 2)
    axes = [
        fig.add_subplot(gs[0, 0]),  # (a) PSA API
        fig.add_subplot(gs[0, 1]),  # (b) VSA API
        fig.add_subplot(gs[1, 0]),  # (c) PSA WC
        fig.add_subplot(gs[1, 1]),  # (d) PSA alpha
        fig.add_subplot(gs[2, 0]),  # (e) VSA WC
        fig.add_subplot(gs[2, 1]),  # (f) VSA alpha
        fig.add_subplot(gs[3, :]),  # (g) QstCH4 — full-width
    ]
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )

    default_color = NATURE_COLORS["purple"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    for ax, metric, panel_label in zip(axes, CLUSTER_PROPERTY_COLUMNS, panel_labels):
        panel_df = summary_df.loc[summary_df["metric"] == metric].sort_values(
            "median", ascending=False
        )
        y_positions = np.arange(len(panel_df))
        for y_pos, row in zip(y_positions, panel_df.itertuples(index=False)):
            ax.hlines(
                y=y_pos,
                xmin=row.q10,
                xmax=row.q90,
                color=default_color,
                linewidth=1.0,
                alpha=0.75,
            )
            ax.scatter(
                row.median,
                y_pos,
                s=layout.marker_area * 0.9,
                color=default_color,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
                alpha=0.75,
            )

        # Draw ATC-Cu benchmark line for API panels
        if benchmark_api is not None and metric in benchmark_api:
            bm_val = benchmark_api[metric]
            if np.isfinite(bm_val):
                ax.axvline(
                    bm_val,
                    color=NATURE_COLORS["red"],
                    linestyle="--",
                    linewidth=1.0,
                    zorder=2,
                )

        tick_labels = [str(int(cluster)) for cluster in panel_df["Cluster"]]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(tick_labels)

        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_ylabel("Cluster", fontsize=layout.body_font)
        ax.set_title(
            compose_panel_title(panel_label, CLUSTER_PROPERTY_TITLES[metric]),
            loc="left",
            fontsize=layout.title_font,
            fontweight="bold",
        )
        ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=layout.tick_font)

    save_figure(fig, file_name, output_dir)
    plt.close(fig)


def export_feature_shift_summary(
    summary_df: pd.DataFrame,
    output_csv: str | Path,
) -> None:
    """Export the ranked feature-shift summary."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")


def generate_assets(
    output_dir: str | Path,
    summary_csv: str | Path | None = None,
    cluster_summary_csv: str | Path | None = None,
    *,
    top_n: int = TOP_N,
    api_csv: str | Path = FULL_LIBRARY_API_CSV,
    cluster_csv: str | Path = CLUSTER_CSV,
    feature_csv: str | Path = FEATURE_CSV,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate Figure 7 (feature-shift KDE) and Figure S4 (cluster property intervals)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_database_analysis_frame(
        api_csv=api_csv,
        cluster_csv=cluster_csv,
        feature_csv=feature_csv,
    )
    summary_df = compute_feature_shift_summary(df, top_n=top_n)
    cluster_summary_df = compute_cluster_property_summary(df)
    selected = FIGURE7_FEATURES

    plot_feature_shift_kde(df, summary_df, selected, output_dir)
    plot_cluster_property_intervals(cluster_summary_df, output_dir)

    if summary_csv is None:
        summary_csv = output_dir / "Figure07_feature_shift_summary.csv"
    export_feature_shift_summary(summary_df, summary_csv)
    if cluster_summary_csv is None:
        cluster_summary_csv = output_dir / "FigureS04_cluster_property_summary.csv"
    Path(cluster_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    cluster_summary_df.to_csv(cluster_summary_csv, index=False)
    print(f"  Saved: {cluster_summary_csv}")
    return summary_df, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 6/7 for database analysis.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures"),
        help="Output directory for figure assets.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional CSV path for the ranked feature-shift summary.",
    )
    parser.add_argument(
        "--cluster_summary_csv",
        type=str,
        default=None,
        help="Optional CSV path for the ranked cluster-property summary.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=TOP_N,
        help="Number of top PSA/VSA candidates used in the feature-shift analysis.",
    )
    parser.add_argument(
        "--api_csv",
        type=str,
        default=str(FULL_LIBRARY_API_CSV),
        help="CSV with ALIGNN full-library API predictions.",
    )
    parser.add_argument(
        "--cluster_csv",
        type=str,
        default=str(CLUSTER_CSV),
        help="CSV with cluster labels.",
    )
    parser.add_argument(
        "--feature_csv",
        type=str,
        default=str(FEATURE_CSV),
        help="CSV with Zeo++ features.",
    )
    args = parser.parse_args()

    summary_df, selected = generate_assets(
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        cluster_summary_csv=args.cluster_summary_csv,
        top_n=args.top_n,
        api_csv=args.api_csv,
        cluster_csv=args.cluster_csv,
        feature_csv=args.feature_csv,
    )
    print(f"Selected features: {selected}")
    print(summary_df[["feature", "display_name", "shift_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
