"""
visualize_gcmc_validation.py
============================
Publication-quality figures for GCMC validation (Task 2.4b enhanced).

Outputs
-------
results/figures/gcmc_validation/gcmc_parity_all199.png
    Parity scatter: GCMC vs ML for 8 adsorption properties (all 199 MOFs).
results/figures/gcmc_validation/gcmc_performance_scatter.png
    WC vs selectivity scatter coloured by API (PSA panel + VSA panel).
results/figures/gcmc_validation/gcmc_api_boxplot.png
    GCMC API distribution boxplot by case (PSA-only / VSA-only / Both).

Also prints per-case statistics for updating gcmc_validation_summary.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as skm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "results" / "alignn" / "gcmc_top_candidates" / "gcmc_vs_ml_comparison.csv"
FIG_DIR   = REPO_ROOT / "results" / "figures" / "gcmc_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style constants (mirroring src/experiments/utils.py)
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

CASE_COLORS = {
    "PSA-only": NATURE_COLORS["blue"],
    "VSA-only": NATURE_COLORS["orange"],
    "Both":     NATURE_COLORS["green"],
}


def setup_matplotlib() -> None:
    """Configure matplotlib for headless publication-quality output."""
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
    plt.rcParams["savefig.dpi"]       = 300
    plt.rcParams["savefig.bbox"]      = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1


def apply_nature_axes(ax) -> None:
    """Apply Nature-journal spine/tick style to an Axes."""
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
# Data loading & subset definition
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the comparison CSV and return (df_all, df_psa, df_vsa, df_psa_only, df_vsa_only).
    """
    df = pd.read_csv(DATA_FILE)
    df_psa      = df[df["psa_rank"].notna()].copy()
    df_vsa      = df[df["vsa_rank"].notna()].copy()
    df_psa_only = df[df["psa_rank"].notna() & df["vsa_rank"].isna()].copy()
    df_vsa_only = df[df["vsa_rank"].notna() & df["psa_rank"].isna()].copy()
    return df, df_psa, df_vsa, df_psa_only, df_vsa_only


# ---------------------------------------------------------------------------
# Figure 1: Parity scatter (2×4, all 199 MOFs)
# ---------------------------------------------------------------------------

def plot_parity_all199(df: pd.DataFrame) -> None:
    """
    2×4 parity scatter: GCMC (x) vs ML predicted (y) for 8 adsorption properties.
    All 199 MOFs are shown. Column mapping:
        ML col          GCMC col
        AdsCH4_Xkpa  →  gcmc_AdsCH4_XkPa
        QstCH4       →  QstCH4_gcmc
        QstN2        →  QstN2_gcmc
    """
    targets = [
        ("AdsCH4_1000kPa", "gcmc_AdsCH4_1000kPa"),
        ("AdsCH4_100kPa",  "gcmc_AdsCH4_100kPa"),
        ("AdsCH4_10kPa",   "gcmc_AdsCH4_10kPa"),
        ("QstCH4",         "QstCH4_gcmc"),
        ("AdsN2_1000kPa",  "gcmc_AdsN2_1000kPa"),
        ("AdsN2_100kPa",   "gcmc_AdsN2_100kPa"),
        ("AdsN2_10kPa",    "gcmc_AdsN2_10kPa"),
        ("QstN2",          "QstN2_gcmc"),
    ]

    # Friendly axis titles
    title_map = {
        "AdsCH4_1000kPa": r"$\mathit{n}_{CH_4}$ @ 1000 kPa (mol/kg)",
        "AdsCH4_100kPa":  r"$\mathit{n}_{CH_4}$ @ 100 kPa (mol/kg)",
        "AdsCH4_10kPa":   r"$\mathit{n}_{CH_4}$ @ 10 kPa (mol/kg)",
        "QstCH4":         r"$Q_{st,CH_4}$ (kJ/mol)",
        "AdsN2_1000kPa":  r"$\mathit{n}_{N_2}$ @ 1000 kPa (mol/kg)",
        "AdsN2_100kPa":   r"$\mathit{n}_{N_2}$ @ 100 kPa (mol/kg)",
        "AdsN2_10kPa":    r"$\mathit{n}_{N_2}$ @ 10 kPa (mol/kg)",
        "QstN2":          r"$Q_{st,N_2}$ (kJ/mol)",
    }

    subplot_labels = [f"({chr(ord('a') + i)})" for i in range(8)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat = axes.flatten()

    fig.suptitle(
        "GCMC Simulation vs ML Predictions — All 199 Top Candidates",
        fontsize=14, fontweight="bold", y=0.998,
    )

    for i, (ml_col, gcmc_col) in enumerate(targets):
        ax = axes_flat[i]

        mask   = df[gcmc_col].notna() & df[ml_col].notna()
        y_true = df.loc[mask, gcmc_col].values   # GCMC = ground truth (x-axis)
        y_pred = df.loc[mask, ml_col].values      # ML predicted (y-axis)

        ax.scatter(
            y_true, y_pred,
            alpha=0.6, s=30,
            color=NATURE_COLORS["blue"],
            edgecolors="black", linewidth=0.3,
        )

        # 1:1 reference line
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, alpha=0.8)

        # Axis labels
        label = title_map[ml_col]
        ax.set_xlabel(f"GCMC {label}", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"ML {label}",   fontsize=10, fontweight="bold")
        ax.set_title(
            f"{subplot_labels[i]} {ml_col.replace('_', ' ')}",
            fontsize=12, fontweight="bold", loc="left",
        )

        # Metrics
        r2   = skm.r2_score(y_true, y_pred)
        mae  = skm.mean_absolute_error(y_true, y_pred)
        mape = skm.mean_absolute_percentage_error(y_true, y_pred)
        textstr = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nMAPE = {mape:.3f}\nn = {len(y_true)}"
        props = dict(boxstyle="round", facecolor="white", alpha=0.8,
                     edgecolor="black", linewidth=1.0)
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=props, family="monospace",
        )

        apply_nature_axes(ax)

    plt.tight_layout()
    savefig(fig, FIG_DIR / "gcmc_parity_all199.png")


# ---------------------------------------------------------------------------
# Figure 2: Performance scatter (PSA + VSA, WC vs selectivity)
# ---------------------------------------------------------------------------

def plot_performance_scatter(df_psa: pd.DataFrame, df_vsa: pd.DataFrame) -> None:
    """
    1×2 WC vs selectivity scatter coloured by GCMC API.
    Uses GCMC-derived columns for both axes and colormap.
    """
    datasets = [
        (df_psa, "PSA", "(a) PSA Process (10 bar ↔ 1 bar)"),
        (df_vsa, "VSA", "(b) VSA Process (1 bar ↔ 0.1 bar)"),
    ]
    fig = plt.figure(figsize=(14, 6))

    for idx, (df_data, process_type, title) in enumerate(datasets, 1):
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

        if process_type == "PSA":
            xlabel = r"CH$_4$ PSA Working Capacity (mol/kg)"
            ylabel = r"(CH$_4$/N$_2$) PSA Selectivity at 10 bar"
        else:
            xlabel = r"CH$_4$ VSA Working Capacity (mol/kg)"
            ylabel = r"(CH$_4$/N$_2$) VSA Selectivity at 1 bar"

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
# Figure 3: API distribution boxplot by case
# ---------------------------------------------------------------------------

def plot_api_boxplot(
    df_psa_only: pd.DataFrame,
    df_vsa_only: pd.DataFrame,
    df_both: pd.DataFrame,
) -> None:
    """
    1×2 boxplot + stripplot: GCMC PSA/VSA API grouped by case.
    Groups: PSA-only (99), VSA-only (99), Both (1).
    """
    # Build labelled frames
    psa_only_psa = pd.DataFrame({
        "API": df_psa_only["gcmc_PSA_API_CH4"].dropna().values,
        "Case": "PSA-only",
        "type": "PSA",
    })
    vsa_only_psa = pd.DataFrame({
        "API": df_vsa_only["gcmc_PSA_API_CH4"].dropna().values,
        "Case": "VSA-only",
        "type": "PSA",
    })
    both_psa = pd.DataFrame({
        "API": df_both["gcmc_PSA_API_CH4"].dropna().values,
        "Case": "Both",
        "type": "PSA",
    })

    psa_only_vsa = pd.DataFrame({
        "API": df_psa_only["gcmc_VSA_API_CH4"].dropna().values,
        "Case": "PSA-only",
        "type": "VSA",
    })
    vsa_only_vsa = pd.DataFrame({
        "API": df_vsa_only["gcmc_VSA_API_CH4"].dropna().values,
        "Case": "VSA-only",
        "type": "VSA",
    })
    both_vsa = pd.DataFrame({
        "API": df_both["gcmc_VSA_API_CH4"].dropna().values,
        "Case": "Both",
        "type": "VSA",
    })

    psa_data = pd.concat([psa_only_psa, vsa_only_psa, both_psa], ignore_index=True)
    vsa_data = pd.concat([psa_only_vsa, vsa_only_vsa, both_vsa], ignore_index=True)

    cases  = ["PSA-only", "VSA-only", "Both"]
    colors = [CASE_COLORS[c] for c in cases]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    rng = np.random.default_rng(42)

    for ax, (data, api_type) in zip(axes, [
        (psa_data, "PSA"),
        (vsa_data, "VSA"),
    ]):
        groups   = [data.loc[data["Case"] == c, "API"].values for c in cases]
        n_groups = [len(g) for g in groups]
        positions = range(len(cases))

        # Boxplot
        bp = ax.boxplot(
            groups,
            positions=list(positions),
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="o", markersize=4, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Strip (jitter)
        for j, (group_vals, pos, color) in enumerate(zip(groups, positions, colors)):
            if len(group_vals) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(group_vals))
            ax.scatter(
                np.full(len(group_vals), pos) + jitter,
                group_vals,
                color=color, s=20, alpha=0.7,
                edgecolors="black", linewidths=0.3, zorder=3,
            )
            # Mean diamond
            mean_val = np.mean(group_vals)
            ax.scatter(
                pos, mean_val,
                marker="D", s=60, color="black", zorder=5,
            )

        ax.set_xticks(list(positions))
        ax.set_xticklabels(
            [f"{c}\n(n={n})" for c, n in zip(cases, n_groups)],
            fontsize=11,
        )
        ax.set_ylabel(
            rf"GCMC {api_type} API$_{{CH_4}}$ (mol²·kg⁻¹·kJ⁻¹)",
            fontsize=11, fontweight="bold",
        )
        panel_label = "(a)" if api_type == "PSA" else "(b)"
        ax.set_title(
            f"{panel_label} {api_type} Process — GCMC API by Case",
            fontsize=12, fontweight="bold", loc="left",
        )
        apply_nature_axes(ax)

    fig.tight_layout()
    savefig(fig, FIG_DIR / "gcmc_api_boxplot.png")


# ---------------------------------------------------------------------------
# Per-case statistics report
# ---------------------------------------------------------------------------

def print_case_statistics(
    df: pd.DataFrame,
    df_psa: pd.DataFrame,
    df_vsa: pd.DataFrame,
    df_psa_only: pd.DataFrame,
    df_vsa_only: pd.DataFrame,
    df_both: pd.DataFrame,
) -> None:
    """Print per-case GCMC statistics and Top-10 VSA table for summary update."""

    def fmt_stats(subset: pd.DataFrame, process: str) -> None:
        api_col = f"gcmc_{process}_API_CH4"
        wc_col  = f"gcmc_{process}_WC_CH4"
        sel_col = f"gcmc_{process}_alpha_CH4_N2"
        ml_api  = f"{process}_API_CH4"

        api_vals = subset[api_col].dropna()
        wc_vals  = subset[wc_col].dropna()
        sel_vals = subset[sel_col].dropna()

        # R² between ML API and GCMC API
        merged = subset[[ml_api, api_col]].dropna()
        if len(merged) >= 2:
            r2 = skm.r2_score(merged[api_col].values, merged[ml_api].values)
        else:
            r2 = float("nan")

        label = "PSA 候选（Top-100 by ML PSA_API）" if process == "PSA" else \
                "VSA 候选（Top-100 by ML VSA_API）"
        print(f"\n### {label}")
        print(f"| 指标 | 值 |")
        print(f"|------|-----|")
        print(f"| 候选数量 | {len(subset)} |")
        print(f"| GCMC {process}_API 范围 | {api_vals.min():.4f} ~ {api_vals.max():.4f} |")
        print(f"| GCMC {process}_API 均值 ± std | {api_vals.mean():.4f} ± {api_vals.std():.4f} |")
        print(f"| GCMC {process}_WC_CH4 均值 | {wc_vals.mean():.4f} mol/kg |")
        print(f"| GCMC {process}_alpha_CH4_N2 均值 | {sel_vals.mean():.2f} |")
        print(f"| ML vs GCMC {process}_API R² | {r2:.4f} |")

    print("\n" + "=" * 60)
    print("PER-CASE GCMC VALIDATION STATISTICS")
    print("=" * 60)

    fmt_stats(df_psa, "PSA")
    fmt_stats(df_vsa, "VSA")

    print(f"\n### 重叠分析")
    print(f"| 分组 | 数量 |")
    print(f"|------|------|")
    print(f"| PSA-only (仅入 PSA Top-100) | {len(df_psa_only)} |")
    print(f"| VSA-only (仅入 VSA Top-100) | {len(df_vsa_only)} |")
    print(f"| 同时入选 PSA+VSA | {len(df_both)} |")

    # Top-10 GCMC VSA candidates
    print(f"\n### Top-10 GCMC-Based VSA Candidates")
    print(f"| Rank | mof_id | GCMC VSA_API | ML VSA_API | PSA Rank | VSA Rank |")
    print(f"|------|--------|-------------|------------|----------|----------|")
    top_vsa = df_vsa.sort_values("gcmc_VSA_API_CH4", ascending=False).head(10)
    for rank, (_, row) in enumerate(top_vsa.iterrows(), 1):
        psa_r = int(row["psa_rank"]) if pd.notna(row["psa_rank"]) else "—"
        vsa_r = int(row["vsa_rank"]) if pd.notna(row["vsa_rank"]) else "—"
        print(
            f"| {rank} | {row['mof_id']} | {row['gcmc_VSA_API_CH4']:.4f} | "
            f"{row['VSA_API_CH4']:.4f} | {psa_r} | {vsa_r} |"
        )

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Loading data from: {DATA_FILE}")
    df, df_psa, df_vsa, df_psa_only, df_vsa_only = load_data()
    df_both = df[df["psa_rank"].notna() & df["vsa_rank"].notna()].copy()

    print(f"[INFO] Total: {len(df)} | PSA: {len(df_psa)} | VSA: {len(df_vsa)} | "
          f"PSA-only: {len(df_psa_only)} | VSA-only: {len(df_vsa_only)} | Both: {len(df_both)}")

    setup_matplotlib()

    print("\n[PLOT 1] Parity scatter (all 199 MOFs) ...")
    plot_parity_all199(df)

    print("[PLOT 2] Performance scatter (PSA + VSA) ...")
    plot_performance_scatter(df_psa, df_vsa)

    print("[PLOT 3] API boxplot by case ...")
    plot_api_boxplot(df_psa_only, df_vsa_only, df_both)

    print_case_statistics(df, df_psa, df_vsa, df_psa_only, df_vsa_only, df_both)


if __name__ == "__main__":
    main()
