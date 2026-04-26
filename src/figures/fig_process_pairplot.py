"""
fig_process_pairplot.py
======================
Seaborn pairwise scatter plots for process metrics and varying operating variables.

Outputs one pairplot per process mode:
  - PSA Top-N materials by IGD
  - VSA Top-N materials by IGD

Usage:
    python src/figures/fig_process_pairplot.py
    python src/figures/fig_process_pairplot.py --top-n 5
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "figures"))

from style import save_figure, set_publication_style
from fig_psa_pareto import _assign_colors_per_panel, _build_short_names

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
DEFAULT_EVAL = MODEL_DIR / "psa_optimization" / "pareto_eval_results.csv"
DEFAULT_RANKING = MODEL_DIR / "psa_optimization" / "material_ranking.csv"
DEFAULT_OUTPUT = REPO_ROOT.parent / "CBM-MOF-paper" / "manuscript" / "figures"


def _select_top_materials(ranking_df: pd.DataFrame, mode: str, top_n: int) -> list[str]:
    """Return the top-N materials by IGD for a given process mode."""
    return (
        ranking_df[ranking_df["mode"] == mode]
        .sort_values("global_rank")
        .head(top_n)["material_name"]
        .tolist()
    )


def _prepare_mode_dataframe(
    df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    mode: str,
    top_n: int,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Build the pairplot dataframe and palette for one process mode."""
    materials = _select_top_materials(ranking_df, mode, top_n)
    sub = df[(df["mode"] == mode) & (df["material_name"].isin(materials))].copy()

    short_names = _build_short_names(materials)
    palette_raw = _assign_colors_per_panel(materials)
    palette = {short_names[name]: color for name, color in palette_raw.items()}

    data = pd.DataFrame(
        {
            "Material": sub["material_name"].map(short_names),
            "CH$_4$\nProductivity": sub["productivity_mol_kg_h"],
            "CH$_4$\nRecovery": sub["recovery"] * 100.0,
            "Energy": sub["energy_kWh_ton"],
        }
    )

    variable_map: list[tuple[str, str, float | None]] = [
        ("x1_P_H", "P_H", 1e5),
        ("x2_t_ads", "t_ads", None),
        ("x3_alpha", "alpha", None),
        ("x4_v0", "v0", None),
        ("x5_beta1", "beta1", None),
        ("x6_P_L", "P_L", 1e5),
        ("x7_t_pres", "t_pres", None),
        ("x8_t_CnC", "t_CnC", None),
        ("x9_t_CoC", "t_CoC", None),
        ("x10_beta2", "beta2", None),
    ]

    selected_labels = ["CH$_4$\nProductivity", "CH$_4$\nRecovery", "Energy"]
    for source_col, label, scale in variable_map:
        if source_col not in sub.columns:
            continue
        series = sub[source_col]
        if series.nunique(dropna=True) <= 1:
            continue
        if scale is not None:
            series = series / scale
        data[label] = series
        selected_labels.append(label)

    return data, selected_labels, palette


def _plot_mode_pairplot(
    data: pd.DataFrame,
    vars_to_plot: list[str],
    palette: dict[str, str],
    mode: str,
    top_n: int,
    output_dir: Path,
) -> None:
    """Render and save one seaborn pairplot."""
    n_vars = len(vars_to_plot)
    cell_size = 1.80 if n_vars <= 10 else 1.65
    title_font_size = 18.0
    label_font_size = 13.0 if n_vars <= 10 else 12.0
    tick_font_size = 10.5 if n_vars <= 10 else 9.5
    legend_font_size = 11.0

    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "font.size": tick_font_size,
            "axes.labelsize": label_font_size,
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
            "legend.fontsize": legend_font_size,
        },
    )
    grid = sns.pairplot(
        data=data,
        vars=vars_to_plot,
        hue="Material",
        palette=palette,
        corner=True,
        diag_kind="hist",
        plot_kws={"s": 26, "alpha": 0.62, "linewidth": 0},
        diag_kws={"bins": 18, "alpha": 0.60, "edgecolor": "white", "linewidth": 0.3},
        height=cell_size,
        aspect=1.0,
    )

    grid.figure.suptitle(
        f"{mode} Top-{top_n} Pairwise Scatter",
        y=0.995,
        fontsize=title_font_size,
        fontweight="bold",
    )

    if grid._legend is not None:
        grid._legend.set_title("")
        grid._legend.set_bbox_to_anchor((0.80, 0.97), transform=grid.figure.transFigure)
        grid._legend._loc = 2
        grid._legend.set_frame_on(True)
        grid._legend.get_frame().set_edgecolor("#DDDDDD")
        grid._legend.get_frame().set_alpha(0.75)
        for text in grid._legend.texts:
            text.set_fontsize(legend_font_size)

    for ax_row in grid.axes:
        if ax_row is None:
            continue
        for ax in ax_row:
            if ax is None:
                continue
            ax.tick_params(
                axis="both",
                labelsize=tick_font_size,
                length=2.5,
                pad=1.5,
                width=0.5,
            )
            ax.xaxis.label.set_size(label_font_size)
            ax.yaxis.label.set_size(label_font_size)
            ax.xaxis.labelpad = 4
            ax.yaxis.labelpad = 4
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    grid.figure.subplots_adjust(
        left=0.085,
        right=0.98,
        bottom=0.085,
        top=0.93,
        wspace=0.06,
        hspace=0.06,
    )
    save_figure(
        grid.figure,
        f"fig_process_pairplot_{mode.lower()}_top{top_n}",
        output_dir,
        tight_layout=False,
    )
    plt.close(grid.figure)


def plot_pairplots(
    eval_csv: Path,
    ranking_csv: Path,
    output_dir: Path,
    top_n: int,
) -> None:
    """Generate PSA and VSA pairplots."""
    set_publication_style()
    df = pd.read_csv(eval_csv)
    ranking_df = pd.read_csv(ranking_csv)
    print(f"Loaded {len(df)} Pareto-evaluated points from {eval_csv}")

    for mode in ["PSA", "VSA"]:
        data, vars_to_plot, palette = _prepare_mode_dataframe(df, ranking_df, mode, top_n)
        print(f"{mode} variables: {vars_to_plot}")
        _plot_mode_pairplot(data, vars_to_plot, palette, mode, top_n, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PSA/VSA pairwise scatter plots for process variables.",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULT_EVAL,
        help="Path to pareto_eval_results.csv",
    )
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=DEFAULT_RANKING,
        help="Path to material_ranking.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for figure files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top-ranked materials to include per mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_pairplots(
        eval_csv=args.eval_csv,
        ranking_csv=args.ranking_csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
