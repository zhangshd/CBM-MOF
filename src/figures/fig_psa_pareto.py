"""
fig_psa_pareto.py
=================
Pareto front comparison for PSA/VSA process optimization (main text figure).

Layout: 2 panels side by side (1x2), double-column width
  (a) PSA Pareto fronts — all materials, colored by material, shaped by cycle
  (b) VSA Pareto fronts — same encoding

Usage:
    python src/figures/fig_psa_pareto.py
    python src/figures/fig_psa_pareto.py --analysis-csv /path/to/pareto_analysis.csv
    python src/figures/fig_psa_pareto.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import functools
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "figures"))

from style import (
    DOUBLE_COL_INCH, DPI,
    NATURE_COLORS,
    set_publication_style, save_figure,
    compute_panel_grid_layout,
    TICK_FONT_SIZE, LABEL_FONT_SIZE, LEGEND_FONT_SIZE,
)

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
DEFAULT_ANALYSIS = MODEL_DIR / "psa_optimization" / "pareto_analysis.csv"
DEFAULT_RANKING = MODEL_DIR / "psa_optimization" / "material_ranking.csv"
DEFAULT_OUTPUT = REPO_ROOT.parent / "CBM-MOF-paper" / "manuscript" / "figures"

ATC_CU_NAME = "CoRE-2020[Cu][pts]3[ASR]1"

# ---------------------------------------------------------------------------
# Material name shortening
# ---------------------------------------------------------------------------
def _shorten_material_name(name: str) -> str:
    """Strip database prefix and trailing suffixes, keep structural ID."""
    # ATC-Cu is a well-known alias
    if name == ATC_CU_NAME:
        return "ATC-Cu"

    # Strip trailing _repeat / _clean_repeat / _full_REPEAT
    cleaned = re.sub(r"_(full_REPEAT|clean_repeat|repeat)$", "", name)

    # CoRE-YYYY[...] → strip "CoRE-YYYY" prefix
    m = re.match(r"CoRE-\d{4}(.+)", cleaned)
    if m:
        return m.group(1)
    # ARC-DB12- prefix (named structures like TAKTOR)
    m = re.match(r"ARC-DB\d{2}-(.+)", cleaned)
    if m:
        return m.group(1)
    # ARC-DB0- prefix (coded structures): keep m/o/f/fsc identifiers
    m = re.match(r"ARC-DB\d+-(.+)", cleaned)
    if m:
        return m.group(1)
    # MOSAEC- prefix
    m = re.match(r"MOSAEC-(.+)", cleaned)
    if m:
        return m.group(1)
    return cleaned


def _build_short_names(materials: list[str]) -> dict[str, str]:
    """Build unique short names; append suffix if collisions exist."""
    raw = {m: _shorten_material_name(m) for m in materials}
    # Check for duplicates and resolve
    from collections import Counter
    counts = Counter(raw.values())
    result = {}
    seen: dict[str, int] = {}
    for m in materials:
        short = raw[m]
        if counts[short] > 1:
            idx = seen.get(short, 0)
            seen[short] = idx + 1
            result[m] = f"{short}-{idx + 1}" if idx > 0 else short
        else:
            result[m] = short
    return result


# ---------------------------------------------------------------------------
# Color palette (qualitative, max ~11 distinct colors)
# ---------------------------------------------------------------------------
_QUALITATIVE_COLORS = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#CC78BC",  # pink
    "#56B4E9",  # sky blue
    "#CA9161",  # brown
    "#D55E00",  # vermillion
    "#ECE133",  # yellow
    "#949494",  # grey
    "#FBAFE4",  # light pink
    "#0072B2",  # dark blue
]


def _assign_colors_per_panel(materials: list[str]) -> dict[str, str]:
    """Assign colors independently per panel. ATC-Cu always red."""
    colors = {}
    ci = 0
    for m in materials:
        if m == ATC_CU_NAME:
            colors[m] = "#D62728"  # distinct red for benchmark
        else:
            colors[m] = _QUALITATIVE_COLORS[ci % len(_QUALITATIVE_COLORS)]
            ci += 1
    return colors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_pareto_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    mode: str,
    short_names: dict[str, str],
    mat_colors: dict[str, str],
    layout,
):
    """Plot one panel (PSA or VSA) of the Pareto front comparison."""
    sub = df[df["mode"] == mode].copy()
    materials = sub["material_name"].unique()

    # Sort materials by global rank (from ranking CSV order, or by best energy)
    mat_best_energy = sub.groupby("material_name")["energy"].min()
    materials_sorted = mat_best_energy.sort_values().index.tolist()

    # Marker settings (single cycle type — no shape distinction needed)
    cycle_markers = {"Basic": "o", "HR": "o"}
    marker_size_normal = max(8, layout.marker_area * 0.8)
    marker_size_gnd = max(18, layout.marker_area * 2.0)

    # Plot each material
    for mat in materials_sorted:
        mat_df = sub[sub["material_name"] == mat]
        color = mat_colors.get(mat, "#999999")
        short = short_names.get(mat, mat[:10])

        for ct in ["Basic", "HR"]:
            ct_df = mat_df[mat_df["cycle_type"] == ct]
            if ct_df.empty:
                continue

            marker = cycle_markers[ct]

            # Normal (non-GND) points
            normal = ct_df[~ct_df["is_globally_nondominated"]]
            if not normal.empty:
                ax.scatter(
                    normal["productivity"], normal["energy"],
                    c=color, marker=marker, s=marker_size_normal,
                    alpha=0.35, edgecolors="none", zorder=2,
                )

            # GND points (larger, bold edge)
            gnd = ct_df[ct_df["is_globally_nondominated"]]
            if not gnd.empty:
                ax.scatter(
                    gnd["productivity"], gnd["energy"],
                    c=color, marker=marker, s=marker_size_gnd,
                    alpha=0.85, edgecolors="white", linewidths=0.3, zorder=4,
                )

    # Connect global Pareto front with dashed line (top layer)
    gnd_all = sub[sub["is_globally_nondominated"]].sort_values("productivity")
    if not gnd_all.empty:
        ax.plot(
            gnd_all["productivity"], gnd_all["energy"],
            color="k", ls="--", lw=0.8, alpha=0.6, zorder=6,
        )

    # ATC-Cu star marker overlay
    atc_df = sub[sub["material_name"] == ATC_CU_NAME]
    if not atc_df.empty:
        ax.scatter(
            atc_df["productivity"], atc_df["energy"],
            marker="*", s=marker_size_gnd * 1.8,
            facecolors="none", edgecolors="#D62728",
            linewidths=0.8, zorder=5,
        )

    # VSA: extend x-axis to give legend more room
    if mode == "VSA":
        ax.set_xlim(right=12)

    # Axis labels
    ax.set_xlabel(r"Productivity (mol$\cdot$kg$^{-1}\cdot$h$^{-1}$)",
                  fontsize=layout.body_font)
    ax.set_ylabel(r"Energy (kWh$\cdot$ton$^{-1}$)", fontsize=layout.body_font)


def _build_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    materials: list[str],
    short_names: dict[str, str],
    mat_colors: dict[str, str],
    layout,
    ncol: int = 3,
):
    """Build a per-panel legend below the axis for materials + global Pareto front."""
    handles = []

    # Material legend entries (colored circles), in provided sorted order
    for mat in materials:
        color = mat_colors.get(mat, "#999999")
        short = short_names.get(mat, mat[:10])
        h = mlines.Line2D(
            [], [], color=color, marker="o", markersize=4,
            linestyle="None", label=short,
        )
        handles.append(h)

    # GND indicator (dashed line entry)
    handles.append(mlines.Line2D(
        [], [], color="k", ls="--", lw=0.6, label="Global Pareto",
    ))

    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=ncol,
        fontsize=layout.annotation_font - 2,
        frameon=True,
        fancybox=False,
        edgecolor="#dddddd",
        framealpha=0.7,
        handletextpad=0.3,
        labelspacing=0.45,
        borderpad=0.4,
    )


def plot_pareto_figure(analysis_csv: Path, ranking_csv: Path, output_dir: Path):
    """Generate the 1x2 (side-by-side) Pareto front comparison figure."""
    set_publication_style()

    df = pd.read_csv(analysis_csv)
    ranking_df = pd.read_csv(ranking_csv)
    print(f"Loaded {len(df)} Pareto points from {analysis_csv}")

    # Collect all unique materials across both modes
    all_materials = sorted(df["material_name"].unique())
    short_names = _build_short_names(all_materials)

    # Per-panel: get materials sorted by global_rank from ranking CSV
    panel_materials_sorted: dict[str, list[str]] = {}
    panel_colors: dict[str, dict[str, str]] = {}
    for mode in ["PSA", "VSA"]:
        mode_rank = ranking_df[ranking_df["mode"] == mode].sort_values("global_rank")
        panel_mats = mode_rank["material_name"].tolist()
        panel_materials_sorted[mode] = panel_mats
        panel_colors[mode] = _assign_colors_per_panel(panel_mats)

    # Layout — 1 row, 2 columns, double-column width; extra bottom for legends
    layout = compute_panel_grid_layout(
        nrows=1, ncols=2,
        figure_width_inch=DOUBLE_COL_INCH,
        bottom_margin_inch=0.60,
        panel_aspect=0.90,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left, right=layout.right,
        bottom=layout.bottom, top=layout.top,
        wspace=layout.wspace,
    )

    for ax, mode, label in zip(axes, ["PSA", "VSA"], ["(a)", "(b)"]):
        mat_colors = panel_colors[mode]
        _plot_pareto_panel(ax, df, mode, short_names, mat_colors, layout)
        ax.set_title(f"{label} {mode}", fontsize=layout.title_font,
                     fontweight="bold", loc="left")
        ax.tick_params(labelsize=layout.tick_font)

        # Per-panel legend, sorted by global_rank from ranking CSV
        panel_materials = panel_materials_sorted[mode]
        _build_legend(fig, ax, panel_materials, short_names, mat_colors, layout, ncol=1)

    save_figure(fig, "fig_psa_pareto", output_dir)
    plt.close(fig)
    print("Done: fig_psa_pareto.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PSA/VSA Pareto front comparison figure.",
    )
    parser.add_argument(
        "--analysis-csv", type=Path, default=DEFAULT_ANALYSIS,
        help="Path to pareto_analysis.csv",
    )
    parser.add_argument(
        "--ranking-csv", type=Path, default=DEFAULT_RANKING,
        help="Path to material_ranking.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for figure files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_pareto_figure(args.analysis_csv, args.ranking_csv, args.output_dir)


if __name__ == "__main__":
    main()
