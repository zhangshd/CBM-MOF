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
DEFAULT_OUTPUT = REPO_ROOT.parent / "CBM-MOF-paper" / "manuscript" / "figures"

ATC_CU_NAME = "CoRE-2020[Cu][pts]3[ASR]1"

# ---------------------------------------------------------------------------
# Material name shortening
# ---------------------------------------------------------------------------
# Priority map for known materials (checked first, order doesn't matter)
_SHORT_NAME_MAP = {
    "CoRE-2020[Cu][pts]3[ASR]1": "ATC-Cu",
    "CoRE-2014[Al][nan]3[ASR]4": "CoRE-Al",
    "CoRE-2023[Cu][pts]3[ASR]2": "CoRE-Cu-2023",
    "CoRE-2009[Cd][nuc]3[ASR]1": "CoRE-Cd",
    "CoRE-2013[Mg][dia]3[ASR]1": "CoRE-Mg",
    "CoRE-2010[Co][pts]3[ASR]2": "CoRE-Co",
    "CoRE-2011[Ni][dia]3[ASR]1": "CoRE-Ni",
    "MOSAEC-YOBPOW_full_REPEAT": "YOBPOW",
    "CoRE-2021[Ni][dia]3[ASR]1": "CoRE-Ni-2021",
    "ARC-DB12-TAKTOR_clean_repeat": "TAKTOR",
    "MOSAEC-QAJDEK_full_REPEAT": "QAJDEK",
}


def _shorten_material_name(name: str) -> str:
    """Create a short, unique display name for a material."""
    if name in _SHORT_NAME_MAP:
        return _SHORT_NAME_MAP[name]

    # ARC-DB1 pattern: extract linker info + No{digits}
    m = re.match(r"ARC-DB1-(\w+)-(\w+)-\w+_\w+_No(\d+)_repeat", name)
    if m:
        formula = m.group(1)
        # Subscript digits in formula (e.g., Al2O6 -> Al2O6)
        return f"{formula} #{m.group(3)}"

    # ARC-DB0 pattern: extract o{digits} identifiers
    m = re.match(r"ARC-DB\d+-m\d+_o(\d+)_o(\d+)_f\d+_fsc(?:\.sym\.(\d+))?_repeat", name)
    if m:
        o1, o2 = m.group(1), m.group(2)
        sym = m.group(3)
        suffix = f".{sym}" if sym else ""
        return f"ARC-o{o1}{suffix}"

    # Fallback: truncate
    return name[:15]


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


def _assign_colors(materials: list[str], short_names: dict[str, str]) -> dict[str, str]:
    """Assign a unique color to each material, ATC-Cu always red."""
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

    # Cycle type markers
    cycle_markers = {"Basic": "o", "HR": "^"}
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

    # Connect global Pareto front with dashed line
    gnd_all = sub[sub["is_globally_nondominated"]].sort_values("productivity")
    if not gnd_all.empty:
        ax.plot(
            gnd_all["productivity"], gnd_all["energy"],
            color="k", ls="--", lw=0.6, alpha=0.5, zorder=3,
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

    # Axis labels
    ax.set_xlabel(r"Productivity (mol$\cdot$kg$^{-1}\cdot$h$^{-1}$)",
                  fontsize=layout.body_font)
    ax.set_ylabel(r"Energy (kWh$\cdot$ton$^{-1}$)", fontsize=layout.body_font)


def _build_legend(
    fig: plt.Figure,
    ax_right: plt.Axes,
    materials_all: list[str],
    short_names: dict[str, str],
    mat_colors: dict[str, str],
    layout,
):
    """Build a combined legend for materials + cycle types."""
    handles = []

    # Material legend entries (colored circles)
    for mat in materials_all:
        color = mat_colors.get(mat, "#999999")
        short = short_names.get(mat, mat[:10])
        h = mlines.Line2D(
            [], [], color=color, marker="o", markersize=4,
            linestyle="None", label=short,
        )
        handles.append(h)

    # Separator
    handles.append(mlines.Line2D([], [], linestyle="None", label=""))

    # Cycle type markers
    handles.append(mlines.Line2D(
        [], [], color="gray", marker="^", markersize=4,
        linestyle="None", label="HR cycle",
    ))

    # GND indicator
    handles.append(mlines.Line2D(
        [], [], color="k", ls="--", lw=0.6, label="Global Pareto front",
    ))

    ax_right.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=layout.annotation_font,
        frameon=False,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0,
    )


def plot_pareto_figure(analysis_csv: Path, output_dir: Path):
    """Generate the 1x2 Pareto front comparison figure."""
    set_publication_style()

    df = pd.read_csv(analysis_csv)
    print(f"Loaded {len(df)} Pareto points from {analysis_csv}")

    # Collect all unique materials across both modes
    all_materials = sorted(df["material_name"].unique())
    short_names = _build_short_names(all_materials)
    mat_colors = _assign_colors(all_materials, short_names)

    # Layout
    layout = compute_panel_grid_layout(
        nrows=1, ncols=2,
        figure_width_inch=DOUBLE_COL_INCH,
        right_margin_inch=1.60,  # extra space for legend
        panel_aspect=0.90,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left, right=layout.right,
        bottom=layout.bottom, top=layout.top,
        wspace=layout.wspace + 0.08,
    )

    for ax, mode, label in zip(axes, ["PSA", "VSA"], ["(a)", "(b)"]):
        _plot_pareto_panel(ax, df, mode, short_names, mat_colors, layout)
        ax.set_title(f"{label} {mode}", fontsize=layout.title_font,
                     fontweight="bold", loc="left")
        ax.tick_params(labelsize=layout.tick_font)

    # Build combined legend on right side
    _build_legend(fig, axes[1], all_materials, short_names, mat_colors, layout)

    save_figure(fig, "fig_psa_pareto", output_dir, formats=("png", "pdf"))
    plt.close(fig)
    print("Done: fig_psa_pareto.png/.pdf")


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
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for figure files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_pareto_figure(args.analysis_csv, args.output_dir)


if __name__ == "__main__":
    main()
