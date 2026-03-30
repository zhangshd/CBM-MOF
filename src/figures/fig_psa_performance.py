"""
fig_psa_performance.py
======================
Performance summary for PSA/VSA process optimization (main text figure).

Layout: 2 panels side by side (1x2), double-column width
  (a) PSA — scatter of best productivity vs best energy per material
  (b) VSA — same

Each point = one material (from material_ranking.csv), annotated with short name.
ATC-Cu highlighted with distinct marker. Global rank shown as annotation.

Usage:
    python src/figures/fig_psa_performance.py
    python src/figures/fig_psa_performance.py --ranking-csv /path/to/material_ranking.csv
    python src/figures/fig_psa_performance.py --output-dir /path/to/output
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
)

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
DEFAULT_RANKING = MODEL_DIR / "psa_optimization" / "material_ranking.csv"
DEFAULT_OUTPUT = REPO_ROOT.parent / "CBM-MOF-paper" / "manuscript" / "figures"

ATC_CU_NAME = "CoRE-2020[Cu][pts]3[ASR]1"

# ---------------------------------------------------------------------------
# Material name shortening
# ---------------------------------------------------------------------------
_SHORT_NAME_MAP = {
    "CoRE-2020[Cu][pts]3[ASR]1": "ATC-Cu",
    "CoRE-2014[Al][nan]3[ASR]4": "CoRE-Al",
    "CoRE-2023[Cu][pts]3[ASR]2": "CoRE-Cu-2023",
    "CoRE-2009[Cd][nuc]3[ASR]1": "CoRE-Cd",
    "CoRE-2013[Mg][dia]3[ASR]1": "CoRE-Mg",
    "CoRE-2010[Co][pts]3[ASR]2": "CoRE-Co",
    "CoRE-2011[Ni][dia]3[ASR]1": "CoRE-Ni",
    "MOSAEC-YOBPOW_full_REPEAT": "YOBPOW",
}


def _shorten_material_name(name: str) -> str:
    """Create a short, unique display name for a material."""
    if name in _SHORT_NAME_MAP:
        return _SHORT_NAME_MAP[name]

    # ARC-DB1 pattern
    m = re.match(r"ARC-DB1-(\w+)-(\w+)-\w+_\w+_No(\d+)_repeat", name)
    if m:
        return f"{m.group(1)} #{m.group(3)}"

    # ARC-DB0 pattern
    m = re.match(r"ARC-DB\d+-m\d+_o(\d+)_o(\d+)_f\d+_fsc(?:\.sym\.(\d+))?_repeat", name)
    if m:
        o1 = m.group(1)
        sym = m.group(3)
        suffix = f".{sym}" if sym else ""
        return f"ARC-o{o1}{suffix}"

    return name[:15]


def _build_short_names(materials: list[str]) -> dict[str, str]:
    """Build unique short names; append suffix if collisions exist."""
    raw = {m: _shorten_material_name(m) for m in materials}
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
# Color + marker assignment
# ---------------------------------------------------------------------------
_CYCLE_COLORS = {
    "Basic": NATURE_COLORS["blue"],
    "HR": NATURE_COLORS["orange"],
}


# ---------------------------------------------------------------------------
# Label placement via greedy stacking
# ---------------------------------------------------------------------------
def _compute_label_offsets(
    data_xy: list[tuple[float, float]],
    x_range: float,
    y_range: float,
    n_points: int,
) -> list[tuple[float, float, str]]:
    """Compute (dx, dy, ha) offsets for labels to minimize overlap.

    Strategy: sort points by y-coordinate (energy), assign labels stacked
    at evenly-spaced y positions. Alternate left/right placement to avoid
    leader-line crossings.

    Returns list of (x_offset_frac, y_offset_frac, horizontal_alignment)
    where offsets are fractions of the data range.
    """
    # Sort indices by energy (y-value) descending (highest energy at top)
    indices = list(range(n_points))
    sorted_by_y = sorted(indices, key=lambda i: data_xy[i][1], reverse=True)

    offsets = [None] * n_points

    # Compute evenly spaced vertical slots over the full y range
    y_min = min(xy[1] for xy in data_xy)
    y_max = max(xy[1] for xy in data_xy)
    y_pad = y_range * 0.05
    slot_positions = np.linspace(y_max + y_pad, y_min - y_pad, n_points)

    for slot_idx, data_idx in enumerate(sorted_by_y):
        x_data, y_data = data_xy[data_idx]
        target_y = slot_positions[slot_idx]
        dy = target_y - y_data

        # Alternate sides; put labels for leftward points to the right and vice versa
        x_mid = np.median([xy[0] for xy in data_xy])
        if x_data < x_mid:
            # Point is on the left side: label goes further left
            dx = -x_range * 0.15
            ha = "right"
        else:
            # Point is on the right side: label goes further right
            dx = x_range * 0.15
            ha = "left"

        offsets[data_idx] = (dx, dy, ha)

    return offsets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_performance_panel(
    ax: plt.Axes,
    df_mode: pd.DataFrame,
    mode: str,
    short_names: dict[str, str],
    layout,
):
    """Scatter plot: best productivity vs best energy, one point per material."""
    df_mode = df_mode.sort_values("global_rank").reset_index(drop=True)
    n = len(df_mode)

    # Collect data
    data_xy = []
    labels = []
    colors_list = []
    is_atc_list = []
    cycles = []

    for _, row in df_mode.iterrows():
        mat = row["material_name"]
        short = short_names.get(mat, mat[:10])
        rank = int(row["global_rank"])
        is_atc = (mat == ATC_CU_NAME)
        cycle = row["cycle_type_best"]

        data_xy.append((row["best_productivity"], row["best_energy"]))
        labels.append(f"{rank}. {short}")
        colors_list.append("#D62728" if is_atc else "0.2")
        is_atc_list.append(is_atc)
        cycles.append(cycle)

    # Determine axis ranges for offset computation
    x_vals = [xy[0] for xy in data_xy]
    y_vals = [xy[1] for xy in data_xy]
    x_range = max(x_vals) - min(x_vals) if len(x_vals) > 1 else 1.0
    y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0

    # Add padding to axis limits (no negative values for productivity)
    x_pad = x_range * 0.30
    y_pad = y_range * 0.12
    ax.set_xlim(max(0, min(x_vals) - x_pad), max(x_vals) + x_pad)
    ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad)

    # Scatter all points
    for i in range(n):
        is_atc = is_atc_list[i]
        cycle = cycles[i]
        color = "#D62728" if is_atc else _CYCLE_COLORS.get(cycle, "#999999")
        marker = "*" if is_atc else ("o" if cycle == "Basic" else "^")
        size = 90 if is_atc else 45

        ax.scatter(
            data_xy[i][0], data_xy[i][1],
            c=color, marker=marker, s=size,
            edgecolors="white" if not is_atc else "#D62728",
            linewidths=0.5, zorder=5 if is_atc else 3,
        )

    # Compute label positions
    offsets = _compute_label_offsets(data_xy, x_range, y_range, n)
    font_size = layout.annotation_font - 0.5

    for i in range(n):
        dx, dy, ha = offsets[i]
        ax.annotate(
            labels[i],
            xy=data_xy[i],
            xytext=(data_xy[i][0] + dx, data_xy[i][1] + dy),
            textcoords="data",
            fontsize=font_size,
            ha=ha, va="center",
            fontweight="bold" if is_atc_list[i] else "normal",
            color=colors_list[i],
            arrowprops=dict(
                arrowstyle="-",
                color="0.65",
                lw=0.4,
                shrinkA=0, shrinkB=3,
            ),
        )

    ax.set_xlabel(r"Best productivity (mol$\cdot$kg$^{-1}\cdot$h$^{-1}$)",
                  fontsize=layout.body_font)
    ax.set_ylabel(r"Best energy (kWh$\cdot$ton$^{-1}$)", fontsize=layout.body_font)

    # Ideal direction arrow (lower-right = better)
    _add_ideal_arrow(ax, layout)


def _add_ideal_arrow(ax: plt.Axes, layout):
    """Add a small 'ideal' direction indicator in the corner."""
    ax.annotate(
        "Ideal",
        xy=(0.92, 0.08), xycoords="axes fraction",
        xytext=(0.78, 0.22), textcoords="axes fraction",
        fontsize=layout.annotation_font - 1,
        color="0.5",
        arrowprops=dict(
            arrowstyle="->",
            color="0.5",
            lw=0.8,
        ),
        ha="center", va="center",
    )


def plot_performance_figure(ranking_csv: Path, output_dir: Path):
    """Generate the 1x2 performance summary figure."""
    set_publication_style()

    df = pd.read_csv(ranking_csv)
    print(f"Loaded {len(df)} material rankings from {ranking_csv}")

    # Build short names from all materials
    all_materials = sorted(df["material_name"].unique())
    short_names = _build_short_names(all_materials)

    # Layout
    layout = compute_panel_grid_layout(
        nrows=1, ncols=2,
        figure_width_inch=DOUBLE_COL_INCH,
        right_margin_inch=0.30,
        panel_aspect=0.90,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left, right=layout.right,
        bottom=layout.bottom, top=layout.top,
        wspace=layout.wspace + 0.10,
    )

    for ax, mode, label in zip(axes, ["PSA", "VSA"], ["(a)", "(b)"]):
        df_mode = df[df["mode"] == mode].copy()
        _plot_performance_panel(ax, df_mode, mode, short_names, layout)
        ax.set_title(f"{label} {mode}", fontsize=layout.title_font,
                     fontweight="bold", loc="left")
        ax.tick_params(labelsize=layout.tick_font)

    # Legend for cycle types
    handles = [
        mlines.Line2D([], [], color=NATURE_COLORS["blue"], marker="o",
                       markersize=5, linestyle="None", label="Basic cycle"),
        mlines.Line2D([], [], color=NATURE_COLORS["orange"], marker="^",
                       markersize=5, linestyle="None", label="HR cycle"),
        mlines.Line2D([], [], color="#D62728", marker="*",
                       markersize=7, linestyle="None", label="ATC-Cu (benchmark)"),
    ]
    axes[1].legend(
        handles=handles,
        loc="upper right",
        fontsize=layout.annotation_font,
        frameon=False,
        handletextpad=0.3,
    )

    save_figure(fig, "fig_psa_performance", output_dir, formats=("png", "pdf"))
    plt.close(fig)
    print("Done: fig_psa_performance.png/.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PSA/VSA performance summary figure.",
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
    plot_performance_figure(args.ranking_csv, args.output_dir)


if __name__ == "__main__":
    main()
