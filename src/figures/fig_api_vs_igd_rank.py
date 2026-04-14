"""
Dual-panel bump chart (slope graph): API rank vs IGD (process) rank
for Top-10 PSA and Top-10 VSA materials.

Figure: fig_api_vs_igd_rank.png
Environment: conda activate skills
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr

# ── Project style import ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    set_publication_style,
    DPI,
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    TICK_FONT_SIZE,
    LABEL_FONT_SIZE,
    BODY_FONT_SIZE,
    TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    save_figure,
)

# ── Data ────────────────────────────────────────────────────────────────────
# API ranks are re-ranked within Top-10 (by descending API value).
# Both API and IGD ranks span 1–10.
MAX_RANK = 10

# PSA Top-10: API re-ranked within these 10 materials
# Labels use abbreviated CIF IDs (ARC-DB0-…_f0_fsc → ARC-…, MOSAEC- dropped)
psa_data = {
    "ARC-m18_o16_o109":        {"igd": 7,  "api": 1},   # API 0.629
    "YOBPOW":                  {"igd": 1,  "api": 2},   # API 0.614
    "ARC-m3_o18_o80":          {"igd": 8,  "api": 3},   # API 0.605
    "ARC-m3_o1480_o156.14":    {"igd": 5,  "api": 4},   # API 0.557
    "ARC-m3_o25_o460.15":      {"igd": 6,  "api": 5},   # API 0.557
    "CoRE-2009-Cd":            {"igd": 2,  "api": 6},   # API 0.505
    "ARC-m3_o10_o146":         {"igd": 9,  "api": 7},   # API 0.504
    "ARC-TAKTOR":              {"igd": 4,  "api": 8},   # API 0.487
    "CoRE-2013-Mg":            {"igd": 10, "api": 9},   # API 0.458
    "ATC-Cu":                  {"igd": 3,  "api": 10},  # API 0.457
}

# VSA Top-10: API re-ranked within these 10 materials
vsa_data = {
    "CoRE-2011-Ni":            {"igd": 7,  "api": 1},   # API 0.556
    "ARC-m3_o47_o15.8":        {"igd": 10, "api": 2},   # API 0.439
    "ARC-m3_o14_o80":          {"igd": 5,  "api": 3},   # API 0.405
    "ARC-m3_o1490_o15.17":     {"igd": 8,  "api": 4},   # API 0.368
    "ARC-m3_o16_o460.20":      {"igd": 4,  "api": 5},   # API 0.351
    "ARC-m3_o1490_o15.16":     {"igd": 9,  "api": 6},   # API 0.300
    "CoRE-2021-Ni":            {"igd": 3,  "api": 7},   # API 0.294
    "QAJDEK":                  {"igd": 6,  "api": 8},   # API 0.270
    "CoRE-2010-Co":            {"igd": 1,  "api": 9},   # API 0.211
    "ATC-Cu":                  {"igd": 2,  "api": 10},  # API 0.173
}


# ── Color helpers ───────────────────────────────────────────────────────────
COLOR_IMPROVE = "#0173B2"   # blue — climbers (API rank worse → IGD rank better)
COLOR_WORSEN  = "#D55E00"   # red–orange — fallers
COLOR_STABLE  = "#949494"   # gray — roughly stable
ATC_DASH      = (0, (4, 2)) # dashed line for ATC-Cu†

RANK_THRESHOLD = 2  # |Δrank| ≤ threshold → "stable"


def rank_color(api: int, igd: int) -> str:
    """Return color based on rank change direction.

    A material *improves* when its process-scale rank (IGD) is better
    (lower number) than its molecular-scale rank (API).  Since we want
    to show materials that were missed by API but surfaced by IGD as
    the interesting "blue" category, improvement = api > igd.
    """
    delta = api - igd  # positive → material climbs from API to IGD
    if abs(delta) <= RANK_THRESHOLD:
        return COLOR_STABLE
    return COLOR_IMPROVE if delta > 0 else COLOR_WORSEN


# ── Drawing one panel ───────────────────────────────────────────────────────
LABEL_FS = 9.5   # Material name font size (≥ TICK_FONT_SIZE + 2)
LABEL_PAD = 0.04  # x-offset from marker to label (data coords)


def draw_bump_panel(
    ax: mpl.axes.Axes,
    data: dict[str, dict],
    title: str,
) -> float:
    """Draw a bump chart on *ax* and return Spearman ρ."""
    api_ranks = [v["api"] for v in data.values()]
    igd_ranks = [v["igd"] for v in data.values()]
    rho, pval = spearmanr(api_ranks, igd_ranks)

    x_api, x_igd = 0.38, 0.7

    # ── Slope lines + markers ───────────────────────────────────────────
    for name, vals in data.items():
        api, igd = vals["api"], vals["igd"]
        color = rank_color(api, igd)
        is_atc = "ATC" in name

        lw = 2.2 if is_atc else 1.3
        ls = ATC_DASH if is_atc else "-"
        zorder = 4 if is_atc else 3

        ax.plot(
            [x_api, x_igd], [api, igd],
            color=color, linewidth=lw, linestyle=ls,
            alpha=1.0 if is_atc else 0.85, zorder=zorder,
            solid_capstyle="round",
        )
        for xp, yp in [(x_api, api), (x_igd, igd)]:
            ax.plot(xp, yp, "o", color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    # ── Labels (data coords, right next to markers) ─────────────────────
    for name, vals in data.items():
        api, igd = vals["api"], vals["igd"]
        color = rank_color(api, igd)
        is_atc = "ATC" in name
        weight = "bold" if is_atc else "normal"

        # Left column: right-aligned, with rank prefix
        ax.text(
            x_api - LABEL_PAD, api, name,
            fontsize=LABEL_FS, fontweight=weight,
            ha="right", va="center", color=color,
            clip_on=False, zorder=6,
        )
        # Right column: left-aligned, with rank prefix
        ax.text(
            x_igd + LABEL_PAD, igd, name,
            fontsize=LABEL_FS, fontweight=weight,
            ha="left", va="center", color=color,
            clip_on=False, zorder=6,
        )

    # ── Axes formatting ─────────────────────────────────────────────────
    ax.set_xlim(-0.15, 1.1)
    ax.set_ylim(10.8, 0.2)  # rank 1 at top (inverted)

    # Y-axis: show rank numbers
    ax.set_yticks(range(1, 11))
    ax.tick_params(axis="y", labelsize=LABEL_FS, length=3, pad=2)

    # X axis: two column headers
    ax.set_xticks([x_api, x_igd])
    ax.set_xticklabels(
        ["API Rank\n(molecular-scale)", "IGD Rank\n(process-scale)"],
        fontsize=LABEL_FONT_SIZE + 1, fontweight="bold",
    )
    ax.tick_params(axis="x", length=0, pad=8)

    # Spines — keep only left for rank axis
    for sp in ["top", "bottom", "right"]:
        ax.spines[sp].set_visible(False)

    # Horizontal guides
    for r in range(1, 11):
        ax.axhline(r, color="#eeeeee", linewidth=0.4, zorder=1)

    # Panel title
    ax.set_title(title, fontsize=TITLE_FONT_SIZE + 1, fontweight="bold",
                 loc="left", pad=10)

    return rho


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    set_publication_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 6),
        gridspec_kw={"wspace": 0.35},
    )
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.15, top=0.92)

    rho_psa = draw_bump_panel(ax1, psa_data, "(a) PSA")
    rho_vsa = draw_bump_panel(ax2, vsa_data, "(b) VSA")

    # ── Shared legend ───────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_IMPROVE, edgecolor="none",
                       label="Rank improves (API → IGD)"),
        mpatches.Patch(facecolor=COLOR_WORSEN, edgecolor="none",
                       label="Rank worsens (API → IGD)"),
        mpatches.Patch(facecolor=COLOR_STABLE, edgecolor="none",
                       label="Rank stable (|Δ| ≤ 2)"),
        plt.Line2D([0], [0], color="k", linestyle=ATC_DASH,
                    linewidth=1.4, label="ATC-Cu (benchmark)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=LEGEND_FONT_SIZE + 1.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    # ── Save ────────────────────────────────────────────────────────────
    out_dir = Path("/home/zhangsd/repos/CBM-MOF-paper/manuscript/Manuscript_CBM/images")
    out_path = out_dir / "fig_api_vs_igd_rank.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"  PSA Spearman ρ = {rho_psa:.4f}")
    print(f"  VSA Spearman ρ = {rho_vsa:.4f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
