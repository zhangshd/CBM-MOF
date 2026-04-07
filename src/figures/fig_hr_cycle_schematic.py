"""
fig_hr_cycle_schematic.py — 7-step HR-PSA/VSA cycle schematic for CBM-MOF manuscript.

Panel (a): Cycle step schematic (bed diagrams with flow arrows)
Panel (b): Bed pressure profile from SuperPSA simulation data

Output: fig_hr_cycle_schematic.png
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from figures.style import (
    set_publication_style, DOUBLE_COL_INCH, DPI, NATURE_COLORS,
    BODY_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE
)

# ── Output paths ────────────────────────────────────────────────────────────
OUT_MAIN = REPO / "src" / "figures" / "fig_hr_cycle_schematic.png"
OUT_PAPER = Path("/home/zhangsd/repos/CBM-MOF-paper/manuscript/figures/fig_hr_cycle_schematic.png")

# ── Data ────────────────────────────────────────────────────────────────────
PROFILE_CSV = REPO / "src" / "SuperPSA" / "Results_extDSL_HR" / "profile_full_PSA_YOBPOW.csv"

# ── Style ───────────────────────────────────────────────────────────────────
set_publication_style()

# Colors from the project palette
C_FEED    = NATURE_COLORS["blue"]      # Feed / cocurrent
C_LIGHT   = NATURE_COLORS["cyan"]      # Light product (N2-rich)
C_HEAVY   = NATURE_COLORS["orange"]    # Heavy product (CH4-rich)
C_BED     = "#E8EDF2"                  # Bed fill (light gray-blue)
C_BED_EDGE = "#5B7FA6"                 # Bed border
C_PRESS   = NATURE_COLORS["blue"]

# ── Step definitions ─────────────────────────────────────────────────────────
STEPS = [
    {
        "num": 1, "short": "CoC-Press",
        "arrows_in":  [("bottom", C_FEED, "Feed")],
        "arrows_out": [],
    },
    {
        "num": 2, "short": "Adsorption",
        "arrows_in":  [("bottom", C_FEED, "Feed")],
        "arrows_out": [("top", C_LIGHT, "Light\nProduct")],
    },
    {
        "num": 3, "short": "HR1",
        "arrows_in":  [("bottom", C_HEAVY, "")],
        "arrows_out": [("top", C_LIGHT, "")],
    },
    {
        "num": 4, "short": "CoC-Depres",
        "arrows_in":  [],
        "arrows_out": [("top", C_LIGHT, "")],
    },
    {
        "num": 5, "short": "HR2",
        "arrows_in":  [("bottom", C_HEAVY, "")],
        "arrows_out": [("top", C_LIGHT, "")],
    },
    {
        "num": 6, "short": "CnC-Blow",
        "arrows_in":  [],
        "arrows_out": [("bottom", C_HEAVY, "CH$_4$")],
    },
    {
        "num": 7, "short": "LR",
        "arrows_in":  [("top", C_LIGHT, "Light\nRecycle")],
        "arrows_out": [],
    },
]

# ── Figure layout ───────────────────────────────────────────────────────────
FIG_W = DOUBLE_COL_INCH          # 6.89 in ≈ 175 mm
FIG_H = 4.5                      # ~114 mm

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Two panels: (a) top 55%, (b) bottom 45%
ax_scheme = fig.add_axes([0.01, 0.46, 0.98, 0.52])   # schematic
ax_press  = fig.add_axes([0.10, 0.06, 0.86, 0.34])   # pressure profile

# ── Panel (a): Bed schematic ─────────────────────────────────────────────────
ax = ax_scheme
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

N = len(STEPS)
# Horizontal layout: beds evenly spaced
bed_w = 0.082       # fraction of axes width
bed_h = 0.44        # fraction of axes height (taller beds, less whitespace)
bed_y0 = 0.25       # bottom of bed — shifted up 0.05 from original 0.20
gap   = (1.0 - N * bed_w) / (N + 1)

arrow_len = 0.10    # arrow length (axes coords)
arrow_lw  = 1.2

# Font sizes: +1 across the board per user request
font_step  = BODY_FONT_SIZE + 0.5
font_label = TICK_FONT_SIZE + 1
font_arrow = max(7.0, TICK_FONT_SIZE - 0.5)

for i, step in enumerate(STEPS):
    # X-centre of this bed
    cx = gap * (i + 1) + bed_w * (i + 0.5)
    bx0 = cx - bed_w / 2
    by0 = bed_y0
    bx1 = bx0 + bed_w
    by1 = by0 + bed_h

    # Hatched bed rectangle (adsorbent)
    rect = FancyBboxPatch(
        (bx0, by0), bed_w, bed_h,
        boxstyle="round,pad=0.003",
        facecolor=C_BED, edgecolor=C_BED_EDGE,
        linewidth=0.8, zorder=2
    )
    ax.add_patch(rect)

    # Light hatching to indicate packed bed
    hatch_rect = FancyBboxPatch(
        (bx0, by0), bed_w, bed_h,
        boxstyle="round,pad=0.003",
        facecolor="none", edgecolor=C_BED_EDGE,
        hatch="///", linewidth=0.0, alpha=0.25, zorder=3
    )
    ax.add_patch(hatch_rect)

    # Step number centered INSIDE the bed rectangle
    ax.text(cx, by0 + bed_h / 2, str(step["num"]),
            ha="center", va="center",
            fontsize=BODY_FONT_SIZE + 1, fontweight="bold",
            color="#333333", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.6),
            zorder=6)

    # Step name at the very bottom of the panel (below all arrows/labels)
    ax.text(cx, 0.0, step["short"],
            ha="center", va="bottom",
            fontsize=font_label, color="#444444",
            transform=ax.transAxes)

    # Draw flow arrows
    def draw_arrow(side, color, label):
        """Draw an arrow entering or leaving the bed at top or bottom."""
        arrowprops = dict(
            arrowstyle="-|>",
            color=color,
            lw=arrow_lw,
            mutation_scale=6,
        )
        if side == "bottom":
            # Arrow pointing INTO bottom
            y_start = by0 - arrow_len
            y_end   = by0 + 0.01
            ax.annotate("", xy=(cx, y_end), xytext=(cx, y_start),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=arrowprops, zorder=5)
            if label:
                ax.text(cx, y_start - 0.01, label,
                        ha="center", va="top",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center")
        elif side == "top":
            # Arrow pointing OUT OF top
            y_start = by1 + 0.01
            y_end   = by1 + arrow_len
            ax.annotate("", xy=(cx, y_end), xytext=(cx, y_start),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=arrowprops, zorder=5)
            if label:
                ax.text(cx, y_end + 0.01, label,
                        ha="center", va="bottom",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center")

    # For "arrows_out" at top, arrow goes upward from bed top
    # For "arrows_in" at bottom, arrow goes upward toward bed bottom
    # For "arrows_in" at top (LR), arrow points downward INTO bed from top
    # For "arrows_out" at bottom (Blow), arrow points downward OUT OF bed
    for (side, color, label) in step["arrows_in"]:
        if side == "bottom":
            draw_arrow("bottom", color, label)
        elif side == "top":
            # LR: light reflux comes IN from top → downward arrow into bed
            arrowprops = dict(arrowstyle="-|>", color=color, lw=arrow_lw, mutation_scale=6)
            y_start = by1 + arrow_len
            y_end   = by1 + 0.01
            ax.annotate("", xy=(cx, y_end), xytext=(cx, y_start),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=arrowprops, zorder=5)
            if label:
                ax.text(cx, y_start + 0.01, label,
                        ha="center", va="bottom",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center")

    for (side, color, label) in step["arrows_out"]:
        if side == "top":
            draw_arrow("top", color, label)
        elif side == "bottom":
            # CnC-Blow: heavy product exits from bottom → downward arrow
            arrowprops = dict(arrowstyle="-|>", color=color, lw=arrow_lw, mutation_scale=6)
            y_start = by0 - 0.01
            y_end   = by0 - arrow_len
            ax.annotate("", xy=(cx, y_end), xytext=(cx, y_start),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=arrowprops, zorder=5)
            if label:
                ax.text(cx, y_end - 0.01, label,
                        ha="center", va="top",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center")

# Legend for arrow colors
legend_handles = [
    Line2D([0], [0], color=C_FEED,  lw=1.5, label="Feed (CH$_4$:N$_2$ = 20:80)"),
    Line2D([0], [0], color=C_LIGHT, lw=1.5, label="Light product (N$_2$-rich)"),
    Line2D([0], [0], color=C_HEAVY, lw=1.5, label="Heavy product (CH$_4$-rich)"),
]
ax.legend(handles=legend_handles, loc="upper center",
          bbox_to_anchor=(0.5, 1.06), ncol=3,
          fontsize=font_label, frameon=False,
          handlelength=1.5, columnspacing=1.0)

# Panel label (a) — use fig coords for alignment with (b)
LABEL_X = 0.035  # shared x position in figure coords for both panel labels
fig.text(LABEL_X, 0.97, "(a)",
         fontsize=TITLE_FONT_SIZE, fontweight="bold",
         va="top", ha="left")

# ── Panel (b): Pressure profile (continuous spatiotemporal data) ──────────────
ax2 = ax_press

# Load full spatiotemporal profile and extract bed exit (last node)
df = pd.read_csv(PROFILE_CSV)
bed_exit = df[df["node"] == 10].sort_values("t_s").copy()
bed_exit["P_bar"] = bed_exit["P_Pa"] / 1e5  # Pa → bar

# Map each step's time range to x ∈ [step_num-1, step_num]
x_all = []
y_all = []
for step_num in sorted(bed_exit["step"].unique()):
    sub = bed_exit[bed_exit["step"] == step_num].sort_values("t_s")
    t = sub["t_s"].values
    p = sub["P_bar"].values
    t_min, t_max = t.min(), t.max()
    step_start_idx = step_num - 1  # step 1 → x ∈ [0, 1], etc.
    if t_max > t_min:
        x_norm = step_start_idx + (t - t_min) / (t_max - t_min)
    else:
        # Edge case: single time point or all identical times → place at midpoint
        x_norm = np.full_like(t, step_start_idx + 0.5)
    x_all.extend(x_norm.tolist())
    y_all.extend(p.tolist())

x_all = np.array(x_all)
y_all = np.array(y_all)

# Continuous pressure line
ax2.plot(x_all, y_all, color=C_PRESS, lw=1.4, zorder=3)

# Shaded region under curve
ax2.fill_between(x_all, y_all, alpha=0.12, color=C_PRESS)

# Step boundary vertical dashed lines
for i in range(1, len(STEPS)):
    ax2.axvline(i, color="#AAAAAA", lw=0.6, ls="--", zorder=1)

# Step labels on x-axis
step_centers = [i + 0.5 for i in range(len(STEPS))]
step_labels  = [s["short"] for s in STEPS]
ax2.set_xticks(step_centers)
ax2.set_xticklabels(step_labels, fontsize=TICK_FONT_SIZE + 0.5, rotation=0)
ax2.set_xlim(0, len(STEPS))

# Y-axis — schematic only, no numerical tick values or units
y_min = 0
y_max = float(y_all.max()) * 1.15
ax2.set_ylim(y_min, y_max)
ax2.set_ylabel("Bed pressure", fontsize=BODY_FONT_SIZE + 1)
ax2.set_yticks([])

# Spine cleanup
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
ax2.spines["left"].set_linewidth(0.5)
ax2.spines["bottom"].set_linewidth(0.5)
ax2.tick_params(axis="both", direction="in", length=3, width=0.5)
ax2.tick_params(axis="x", bottom=False)  # remove bottom tick marks (labels only)

# Panel label (b) — aligned with (a) via shared fig x
fig.text(LABEL_X, 0.42, "(b)",
         fontsize=TITLE_FONT_SIZE, fontweight="bold",
         va="top", ha="left")

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT_MAIN, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_MAIN}")

OUT_PAPER.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PAPER, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PAPER}")

plt.close(fig)
