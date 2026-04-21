"""
fig_hr_cycle_schematic.py — 7-step HR-PSA/VSA cycle schematic for CBM-MOF manuscript.

Panel (a): Cycle step schematic (bed diagrams with flow arrows)
Panel (b): Bed pressure profile from SuperPSA simulation data

Output:
  - fig_hr_cycle_schematic.png
  - fig_hr_cycle_panel_b.png (optional standalone panel for manual assembly)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# ── Path setup ─────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from figures.style import (
    set_publication_style, DOUBLE_COL_INCH, DPI, NATURE_COLORS,
    BODY_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE
)

# ── Output paths ────────────────────────────────────────────────────────────
PAPER_REPO = REPO.parent / "CBM-MOF-paper"
OUT_MAIN = REPO / "src" / "figures" / "fig_hr_cycle_schematic.png"
OUT_PAPER = PAPER_REPO / "manuscript" / "figures" / "fig_hr_cycle_schematic.png"
OUT_PANEL_B_MAIN = REPO / "src" / "figures" / "fig_hr_cycle_panel_b.png"
OUT_PANEL_B_PAPER = PAPER_REPO / "manuscript" / "figures" / "fig_hr_cycle_panel_b.png"

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

def _get_pressure_profile():
    df = pd.read_csv(PROFILE_CSV)
    bed_exit_node = int(df["node"].max())
    bed_exit = df[df["node"] == bed_exit_node].sort_values("t_s").copy()
    bed_exit["P_bar"] = bed_exit["P_Pa"] / 1e5

    x_all = []
    y_all = []
    for step_num in sorted(bed_exit["step"].unique()):
        sub = bed_exit[bed_exit["step"] == step_num].sort_values("t_s")
        t = sub["t_s"].to_numpy()
        p = sub["P_bar"].to_numpy()
        t_min = t.min()
        t_max = t.max()
        step_start_idx = step_num - 1
        if t_max > t_min:
            x_norm = step_start_idx + (t - t_min) / (t_max - t_min)
        else:
            x_norm = np.full_like(t, step_start_idx + 0.5, dtype=float)
        x_all.extend(x_norm.tolist())
        y_all.extend(p.tolist())

    return np.array(x_all), np.array(y_all)


def _plot_pressure_panel(ax):
    x_all, y_all = _get_pressure_profile()

    ax.plot(x_all, y_all, color=C_PRESS, lw=1.4, zorder=3)
    ax.fill_between(x_all, y_all, alpha=0.12, color=C_PRESS)

    for i in range(1, len(STEPS)):
        ax.axvline(i, color="#AAAAAA", lw=0.6, ls="--", zorder=1)

    step_centers = [i + 0.5 for i in range(len(STEPS))]
    step_labels = [s["short"] for s in STEPS]
    ax.set_xticks(step_centers)
    ax.set_xticklabels(step_labels, fontsize=TICK_FONT_SIZE + 0.5, rotation=0)
    ax.set_xlim(0, len(STEPS))
    ax.set_ylim(0, float(y_all.max()) * 1.15)
    ax.set_ylabel("Bed pressure", fontsize=BODY_FONT_SIZE + 1)
    ax.set_yticks([])

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", direction="in", length=3, width=0.5)
    ax.tick_params(axis="x", bottom=False)


def _draw_schematic_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_steps = len(STEPS)
    bed_w = 0.082
    bed_h = 0.44
    bed_y0 = 0.25
    gap = (1.0 - n_steps * bed_w) / (n_steps + 1)

    arrow_len = 0.10
    arrow_lw = 1.2

    font_label = TICK_FONT_SIZE + 1
    font_arrow = max(7.0, TICK_FONT_SIZE - 0.5)

    for i, step in enumerate(STEPS):
        cx = gap * (i + 1) + bed_w * (i + 0.5)
        bx0 = cx - bed_w / 2
        by0 = bed_y0
        by1 = by0 + bed_h

        rect = FancyBboxPatch(
            (bx0, by0), bed_w, bed_h,
            boxstyle="round,pad=0.003",
            facecolor=C_BED, edgecolor=C_BED_EDGE,
            linewidth=0.8, zorder=2
        )
        ax.add_patch(rect)

        hatch_rect = FancyBboxPatch(
            (bx0, by0), bed_w, bed_h,
            boxstyle="round,pad=0.003",
            facecolor="none", edgecolor=C_BED_EDGE,
            hatch="///", linewidth=0.0, alpha=0.25, zorder=3
        )
        ax.add_patch(hatch_rect)

        ax.text(
            cx, by0 + bed_h / 2, str(step["num"]),
            ha="center", va="center",
            fontsize=BODY_FONT_SIZE + 1, fontweight="bold",
            color="#333333", transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.6,
            ),
            zorder=6,
        )

        ax.text(
            cx, 0.0, step["short"],
            ha="center", va="bottom",
            fontsize=font_label, color="#444444",
            transform=ax.transAxes,
        )

        def draw_arrow(side, color, label):
            arrowprops = dict(
                arrowstyle="-|>",
                color=color,
                lw=arrow_lw,
                mutation_scale=6,
            )
            if side == "bottom":
                y_start = by0 - arrow_len
                y_end = by0 + 0.01
                ax.annotate(
                    "", xy=(cx, y_end), xytext=(cx, y_start),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=arrowprops, zorder=5,
                )
                if label:
                    ax.text(
                        cx, y_start - 0.01, label,
                        ha="center", va="top",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center",
                    )
            elif side == "top":
                y_start = by1 + 0.01
                y_end = by1 + arrow_len
                ax.annotate(
                    "", xy=(cx, y_end), xytext=(cx, y_start),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=arrowprops, zorder=5,
                )
                if label:
                    ax.text(
                        cx, y_end + 0.01, label,
                        ha="center", va="bottom",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center",
                    )

        for side, color, label in step["arrows_in"]:
            if side == "bottom":
                draw_arrow("bottom", color, label)
            elif side == "top":
                arrowprops = dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=arrow_lw,
                    mutation_scale=6,
                )
                y_start = by1 + arrow_len
                y_end = by1 + 0.01
                ax.annotate(
                    "", xy=(cx, y_end), xytext=(cx, y_start),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=arrowprops, zorder=5,
                )
                if label:
                    ax.text(
                        cx, y_start + 0.01, label,
                        ha="center", va="bottom",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center",
                    )

        for side, color, label in step["arrows_out"]:
            if side == "top":
                draw_arrow("top", color, label)
            elif side == "bottom":
                arrowprops = dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=arrow_lw,
                    mutation_scale=6,
                )
                y_start = by0 - 0.01
                y_end = by0 - arrow_len
                ax.annotate(
                    "", xy=(cx, y_end), xytext=(cx, y_start),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=arrowprops, zorder=5,
                )
                if label:
                    ax.text(
                        cx, y_end - 0.01, label,
                        ha="center", va="top",
                        fontsize=font_arrow, color=color,
                        transform=ax.transAxes,
                        multialignment="center",
                    )

    legend_handles = [
        Line2D([0], [0], color=C_FEED, lw=1.5, label="Feed (CH$_4$:N$_2$ = 20:80)"),
        Line2D([0], [0], color=C_LIGHT, lw=1.5, label="Light product (N$_2$-rich)"),
        Line2D([0], [0], color=C_HEAVY, lw=1.5, label="Heavy product (CH$_4$-rich)"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.06), ncol=3,
        fontsize=font_label, frameon=False,
        handlelength=1.5, columnspacing=1.0,
    )


def _save_figure(fig, output_paths):
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")


def render_full_figure(output_paths):
    fig_w = DOUBLE_COL_INCH
    fig_h = 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax_scheme = fig.add_axes([0.01, 0.46, 0.98, 0.52])
    ax_press = fig.add_axes([0.10, 0.06, 0.86, 0.34])

    _draw_schematic_panel(ax_scheme)
    _plot_pressure_panel(ax_press)

    label_x = 0.035
    fig.text(
        label_x, 0.97, "(a)",
        fontsize=TITLE_FONT_SIZE, fontweight="bold",
        va="top", ha="left",
    )
    fig.text(
        label_x, 0.42, "(b)",
        fontsize=TITLE_FONT_SIZE, fontweight="bold",
        va="top", ha="left",
    )

    _save_figure(fig, output_paths)
    plt.close(fig)


def render_panel_b(output_paths):
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_INCH * 0.86, 2.0))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.28, top=0.98)
    _plot_pressure_panel(ax)
    _save_figure(fig, output_paths)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        choices=["full", "b"],
        default="full",
        help="Render the full Figure 1 layout or the standalone panel b.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path. If omitted, use the default repo targets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_publication_style()

    if args.panel == "b":
        output_paths = [args.output_path] if args.output_path else [OUT_PANEL_B_MAIN, OUT_PANEL_B_PAPER]
        render_panel_b(output_paths)
        return

    output_paths = [args.output_path] if args.output_path else [OUT_MAIN, OUT_PAPER]
    render_full_figure(output_paths)


if __name__ == "__main__":
    main()
