#!/usr/bin/env python
"""Generate publication-quality schematic panels for Cluster 2 (rna) and Cluster 8 (fsc) structural motifs.

Panel (c): Cluster 2 — rna Al-rod architecture with dicarboxylate linkers
Panel (d): Cluster 8 — fsc Zn-paddlewheel pillared-layer architecture

These panels are designed as (c) and (d) additions to Figure 11 in the manuscript.

Usage:
    conda run -n alignn_env python src/analysis/plot_cluster_schematics.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# ── Project style import ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from style import set_publication_style, NATURE_COLORS, DPI, save_figure

logger = logging.getLogger(__name__)

# ── Color palette ───────────────────────────────────────────────────────────
SBU_COLOR = NATURE_COLORS["blue"]       # Blue for SBU / metal nodes
LINKER_COLOR = NATURE_COLORS["green"]   # Green for carboxylate linkers
PILLAR_COLOR = NATURE_COLORS["orange"]  # Orange for N-donor pillar linkers
PORE_COLOR = "#E8F4FD"                  # Light blue for pore regions
BOND_COLOR = "#444444"                  # Dark gray for bonds
LABEL_COLOR = "#333333"                 # Dark gray for labels

# Slightly muted versions for fills
SBU_FILL = "#5DADE2"      # Lighter blue for SBU fill
LINKER_FILL = "#82E0AA"   # Lighter green for carboxylate fill
PILLAR_FILL = "#F8C471"   # Lighter orange for N-donor fill

# ── Shared cross-panel layout constants ──────────────────────────────────────
VIS_TOP = 9.3       # top edge shared by framework & linker boxes
VIS_BOTTOM = 3.1    # bottom edge shared by framework & linker boxes
LABEL_Y = 9.7       # unified y for all top labels (va="bottom")
TITLE_Y = 10.9      # title y
TOPO_Y = 2.3        # topology label y


def render_molecule_image(smiles: str, size: tuple[int, int] = (500, 350),
                          label: str = "") -> Image.Image:
    """Render a molecule from SMILES to a PIL Image using RDKit SVG drawer.

    Args:
        smiles: SMILES string.
        size: (width, height) in pixels.
        label: Optional label (not drawn on image, for logging).

    Returns:
        PIL Image with white background.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    AllChem.Compute2DCoords(mol)

    # Use MolDraw2DCairo for high-quality rendering
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2.5
    opts.padding = 0.15
    opts.additionalAtomLabelPadding = 0.1
    # Make atom labels readable
    opts.minFontSize = 16
    opts.maxFontSize = 24
    opts.annotationFontScale = 0.8
    # Don't show atom indices
    opts.addAtomIndices = False

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()
    img = Image.open(BytesIO(png_data)).convert("RGBA")
    # Make white background transparent
    data = np.array(img)
    white_mask = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
    data[white_mask, 3] = 0
    img = Image.fromarray(data)
    return img




def add_mol_inset(ax: plt.Axes, smiles: str, xy: tuple[float, float],
                  zoom: float = 0.18, label: str = "",
                  box_alignment: tuple[float, float] = (0.5, 0.5),
                  mol_size: tuple[int, int] = (500, 350),
                  frameon: bool = False) -> AnnotationBbox:
    """Add a molecule structure inset to a matplotlib axes.

    Args:
        ax: Target axes.
        smiles: SMILES string.
        xy: Position in data coordinates.
        zoom: Scale factor for the inset.
        label: Display name for the molecule.
        box_alignment: Alignment of the inset box.
        mol_size: Size of the rendered molecule image.
        frameon: Whether to draw a frame around the molecule.

    Returns:
        The AnnotationBbox artist.
    """
    img = render_molecule_image(smiles, size=mol_size, label=label)
    arr = np.array(img)
    imagebox = OffsetImage(arr, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy, frameon=frameon,
                        box_alignment=box_alignment,
                        bboxprops=dict(edgecolor="#AAAAAA", linewidth=0.5,
                                       facecolor="white", alpha=0.95),
                        pad=0.05)
    ax.add_artist(ab)
    return ab


# ── Panel (c): Cluster 2 — rna Al-rod ───────────────────────────────────────

def draw_cluster_2_panel(ax: plt.Axes) -> None:
    """Draw Cluster 2 (rna topology, Al-rod) schematic on the given axes.

    Layout: 3 rods × 4 bridges → 6 windows (3 rows × 2 columns).
    Bridges align with Al atom positions on the rods.
    """
    ROD_PAD = 0.05      # rod FancyBboxPatch pad

    ax.set_xlim(0, 10)
    ax.set_ylim(TOPO_Y - 0.5, TITLE_Y + 0.1)
    ax.axis("off")

    # ── Title ──
    ax.text(0.4, TITLE_Y, "(a) Cluster 2: PSA-dominant", fontsize=10,
            fontweight="bold", ha="left", va="top", color=LABEL_COLOR)

    # ── Assembly schematic (left side) ──
    rod_x = [0.8, 2.6, 4.4]
    bridge_y = [3.8, 5.4, 7.0, 8.6]
    # Rod rect coords: visual edges = VIS_TOP / VIS_BOTTOM
    rod_rect_bottom = VIS_BOTTOM + ROD_PAD   # so visual bottom = VIS_BOTTOM
    rod_rect_top = VIS_TOP - ROD_PAD         # so visual top = VIS_TOP
    rod_rect_h = rod_rect_top - rod_rect_bottom

    for rx in rod_x:
        rod_rect = mpatches.FancyBboxPatch(
            (rx - 0.22, rod_rect_bottom), 0.44, rod_rect_h,
            boxstyle=f"round,pad={ROD_PAD}", facecolor=SBU_FILL, edgecolor=SBU_COLOR,
            linewidth=1.2, alpha=0.9, zorder=5
        )
        ax.add_patch(rod_rect)
        for by in bridge_y:
            ax.plot(rx, by, "o", color=SBU_COLOR, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=6)
        for k in range(len(bridge_y) - 1):
            y_mid = (bridge_y[k] + bridge_y[k + 1]) / 2
            ax.plot(rx, y_mid, ".", color="#CC3333", markersize=3.5, zorder=6)

    # SBU label
    ax.text(rod_x[1], LABEL_Y, "Al-OH rod", fontsize=7.5,
            ha="center", va="bottom", color=SBU_COLOR, fontweight="bold")
    ax.annotate("", xy=(rod_x[1], VIS_TOP - 0.05),
                xytext=(rod_x[1], LABEL_Y + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=SBU_COLOR, lw=0.8))

    # Draw linker bridges at Al atom positions
    for by in bridge_y:
        for j in range(len(rod_x) - 1):
            x_start = rod_x[j] + 0.25
            x_end = rod_x[j + 1] - 0.25
            linker_rect = mpatches.FancyBboxPatch(
                (x_start, by - 0.12), x_end - x_start, 0.24,
                boxstyle="round,pad=0.03", facecolor=LINKER_FILL,
                edgecolor=LINKER_COLOR, linewidth=1.0, alpha=0.85, zorder=4
            )
            ax.add_patch(linker_rect)

    # ── 6 windows ──
    win_row_y = [(bridge_y[0] + bridge_y[1]) / 2,
                 (bridge_y[1] + bridge_y[2]) / 2,
                 (bridge_y[2] + bridge_y[3]) / 2]
    win_col_x = [(rod_x[0] + rod_x[1]) / 2,
                 (rod_x[1] + rod_x[2]) / 2]
    win_half_h = (bridge_y[1] - bridge_y[0]) / 2 - 0.15

    for wy in win_row_y:
        for wx in win_col_x:
            win_rect = mpatches.FancyBboxPatch(
                (wx - 0.65, wy - win_half_h), 1.3, 2 * win_half_h,
                boxstyle="round,pad=0.05", facecolor=PORE_COLOR, edgecolor="none",
                alpha=0.4, zorder=1
            )
            ax.add_patch(win_rect)

    ax.text(win_col_x[0], win_row_y[1], "1D\nchannel", fontsize=7,
            ha="center", va="center", color="#2471A3",
            fontweight="bold", style="italic", zorder=2)

    arrow_x = win_col_x[0]
    arr_extend = 0.35
    ax.annotate("", xy=(arrow_x, bridge_y[0] - arr_extend),
                xytext=(arrow_x, bridge_y[1] + arr_extend),
                arrowprops=dict(arrowstyle="->", color="#2471A3", lw=1.2), zorder=7)
    ax.annotate("", xy=(arrow_x, bridge_y[3] + arr_extend),
                xytext=(arrow_x, bridge_y[2] - arr_extend),
                arrowprops=dict(arrowstyle="->", color="#2471A3", lw=1.2), zorder=7)

    # ── Molecule structures (right side) ──
    linker_data = [
        ("[O-]C(=O)/C=C/C(=O)[O-]", "Fumarate"),
        ("[O-]C(=O)c1ccc(cc1)C(=O)[O-]", "BDC"),
        ("[O-]C(=O)c1ccc2c(c1)ccc(c2)C(=O)[O-]", "NDC"),
    ]

    box_left = 5.7
    box_w = 3.0
    mol_x = box_left + box_w / 2   # 7.2 — centered in box
    mol_y_positions = [8.1, 6.4, 4.7]

    # Bounding box — pad=0, exact visual alignment with framework
    box_rect = mpatches.FancyBboxPatch(
        (box_left, VIS_BOTTOM), box_w, VIS_TOP - VIS_BOTTOM,
        boxstyle="round,pad=0,rounding_size=0.2",
        facecolor="white", edgecolor="#AAAAAA",
        linewidth=0.8, alpha=0.95, zorder=0
    )
    ax.add_patch(box_rect)

    # Header
    ax.text(mol_x, LABEL_Y, "Dicarboxylate linkers", fontsize=7.5,
            ha="center", va="bottom", color=LINKER_COLOR, fontweight="bold")

    for (smi, name), my in zip(linker_data, mol_y_positions):
        add_mol_inset(ax, smi, (mol_x, my), zoom=0.21, label=name,
                      mol_size=(450, 230), frameon=False)
        ax.text(mol_x, my - 0.75, name, fontsize=7, ha="center", va="top",
                fontweight="bold", color=LINKER_COLOR)

    # Arrow from box to framework
    ax.annotate("", xy=(rod_x[-1] + 0.4, bridge_y[2]),
                xytext=(5.7, (mol_y_positions[0] + mol_y_positions[-1]) / 2),
                arrowprops=dict(arrowstyle="-|>", color="#BBBBBB",
                               connectionstyle="arc3,rad=-0.10",
                               lw=0.8, linestyle="--"), zorder=3)

    # ── Topology label ──
    ax.text(5.0, TOPO_Y,
            r"rna topology", fontweight="bold",
            fontsize=8.5, ha="center", va="center", color="#555555",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", linewidth=0.5))


# ── Panel (d): Cluster 8 — fsc Zn-paddlewheel ──────────────────────────────

def draw_paddlewheel(ax: plt.Axes, cx: float, cy: float, size: float = 0.35) -> None:
    """Draw a simplified Zn2 paddlewheel SBU at (cx, cy).

    Two connected spheres with 4 equatorial bond stubs + 2 axial stubs.
    """
    # Two Zn atoms (vertically stacked, close together)
    offset = size * 0.3
    ax.plot(cx, cy + offset, "o", color=SBU_COLOR, markersize=6,
            markeredgecolor="white", markeredgewidth=0.5, zorder=10)
    ax.plot(cx, cy - offset, "o", color=SBU_COLOR, markersize=6,
            markeredgecolor="white", markeredgewidth=0.5, zorder=10)
    # Bond between Zn atoms
    ax.plot([cx, cx], [cy - offset, cy + offset], "-", color=SBU_COLOR,
            linewidth=1.5, zorder=9)

    # Equatorial stubs (4 directions: left, right, and diagonals)
    eq_len = size * 1.1
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.radians(angle_deg)
        dx = eq_len * np.cos(angle_rad)
        dy = eq_len * np.sin(angle_rad) * 0.4  # Flatten to show perspective
        ax.plot([cx, cx + dx], [cy, cy + dy], "-", color=BOND_COLOR,
                linewidth=0.8, zorder=8)

    # Axial stubs (up and down — for pillar coordination)
    ax_len = size * 1.4
    ax.plot([cx, cx], [cy + offset, cy + offset + ax_len], "-",
            color=PILLAR_COLOR, linewidth=1.0, zorder=8)
    ax.plot([cx, cx], [cy - offset, cy - offset - ax_len], "-",
            color=PILLAR_COLOR, linewidth=1.0, zorder=8)


def draw_cluster_8_panel(ax: plt.Axes) -> None:
    """Draw Cluster 8 (fsc topology, Zn-paddlewheel pillared-layer) schematic."""
    LAYER_PAD = 0.08

    ax.set_xlim(0, 10)
    ax.set_ylim(TOPO_Y - 0.2, TITLE_Y + 0.1)
    ax.axis("off")

    # ── Title ──
    ax.text(0.3, TITLE_Y, "(b) Cluster 8: VSA-dominant", fontsize=10,
            fontweight="bold", ha="left", va="top", color=LABEL_COLOR)

    # ── Assembly schematic (left side) ──
    pw_x = [1.2, 3.0, 4.8]
    # Layer y computed so visual edges = VIS_TOP / VIS_BOTTOM
    ly_upper = VIS_TOP - 0.55 - LAYER_PAD
    ly_lower = VIS_BOTTOM + 0.55 + LAYER_PAD
    layer_y = [ly_lower, ly_upper]

    for ly in layer_y:
        layer_rect = mpatches.FancyBboxPatch(
            (0.4, ly - 0.55), 5.1, 1.1,
            boxstyle=f"round,pad={LAYER_PAD}", facecolor=LINKER_FILL,
            edgecolor=LINKER_COLOR, linewidth=0.8, alpha=0.25, zorder=1
        )
        ax.add_patch(layer_rect)
        ax.text(0.3, ly, "2D layer", fontsize=6, ha="right", va="center",
                color=LINKER_COLOR, fontweight="bold", style="italic")
        for px in pw_x:
            draw_paddlewheel(ax, px, ly, size=0.3)
        for j in range(len(pw_x) - 1):
            x1 = pw_x[j] + 0.35
            x2 = pw_x[j + 1] - 0.35
            ax.plot([x1, x2], [ly, ly], "-", color=LINKER_COLOR,
                    linewidth=2.5, alpha=0.7, zorder=3)
            ax.plot([x1, x2], [ly, ly], "-", color=LINKER_COLOR,
                    linewidth=1.0, alpha=1.0, zorder=4)

    # Intralayer window
    win_x1 = pw_x[0] + 0.4
    win_x2 = pw_x[1] - 0.4
    win_rect = mpatches.FancyBboxPatch(
        (win_x1, layer_y[0] - 0.32), win_x2 - win_x1, 0.64,
        boxstyle="round,pad=0.05", facecolor=PORE_COLOR, edgecolor="none",
        alpha=0.5, zorder=2
    )
    ax.add_patch(win_rect)
    ax.text((win_x1 + win_x2) / 2, layer_y[0], "intralayer\n\nwindow", fontsize=6,
            ha="center", va="center", color="#2471A3", style="italic", zorder=2)

    # Pillars
    gallery_y_mid = (layer_y[0] + layer_y[1]) / 2
    for px in pw_x:
        y_bottom = layer_y[0] + 0.65
        y_top = layer_y[1] - 0.65
        pillar_rect = mpatches.FancyBboxPatch(
            (px - 0.12, y_bottom), 0.24, y_top - y_bottom,
            boxstyle="round,pad=0.03", facecolor=PILLAR_FILL,
            edgecolor=PILLAR_COLOR, linewidth=1.0, alpha=0.8, zorder=4
        )
        ax.add_patch(pillar_rect)

    # N-pillar label — aligned with pillar center
    ax.text(pw_x[-1] + 0.55, gallery_y_mid,
            "N-pillar", fontsize=7, ha="left", va="center",
            color=PILLAR_COLOR, fontweight="bold", rotation=90)

    # Interlayer gallery
    gal_rect = mpatches.FancyBboxPatch(
        (pw_x[0] + 0.25, layer_y[0] + 0.7), pw_x[1] - pw_x[0] - 0.5,
        layer_y[1] - layer_y[0] - 1.4,
        boxstyle="round,pad=0.08", facecolor=PORE_COLOR, edgecolor="none",
        alpha=0.35, zorder=1
    )
    ax.add_patch(gal_rect)
    ax.text((pw_x[0] + pw_x[1]) / 2, gallery_y_mid, "interlayer\ngallery",
            fontsize=6, ha="center", va="center", color="#2471A3",
            fontweight="bold", style="italic", zorder=2)

    # SBU label
    ax.text(pw_x[1], LABEL_Y, r"Zn$_2$ paddlewheel", fontsize=7.5,
            ha="center", va="bottom", color=SBU_COLOR, fontweight="bold")
    ax.annotate("", xy=(pw_x[1], VIS_TOP - 0.05),
                xytext=(pw_x[1], LABEL_Y + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=SBU_COLOR, lw=0.8))

    # ── Molecule structures (right side) — pad=0 for exact alignment ──
    mol_x = 7.8
    box_gap = 0.25
    box_mid = (VIS_BOTTOM + VIS_TOP) / 2
    tetra_box_y = box_mid + box_gap / 2
    tetra_box_h = VIS_TOP - tetra_box_y
    ndonor_box_y = VIS_BOTTOM
    ndonor_box_h = box_mid - box_gap / 2 - ndonor_box_y

    # Tetracarboxylate bounding box
    box_rect_tetra = mpatches.FancyBboxPatch(
        (6.2, tetra_box_y-0.25), 3.2, tetra_box_h+0.25, facecolor="white", edgecolor="#AAAAAA",
        boxstyle="round,pad=0,rounding_size=0.2",
        linewidth=0.8, alpha=0.95, zorder=0
    )
    ax.add_patch(box_rect_tetra)

    # Tetracarboxylate molecules — shifted up within box to avoid label overlap
    tetra_center = tetra_box_y + tetra_box_h * 0.55
    tetra_y1 = tetra_center + 0.75
    tetra_y2 = tetra_center - 0.8

    tetra_smi1 = "[O-]C(=O)c1cc(C(=O)[O-])c(cc1C(=O)[O-])C(=O)[O-]"
    add_mol_inset(ax, tetra_smi1, (mol_x, tetra_y1), zoom=0.17, label="pyromellitate",
                  mol_size=(420, 250), frameon=False)
    ax.text(mol_x, tetra_y1 - 0.55, "Pyromellitate", fontsize=6, ha="center",
            va="top", fontweight="bold", color=LINKER_COLOR)

    tetra_smi2 = "O=C([O-])c1cc(C(=O)[O-])c2cc(C(=O)[O-])cc(C(=O)[O-])c2c1"  # 1,3,5,7-NTC (freq=5)
    add_mol_inset(ax, tetra_smi2, (mol_x, tetra_y2), zoom=0.18, label="NTC",
                  mol_size=(550, 350), frameon=False)
    ax.text(mol_x, tetra_y2 - 0.8, "NTC", fontsize=6, ha="center",
            va="top", fontweight="bold", color=LINKER_COLOR)

    # N-donor bounding box
    box_rect_ndonor = mpatches.FancyBboxPatch(
        (6.2, ndonor_box_y), 3.2, ndonor_box_h-0.2, facecolor="white", edgecolor="#AAAAAA",
        boxstyle="round,pad=0,rounding_size=0.2",
        linewidth=0.8, alpha=0.95, zorder=0
    )
    ax.add_patch(box_rect_ndonor)

    # N-donor molecules — shifted up within box to avoid label overlap
    ndonor_center = ndonor_box_y + ndonor_box_h * 0.55
    pillar_y1 = ndonor_center + 0.6
    pillar_y2 = ndonor_center - 0.8

    pillar_smi1 = "c1ccc2c(c1)nccn2"
    add_mol_inset(ax, pillar_smi1, (mol_x, pillar_y1), zoom=0.15, label="quinoxaline",
                  mol_size=(380, 220), frameon=False)
    ax.text(mol_x, pillar_y1 - 0.55, "Quinoxaline", fontsize=6, ha="center",
            va="top", fontweight="bold", color=PILLAR_COLOR)

    pillar_smi2 = "c1ccc2nc3ccccc3nc2c1"  # Phenazine (freq=5, rank #2)
    add_mol_inset(ax, pillar_smi2, (mol_x, pillar_y2), zoom=0.18, label="phenazine",
                  mol_size=(380, 220), frameon=False)
    ax.text(mol_x, pillar_y2 - 0.55, "Phenazine", fontsize=6, ha="center",
            va="top", fontweight="bold", color=PILLAR_COLOR)

    # Header
    ax.text(mol_x, LABEL_Y, "Dual-linker system", fontsize=7.5,
            ha="center", va="bottom", color="#555555", fontweight="bold")

    # Connecting arrows
    ax.annotate("", xy=(pw_x[-1] + 0.5, layer_y[1]),
                xytext=(6.2, tetra_center),
                arrowprops=dict(arrowstyle="-|>", color=LINKER_COLOR,
                               connectionstyle="arc3,rad=-0.12",
                               lw=0.6, linestyle="--"), zorder=3)
    ax.annotate("", xy=(pw_x[-1] + 0.2, gallery_y_mid - 0.8),
                xytext=(6.2, ndonor_center),
                arrowprops=dict(arrowstyle="-|>", color=PILLAR_COLOR,
                               connectionstyle="arc3,rad=0.12",
                               lw=0.6, linestyle="--"), zorder=3)

    # ── Topology label ──
    ax.text(5.0, TOPO_Y,
            r"fsc topology", fontweight="bold",
            fontsize=8.5, ha="center", va="center", color="#555555",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", linewidth=0.5))



# ── Main figure generation ──────────────────────────────────────────────────

def generate_single_panel(draw_func, output_path: Path) -> None:
    """Generate a single panel figure."""
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.9))
    draw_func(ax)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def generate_combined_figure(output_path: Path) -> None:
    """Generate combined side-by-side figure with both panels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.9))
    fig.subplots_adjust(wspace=0.05)
    draw_cluster_2_panel(ax1)
    draw_cluster_8_panel(ax2)

    # Add a subtle vertical separator between the two panels
    line = mlines.Line2D([0.48, 0.48], [0.02, 0.95], transform=fig.transFigure,
                         color="#DDDDDD", linewidth=0.8, linestyle="-")
    fig.add_artist(line)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Cluster 2 and Cluster 8 structural schematic panels."
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150/structural_analysis"),
        help="Output directory for figures."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_publication_style()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual panels
    logger.info("Generating Cluster 2 panel...")
    generate_single_panel(
        draw_cluster_2_panel,
        output_dir / "cluster_2_schematic.png",
    )

    logger.info("Generating Cluster 8 panel...")
    generate_single_panel(
        draw_cluster_8_panel,
        output_dir / "cluster_8_schematic.png",
    )

    # Generate combined figure
    logger.info("Generating combined figure...")
    generate_combined_figure(output_dir / "cluster_2_vs_8_schematic_combined.png")

    logger.info("All figures generated successfully.")


if __name__ == "__main__":
    main()
