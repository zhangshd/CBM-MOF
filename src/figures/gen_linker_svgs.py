"""
Generate 2D linker structure diagrams as SVG (and PNG backup).

Outputs:
  - SVG via rdMolDraw2D.MolDraw2DSVG (vector, PPT-friendly)
  - PNG via rdMolDraw2D.MolDraw2DCairo (raster backup)

No titles; large atom labels; functional-group highlighting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "alignn" / "model_ep150"
    / "structural_analysis" / "psa_vs_vsa_beaters" / "linker_structures"
)

# ── Linker definitions ───────────────────────────────────────────────────────
LINKERS = [
    {
        "filename": "linker_PSA_polycarboxylate",
        "smiles": "[O-]C(=O)c1c(cc2c3c1ccc1c3c(cc2)c(c(c1)C(=O)[O-])C(=O)[O-])C(=O)[O-]",
        "highlight_red": "[C](=O)[O-]",
        "highlight_blue": None,
        "scale": 1.00,  # 4-ring pyrene core (reference)
    },
    {
        "filename": "linker_PSA_Ndonor",
        "smiles": "n1cc2cc3ccc4c5c3c3c2c(c1)cc1c3c2c5c3c(c4)cncc3cc2cc1",
        "highlight_red": None,
        "highlight_blue": "[#7]",
        "scale": 1.00,  # large polycyclic N-donor
    },
    {
        "filename": "linker_VSA_polycarboxylate",
        "smiles": "[O-]C(=O)c1cc(C(=O)[O-])c(cc1C(=O)[O-])C(=O)[O-]",
        "highlight_red": "[C](=O)[O-]",
        "highlight_blue": None,
        "scale": 0.65,  # single benzene ring
    },
    {
        "filename": "linker_VSA_Ndonor",
        "smiles": "c1ccc2c(c1)nccn2",
        "highlight_red": None,
        "highlight_blue": "[#7]",
        "scale": 0.50,  # benzimidazole (smallest)
    },
]

# ── Highlight colors (RGB floats) ────────────────────────────────────────────
RED_HIGHLIGHT = (1.0, 0.3, 0.3)
BLUE_HIGHLIGHT = (0.3, 0.5, 1.0)


def _get_highlight(mol: Chem.Mol, smarts_red: str | None, smarts_blue: str | None):
    """Build highlight atom list and color mapping."""
    highlight_atoms: list[int] = []
    highlight_colors: dict[int, tuple[float, float, float]] = {}

    if smarts_red:
        pattern = Chem.MolFromSmarts(smarts_red)
        if pattern:
            for match in mol.GetSubstructMatches(pattern):
                for idx in match:
                    if idx not in highlight_colors:
                        highlight_atoms.append(idx)
                    highlight_colors[idx] = RED_HIGHLIGHT

    if smarts_blue:
        pattern = Chem.MolFromSmarts(smarts_blue)
        if pattern:
            for match in mol.GetSubstructMatches(pattern):
                for idx in match:
                    if idx not in highlight_colors:
                        highlight_atoms.append(idx)
                    highlight_colors[idx] = BLUE_HIGHLIGHT

    return highlight_atoms, highlight_colors


def draw_linker_svg(smiles: str, smarts_red: str | None, smarts_blue: str | None,
                    output_path: Path, width: int = 800, height: int = 600) -> None:
    """Draw a single linker as SVG using MolDraw2DSVG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Failed to parse SMILES: %s", smiles)
        return
    AllChem.Compute2DCoords(mol)

    highlight_atoms, highlight_colors = _get_highlight(mol, smarts_red, smarts_blue)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.baseFontSize = 0.9
    opts.minFontSize = 30
    opts.maxFontSize = 60
    opts.bondLineWidth = 5.0
    opts.padding = 0.12
    opts.additionalAtomLabelPadding = 0.15

    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                        highlightAtomColors=highlight_colors)
    drawer.FinishDrawing()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(drawer.GetDrawingText())
    logger.info("SVG saved (%dx%d): %s", width, height, output_path)


def draw_linker_png(smiles: str, smarts_red: str | None, smarts_blue: str | None,
                    output_path: Path, width: int = 1200, height: int = 900) -> None:
    """Draw a single linker as PNG using MolDraw2DCairo (backup raster)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Failed to parse SMILES: %s", smiles)
        return
    AllChem.Compute2DCoords(mol)

    highlight_atoms, highlight_colors = _get_highlight(mol, smarts_red, smarts_blue)

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    except Exception:
        logger.warning("MolDraw2DCairo unavailable, skipping PNG for %s", output_path.name)
        return

    opts = drawer.drawOptions()
    opts.baseFontSize = 0.9
    opts.minFontSize = 36
    opts.maxFontSize = 72
    opts.bondLineWidth = 6.0
    opts.padding = 0.12
    opts.additionalAtomLabelPadding = 0.15

    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                        highlightAtomColors=highlight_colors)
    drawer.FinishDrawing()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(drawer.GetDrawingText())
    logger.info("PNG saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate linker 2D structure SVGs for Figure 11.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--width", type=int, default=800, help="SVG width in px (default: 800)")
    parser.add_argument("--height", type=int, default=600, help="SVG height in px (default: 600)")
    args = parser.parse_args()

    for linker in LINKERS:
        svg_path = args.output_dir / f"{linker['filename']}.svg"
        png_path = args.output_dir / f"{linker['filename']}.png"
        scale = linker.get("scale", 1.0)
        svg_w = int(args.width * scale)
        svg_h = int(args.height * scale)
        png_w = int(args.width * 1.5 * scale)
        png_h = int(args.height * 1.5 * scale)

        draw_linker_svg(
            smiles=linker["smiles"],
            smarts_red=linker["highlight_red"],
            smarts_blue=linker["highlight_blue"],
            output_path=svg_path,
            width=svg_w,
            height=svg_h,
        )
        draw_linker_png(
            smiles=linker["smiles"],
            smarts_red=linker["highlight_red"],
            smarts_blue=linker["highlight_blue"],
            output_path=png_path,
            width=png_w,
            height=png_h,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
