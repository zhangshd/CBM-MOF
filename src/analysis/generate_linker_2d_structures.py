#!/usr/bin/env python
"""Generate 2D molecular structure images of representative MOF linkers.

Produces annotated PNG images for PSA-enriched (large) and VSA-enriched (small)
representative linkers, with functional group highlighting suitable for
Figure 11 Panel (c) of the CBM-MOF manuscript.

Output:
    linker_PSA_polycarboxylate.png  -- pyrene-1,2,6,7-tetracarboxylate (o80)
    linker_PSA_Ndonor.png           -- large N-donor pillar (o47/o156)
    linker_VSA_polycarboxylate.png  -- pyromellitate/BTEC (o118)
    linker_VSA_Ndonor.png           -- quinoxaline (o14)
"""

import argparse
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Linker definitions
# ---------------------------------------------------------------------------

LINKERS = {
    "PSA_polycarboxylate": {
        "smiles": "[O-]C(=O)c1c(cc2c3c1ccc1c3c(cc2)c(c(c1)C(=O)[O-])C(=O)[O-])C(=O)[O-]",
        "label": "Pyrene-1,2,6,7-tetracarboxylate",
        "highlight_type": "carboxylate",
        "process": "PSA",
    },
    "PSA_Ndonor": {
        "smiles": "n1cc2cc3ccc4c5c3c3c2c(c1)cc1c3c2c5c3c(c4)cncc3cc2cc1",
        "label": "Large N-donor pillar",
        "highlight_type": "nitrogen",
        "process": "PSA",
    },
    "VSA_polycarboxylate": {
        "smiles": "[O-]C(=O)c1cc(C(=O)[O-])c(cc1C(=O)[O-])C(=O)[O-]",
        "label": "Pyromellitate (BTEC)",
        "highlight_type": "carboxylate",
        "process": "VSA",
    },
    "VSA_Ndonor": {
        "smiles": "c1ccc2c(c1)nccn2",
        "label": "Quinoxaline",
        "highlight_type": "nitrogen",
        "process": "VSA",
    },
}

# Highlight colours (RGB float tuples for RDKit)
COLOR_RED = (0.90, 0.20, 0.20)       # carboxylate -COO-
COLOR_BLUE = (0.15, 0.35, 0.80)      # N atoms
COLOR_RED_BOND = (0.90, 0.20, 0.20)
COLOR_BLUE_BOND = (0.15, 0.35, 0.80)

# Font search paths (system TTF fonts)
FONT_PATHS_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
FONT_PATHS_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _load_font(paths, size):
    """Try each path in order; fall back to default bitmap font."""
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def count_aromatic_rings(mol):
    """Return the number of aromatic rings in *mol*."""
    ri = mol.GetRingInfo()
    n_aromatic = 0
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            n_aromatic += 1
    return n_aromatic


def get_highlight_atoms_and_bonds(mol, highlight_type):
    """Return (highlight_atoms, highlight_bonds, atom_colors, bond_colors).

    For 'carboxylate': highlight all atoms in -C(=O)[O-] groups in red.
    For 'nitrogen': highlight all N atoms in blue.
    """
    atom_colors = {}
    bond_colors = {}
    highlight_atoms = []
    highlight_bonds = []

    if highlight_type == "carboxylate":
        pattern = Chem.MolFromSmarts("[CX3](=[OX1])[O-,OH1]")
        if pattern is None:
            return highlight_atoms, highlight_bonds, atom_colors, bond_colors
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            for idx in match:
                if idx not in atom_colors:
                    highlight_atoms.append(idx)
                atom_colors[idx] = COLOR_RED
            for i, idx1 in enumerate(match):
                for idx2 in match[i + 1:]:
                    bond = mol.GetBondBetweenAtoms(idx1, idx2)
                    if bond is not None:
                        bidx = bond.GetIdx()
                        if bidx not in bond_colors:
                            highlight_bonds.append(bidx)
                        bond_colors[bidx] = COLOR_RED_BOND

    elif highlight_type == "nitrogen":
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                idx = atom.GetIdx()
                highlight_atoms.append(idx)
                atom_colors[idx] = COLOR_BLUE
                for bond in atom.GetBonds():
                    bidx = bond.GetIdx()
                    if bidx not in bond_colors:
                        highlight_bonds.append(bidx)
                    bond_colors[bidx] = COLOR_BLUE_BOND

    return highlight_atoms, highlight_bonds, atom_colors, bond_colors


def _draw_mol_to_png(mol, hl_atoms, hl_bonds, atom_colors, bond_colors, width, height):
    """Render molecule with RDKit Cairo drawer, return PNG bytes."""
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2.5
    opts.addAtomIndices = False
    opts.backgroundColour = (1.0, 1.0, 1.0, 1.0)
    # Let RDKit auto-scale to fit the canvas
    opts.padding = 0.12

    drawer.DrawMolecule(
        mol,
        highlightAtoms=hl_atoms,
        highlightBonds=hl_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_linker(
    smiles: str,
    label: str,
    highlight_type: str,
    process: str,
    output_path: Path,
    width: int = 600,
    height: int = 500,
    no_title: bool = False,
):
    """Draw a single linker with highlighting and optional annotation, save as PNG.

    Args:
        no_title: If True, omit the annotation banner and output only the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Failed to parse SMILES: %s", smiles)
        return False

    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Count properties
    n_heavy = mol.GetNumHeavyAtoms()
    n_aromatic_rings = count_aromatic_rings(mol)

    # Get highlight info
    hl_atoms, hl_bonds, atom_colors, bond_colors = get_highlight_atoms_and_bonds(
        mol, highlight_type
    )

    # -----------------------------------------------------------------------
    # Step 1: Render molecule to PNG via RDKit Cairo
    # -----------------------------------------------------------------------
    mol_png = _draw_mol_to_png(
        mol, hl_atoms, hl_bonds, atom_colors, bond_colors, width, height
    )
    mol_img = Image.open(BytesIO(mol_png)).convert("RGBA")

    if no_title:
        # No annotation banner — save molecule image directly
        final_img = Image.new("RGB", (width, height), "white")
        final_img.paste(mol_img, (0, 0))
        final_img.save(str(output_path), dpi=(400, 400))
        logger.info("Saved (no title): %s (%dx%d px)", output_path.name, final_img.width, final_img.height)
        return True

    # -----------------------------------------------------------------------
    # Step 2: Compose final image with annotation banner on top
    # -----------------------------------------------------------------------
    annotation_h = 50
    final_img = Image.new("RGB", (width, height + annotation_h), "white")
    final_img.paste(mol_img, (0, annotation_h))

    draw = ImageDraw.Draw(final_img)
    font_bold = _load_font(FONT_PATHS_BOLD, 16)
    font_regular = _load_font(FONT_PATHS_REGULAR, 13)

    # Line 1: linker name
    text1 = label
    # Line 2: structural statistics
    text2 = f"{n_heavy} heavy atoms, {n_aromatic_rings} aromatic ring{'s' if n_aromatic_rings != 1 else ''}"
    if highlight_type == "carboxylate":
        n_coo = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=[OX1])[O-,OH1]")))
        text2 += f", {n_coo} -COO\u207B"
    elif highlight_type == "nitrogen":
        n_N = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        text2 += f", {n_N} N atom{'s' if n_N != 1 else ''}"

    # Center-align text
    bbox1 = draw.textbbox((0, 0), text1, font=font_bold)
    x1 = (width - (bbox1[2] - bbox1[0])) // 2
    draw.text((x1, 4), text1, fill="black", font=font_bold)

    bbox2 = draw.textbbox((0, 0), text2, font=font_regular)
    x2 = (width - (bbox2[2] - bbox2[0])) // 2
    draw.text((x2, 24), text2, fill=(100, 100, 100), font=font_regular)

    # Save at 400 DPI
    final_img.save(str(output_path), dpi=(400, 400))
    logger.info("Saved: %s (%dx%d px)", output_path.name, final_img.width, final_img.height)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150/"
            "structural_analysis/psa_vs_vsa_beaters/linker_structures"
        ),
        help="Output directory for PNG images",
    )
    parser.add_argument("--width", type=int, default=600, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=500, help="Molecule canvas height in pixels")
    parser.add_argument("--no-title", action="store_true",
                        help="Omit annotation banner, output molecule image only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, info in LINKERS.items():
        out_path = args.output_dir / f"linker_{key}.png"
        logger.info("Drawing %s: %s", key, info["label"])
        success = draw_linker(
            smiles=info["smiles"],
            label=info["label"],
            highlight_type=info["highlight_type"],
            process=info["process"],
            output_path=out_path,
            width=args.width,
            height=args.height,
            no_title=args.no_title,
        )
        if not success:
            logger.error("Failed to draw %s", key)

    logger.info("All linker images saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
