"""Figure S1: Statistical overview of the integrated MOF database after deduplication.

Generates a two-panel figure:
  (a) Venn diagram — cross-dataset overlap among ARC, CoRE2024, and MOSAEC.
  (b) Bar chart   — experimental vs hypothetical MOF counts post-dedup.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    NATURE_COLORS,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEDUP_CSV = PROJECT_ROOT / "data" / "processed" / "dedup_cifs" / "duplicate_pdd.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "figures"


# ── Helper classes / functions ──────────────────────────────────────────────


class UnionFind:
    """Disjoint-set (Union-Find) with path compression."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def extract_prefix(cif_name: str) -> str | None:
    """Return the dataset prefix (ARC / CoRE / MOSAEC) or *None*."""
    match = re.match(r"^(ARC|CoRE|MOSAEC)", cif_name)
    return match.group(1) if match else None


_DATASET_PRIORITY = {"CoRE": 1, "MOSAEC": 2, "ARC": 3}


def _dataset_priority(dataset: str | None) -> int:
    return _DATASET_PRIORITY.get(dataset, 4)  # type: ignore[arg-type]


def _select_representative(members_dict: dict[str, int], group_members: list[str]) -> str | None:
    """Pick one representative per connected component.

    Selection order:
      1. Highest occurrence count in ``duplicate_ids``
      2. Dataset priority  CoRE > MOSAEC > ARC
      3. Alphabetical
    """
    ranked = sorted(
        group_members,
        key=lambda x: (-members_dict[x], _dataset_priority(extract_prefix(x)), x),
    )
    return ranked[0] if ranked else None


# ── Data processing ─────────────────────────────────────────────────────────


def analyze_duplicates(csv_path: Path) -> tuple[dict[str, int], int, int]:
    """Parse the dedup CSV and return Venn counts plus exp/hypo bar counts.

    Returns:
        venn_counts: dict with keys like ``'ARC'``, ``'ARC&CoRE'``, etc.
        n_exp:       number of experimental MOFs after dedup.
        n_hypo:      number of hypothetical MOFs after dedup.
    """
    logger.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows", len(df))

    # --- Build Union-Find and count occurrences --------------------------------
    uf = UnionFind()
    all_cifs = set(df["cif"])
    dup_counts: Counter[str] = Counter()

    for _, row in df.iterrows():
        current: str = row["cif"]
        duplicates = [] if pd.isna(row["duplicate_ids"]) else row["duplicate_ids"].split("/")

        dup_counts[current] += 1
        for d in duplicates:
            dup_counts[d] += 1

        # Register nodes
        for d in [current, *duplicates]:
            if d not in uf.parent:
                uf.parent[d] = d

        # Merge with known CIFs only
        for d in duplicates:
            if d in all_cifs:
                uf.union(current, d)

    # --- Group components and determine dataset membership ---------------------
    groups_prefix: dict[str, set[str]] = defaultdict(set)
    groups_cif: dict[str, set[str]] = defaultdict(set)
    for cif in df["cif"]:
        root = uf.find(cif)
        groups_cif[root].add(cif)
        prefix = extract_prefix(cif)
        if prefix:
            groups_prefix[root].add(prefix)

    # --- Select representatives ------------------------------------------------
    deduplicated: list[str] = []
    for root, members in groups_cif.items():
        unique_members = sorted(members)
        members_dict = {m: dup_counts[m] for m in unique_members}
        rep = _select_representative(members_dict, unique_members)
        if rep:
            deduplicated.append(rep)

    # --- Venn diagram counts ---------------------------------------------------
    venn_counts: dict[str, int] = {
        "ARC": 0,
        "CoRE": 0,
        "MOSAEC": 0,
        "ARC&CoRE": 0,
        "ARC&MOSAEC": 0,
        "CoRE&MOSAEC": 0,
        "ARC&CoRE&MOSAEC": 0,
    }
    for prefixes in groups_prefix.values():
        key_parts: list[str] = []
        if "ARC" in prefixes:
            key_parts.append("ARC")
        if "CoRE" in prefixes:
            key_parts.append("CoRE")
        if "MOSAEC" in prefixes:
            key_parts.append("MOSAEC")
        if len(key_parts) == 1:
            venn_counts[key_parts[0]] += 1
        elif len(key_parts) == 2:
            venn_counts["&".join(key_parts)] += 1
        elif len(key_parts) == 3:
            venn_counts["ARC&CoRE&MOSAEC"] += 1

    # --- Experimental / hypothetical -------------------------------------------
    exp_pattern = re.compile(r"^(ARC-DB12|ARC-DB14|CoRE|MOSAEC)")
    n_exp = sum(1 for c in deduplicated if exp_pattern.match(c))
    n_hypo = len(deduplicated) - n_exp

    logger.info("Venn counts: %s", venn_counts)
    logger.info("Deduplicated total: %d  (exp=%d, hypo=%d)", len(deduplicated), n_exp, n_hypo)
    return venn_counts, n_exp, n_hypo


# ── Figure construction ─────────────────────────────────────────────────────


def make_figure(venn_counts: dict[str, int], n_exp: int, n_hypo: int) -> plt.Figure:
    """Build the two-panel Figure S1.

    Args:
        venn_counts: Seven-zone Venn counts keyed by dataset combination.
        n_exp:       Experimental MOF count.
        n_hypo:      Hypothetical MOF count.

    Returns:
        Matplotlib Figure ready for saving.
    """
    set_publication_style()
    layout = compute_panel_grid_layout(nrows=1, ncols=2, figure_width_inch=DOUBLE_COL_INCH)

    fig, (ax_venn, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(layout.figure_width, layout.figure_height),
    )
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=layout.wspace,
    )

    # ── Panel (a): Venn diagram ───────────────────────────────────────────────
    # venn3 expects subset sizes in the order:
    #   (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    # where A=ARC, B=CoRE, C=MOSAEC
    actual_counts = [
        venn_counts["ARC"],
        venn_counts["CoRE"],
        venn_counts["ARC&CoRE"],
        venn_counts["MOSAEC"],
        venn_counts["ARC&MOSAEC"],
        venn_counts["CoRE&MOSAEC"],
        venn_counts["ARC&CoRE&MOSAEC"],
    ]
    # Sqrt-scale for balanced circle sizes; real counts displayed as labels.
    scaled_subsets = tuple(v**0.5 for v in actual_counts)

    venn_colors = (NATURE_COLORS["blue"], NATURE_COLORS["orange"], NATURE_COLORS["green"])

    v = venn3(
        subsets=scaled_subsets,
        set_labels=("ARC", "CoRE2024", "MOSAEC"),
        set_colors=venn_colors,
        alpha=0.5,
        ax=ax_venn,
    )

    if v is not None:
        # Set labels (dataset names)
        for text in v.set_labels or []:
            if text is not None:
                text.set_fontsize(layout.body_font)
                text.set_fontweight("bold")

        # Subset count labels — overwrite with real (non-scaled) counts
        for idx, text in enumerate(v.subset_labels or []):
            if text is not None:
                text.set_text(f"{actual_counts[idx]:,}")
                text.set_fontsize(layout.tick_font)
                text.set_fontweight("bold")

        # Patch edges
        for patch in v.patches or []:
            if patch is not None:
                patch.set_edgecolor("black")
                patch.set_linewidth(mpl.rcParams["axes.linewidth"])

    # Title placed via fig.text below for cross-panel alignment

    # ── Panel (b): Bar chart ──────────────────────────────────────────────────
    categories = ["Experimental", "Hypothetical"]
    counts = [n_exp, n_hypo]
    colors = [NATURE_COLORS["cyan"], NATURE_COLORS["magenta"]]

    bars = ax_bar.bar(
        categories,
        counts,
        width=0.6,
        color=colors,
        edgecolor="black",
        linewidth=mpl.rcParams["axes.linewidth"],
    )

    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=layout.tick_font,
            fontweight="bold",
        )

    # Spine styling
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    ax_bar.set_ylabel("Number of MOFs", fontsize=layout.body_font)
    ax_bar.tick_params(axis="both", which="major", labelsize=layout.tick_font)
    # Title placed via fig.text below for cross-panel alignment

    # Panel titles — use fig.text at a shared y so they align exactly
    title_y = layout.top + 0.035
    for ax, label in [(ax_venn, "(a)"), (ax_bar, "(b)")]:
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.005, title_y, label,
            fontsize=layout.title_font, fontweight="bold",
            ha="left", va="bottom",
            transform=fig.transFigure,
        )

    return fig


# ── CLI entry ────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, run analysis, and save figure."""
    parser = argparse.ArgumentParser(
        description="Generate Figure S1: deduplication overview (Venn + bar chart).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEDUP_CSV,
        help="Path to duplicate_pdd.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved figure (default: %(default)s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    venn_counts, n_exp, n_hypo = analyze_duplicates(args.csv)
    fig = make_figure(venn_counts, n_exp, n_hypo)
    save_figure(fig, "FigureS1_dedup_overview", args.output_dir, formats=("png",))

    # Summary to stdout (useful for verification)
    logger.info("=== Venn diagram counts ===")
    for k, v in venn_counts.items():
        logger.info("  %-20s %d", k, v)
    logger.info("=== Bar chart values ===")
    logger.info("  Experimental:  %d", n_exp)
    logger.info("  Hypothetical:  %d", n_hypo)


if __name__ == "__main__":
    main()
