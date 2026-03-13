"""Regression checks for model-comparison figure layout helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.figures.annotation_layout import (  # noqa: E402
    build_corner_annotation_candidates,
    choose_annotation_anchor,
    choose_common_annotation_anchor,
)
from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    compute_panel_grid_layout,
    derive_word_equivalent_fonts,
)


def test_font_scaling_matches_word_target() -> None:
    """The Word-calibrated fonts should be smaller than a 1:1 point mapping."""
    title_font, body_font = derive_word_equivalent_fonts(DOUBLE_COL_INCH)
    assert 9.0 <= title_font <= 10.5
    assert 8.0 <= body_font <= 9.0
    assert title_font > body_font


def test_grid_layout_compacts_two_row_parity_figure() -> None:
    """Two-row parity figures should not require an excessively tall canvas."""
    layout = compute_panel_grid_layout(nrows=2, ncols=4, figure_width_inch=DOUBLE_COL_INCH)
    assert layout.figure_height < 5.0
    assert layout.hspace < 0.12
    assert layout.marker_area > 0
    assert layout.tick_font < layout.body_font
    assert layout.annotation_font <= layout.tick_font


def test_annotation_anchor_avoids_dense_corner() -> None:
    """Annotation should move away from the densest occupied quadrant."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.15, 0.03, size=300)
    y = rng.normal(0.85, 0.03, size=300)
    candidates = build_corner_annotation_candidates(
        panel_width_inch=1.4,
        panel_height_inch=1.35,
        font_size_pt=7.0,
        n_lines=3,
        max_line_chars=12,
    )
    position = choose_annotation_anchor(x, y, limits=(0.0, 1.0), candidates=candidates)
    assert position["name"] != "upper left"


def test_common_annotation_anchor_returns_valid_corner() -> None:
    """Global anchor selection should return one shared valid corner name."""
    panels = [
        (np.array([0.1, 0.2]), np.array([0.2, 0.3]), (0.0, 1.0)),
        (np.array([0.7, 0.8]), np.array([0.8, 0.9]), (0.0, 1.0)),
    ]
    candidates = build_corner_annotation_candidates(
        panel_width_inch=1.4,
        panel_height_inch=1.35,
        font_size_pt=7.0,
        n_lines=3,
        max_line_chars=12,
    )
    assert choose_common_annotation_anchor(panels, candidates=candidates) in {
        "upper left",
        "upper right",
        "lower left",
        "lower right",
    }


def test_corner_candidates_are_generated_from_panel_geometry() -> None:
    """Corner candidates should be computed from panel size and text geometry."""
    candidates = build_corner_annotation_candidates(
        panel_width_inch=1.4,
        panel_height_inch=1.35,
        font_size_pt=7.0,
        n_lines=3,
        max_line_chars=12,
    )
    assert len(candidates) == 4
    assert {candidate["name"] for candidate in candidates} == {
        "upper left",
        "upper right",
        "lower left",
        "lower right",
    }


if __name__ == "__main__":
    test_font_scaling_matches_word_target()
    test_grid_layout_compacts_two_row_parity_figure()
    test_annotation_anchor_avoids_dense_corner()
    test_common_annotation_anchor_returns_valid_corner()
    test_corner_candidates_are_generated_from_panel_geometry()
    print("5 tests passed")
