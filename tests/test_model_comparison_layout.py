"""Regression checks for model-comparison figure layout helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.figures.fig_model_comparison as fig_model_comparison  # noqa: E402
from src.figures.annotation_layout import (  # noqa: E402
    build_corner_annotation_candidates,
    choose_annotation_anchor,
    choose_common_annotation_anchor,
)
from src.figures.data_loader import MODEL_ORDER, TASK_LIST  # noqa: E402
from src.figures.style import (  # noqa: E402
    DOUBLE_COL_INCH,
    LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
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


def test_heatmap_keeps_mean_inside_single_panel_and_uses_larger_fonts() -> None:
    """Figure 4 should keep the mean column inside one heatmap without a title."""
    rows: list[dict[str, float | str]] = []
    for model_idx, model_name in enumerate(MODEL_ORDER):
        for task_idx, task in enumerate(TASK_LIST):
            rows.append(
                {
                    "Model": model_name,
                    "Target": task,
                    "R2": 0.74 + 0.02 * model_idx + 0.005 * task_idx,
                    "MAE": 0.1,
                    "MAPE": 0.2,
                }
            )
    metrics_long = pd.DataFrame(rows)

    captured: dict[str, object] = {}
    original_save = fig_model_comparison.save_figure

    def capture_save(fig, name, output_dir, formats=("png",), tight_layout=True):
        captured["fig"] = fig
        captured["name"] = name

    fig_model_comparison.save_figure = capture_save
    try:
        fig_model_comparison.plot_figure4(Path("/tmp"), metrics_long)
        fig = captured["fig"]
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        cbar_ax = fig.axes[1]

        assert captured["name"] == "Figure04_model_heatmap"
        assert ax.get_title() == ""
        assert [tick.get_text() for tick in ax.get_xticklabels()][-1] == "Mean"
        assert len(ax.get_xticklabels()) == len(TASK_LIST) + 1
        assert len(ax.lines) == 0
        assert ax.get_xticklabels()[0].get_fontsize() == TICK_FONT_SIZE + 1.0
        assert ax.get_yticklabels()[0].get_fontsize() == LABEL_FONT_SIZE + 1.0
        assert cbar_ax.get_ylabel() == r"$R^2$"
    finally:
        fig_model_comparison.save_figure = original_save
        if "fig" in captured:
            plt.close(captured["fig"])


if __name__ == "__main__":
    test_font_scaling_matches_word_target()
    test_grid_layout_compacts_two_row_parity_figure()
    test_annotation_anchor_avoids_dense_corner()
    test_common_annotation_anchor_returns_valid_corner()
    test_corner_candidates_are_generated_from_panel_geometry()
    test_heatmap_keeps_mean_inside_single_panel_and_uses_larger_fonts()
    print("6 tests passed")
