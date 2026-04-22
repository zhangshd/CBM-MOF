"""Regression checks for the multi-temperature isotherm-fit figure layout."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.figures.fig_isotherm_fits_multitemp import (  # noqa: E402
    CH4_FIG_NAME,
    DEFAULT_LAYOUT_WIDTH,
    N2_FIG_NAME,
    build_figure_layout,
    choose_grid,
    simplify_mof_name,
)
from src.figures.style import MAX_HEIGHT_INCH  # noqa: E402


def test_choose_grid_prefers_taller_layout_for_19_mofs() -> None:
    """Top-19 process candidates should use a 5x4 grid for larger panels."""
    assert choose_grid(19) == (5, 4)


def test_isotherm_fit_layout_stays_page_friendly() -> None:
    """The composite figure should fit within a single manuscript page height."""
    nrows, ncols = choose_grid(19)
    layout = build_figure_layout(nrows, ncols, DEFAULT_LAYOUT_WIDTH)

    assert layout.figure_width == DEFAULT_LAYOUT_WIDTH
    assert layout.figure_height < MAX_HEIGHT_INCH
    assert layout.panel_width > 1.3
    assert layout.panel_height > 0.9
    assert layout.hspace > layout.wspace


def test_short_name_matches_figure11_rules_for_benchmark_and_arc_codes() -> None:
    """Benchmark alias and ARC abbreviations should match the Figure 11 scheme."""
    assert simplify_mof_name("CoRE-2020[Cu][pts]3[ASR]1") == "ATC-Cu"
    assert (
        simplify_mof_name("ARC-DB0-m3_o1480_o156_f0_fsc.sym.14_repeat")
        == "m3_o1480_o156.14"
    )
    assert simplify_mof_name("MOSAEC-YOBPOW_full_REPEAT") == "YOBPOW"
    assert simplify_mof_name("CoRE-2009[Cd][nuc]3[ASR]1") == "2009[Cd][nuc]3[ASR]1"


def test_output_filenames_follow_manuscript_figure_naming() -> None:
    """Generated SI images should use the same fig_* naming convention as peers."""
    assert CH4_FIG_NAME == "fig_isotherm_fits_ch4.png"
    assert N2_FIG_NAME == "fig_isotherm_fits_n2.png"


if __name__ == "__main__":
    test_choose_grid_prefers_taller_layout_for_19_mofs()
    test_isotherm_fit_layout_stays_page_friendly()
    test_short_name_matches_figure11_rules_for_benchmark_and_arc_codes()
    test_output_filenames_follow_manuscript_figure_naming()
    print("4 tests passed")
