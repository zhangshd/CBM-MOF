"""Regression checks for the UQ validation figure helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.alignn.apply_uq_to_library import load_composite_threshold  # noqa: E402
from src.figures.fig_uq_validation import (  # noqa: E402
    PCA_TARGETS,
    PANEL_TITLES,
    TARGET_COLS,
    build_threshold_table,
)


def test_pca_targets_match_main_text_representative_tasks() -> None:
    """The main-text UQ figure should use the three agreed representative targets."""
    assert PCA_TARGETS == ["AdsCH4_10kPa", "AdsCH4_1000kPa", "QstCH4"]


def test_panel_titles_cover_selected_targets_and_sr_panel() -> None:
    """The UQ figure should define titles for the three PCA panels plus SR."""
    assert set(PANEL_TITLES) == {"AdsCH4_10kPa", "AdsCH4_1000kPa", "QstCH4", "SR"}
    assert "SR-based cutoff selection" in PANEL_TITLES["SR"]


def test_build_threshold_table_exports_all_targets_in_order() -> None:
    """The threshold CSV should preserve the canonical target ordering."""
    payload = {
        "percentile": 80,
        "baseline_lsv_mean": {target: idx + 0.1 for idx, target in enumerate(TARGET_COLS)},
        "per_target_p80_lsv_norm": {target: idx + 1.1 for idx, target in enumerate(TARGET_COLS)},
        "sr_sweep": {"pcts": [0, 80, 100], "sr": [1.0, 2.0, 3.0]},
        "composite_threshold": 1.23,
        "composite_retain_fraction": 0.8,
    }

    table = build_threshold_table(payload)

    assert list(table["Target"]) == TARGET_COLS
    assert list(table.columns) == ["Target", "BaselineMeanRawLSV", "P80_LSV_norm"]


def test_load_composite_threshold_reads_json_sidecar(tmp_path: Path) -> None:
    """Library-scale screening should default to the calibrated JSON threshold."""
    threshold_json = tmp_path / "lsv_thresholds.json"
    threshold_json.write_text(
        json.dumps({"composite_threshold": 1.3658608198165902}, indent=2)
    )

    threshold = load_composite_threshold(threshold_json)
    assert threshold == 1.3658608198165902


if __name__ == "__main__":
    test_pca_targets_match_main_text_representative_tasks()
    test_panel_titles_cover_selected_targets_and_sr_panel()
    test_build_threshold_table_exports_all_targets_in_order()
    test_load_composite_threshold_reads_json_sidecar(Path("/tmp"))
    print("4 tests passed")
