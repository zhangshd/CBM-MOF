"""Regression checks for the combined Top-100 GCMC validation figure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.figures.fig_top100_validation import (  # noqa: E402
    GCMC_COMPARE_CSV,
    TOP100_SPLIT_CSV,
    load_top100_validation_predictions,
)


def test_top100_validation_loader_uses_current_gcmc_outputs() -> None:
    """Current parity plots should read from the official GCMC comparison CSV."""
    assert GCMC_COMPARE_CSV.name == "gcmc_vs_ml_comparison.csv"
    assert TOP100_SPLIT_CSV["top_100_psa"].name == "top100_psa.csv"
    assert TOP100_SPLIT_CSV["top_100_vsa"].name == "top100_vsa.csv"


def test_top100_validation_loader_returns_100_rows_per_split() -> None:
    """PSA/VSA parity inputs should each contain exactly 100 validated MOFs."""
    psa = load_top100_validation_predictions("top_100_psa")
    vsa = load_top100_validation_predictions("top_100_vsa")

    assert len(psa) == 100
    assert len(vsa) == 100
    assert "AdsCH4_10kPa_true" in psa.columns
    assert "AdsCH4_10kPa_pred" in psa.columns
    assert "QstN2_true" in vsa.columns
    assert "QstN2_pred" in vsa.columns


if __name__ == "__main__":
    test_top100_validation_loader_uses_current_gcmc_outputs()
    test_top100_validation_loader_returns_100_rows_per_split()
    print("2 tests passed")
