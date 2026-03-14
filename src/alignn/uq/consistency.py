"""Consistency checks for canonical UQ artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.alignn.uq.core import load_lsv_thresholds


def validate_full_library_uq_consistency(
    full_library_uq_csv: Path,
    thresholds_json: Path,
    composite_col: str = "lsv_norm_composite",
    flag_col: str = "flag_high_uq",
) -> dict:
    """Check whether the persisted high-UQ flags match the persisted threshold."""
    df = pd.read_csv(full_library_uq_csv)
    thresholds = load_lsv_thresholds(thresholds_json)
    threshold = float(thresholds["composite_threshold"])

    expected = (df[composite_col] > threshold).astype(np.int8)
    actual = df[flag_col].astype(np.int8)
    mismatch = int((expected != actual).sum())
    return {
        "rows": int(len(df)),
        "threshold": threshold,
        "mismatch_count": mismatch,
        "expected_flagged": int(expected.sum()),
        "actual_flagged": int(actual.sum()),
        "is_consistent": mismatch == 0,
    }
