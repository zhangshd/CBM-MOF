"""Batch application helpers for full-library uncertainty scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alignn.common.constants import TARGET_COLS
from src.alignn.uq.core import compute_lsv


def score_full_library_batch(
    mol_ids: list[str],
    embeddings: np.ndarray,
    train_labels_orig: np.ndarray,
    index,
    baseline_dist: float,
    baseline_lsv_mean: np.ndarray,
    k: int,
    threshold: float,
) -> pd.DataFrame:
    """Score one full-library feature batch and return the canonical UQ table."""
    lsv_norm = compute_lsv(
        embeddings,
        train_labels_orig,
        index,
        baseline_dist,
        k=k,
        baseline_lsv_mean=baseline_lsv_mean,
    )
    composite = lsv_norm.mean(axis=1)
    batch_df = pd.DataFrame(lsv_norm, columns=[f"{col}_lsv_norm" for col in TARGET_COLS])
    batch_df.insert(0, "mof_id", mol_ids)
    batch_df["lsv_norm_composite"] = composite
    batch_df["flag_high_uq"] = (composite > threshold).astype(np.int8)
    return batch_df
