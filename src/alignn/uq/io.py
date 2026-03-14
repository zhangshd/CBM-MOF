"""I/O helpers for deployment-based UQ calibration and full-library application."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.alignn.common.constants import TARGET_COLS


def load_split(input_dir: Path, split: str) -> dict:
    """Load latent features, predictions, and ground truth for one split."""
    npz = np.load(input_dir / f"{split}_latent_features.npz", allow_pickle=True)
    pred = pd.read_csv(input_dir / f"{split}_predictions.csv")
    truth = pd.read_csv(input_dir / f"{split}_groundtruth.csv")

    features = npz["features"].astype(np.float32)
    preds = pred[TARGET_COLS].values.astype(np.float32)
    truths = truth[TARGET_COLS].values.astype(np.float32)
    return {
        "mol_ids": list(npz["mol_ids"]),
        "features": features,
        "preds": preds,
        "truths": truths,
    }


def load_deployment_splits(input_dir: Path) -> dict[str, dict]:
    """Load the canonical train/val/test deployment artifacts."""
    return {split: load_split(input_dir, split) for split in ("train", "val", "test")}


def load_full_library_batch_features(batches_dir: Path) -> list[tuple[list[str], np.ndarray]]:
    """Load batch feature NPZs in canonical order for full-library UQ application."""
    rows = []
    for npz_path in sorted(batches_dir.glob("batch_*_features.npz")):
        data = np.load(npz_path, allow_pickle=True)
        rows.append((data["mol_ids"].tolist(), data["features"].astype(np.float32)))
    return rows
