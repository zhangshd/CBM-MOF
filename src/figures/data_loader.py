"""
Unified data loading and metric calculation for CBM-MOF figure scripts.

All loaders return predictions in physical units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TASK_LIST = [
    "AdsCH4_10kPa",
    "AdsCH4_100kPa",
    "AdsCH4_1000kPa",
    "AdsN2_10kPa",
    "AdsN2_100kPa",
    "AdsN2_1000kPa",
    "QstCH4",
    "QstN2",
]

UPTAKE_TASKS = [task for task in TASK_LIST if task.startswith("Ads")]
QST_TASKS = [task for task in TASK_LIST if task.startswith("Qst")]
MODEL_ORDER = ["CGCNN", "MOFTransformer", "ALIGNN"]


CGCNN_BASE = (
    PROJECT_ROOT
    / "results"
    / "cgcnn_models"
    / "ads_qst_ch4_n2_symlog_1e3_seed42_att_cgcnn"
    / "version_3"
)

MFT_BASE = (
    PROJECT_ROOT
    / "results"
    / "moftransformer_models"
    / "ads_qst_ch4_n2_symlog_1e3_seed42_moftransformer_from_pmtransformer"
    / "version_2"
)

ALIGNN_TEST_DIR = PROJECT_ROOT / "results" / "alignn" / "model_ep150" / "deployment"
ALIGNN_TOP100_DIR = (
    PROJECT_ROOT
    / "results"
    / "alignn"
    / "500ep_symlog_1e-3_ddp2g"
    / "evaluation_ep150"
)


def inv_symlog(y: np.ndarray, tau: float = 1e-3) -> np.ndarray:
    """Inverse of sign(x) * log10(1 + |x| / tau)."""
    return np.sign(y) * tau * (10.0 ** np.abs(y) - 1.0)


def _load_task_frame(base_dir: Path, task: str) -> pd.DataFrame:
    if task in UPTAKE_TASKS:
        path = base_dir / f"test_results_symlog{task}_1e3.csv"
    else:
        path = base_dir / f"test_results_{task}.csv"

    df = pd.read_csv(path)
    ground_truth = df["GroundTruth"].to_numpy()
    predicted = df["Predicted"].to_numpy()
    if task in UPTAKE_TASKS:
        ground_truth = inv_symlog(ground_truth)
        predicted = inv_symlog(predicted)

    return pd.DataFrame(
        {
            "CifId": df["CifId"].to_numpy(),
            f"{task}_true": ground_truth,
            f"{task}_pred": predicted,
        }
    )


def _merge_task_frames(task_frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = task_frames[0]
    for frame in task_frames[1:]:
        merged = merged.merge(frame, on="CifId", how="inner")
    merged = merged.set_index("CifId")
    merged.index.name = "CifId"
    return merged


def load_cgcnn_predictions() -> pd.DataFrame:
    """Load CGCNN test-set predictions in physical units."""
    frames = [_load_task_frame(CGCNN_BASE, task) for task in TASK_LIST]
    return _merge_task_frames(frames)


def load_mft_predictions() -> pd.DataFrame:
    """Load MOFTransformer test-set predictions in physical units."""
    frames = [_load_task_frame(MFT_BASE, task) for task in TASK_LIST]
    return _merge_task_frames(frames)


def load_alignn_predictions(split: str = "test") -> pd.DataFrame:
    """Load ALIGNN predictions in physical units for test or top-100 splits."""
    if split == "test":
        pred_path = ALIGNN_TEST_DIR / "test_predictions.csv"
        true_path = ALIGNN_TEST_DIR / "test_groundtruth.csv"
        pred_df = pd.read_csv(pred_path).rename(columns={"mof_id": "CifId"})
        true_df = pd.read_csv(true_path).rename(columns={"mof_id": "CifId"})
        merged = pred_df.merge(true_df, on="CifId", suffixes=("_pred", "_true"))
        merged = merged.set_index("CifId")
        merged.index.name = "CifId"
        return merged

    if split in {"top_100_psa", "top_100_vsa"}:
        path = ALIGNN_TOP100_DIR / split / f"{split}_predictions_vs_truth.csv"
        df = pd.read_csv(path).rename(columns={"Unnamed: 0": "CifId"})
        df = df.set_index("CifId")
        df.index.name = "CifId"
        return df

    raise ValueError(f"Unknown split: {split}")


def load_model_predictions(model_name: str, split: str = "test") -> pd.DataFrame:
    """Load predictions for a named model."""
    if model_name == "CGCNN":
        if split != "test":
            raise ValueError("CGCNN predictions are only available for the test split.")
        return load_cgcnn_predictions()
    if model_name == "MOFTransformer":
        if split != "test":
            raise ValueError("MOFTransformer predictions are only available for the test split.")
        return load_mft_predictions()
    if model_name == "ALIGNN":
        return load_alignn_predictions(split)
    raise ValueError(f"Unknown model: {model_name}")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute percentage error."""
    mask = np.abs(y_true) > 1e-12
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_task_metrics(df: pd.DataFrame, task: str) -> dict[str, float]:
    """Compute R^2, MAE, and MAPE for a single task."""
    y_true = df[f"{task}_true"].to_numpy()
    y_pred = df[f"{task}_pred"].to_numpy()
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mae_score(y_true, y_pred),
        "MAPE": mape_score(y_true, y_pred),
    }


def build_model_metrics_long() -> pd.DataFrame:
    """Return long-format metrics table for the three retained models."""
    rows: list[dict[str, float | str]] = []
    for model_name in MODEL_ORDER:
        df = load_model_predictions(model_name, split="test")
        for task in TASK_LIST:
            metrics = compute_task_metrics(df, task)
            rows.append(
                {
                    "Model": model_name,
                    "Target": task,
                    "R2": metrics["R2"],
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                }
            )
    return pd.DataFrame(rows)
