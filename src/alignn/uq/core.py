"""Core LSV and faiss helpers shared by UQ calibration and application."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.alignn.common.constants import DEFAULT_K_NEIGHBORS, MIN_SPEARMAN_RHO, TARGET_COLS


def build_faiss_index(train_emb: np.ndarray):
    """Build a faiss IndexFlatL2, preferring GPU when available."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss not installed. Install faiss-gpu or faiss-cpu.") from exc

    dim = train_emb.shape[1]
    index_cpu = faiss.IndexFlatL2(dim)
    index_cpu.add(train_emb)

    try:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index_cpu)
    except Exception:
        return index_cpu


def serialize_index(index) -> bytes:
    """Serialize a faiss index to portable bytes."""
    import faiss

    try:
        index = faiss.index_gpu_to_cpu(index)
    except Exception:
        pass
    return faiss.serialize_index(index)


def deserialize_index(index_bytes: bytes):
    """Deserialize a faiss index from portable bytes."""
    import faiss

    return faiss.deserialize_index(index_bytes)


def compute_baseline_dist(index, train_emb: np.ndarray, k: int = DEFAULT_K_NEIGHBORS) -> float:
    """Compute the mean k-NN L2 distance within the training set, excluding self."""
    distances, _ = index.search(train_emb, k + 1)
    return float(distances[:, 1:].mean())


def compute_lsv(
    query_emb: np.ndarray,
    train_labels_orig: np.ndarray,
    index,
    baseline_dist: float,
    k: int = DEFAULT_K_NEIGHBORS,
    baseline_lsv_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Compute raw or normalized latent-space variance."""
    distances, indices = index.search(query_emb.astype(np.float32), k)
    sigma2 = max(baseline_dist**2, 1e-8)
    weights = np.exp(-distances / sigma2)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)

    neighbor_labels = train_labels_orig[indices]
    weighted_mean = (weights[:, :, None] * neighbor_labels).sum(axis=1)
    diff = neighbor_labels - weighted_mean[:, None, :]
    lsv = (weights[:, :, None] * diff**2).sum(axis=1)
    if baseline_lsv_mean is not None:
        lsv = lsv / (baseline_lsv_mean + 1e-12)
    return lsv.astype(np.float32)


def calibrate_uq(lsv: np.ndarray, pred_orig: np.ndarray, true_orig: np.ndarray) -> dict:
    """Compute per-target Spearman correlations between LSV and absolute error."""
    from scipy.stats import spearmanr

    results: dict[str, dict[str, float | int | None]] = {}
    for index, col in enumerate(TARGET_COLS):
        lsv_vals = lsv[:, index]
        abs_error = np.abs(pred_orig[:, index] - true_orig[:, index])
        mask = np.isfinite(lsv_vals) & np.isfinite(abs_error)
        count = int(mask.sum())
        if count < 10:
            results[col] = {"rho": None, "pval": None, "n": count}
            continue
        rho, pval = spearmanr(lsv_vals[mask], abs_error[mask])
        results[col] = {"rho": float(rho), "pval": float(pval), "n": count}

    valid_rhos = [entry["rho"] for entry in results.values() if entry["rho"] is not None]
    results["_summary"] = {
        "min_spearman_rho": MIN_SPEARMAN_RHO,
        "n_pass": sum(1 for rho in valid_rhos if rho > MIN_SPEARMAN_RHO),
        "n_total": len(valid_rhos),
        "mean_rho": float(np.mean(valid_rhos)) if valid_rhos else None,
    }
    return results


def save_uncertainty_trees(
    index,
    train_labels_orig: np.ndarray,
    baseline_dist: float,
    baseline_lsv_mean: np.ndarray,
    k: int,
    embedding_dim: int,
    out_path: Path,
) -> None:
    """Serialize the UQ index payload used by calibration and full-library application."""
    payload = {
        "index_bytes": serialize_index(index),
        "train_labels_orig": train_labels_orig,
        "baseline_dist": baseline_dist,
        "baseline_lsv_mean": baseline_lsv_mean,
        "k": k,
        "embedding_dim": embedding_dim,
        "target_cols": TARGET_COLS,
    }
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)


def load_uncertainty_payload(payload_path: Path) -> dict:
    """Load the canonical serialized UQ payload."""
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["index"] = deserialize_index(payload["index_bytes"])
    return payload


def load_lsv_thresholds(threshold_path: Path) -> dict:
    """Load the canonical threshold JSON payload."""
    with threshold_path.open() as handle:
        return json.load(handle)
