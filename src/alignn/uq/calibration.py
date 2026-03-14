"""Calibration utilities for deployment-based latent-space uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.alignn.common.constants import DEFAULT_LSV_PERCENTILES, MIN_SPEARMAN_RHO, TARGET_COLS
from src.alignn.uq.core import calibrate_uq, compute_baseline_dist, compute_lsv


def compute_training_baseline(train_emb: np.ndarray, train_labels: np.ndarray, index, k: int) -> tuple[float, np.ndarray]:
    """Compute baseline distance and mean raw training LSV per target."""
    baseline_dist = compute_baseline_dist(index, train_emb, k=k)
    lsv_train_raw = compute_lsv(train_emb, train_labels, index, baseline_dist, k=k)
    baseline_lsv_mean = lsv_train_raw.mean(axis=0).astype(np.float32)
    return baseline_dist, baseline_lsv_mean


def compute_normalized_split_lsv(
    split_data: dict[str, dict],
    train_labels: np.ndarray,
    index,
    baseline_dist: float,
    baseline_lsv_mean: np.ndarray,
    k: int,
) -> dict[str, dict]:
    """Attach normalized per-target LSV arrays to val/test split payloads."""
    enriched = {}
    for split_name, payload in split_data.items():
        enriched[split_name] = dict(payload)
        enriched[split_name]["lsv"] = compute_lsv(
            payload["features"],
            train_labels,
            index,
            baseline_dist,
            k=k,
            baseline_lsv_mean=baseline_lsv_mean,
        )
    return enriched


def combined_val_test(enriched_splits: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return combined val+test arrays for features, predictions, truths, and LSV."""
    features = np.vstack([enriched_splits["val"]["features"], enriched_splits["test"]["features"]])
    preds = np.vstack([enriched_splits["val"]["preds"], enriched_splits["test"]["preds"]])
    truths = np.vstack([enriched_splits["val"]["truths"], enriched_splits["test"]["truths"]])
    lsv = np.vstack([enriched_splits["val"]["lsv"], enriched_splits["test"]["lsv"]])
    return features, preds, truths, lsv


def run_k_sweep(
    query_emb: np.ndarray,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    index,
    preds: np.ndarray,
    truths: np.ndarray,
    k_values: list[int],
) -> dict:
    """Compute per-target Spearman correlations across a list of k values."""
    from scipy.stats import spearmanr

    sweep_results: dict[int, dict[str, float | None]] = {}
    for k in k_values:
        baseline_dist = compute_baseline_dist(index, train_emb, k=k)
        lsv = compute_lsv(query_emb, train_labels, index, baseline_dist, k=k)
        target_rhos: dict[str, float | None] = {}
        for idx, col in enumerate(TARGET_COLS):
            abs_error = np.abs(preds[:, idx] - truths[:, idx])
            mask = np.isfinite(lsv[:, idx]) & np.isfinite(abs_error)
            if int(mask.sum()) < 10:
                target_rhos[col] = None
                continue
            rho, _ = spearmanr(lsv[mask, idx], abs_error[mask])
            target_rhos[col] = float(rho)
        sweep_results[k] = target_rhos
    return sweep_results


def run_sr_sweep(lsv_score: np.ndarray, truths: np.ndarray, preds: np.ndarray, percentiles: np.ndarray) -> dict:
    """Run the SR sweep used to choose the canonical LSV percentile cutoff."""
    mae_in, mae_out, sr, retention = [], [], [], []
    for pct in percentiles:
        threshold = np.percentile(lsv_score, pct)
        low_uq = lsv_score <= threshold
        high_uq = ~low_uq

        def _mae(mask: np.ndarray) -> float:
            return float(np.abs(truths[mask] - preds[mask]).mean()) if int(mask.sum()) >= 10 else float("nan")

        low_mae = _mae(low_uq)
        high_mae = _mae(high_uq)
        mae_in.append(low_mae)
        mae_out.append(high_mae)
        sr.append(high_mae / low_mae if np.isfinite(low_mae) and np.isfinite(high_mae) and low_mae > 0 else float("nan"))
        retention.append(float(low_uq.mean()))

    return {
        "percentiles": percentiles.tolist(),
        "mae_in": np.asarray(mae_in, dtype=float).tolist(),
        "mae_out": np.asarray(mae_out, dtype=float).tolist(),
        "sr": np.asarray(sr, dtype=float).tolist(),
        "retention": np.asarray(retention, dtype=float).tolist(),
    }


def assemble_threshold_payload(
    combined_lsv: np.ndarray,
    baseline_lsv_mean: np.ndarray,
    k: int,
    recommended_pct: int,
    sr_sweep: dict,
) -> dict:
    """Assemble the canonical threshold JSON payload."""
    composite = combined_lsv.mean(axis=1)
    threshold = float(np.percentile(composite, recommended_pct))
    retention = float((composite <= threshold).mean())
    per_target = {
        col: float(np.percentile(combined_lsv[:, idx], recommended_pct))
        for idx, col in enumerate(TARGET_COLS)
    }
    percentile_index = sr_sweep["percentiles"].index(recommended_pct)
    sr_at_pct = float(sr_sweep["sr"][percentile_index])

    return {
        "description": (
            "LSV_norm filtering thresholds calibrated on val+test. "
            "LSV_norm_t = LSV_t / mean(LSV_train_t), and the composite score is the mean over targets."
        ),
        "calibration_set": "val+test combined",
        "n_calibration": int(len(composite)),
        "k": k,
        "lsv_normalised": True,
        "percentile": recommended_pct,
        "composite_threshold": threshold,
        "composite_retain_fraction": retention,
        f"per_target_p{recommended_pct}_lsv_norm": per_target,
        "baseline_lsv_mean": {col: float(value) for col, value in zip(TARGET_COLS, baseline_lsv_mean)},
        "elbow_analysis": {
            "recommended_pct": recommended_pct,
            f"sr_at_p{recommended_pct}": sr_at_pct,
        },
        "sr_sweep": {
            str(int(pct)): {
                "mae_in": float(sr_sweep["mae_in"][idx]) if np.isfinite(sr_sweep["mae_in"][idx]) else None,
                "mae_out": float(sr_sweep["mae_out"][idx]) if np.isfinite(sr_sweep["mae_out"][idx]) else None,
                "sr": float(sr_sweep["sr"][idx]) if np.isfinite(sr_sweep["sr"][idx]) else None,
                "retention": float(sr_sweep["retention"][idx]),
            }
            for idx, pct in enumerate(sr_sweep["percentiles"])
        },
        "note": (
            f"Primary filter for library screening: flag MOFs where mean(LSV_norm_8targets) "
            f"> composite_threshold (= p{recommended_pct} of val+test)."
        ),
    }


def save_json(payload: dict, output_path: Path) -> None:
    """Write a JSON payload with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_calibration_summary(
    input_dir: Path,
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    test_emb: np.ndarray,
    baseline_dist: float,
    k: int,
    calib_results: dict,
) -> dict:
    """Assemble the canonical uq_calibration.json payload."""
    valid_rhos = [entry["rho"] for key, entry in calib_results.items() if key in TARGET_COLS and entry["rho"] is not None]
    return {
        "input_dir": str(input_dir),
        "k": k,
        "baseline_dist": float(baseline_dist),
        "n_train": int(len(train_emb)),
        "embedding_dim": int(train_emb.shape[1]),
        "n_val": int(len(val_emb)),
        "n_test": int(len(test_emb)),
        "calibration": {key: value for key, value in calib_results.items() if key != "_summary"},
        "n_pass_rho_threshold": int(calib_results["_summary"]["n_pass"]),
        "mean_rho": float(np.mean(valid_rhos)) if valid_rhos else None,
        "min_rho_threshold": MIN_SPEARMAN_RHO,
    }
