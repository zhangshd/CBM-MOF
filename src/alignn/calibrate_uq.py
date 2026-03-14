"""
calibrate_uq.py
===============
Canonical deployment-based UQ calibration entrypoint.

This script is the only authoritative producer of:
  - uncertainty_trees.pkl
  - uq_calibration.json
  - uq_calibration.png
  - lsv_thresholds.json
  - k_sensitivity_sweep.json / .png
  - latent_space_pca_by_targets.png
  - lsv_norm_distribution.png
  - ALIGNN_ep150_LSV_sr_analysis.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.alignn.common.constants import DEFAULT_K_NEIGHBORS
from src.alignn.common.paths import REPO_ROOT, resolve_model_paths
from src.alignn.uq.calibration import (
    assemble_threshold_payload,
    build_calibration_summary,
    combined_val_test,
    compute_normalized_split_lsv,
    compute_training_baseline,
    run_k_sweep,
    run_sr_sweep,
    save_json,
)
from src.alignn.uq.core import build_faiss_index, save_uncertainty_trees, calibrate_uq
from src.alignn.uq.io import load_deployment_splits
from src.alignn.uq.plots import (
    plot_calibration,
    plot_distribution_panel,
    plot_k_sweep,
    plot_pca_by_targets,
    plot_sr_panel,
)


DEFAULT_MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Model-specific results dir containing deployment/ and uq/.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional override for deployment artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for uq output directory.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help="Number of nearest neighbors for LSV.",
    )
    parser.add_argument(
        "--recommended-pct",
        type=int,
        default=85,
        help="Canonical LSV percentile cutoff to serialize.",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA visualization.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir.is_absolute() else REPO_ROOT / args.model_dir
    paths = resolve_model_paths(model_dir)
    input_dir = args.input_dir or paths.deployment_dir
    output_dir = args.output_dir or paths.uq_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Task 1.1b — Calibrate UQ")
    print("=" * 60)
    print(f"  input_dir  : {input_dir}")
    print(f"  output_dir : {output_dir}")
    print(f"  k          : {args.k}")
    print(f"  percentile : {args.recommended_pct}")

    splits = load_deployment_splits(input_dir)
    train_emb = splits["train"]["features"]
    train_labels = splits["train"]["truths"]

    index = build_faiss_index(train_emb)
    baseline_dist, baseline_lsv_mean = compute_training_baseline(train_emb, train_labels, index, k=args.k)
    save_uncertainty_trees(
        index=index,
        train_labels_orig=train_labels,
        baseline_dist=baseline_dist,
        baseline_lsv_mean=baseline_lsv_mean,
        k=args.k,
        embedding_dim=int(train_emb.shape[1]),
        out_path=output_dir / "uncertainty_trees.pkl",
    )

    enriched = compute_normalized_split_lsv(
        split_data=splits,
        train_labels=train_labels,
        index=index,
        baseline_dist=baseline_dist,
        baseline_lsv_mean=baseline_lsv_mean,
        k=args.k,
    )
    combined_features, combined_preds, combined_truths, combined_lsv = combined_val_test(enriched)

    calib_results = calibrate_uq(combined_lsv, combined_preds, combined_truths)
    save_json(
        build_calibration_summary(
            input_dir=input_dir,
            train_emb=train_emb,
            val_emb=splits["val"]["features"],
            test_emb=splits["test"]["features"],
            baseline_dist=baseline_dist,
            k=args.k,
            calib_results=calib_results,
        ),
        output_dir / "uq_calibration.json",
    )

    k_sweep = run_k_sweep(
        query_emb=combined_features,
        train_emb=train_emb,
        train_labels=train_labels,
        index=index,
        preds=combined_preds,
        truths=combined_truths,
        k_values=[3, 5, 10, 20, 50],
    )
    save_json(
        {
            "k_sweep": {str(k): values for k, values in k_sweep.items()},
            "min_rho_threshold": calib_results["_summary"]["min_spearman_rho"],
        },
        output_dir / "k_sensitivity_sweep.json",
    )

    sr_sweep = run_sr_sweep(
        lsv_score=combined_lsv.mean(axis=1),
        truths=combined_truths,
        preds=combined_preds,
        percentiles=np.concatenate([[0], np.arange(5, 100, 5), [100]]),
    )
    threshold_payload = assemble_threshold_payload(
        combined_lsv=combined_lsv,
        baseline_lsv_mean=baseline_lsv_mean,
        k=args.k,
        recommended_pct=args.recommended_pct,
        sr_sweep=sr_sweep,
    )
    save_json(threshold_payload, output_dir / "lsv_thresholds.json")

    plot_calibration(combined_lsv, combined_preds, combined_truths, calib_results, output_dir / "uq_calibration.png")
    plot_k_sweep(k_sweep, output_dir / "k_sensitivity_sweep.png")
    plot_distribution_panel(
        enriched["train"]["lsv"],
        enriched["val"]["lsv"],
        enriched["test"]["lsv"],
        output_dir / "lsv_norm_distribution.png",
    )
    plot_sr_panel(
        sr_sweep=sr_sweep,
        threshold_value=float(threshold_payload["composite_threshold"]),
        recommended_pct=args.recommended_pct,
        out_path=output_dir / "ALIGNN_ep150_LSV_sr_analysis.png",
    )
    if not args.skip_pca:
        plot_pca_by_targets(
            all_features=np.vstack([splits["train"]["features"], splits["val"]["features"], splits["test"]["features"]]),
            all_truths=np.vstack([splits["train"]["truths"], splits["val"]["truths"], splits["test"]["truths"]]),
            out_path=output_dir / "latent_space_pca_by_targets.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
