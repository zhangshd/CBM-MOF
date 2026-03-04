"""
Exp06b – Summarise and compare training results across all models.

Source: src/jupyter/6.5_training_results.ipynb

Steps
-----
1. Load test-set predictions for CGCNN, CGCNN+ZEO, MOFTransformer, XGBoost.
2. Compute R², MAE, MAPE for each task × model combination.
3. Save comparison table as Excel file.
4. Plot parity scatter plots (8 panels, one per task) for XGBoost.

Outputs (normal mode)
----------------------
results/model_comparison_results.xlsx
results/figures/exp06b_model_comparison_table.csv
results/figures/exp06b_parity_{model}.png

Run
---
python src/experiments/exp06b_training_results.py
python src/experiments/exp06b_training_results.py --test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    REPO_ROOT,
    NATURE_COLORS,
    add_test_arg,
    apply_nature_axes,
    resolve_output_dir,
    savefig,
    setup_matplotlib,
)

import numpy as np
import pandas as pd
from sklearn import metrics as skm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ML_MODEL_DIR = REPO_ROOT / "results" / "ml_models" / "round2" / "RAC_and_zeo_features_with_id_prop"
CGCNN_DIR    = REPO_ROOT / "results" / "cgcnn_models" / "ads_qst_ch4_n2_org_seed42_att_cgcnn" / "version_0"
CGCNN_ZEO_DIR = REPO_ROOT / "results" / "cgcnn_models" / "ads_qst_ch4_n2_org_seed42_att_cgcnn" / "version_1"
MFT_DIR      = (
    REPO_ROOT / "results" / "moftransformer_models"
    / "ads_qst_ch4_n2_org_seed42_moftransformer_from_pmtransformer" / "version_8"
)

TASKS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa", "QstCH4",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",  "QstN2",
]

MODEL_DIRS = {
    "CGCNN":         CGCNN_DIR,
    "CGCNN_ZEO":     CGCNN_ZEO_DIR,
    "MOFTransformer": MFT_DIR,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":   skm.r2_score(y_true, y_pred),
        "MAE":  skm.mean_absolute_error(y_true, y_pred),
        "MAPE": skm.mean_absolute_percentage_error(y_true, y_pred),
    }


def load_gnn_preds(model_dir: Path, task: str) -> tuple[np.ndarray, np.ndarray] | None:
    csv_p = model_dir / f"test_results_{task}.csv"
    if not csv_p.exists():
        return None
    df = pd.read_csv(csv_p)
    return df["GroundTruth"].values, df["Predicted"].values


def load_ml_preds(task: str) -> tuple[np.ndarray, np.ndarray] | None:
    csv_p = ML_MODEL_DIR / task / "test_predicted_XGBRegressor.csv"
    if not csv_p.exists():
        return None
    df = pd.read_csv(csv_p)
    return df["GroundTruth"].values, df["Predicted"].values


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def plot_parity_xgb(fig_dir: Path) -> None:
    """8-panel parity plot for XGBoost predictions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        result = load_ml_preds(task)
        if result is None:
            axes[i].set_visible(False)
            continue
        y_true, y_pred = result
        m = compute_metrics(y_true, y_pred)
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=NATURE_COLORS["blue"],
                   edgecolors="black", linewidth=0.3)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
        ax.set_title(task, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("Ground Truth", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.text(0.05, 0.93, f"R²={m['R2']:.3f}\nMAE={m['MAE']:.3f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8))
        apply_nature_axes(ax)

    fig.suptitle("XGBoost Parity Plots (Test Set)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_dir / "exp06b_parity_xgboost.png")


def plot_r2_comparison(metric_df: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bar chart: R² by model and task."""
    import matplotlib.pyplot as plt

    model_names = metric_df.columns.get_level_values("Model").unique().tolist()
    x = np.arange(len(TASKS))
    width = 0.2
    colors = [NATURE_COLORS["blue"], NATURE_COLORS["orange"], NATURE_COLORS["green"], NATURE_COLORS["red"]]

    fig, ax = plt.subplots(figsize=(16, 6))
    for j, (model, color) in enumerate(zip(model_names, colors)):
        vals = metric_df[(model, "R2")].values if (model, "R2") in metric_df.columns else np.full(len(TASKS), np.nan)
        ax.bar(x + j * width - width * len(model_names) / 2, vals, width,
               label=model, color=color, edgecolor="black", linewidth=0.6, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("R²", fontsize=12, fontweight="bold")
    ax.set_title("Model R² Comparison across Tasks", fontsize=13, fontweight="bold", loc="left")
    ax.legend(fontsize=10, frameon=True)
    apply_nature_axes(ax)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    savefig(fig, fig_dir / "exp06b_r2_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp06b: Summarise training results.")
    add_test_arg(parser)
    args = parser.parse_args()

    setup_matplotlib()
    fig_dir = resolve_output_dir(args.test, "figures")
    results_dir = resolve_output_dir(args.test, "")  # results/ root

    # Build metrics table
    metric_result: dict = {}
    for task in TASKS:
        metric_result[task] = {}
        # GNN models
        for model_name, model_dir in MODEL_DIRS.items():
            result = load_gnn_preds(model_dir, task)
            if result:
                metric_result[task][model_name] = compute_metrics(*result)
            else:
                print(f"  [SKIP] {model_name} / {task}")
        # XGBoost
        result_ml = load_ml_preds(task)
        if result_ml:
            metric_result[task]["XGBRegressor"] = compute_metrics(*result_ml)

    # Build multi-level DataFrame
    rows = []
    for task_name, task_metrics in metric_result.items():
        row = {"Task": task_name}
        for model_name, m in task_metrics.items():
            for metric_name, val in m.items():
                row[(model_name, metric_name)] = val
        rows.append(row)

    metric_df = pd.DataFrame(rows).set_index("Task")
    metric_df.columns = pd.MultiIndex.from_tuples(metric_df.columns, names=["Model", "Metric"])

    # Save Excel (production only; in test mode save CSV instead)
    if not args.test:
        xlsx_path = results_dir / "model_comparison_results.xlsx"
        metric_df.to_excel(str(xlsx_path), float_format="%.4f", merge_cells=True)
        print(f"Excel saved → {xlsx_path}")

    csv_path = fig_dir / "exp06b_model_comparison_table.csv"
    metric_df.to_csv(csv_path, float_format="%.4f")
    print(f"CSV saved → {csv_path}")
    print(metric_df.to_string())

    # Figures
    plot_parity_xgb(fig_dir)
    plot_r2_comparison(metric_df, fig_dir)

    if args.test:
        print("[TEST MODE] All outputs in results/test_run/")


if __name__ == "__main__":
    main()
