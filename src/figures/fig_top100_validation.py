"""
Validated-MOF parity figure based on the 186-MOF dual-track GCMC validation.

Layout: one combined 4 x 4 figure.
Rows 1-2 = PSA candidates (top-50 exp + top-50 hypo = 100 MOFs)
Rows 3-4 = VSA candidates (top-50 exp + top-50 hypo = 100 MOFs)
Columns = three uptake tasks + one heat-of-adsorption task per gas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.data_loader import compute_task_metrics
from src.figures.style import (
    DOUBLE_COL_INCH,
    MODEL_COLORS,
    MODEL_MARKERS,
    TASK_LABELS,
    TASK_UNITS,
    compute_panel_grid_layout,
    save_figure,
    set_emphasized_title,
    set_publication_style,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
# New 186-MOF dual-track validation data
GCMC_COMPARE_CSV = MODEL_DIR / "bkt_candidates_new" / "gcmc_vs_ml_comparison.csv"
# PSA/VSA splits derived from top-50 exp + top-50 hypo per process
TOP100_SPLIT_CSV = {
    "top_100_psa": [
        MODEL_DIR / "top_candidates" / "exp_top50_psa.csv",
        MODEL_DIR / "top_candidates" / "hypo_top50_psa.csv",
    ],
    "top_100_vsa": [
        MODEL_DIR / "top_candidates" / "exp_top50_vsa.csv",
        MODEL_DIR / "top_candidates" / "hypo_top50_vsa.csv",
    ],
}

PANEL_ORDER = [
    ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa", "QstCH4"],
    ["AdsN2_10kPa", "AdsN2_100kPa", "AdsN2_1000kPa", "QstN2"],
]

TASK_COLUMN_MAP = {
    "AdsCH4_10kPa": ("gcmc_AdsCH4_10kPa", "AdsCH4_10kPa"),
    "AdsCH4_100kPa": ("gcmc_AdsCH4_100kPa", "AdsCH4_100kPa"),
    "AdsCH4_1000kPa": ("gcmc_AdsCH4_1000kPa", "AdsCH4_1000kPa"),
    "AdsN2_10kPa": ("gcmc_AdsN2_10kPa", "AdsN2_10kPa"),
    "AdsN2_100kPa": ("gcmc_AdsN2_100kPa", "AdsN2_100kPa"),
    "AdsN2_1000kPa": ("gcmc_AdsN2_1000kPa", "AdsN2_1000kPa"),
    "QstCH4": ("QstCH4_gcmc", "QstCH4"),
    "QstN2": ("QstN2_gcmc", "QstN2"),
}


def load_top100_validation_predictions(split: str) -> pd.DataFrame:
    """Load 186-MOF ML-vs-GCMC validation data for a given process split."""
    if split not in TOP100_SPLIT_CSV:
        raise ValueError(f"Unknown split: {split}")

    compare_df = pd.read_csv(GCMC_COMPARE_CSV)
    # Concatenate exp_top50 + hypo_top50 for this process
    split_dfs = [pd.read_csv(p, usecols=["mof_id"]) for p in TOP100_SPLIT_CSV[split]]
    top_df = pd.concat(split_dfs, ignore_index=True).drop_duplicates(subset="mof_id")
    compare_df["mof_id"] = compare_df["mof_id"].astype(str)
    top_df["mof_id"] = top_df["mof_id"].astype(str)
    merged = top_df.merge(compare_df, on="mof_id", how="left", validate="one_to_one")

    required_cols = []
    for true_col, pred_col in TASK_COLUMN_MAP.values():
        required_cols.extend([true_col, pred_col])
    if merged[required_cols].isna().any().any():
        missing = merged.loc[merged[required_cols].isna().any(axis=1), "mof_id"].tolist()
        raise ValueError(
            f"Missing GCMC validation rows for {split}: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    records = {"CifId": merged["mof_id"].to_numpy()}
    for task, (true_col, pred_col) in TASK_COLUMN_MAP.items():
        records[f"{task}_true"] = merged[true_col].to_numpy()
        records[f"{task}_pred"] = merged[pred_col].to_numpy()
    return pd.DataFrame(records).set_index("CifId")


def plot_figure9(output_dir: Path) -> None:
    """Generate the combined 4x4 Top-100 validation figure."""
    set_publication_style()
    layout = compute_panel_grid_layout(
        nrows=4,
        ncols=4,
        figure_width_inch=DOUBLE_COL_INCH,
        gap_ratio_x=0.20,
        gap_ratio_y=0.28,
        panel_aspect=0.94,
    )

    fig, axes = plt.subplots(4, 4, figsize=(layout.figure_width, layout.figure_height))
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )

    psa_df = load_top100_validation_predictions("top_100_psa")
    vsa_df = load_top100_validation_predictions("top_100_vsa")
    panel_specs = [
        ("(PSA Elites)", psa_df, PANEL_ORDER[0][0]),
        ("(PSA Elites)", psa_df, PANEL_ORDER[0][1]),
        ("(PSA Elites)", psa_df, PANEL_ORDER[0][2]),
        ("(PSA Elites)", psa_df, PANEL_ORDER[0][3]),
        ("(PSA Elites)", psa_df, PANEL_ORDER[1][0]),
        ("(PSA Elites)", psa_df, PANEL_ORDER[1][1]),
        ("(PSA Elites)", psa_df, PANEL_ORDER[1][2]),
        ("(PSA Elites)", psa_df, PANEL_ORDER[1][3]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[0][0]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[0][1]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[0][2]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[0][3]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[1][0]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[1][1]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[1][2]),
        ("(VSA Elites)", vsa_df, PANEL_ORDER[1][3]),
    ]

    for idx, (split_suffix, df, task) in enumerate(panel_specs):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        y_true = df[f"{task}_true"].to_numpy()
        y_pred = df[f"{task}_pred"].to_numpy()
        metrics = compute_task_metrics(df, task)

        ax.scatter(
            y_true,
            y_pred,
            s=layout.marker_area,
            alpha=0.35,
            c=MODEL_COLORS["ALIGNN"],
            marker=MODEL_MARKERS["ALIGNN"],
            linewidths=0,
            rasterized=True,
        )

        lower = min(y_true.min(), y_pred.min())
        upper = max(y_true.max(), y_pred.max())
        margin = (upper - lower) * 0.06 if upper > lower else 0.1
        limits = [lower - margin, upper + margin]
        ax.plot(limits, limits, linestyle="--", linewidth=0.5, color="black", alpha=0.6)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")

        ax.text(
            0.05,
            0.95,
            f"MAPE = {metrics['MAPE']:.3f}\n"
            f"MAE = {metrics['MAE']:.3f}\n"
            f"$R^2$ = {metrics['R2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=layout.annotation_font,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )

        set_emphasized_title(
            ax,
            f"{TASK_LABELS[task]} {split_suffix}",
            loc="left",
            fontsize=layout.body_font,
            pad=5,
        )

        if row == 3:
            ax.set_xlabel(f"GCMC ({TASK_UNITS[task]})", fontsize=layout.body_font)
        if col == 0:
            ax.set_ylabel(f"Predicted ({TASK_UNITS[task]})", fontsize=layout.body_font)
        ax.tick_params(labelsize=layout.tick_font)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_figure(fig, "Figure9_validated_186", output_dir, tight_layout=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="manuscript/figures")
    args = parser.parse_args()
    out = Path(args.output_dir)
    plot_figure9(out)
    print("Done: combined 186-MOF PSA/VSA validation figure (Figure 9).")


if __name__ == "__main__":
    main()
