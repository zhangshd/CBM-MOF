"""
Validated-MOF parity figure based on the 186-MOF dual-track GCMC validation.

Two layouts available:
  plot_figure9_4x4()  -- Legacy 4x4 grid: Rows 1-2 = PSA, Rows 3-4 = VSA (uniform color)
  plot_figure9_2x4()  -- New 2x4 grid: all 186 unique MOFs, exp vs hypo markers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.data_loader import compute_task_metrics, r2_score
from src.figures.style import (
    DOUBLE_COL_INCH,
    LEGEND_FONT_SIZE,
    MODEL_COLORS,
    MODEL_MARKERS,
    NATURE_COLORS,
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
GCMC_COMPARE_CSV = MODEL_DIR / "bkt_candidates" / "gcmc_vs_ml_comparison.csv"
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

# Experimental MOF ID prefixes
_EXP_PREFIXES = ("CoRE-", "MOSAEC-", "ARC-DB12-", "ARC-DB14-")

# Visual identity for exp/hypo distinction
EXP_HYPO_COLORS = {
    "exp": NATURE_COLORS["blue"],
    "hypo": NATURE_COLORS["orange"],
}
EXP_HYPO_MARKERS = {
    "exp": "^",     # triangle
    "hypo": "o",    # circle
}
EXP_HYPO_LABELS = {
    "exp": "Experimental",
    "hypo": "Hypothetical",
}


def _classify_exp_hypo(mof_ids: pd.Index | pd.Series) -> np.ndarray:
    """Return boolean array: True for experimental MOFs, False for hypothetical."""
    ids = mof_ids.astype(str)
    return np.array([mid.startswith(_EXP_PREFIXES) for mid in ids])


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


def load_all_186_validation() -> pd.DataFrame:
    """Load all 186 unique MOFs by merging PSA and VSA top-100 splits.

    Returns a DataFrame indexed by CifId with ``{task}_true`` / ``{task}_pred``
    columns and a boolean ``is_exp`` column.
    """
    compare_df = pd.read_csv(GCMC_COMPARE_CSV)
    # Collect all unique MOF IDs across both PSA and VSA splits
    all_split_dfs = []
    for split_csvs in TOP100_SPLIT_CSV.values():
        for p in split_csvs:
            all_split_dfs.append(pd.read_csv(p, usecols=["mof_id"]))
    all_ids = pd.concat(all_split_dfs, ignore_index=True).drop_duplicates(subset="mof_id")

    compare_df["mof_id"] = compare_df["mof_id"].astype(str)
    all_ids["mof_id"] = all_ids["mof_id"].astype(str)
    merged = all_ids.merge(compare_df, on="mof_id", how="left", validate="one_to_one")

    required_cols = []
    for true_col, pred_col in TASK_COLUMN_MAP.values():
        required_cols.extend([true_col, pred_col])
    if merged[required_cols].isna().any().any():
        missing = merged.loc[merged[required_cols].isna().any(axis=1), "mof_id"].tolist()
        raise ValueError(
            f"Missing GCMC validation rows: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    records = {"CifId": merged["mof_id"].to_numpy()}
    for task, (true_col, pred_col) in TASK_COLUMN_MAP.items():
        records[f"{task}_true"] = merged[true_col].to_numpy()
        records[f"{task}_pred"] = merged[pred_col].to_numpy()
    df = pd.DataFrame(records).set_index("CifId")
    df["is_exp"] = _classify_exp_hypo(df.index)
    return df


# ---------------------------------------------------------------------------
# Legacy 4x4 layout (kept for backward compatibility)
# ---------------------------------------------------------------------------


def plot_figure9_4x4(output_dir: Path) -> None:
    """Generate the combined 4x4 Top-100 validation figure (legacy layout)."""
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

    save_figure(fig, "Figure09_validated_186", output_dir, tight_layout=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# New 2x4 layout: exp vs hypo on merged 186 MOFs
# ---------------------------------------------------------------------------


def plot_figure9_2x4(output_dir: Path) -> None:
    """Generate a 2x4 parity figure with exp/hypo marker distinction.

    Row 1: CH4 properties (3 uptakes + Qst)
    Row 2: N2 properties (3 uptakes + Qst)
    Each panel plots all 186 unique MOFs, coloring experimental and hypothetical
    MOFs differently with separate R^2 annotations.
    """
    set_publication_style()
    layout = compute_panel_grid_layout(
        nrows=2,
        ncols=4,
        figure_width_inch=DOUBLE_COL_INCH,
        gap_ratio_x=0.20,
        gap_ratio_y=0.32,
        panel_aspect=0.94,
    )

    fig, axes = plt.subplots(2, 4, figsize=(layout.figure_width, layout.figure_height))
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )

    df = load_all_186_validation()
    is_exp = df["is_exp"].to_numpy()
    is_hypo = ~is_exp
    n_exp = int(is_exp.sum())
    n_hypo = int(is_hypo.sum())

    # Flat panel list: row0 = CH4 tasks, row1 = N2 tasks
    panel_tasks = PANEL_ORDER[0] + PANEL_ORDER[1]

    for idx, task in enumerate(panel_tasks):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        y_true = df[f"{task}_true"].to_numpy()
        y_pred = df[f"{task}_pred"].to_numpy()

        # Plot all 186 MOFs with uniform style
        ax.scatter(
            y_true,
            y_pred,
            s=layout.marker_area * 1.4,
            alpha=0.45,
            c=MODEL_COLORS["ALIGNN"],
            marker="s",
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        # Diagonal parity line
        lower = min(y_true.min(), y_pred.min())
        upper = max(y_true.max(), y_pred.max())
        margin = (upper - lower) * 0.06 if upper > lower else 0.1
        limits = [lower - margin, upper + margin]
        ax.plot(limits, limits, linestyle="--", linewidth=0.5, color="black", alpha=0.6, zorder=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")

        # Compute overall metrics only
        metrics_all = compute_task_metrics(df, task)

        # Annotation: overall R^2, MAE, MAPE
        ax.text(
            0.05,
            0.95,
            f"$R^2$ = {metrics_all['R2']:.3f}\n"
            f"MAE = {metrics_all['MAE']:.3f}\n"
            f"MAPE = {metrics_all['MAPE']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=layout.annotation_font,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )

        set_emphasized_title(
            ax,
            TASK_LABELS[task],
            loc="left",
            fontsize=layout.body_font,
            pad=5,
        )

        if row == 1:
            ax.set_xlabel(f"GCMC ({TASK_UNITS[task]})", fontsize=layout.body_font)
        if col == 0:
            ax.set_ylabel(f"Predicted ({TASK_UNITS[task]})", fontsize=layout.body_font)
        ax.tick_params(labelsize=layout.tick_font)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_figure(fig, "Figure09_validated_186_exp_hypo", output_dir, tight_layout=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Alias: default plot_figure9 now points to the new 2x4 layout
# ---------------------------------------------------------------------------
plot_figure9 = plot_figure9_2x4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GCMC-vs-ML parity plots for the 186 validated MOFs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "alignn" / "model_ep150" / "figures"),
    )
    parser.add_argument(
        "--layout",
        choices=["2x4", "4x4"],
        default="2x4",
        help="Figure layout: '2x4' (exp/hypo, default) or '4x4' (legacy PSA/VSA)",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)

    if args.layout == "4x4":
        plot_figure9_4x4(out)
        print("Done: legacy 4x4 PSA/VSA validation figure (Figure 9).")
    else:
        plot_figure9_2x4(out)
        print("Done: 2x4 exp/hypo validation figure (Figure 9).")


if __name__ == "__main__":
    main()
