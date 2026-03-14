"""
Generate Figure 4, Figure 5, and Table S3 for model comparison.

Figure 4: test-set R^2 heatmap across the retained models.
Figure 5: parity plots of the selected ALIGNN model on the test split.
Table S3: task-specific R^2, MAE, and MAPE values exported as CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.figures.annotation_layout import (  # noqa: E402
    build_corner_annotation_candidates,
    choose_annotation_anchor,
    choose_common_annotation_anchor,
)
from src.figures.data_loader import (  # noqa: E402
    MODEL_ORDER,
    TASK_LIST,
    build_model_metrics_long,
    compute_task_metrics,
    load_alignn_predictions,
)
from src.figures.style import (  # noqa: E402
    BODY_FONT_SIZE,
    DOUBLE_COL_INCH,
    LABEL_FONT_SIZE,
    MODEL_COLORS,
    NATURE_COLORS,
    MODEL_MARKERS,
    TASK_LABELS,
    TASK_UNITS,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    compute_panel_grid_layout,
    save_figure,
    set_publication_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PANEL_ORDER = [
    ["AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa", "QstCH4"],
    ["AdsN2_10kPa", "AdsN2_100kPa", "AdsN2_1000kPa", "QstN2"],
]

ALIGNN_CMAP = LinearSegmentedColormap.from_list(
    "nature_magenta_cyan",
    [
        NATURE_COLORS["magenta"],
        NATURE_COLORS["cyan"],
    ],
)


def build_table_s3(metrics_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format metrics to a wide Table S3 layout."""
    rows: list[dict[str, float | str]] = []
    for task in TASK_LIST:
        row: dict[str, float | str] = {"Target": task}
        for model_name in MODEL_ORDER:
            metric_row = metrics_long[
                (metrics_long["Model"] == model_name)
                & (metrics_long["Target"] == task)
            ].iloc[0]
            row[f"{model_name}_R2"] = metric_row["R2"]
            row[f"{model_name}_MAE"] = metric_row["MAE"]
            row[f"{model_name}_MAPE"] = metric_row["MAPE"]
        rows.append(row)

    mean_row: dict[str, float | str] = {"Target": "Mean"}
    for model_name in MODEL_ORDER:
        subset = metrics_long[metrics_long["Model"] == model_name]
        mean_row[f"{model_name}_R2"] = subset["R2"].mean()
        mean_row[f"{model_name}_MAE"] = subset["MAE"].mean()
        mean_row[f"{model_name}_MAPE"] = subset["MAPE"].mean()
    rows.append(mean_row)

    return pd.DataFrame(rows)


def export_table_s3(csv_path: Path, table_s3: pd.DataFrame) -> None:
    """Export the wide-format Table S3 CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table_s3.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved: {csv_path}")


def plot_figure4(output_dir: Path, metrics_long: pd.DataFrame) -> None:
    """Plot Figure 4: test-set R^2 heatmap."""
    set_publication_style()

    r2_matrix = np.zeros((len(MODEL_ORDER), len(TASK_LIST) + 1))
    for i, model_name in enumerate(MODEL_ORDER):
        subset = metrics_long[metrics_long["Model"] == model_name]
        r2_values = [
            subset.loc[subset["Target"] == task, "R2"].iloc[0]
            for task in TASK_LIST
        ]
        r2_matrix[i, :-1] = r2_values
        r2_matrix[i, -1] = np.mean(r2_values)

    col_labels = [TASK_LABELS[task] for task in TASK_LIST] + ["Mean"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_INCH, 3.1))
    image = ax.imshow(
        r2_matrix,
        cmap=ALIGNN_CMAP,
        aspect="auto",
        vmin=0.72,
        vmax=0.98,
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(["CGCNN", "MFT", "ALIGNN"], fontsize=LABEL_FONT_SIZE)

    for i in range(r2_matrix.shape[0]):
        for j in range(r2_matrix.shape[1]):
            value = r2_matrix[i, j]
            fontweight = "bold" if j == r2_matrix.shape[1] - 1 else "normal"
            text_color = "black"
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=TICK_FONT_SIZE,
                fontweight=fontweight,
                color=text_color,
            )

    ax.axvline(x=r2_matrix.shape[1] - 1.5, color="white", linewidth=1.5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(r"$R^2$", fontsize=LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title(
        r"Test-set $R^2$ Comparison",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=8,
    )

    save_figure(fig, "Figure4", output_dir)
    plt.close(fig)


def plot_figure5(output_dir: Path) -> None:
    """Plot Figure 5: ALIGNN test-set parity plots."""
    layout = compute_panel_grid_layout(nrows=2, ncols=4, figure_width_inch=DOUBLE_COL_INCH)
    set_publication_style()
    df = load_alignn_predictions(split="test")
    annotation_candidates = build_corner_annotation_candidates(
        panel_width_inch=layout.panel_width,
        panel_height_inch=layout.panel_height,
        font_size_pt=layout.annotation_font,
        n_lines=3,
        max_line_chars=12,
    )
    panel_data = []
    for row in PANEL_ORDER:
        for task in row:
            y_true = df[f"{task}_true"].to_numpy()
            y_pred = df[f"{task}_pred"].to_numpy()
            lower = min(y_true.min(), y_pred.min())
            upper = max(y_true.max(), y_pred.max())
            margin = (upper - lower) * 0.06
            panel_data.append((y_true, y_pred, (lower - margin, upper + margin)))
    common_anchor_name = choose_common_annotation_anchor(
        panel_data,
        candidates=annotation_candidates,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(layout.figure_width, layout.figure_height),
    )

    for row in range(2):
        for col in range(4):
            task = PANEL_ORDER[row][col]
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
            margin = (upper - lower) * 0.06
            limits = [lower - margin, upper + margin]
            ax.plot(limits, limits, linestyle="--", linewidth=0.5, color="black", alpha=0.6)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal", adjustable="box")

            anchor = choose_annotation_anchor(
                y_true,
                y_pred,
                limits=tuple(limits),
                candidates=annotation_candidates,
                preferred_name=common_anchor_name,
            )

            ax.text(
                anchor["x"],
                anchor["y"],
                (
                    f"$R^2$ = {metrics['R2']:.3f}\n"
                    f"MAE = {metrics['MAE']:.3f}\n"
                    f"MAPE = {metrics['MAPE']:.3f}"
                ),
                transform=ax.transAxes,
                va=anchor["va"],
                ha=anchor["ha"],
                fontsize=layout.annotation_font,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            )

            ax.set_title(
                TASK_LABELS[task],
                fontsize=layout.title_font,
                fontweight="bold",
                pad=5,
            )
            if row == 1:
                ax.set_xlabel(f"GCMC ({TASK_UNITS[task]})", fontsize=layout.body_font)
            if col == 0:
                ax.set_ylabel(f"Predicted ({TASK_UNITS[task]})", fontsize=layout.body_font)
            ax.tick_params(labelsize=layout.tick_font)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        bottom=layout.bottom,
        top=layout.top,
        hspace=layout.hspace,
        wspace=layout.wspace,
    )
    save_figure(fig, "Figure5_alignn_parity", output_dir)
    plt.close(fig)


def generate_assets(output_dir: Path, table_csv: Path) -> None:
    """Generate Figure 4, Figure 5, and Table S3 CSV."""
    metrics_long = build_model_metrics_long()
    table_s3 = build_table_s3(metrics_long)
    plot_figure4(output_dir, metrics_long)
    plot_figure5(output_dir)
    export_table_s3(table_csv, table_s3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
        help="Directory for Figure 4 and Figure 5 outputs.",
    )
    parser.add_argument(
        "--table_csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "summary" / "Table_S3_model_metrics.csv",
        help="CSV path for Table S3.",
    )
    args = parser.parse_args()

    generate_assets(args.output_dir, args.table_csv)
    print("Done: Figure 4, Figure 5, and Table S3 CSV.")


if __name__ == "__main__":
    main()
