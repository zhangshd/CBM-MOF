"""
Exp05 – Build training data for Round 2 (merge RASPA3 GCMC + Widom results).

Source: src/jupyter/5_make_training_data_round2.ipynb

Steps
-----
1. Load RASPA3 adsorption and RASPA2 Widom results.
2. Match to sample splits (train/val/test).
3. Save task-specific CSV files with normalised properties.
4. Plot label distributions and PCA/UMAP scatter.

Outputs (normal mode)
----------------------
data/processed/training_data/{task}/{train,val,test}_set.csv
results/figures/exp05_label_distributions.png
results/figures/exp05_pca_split.png

Run
---
python src/experiments/exp05_make_training_data.py
python src/experiments/exp05_make_training_data.py --test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    REPO_ROOT,
    NATURE_COLORS,
    add_test_arg,
    apply_nature_axes,
    resolve_data_dir,
    resolve_output_dir,
    savefig,
    setup_matplotlib,
)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RASPA3_CSV  = REPO_ROOT / "results" / "cbm_screening" / "raspa3_parsed_results_round2_0917.csv"
WIDOM_CSV   = REPO_ROOT / "results" / "cbm_screening" / "widom_results_round2_0917.csv"
SPLIT_DIR   = REPO_ROOT / "data" / "processed" / "stratified_datasets"
FEAT_CSV    = REPO_ROOT / "data" / "processed" / "RAC_and_zeo_features.csv"

TASKS = [
    "AdsCH4_10kPa",
    "AdsCH4_100kPa",
    "AdsCH4_1000kPa",
    "QstCH4",
    "AdsN2_10kPa",
    "AdsN2_100kPa",
    "AdsN2_1000kPa",
    "QstN2",
]

# Pressures are in bar (0.1 bar = 10 kPa, 1.0 bar = 100 kPa, 10.0 bar = 1000 kPa)
# Gas names as stored in RASPA3 CSV: "methane" / "N2"
# Widom Qst column: "AdsorptionHeat" (kJ/mol)
TASK_COL_MAP = {
    "AdsCH4_10kPa":   ("methane", 0.1,  "AbsLoading"),
    "AdsCH4_100kPa":  ("methane", 1.0,  "AbsLoading"),
    "AdsCH4_1000kPa": ("methane", 10.0, "AbsLoading"),
    "QstCH4":         ("methane", None, "AdsorptionHeat"),
    "AdsN2_10kPa":    ("N2",      0.1,  "AbsLoading"),
    "AdsN2_100kPa":   ("N2",      1.0,  "AbsLoading"),
    "AdsN2_1000kPa":  ("N2",      10.0, "AbsLoading"),
    "QstN2":          ("N2",      None, "AdsorptionHeat"),
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_raspa3(csv_path: Path) -> pd.DataFrame:
    """Load and pivot RASPA3 adsorption data into per-MOF wide format."""
    if not csv_path.exists():
        print(f"[WARN] RASPA3 CSV not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"RASPA3 raw rows: {len(df)}")

    records = []
    for mof, grp in df.groupby("MofName"):
        row = {"MofName": mof}
        for task, (gas, pressure, col) in TASK_COL_MAP.items():
            if "Qst" in task:
                continue
            sub = grp[(grp["GasName"] == gas) & (np.isclose(grp["Pressure[bar]"], pressure, rtol=0.01))]
            row[task] = sub[col].mean() if not sub.empty else np.nan
        records.append(row)

    return pd.DataFrame(records)


def load_widom(csv_path: Path) -> pd.DataFrame:
    """Load Widom insertion results and pivot Qst values."""
    if not csv_path.exists():
        print(f"[WARN] Widom CSV not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Widom raw rows: {len(df)}")

    qst_map = {"QstCH4": "methane", "QstN2": "N2"}
    records = []
    for mof, grp in df.groupby("MofName"):
        row = {"MofName": mof}
        for task, gas in qst_map.items():
            sub = grp[grp["GasName"] == gas]
            row[task] = sub["AdsorptionHeat"].mean() if not sub.empty else np.nan
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def plot_label_distributions(df_labeled: pd.DataFrame, fig_dir: Path) -> None:
    """8-panel histogram of property distributions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        if task not in df_labeled.columns:
            axes[i].set_visible(False)
            continue
        vals = df_labeled[task].dropna()
        axes[i].hist(vals, bins=80, color=NATURE_COLORS["blue"], alpha=0.7,
                     edgecolor="black", linewidth=0.4)
        axes[i].set_title(task, fontsize=11, fontweight="bold", loc="left")
        axes[i].set_xlabel("Value", fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)
        apply_nature_axes(axes[i])

    fig.suptitle("Label Distributions (Round 2 Training Data)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_dir / "exp05_label_distributions.png")


def plot_pca_split(df_all: pd.DataFrame, train_ids: set, val_ids: set, test_ids: set, fig_dir: Path) -> None:
    """PCA scatter coloured by split membership."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    feat_csv = FEAT_CSV
    if not feat_csv.exists():
        print("[SKIP] Feature CSV not found for PCA plot.")
        return

    df_feat = pd.read_csv(feat_csv)
    id_col = "name" if "name" in df_feat.columns else "cif_file"
    df_feat = df_feat.rename(columns={id_col: "MofName"})
    df_merged = df_all[["MofName"]].merge(df_feat, on="MofName", how="left")

    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    X = df_merged[numeric_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    colors = []
    for mid in df_merged["MofName"]:
        if mid in train_ids:
            colors.append(NATURE_COLORS["blue"])
        elif mid in val_ids:
            colors.append(NATURE_COLORS["orange"])
        elif mid in test_ids:
            colors.append(NATURE_COLORS["green"])
        else:
            colors.append("#CCCCCC")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, s=8, edgecolors="none")
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=NATURE_COLORS["blue"],   label=f"Train ({len(train_ids):,})"),
        Patch(facecolor=NATURE_COLORS["orange"], label=f"Val ({len(val_ids):,})"),
        Patch(facecolor=NATURE_COLORS["green"],  label=f"Test ({len(test_ids):,})"),
    ]
    ax.legend(handles=legend_els, fontsize=10, frameon=True)
    ax.set_xlabel("PC 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PC 2", fontsize=12, fontweight="bold")
    ax.set_title("PCA of Training / Validation / Test Split", fontsize=13, fontweight="bold", loc="left")
    apply_nature_axes(ax)
    fig.tight_layout()
    savefig(fig, fig_dir / "exp05_pca_split.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp05: Build Round 2 training data.")
    add_test_arg(parser)
    args = parser.parse_args()

    setup_matplotlib()

    data_out_dir = resolve_data_dir(args.test, "processed/training_data")
    fig_dir      = resolve_output_dir(args.test, "figures")

    # Load adsorption + Qst data
    df_ads   = load_raspa3(RASPA3_CSV)
    df_widom = load_widom(WIDOM_CSV)

    if df_ads.empty and df_widom.empty:
        print("[WARN] No simulation results found. Saving empty placeholder files.")
        df_labeled = pd.DataFrame(columns=["MofName"] + TASKS)
    else:
        if not df_ads.empty and not df_widom.empty:
            df_labeled = df_ads.merge(df_widom, on="MofName", how="outer")
        elif not df_ads.empty:
            df_labeled = df_ads
        else:
            df_labeled = df_widom

    print(f"Combined labeled data: {df_labeled.shape}")

    # Load splits
    train_ids, val_ids, test_ids = set(), set(), set()
    for split_name in ["train", "val", "test"]:
        csv_p = SPLIT_DIR / f"{split_name}_set.csv"
        if csv_p.exists():
            split_df = pd.read_csv(csv_p)
            # Column may be "name" or "CifId" depending on which script generated the split
            id_col = "CifId" if "CifId" in split_df.columns else "name"
            ids = split_df[id_col].tolist()
            if split_name == "train":
                train_ids = set(ids)
            elif split_name == "val":
                val_ids = set(ids)
            else:
                test_ids = set(ids)

    # Save per-task split CSVs
    for task in TASKS:
        task_dir = data_out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        if task not in df_labeled.columns:
            continue
        df_task = df_labeled[["MofName", task]].dropna()
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            split_df = df_task[df_task["MofName"].isin(ids)]
            out_path = task_dir / f"{split_name}_set.csv"
            split_df.to_csv(out_path, index=False)
        print(f"  Task {task}: {len(df_task)} labeled MOFs saved.")

    # Figures
    plot_label_distributions(df_labeled, fig_dir)
    plot_pca_split(df_labeled, train_ids, val_ids, test_ids, fig_dir)

    if args.test:
        print("[TEST MODE] All outputs in results/test_run/")


if __name__ == "__main__":
    main()
