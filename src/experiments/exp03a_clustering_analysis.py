"""
Exp03a – Clustering analysis, UMAP visualization, and stratified sampling.

Source: src/jupyter/3_clustering_analysis.ipynb

Steps
-----
1. Load screened MOF list and features.
2. K-Means clustering (optimal k via elbow + Davies-Bouldin index).
3. UMAP dimensionality reduction.
4. Stratified sampling → training / validation / test splits.
5. Copy / symlink sampled CIFs.

Outputs (normal mode)
----------------------
data/processed/textural_screened/textural_screened_clustered_with_umap.csv
data/processed/stratified_datasets/{train,val,test}_set.csv
data/processed/stratified_datasets/cifs/   (symlinks)
results/figures/exp03a_clustering_umap.png
results/figures/exp03a_dataset_split.png

Run
---
python src/experiments/exp03a_clustering_analysis.py
python src/experiments/exp03a_clustering_analysis.py --test
"""
import argparse
import os
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
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INTEGRATED_CIFS = Path("/home/zhangsd/repos/MOF-HTS/data/processed/integrated_cifs")
OPTIMAL_K = 22
TRAIN_SIZE = 20_000
VAL_SIZE   = 1_000
TEST_SIZE  = 1_000
RANDOM_STATE = 42

FEATURE_COLS = [
    "Di", "Df", "Dif", "rho", "VPOV", "GPOV", "POAV_vol_frac", "GPOAV",
    "Di_dist", "Df_dist", "Dif_dist", "GSA", "VSA",
    "lc-chi-0-all", "lc-chi-1-all", "lc-chi-2-all", "lc-chi-3-all",
    "lc-Z-0-all", "lc-Z-1-all", "lc-Z-2-all", "lc-Z-3-all",
    "lc-I-0-all", "lc-I-1-all", "lc-I-2-all", "lc-I-3-all",
    "lc-T-0-all", "lc-T-1-all", "lc-T-2-all", "lc-T-3-all",
    "lc-S-0-all", "lc-S-1-all", "lc-S-2-all", "lc-S-3-all",
    "mc-chi-0-all", "mc-chi-1-all", "mc-chi-2-all", "mc-chi-3-all",
    "mc-Z-0-all", "mc-Z-1-all", "mc-Z-2-all", "mc-Z-3-all",
    "mc-I-0-all", "mc-I-1-all", "mc-I-2-all", "mc-I-3-all",
    "mc-T-0-all", "mc-T-1-all", "mc-T-2-all", "mc-T-3-all",
    "mc-S-0-all", "mc-S-1-all", "mc-S-2-all", "mc-S-3-all",
    "f-lig-chi-0", "f-lig-chi-1", "f-lig-chi-2", "f-lig-chi-3",
    "f-lig-Z-0", "f-lig-Z-1", "f-lig-Z-2", "f-lig-Z-3",
    "f-lig-I-0", "f-lig-I-1", "f-lig-I-2", "f-lig-I-3",
    "f-lig-T-0", "f-lig-T-1", "f-lig-T-2", "f-lig-T-3",
    "f-lig-S-0", "f-lig-S-1", "f-lig-S-2", "f-lig-S-3",
    "func-chi-0-all", "func-chi-1-all", "func-chi-2-all", "func-chi-3-all",
    "func-Z-0-all", "func-Z-1-all", "func-Z-2-all", "func-Z-3-all",
    "func-I-0-all", "func-I-1-all", "func-I-2-all", "func-I-3-all",
    "func-T-0-all", "func-T-1-all", "func-T-2-all", "func-T-3-all",
    "func-S-0-all", "func-S-1-all", "func-S-2-all", "func-S-3-all",
    "PONAV", "GPONAV",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_clustering(X_scaled: np.ndarray, k_range, sample_size: int = 50000):
    """Compute inertia and Davies-Bouldin scores; mirrors notebook evaluate_clustering()."""
    if X_scaled.shape[0] > sample_size:
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
        X_s = X_scaled[idx]
    else:
        X_s = X_scaled

    inertias, db_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbls = km.fit_predict(X_s)
        inertias.append(km.inertia_)
        if k > 1:
            db_scores.append(davies_bouldin_score(X_s, lbls))
        else:
            db_scores.append(float("inf"))
    return inertias, db_scores


def create_extended_colormap(n_colors: int):
    """Return a ListedColormap with at least *n_colors* distinct entries."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if n_colors <= 10:
        return plt.cm.tab10
    if n_colors <= 20:
        return plt.cm.tab20
    colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    if n_colors > 20:
        colors += list(plt.cm.Set3(np.linspace(0, 1, min(12, n_colors - 20))))
    if n_colors > 32:
        colors += list(plt.cm.Paired(np.linspace(0, 1, min(12, n_colors - 32))))
    return mcolors.ListedColormap(colors[:n_colors])


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def load_features(feat_csv: Path, screened_list_txt: Path) -> pd.DataFrame:
    """Load screened features, clean inf/NaN row-wise, return clean DataFrame.

    Mirrors notebook exactly:
      - Filter rows by screened list via ``cif_file`` column
      - Replace inf → NaN, then drop rows with any NaN (not columns)
      - Feature columns = df.columns[2:] (skip name + cif_file identifiers)
    """
    with open(screened_list_txt) as f:
        screened = f.read().splitlines()

    df = pd.read_csv(feat_csv)
    df = df[df["cif_file"].isin(screened)].copy()

    # Row-wise NaN/inf cleaning – mirrors notebook's X.replace(...).dropna()
    X = df.iloc[:, 2:].replace([np.inf, -np.inf], np.nan)
    X_clean = X.dropna()
    df_clean = df.loc[X_clean.index].copy().reset_index(drop=True)

    print(f"Loaded {len(df_clean)} screened MOFs with features "
          f"({len(df) - len(df_clean)} rows dropped due to NaN/inf).")
    return df_clean


def cluster_mofs(df: pd.DataFrame):
    """Run K-Means; return (labels, X_scaled, feature_cols_used, km_model).

    Uses df.columns[2:] as feature columns to mirror notebook exactly.
    Scaling is applied here and shared with evaluate_clustering caller.
    """
    # Feature columns = all columns after the two identifier columns (name, cif_file)
    feature_cols = list(df.columns[2:])
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Running K-Means with k={OPTIMAL_K}, features={len(feature_cols)} …")
    km = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"Cluster sizes: min={np.bincount(labels).min()}, max={np.bincount(labels).max()}")
    return labels, X_scaled, np.array(feature_cols), km


def compute_umap(X_scaled: np.ndarray) -> np.ndarray:
    """Fit UMAP and return 2-D embedding."""
    try:
        import umap.umap_ as umap
    except ImportError:
        import umap

    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="euclidean", random_state=RANDOM_STATE, verbose=True,
    )
    print("Computing UMAP …")
    return reducer.fit_transform(X_scaled)


def stratified_sampling(df_clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Proportional stratified sampling by cluster → (train, val, test)."""
    np.random.seed(RANDOM_STATE)
    cluster_col = "Cluster"
    total = len(df_clustered)
    cluster_counts = df_clustered[cluster_col].value_counts().sort_index()

    train_parts, val_parts, test_parts = [], [], []
    for cid, csize in cluster_counts.items():
        prop = csize / total
        n_tr = max(1, int(TRAIN_SIZE * prop))
        n_va = max(1, int(VAL_SIZE * prop))
        n_te = max(1, int(TEST_SIZE * prop))
        # Clip to available
        n_tr = min(n_tr, csize)
        n_va = min(n_va, csize - n_tr)
        n_te = min(n_te, csize - n_tr - n_va)
        chunk = df_clustered[df_clustered[cluster_col] == cid].sample(
            frac=1, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        train_parts.append(chunk.iloc[:n_tr])
        val_parts.append(chunk.iloc[n_tr : n_tr + n_va])
        test_parts.append(chunk.iloc[n_tr + n_va : n_tr + n_va + n_te])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df   = pd.concat(val_parts,   ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df  = pd.concat(test_parts,  ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Stratified split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Leakage check
    tr_ids, va_ids, te_ids = set(train_df["CifId"]), set(val_df["CifId"]), set(test_df["CifId"])
    ol_tv = tr_ids & va_ids; ol_tt = tr_ids & te_ids; ol_vt = va_ids & te_ids
    if ol_tv or ol_tt or ol_vt:
        print(f"[WARN] Data leakage: tr-val={len(ol_tv)}, tr-test={len(ol_tt)}, val-test={len(ol_vt)}")
    else:
        print("[OK] No data leakage between splits.")

    return train_df, val_df, test_df


def plot_clustering_and_split(df_clustered: pd.DataFrame, fig_dir: Path) -> None:
    """Two-panel figure: cluster distribution + dataset split on UMAP."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    X_2d = df_clustered[["UMAP1", "UMAP2"]].values
    labels = df_clustered["Cluster"].values
    unique_clusters = sorted(np.unique(labels))
    n_clusters = len(unique_clusters)
    cmap = create_extended_colormap(n_clusters)
    alpha = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: clusters ---
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap,
                alpha=alpha, s=4, edgecolors="black", linewidth=0.1)
    ax1.set_title(f"(a) Clustering Results (k={n_clusters})",
                  fontsize=14, fontweight="bold", pad=12, loc="left")
    ax1.set_xlabel("UMAP Component 1", fontsize=12, fontweight="bold")
    ax1.set_ylabel("UMAP Component 2", fontsize=12, fontweight="bold")
    apply_nature_axes(ax1)
    legend_els = []
    for cid in unique_clusters:
        cnt = (labels == cid).sum()
        pct = cnt / len(labels) * 100
        color = cmap(cid / max(n_clusters - 1, 1))
        legend_els.append(Patch(facecolor=color, edgecolor="black", linewidth=0.5,
                                label=f"Cluster {cid+1:2d}: {cnt} ({pct:.1f}%)", alpha=alpha))
    ax1.legend(handles=legend_els, title="Cluster Statistics",
               loc="center left", bbox_to_anchor=(1.01, 0.5),
               frameon=True, fancybox=False, fontsize=8,
               title_fontsize=10, edgecolor="black", framealpha=0.9).get_title().set_fontweight("bold")

    # --- Right: sampled vs all ---
    # Determine sample membership (train+val+test will be set to "Sampled")
    sampled_mask = pd.Series(False, index=df_clustered.index)
    if "Split" in df_clustered.columns:
        sampled_mask = df_clustered["Split"].notna()

    colors_bg = ["#CCCCCC"] * len(X_2d)
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c="#CCCCCC",
                alpha=0.4, s=3, edgecolors="none", label=f"All ({len(X_2d):,})")
    if sampled_mask.any():
        s_2d = X_2d[sampled_mask]
        ax2.scatter(s_2d[:, 0], s_2d[:, 1], c="#E41A1C",
                    alpha=0.8, s=4, edgecolors="black", linewidth=0.1,
                    label=f"Sampled ({sampled_mask.sum():,})")
    ax2.set_title("(b) Sampled Data Distribution",
                  fontsize=14, fontweight="bold", pad=12, loc="left")
    ax2.set_xlabel("UMAP Component 1", fontsize=12, fontweight="bold")
    ax2.set_ylabel("UMAP Component 2", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    apply_nature_axes(ax2)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp03a_clustering_umap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp03a: Clustering analysis and stratified sampling.")
    add_test_arg(parser)
    args = parser.parse_args()

    setup_matplotlib()

    # Input paths: always point to production data/ (never rerouted to test_run/)
    prod_processed = REPO_ROOT / "data" / "processed"
    feat_csv      = prod_processed / "RAC_and_zeo_features.csv"
    screened_list = prod_processed / "textural_screened" / "textural_screened_list.txt"

    # Output paths: routed to test_run/ when --test
    screened_dir = resolve_data_dir(args.test, "processed/textural_screened")
    strat_dir    = resolve_data_dir(args.test, "processed/stratified_datasets")
    cif_out_dir  = strat_dir / "cifs"
    cif_out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir      = resolve_output_dir(args.test, "figures")

    clustered_csv = screened_dir / "textural_screened_clustered_with_umap.csv"

    # ----- Load or recompute -----
    if clustered_csv.exists():
        print(f"Loading existing clustered data: {clustered_csv}")
        df_clustered = pd.read_csv(clustered_csv)
    else:
        df = load_features(feat_csv, screened_list)

        # K optimization – mirrors notebook: np.random.seed(42) outside, then evaluate
        feature_cols = list(df.columns[2:])
        X_for_eval = StandardScaler().fit_transform(df[feature_cols].values)
        k_range = range(2, 51, 2)
        print(f"Running K-Means optimization for k in {list(k_range)} …")
        np.random.seed(RANDOM_STATE)   # mirror notebook's seed call before evaluate
        inertias, db_scores = evaluate_clustering(X_for_eval, k_range)
        k_list = list(k_range)
        best_db_k = k_list[int(np.argmin(db_scores))]
        print(f"Best k by Davies-Bouldin: {best_db_k}  |  Using OPTIMAL_K={OPTIMAL_K}")
        for k, iner, db in zip(k_list, inertias, db_scores):
            print(f"  k={k:2d}  inertia={iner:.1f}  DB={db:.4f}")

        # Save k-optimization figure
        import matplotlib.pyplot as plt
        fig_k, ax_k = plt.subplots(1, 2, figsize=(12, 5))
        ax_k[0].plot(k_list, inertias, "o-", color=NATURE_COLORS["blue"])
        ax_k[0].set_xlabel("k"); ax_k[0].set_ylabel("Inertia"); ax_k[0].set_title("Elbow Method")
        apply_nature_axes(ax_k[0])
        db_k_plot = [k for k in k_list if k > 1]
        db_vals_plot = [db for k, db in zip(k_list, db_scores) if k > 1]
        ax_k[1].plot(db_k_plot, db_vals_plot, "o-", color=NATURE_COLORS["orange"])
        ax_k[1].axvline(OPTIMAL_K, color="red", linestyle="--", label=f"k={OPTIMAL_K}")
        ax_k[1].set_xlabel("k"); ax_k[1].set_ylabel("Davies-Bouldin Score")
        ax_k[1].set_title("Davies-Bouldin Index"); ax_k[1].legend()
        apply_nature_axes(ax_k[1])
        fig_k.tight_layout()
        savefig(fig_k, fig_dir / "exp03a_k_optimization.png")

        labels, X_scaled, _, km_model = cluster_mofs(df)
        X_2d = compute_umap(X_scaled)

        # Compute IsCentroid: nearest sample to each cluster centroid
        from sklearn.metrics.pairwise import euclidean_distances
        centroids = km_model.cluster_centers_
        is_centroid = np.zeros(len(labels), dtype=bool)
        for cid in range(OPTIMAL_K):
            mask = np.where(labels == cid)[0]
            if len(mask) == 0:
                continue
            dists = euclidean_distances(X_scaled[mask], centroids[cid].reshape(1, -1)).ravel()
            is_centroid[mask[np.argmin(dists)]] = True

        cif_ids = df["name"] if "name" in df.columns else df["cif_file"]
        df_clustered = pd.DataFrame({
            "CifId":       cif_ids.values,
            "Cluster":     labels,
            "IsCentroid":  is_centroid,
            "UMAP1":       X_2d[:, 0],
            "UMAP2":       X_2d[:, 1],
        })
        df_clustered.to_csv(clustered_csv, index=False)
        print(f"Clustered data saved → {clustered_csv}")

    # ----- Stratified sampling -----
    train_df, val_df, test_df = stratified_sampling(df_clustered)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_csv = strat_dir / f"{split_name}_set.csv"
        split_df.to_csv(out_csv, index=False)
        print(f"Saved {split_name} set → {out_csv}")

    # ----- Symlink CIFs for training -----
    all_sampled = pd.concat([train_df, val_df, test_df], ignore_index=True)
    n_linked = 0
    for name in all_sampled["CifId"].tolist():
        src = INTEGRATED_CIFS / (name.strip() + ".cif")
        dst = cif_out_dir / (name.strip() + ".cif")
        if not src.exists():
            continue
        if not dst.exists():
            try:
                os.symlink(src, dst)
                n_linked += 1
            except OSError:
                pass
    print(f"CIF symlinks created: {n_linked}  →  {cif_out_dir}")

    # ----- Merge Split labels for visualization -----
    split_labels = pd.concat([
        train_df[["CifId"]].assign(Split="train"),
        val_df[["CifId"]].assign(Split="val"),
        test_df[["CifId"]].assign(Split="test"),
    ], ignore_index=True)
    df_plot = df_clustered.merge(split_labels, on="CifId", how="left")

    # ----- Figures -----
    plot_clustering_and_split(df_plot, fig_dir)

    if args.test:
        print("[TEST MODE] All outputs in results/test_run/")


if __name__ == "__main__":
    main()
