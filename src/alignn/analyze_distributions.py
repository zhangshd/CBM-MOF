"""
analyze_distributions.py
========================
Diagnostic script to quantify distribution distortion caused by different symlog τ choices.

For each of the 8 prediction targets, compares:
  - Raw distribution: skewness, kurtosis, range
  - After symlog(τ=1e-4): the old global baseline
  - After per-column τ*: the new optimized config

Also generates a 3-row × 8-col grid of histograms saved to
results/alignn/dist_analysis/

Usage:
    conda activate mofmthnn
    python src/alignn/analyze_distributions.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
REPO_ROOT  = Path("/home/zhangsd/repos/CBM-MOF")
LABEL_CSV  = REPO_ROOT / "src/ml/data/round2/RAC_and_zeo_features_with_id_prop.csv"
SPLIT_DIR  = REPO_ROOT / "data/processed/stratified_datasets"
OUT_DIR    = REPO_ROOT / "results/alignn/dist_analysis"

# ──────────────────────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────────────────────
ALL_TARGETS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
    "QstCH4",       "QstN2",
]

# ──────────────────────────────────────────────────────────────
# Per-column τ* config (from CBM-MOF-symlog v2 Brent search)
# ──────────────────────────────────────────────────────────────
TRANSFORM_CONFIG = {
    "AdsCH4_10kPa":   {"type": "symlog", "tau": 1e-6},
    "AdsCH4_100kPa":  {"type": "symlog", "tau": 1e-6},
    "AdsCH4_1000kPa": {"type": "symlog", "tau": 0.177},
    "AdsN2_10kPa":    {"type": "symlog", "tau": 1e-6},
    "AdsN2_100kPa":   {"type": "symlog", "tau": 0.013},
    "AdsN2_1000kPa":  {"type": "raw"},
    "QstCH4":         {"type": "raw"},
    "QstN2":          {"type": "raw"},
}

OLD_THRESH = 1e-4  # global τ baseline


# ──────────────────────────────────────────────────────────────
# Transform helpers
# ──────────────────────────────────────────────────────────────
def symlog(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.log10(1.0 + np.abs(x) / tau)


def apply_transform(x: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["type"] == "symlog":
        return symlog(x, cfg["tau"])
    return x.copy()   # raw


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training set only (to match what the model sees)
    train_names = pd.read_csv(SPLIT_DIR / "train_set.csv")["name"].tolist()
    df_all = pd.read_csv(LABEL_CSV)
    df = df_all[df_all["MofName"].isin(train_names)].copy()
    print(f"Training set rows: {len(df)}")

    rows = []
    print(f"\n{'Target':25s}  {'raw_skew':>9}  {'old(1e-4)_skew':>14}  {'optau_skew':>10}  {'type':>6}  {'tau*':>8}")
    print("─" * 78)
    for col in ALL_TARGETS:
        x = df[col].values
        cfg = TRANSFORM_CONFIG[col]

        x_old = symlog(x, OLD_THRESH) if col in [
            "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
            "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
        ] else x
        x_new = apply_transform(x, cfg)

        sk_raw = float(skew(x))
        sk_old = float(skew(x_old))
        sk_new = float(skew(x_new))
        ku_raw = float(kurtosis(x))

        tau_str = str(cfg.get("tau", "—"))
        print(f"  {col:23s}  {sk_raw:9.3f}  {sk_old:14.3f}  {sk_new:10.3f}  "
              f"{cfg['type']:>6}  {tau_str:>8}")

        rows.append({
            "column": col, "type": cfg["type"], "tau": cfg.get("tau"),
            "raw_skew": round(sk_raw, 4), "raw_kurtosis": round(ku_raw, 4),
            "old_tau1e4_skew": round(sk_old, 4), "optau_skew": round(sk_new, 4),
            "raw_min": float(x.min()), "raw_max": float(x.max()),
            "raw_mean": float(x.mean()), "raw_std": float(x.std()),
        })

    # Save stats table
    stats_df = pd.DataFrame(rows)
    stats_csv = OUT_DIR / "distribution_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nStats saved: {stats_csv}")

    # ── Histogram grid: raw | old symlog | new symlog ──────────
    n = len(ALL_TARGETS)
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))

    for j, col in enumerate(ALL_TARGETS):
        x = df[col].values
        cfg = TRANSFORM_CONFIG[col]
        x_old = symlog(x, OLD_THRESH) if cfg["type"] == "symlog" else x
        x_new = apply_transform(x, cfg)

        sk_r = float(skew(x))
        sk_o = float(skew(x_old))
        sk_n = float(skew(x_new))

        tau_label = f", τ={cfg['tau']}" if cfg["type"] == "symlog" else ""
        optau_label = f"optau({cfg['type']}{tau_label})"

        for row_idx, (arr, label, color, sk) in enumerate([
            (x,     "raw",              "#4e79a7", sk_r),
            (x_old, "symlog(τ=1e-4)",  "#e15759", sk_o),
            (x_new, optau_label,        "#59a14f", sk_n),
        ]):
            ax = axes[row_idx, j]
            ax.hist(arr, bins=80, color=color, alpha=0.7)
            if row_idx == 0:
                ax.set_title(col, fontsize=7, fontweight="bold")
            ax.set_xlabel(f"skew={sk:.2f}", fontsize=6)
            ax.tick_params(labelsize=5)
            if j == 0:
                ax.set_ylabel(label, fontsize=6)

    plt.tight_layout()
    hist_path = OUT_DIR / "distribution_comparison.png"
    fig.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram grid saved: {hist_path}")

    # ── Highlight worst distortions ────────────────────────────
    print("\n⚠️  Worst distortions (|old_skew| - |raw_skew| > 0.5):")
    for r in rows:
        delta = abs(r["old_tau1e4_skew"]) - abs(r["raw_skew"])
        if delta > 0.5:
            print(f"  {r['column']:25s}  raw={r['raw_skew']:+.3f}  "
                  f"old={r['old_tau1e4_skew']:+.3f}  new={r['optau_skew']:+.3f}  "
                  f"Δ={delta:+.3f}")

    print(f"\n✓ Analysis complete. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
