"""
Exp02 – Textural / geometric screening of MOFs for CBM upgrading.

Source: src/jupyter/2_textural_screening_cbm.ipynb

Steps
-----
1. Merge RAC + Zeo++ features from all batch directories.
2. Keep de-duplicated entries only (duplicate_pdd_deduplicated.txt).
3. Apply screening thresholds: PLD > 3 Å  AND  GSA > 100 m²/g.
4. Save the screened list and a distribution figure.

Outputs (normal mode)
----------------------
data/processed/RAC_and_zeo_features.csv
data/processed/textural_screened/textural_screened_list.txt
results/figures/exp02_pore_distribution.png
results/figures/exp02_source_distribution.png

Run
---
python src/experiments/exp02_textural_screening.py
python src/experiments/exp02_textural_screening.py --test
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

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIR = Path("/home/zhangsd/repos/MOF-HTS/data/processed/features")
DEDUP_TXT = Path("/home/zhangsd/repos/MOF-HTS/data/processed/dedup_cifs/duplicate_pdd_deduplicated.txt")

PLD_CUTOFF = 3.0    # Å  – pore limiting diameter
GSA_CUTOFF = 100.0  # m²/g – gravimetric surface area

# Known ATC-Cu duplicates to keep for benchmark comparisons
ATC_CU_DUPLICATES = [
    "ARC-DB12-BIMDIL_clean_repeat.cif",
    "ARC-DB12-BIMDIL_freeONLY_repeat.cif",
    "MOSAEC-IMAYUT_full_REPEAT.cif",
    "MOSAEC-IMAZAA_full_REPEAT.cif",
]


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def merge_features(output_csv: Path) -> pd.DataFrame:
    """Merge all batch RAC+Zeo feature CSVs and save the combined file."""
    dfs = []
    failed_samples: list = []

    for batch_dir in sorted(FEATURE_DIR.glob("batch_*")):
        feat_csv = batch_dir / "RAC_and_zeo_features.csv"
        if feat_csv.exists():
            dfs.append(pd.read_csv(feat_csv))
        failed_txt = batch_dir / "unsuccessful_featurizations.txt"
        if failed_txt.exists():
            failed_samples.extend(failed_txt.read_text().splitlines())

    df_des = pd.concat(dfs, ignore_index=True)
    df_des.to_csv(output_csv, index=False)
    print(f"Merged features: {df_des.shape[0]} rows  →  {output_csv}")
    print(f"Failed featurizations: {len(failed_samples)}")
    return df_des


def apply_dedup(df_des: pd.DataFrame) -> pd.DataFrame:
    """Keep only de-duplicated entries + benchmark duplicates."""
    if not DEDUP_TXT.exists():
        print(f"[WARN] Dedup list not found: {DEDUP_TXT}. Skipping dedup step.")
        return df_des

    with open(DEDUP_TXT) as f:
        dedup_list = f.read().splitlines()
    dedup_list.extend(ATC_CU_DUPLICATES)

    df_dedup = df_des[df_des["cif_file"].isin(dedup_list)]
    print(f"After dedup: {df_dedup.shape[0]} rows  (from {df_des.shape[0]})")
    return df_dedup


def screen_mofs(df_dedup: pd.DataFrame) -> pd.DataFrame:
    """Apply PLD and GSA thresholds."""
    mask = (df_dedup["Df"] > PLD_CUTOFF) & (df_dedup["GSA"] > GSA_CUTOFF)
    df_screened = df_dedup[mask].copy()
    print(
        f"Screened: {df_screened.shape[0]} / {df_dedup.shape[0]}  "
        f"(PLD > {PLD_CUTOFF} Å  AND  GSA > {GSA_CUTOFF} m²/g)"
    )
    return df_screened


def plot_pore_distribution(df_dedup: pd.DataFrame, fig_dir: Path) -> None:
    """Plot PLD and GSA distributions with screening thresholds."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    total = df_dedup.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: PLD ---
    sns.histplot(data=df_dedup["Df"], bins=100, color=NATURE_COLORS["cyan"],
                 alpha=0.5, edgecolor="black", linewidth=1.0, ax=ax1)
    ax1.axvline(PLD_CUTOFF, color=NATURE_COLORS["orange"], linestyle="--",
                linewidth=2.5, label=f"Cutoff = {PLD_CUTOFF} Å")
    pass_pld = (df_dedup["Df"] > PLD_CUTOFF).sum()
    ax1.set_xlabel("Pore Limiting Diameter (Å)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"(a) PLD Distribution\nPass: {pass_pld}/{total} ({pass_pld/total*100:.1f}%)",
        fontsize=14, fontweight="bold", loc="left",
    )
    ax1.legend(frameon=True, edgecolor="black", loc="upper right", framealpha=0.9, fontsize=12)
    apply_nature_axes(ax1)

    # --- Panel B: GSA ---
    sns.histplot(data=df_dedup["GSA"], bins=100, color=NATURE_COLORS["cyan"],
                 alpha=0.5, edgecolor="black", linewidth=1.0, ax=ax2)
    ax2.axvline(GSA_CUTOFF, color=NATURE_COLORS["orange"], linestyle="--",
                linewidth=2.5, label=f"Cutoff = {GSA_CUTOFF} m²/g")
    pass_gsa = (df_dedup["GSA"] > GSA_CUTOFF).sum()
    ax2.set_xlabel("Gravimetric Surface Area (m²/g)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"(b) GSA Distribution\nPass: {pass_gsa}/{total} ({pass_gsa/total*100:.1f}%)",
        fontsize=14, fontweight="bold", loc="left",
    )
    ax2.legend(frameon=True, edgecolor="black", loc="upper right", framealpha=0.9, fontsize=12)
    apply_nature_axes(ax2)

    fig.tight_layout()
    savefig(fig, fig_dir / "exp02_pore_distribution.png")


def plot_source_distribution(df_screened: pd.DataFrame, fig_dir: Path) -> None:
    """Bar chart: experimental vs hypothetical MOF counts after screening."""
    import matplotlib.pyplot as plt

    exp_mask = df_screened["cif_file"].str.match(r"^(ARC-DB12|ARC-DB14|CoRE|MOSAEC)")
    n_exp = exp_mask.sum()
    n_hypo = (~exp_mask).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Experimental", "Hypothetical"], [n_exp, n_hypo],
                  width=0.3, color=[NATURE_COLORS["blue"], NATURE_COLORS["orange"]],
                  edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 20,
                str(int(h)), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("Screened MOF Source Distribution", fontsize=13, fontweight="bold", loc="left")
    ax.tick_params(axis="x", labelsize=12)
    apply_nature_axes(ax)
    ax.grid(False)
    fig.tight_layout()
    savefig(fig, fig_dir / "exp02_source_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp02: Textural screening of MOFs.")
    add_test_arg(parser)
    args = parser.parse_args()

    setup_matplotlib()

    # Input paths always point to production data (same in normal and test mode)
    prod_feat_csv = REPO_ROOT / "data" / "processed" / "RAC_and_zeo_features.csv"

    # Output paths route to results/test_run/** in test mode
    screened_dir = resolve_data_dir(args.test, "processed/textural_screened")
    feat_out_csv = resolve_data_dir(args.test, "processed") / "RAC_and_zeo_features.csv"
    fig_dir = resolve_output_dir(args.test, "figures")

    # 1. Load feature CSV from production dir; merge from batches only if it doesn't exist
    if prod_feat_csv.exists():
        print(f"Loading existing feature CSV: {prod_feat_csv}")
        df_des = pd.read_csv(prod_feat_csv)
    else:
        df_des = merge_features(feat_out_csv)

    # 2. De-duplicate
    df_dedup = apply_dedup(df_des)

    # 3. Screen
    df_screened = screen_mofs(df_dedup)

    # 4. Save screened list
    screened_list_path = screened_dir / "textural_screened_list.txt"
    with open(screened_list_path, "w") as f:
        f.write("\n".join(df_screened["cif_file"].tolist()))
    print(f"Screened list saved → {screened_list_path}")

    # 5. Figures
    plot_pore_distribution(df_dedup, fig_dir)
    plot_source_distribution(df_screened, fig_dir)

    if args.test:
        print("[TEST MODE] All outputs in results/test_run/")


if __name__ == "__main__":
    main()
