#!/usr/bin/env python3
"""Compare BKT / GCMC / IAST CH4 uptake at feed pressure → volumetric capacity → t₅₀.

BKT cumulative adsorption = total CH4 adsorbed at feed pressure (single-pressure,
NOT working capacity).  The correct comparison is with GCMC and IAST mixture
CH4 uptake at the SAME feed pressure and composition (CH4:N2 = 20:80).

PSA feed = 10 bar → gcmc_AdsCH4_1000kPa, q_CH4_IAST_PSA
VSA feed = 1 bar  → gcmc_AdsCH4_100kPa,  q_CH4_IAST_VSA
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BKT_DIR = Path("/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150/bkt_candidates")

# ── Load data ──
fig13 = pd.read_csv(BKT_DIR / "Figure13_selectivity_dumbbell_metrics.csv")
top20 = pd.read_csv(BKT_DIR / "top20_combined.csv")
iast  = pd.read_csv(BKT_DIR / "iast_selectivity.csv")

# Rename for consistent merge key
top20 = top20.rename(columns={"mof_id": "mof"})
iast  = iast.rename(columns={"MofName": "mof"})

# ATC-Cu GCMC mixture uptakes (from raspa3_parsed_results_0911.csv)
ATC_CU = {
    "mof": "CoRE-2020[Cu][pts]3[ASR]1",
    "gcmc_AdsCH4_1000kPa": 2.607836,  # PSA feed
    "gcmc_AdsCH4_100kPa":  0.831050,  # VSA feed
    "gcmc_AdsN2_1000kPa":  1.439863,
    "gcmc_AdsN2_100kPa":   0.491690,
}

# ATC-Cu IAST (from iast_selectivity.csv — check if present, else from memory)
atc_iast = iast[iast["mof"] == ATC_CU["mof"]]
if atc_iast.empty:
    # Values from previous analysis (IAST computed for ATC-Cu)
    ATC_CU["q_CH4_IAST_PSA"] = 2.651  # approximate
    ATC_CU["q_CH4_IAST_VSA"] = 0.908
else:
    ATC_CU["q_CH4_IAST_PSA"] = atc_iast.iloc[0]["q_CH4_IAST_PSA"]
    ATC_CU["q_CH4_IAST_VSA"] = atc_iast.iloc[0]["q_CH4_IAST_VSA"]

# ── Per-process analysis ──
PROCESS_CFG = {
    "PSA": {"gcmc_col": "gcmc_AdsCH4_1000kPa", "iast_col": "q_CH4_IAST_PSA",
            "gcmc_n2": "gcmc_AdsN2_1000kPa",   "iast_n2": "q_N2_IAST_PSA"},
    "VSA": {"gcmc_col": "gcmc_AdsCH4_100kPa",  "iast_col": "q_CH4_IAST_VSA",
            "gcmc_n2": "gcmc_AdsN2_100kPa",    "iast_n2": "q_N2_IAST_VSA"},
}

for process, cfg in PROCESS_CFG.items():
    proc = fig13[fig13["process"] == process].copy()

    # Merge GCMC feed-pressure uptake
    gcmc_cols = ["mof", cfg["gcmc_col"], cfg["gcmc_n2"]]
    merged = proc.merge(top20[gcmc_cols].drop_duplicates("mof"), on="mof", how="left")

    # Merge IAST feed-pressure uptake
    iast_cols = ["mof", cfg["iast_col"], cfg["iast_n2"]]
    iast_sub = iast[iast_cols].drop_duplicates("mof")
    merged = merged.merge(iast_sub, on="mof", how="left")

    # Fill ATC-Cu
    atc_mask = merged["is_benchmark"] == True
    merged.loc[atc_mask, cfg["gcmc_col"]] = ATC_CU[cfg["gcmc_col"]]
    merged.loc[atc_mask, cfg["iast_col"]] = ATC_CU[cfg["iast_col"]]

    # Compute volumetric capacities (mol/m³)
    merged["vol_BKT"]  = merged["q_CH4_mol_per_kg"] * merged["rho_s"]
    merged["vol_GCMC"] = merged[cfg["gcmc_col"]]     * merged["rho_s"]
    merged["vol_IAST"] = merged[cfg["iast_col"]]     * merged["rho_s"]

    valid = merged.dropna(subset=["vol_BKT", "vol_GCMC", "vol_IAST", "t50_min"])
    n = len(valid)

    print(f"\n{'='*70}")
    print(f"  {process}  (n={n} MOFs, feed = {'10 bar' if process=='PSA' else '1 bar'})")
    print(f"{'='*70}")

    # ── Correlations with t₅₀ ──
    metrics = [
        ("vol_BKT",  "vol(BKT)  = q_CH4_BKT  × ρ_s"),
        ("vol_GCMC", "vol(GCMC) = q_CH4_GCMC × ρ_s"),
        ("vol_IAST", "vol(IAST) = q_CH4_IAST × ρ_s"),
        ("q_CH4_mol_per_kg", "q_CH4_BKT  (gravimetric)"),
        (cfg["gcmc_col"],    "q_CH4_GCMC (gravimetric)"),
        (cfg["iast_col"],    "q_CH4_IAST (gravimetric)"),
    ]

    print(f"\n  {'Metric':<38} {'Pearson r':>10} {'Spearman ρ':>11}")
    print(f"  {'-'*38} {'-'*10} {'-'*11}")
    for col, label in metrics:
        rp, _ = stats.pearsonr(valid[col], valid["t50_min"])
        rs, _ = stats.spearmanr(valid[col], valid["t50_min"])
        print(f"  {label:<38} {rp:10.4f} {rs:11.4f}")

    # ── BKT vs GCMC vs IAST uptake comparison ──
    print(f"\n  Uptake ratios (feed-pressure CH4 adsorption):")
    r_bg = valid["q_CH4_mol_per_kg"] / valid[cfg["gcmc_col"]]
    r_bi = valid["q_CH4_mol_per_kg"] / valid[cfg["iast_col"]]
    r_ig = valid[cfg["iast_col"]]    / valid[cfg["gcmc_col"]]
    print(f"    q_BKT/q_GCMC:  mean={r_bg.mean():.4f}  std={r_bg.std():.4f}  "
          f"range=[{r_bg.min():.4f}, {r_bg.max():.4f}]")
    print(f"    q_BKT/q_IAST:  mean={r_bi.mean():.4f}  std={r_bi.std():.4f}  "
          f"range=[{r_bi.min():.4f}, {r_bi.max():.4f}]")
    print(f"    q_IAST/q_GCMC: mean={r_ig.mean():.4f}  std={r_ig.std():.4f}  "
          f"range=[{r_ig.min():.4f}, {r_ig.max():.4f}]")

    # ── Detailed table ──
    print(f"\n  {'MOF':<45} {'ρ_s':>6} {'q_BKT':>7} {'q_GCMC':>7} {'q_IAST':>7} "
          f"{'vBKT':>7} {'vGCMC':>7} {'vIAST':>7} {'t50':>7} {'bm':>2}")
    print(f"  {'-'*45} {'-'*6} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*2}")
    for _, row in valid.sort_values("t50_min", ascending=False).iterrows():
        mof_short = row["mof"][:44]
        bm = "★" if row.get("is_benchmark") else ""
        print(f"  {mof_short:<45} {row['rho_s']:6.0f} "
              f"{row['q_CH4_mol_per_kg']:7.3f} {row[cfg['gcmc_col']]:7.3f} "
              f"{row[cfg['iast_col']]:7.3f} "
              f"{row['vol_BKT']:7.0f} {row['vol_GCMC']:7.0f} {row['vol_IAST']:7.0f} "
              f"{row['t50_min']:7.1f} {bm:>2}")

# ── Final interpretation ──
print(f"\n{'='*70}")
print("  KEY QUESTION: Can GCMC (or IAST) volumetric uptake replace BKT for ranking?")
print(f"{'='*70}")
print("""
  Compare vol(GCMC) vs t₅₀ correlation with vol(BKT) vs t₅₀:
  - If vol(GCMC) ≈ vol(BKT) in ranking → BKT adds no ranking info
  - If vol(GCMC) diverges from vol(BKT) → isotherm fitting + IAST mixing
    introduce deviations that BKT captures but GCMC alone does not

  Also compare q_BKT/q_GCMC ratio:
  - If ratio ≈ 1 and low variance → BKT just reproduces GCMC equilibrium
  - If ratio ≠ 1 or high variance → BKT capacity differs from GCMC
    (due to Langmuir fit error, IAST mixing deviation, or kinetic effects)
""")
