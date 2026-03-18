"""
iast_selectivity_vs_pressure.py — Plot IAST selectivity vs pressure for ATC-Cu.

Compares our GCMC-fitted DSLF with Niu et al. 2019 experimental DSLF,
across multiple compositions and the full pressure range.

Output:
  results/alignn/model_ep150/literature_validation/iast_alpha_vs_pressure.png
  results/alignn/model_ep150/literature_validation/iast_alpha_vs_pressure.csv

Usage:
    python src/alignn/validation/iast_selectivity_vs_pressure.py
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# DSLF formula: q = qs1*b1*P^n1/(1+b1*P^n1) + qs2*b2*P^n2/(1+b2*P^n2)
# Spreading pressure: π = (qs1/n1)*ln(1+b1*P^n1) + (qs2/n2)*ln(1+b2*P^n2)
# =====================================================================

# --- Our GCMC-fitted DSLF (P in bar, n bounded [0.8, 3.0]) ---
OUR_CH4 = dict(qs1=1.74, b1=2.960, n1=1.763,
               qs2=3.64, b2=0.745, n2=0.800)
OUR_N2 = dict(qs1=0.24, b1=1.995, n1=1.055,
              qs2=4.56, b2=0.131, n2=1.153)

# --- Niu et al. 2019 experimental DSLF (P in kPa) ---
# Original params: t1, t2 (their convention). Our n = 1/t.
NIU_CH4 = dict(qs1=2.4132725, b1=0.0016885, n1=1/0.7587481,
               qs2=2.5963261, b2=0.0262649, n2=1/1.0010558)
NIU_N2 = dict(qs1=1.4898984, b1=0.0062025, n1=1/0.9480648,
              qs2=0.5483496, b2=0.0054842, n2=1/1.2880086)


def dslf_loading(P, qs1, b1, n1, qs2, b2, n2):
    Pn1 = np.float_power(max(P, 1e-15), n1)
    Pn2 = np.float_power(max(P, 1e-15), n2)
    return qs1 * b1 * Pn1 / (1 + b1 * Pn1) + qs2 * b2 * Pn2 / (1 + b2 * Pn2)


def dslf_sp(P, qs1, b1, n1, qs2, b2, n2):
    Pn1 = np.float_power(max(P, 1e-15), n1)
    Pn2 = np.float_power(max(P, 1e-15), n2)
    return (qs1 / n1) * np.log(1 + b1 * Pn1) + (qs2 / n2) * np.log(1 + b2 * Pn2)


def iast_binary(p1, p2, y, P_total):
    y1, y2 = y

    def obj(x1):
        if x1 <= 0 or x1 >= 1:
            return 1e10
        return dslf_sp(P_total * y1 / x1, **p1) - dslf_sp(P_total * y2 / (1 - x1), **p2)

    eps = 1e-10
    try:
        xx = np.linspace(eps, 1 - eps, 500)
        ff = np.array([obj(x) for x in xx])
        sc = np.where(np.diff(np.sign(ff)))[0]
        if len(sc) == 0:
            return np.nan, np.nan, np.nan
        x1 = brentq(obj, xx[sc[0]], xx[sc[0] + 1], xtol=1e-12)
    except Exception:
        return np.nan, np.nan, np.nan

    x2 = 1 - x1
    q1p = dslf_loading(P_total * y1 / x1, **p1)
    q2p = dslf_loading(P_total * y2 / x2, **p2)
    if q1p <= 0 or q2p <= 0:
        return np.nan, np.nan, np.nan
    qt = 1 / (x1 / q1p + x2 / q2p)
    q1, q2 = x1 * qt, x2 * qt
    alpha = (q1 / q2) * (y2 / y1) if q2 > 0 else np.nan
    return alpha, q1, q2


def main():
    REPO = "/home/zhangsd/repos/CBM-MOF"
    out_dir = os.path.join(REPO, "results/alignn/model_ep150/literature_validation")
    os.makedirs(out_dir, exist_ok=True)

    # Pressure grid (log-spaced, in kPa for Niu, bar for ours)
    P_kpa = np.logspace(np.log10(1), np.log10(1500), 80)  # 1 kPa to 1500 kPa
    P_bar = P_kpa / 100.0  # convert to bar

    compositions = [
        (0.15, "15:85"),
        (0.20, "20:80"),
        (0.30, "30:70"),
        (0.50, "50:50"),
    ]

    # Collect all data
    rows = []
    for y_ch4, label in compositions:
        y = (y_ch4, 1 - y_ch4)
        for i, (Pk, Pb) in enumerate(zip(P_kpa, P_bar)):
            a_niu, q1_niu, q2_niu = iast_binary(NIU_CH4, NIU_N2, y, Pk)
            a_our, q1_our, q2_our = iast_binary(OUR_CH4, OUR_N2, y, Pb)
            rows.append(dict(
                P_kPa=Pk, P_bar=Pb, y_CH4=y_ch4, composition=label,
                alpha_Niu=a_niu, q_CH4_Niu=q1_niu, q_N2_Niu=q2_niu,
                alpha_Ours=a_our, q_CH4_Ours=q1_our, q_N2_Ours=q2_our,
            ))

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(out_dir, "iast_alpha_vs_pressure.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Data saved: {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    colors = {
        "15:85": "#9467bd",
        "20:80": "#2ca02c",
        "30:70": "#17becf",
        "50:50": "#1f77b4",
    }

    # Panel (a): Our GCMC-fitted DSLF
    ax = axes[0]
    for y_ch4, label in compositions:
        sub = df[df["composition"] == label]
        ax.plot(sub["P_kPa"], sub["alpha_Ours"], "-", color=colors[label],
                linewidth=1.8, label=f"CH$_4$:N$_2$ = {label}")
    ax.set_xscale("log")
    ax.set_xlabel("Total Pressure (kPa)", fontsize=12)
    ax.set_ylabel(r"IAST Selectivity $\alpha$(CH$_4$/N$_2$)", fontsize=12)
    ax.set_title("(a) This work (GCMC-fitted DSLF)", fontsize=12)
    ax.set_xlim(1, 1500)
    ax.set_ylim(0, max(df["alpha_Ours"].max() * 1.1, 25))
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    # Panel (b): Niu et al. experimental DSLF
    ax = axes[1]
    for y_ch4, label in compositions:
        sub = df[df["composition"] == label]
        # Niu's params only valid up to ~110 kPa (experimental range)
        # Plot full range but mark extrapolation
        mask_valid = sub["P_kPa"] <= 120
        mask_extrap = sub["P_kPa"] > 120
        ax.plot(sub.loc[mask_valid, "P_kPa"], sub.loc[mask_valid, "alpha_Niu"],
                "-", color=colors[label], linewidth=1.8, label=f"CH$_4$:N$_2$ = {label}")
        ax.plot(sub.loc[mask_extrap, "P_kPa"], sub.loc[mask_extrap, "alpha_Niu"],
                "--", color=colors[label], linewidth=1.0, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Total Pressure (kPa)", fontsize=12)
    ax.set_ylabel(r"IAST Selectivity $\alpha$(CH$_4$/N$_2$)", fontsize=12)
    ax.set_title("(b) Niu et al. 2019 (Exp-fitted DSLF)", fontsize=12)
    ax.set_xlim(1, 1500)
    # Cap y-axis to show the physically meaningful range
    niu_valid = df[(df["P_kPa"] <= 120) & df["alpha_Niu"].notna()]
    ax.set_ylim(0, max(niu_valid["alpha_Niu"].max() * 1.3, 15))
    ax.axvline(x=110, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(115, ax.get_ylim()[1] * 0.9, "exp. limit", fontsize=8,
            color="gray", ha="left", va="top")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    png_path = os.path.join(out_dir, "iast_alpha_vs_pressure.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {png_path}")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: α at key pressures (50:50)")
    print(f"{'='*70}")
    sub50 = df[df["composition"] == "50:50"]
    key_P = [5, 10, 50, 100, 500, 1000]
    print(f"{'P (kPa)':>10}  {'Niu':>10}  {'Ours':>10}")
    print(f"{'-'*10}  {'-'*10}  {'-'*10}")
    for Pk in key_P:
        row = sub50.iloc[(sub50["P_kPa"] - Pk).abs().argsort()[:1]]
        print(f"{Pk:>10}  {row['alpha_Niu'].values[0]:>10.2f}  {row['alpha_Ours'].values[0]:>10.2f}")

    print(f"\nSUMMARY: α at key pressures (20:80)")
    sub20 = df[df["composition"] == "20:80"]
    print(f"{'P (kPa)':>10}  {'Niu':>10}  {'Ours':>10}")
    print(f"{'-'*10}  {'-'*10}  {'-'*10}")
    for Pk in key_P:
        row = sub20.iloc[(sub20["P_kPa"] - Pk).abs().argsort()[:1]]
        print(f"{Pk:>10}  {row['alpha_Niu'].values[0]:>10.2f}  {row['alpha_Ours'].values[0]:>10.2f}")


if __name__ == "__main__":
    main()
