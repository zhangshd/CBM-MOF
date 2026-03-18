"""
BKT Literature Comparison: ATC-Cu at Niu et al. 2019 Experimental Conditions
=============================================================================
Simulates breakthrough at the experimental column geometry and operating
conditions reported in Niu et al. 2019 (10.1002/anie.201904507):
  - Column: 4.6 mm ID x 50 mm length
  - Packed mass: 0.697 g ATC-Cu
  - Feed: CH4:N2 = 50:50
  - Flow: ~2 mL/min, 298 K, 1 bar
  - Experimental CH4 breakthrough: 10.69 min

Supports both DSL and DSLF isotherm models via --model flag.

Usage:
    python src/alignn/validation/bkt_literature_comparison.py --model DSLF
    python src/alignn/validation/bkt_literature_comparison.py --model DSL
"""

import sys
import os
import argparse
import collections
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
REPO = "/home/zhangsd/repos/CBM-MOF"
sys.path.insert(0, os.path.join(REPO, "src"))

from bkt.src.util import calculate_ki_Dax
import bkt.src.params as params
import bkt.src.model as model
from bkt.src.plot import data_to_state

# ── ATC-Cu Isotherm Parameters (298 K, from GCMC fitting) ─────────────────

# DSLF: q = qs1*b1*P^n1/(1+b1*P^n1) + qs2*b2*P^n2/(1+b2*P^n2)
DSLF_ISO = {
    "qs1_CH4": 3.5335667698250246,  "b1_CH4": 1.8784135750497013,  "n1_CH4": 1.341155532664619,
    "qs2_CH4": 2.6199322227244837,  "b2_CH4": 0.2639442399734971,  "n2_CH4": 0.5582238531029198,
    "qs1_N2":  0.24025773537762307, "b1_N2":  1.971125617221562,   "n1_N2":  1.047725040399253,
    "qs2_N2":  4.564896972174838,   "b2_N2":  0.13141683535735635, "n2_N2":  1.1517692312746464,
}

# DSL: DSLF with n=1
DSL_ISO = {
    "qs1_CH4": 5.108284256733254,     "b1_CH4": 1.2197703391704942,
    "qs2_CH4": 9.106397449233402,     "b2_CH4": 0.00024193549307004118,
    "qs1_N2":  0.26393114455779715,   "b1_N2":  0.15285116252643005,
    "qs2_N2":  5.014691746598145,     "b2_N2":  0.15282903514620563,
}

# ── Literature Experimental Conditions ─────────────────────────────────────
D_col  = 4.6e-3    # column inner diameter [m]
L_col  = 50e-3     # column length [m]
T      = 298       # temperature [K]
P_feed = 1         # feed pressure [bar]
Q_feed = 2.0       # volumetric flow rate [mL/min]
y_CH4  = 0.5       # CH4 mole fraction in feed
y_N2   = 0.5       # N2 mole fraction in feed
rho_s  = 1408.79   # crystal density [kg/m³]


def main():
    parser = argparse.ArgumentParser(description="BKT literature comparison")
    parser.add_argument("--model", choices=["DSL", "DSLF"], default="DSLF",
                        help="Isotherm model (default: DSLF)")
    args = parser.parse_args()
    iso_model = args.model
    ISO = DSLF_ISO if iso_model == "DSLF" else DSL_ISO

    # ── Bed geometry ────────────────────────────────────────────────────────
    A_col = math.pi / 4 * D_col**2
    V_bed = A_col * L_col

    print("=" * 70)
    print(f"BKT Literature Comparison: ATC-Cu — Model: {iso_model}")
    print("=" * 70)
    print(f"Column: D = {D_col*1e3:.1f} mm, L = {L_col*1e3:.0f} mm")
    print(f"  A = {A_col:.6e} m²  V_bed = {V_bed:.6e} m³")
    print(f"Feed: CH4:N2 = {y_CH4:.0%}:{y_N2:.0%}")
    print(f"Flow: {Q_feed} mL/min = {Q_feed*1e-6/60:.4e} m³/s")
    print(f"T = {T} K, P = {P_feed} bar")
    print(f"rho_s = {rho_s:.1f} kg/m³")

    # Epsilon from mass balance
    m_packed = 0.697e-3  # 0.697 g → kg
    epsilon = 1 - m_packed / (rho_s * V_bed)
    print(f"\nBed void fraction: epsilon = {epsilon:.4f}")

    # Transport parameters
    rp       = 2e-4     # particle radius [m]
    r_pore   = 25e-9    # pore radius [m]
    tor      = 3        # tortuosity
    epsilon_p = 0.35    # particle porosity

    Q_m3_s = Q_feed * 1e-6 / 60
    v_superficial = Q_m3_s / A_col
    v_interstitial = v_superficial / epsilon
    print(f"v_interstitial = {v_interstitial:.6f} m/s")

    k1, k2, Dax = calculate_ki_Dax(
        "CH4", "N2", T, P_feed, rp, r_pore, epsilon_p, tor, v_interstitial
    )
    print(f"k1 (CH4) = {k1:.4f} 1/s,  k2 (N2) = {k2:.4f} 1/s,  Dax = {Dax:.4e} m²/s")

    # ── Build BKT parameter dict ───────────────────────────────────────────
    tstop = 3000  # [s]
    tN = 500
    N_spatial = 30

    mods = collections.defaultdict()
    mods["nocomponents"] = 2
    mods["feed_yi"] = [y_CH4, y_N2]
    mods["ini_yi"] = [1e-10, 1e-10]
    mods["isomodel"] = iso_model
    mods["eq_method"] = "IAST"
    mods["component_names"] = ["CH4", "N2"]

    # Isotherm parameters
    mods["bi"]   = [ISO["b1_CH4"],  ISO["b1_N2"]]
    mods["qsbi"] = [ISO["qs1_CH4"], ISO["qs1_N2"]]
    mods["di"]   = [ISO["b2_CH4"],  ISO["b2_N2"]]
    mods["qsdi"] = [ISO["qs2_CH4"], ISO["qs2_N2"]]
    mods["Hi"]   = [0, 0]  # isothermal

    if iso_model == "DSLF":
        mods["n1i"] = [ISO["n1_CH4"], ISO["n1_N2"]]
        mods["n2i"] = [ISO["n2_CH4"], ISO["n2_N2"]]

    # Bed geometry
    mods["R"] = 8.314
    mods["D"] = D_col
    mods["A"] = A_col
    mods["L"] = L_col
    mods["epsilon"] = epsilon
    mods["rp"] = rp
    mods["Ta"] = T
    mods["feed_pressure"] = P_feed
    mods["vfeed"] = v_interstitial
    mods["rho_s"] = rho_s
    mods["ki"] = [k1, k2]
    mods["DL"] = Dax
    mods["bed"] = "Breakthrough"
    mods["tstart"] = 0
    mods["tstop"] = tstop
    mods["tbreak"] = 0
    mods["tN"] = tN
    mods["N"] = N_spatial

    print(f"\nSimulation: tstop = {tstop} s ({tstop/60:.1f} min), tN = {tN}, N = {N_spatial}")

    # ── Solve ──────────────────────────────────────────────────────────────
    print("\nCreating parameters and solving ODE system...")
    localparam = params.create_param(mods)
    print(f"Pe = {localparam.Pe:.2f}, psi = {localparam.psi:.4f}")

    import scipy.integrate
    x0_all = model.init(localparam)
    x0 = np.hstack([x0_all[item] for item in localparam.state_names])
    breakthroughmodel = model.oadesmodel
    ev = np.linspace(0, localparam.norm_tbreak, localparam.tN + 1, endpoint=True)
    t_span = (0, localparam.norm_tbreak)

    strategies = [
        ("BDF",   1e-6, 1e-9, "BDF tight"),
        ("Radau", 1e-6, 1e-9, "Radau tight"),
        ("BDF",   1e-4, 1e-7, "BDF default"),
        ("Radau", 1e-4, 1e-7, "Radau default"),
        ("LSODA", 1e-3, 1e-6, "LSODA relaxed"),
    ]

    outcome = None
    for method, rtol, atol, label in strategies:
        try:
            outcome = scipy.integrate.solve_ivp(
                breakthroughmodel, t_span, x0,
                vectorized=False, t_eval=ev, method=method,
                rtol=rtol, atol=atol, args=(localparam,),
            )
            if outcome.success and len(outcome.t) >= 10:
                print(f"  Solver OK: {label} ({len(outcome.t)} points)")
                break
            else:
                print(f"  Solver {label}: success={outcome.success}, points={len(outcome.t)}")
        except Exception as e:
            print(f"  Solver {label} exception: {e}")

    if outcome is None or not outcome.success:
        print("ERROR: All solvers failed!")
        sys.exit(1)

    # ── Extract breakthrough curves ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    curve_state = data_to_state(outcome.y, localparam)
    time_s = outcome.t * localparam.norm_t0
    time_min = time_s / 60

    yA_outlet = curve_state.yA[-1]  # CH4
    yB_outlet = curve_state.yB[-1]  # N2
    cc0_ch4 = yA_outlet / y_CH4
    cc0_n2  = yB_outlet / y_N2

    # ── Breakthrough times ─────────────────────────────────────────────────
    def find_breakthrough_time(time, cc0, threshold):
        start_idx = 1
        idx = np.where(cc0[start_idx:] >= threshold)[0]
        if len(idx) == 0:
            return None
        i = idx[0] + start_idx
        if i > 1:
            return np.interp(threshold, [cc0[i-1], cc0[i]], [time[i-1], time[i]])
        return time[i]

    fmt = lambda v: f"{v:.2f}" if v is not None else "N/A"

    t_ch4_1  = find_breakthrough_time(time_min, cc0_ch4, 0.01)
    t_ch4_5  = find_breakthrough_time(time_min, cc0_ch4, 0.05)
    t_ch4_50 = find_breakthrough_time(time_min, cc0_ch4, 0.50)
    t_ch4_95 = find_breakthrough_time(time_min, cc0_ch4, 0.95)
    t_n2_1   = find_breakthrough_time(time_min, cc0_n2,  0.01)
    t_n2_5   = find_breakthrough_time(time_min, cc0_n2,  0.05)
    t_n2_50  = find_breakthrough_time(time_min, cc0_n2,  0.50)
    t_n2_95  = find_breakthrough_time(time_min, cc0_n2,  0.95)

    print(f"\n{'Component':<12} {'1% C/C0':>10} {'5% C/C0':>10} {'50% C/C0':>10} {'95% C/C0':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"{'CH4':<12} {fmt(t_ch4_1):>10} {fmt(t_ch4_5):>10} {fmt(t_ch4_50):>10} {fmt(t_ch4_95):>10}")
    print(f"{'N2':<12} {fmt(t_n2_1):>10} {fmt(t_n2_5):>10} {fmt(t_n2_50):>10} {fmt(t_n2_95):>10}")

    t_exp = 10.69  # experimental CH4 breakthrough time (min)
    print(f"\nExperimental CH4 breakthrough: {t_exp:.2f} min")
    if t_ch4_5 is not None:
        print(f"Simulated CH4 5% C/C0: {t_ch4_5:.2f} min (ratio = {t_ch4_5/t_exp:.2f}×)")
    if t_ch4_1 is not None:
        print(f"Simulated CH4 1% C/C0: {t_ch4_1:.2f} min (ratio = {t_ch4_1/t_exp:.2f}×)")

    # ── Plot ───────────────────────────────────────────────────────────────
    out_dir = os.path.join(REPO, "results/alignn/model_ep150/literature_validation")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(time_min, cc0_ch4, "r-", linewidth=2, label="CH$_4$ (simulated)")
    ax.plot(time_min, cc0_n2, "b-", linewidth=2, label="N$_2$ (simulated)")
    ax.axvline(x=t_exp, color="gray", linestyle="--", linewidth=1.5,
               label=f"Exp. CH$_4$ breakthrough = {t_exp:.2f} min")
    if t_ch4_5 is not None:
        ax.axvline(x=t_ch4_5, color="r", linestyle=":", linewidth=1,
                   label=f"Sim. CH$_4$ 5% C/C$_0$ = {t_ch4_5:.2f} min")

    ax.set_xlabel("Time (min)", fontsize=13)
    ax.set_ylabel("C/C$_0$", fontsize=13)
    ax.set_title(f"BKT vs. Niu et al. 2019 (ATC-Cu, 50:50, 1 bar, {iso_model}+IAST)", fontsize=11)
    ax.set_xlim(0, max(time_min[-1], 2 * t_exp))
    ax.set_ylim(-0.02, max(cc0_n2.max(), cc0_ch4.max()) * 1.08)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = iso_model.lower()
    png_path = os.path.join(out_dir, f"bkt_literature_comparison_{suffix}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {png_path}")

    import pandas as pd
    df = pd.DataFrame({"time_s": time_s, "time_min": time_min,
                        "cc0_ch4": cc0_ch4, "cc0_n2": cc0_n2})
    csv_path = os.path.join(out_dir, f"bkt_literature_comparison_{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved: {csv_path}")

    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
