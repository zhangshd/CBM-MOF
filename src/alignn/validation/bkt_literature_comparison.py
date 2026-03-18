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
"""

import sys
import os
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

# ── ATC-Cu DSL Isotherm Parameters (298 K, from GCMC fitting) ─────────────
ISO = {
    # CH4
    "qs1_CH4": 5.108284256733254,     # mol/kg
    "b1_CH4":  1.2197703391704942,     # bar^-1
    "qs2_CH4": 9.106397449233402,      # mol/kg
    "b2_CH4":  0.00024193549307004118, # bar^-1
    # N2
    "qs1_N2":  0.26393114455779715,    # mol/kg
    "b1_N2":   0.15285116252643005,    # bar^-1
    "qs2_N2":  5.014691746598145,      # mol/kg
    "b2_N2":   0.15282903514620563,    # bar^-1
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

# ── Bed geometry calculations ──────────────────────────────────────────────
A_col = math.pi / 4 * D_col**2  # cross-section area [m²]
V_bed = A_col * L_col            # bed volume [m³]

print("=" * 70)
print("BKT Literature Comparison: ATC-Cu @ Niu et al. 2019 conditions")
print("=" * 70)
print(f"Column: D = {D_col*1e3:.1f} mm, L = {L_col*1e3:.0f} mm")
print(f"  A = {A_col:.6e} m²  V_bed = {V_bed:.6e} m³")
print(f"Feed: CH4:N2 = {y_CH4:.0%}:{y_N2:.0%}")
print(f"Flow: {Q_feed} mL/min = {Q_feed*1e-6/60:.4e} m³/s")
print(f"T = {T} K, P = {P_feed} bar")
print(f"rho_s = {rho_s:.1f} kg/m³")

# ── Check epsilon from mass balance ───────────────────────────────────────
# m_packed = rho_s * (1 - epsilon) * V_bed
# => epsilon = 1 - m_packed / (rho_s * V_bed)
m_packed = 0.697e-3  # 0.697 g → kg
epsilon_calc = 1 - m_packed / (rho_s * V_bed)
print(f"\nBed void fraction check:")
print(f"  m_packed = {m_packed*1e3:.3f} g")
print(f"  epsilon = 1 - m/(rho_s*V) = 1 - {m_packed:.6f}/({rho_s:.1f}*{V_bed:.6e})")
print(f"  epsilon = {epsilon_calc:.4f}")

# Use the calculated epsilon (more physically consistent with literature mass)
epsilon = epsilon_calc
print(f"  Using epsilon = {epsilon:.4f}")

# ── Transport parameters (keep defaults for unknowns) ─────────────────────
rp       = 2e-4     # particle radius [m] (our default)
r_pore   = 25e-9    # pore radius [m]
tor      = 3        # tortuosity
epsilon_p = 0.35    # particle porosity

# Flow velocity
Q_m3_s = Q_feed * 1e-6 / 60  # mL/min → m³/s
v_superficial = Q_m3_s / A_col
v_interstitial = v_superficial / epsilon
print(f"\nVelocity:")
print(f"  Q = {Q_m3_s:.4e} m³/s")
print(f"  v_superficial = {v_superficial:.6f} m/s")
print(f"  v_interstitial (vfeed) = {v_interstitial:.6f} m/s")

# ── Build parameter dict ──────────────────────────────────────────────────
k1, k2, Dax = calculate_ki_Dax(
    "CH4", "N2", T, P_feed, rp, r_pore, epsilon_p, tor, v_interstitial
)
print(f"\nTransport parameters:")
print(f"  k1 (CH4) = {k1:.4f} 1/s")
print(f"  k2 (N2)  = {k2:.4f} 1/s")
print(f"  Dax = {Dax:.4e} m²/s")

# ── Simulation time ───────────────────────────────────────────────────────
# Experimental CH4 breakthrough: 10.69 min = 641 s
# Simulate 3000 s (~50 min) to capture full breakthrough
tstop = 3000  # [s]
tN = 500      # time points
N_spatial = 30  # spatial discretization

mods = collections.defaultdict()
mods["nocomponents"] = 2
mods["feed_yi"] = [y_CH4, y_N2]
mods["ini_yi"] = [1e-10, 1e-10]
mods["isomodel"] = "DSL"
mods["eq_method"] = "IAST"
mods["component_names"] = ["CH4", "N2"]

# DSL isotherm parameters
mods["bi"]   = [ISO["b1_CH4"],  ISO["b1_N2"]]
mods["qsbi"] = [ISO["qs1_CH4"], ISO["qs1_N2"]]
mods["di"]   = [ISO["b2_CH4"],  ISO["b2_N2"]]
mods["qsdi"] = [ISO["qs2_CH4"], ISO["qs2_N2"]]
mods["Hi"]   = [0, 0]  # isothermal

# Bed geometry
mods["R"] = 8.314
mods["D"] = D_col
mods["A"] = A_col
mods["L"] = L_col
mods["epsilon"] = epsilon
mods["rp"] = rp
mods["Ta"] = T

# Pressure and flow
mods["feed_pressure"] = P_feed
mods["vfeed"] = v_interstitial

# Adsorbent
mods["rho_s"] = rho_s

# Transport
mods["ki"] = [k1, k2]
mods["DL"] = Dax

# Simulation control
mods["bed"] = "Breakthrough"
mods["tstart"] = 0
mods["tstop"] = tstop
mods["tbreak"] = 0
mods["tN"] = tN
mods["N"] = N_spatial

print(f"\nSimulation settings:")
print(f"  tstop = {tstop} s ({tstop/60:.1f} min)")
print(f"  tN = {tN} time points, N = {N_spatial} spatial cells")

# ── Create parameters and solve ───────────────────────────────────────────
print("\n" + "=" * 70)
print("Creating BKT parameters...")
localparam = params.create_param(mods)

print(f"\nNormalization constants:")
print(f"  norm_v0 = {localparam.norm_v0:.6f} m/s")
print(f"  norm_t0 = {localparam.norm_t0:.4f} s")
print(f"  norm_P0 = {localparam.norm_P0:.1f} Pa")
print(f"  Pe = {localparam.Pe:.2f}")
print(f"  psi = {localparam.psi:.4f}")

print("\nSolving BKT ODE system...")
import scipy.integrate

x0_all = model.init(localparam)
x0 = np.hstack([x0_all[item] for item in localparam.state_names])
breakthroughmodel = model.oadesmodel
ev = np.linspace(0, localparam.norm_tbreak, localparam.tN + 1, endpoint=True)
t_span = (0, localparam.norm_tbreak)

# Try solvers in order of preference
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

# ── Extract breakthrough curves ───────────────────────────────────────────
print("\n" + "=" * 70)
print("Extracting breakthrough curves...")

curve_state = data_to_state(outcome.y, localparam)
time_s = outcome.t * localparam.norm_t0  # dimensionless → seconds
time_min = time_s / 60

# yA = CH4 (component A), yB = N2 (component B)
# Shape: (N_spatial, tN+1) — spatial cells x time points
# Outlet = last cell
yA_outlet = curve_state.yA[-1]  # CH4 outlet mole fraction over time
yB_outlet = curve_state.yB[-1]  # N2 outlet mole fraction over time

# C/C0 ratios — normalize by FEED composition (constant reference),
# not instantaneous inlet (which starts at ini_yi ≈ 0 and ramps up).
cc0_ch4 = yA_outlet / y_CH4
cc0_n2  = yB_outlet / y_N2

print(f"Time range: {time_min[0]:.2f} to {time_min[-1]:.2f} min")
print(f"CH4 C/C0 range: {cc0_ch4.min():.4f} to {cc0_ch4.max():.4f}")
print(f"N2  C/C0 range: {cc0_n2.min():.4f} to {cc0_n2.max():.4f}")

# ── Find breakthrough times ──────────────────────────────────────────────
def find_breakthrough_time(time, cc0, threshold):
    """Find time when C/C0 first exceeds threshold (skipping t=0 initialization artifact)."""
    # Skip the first point (t=0) which may have C/C0=1 due to initialization
    # Find the first index where cc0 rises above threshold AFTER the initial drop
    start_idx = 1  # skip t=0
    idx = np.where(cc0[start_idx:] >= threshold)[0]
    if len(idx) == 0:
        return None
    i = idx[0] + start_idx  # adjust back to original indexing
    if i > 1:
        t_interp = np.interp(threshold, [cc0[i-1], cc0[i]], [time[i-1], time[i]])
        return t_interp
    return time[i]

t_ch4_1pct  = find_breakthrough_time(time_min, cc0_ch4, 0.01)
t_ch4_5pct  = find_breakthrough_time(time_min, cc0_ch4, 0.05)
t_ch4_50pct = find_breakthrough_time(time_min, cc0_ch4, 0.50)
t_ch4_95pct = find_breakthrough_time(time_min, cc0_ch4, 0.95)

t_n2_1pct  = find_breakthrough_time(time_min, cc0_n2, 0.01)
t_n2_5pct  = find_breakthrough_time(time_min, cc0_n2, 0.05)
t_n2_50pct = find_breakthrough_time(time_min, cc0_n2, 0.50)
t_n2_95pct = find_breakthrough_time(time_min, cc0_n2, 0.95)

print(f"\n{'='*70}")
print(f"BREAKTHROUGH TIMES (min)")
print(f"{'='*70}")
print(f"{'Component':<12} {'1% C/C0':>10} {'5% C/C0':>10} {'50% C/C0':>10} {'95% C/C0':>10}")
print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

def fmt(v):
    return f"{v:.2f}" if v is not None else "N/A"

print(f"{'CH4':<12} {fmt(t_ch4_1pct):>10} {fmt(t_ch4_5pct):>10} {fmt(t_ch4_50pct):>10} {fmt(t_ch4_95pct):>10}")
print(f"{'N2':<12} {fmt(t_n2_1pct):>10} {fmt(t_n2_5pct):>10} {fmt(t_n2_50pct):>10} {fmt(t_n2_95pct):>10}")

print(f"\n{'='*70}")
print(f"COMPARISON WITH LITERATURE")
print(f"{'='*70}")
t_exp = 10.69  # experimental CH4 breakthrough time (min)
print(f"Experimental CH4 breakthrough time: {t_exp:.2f} min")
if t_ch4_5pct is not None:
    ratio = t_ch4_5pct / t_exp
    print(f"Simulated CH4 breakthrough (5% C/C0): {t_ch4_5pct:.2f} min  (ratio = {ratio:.2f})")
if t_ch4_1pct is not None:
    ratio = t_ch4_1pct / t_exp
    print(f"Simulated CH4 breakthrough (1% C/C0): {t_ch4_1pct:.2f} min  (ratio = {ratio:.2f})")

# ── Additional bed capacity analysis ─────────────────────────────────────
print(f"\n{'='*70}")
print(f"BED CAPACITY ANALYSIS")
print(f"{'='*70}")

# Total CH4 capacity from DSL at feed conditions (pure-component at partial pressure)
P_ch4 = P_feed * y_CH4  # partial pressure
q_ch4_dsl = (ISO["qs1_CH4"] * ISO["b1_CH4"] * P_ch4 / (1 + ISO["b1_CH4"] * P_ch4) +
             ISO["qs2_CH4"] * ISO["b2_CH4"] * P_ch4 / (1 + ISO["b2_CH4"] * P_ch4))
print(f"Pure-component CH4 loading at P_CH4={P_ch4:.2f} bar: {q_ch4_dsl:.3f} mol/kg")

# Mass of adsorbent in bed
m_ads = rho_s * (1 - epsilon) * V_bed  # kg
print(f"Adsorbent mass in bed: {m_ads*1e3:.3f} g (literature: 0.697 g)")

# Molar flow of CH4 in feed
C_total = P_feed * 1e5 / (8.314 * T)  # mol/m³ (ideal gas)
F_ch4 = Q_m3_s * C_total * y_CH4  # mol/s
print(f"Total gas concentration: {C_total:.2f} mol/m³")
print(f"CH4 molar feed rate: {F_ch4:.4e} mol/s = {F_ch4*60*1e3:.4f} mmol/min")

# Simple stoichiometric breakthrough time estimate
q_ch4_total = q_ch4_dsl * m_ads  # mol total CH4 at saturation
t_stoich = q_ch4_total / F_ch4 / 60  # minutes
print(f"Total CH4 capacity in bed: {q_ch4_total*1e3:.4f} mmol")
print(f"Stoichiometric breakthrough time (pure-comp): {t_stoich:.2f} min")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.plot(time_min, cc0_ch4, "r-", linewidth=2, label="CH$_4$ (simulated)")
ax.plot(time_min, cc0_n2, "b-", linewidth=2, label="N$_2$ (simulated)")

# Mark experimental breakthrough time
ax.axvline(x=t_exp, color="gray", linestyle="--", linewidth=1.5,
           label=f"Exp. CH$_4$ breakthrough = {t_exp:.2f} min")

# Mark simulated breakthrough times
if t_ch4_5pct is not None:
    ax.axvline(x=t_ch4_5pct, color="r", linestyle=":", linewidth=1,
               label=f"Sim. CH$_4$ 5% C/C$_0$ = {t_ch4_5pct:.2f} min")

ax.set_xlabel("Time (min)", fontsize=13)
ax.set_ylabel("C/C$_0$", fontsize=13)
ax.set_title("BKT Simulation vs. Niu et al. 2019 (ATC-Cu, CH$_4$:N$_2$ = 50:50, 1 bar, 298 K)",
             fontsize=11)
ax.set_xlim(0, max(time_min[-1], 2 * t_exp))
y_max = max(cc0_n2.max(), cc0_ch4.max()) * 1.08
ax.set_ylim(-0.02, y_max)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)

# Secondary x-axis in seconds
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim()[0] * 60, ax.get_xlim()[1] * 60)
ax2.set_xlabel("Time (s)", fontsize=11)

plt.tight_layout()
outpath = os.path.join(REPO, "results/alignn/model_ep150/literature_validation/bkt_literature_comparison.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {outpath}")

# Also save as CSV for reference
import pandas as pd
df = pd.DataFrame({
    "time_s": time_s,
    "time_min": time_min,
    "cc0_ch4": cc0_ch4,
    "cc0_n2": cc0_n2,
})
csv_path = os.path.join(REPO, "results/alignn/model_ep150/literature_validation/bkt_literature_comparison.csv")
df.to_csv(csv_path, index=False)
print(f"Data saved: {csv_path}")

print(f"\n{'='*70}")
print(f"DONE")
print(f"{'='*70}")
