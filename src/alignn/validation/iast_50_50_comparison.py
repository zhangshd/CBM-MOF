"""
iast_50_50_comparison.py — Compare IAST selectivity with Niu et al. 2019 literature values.

Computes:
1. Pure-component DSL loadings at 1 bar vs literature
2. IAST selectivity at 50:50, 1 bar (literature condition)
3. IAST selectivity across multiple pressures at 50:50
4. IAST selectivity at 20:80 (CBM condition) at 1 bar and 10 bar (cross-check)

Usage:
    conda run -n mofmthnn python /tmp/iast_50_50_comparison.py
"""

import numpy as np
from scipy.optimize import brentq

# =====================================================================
# ATC-Cu DSL parameters (fitted from GCMC data at 298K)
# =====================================================================
CH4_PARAMS = dict(
    qs1=5.108284256733254, b1=1.2197703391704942,
    qs2=9.106397449233402, b2=0.00024193549307004118,
)
N2_PARAMS = dict(
    qs1=0.26393114455779715, b1=0.15285116252643005,
    qs2=5.014691746598145,  b2=0.15282903514620563,
)

# Literature values (Niu et al. 2019, doi:10.1002/anie.201904507)
LIT_ALPHA_50_50 = 9.7           # IAST α(CH4/N2) at 50:50, 1 bar, 298K
LIT_Q_CH4_MIX = 1.86            # mmol/g = mol/kg, CH4 from equimolar mixture
LIT_Q_CH4_PURE = 2.90           # mmol/g, pure CH4 at 1 bar
LIT_Q_N2_PURE = 0.75            # mmol/g, pure N2 at 1 bar


# =====================================================================
# DSL isotherm functions
# =====================================================================
def dsl_loading(P, qs1, b1, qs2, b2):
    """DSL loading at pressure P [bar]. Returns q [mol/kg]."""
    return qs1 * b1 * P / (1.0 + b1 * P) + qs2 * b2 * P / (1.0 + b2 * P)


def dsl_spreading_pressure(P, qs1, b1, qs2, b2):
    """Analytical spreading pressure for DSL: qs1*ln(1+b1*P) + qs2*ln(1+b2*P)."""
    return qs1 * np.log(1.0 + b1 * P) + qs2 * np.log(1.0 + b2 * P)


# =====================================================================
# IAST binary solver (from compute_iast_selectivity.py)
# =====================================================================
def iast_binary(params_1, params_2, y, P_total):
    """
    Solve binary IAST for DSL isotherms.
    params_1, params_2: dicts {qs1, b1, qs2, b2}
    y: (y1, y2) gas-phase mole fractions
    P_total: total pressure [bar]
    Returns (alpha, q1, q2)
    """
    y1, y2 = y

    def sp1(P): return dsl_spreading_pressure(P, **params_1)
    def sp2(P): return dsl_spreading_pressure(P, **params_2)
    def q1_fn(P): return dsl_loading(P, **params_1)
    def q2_fn(P): return dsl_loading(P, **params_2)

    def objective(x1):
        if x1 <= 0 or x1 >= 1:
            return 1e10
        P1_0 = P_total * y1 / x1
        P2_0 = P_total * y2 / (1.0 - x1)
        return sp1(P1_0) - sp2(P2_0)

    eps = 1e-10
    try:
        f_lo = objective(eps)
        f_hi = objective(1.0 - eps)
        if f_lo * f_hi > 0:
            xx = np.linspace(eps, 1.0 - eps, 200)
            ff = np.array([objective(x) for x in xx])
            sign_changes = np.where(np.diff(np.sign(ff)))[0]
            if len(sign_changes) == 0:
                return np.nan, np.nan, np.nan
            idx = sign_changes[0]
            x1_sol = brentq(objective, xx[idx], xx[idx + 1], xtol=1e-12)
        else:
            x1_sol = brentq(objective, eps, 1.0 - eps, xtol=1e-12)
    except Exception:
        return np.nan, np.nan, np.nan

    x2_sol = 1.0 - x1_sol
    P1_0 = P_total * y1 / x1_sol
    P2_0 = P_total * y2 / x2_sol

    q1_pure = q1_fn(P1_0)
    q2_pure = q2_fn(P2_0)

    if q1_pure <= 0 or q2_pure <= 0:
        return np.nan, np.nan, np.nan

    q_total = 1.0 / (x1_sol / q1_pure + x2_sol / q2_pure)
    q1 = x1_sol * q_total
    q2 = x2_sol * q_total

    alpha = (q1 / q2) * (y2 / y1) if q2 > 0 else np.nan

    return alpha, q1, q2


# =====================================================================
# Main
# =====================================================================
def main():
    sep = "=" * 70

    # ------------------------------------------------------------------
    # 1. Pure-component loadings at 1 bar
    # ------------------------------------------------------------------
    print(sep)
    print("1. PURE-COMPONENT DSL LOADINGS AT 1 BAR, 298K")
    print(sep)
    q_ch4_pure = dsl_loading(1.0, **CH4_PARAMS)
    q_n2_pure = dsl_loading(1.0, **N2_PARAMS)

    print(f"{'':>30s}  {'Our DSL':>10s}  {'Lit (Niu)':>10s}  {'Δ%':>8s}")
    print(f"{'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    pct_ch4 = (q_ch4_pure - LIT_Q_CH4_PURE) / LIT_Q_CH4_PURE * 100
    pct_n2 = (q_n2_pure - LIT_Q_N2_PURE) / LIT_Q_N2_PURE * 100
    print(f"{'CH4 pure (mol/kg)':>30s}  {q_ch4_pure:10.4f}  {LIT_Q_CH4_PURE:10.4f}  {pct_ch4:+7.1f}%")
    print(f"{'N2 pure (mol/kg)':>30s}  {q_n2_pure:10.4f}  {LIT_Q_N2_PURE:10.4f}  {pct_n2:+7.1f}%")
    print(f"\nNote: Our DSL is fitted to GCMC data; literature values are experimental.")
    print(f"      Some deviation is expected (GCMC vs experiment).\n")

    # Also check pure-component "selectivity" ratio
    ratio_pure = q_ch4_pure / q_n2_pure
    lit_ratio_pure = LIT_Q_CH4_PURE / LIT_Q_N2_PURE
    print(f"{'Pure loading ratio q_CH4/q_N2':>30s}  {ratio_pure:10.4f}  {lit_ratio_pure:10.4f}")

    # ------------------------------------------------------------------
    # 2. IAST at 50:50, 1 bar (literature condition)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("2. IAST SELECTIVITY AT CH4:N2 = 50:50, 1 BAR, 298K")
    print(sep)

    alpha, q_ch4, q_n2 = iast_binary(CH4_PARAMS, N2_PARAMS, (0.5, 0.5), 1.0)
    print(f"{'':>30s}  {'Our DSL':>10s}  {'Lit (Niu)':>10s}  {'Δ%':>8s}")
    print(f"{'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    pct_alpha = (alpha - LIT_ALPHA_50_50) / LIT_ALPHA_50_50 * 100
    pct_q = (q_ch4 - LIT_Q_CH4_MIX) / LIT_Q_CH4_MIX * 100
    print(f"{'α(CH4/N2)':>30s}  {alpha:10.4f}  {LIT_ALPHA_50_50:10.4f}  {pct_alpha:+7.1f}%")
    print(f"{'q_CH4 mix (mol/kg)':>30s}  {q_ch4:10.4f}  {LIT_Q_CH4_MIX:10.4f}  {pct_q:+7.1f}%")
    print(f"{'q_N2 mix (mol/kg)':>30s}  {q_n2:10.4f}  {'—':>10s}  {'—':>8s}")
    print(f"{'x_CH4 (adsorbed fraction)':>30s}  {q_ch4/(q_ch4+q_n2):10.4f}")

    # ------------------------------------------------------------------
    # 3. Pressure dependence at 50:50
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("3. IAST SELECTIVITY vs PRESSURE (CH4:N2 = 50:50, 298K)")
    print(sep)
    pressures = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"{'P (bar)':>10s}  {'α(CH4/N2)':>10s}  {'q_CH4':>10s}  {'q_N2':>10s}  {'q_total':>10s}")
    print(f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for P in pressures:
        a, q1, q2 = iast_binary(CH4_PARAMS, N2_PARAMS, (0.5, 0.5), P)
        print(f"{P:10.2f}  {a:10.4f}  {q1:10.4f}  {q2:10.4f}  {q1+q2:10.4f}")

    # ------------------------------------------------------------------
    # 4. Cross-check: CBM conditions (20:80) at 1 bar and 10 bar
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("4. CROSS-CHECK: CBM CONDITIONS (CH4:N2 = 20:80, 298K)")
    print(sep)
    print(f"{'Condition':>15s}  {'α(CH4/N2)':>10s}  {'q_CH4':>10s}  {'q_N2':>10s}  {'Ref α':>10s}")
    print(f"{'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    # VSA: 1 bar
    a_vsa, q1_vsa, q2_vsa = iast_binary(CH4_PARAMS, N2_PARAMS, (0.2, 0.8), 1.0)
    print(f"{'VSA (1 bar)':>15s}  {a_vsa:10.4f}  {q1_vsa:10.4f}  {q2_vsa:10.4f}  {'7.686':>10s}")

    # PSA: 10 bar
    a_psa, q1_psa, q2_psa = iast_binary(CH4_PARAMS, N2_PARAMS, (0.2, 0.8), 10.0)
    print(f"{'PSA (10 bar)':>15s}  {a_psa:10.4f}  {q1_psa:10.4f}  {q2_psa:10.4f}  {'7.498':>10s}")

    print(f"\nRef values from iast_selectivity.csv (α_IAST_PSA=7.498, α_IAST_VSA=7.686)")

    # ------------------------------------------------------------------
    # 5. Henry's law selectivity (zero-pressure limit)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("5. HENRY'S LAW SELECTIVITY (ZERO-PRESSURE LIMIT)")
    print(sep)
    K_H_ch4 = CH4_PARAMS["qs1"] * CH4_PARAMS["b1"] + CH4_PARAMS["qs2"] * CH4_PARAMS["b2"]
    K_H_n2 = N2_PARAMS["qs1"] * N2_PARAMS["b1"] + N2_PARAMS["qs2"] * N2_PARAMS["b2"]
    alpha_H = K_H_ch4 / K_H_n2
    print(f"K_H(CH4) = qs1*b1 + qs2*b2 = {K_H_ch4:.4f} mol/(kg·bar)")
    print(f"K_H(N2)  = qs1*b1 + qs2*b2 = {K_H_n2:.4f} mol/(kg·bar)")
    print(f"α_Henry = K_H(CH4)/K_H(N2) = {alpha_H:.4f}")
    print(f"\nNote: At very low pressure, IAST → Henry selectivity.")
    print(f"      At high pressure, site saturation → lower selectivity.")

    # ------------------------------------------------------------------
    # 6. DSL site breakdown
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("6. DSL SITE BREAKDOWN AT 1 BAR")
    print(sep)
    for name, params in [("CH4", CH4_PARAMS), ("N2", N2_PARAMS)]:
        q_site1 = params["qs1"] * params["b1"] * 1.0 / (1.0 + params["b1"] * 1.0)
        q_site2 = params["qs2"] * params["b2"] * 1.0 / (1.0 + params["b2"] * 1.0)
        q_tot = q_site1 + q_site2
        print(f"{name}: Site1 = {q_site1:.4f} mol/kg ({q_site1/q_tot*100:.1f}%), "
              f"Site2 = {q_site2:.4f} mol/kg ({q_site2/q_tot*100:.1f}%), "
              f"Total = {q_tot:.4f} mol/kg")
    print()

    # ------------------------------------------------------------------
    # 7. N2 DSL parameter analysis
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("7. N2 DSL PARAMETER ANALYSIS")
    print(sep)
    print(f"N2 Site 1: qs1={N2_PARAMS['qs1']:.4f}, b1={N2_PARAMS['b1']:.4f}")
    print(f"N2 Site 2: qs2={N2_PARAMS['qs2']:.4f}, b2={N2_PARAMS['b2']:.4f}")
    print(f"Note: b1 ≈ b2 ({N2_PARAMS['b1']:.6f} vs {N2_PARAMS['b2']:.6f})")
    print(f"      This means the two N2 sites are essentially degenerate")
    print(f"      (same affinity), making the DSL effectively a single-Langmuir")
    print(f"      with qs_eff = {N2_PARAMS['qs1']+N2_PARAMS['qs2']:.4f} mol/kg, "
          f"b_eff ≈ {N2_PARAMS['b1']:.4f} bar⁻¹")
    q_n2_equiv = (N2_PARAMS['qs1']+N2_PARAMS['qs2']) * N2_PARAMS['b1'] * 1.0 / (1.0 + N2_PARAMS['b1'] * 1.0)
    print(f"      Equivalent single-Langmuir loading at 1 bar: {q_n2_equiv:.4f} mol/kg")
    print(f"      Actual DSL loading at 1 bar: {q_n2_pure:.4f} mol/kg")


if __name__ == "__main__":
    main()
