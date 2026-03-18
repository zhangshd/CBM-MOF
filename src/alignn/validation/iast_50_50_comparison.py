"""
iast_50_50_comparison.py — Compare IAST selectivity with Niu et al. 2019 literature values.

Supports both DSL and DSLF isotherm models via --model flag.

Computes:
1. Pure-component loadings at 1 bar vs literature
2. IAST selectivity at 50:50, 1 bar (literature condition)
3. IAST selectivity across multiple pressures at 50:50
4. IAST selectivity at 20:80 (CBM condition) at 1 bar and 10 bar (cross-check)
5. Henry's law selectivity (zero-pressure limit)

Usage:
    python src/alignn/validation/iast_50_50_comparison.py --model DSLF
    python src/alignn/validation/iast_50_50_comparison.py --model DSL
"""

import argparse
import numpy as np
from scipy.optimize import brentq

# =====================================================================
# ATC-Cu isotherm parameters (fitted from GCMC pure-component data, 298K)
# =====================================================================

# DSLF: q = qs1*b1*P^n1/(1+b1*P^n1) + qs2*b2*P^n2/(1+b2*P^n2)
DSLF_CH4 = dict(qs1=3.5335667698250246, b1=1.8784135750497013, n1=1.341155532664619,
                qs2=2.6199322227244837, b2=0.2639442399734971, n2=0.5582238531029198)
DSLF_N2 = dict(qs1=0.24025773537762307, b1=1.971125617221562, n1=1.047725040399253,
               qs2=4.564896972174838, b2=0.13141683535735635, n2=1.1517692312746464)

# DSL: q = qs1*b1*P/(1+b1*P) + qs2*b2*P/(1+b2*P)  (DSLF with n1=n2=1)
DSL_CH4 = dict(qs1=5.108284256733254, b1=1.2197703391704942,
               qs2=9.106397449233402, b2=0.00024193549307004118)
DSL_N2 = dict(qs1=0.26393114455779715, b1=0.15285116252643005,
              qs2=5.014691746598145, b2=0.15282903514620563)

# Literature values (Niu et al. 2019, doi:10.1002/anie.201904507)
LIT_ALPHA_50_50 = 9.7           # IAST α(CH4/N2) at 50:50, 1 bar, 298K
LIT_Q_CH4_MIX = 1.86            # mmol/g = mol/kg, CH4 from equimolar mixture
LIT_Q_CH4_PURE = 2.90           # mmol/g, pure CH4 at 1 bar
LIT_Q_N2_PURE = 0.75            # mmol/g, pure N2 at 1 bar


# =====================================================================
# Isotherm functions
# =====================================================================
def dslf_loading(P, qs1, b1, n1, qs2, b2, n2):
    """DSLF loading at pressure P [bar]. Returns q [mol/kg]."""
    Pn1 = np.float_power(np.maximum(P, 0), n1)
    Pn2 = np.float_power(np.maximum(P, 0), n2)
    return qs1 * b1 * Pn1 / (1.0 + b1 * Pn1) + qs2 * b2 * Pn2 / (1.0 + b2 * Pn2)


def dslf_spreading_pressure(P, qs1, b1, n1, qs2, b2, n2):
    """Analytical spreading pressure for DSLF: (qs1/n1)*ln(1+b1*P^n1) + (qs2/n2)*ln(1+b2*P^n2)."""
    Pn1 = np.float_power(np.maximum(P, 0), n1)
    Pn2 = np.float_power(np.maximum(P, 0), n2)
    return (qs1 / n1) * np.log(1.0 + b1 * Pn1) + (qs2 / n2) * np.log(1.0 + b2 * Pn2)


def dsl_loading(P, qs1, b1, qs2, b2):
    """DSL loading at pressure P [bar]. Returns q [mol/kg]."""
    return qs1 * b1 * P / (1.0 + b1 * P) + qs2 * b2 * P / (1.0 + b2 * P)


def dsl_spreading_pressure(P, qs1, b1, qs2, b2):
    """Analytical spreading pressure for DSL: qs1*ln(1+b1*P) + qs2*ln(1+b2*P)."""
    return qs1 * np.log(1.0 + b1 * P) + qs2 * np.log(1.0 + b2 * P)


# =====================================================================
# IAST binary solver (generic — works for both DSL and DSLF)
# =====================================================================
def iast_binary(loading_fn_1, sp_fn_1, loading_fn_2, sp_fn_2, y, P_total):
    """
    Solve binary IAST.
    loading_fn_i: callable P → q [mol/kg]
    sp_fn_i: callable P → π*A/(RT) [mol/kg] (dimensionless spreading pressure)
    y: (y1, y2) gas-phase mole fractions
    P_total: total pressure [bar]
    Returns (alpha, q1, q2)
    """
    y1, y2 = y

    def objective(x1):
        if x1 <= 0 or x1 >= 1:
            return 1e10
        P1_0 = P_total * y1 / x1
        P2_0 = P_total * y2 / (1.0 - x1)
        return sp_fn_1(P1_0) - sp_fn_2(P2_0)

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

    q1_pure = loading_fn_1(P1_0)
    q2_pure = loading_fn_2(P2_0)

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
    parser = argparse.ArgumentParser(description="IAST 50:50 comparison with literature")
    parser.add_argument("--model", choices=["DSL", "DSLF"], default="DSLF",
                        help="Isotherm model (default: DSLF)")
    args = parser.parse_args()
    model = args.model

    # Build model-specific callables
    if model == "DSLF":
        ch4_params, n2_params = DSLF_CH4, DSLF_N2
        q_fn_1 = lambda P: dslf_loading(P, **ch4_params)
        q_fn_2 = lambda P: dslf_loading(P, **n2_params)
        sp_fn_1 = lambda P: dslf_spreading_pressure(P, **ch4_params)
        sp_fn_2 = lambda P: dslf_spreading_pressure(P, **n2_params)
        pure_q_ch4 = lambda P: dslf_loading(P, **ch4_params)
        pure_q_n2 = lambda P: dslf_loading(P, **n2_params)
    else:
        ch4_params, n2_params = DSL_CH4, DSL_N2
        q_fn_1 = lambda P: dsl_loading(P, **ch4_params)
        q_fn_2 = lambda P: dsl_loading(P, **n2_params)
        sp_fn_1 = lambda P: dsl_spreading_pressure(P, **ch4_params)
        sp_fn_2 = lambda P: dsl_spreading_pressure(P, **n2_params)
        pure_q_ch4 = lambda P: dsl_loading(P, **ch4_params)
        pure_q_n2 = lambda P: dsl_loading(P, **n2_params)

    def solve(y, P):
        return iast_binary(q_fn_1, sp_fn_1, q_fn_2, sp_fn_2, y, P)

    sep = "=" * 70

    # ------------------------------------------------------------------
    # 0. Model info
    # ------------------------------------------------------------------
    print(sep)
    print(f"IAST 50:50 COMPARISON — Model: {model}")
    print(sep)
    if model == "DSLF":
        print(f"CH4 DSLF: qs1={ch4_params['qs1']:.4f}, b1={ch4_params['b1']:.4f}, n1={ch4_params['n1']:.4f}")
        print(f"          qs2={ch4_params['qs2']:.4f}, b2={ch4_params['b2']:.4f}, n2={ch4_params['n2']:.4f}")
        print(f"N2  DSLF: qs1={n2_params['qs1']:.4f}, b1={n2_params['b1']:.4f}, n1={n2_params['n1']:.4f}")
        print(f"          qs2={n2_params['qs2']:.4f}, b2={n2_params['b2']:.4f}, n2={n2_params['n2']:.4f}")
    else:
        print(f"CH4 DSL: qs1={ch4_params['qs1']:.4f}, b1={ch4_params['b1']:.4f}")
        print(f"         qs2={ch4_params['qs2']:.4f}, b2={ch4_params['b2']:.4f}")
        print(f"N2  DSL: qs1={n2_params['qs1']:.4f}, b1={n2_params['b1']:.4f}")
        print(f"         qs2={n2_params['qs2']:.4f}, b2={n2_params['b2']:.4f}")

    # ------------------------------------------------------------------
    # 1. Pure-component loadings at 1 bar
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("1. PURE-COMPONENT LOADINGS AT 1 BAR, 298K")
    print(sep)
    q_ch4_pure = pure_q_ch4(1.0)
    q_n2_pure = pure_q_n2(1.0)

    print(f"{'':>30s}  {'Our '+model:>10s}  {'Lit (Niu)':>10s}  {'Δ%':>8s}")
    print(f"{'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    pct_ch4 = (q_ch4_pure - LIT_Q_CH4_PURE) / LIT_Q_CH4_PURE * 100
    pct_n2 = (q_n2_pure - LIT_Q_N2_PURE) / LIT_Q_N2_PURE * 100
    print(f"{'CH4 pure (mol/kg)':>30s}  {q_ch4_pure:10.4f}  {LIT_Q_CH4_PURE:10.4f}  {pct_ch4:+7.1f}%")
    print(f"{'N2 pure (mol/kg)':>30s}  {q_n2_pure:10.4f}  {LIT_Q_N2_PURE:10.4f}  {pct_n2:+7.1f}%")

    ratio_pure = q_ch4_pure / q_n2_pure
    lit_ratio_pure = LIT_Q_CH4_PURE / LIT_Q_N2_PURE
    print(f"{'Pure loading ratio q_CH4/q_N2':>30s}  {ratio_pure:10.4f}  {lit_ratio_pure:10.4f}")

    # ------------------------------------------------------------------
    # 2. IAST at 50:50, 1 bar (literature condition)
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("2. IAST SELECTIVITY AT CH4:N2 = 50:50, 1 BAR, 298K")
    print(sep)

    alpha, q_ch4, q_n2 = solve((0.5, 0.5), 1.0)
    print(f"{'':>30s}  {'Our '+model:>10s}  {'Lit (Niu)':>10s}  {'Δ%':>8s}")
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
        a, q1, q2 = solve((0.5, 0.5), P)
        print(f"{P:10.2f}  {a:10.4f}  {q1:10.4f}  {q2:10.4f}  {q1+q2:10.4f}")

    # ------------------------------------------------------------------
    # 4. Cross-check: CBM conditions (20:80) at 1 bar and 10 bar
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("4. CROSS-CHECK: CBM CONDITIONS (CH4:N2 = 20:80, 298K)")
    print(sep)
    print(f"{'Condition':>15s}  {'α(CH4/N2)':>10s}  {'q_CH4':>10s}  {'q_N2':>10s}")
    print(f"{'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}")

    a_vsa, q1_vsa, q2_vsa = solve((0.2, 0.8), 1.0)
    print(f"{'VSA (1 bar)':>15s}  {a_vsa:10.4f}  {q1_vsa:10.4f}  {q2_vsa:10.4f}")

    a_psa, q1_psa, q2_psa = solve((0.2, 0.8), 10.0)
    print(f"{'PSA (10 bar)':>15s}  {a_psa:10.4f}  {q1_psa:10.4f}  {q2_psa:10.4f}")

    # ------------------------------------------------------------------
    # 5. Henry's law selectivity
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("5. HENRY'S LAW SELECTIVITY (ZERO-PRESSURE LIMIT)")
    print(sep)
    if model == "DSLF":
        K_H_ch4 = ch4_params["qs1"] * ch4_params["b1"] * ch4_params["n1"] + \
                  ch4_params["qs2"] * ch4_params["b2"] * ch4_params["n2"]
        K_H_n2 = n2_params["qs1"] * n2_params["b1"] * n2_params["n1"] + \
                 n2_params["qs2"] * n2_params["b2"] * n2_params["n2"]
        print(f"Note: For DSLF, K_H = Σ qs_i * b_i * n_i (derivative at P→0)")
    else:
        K_H_ch4 = ch4_params["qs1"] * ch4_params["b1"] + ch4_params["qs2"] * ch4_params["b2"]
        K_H_n2 = n2_params["qs1"] * n2_params["b1"] + n2_params["qs2"] * n2_params["b2"]
    alpha_H = K_H_ch4 / K_H_n2
    print(f"K_H(CH4) = {K_H_ch4:.4f} mol/(kg·bar)")
    print(f"K_H(N2)  = {K_H_n2:.4f} mol/(kg·bar)")
    print(f"α_Henry = K_H(CH4)/K_H(N2) = {alpha_H:.4f}")

    # ------------------------------------------------------------------
    # 6. Multi-composition sweep at 1 bar and 10 bar
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("6. COMPOSITION SWEEP (298K)")
    print(sep)
    y_ch4_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    for P_tot in [1.0, 10.0]:
        print(f"\n  P = {P_tot} bar:")
        print(f"  {'y_CH4':>8s}  {'α(CH4/N2)':>10s}  {'q_CH4':>10s}  {'q_N2':>10s}  {'WC_CH4':>10s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        prev_q = None
        for yc in y_ch4_values:
            a, q1, q2 = solve((yc, 1 - yc), P_tot)
            wc_str = "—"
            if P_tot == 1.0 and prev_q is not None:
                # VSA WC: q(current) - q(previous lower pressure would need separate calc)
                pass
            print(f"  {yc:8.2f}  {a:10.4f}  {q1:10.4f}  {q2:10.4f}  {wc_str:>10s}")


if __name__ == "__main__":
    main()
