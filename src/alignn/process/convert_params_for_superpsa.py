"""Convert DSL isotherm fits + Qst + density to SuperPSA Adsorbents.csv format.

Library module — called by select_optimization_candidates.py.

Key conversions:
  - Pressure units: b [bar^-1] -> b0 [Pa^-1] = b / 1e5
  - Temperature dependence: deltaU_b = deltaU_d = 0 (single-T 298K data)
  - Column mapping: SuperPSA "CO2" columns -> CH4 params (repurposed)
  - Q_st -> deltaU:  deltaU [J/mol] = R*T - Q_st*1000
    (Q_st in kJ/mol, positive-exothermic; R=8.314, T=298K)

Public API:
  - build_adsorbents_table(fits_csv, density_csv, qst_csv) -> DataFrame
  - compute_ldf_coefficients(...) -> dict
  - qst_to_deltaU(qst_kJ, T) -> float
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Physical constants
R_GAS = 8.314       # J/mol/K
T_REF = 298.0       # K (reference temperature for single-T fits)
BAR_TO_PA = 1e5     # 1 bar = 1e5 Pa


def compute_ldf_coefficients(
    T: float = 298.0,
    P_bar: float = 1.0,
    r_p: float = 1e-3,
    d_pore: float = 100e-9,
    epsilon_p: float = 0.35,
    tau_p: float = 3.0,
) -> dict[str, float]:
    """Compute LDF mass transfer coefficients for CH4 and N2.

    Uses Chapman-Enskog molecular diffusivity and Knudsen diffusion
    with the combined resistance model.

    Args:
        T: Temperature [K].
        P_bar: Pressure [bar].
        r_p: Particle radius [m].  SuperPSA default = 1e-3 m (1 mm).
        d_pore: Mean macropore diameter [m].
        epsilon_p: Particle porosity.
        tau_p: Tortuosity.

    Returns:
        Dictionary with k_CH4_LDF, k_N2_LDF [1/s] and D_m [m^2/s].
    """
    # Lennard-Jones parameters (from Molecular Theory of Gases and Liquids)
    fc = {
        "CH4": {"MW": 16.0, "epsilon": 137.0, "theta": 3.822},
        "N2":  {"MW": 28.0, "epsilon": 91.5,  "theta": 3.681},
    }
    # Collision integral table (T*, Omega_11)
    collision_data = np.array([
        [0.30, 2.662], [0.35, 2.476], [0.40, 2.318], [0.45, 2.184],
        [0.50, 2.066], [0.55, 1.966], [0.60, 1.877], [0.65, 1.798],
        [0.70, 1.729], [0.75, 1.667], [0.80, 1.612], [0.85, 1.562],
        [0.90, 1.517], [0.95, 1.476], [1.00, 1.439], [1.05, 1.406],
        [1.10, 1.375], [1.15, 1.346], [1.20, 1.320], [1.25, 1.296],
        [1.30, 1.273], [1.35, 1.253], [1.40, 1.233], [1.45, 1.215],
        [1.50, 1.198], [1.55, 1.182], [1.60, 1.167], [1.65, 1.153],
        [1.70, 1.140], [1.75, 1.128], [1.80, 1.116], [1.85, 1.105],
        [1.90, 1.094], [1.95, 1.084], [2.00, 1.075], [2.10, 1.057],
        [2.20, 1.041], [2.30, 1.026], [2.40, 1.012], [2.50, 0.9996],
        [2.60, 0.9878], [2.70, 0.9770], [2.80, 0.9672], [2.90, 0.9576],
        [3.00, 0.9490], [3.10, 0.9406], [3.20, 0.9328], [3.30, 0.9256],
        [3.40, 0.9186], [3.50, 0.9120], [3.60, 0.9058], [3.70, 0.8998],
        [3.80, 0.8942], [3.90, 0.8888], [4.00, 0.8836], [5.00, 0.8422],
        [6.00, 0.8124], [7.00, 0.7896], [8.00, 0.7712], [9.00, 0.7556],
        [10.0, 0.7424],
    ])

    # Collision diameter and reduced temperature
    theta_12 = 0.5 * (fc["CH4"]["theta"] + fc["N2"]["theta"])
    T_star = T / (fc["CH4"]["epsilon"] * fc["N2"]["epsilon"]) ** 0.5
    Omega = np.interp(T_star, collision_data[:, 0], collision_data[:, 1])

    # Chapman-Enskog molecular diffusivity [m^2/s]
    MW_1, MW_2 = fc["CH4"]["MW"], fc["N2"]["MW"]
    D_m = (1.86e-7 * (T**3 * (1 / MW_1 + 1 / MW_2)) ** 0.5
           / (P_bar * theta_12**2 * Omega))

    # Knudsen diffusivity [m^2/s]  (r_pore = d_pore / 2)
    r_pore = d_pore / 2
    DK_CH4 = 97 * r_pore * (T / MW_1) ** 0.5
    DK_N2 = 97 * r_pore * (T / MW_2) ** 0.5

    # Combined effective diffusivity [m^2/s]
    D_eff_CH4 = epsilon_p / tau_p * (1 / D_m + 1 / DK_CH4) ** -1
    D_eff_N2 = epsilon_p / tau_p * (1 / D_m + 1 / DK_N2) ** -1

    # LDF mass transfer coefficient [1/s]
    k_CH4 = 15 * D_eff_CH4 / r_p**2
    k_N2 = 15 * D_eff_N2 / r_p**2

    return {
        "k_CH4_LDF": k_CH4,
        "k_N2_LDF": k_N2,
        "D_m": D_m,
        "DK_CH4": DK_CH4,
        "DK_N2": DK_N2,
        "D_eff_CH4": D_eff_CH4,
        "D_eff_N2": D_eff_N2,
        "T_star": T_star,
        "Omega": Omega,
    }


def qst_to_deltaU(qst_kJ: float, T: float = T_REF) -> float:
    """Convert positive-exothermic Q_st [kJ/mol] to deltaU [J/mol].

    SuperPSA convention: deltaU is negative (exothermic adsorption).
    Q_st = |deltaH| = |deltaU + R*T|  =>  deltaU = R*T - Q_st*1000

    Args:
        qst_kJ: Isosteric heat of adsorption [kJ/mol], positive-exothermic.
        T: Temperature [K].

    Returns:
        deltaU [J/mol], negative for exothermic adsorption.
    """
    return R_GAS * T - qst_kJ * 1000.0


def load_density(density_csv: Path) -> pd.DataFrame:
    """Load crystal density from RAC_and_zeo_features.csv.

    Args:
        density_csv: Path to RAC_and_zeo_features.csv (contains 'name' and 'rho' columns).

    Returns:
        DataFrame with columns [mof, rho_s], deduplicated.
    """
    df = pd.read_csv(density_csv, usecols=["name", "rho"])
    df = df.rename(columns={"name": "mof", "rho": "rho_s"}).drop_duplicates(subset="mof")
    # Convert g/cm^3 -> kg/m^3
    df["rho_s"] = df["rho_s"] * 1000.0
    logger.info("Loaded density for %d MOFs from %s", len(df), density_csv)
    return df


def load_qst(qst_csv: Path) -> pd.DataFrame:
    """Load Widom Q_st values from top20_combined.csv.

    Args:
        qst_csv: Path to top20_combined.csv.

    Returns:
        DataFrame with columns [mof_id, QstCH4_gcmc, QstN2_gcmc].
    """
    df = pd.read_csv(qst_csv, usecols=["mof_id", "QstCH4_gcmc", "QstN2_gcmc"])
    logger.info("Loaded Q_st for %d MOFs", len(df))
    return df


def build_adsorbents_table(
    fits_csv: Path,
    density_csv: Path,
    qst_csv: Path,
) -> pd.DataFrame:
    """Build the SuperPSA Adsorbents.csv table from pipeline data.

    Args:
        fits_csv: Path to best_isotherm_fits.csv.
        density_csv: Path to RAC_and_zeo_features.csv (crystal density).
        qst_csv: Path to top20_combined.csv.

    Returns:
        DataFrame in SuperPSA Adsorbents.csv format.
    """
    # Load input data
    fits = pd.read_csv(fits_csv)
    density = load_density(density_csv)
    qst = load_qst(qst_csv)

    # Separate CH4 and N2 fits
    ch4_mask = fits["GasName"].str.lower().isin(["methane", "ch4"])
    n2_mask = fits["GasName"].str.lower() == "n2"

    ch4_fits = fits[ch4_mask].set_index("MofName")
    n2_fits = fits[n2_mask].set_index("MofName")

    # Find common MOFs across all data sources
    common_mofs = sorted(
        set(ch4_fits.index) & set(n2_fits.index)
        & set(density["mof"]) & set(qst["mof_id"])
    )
    if not common_mofs:
        raise ValueError("No common MOFs found across fits, density, and Q_st data")
    logger.info("Building Adsorbents table for %d MOFs", len(common_mofs))

    density_map = density.set_index("mof")["rho_s"].to_dict()
    qst_map_ch4 = qst.set_index("mof_id")["QstCH4_gcmc"].to_dict()
    qst_map_n2 = qst.set_index("mof_id")["QstN2_gcmc"].to_dict()

    rows = []
    for mof in common_mofs:
        ch4 = ch4_fits.loc[mof]
        n2 = n2_fits.loc[mof]
        rho_s = density_map[mof]
        qst_ch4 = qst_map_ch4[mof]
        qst_n2 = qst_map_n2[mof]

        # Convert b from bar^-1 to Pa^-1
        b1_ch4_pa = ch4["b1"] / BAR_TO_PA
        b2_ch4_pa = ch4["b2"] / BAR_TO_PA
        b1_n2_pa = n2["b1"] / BAR_TO_PA
        b2_n2_pa = n2["b2"] / BAR_TO_PA

        # Overall deltaU from Widom Q_st
        deltaU_ch4 = qst_to_deltaU(qst_ch4)
        deltaU_n2 = qst_to_deltaU(qst_n2)

        row = {
            "material_name": mof,
            # --- CH4 params mapped to "CO2" columns ---
            "q_s_b_CO2 [mol/kg]": ch4["qs1"],
            "q_s_d_CO2 [mol/kg]": ch4["qs2"],
            "b0_CO2 [kPa^-1]": b1_ch4_pa,      # actually Pa^-1, header kept for compat
            "d0_CO2 [kPa^-1]": b2_ch4_pa,       # actually Pa^-1, header kept for compat
            "deltaU_b_CO2 [J/mol]": 0.0,         # single-T: no Arrhenius
            "deltaU_d_CO2 [J/mol]": 0.0,
            # --- N2 params ---
            "q_s_b_N2 [mol/kg]": n2["qs1"],
            "q_s_d_N2 [mol/kg]": n2["qs2"],
            "b0_N2 [kPa^-1]": b1_n2_pa,         # actually Pa^-1
            "d0_N2 [kPa^-1]": b2_n2_pa,         # actually Pa^-1
            "deltaU_b_N2 [J/mol]": 0.0,
            "deltaU_d_N2 [J/mol]": 0.0,
            # --- Flags & material properties ---
            "isotherm_type": 0,                   # 0 = partial pressure basis [Pa]
            "ro_s [kg/m^3]": rho_s,
            "deltaU_CO2 [J/mol]": deltaU_ch4,    # overall deltaU for CH4
            "deltaU_N2 [J/mol]": deltaU_n2,      # overall deltaU for N2
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Validation: check for negative qs or zero b values
    for col in ["q_s_b_CO2 [mol/kg]", "q_s_d_CO2 [mol/kg]",
                "q_s_b_N2 [mol/kg]", "q_s_d_N2 [mol/kg]"]:
        neg = df[col] < 0
        if neg.any():
            logger.warning("Negative saturation capacity in %s for: %s",
                           col, df.loc[neg, "material_name"].tolist())

    return df
