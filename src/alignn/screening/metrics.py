"""Shared screening metrics for PSA/VSA ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alignn.common.constants import API_A, API_B, API_C, API_Y_CH4, API_Y_N2


def calculate_separation_metrics(
    df: pd.DataFrame,
    y_ch4: float = API_Y_CH4,
    y_n2: float = API_Y_N2,
    a_param: float = API_A,
    b_param: float = API_B,
    c_param: float = API_C,
) -> pd.DataFrame:
    """Add PSA/VSA working-capacity, selectivity, and API columns."""
    result_df = df.copy()
    qst_ch4_abs = np.abs(result_df["QstCH4"])

    for process, ads_p, des_p in [("PSA", "1000kPa", "100kPa"), ("VSA", "100kPa", "10kPa")]:
        q_ch4_ads = result_df[f"AdsCH4_{ads_p}"]
        q_n2_ads = result_df[f"AdsN2_{ads_p}"]

        result_df[f"{process}_WC_CH4"] = result_df[f"AdsCH4_{ads_p}"] - result_df[f"AdsCH4_{des_p}"]
        result_df[f"{process}_WC_N2"] = result_df[f"AdsN2_{ads_p}"] - result_df[f"AdsN2_{des_p}"]

        alpha = np.where(
            q_n2_ads > 1e-10,
            (q_ch4_ads / q_n2_ads) * (y_n2 / y_ch4),
            np.nan,
        )
        result_df[f"{process}_alpha_CH4_N2"] = alpha

        valid = (qst_ch4_abs > 1e-10) & (result_df[f"{process}_alpha_CH4_N2"] > 1e-10)
        result_df[f"{process}_API_CH4"] = np.where(
            valid,
            ((result_df[f"{process}_alpha_CH4_N2"] - 1) ** a_param * result_df[f"{process}_WC_CH4"] ** b_param)
            / (qst_ch4_abs ** c_param),
            np.nan,
        )

    return result_df
