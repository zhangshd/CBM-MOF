import pandas as pd

from src.alignn.parse_validation_results import (
    calculate_validation_metrics,
    compute_metrics,
)


def _build_validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mof_id": ["MOF_A", "MOF_B", "MOF_C"],
            "AdsCH4_10kPa": [0.10, 0.20, 0.30],
            "AdsCH4_100kPa": [0.30, 0.50, 0.70],
            "AdsCH4_1000kPa": [0.90, 1.20, 1.50],
            "AdsN2_10kPa": [0.08, 0.12, 0.16],
            "AdsN2_100kPa": [0.20, 0.28, 0.36],
            "AdsN2_1000kPa": [0.70, 0.92, 1.14],
            "QstCH4": [18.0, 19.0, 20.0],
            "QstN2": [12.0, 12.5, 13.0],
            "gcmc_AdsCH4_10kPa": [0.11, 0.19, 0.31],
            "gcmc_AdsCH4_100kPa": [0.29, 0.52, 0.69],
            "gcmc_AdsCH4_1000kPa": [0.88, 1.18, 1.53],
            "gcmc_AdsN2_10kPa": [0.09, 0.11, 0.15],
            "gcmc_AdsN2_100kPa": [0.21, 0.27, 0.35],
            "gcmc_AdsN2_1000kPa": [0.68, 0.95, 1.10],
            "QstCH4_gcmc": [17.5, 18.8, 20.4],
            "QstN2_gcmc": [11.8, 12.3, 13.2],
        }
    )


def test_calculate_validation_metrics_preserves_raw_columns():
    df = _build_validation_frame()
    result = calculate_validation_metrics(df)

    for col in [
        "AdsCH4_10kPa",
        "AdsN2_1000kPa",
        "QstCH4",
        "gcmc_AdsCH4_100kPa",
        "QstCH4_gcmc",
        "QstN2_gcmc",
    ]:
        assert col in result.columns

    for col in [
        "gcmc_PSA_WC_CH4",
        "gcmc_PSA_alpha_CH4_N2",
        "gcmc_PSA_API_CH4",
        "gcmc_VSA_API_CH4",
    ]:
        assert col in result.columns


def test_compute_metrics_returns_expected_property_rows():
    df = _build_validation_frame()
    result = calculate_validation_metrics(df)
    metrics = compute_metrics(result)

    assert not metrics.empty
    assert set(metrics["property"]) == {
        "AdsCH4_10kPa",
        "AdsCH4_100kPa",
        "AdsCH4_1000kPa",
        "AdsN2_10kPa",
        "AdsN2_100kPa",
        "AdsN2_1000kPa",
        "QstCH4",
        "QstN2",
    }
