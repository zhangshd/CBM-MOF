"""Regression checks for ALIGNN-based database-analysis figure helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.figures.fig_database_analysis import (  # noqa: E402
    FIGURE6_API_UNIT,
    FIGURE6_PANEL_TITLES,
    FIGURE7_FEATURES,
    FIGURE7_CLIP_FEATURES,
    FEATURE_LABELS,
    FEATURE_TITLES,
    FEATURE_UNITS,
    ZEO_FEATURE_COLUMNS,
    build_database_analysis_frame,
    clip_feature_for_kde,
    compute_cluster_property_summary,
    compute_feature_shift_summary,
    format_range_value,
    select_shift_features,
)
from src.figures.style import TITLE_EMPHASIS_LINEWIDTH  # noqa: E402


def test_build_database_analysis_frame_merges_api_cluster_and_features(
    tmp_path: Path,
) -> None:
    """Merged analysis table should combine API, cluster, and Zeo++ features."""
    api_csv = tmp_path / "api.csv"
    cluster_csv = tmp_path / "cluster.csv"
    feature_csv = tmp_path / "features.csv"

    pd.DataFrame(
        {
            "mof_id": ["mof_a", "mof_b"],
            "PSA_API_CH4": [0.10, 0.20],
            "VSA_API_CH4": [0.03, 0.05],
            "PSA_WC_CH4": [1.0, 1.2],
            "PSA_alpha_CH4_N2": [2.5, 3.0],
            "VSA_WC_CH4": [0.10, 0.14],
            "VSA_alpha_CH4_N2": [2.2, 2.8],
            "QstCH4": [12.0, 13.5],
        }
    ).to_csv(api_csv, index=False)
    pd.DataFrame(
        {
            "CifId": ["mof_a", "mof_b"],
            "Cluster": [0, 4],
        }
    ).to_csv(cluster_csv, index=False)
    pd.DataFrame(
        {
            "name": ["mof_a", "mof_b"],
            "Di": [5.0, 8.0],
            "Df": [4.0, 6.0],
            "Dif": [5.5, 8.5],
            "rho": [0.9, 1.1],
            "VSA": [1200.0, 900.0],
            "GSA": [1800.0, 1400.0],
            "VPOV": [0.55, 0.40],
            "GPOV": [0.70, 0.52],
            "POAV_vol_frac": [0.52, 0.38],
            "PONAV_vol_frac": [0.03, 0.02],
            "GPOAV": [0.68, 0.50],
            "GPONAV": [0.02, 0.02],
            "POAV": [3200.0, 2500.0],
            "PONAV": [120.0, 80.0],
        }
    ).to_csv(feature_csv, index=False)

    merged = build_database_analysis_frame(api_csv, cluster_csv, feature_csv)

    assert list(merged["mof_id"]) == ["mof_a", "mof_b"]
    assert list(merged["Cluster"]) == [1, 5]
    assert set(["PSA_API_CH4", "VSA_API_CH4"]).issubset(merged.columns)
    assert set(ZEO_FEATURE_COLUMNS).issubset(merged.columns)


def test_compute_feature_shift_summary_scores_larger_psa_shift_higher() -> None:
    """Features with stronger top-candidate shifts should receive higher scores."""
    df = pd.DataFrame(
        {
            "mof_id": [f"m{i}" for i in range(8)],
            "PSA_API_CH4": [0.90, 0.85, 0.80, 0.75, 0.20, 0.18, 0.16, 0.14],
            "VSA_API_CH4": [0.15, 0.14, 0.13, 0.12, 0.95, 0.90, 0.88, 0.83],
            "Df": [8.0, 8.2, 8.1, 7.9, 2.0, 2.1, 1.9, 2.2],
            "rho": [0.90, 0.91, 0.89, 0.90, 0.90, 0.91, 0.89, 0.90],
        }
    )

    summary = compute_feature_shift_summary(
        df,
        feature_columns=["Df", "rho"],
        top_n=4,
    )

    top_feature = summary.sort_values("shift_score", ascending=False).iloc[0]["feature"]
    assert top_feature == "Df"


def test_select_shift_features_returns_between_four_and_six_when_possible() -> None:
    """Automatic selection should keep 4-6 strongest non-redundant features."""
    summary = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d", "e", "f", "g"],
            "shift_score": [1.10, 0.95, 0.82, 0.71, 0.54, 0.41, 0.18],
        }
    )
    feature_df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "c": [5, 4, 3, 2, 1],
            "d": [2, 2, 3, 3, 4],
            "e": [10, 9, 8, 7, 6],
            "f": [0, 1, 0, 1, 0],
            "g": [3, 1, 4, 1, 5],
        }
    )

    selected = select_shift_features(
        summary,
        feature_df,
        min_features=4,
        max_features=6,
        threshold=0.40,
        redundancy_corr_threshold=0.98,
    )

    assert 4 <= len(selected) <= 6
    assert selected[0] == "a"
    assert "b" not in selected
    assert "c" not in selected
    assert "e" not in selected
    assert selected[:4] == ["a", "d", "f", "g"]


def test_feature_labels_include_human_readable_zeopp_mappings() -> None:
    """Mapped labels should use manuscript-friendly Zeo++ names."""
    assert FEATURE_LABELS["Dif"] == "LCD"
    assert FEATURE_LABELS["Df"] == "PLD"
    assert FEATURE_LABELS["rho"] == "Density"
    assert FEATURE_LABELS["POAV_vol_frac"] == "Void Fraction"
    assert FEATURE_LABELS["VSA"] == "Volumetric Surface Area"


def test_feature_titles_include_units_for_kde_panels() -> None:
    """Figure 7 should use short titles while units are tracked separately."""
    assert FEATURE_TITLES["rho"] == "Density"
    assert FEATURE_TITLES["POAV_vol_frac"] == "VF"
    assert FEATURE_TITLES["Df"] == "PLD"
    assert FEATURE_UNITS["rho"] == r"g/cm$^3$"
    assert FEATURE_UNITS["GSA"] == r"m$^2$/g"
    assert FEATURE_UNITS["Df"] == r"$\AA$"


def test_figure7_feature_list_uses_manual_common_geometry_selection() -> None:
    """Figure 7 should show the user-approved common Zeo++ properties."""
    assert FIGURE7_FEATURES == ["rho", "POAV_vol_frac", "GSA", "VSA", "Dif", "Df"]
    assert FIGURE7_CLIP_FEATURES == {"Dif", "Df"}


def test_figure6_panel_titles_embed_api_units() -> None:
    """Figure 6 should move API units into the panel titles."""
    assert FIGURE6_API_UNIT == r"mol$^2$ kg$^{-1}$ kJ$^{-1}$"
    assert FIGURE6_PANEL_TITLES["PSA_API_CH4"] == r"Predicted PSA API (mol$^2$ kg$^{-1}$ kJ$^{-1}$)"
    assert FIGURE6_PANEL_TITLES["VSA_API_CH4"] == r"Predicted VSA API (mol$^2$ kg$^{-1}$ kJ$^{-1}$)"


def test_format_range_value_uses_scale_aware_precision() -> None:
    """Percentile annotations should use stable precision across scales."""
    assert format_range_value(1234.56) == "1235"
    assert format_range_value(87.654) == "87.7"
    assert format_range_value(5.4321) == "5.43"
    assert format_range_value(0.07891) == "0.079"


def test_clip_feature_for_kde_trims_pld_lcd_by_all_sample_q99() -> None:
    """PLD/LCD panels should trim only the extreme upper tail for visualization."""
    all_data = pd.Series([1, 2, 3, 4, 100])
    psa_data = pd.Series([2, 3, 150])
    vsa_data = pd.Series([1, 4, 120])

    clipped_all, clipped_psa, clipped_vsa = clip_feature_for_kde(
        "Df",
        all_data,
        psa_data,
        vsa_data,
    )

    assert clipped_all.max() < 100
    assert clipped_psa.max() < 150
    assert clipped_vsa.max() < 120

    same_all, same_psa, same_vsa = clip_feature_for_kde(
        "rho",
        all_data,
        psa_data,
        vsa_data,
    )
    assert same_all.equals(all_data)
    assert same_psa.equals(psa_data)
    assert same_vsa.equals(vsa_data)


def test_compute_cluster_property_summary_orders_cluster_medians() -> None:
    """Cluster summaries should preserve per-metric medians and ranks."""
    df = pd.DataFrame(
        {
            "Cluster": [11, 11, 5, 5, 8, 8],
            "PSA_WC_CH4": [0.80, 0.82, 1.40, 1.45, 1.10, 1.12],
            "PSA_alpha_CH4_N2": [1.45, 1.50, 3.50, 3.55, 2.80, 2.75],
            "VSA_WC_CH4": [0.08, 0.09, 0.30, 0.31, 0.18, 0.17],
            "VSA_alpha_CH4_N2": [1.42, 1.46, 3.60, 3.55, 2.70, 2.80],
            "QstCH4": [7.4, 7.6, 18.4, 18.6, 15.0, 15.2],
        }
    )

    summary = compute_cluster_property_summary(df)
    psa_alpha = summary.loc[summary["metric"] == "PSA_alpha_CH4_N2"].sort_values(
        "median", ascending=False
    )

    assert int(psa_alpha.iloc[0]["Cluster"]) == 5
    assert int(psa_alpha.iloc[-1]["Cluster"]) == 11
    assert float(psa_alpha.iloc[-1]["median"]) == 1.475


def test_title_emphasis_linewidth_is_centralized_in_style() -> None:
    """Math-aware title emphasis should come from the shared style module."""
    assert TITLE_EMPHASIS_LINEWIDTH == 0.6


if __name__ == "__main__":
    test_build_database_analysis_frame_merges_api_cluster_and_features(Path("/tmp"))
    test_compute_feature_shift_summary_scores_larger_psa_shift_higher()
    test_select_shift_features_returns_between_four_and_six_when_possible()
    test_feature_labels_include_human_readable_zeopp_mappings()
    test_feature_titles_include_units_for_kde_panels()
    test_figure7_feature_list_uses_manual_common_geometry_selection()
    test_figure6_panel_titles_embed_api_units()
    test_format_range_value_uses_scale_aware_precision()
    test_clip_feature_for_kde_trims_pld_lcd_by_all_sample_q99()
    test_compute_cluster_property_summary_orders_cluster_medians()
    test_title_emphasis_linewidth_is_centralized_in_style()
    print("11 tests passed")
