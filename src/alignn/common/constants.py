"""Canonical constants used across the ALIGNN mainline pipeline."""

from __future__ import annotations

UPTAKE_COLS = [
    "AdsCH4_10kPa",
    "AdsCH4_100kPa",
    "AdsCH4_1000kPa",
    "AdsN2_10kPa",
    "AdsN2_100kPa",
    "AdsN2_1000kPa",
]

QST_COLS = ["QstCH4", "QstN2"]
TARGET_COLS = UPTAKE_COLS + QST_COLS
N_TARGETS = len(TARGET_COLS)

MIN_SPEARMAN_RHO = 0.3
DEFAULT_K_NEIGHBORS = 10
DEFAULT_LSV_PERCENTILES = (50, 75, 80, 85, 90, 95)

UPTAKE_FLOOR_MOL_PER_KG = 0.01

API_Y_CH4 = 0.2
API_Y_N2 = 0.8
API_A = 1.0
API_B = 1.0
API_C = 1.0
