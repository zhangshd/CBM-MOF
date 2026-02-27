"""
verify_data_pipeline.py
=======================
Temporary test script to verify that each model's prepared data matches
the documented transformations in docs/data-pipeline.md.

Starting point: results/cbm_screening/raspa3_parsed_results_round2_0917.csv
                results/cbm_screening/widom_results_round2_0917.csv

Checks:
  [A] ML (XGBoost)      — raw mol/kg labels, no transform
  [B] CGCNN             — raw mol/kg labels, no transform
  [C] MOFTransformer    — raw mol/kg labels in CSV (transform happens in training)
  [D] ALIGNN id_prop    — symlog-transformed uptakes; raw Qst
  [E] Symlog round-trip — inv_symlog(symlog(x)) ≈ x

Run:
  cd /home/zhangsd/repos/CBM-MOF
  python tests/verify_data_pipeline.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path("/home/zhangsd/repos/CBM-MOF")

SRC_ADS   = REPO / "results/cbm_screening/raspa3_parsed_results_round2_0917.csv"
SRC_WIDOM = REPO / "results/cbm_screening/widom_results_round2_0917.csv"

ML_CSV    = REPO / "src/ml/data/round2/RAC_and_zeo_features_with_id_prop.csv"
CGCNN_TRAIN = REPO / "src/cgcnn/data/round2/train.csv"
CGCNN_VAL   = REPO / "src/cgcnn/data/round2/val.csv"
MFT_CSV   = REPO / "src/moftransformer/data/round2/integrated_ads_qst_metric_data.csv"
ALIGNN_TRAIN = REPO / "data/alignn/train/id_prop.csv"
ALIGNN_VAL   = REPO / "data/alignn/val/id_prop.csv"
ALIGNN_TEST  = REPO / "data/alignn/test/id_prop.csv"

# ── Target columns ─────────────────────────────────────────────────────────────
UPTAKE_COLS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
]
QST_COLS = ["QstCH4", "QstN2"]
ALL_TARGETS = UPTAKE_COLS + QST_COLS

# ── Symlog helpers (must match prepare_data.py exactly) ───────────────────────
SYMLOG_THRESH = 1e-4

def symlog(x: np.ndarray, threshold: float = SYMLOG_THRESH) -> np.ndarray:
    """sign(x) * log10(1 + |x| / threshold)"""
    return np.sign(x) * np.log10(1.0 + np.abs(x) / threshold)

def inv_symlog(y: np.ndarray, threshold: float = SYMLOG_THRESH) -> np.ndarray:
    return np.sign(y) * threshold * (10.0 ** np.abs(y) - 1.0)

# ── Helpers ────────────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

def check(ok: bool, msg: str, detail: str = "") -> bool:
    label = PASS if ok else FAIL
    print(f"  {label} {msg}")
    if detail:
        print(f"         {detail}")
    return ok

results = []  # (name, passed)

# ══════════════════════════════════════════════════════════════════════════════
# Build reference integrated dataset from source CSV files
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{INFO} Loading source data …")

ads_df = pd.read_csv(SRC_ADS)
ads_df["GasName"] = ads_df["GasName"].str.replace("methane", "CH4")

widom_df = pd.read_csv(SRC_WIDOM)
widom_df["GasName"] = widom_df["GasName"].str.replace("methane", "CH4")

# Build adsorption pivot: 6 uptake columns (mol/kg)
ads_pivot = (
    ads_df.pivot_table(
        index="MofName",
        columns=["GasName", "Pressure[bar]"],
        values="AbsLoading",
        aggfunc="first",
    )
    .rename(columns=lambda c: c, level=0)
)
ads_pivot.columns = [
    f"Ads{gas}_{pressure * 100:.0f}kPa"   # 0.1 bar * 100 = 10kPa, 1 bar = 100kPa, 10 bar = 1000kPa
    for gas, pressure in ads_pivot.columns
]
ads_pivot = ads_pivot.reset_index()

# Build Widom pivot: 2 Qst columns (kJ/mol)
widom_pivot = (
    widom_df.pivot_table(
        index="MofName",
        columns="GasName",
        values="AdsorptionHeat",
        aggfunc="first",
    )
    .rename(columns=lambda c: f"Qst{c}")
    .reset_index()
)

ref_df = pd.merge(ads_pivot, widom_pivot, on="MofName", how="outer")
# Keep only rows with complete 8-label set
ref_df = ref_df.dropna(subset=ALL_TARGETS)
print(f"  Reference dataset: {len(ref_df)} MOFs with complete 8-label set")
print(f"  Uptake range (CH4 @ 1bar): "
      f"{ref_df['AdsCH4_100kPa'].min():.4f} – {ref_df['AdsCH4_100kPa'].max():.4f} mol/kg")
print(f"  Qst range (CH4): "
      f"{ref_df['QstCH4'].min():.2f} – {ref_df['QstCH4'].max():.2f} kJ/mol")

ref_index = ref_df.set_index("MofName")
ATOL_RAW    = 1e-6   # tolerance for raw value comparison
ATOL_SYMLOG = 1e-5   # tolerance for symlog value comparison
SAMPLE_N    = 200    # random MOFs to spot-check


# ══════════════════════════════════════════════════════════════════════════════
# [A] ML (XGBoost) — raw labels
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("[A] ML (XGBoost) — src/ml/data/round2/RAC_and_zeo_features_with_id_prop.csv")
print(f"{'='*60}")

ml_df = pd.read_csv(ML_CSV)
print(f"  Rows: {len(ml_df)}, Partitions: {dict(ml_df['Partition'].value_counts())}")

# Check required columns present
has_cols = all(c in ml_df.columns for c in ALL_TARGETS)
r = check(has_cols, "All 8 target columns present",
          f"Missing: {[c for c in ALL_TARGETS if c not in ml_df.columns]}")
results.append(("[A] ML columns", r))

# Spot-check: values should match raw reference exactly
sample_mofs = ml_df[ml_df["MofName"].isin(ref_index.index)].sample(
    min(SAMPLE_N, len(ml_df)), random_state=42
)
common = sample_mofs[sample_mofs["MofName"].isin(ref_index.index)]

mismatches = []
for col in UPTAKE_COLS:
    ml_vals = common.set_index("MofName")[col]
    ref_vals = ref_index.loc[ml_vals.index, col]
    if not np.allclose(ml_vals.values, ref_vals.values, atol=ATOL_RAW, equal_nan=True):
        max_diff = np.nanmax(np.abs(ml_vals.values - ref_vals.values))
        mismatches.append(f"{col}: max_diff={max_diff:.2e}")

r = check(len(mismatches) == 0, f"Uptake values are RAW (spot-check {len(common)} MOFs)",
          "; ".join(mismatches) if mismatches else "")
results.append(("[A] ML raw uptakes", r))

# Sanity: values should NOT look like symlog (symlog of typical mol/kg >> 1 gives ~4-5)
median_ch4 = ml_df["AdsCH4_100kPa"].median()
r = check(median_ch4 > 0.01, "Median AdsCH4_100kPa plausible for raw mol/kg (>0.01)",
          f"Got {median_ch4:.4f} mol/kg")
results.append(("[A] ML raw magnitude", r))


# ══════════════════════════════════════════════════════════════════════════════
# [B] CGCNN — raw labels
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("[B] CGCNN — src/cgcnn/data/round2/{{train,val}}.csv")
print(f"{'='*60}")

cgcnn_train = pd.read_csv(CGCNN_TRAIN)
cgcnn_val   = pd.read_csv(CGCNN_VAL)
print(f"  train: {len(cgcnn_train)}, val: {len(cgcnn_val)}")

has_cols = all(c in cgcnn_train.columns for c in ALL_TARGETS)
r = check(has_cols, "All 8 target columns present in train.csv",
          f"Missing: {[c for c in ALL_TARGETS if c not in cgcnn_train.columns]}")
results.append(("[B] CGCNN columns", r))

# Spot-check raw values
common_c = cgcnn_train[cgcnn_train["MofName"].isin(ref_index.index)].sample(
    min(SAMPLE_N, len(cgcnn_train)), random_state=42
)
mismatches = []
for col in UPTAKE_COLS:
    cv = common_c.set_index("MofName")[col]
    rv = ref_index.loc[cv.index, col]
    if not np.allclose(cv.values, rv.values, atol=ATOL_RAW, equal_nan=True):
        max_diff = np.nanmax(np.abs(cv.values - rv.values))
        mismatches.append(f"{col}: max_diff={max_diff:.2e}")

r = check(len(mismatches) == 0, f"Uptake values are RAW (spot-check {len(common_c)} MOFs)",
          "; ".join(mismatches) if mismatches else "")
results.append(("[B] CGCNN raw uptakes", r))

# Check geometric feature columns appended (Di ... PONAV)
geo_cols_present = "Di" in cgcnn_train.columns and "PONAV" in cgcnn_train.columns
r = check(geo_cols_present, "Geometric feature columns Di ... PONAV appended")
results.append(("[B] CGCNN geo features", r))


# ══════════════════════════════════════════════════════════════════════════════
# [C] MOFTransformer — raw labels in CSV (transform happens in training)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("[C] MOFTransformer — src/moftransformer/data/round2/integrated_ads_qst_metric_data.csv")
print(f"{'='*60}")

mft_df = pd.read_csv(MFT_CSV)
print(f"  Rows: {len(mft_df)}")

has_cols = all(c in mft_df.columns for c in ALL_TARGETS)
r = check(has_cols, "All 8 target columns present",
          f"Missing: {[c for c in ALL_TARGETS if c not in mft_df.columns]}")
results.append(("[C] MFT columns", r))

# Spot-check raw values
common_m = mft_df[mft_df["MofName"].isin(ref_index.index)].sample(
    min(SAMPLE_N, len(mft_df)), random_state=42
)
mismatches = []
for col in UPTAKE_COLS:
    mv = common_m.set_index("MofName")[col]
    rv = ref_index.loc[mv.index, col]
    if not np.allclose(mv.values, rv.values, atol=ATOL_RAW, equal_nan=True):
        max_diff = np.nanmax(np.abs(mv.values - rv.values))
        mismatches.append(f"{col}: max_diff={max_diff:.2e}")

r = check(len(mismatches) == 0, f"Uptake values are RAW (spot-check {len(common_m)} MOFs)",
          "; ".join(mismatches) if mismatches else "")
results.append(("[C] MFT raw uptakes", r))

# Check separation metric columns present (derived columns)
sep_cols = ["PSA_WC_CH4", "PSA_alpha_CH4_N2", "VSA_WC_CH4", "VSA_alpha_CH4_N2"]
has_sep = all(c in mft_df.columns for c in sep_cols)
r = check(has_sep, "Separation metric columns (PSA/VSA) present",
          f"Missing: {[c for c in sep_cols if c not in mft_df.columns]}")
results.append(("[C] MFT sep metrics", r))


# ══════════════════════════════════════════════════════════════════════════════
# [D] ALIGNN id_prop.csv — symlog-transformed uptakes, raw Qst
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("[D] ALIGNN id_prop — data/alignn/{train,val,test}/id_prop.csv")
print(f"{'='*60}")

alignn_dfs = {}
for split, path in [("train", ALIGNN_TRAIN), ("val", ALIGNN_VAL), ("test", ALIGNN_TEST)]:
    if path.exists():
        alignn_dfs[split] = pd.read_csv(path)
        print(f"  {split}: {len(alignn_dfs[split])} rows")
    else:
        print(f"  {WARN} {split} not found: {path}")

if not alignn_dfs:
    print(f"  {FAIL} No ALIGNN id_prop.csv found — skipping [D]")
else:
    # Merge all splits for checks
    alignn_all = pd.concat(alignn_dfs.values(), ignore_index=True)
    # Column name: ALIGNN uses 'mol_id' not 'MofName'
    id_col = "mol_id" if "mol_id" in alignn_all.columns else "MofName"
    print(f"  ID column: '{id_col}'")

    # Expected columns
    has_cols = all(c in alignn_all.columns for c in ALL_TARGETS)
    r = check(has_cols, "All 8 target columns present",
              f"Missing: {[c for c in ALL_TARGETS if c not in alignn_all.columns]}")
    results.append(("[D] ALIGNN columns", r))

    # Core check: uptake columns should be symlog-transformed (i.e., inv_symlog recovers raw)
    alignn_indexed = alignn_all.set_index(id_col)
    common_a = alignn_all[alignn_all[id_col].isin(ref_index.index)].sample(
        min(SAMPLE_N, len(alignn_all)), random_state=42
    )

    print(f"\n  Spot-check {len(common_a)} MOFs for symlog transform …")
    uptake_sym_ok = []
    for col in UPTAKE_COLS:
        av = common_a.set_index(id_col)[col].values          # symlog-space values in file
        rv = ref_index.loc[common_a.set_index(id_col).index, col].values  # raw reference

        # 1. Check values do NOT match raw (would fail if symlog was not applied)
        is_raw = np.allclose(av, rv, atol=1e-4, equal_nan=True)

        # 2. Check inv_symlog recovers raw values
        recovered = inv_symlog(av)
        is_roundtrip = np.allclose(recovered, rv, atol=ATOL_SYMLOG, equal_nan=True)

        max_raw_diff = np.nanmax(np.abs(av - rv))
        max_rt_diff  = np.nanmax(np.abs(recovered - rv))
        ok = (not is_raw) and is_roundtrip
        uptake_sym_ok.append(ok)
        status = PASS if ok else FAIL
        print(f"    {status} {col}: "
              f"is_raw={is_raw}, roundtrip_ok={is_roundtrip} "
              f"(max raw diff={max_raw_diff:.3f}, max roundtrip diff={max_rt_diff:.2e})")

    r = all(uptake_sym_ok)
    results.append(("[D] ALIGNN uptakes symlog", r))

    # Qst columns should be raw (NOT symlog-transformed)
    qst_raw_ok = []
    for col in QST_COLS:
        av = common_a.set_index(id_col)[col].values
        rv = ref_index.loc[common_a.set_index(id_col).index, col].values
        is_raw = np.allclose(av, rv, atol=ATOL_RAW, equal_nan=True)
        qst_raw_ok.append(is_raw)
        status = PASS if is_raw else FAIL
        max_diff = np.nanmax(np.abs(av - rv))
        print(f"    {status} {col}: raw_match={is_raw} (max diff={max_diff:.2e})")

    r = all(qst_raw_ok)
    results.append(("[D] ALIGNN Qst raw", r))

    # Cross-check: ALIGNN uptake values numerically look like ~3-5 range (symlog domain)
    # symlog(1 mol/kg) ≈ log10(1 + 1/1e-4) ≈ 4.0
    median_alignn = alignn_all["AdsCH4_100kPa"].median()
    in_symlog_range = 1.0 < abs(median_alignn) < 6.0
    r = check(in_symlog_range,
              f"Median AdsCH4_100kPa in plausible symlog range (1~6)",
              f"Got {median_alignn:.4f}")
    results.append(("[D] ALIGNN symlog magnitude", r))


# ══════════════════════════════════════════════════════════════════════════════
# [E] Symlog round-trip self-test
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("[E] Symlog round-trip self-test")
print(f"{'='*60}")

test_vals = np.array([0.0, 1e-6, 1e-4, 0.01, 0.1, 1.0, 5.0, 50.0, -0.5, -10.0])
recovered  = inv_symlog(symlog(test_vals))
rt_ok = np.allclose(test_vals, recovered, atol=1e-10)
r = check(rt_ok, "inv_symlog(symlog(x)) ≈ x for sample values",
          f"Max error: {np.max(np.abs(test_vals - recovered)):.2e}")
results.append(("[E] Symlog round-trip", r))

# Specific value check: symlog(1.0 mol/kg) ≈ 4.0
expected = np.log10(1.0 + 1.0 / SYMLOG_THRESH)   # ≈ 4.0
got = symlog(np.array([1.0]))[0]
r = check(abs(got - expected) < 1e-10,
          f"symlog(1.0) = {got:.6f} (expected ≈ {expected:.6f})")
results.append(("[E] Symlog value check", r))


# ══════════════════════════════════════════════════════════════════════════════
# [F] CrystalFramer pkl — data/cbm_mof/{train,val,test}/raw/raw_data.pkl
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("[F] CrystalFramer pkl (data/cbm_mof/*/raw/raw_data.pkl)")
print(f"{'='*60}")

CF_PKL_ROOT = REPO / "data" / "cbm_mof"
CF_SPLITS   = ["train", "val", "test"]

# Expected split sizes (from prepare_data_cf.py run; use loose bounds for robustness)
EXPECTED_SIZES = {"train": (19000, 21000), "val": (900, 1100), "test": (900, 1100)}
CF_TARGET_COLS = [
    "AdsCH4_10kPa", "AdsCH4_100kPa", "AdsCH4_1000kPa",
    "AdsN2_10kPa",  "AdsN2_100kPa",  "AdsN2_1000kPa",
    "QstCH4",       "QstN2",
]
CF_SYMLOG_COLS = CF_TARGET_COLS[:6]

pkl_paths = {s: CF_PKL_ROOT / s / "raw" / "raw_data.pkl" for s in CF_SPLITS}
missing_splits = [s for s, p in pkl_paths.items() if not p.exists()]
if missing_splits:
    print(f"  {FAIL} Missing pkl splits: {missing_splits} — skipping [F]")
else:
    import pickle as _pickle

    cf_data: dict[str, list] = {}
    for split, pth in pkl_paths.items():
        with open(pth, "rb") as f:
            cf_data[split] = _pickle.load(f)

    # F-1: Split sizes within expected range
    for split, records in cf_data.items():
        lo, hi = EXPECTED_SIZES[split]
        r = check(lo <= len(records) <= hi,
                  f"[F-1] {split} split size in [{lo}, {hi}]",
                  f"Got {len(records)}")
        results.append((f"[F] CrystalFramer {split} size", r))

    # F-2: All target columns present in every item
    sample_item = cf_data["train"][0]
    all_keys_present = all(col in sample_item for col in CF_TARGET_COLS)
    r = check(all_keys_present,
              "[F-2] All 8 target columns present in pkl items",
              f"Keys found: {[k for k in sample_item if k != 'structure']}")
    results.append(("[F] CrystalFramer target keys", r))

    # F-3: Uptake values are in symlog range (1–6), NOT raw mol/kg (0.001–300)
    train_uptakes = np.array([item["AdsCH4_100kPa"] for item in cf_data["train"]])
    median_uptake = float(np.median(train_uptakes))
    in_symlog_range = 1.0 < abs(median_uptake) < 6.0
    r = check(in_symlog_range,
              f"[F-3] Train AdsCH4_100kPa median in symlog range (1–6)",
              f"Got {median_uptake:.4f}")
    results.append(("[F] CrystalFramer uptake symlog range", r))

    # F-4: Cross-check uptakes — inv_symlog(pkl) should match reference raw values
    # Build a lookup from the reference label CSV (same source used by prepare_data_cf.py)
    ref_label_csv = REPO / "src/ml/data/round2/RAC_and_zeo_features_with_id_prop.csv"
    if ref_label_csv.exists():
        ref_labels = pd.read_csv(ref_label_csv)
        ref_labels = ref_labels.set_index("MofName")
        sample_train = cf_data["train"][:200]
        errors_uptake = []
        errors_qst    = []
        skipped = 0
        for item in sample_train:
            mid = item["material_id"]
            if mid not in ref_labels.index:
                skipped += 1
                continue
            for col in CF_SYMLOG_COLS:
                raw_ref  = float(ref_labels.loc[mid, col])
                raw_pred = float(inv_symlog(np.array([item[col]]))[0])
                errors_uptake.append(abs(raw_pred - raw_ref))
            for col in ["QstCH4", "QstN2"]:
                raw_ref  = float(ref_labels.loc[mid, col])
                raw_pkg  = float(item[col])
                errors_qst.append(abs(raw_pkg - raw_ref))

        if errors_uptake:
            max_err_uptake = max(errors_uptake)
            r = check(max_err_uptake < 0.01,
                      f"[F-4a] inv_symlog(pkl uptake) matches reference raw (tol=0.01 mol/kg, "
                      f"n={len(errors_uptake)//6}, skipped={skipped})",
                      f"Max error: {max_err_uptake:.6f}")
            results.append(("[F] CrystalFramer uptake cross-check", r))

        if errors_qst:
            max_err_qst = max(errors_qst)
            r = check(max_err_qst < 0.01,
                      f"[F-4b] Pkl Qst matches reference raw kJ/mol (tol=0.01)",
                      f"Max error: {max_err_qst:.6f}")
            results.append(("[F] CrystalFramer Qst cross-check", r))
    else:
        print(f"  ⚠  Reference label CSV not found — skipping F-4 cross-check")

    # F-5: Qst values are NOT in symlog range — they should be 4–60 kJ/mol (raw)
    train_qst = np.array([item["QstCH4"] for item in cf_data["train"]])
    median_qst = float(np.median(train_qst))
    r = check(4.0 < median_qst < 60.0,
              f"[F-5] Train QstCH4 median in plausible raw range (4–60 kJ/mol)",
              f"Got {median_qst:.4f}")
    results.append(("[F] CrystalFramer Qst raw range", r))

    # F-6: material_id column present and non-empty
    has_mat_id = all("material_id" in item and item["material_id"] for item in cf_data["train"][:10])
    r = check(has_mat_id, "[F-6] material_id present and non-empty in train items")
    results.append(("[F] CrystalFramer material_id", r))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
for name, ok in results:
    label = PASS if ok else FAIL
    print(f"  {label} {name}")

print(f"\n  {'✓' if passed == total else '✗'} {passed}/{total} checks passed")
if passed < total:
    sys.exit(1)
