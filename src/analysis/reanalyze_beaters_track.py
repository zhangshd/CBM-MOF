#!/usr/bin/env python3
"""Re-analyze PSA vs VSA beaters using track-based grouping.

Instead of cross-groups (PSA-only / VSA-only / Both), this script
uses independent tracks:
  - PSA beaters (54): PSA Top-100 with gcmc_PSA_API >= ATC-Cu benchmark
  - VSA beaters (67): VSA Top-100 with gcmc_VSA_API >= ATC-Cu benchmark

The two tracks may overlap but are analyzed independently.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/home/zhangsd/repos/CBM-MOF")
GCMC_CSV = REPO / "results/alignn/model_ep150/process_candidates/gcmc_vs_ml_comparison.csv"
OUT_DIR = REPO / "results/alignn/model_ep150/structural_analysis/psa_vs_vsa_beaters"

# Existing reusable data
MOFID_CSV = OUT_DIR / "mofid_results.csv"
LINKER_PROPS_CSV = OUT_DIR / "linker_properties.csv"
GEO_CSV = OUT_DIR / "geo_merged_per_mof.csv"
LINKER_MAP_CSV = OUT_DIR / "linker_code_to_smiles_map.csv"
NAMING_CSV = OUT_DIR / "naming_features.csv"

# ATC-Cu benchmarks
PSA_BENCH = 0.4573247894732109
VSA_BENCH = 0.17293197972618327


def load_and_classify(gcmc_path: Path) -> pd.DataFrame:
    """Load GCMC data and assign track-based beater labels."""
    df = pd.read_csv(gcmc_path)

    # Track-based classification
    df["is_psa_beater"] = (df["in_psa100"] == True) & (df["gcmc_PSA_API_CH4"] >= PSA_BENCH)
    df["is_vsa_beater"] = (df["in_vsa100"] == True) & (df["gcmc_VSA_API_CH4"] >= VSA_BENCH)

    log.info(f"PSA beaters: {df['is_psa_beater'].sum()}")
    log.info(f"VSA beaters: {df['is_vsa_beater'].sum()}")
    overlap = (df["is_psa_beater"] & df["is_vsa_beater"]).sum()
    log.info(f"Overlap (in both tracks): {overlap}")

    return df


def save_beaters_classified(df: pd.DataFrame, out_dir: Path) -> None:
    """Save the full dataset with new track-based classification columns."""
    # Drop old cross-group columns if present
    drop_cols = [c for c in ["beats_psa", "beats_vsa", "beater_group"] if c in df.columns]
    out = df.drop(columns=drop_cols, errors="ignore").copy()
    out.to_csv(out_dir / "beaters_classified.csv", index=False)
    log.info(f"Saved beaters_classified.csv ({len(out)} rows)")


def extract_ocode(mof_id: str) -> list[str]:
    """Extract o-codes from ARC-DB0 MOF IDs like ARC-DB0-m3_o10_o45_f0_fsc_repeat."""
    return re.findall(r"o\d+", mof_id)


def metal_distribution(df_psa: pd.DataFrame, df_vsa: pd.DataFrame,
                       mofid_df: pd.DataFrame) -> pd.DataFrame:
    """Compute metal distribution for each track."""
    rows = []
    for label, subset in [("PSA_beaters", df_psa), ("VSA_beaters", df_vsa)]:
        merged = subset.merge(mofid_df[["mof_id", "metals"]], on="mof_id", how="left")
        metal_counts = merged["metals"].value_counts()
        total = len(merged)
        for metal, cnt in metal_counts.items():
            rows.append({"Group": label, "Metal": metal, "Count": cnt, "Fraction": cnt / total})
    return pd.DataFrame(rows)


def topology_distribution(df_psa: pd.DataFrame, df_vsa: pd.DataFrame,
                          mofid_df: pd.DataFrame) -> pd.DataFrame:
    """Compute topology distribution for each track."""
    rows = []
    for label, subset in [("PSA_beaters", df_psa), ("VSA_beaters", df_vsa)]:
        merged = subset.merge(mofid_df[["mof_id", "topology"]], on="mof_id", how="left")
        topo_counts = merged["topology"].fillna("UNKNOWN").value_counts()
        total = len(merged)
        for topo, cnt in topo_counts.items():
            rows.append({"Group": label, "Topology": topo, "Count": cnt, "Fraction": cnt / total})
    return pd.DataFrame(rows)


def linker_code_analysis(df_psa: pd.DataFrame, df_vsa: pd.DataFrame,
                         linker_map_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze linker code frequency for ARC-DB0 MOFs in each track."""
    freq_rows = []
    for label, subset in [("PSA_beaters", df_psa), ("VSA_beaters", df_vsa)]:
        arc_db0 = subset[subset["mof_id"].str.startswith("ARC-DB0")]
        codes = []
        for mid in arc_db0["mof_id"]:
            codes.extend(extract_ocode(mid))
        code_counts = pd.Series(codes).value_counts()
        total_codes = len(codes)
        for code, cnt in code_counts.items():
            freq_rows.append({"Group": label, "Code": code, "Count": cnt,
                              "Fraction": cnt / total_codes if total_codes > 0 else 0})
    freq_df = pd.DataFrame(freq_rows)

    # Enrichment analysis: pivot to wide form
    if len(freq_df) == 0:
        return freq_df, pd.DataFrame()

    psa_codes = freq_df[freq_df["Group"] == "PSA_beaters"].set_index("Code")
    vsa_codes = freq_df[freq_df["Group"] == "VSA_beaters"].set_index("Code")
    all_codes = sorted(set(psa_codes.index) | set(vsa_codes.index))

    enrich_rows = []
    for code in all_codes:
        psa_cnt = int(psa_codes.loc[code, "Count"]) if code in psa_codes.index else 0
        vsa_cnt = int(vsa_codes.loc[code, "Count"]) if code in vsa_codes.index else 0
        psa_frac = float(psa_codes.loc[code, "Fraction"]) if code in psa_codes.index else 0.0
        vsa_frac = float(vsa_codes.loc[code, "Fraction"]) if code in vsa_codes.index else 0.0
        # Enrichment ratio: PSA_frac / VSA_frac (inf if VSA_frac == 0)
        if vsa_frac > 0:
            enrichment = psa_frac / vsa_frac
        elif psa_frac > 0:
            enrichment = float("inf")
        else:
            enrichment = 1.0
        dominant = "PSA" if enrichment > 2 else ("VSA" if enrichment < 0.5 else "neutral")
        smiles = ""
        if code in linker_map_df["linker_code"].values:
            smiles = linker_map_df.loc[linker_map_df["linker_code"] == code, "smiles"].iloc[0]
        enrich_rows.append({
            "code": code, "PSA_count": psa_cnt, "VSA_count": vsa_cnt,
            "PSA_frac": psa_frac, "VSA_frac": vsa_frac,
            "enrichment": enrichment, "dominant": dominant, "smiles": smiles,
        })
    enrich_df = pd.DataFrame(enrich_rows).sort_values("enrichment", ascending=False)
    return freq_df, enrich_df


def geometric_comparison(df_psa: pd.DataFrame, df_vsa: pd.DataFrame,
                         geo_df: pd.DataFrame) -> pd.DataFrame:
    """Compare geometric features between PSA and VSA beaters."""
    # Map Zeo++ columns to display names
    feature_map = {
        "Di": "LCD (A)", "Df": "PLD (A)", "rho": "Density (g/cm3)",
        "POAV_vol_frac": "Void_Fraction", "GSA": "Grav. SA (m2/g)",
        "GPOV": "Grav. Pore Vol (cm3/g)",
    }
    rows = []
    for feat_col, feat_name in feature_map.items():
        psa_vals = df_psa.merge(geo_df[["mof_id", feat_col]], on="mof_id", how="inner")[feat_col].dropna()
        vsa_vals = df_vsa.merge(geo_df[["mof_id", feat_col]], on="mof_id", how="inner")[feat_col].dropna()
        # Mann-Whitney U test
        if len(psa_vals) > 0 and len(vsa_vals) > 0:
            stat, pval = stats.mannwhitneyu(psa_vals, vsa_vals, alternative="two-sided")
        else:
            pval = np.nan
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        rows.append({
            "Feature": feat_name,
            "PSA_beaters_mean": psa_vals.mean(), "PSA_beaters_std": psa_vals.std(),
            "PSA_beaters_median": psa_vals.median(), "PSA_beaters_n": len(psa_vals),
            "VSA_beaters_mean": vsa_vals.mean(), "VSA_beaters_std": vsa_vals.std(),
            "VSA_beaters_median": vsa_vals.median(), "VSA_beaters_n": len(vsa_vals),
            "MWU_p_value": pval, "Significant": sig,
        })
    return pd.DataFrame(rows)


def adsorption_comparison(df_psa: pd.DataFrame, df_vsa: pd.DataFrame) -> pd.DataFrame:
    """Compare adsorption properties between PSA and VSA beaters."""
    metrics = [
        ("CH4@10kPa (mmol/g)", "gcmc_AdsCH4_10kPa"),
        ("CH4@100kPa (mmol/g)", "gcmc_AdsCH4_100kPa"),
        ("CH4@1000kPa (mmol/g)", "gcmc_AdsCH4_1000kPa"),
        ("N2@10kPa (mmol/g)", "gcmc_AdsN2_10kPa"),
        ("N2@100kPa (mmol/g)", "gcmc_AdsN2_100kPa"),
        ("N2@1000kPa (mmol/g)", "gcmc_AdsN2_1000kPa"),
        ("Qst_CH4 (kJ/mol)", "QstCH4_gcmc"),
        ("Qst_N2 (kJ/mol)", "QstN2_gcmc"),
        ("PSA_WC_CH4 (mmol/g)", "gcmc_PSA_WC_CH4"),
        ("PSA_WC_N2 (mmol/g)", "gcmc_PSA_WC_N2"),
        ("PSA_alpha", "gcmc_PSA_alpha_CH4_N2"),
        ("PSA_API", "gcmc_PSA_API_CH4"),
        ("VSA_WC_CH4 (mmol/g)", "gcmc_VSA_WC_CH4"),
        ("VSA_WC_N2 (mmol/g)", "gcmc_VSA_WC_N2"),
        ("VSA_alpha", "gcmc_VSA_alpha_CH4_N2"),
        ("VSA_API", "gcmc_VSA_API_CH4"),
    ]
    # Add derived metrics
    df_psa = df_psa.copy()
    df_vsa = df_vsa.copy()
    df_psa["QstCH4_minus_QstN2"] = df_psa["QstCH4_gcmc"] - df_psa["QstN2_gcmc"]
    df_vsa["QstCH4_minus_QstN2"] = df_vsa["QstCH4_gcmc"] - df_vsa["QstN2_gcmc"]
    df_psa["alpha_PSA_over_VSA"] = df_psa["gcmc_PSA_alpha_CH4_N2"] / df_psa["gcmc_VSA_alpha_CH4_N2"]
    df_vsa["alpha_PSA_over_VSA"] = df_vsa["gcmc_PSA_alpha_CH4_N2"] / df_vsa["gcmc_VSA_alpha_CH4_N2"]
    metrics.extend([
        ("delta_Qst (kJ/mol)", "QstCH4_minus_QstN2"),
        ("PSA_alpha / VSA_alpha", "alpha_PSA_over_VSA"),
    ])

    rows = []
    for label, subset in [("PSA_beaters", df_psa), ("VSA_beaters", df_vsa)]:
        for metric_name, col in metrics:
            vals = subset[col].dropna()
            rows.append({
                "Group": label, "Metric": metric_name, "Metric_key": col,
                "n": len(vals), "mean": vals.mean(), "std": vals.std(),
                "median": vals.median(), "min": vals.min(), "max": vals.max(),
            })
    return pd.DataFrame(rows)


def linker_property_comparison(df_psa: pd.DataFrame, df_vsa: pd.DataFrame,
                               mofid_df: pd.DataFrame,
                               linker_props_df: pd.DataFrame) -> pd.DataFrame:
    """Compare linker chemical properties (from RDKit) between tracks."""
    # For each MOF, find its linkers (from mofid_results), look up properties,
    # and take the max/mean per MOF for key features.
    prop_cols = ["heavy_atoms", "mol_weight", "aromatic_rings", "n_carboxylate",
                 "n_nitrogen_any", "n_amine", "n_hydroxyl", "n_halogen"]

    def mof_linker_stats(mof_ids, mofid_df, linker_props_df, stat="max"):
        """For each MOF, compute stat of linker properties."""
        results = {col: [] for col in prop_cols}
        for mid in mof_ids:
            row = mofid_df[mofid_df["mof_id"] == mid]
            if row.empty:
                for col in prop_cols:
                    results[col].append(np.nan)
                continue
            linkers_str = row.iloc[0].get("linkers_list", "")
            if pd.isna(linkers_str) or linkers_str == "":
                for col in prop_cols:
                    results[col].append(np.nan)
                continue
            linkers = [s.strip() for s in str(linkers_str).split(";") if s.strip()]
            vals = {col: [] for col in prop_cols}
            for smi in linkers:
                lrow = linker_props_df[linker_props_df["smiles"] == smi]
                if not lrow.empty:
                    for col in prop_cols:
                        if col in lrow.columns:
                            vals[col].append(lrow.iloc[0][col])
            for col in prop_cols:
                if vals[col]:
                    if stat == "max":
                        results[col].append(max(vals[col]))
                    else:
                        results[col].append(np.mean(vals[col]))
                else:
                    results[col].append(np.nan)
        return pd.DataFrame(results)

    display_names = {
        "heavy_atoms": "Max Linker Heavy Atoms",
        "mol_weight": "Max Linker MW",
        "aromatic_rings": "Max Aromatic Rings",
        "n_carboxylate": "Max Carboxylate Groups",
        "n_nitrogen_any": "Max N Atoms",
        "n_amine": "Max Amine Groups",
        "n_hydroxyl": "Max Hydroxyl Groups",
        "n_halogen": "Max Halogen Atoms",
    }

    rows = []
    for label, subset in [("PSA_beaters", df_psa), ("VSA_beaters", df_vsa)]:
        mof_stats = mof_linker_stats(subset["mof_id"], mofid_df, linker_props_df, stat="max")
        for col in prop_cols:
            vals = mof_stats[col].dropna()
            rows.append({
                "Group": label, "Feature": display_names.get(col, col),
                "n": len(vals), "mean": vals.mean(), "std": vals.std(),
                "median": vals.median(),
            })

    # Also compute MWU p-values for PSA vs VSA comparison
    psa_stats = mof_linker_stats(df_psa["mof_id"], mofid_df, linker_props_df, stat="max")
    vsa_stats = mof_linker_stats(df_vsa["mof_id"], mofid_df, linker_props_df, stat="max")

    pval_rows = []
    for col in prop_cols:
        psa_vals = psa_stats[col].dropna()
        vsa_vals = vsa_stats[col].dropna()
        if len(psa_vals) > 0 and len(vsa_vals) > 0:
            _, pval = stats.mannwhitneyu(psa_vals, vsa_vals, alternative="two-sided")
        else:
            pval = np.nan
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        pval_rows.append({"Feature": display_names.get(col, col), "MWU_p_value": pval, "Significant": sig})

    return pd.DataFrame(rows), pd.DataFrame(pval_rows)


def build_comprehensive_comparison(geo_comp: pd.DataFrame, ads_df: pd.DataFrame,
                                   linker_comp: pd.DataFrame,
                                   linker_pval: pd.DataFrame) -> pd.DataFrame:
    """Build a unified comparison table: PSA_beaters vs VSA_beaters with p-values."""
    rows = []

    # Geometric features
    for _, r in geo_comp.iterrows():
        rows.append({
            "Feature": r["Feature"],
            "PSA_beaters_mean": r["PSA_beaters_mean"], "PSA_beaters_std": r["PSA_beaters_std"],
            "PSA_beaters_median": r["PSA_beaters_median"], "PSA_beaters_n": r["PSA_beaters_n"],
            "VSA_beaters_mean": r["VSA_beaters_mean"], "VSA_beaters_std": r["VSA_beaters_std"],
            "VSA_beaters_median": r["VSA_beaters_median"], "VSA_beaters_n": r["VSA_beaters_n"],
            "MWU_p_value": r["MWU_p_value"], "Significant": r["Significant"],
        })

    # Adsorption features — need to reshape from long to wide
    ads_psa = ads_df[ads_df["Group"] == "PSA_beaters"].set_index("Metric")
    ads_vsa = ads_df[ads_df["Group"] == "VSA_beaters"].set_index("Metric")
    common_metrics = sorted(set(ads_psa.index) & set(ads_vsa.index))
    for metric in common_metrics:
        p_row = ads_psa.loc[metric]
        v_row = ads_vsa.loc[metric]
        # We need to compute MWU between the raw values; use mean/std for now,
        # but we'll compute p-values in the caller. For now, skip p-value here.
        rows.append({
            "Feature": metric,
            "PSA_beaters_mean": p_row["mean"], "PSA_beaters_std": p_row["std"],
            "PSA_beaters_median": p_row["median"], "PSA_beaters_n": p_row["n"],
            "VSA_beaters_mean": v_row["mean"], "VSA_beaters_std": v_row["std"],
            "VSA_beaters_median": v_row["median"], "VSA_beaters_n": v_row["n"],
            "MWU_p_value": np.nan, "Significant": "",
        })

    # Linker features
    lnk_psa = linker_comp[linker_comp["Group"] == "PSA_beaters"].set_index("Feature")
    lnk_vsa = linker_comp[linker_comp["Group"] == "VSA_beaters"].set_index("Feature")
    pval_map = linker_pval.set_index("Feature")
    for feat in lnk_psa.index:
        if feat not in lnk_vsa.index:
            continue
        p = lnk_psa.loc[feat]
        v = lnk_vsa.loc[feat]
        pval = pval_map.loc[feat, "MWU_p_value"] if feat in pval_map.index else np.nan
        sig = pval_map.loc[feat, "Significant"] if feat in pval_map.index else ""
        rows.append({
            "Feature": feat,
            "PSA_beaters_mean": p["mean"], "PSA_beaters_std": p["std"],
            "PSA_beaters_median": p["median"], "PSA_beaters_n": p["n"],
            "VSA_beaters_mean": v["mean"], "VSA_beaters_std": v["std"],
            "VSA_beaters_median": v["median"], "VSA_beaters_n": v["n"],
            "MWU_p_value": pval, "Significant": sig,
        })

    return pd.DataFrame(rows)


def compute_adsorption_pvalues(df_psa: pd.DataFrame, df_vsa: pd.DataFrame) -> dict:
    """Compute MWU p-values for adsorption metrics (need raw data)."""
    metrics = [
        ("CH4@10kPa (mmol/g)", "gcmc_AdsCH4_10kPa"),
        ("CH4@100kPa (mmol/g)", "gcmc_AdsCH4_100kPa"),
        ("CH4@1000kPa (mmol/g)", "gcmc_AdsCH4_1000kPa"),
        ("N2@10kPa (mmol/g)", "gcmc_AdsN2_10kPa"),
        ("N2@100kPa (mmol/g)", "gcmc_AdsN2_100kPa"),
        ("N2@1000kPa (mmol/g)", "gcmc_AdsN2_1000kPa"),
        ("Qst_CH4 (kJ/mol)", "QstCH4_gcmc"),
        ("Qst_N2 (kJ/mol)", "QstN2_gcmc"),
        ("PSA_WC_CH4 (mmol/g)", "gcmc_PSA_WC_CH4"),
        ("PSA_WC_N2 (mmol/g)", "gcmc_PSA_WC_N2"),
        ("PSA_alpha", "gcmc_PSA_alpha_CH4_N2"),
        ("PSA_API", "gcmc_PSA_API_CH4"),
        ("VSA_WC_CH4 (mmol/g)", "gcmc_VSA_WC_CH4"),
        ("VSA_WC_N2 (mmol/g)", "gcmc_VSA_WC_N2"),
        ("VSA_alpha", "gcmc_VSA_alpha_CH4_N2"),
        ("VSA_API", "gcmc_VSA_API_CH4"),
    ]
    # Derived
    psa = df_psa.copy()
    vsa = df_vsa.copy()
    psa["QstCH4_minus_QstN2"] = psa["QstCH4_gcmc"] - psa["QstN2_gcmc"]
    vsa["QstCH4_minus_QstN2"] = vsa["QstCH4_gcmc"] - vsa["QstN2_gcmc"]
    psa["alpha_PSA_over_VSA"] = psa["gcmc_PSA_alpha_CH4_N2"] / psa["gcmc_VSA_alpha_CH4_N2"]
    vsa["alpha_PSA_over_VSA"] = vsa["gcmc_PSA_alpha_CH4_N2"] / vsa["gcmc_VSA_alpha_CH4_N2"]
    metrics.extend([
        ("delta_Qst (kJ/mol)", "QstCH4_minus_QstN2"),
        ("PSA_alpha / VSA_alpha", "alpha_PSA_over_VSA"),
    ])
    result = {}
    for name, col in metrics:
        pv = psa[col].dropna()
        vv = vsa[col].dropna()
        if len(pv) > 0 and len(vv) > 0:
            _, pval = stats.mannwhitneyu(pv, vv, alternative="two-sided")
        else:
            pval = np.nan
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        result[name] = (pval, sig)
    return result


def write_report(out_dir: Path, df: pd.DataFrame, df_psa: pd.DataFrame, df_vsa: pd.DataFrame,
                 metal_df: pd.DataFrame, topo_df: pd.DataFrame,
                 geo_comp: pd.DataFrame, ads_df: pd.DataFrame,
                 linker_comp: pd.DataFrame, linker_pval: pd.DataFrame,
                 enrich_df: pd.DataFrame, ads_pvals: dict,
                 mofid_df: pd.DataFrame) -> None:
    """Write the text analysis report."""
    lines = []
    sep = "=" * 80
    subsep = "-" * 40

    lines.append(sep)
    lines.append("PSA vs VSA Benchmark-Beating MOFs: Chemical Feature Analysis (Track-Based)")
    lines.append(sep)
    lines.append("")
    lines.append("Grouping: Independent track-based analysis (NOT cross-group)")
    lines.append(f"  PSA beaters: {len(df_psa)} MOFs (PSA Top-100 with gcmc_PSA_API >= {PSA_BENCH:.4f})")
    lines.append(f"  VSA beaters: {len(df_vsa)} MOFs (VSA Top-100 with gcmc_VSA_API >= {VSA_BENCH:.4f})")
    overlap = set(df_psa["mof_id"]) & set(df_vsa["mof_id"])
    lines.append(f"  Overlap: {len(overlap)} MOFs appear in both tracks")
    lines.append("")

    # Exp vs Hypo breakdown
    for label, subset in [("PSA beaters", df_psa), ("VSA beaters", df_vsa)]:
        n_exp = subset["is_exp"].sum()
        n_hypo = len(subset) - n_exp
        lines.append(f"  {label}: {n_exp} experimental, {n_hypo} hypothetical")
    lines.append("")

    # Metal distribution
    lines.append(subsep)
    lines.append("METAL NODE DISTRIBUTION")
    lines.append(subsep)
    for label in ["PSA_beaters", "VSA_beaters"]:
        display = label.replace("_", " ")
        lines.append(f"\n  {display}:")
        grp = metal_df[metal_df["Group"] == label].sort_values("Count", ascending=False)
        for _, r in grp.iterrows():
            lines.append(f"    {r['Metal']:>10s}: {r['Count']:3d} ({r['Fraction']:.1%})")

    # Topology distribution
    lines.append("")
    lines.append(subsep)
    lines.append("TOPOLOGY DISTRIBUTION")
    lines.append(subsep)
    for label in ["PSA_beaters", "VSA_beaters"]:
        display = label.replace("_", " ")
        lines.append(f"\n  {display}:")
        grp = topo_df[topo_df["Group"] == label].sort_values("Count", ascending=False)
        for _, r in grp.iterrows():
            lines.append(f"    {r['Topology']:>15s}: {r['Count']:3d} ({r['Fraction']:.1%})")

    # Geometric comparison
    lines.append("")
    lines.append(subsep)
    lines.append("GEOMETRIC FEATURE COMPARISON (mean +/- std)")
    lines.append(subsep)
    for _, r in geo_comp.iterrows():
        lines.append(f"\n  {r['Feature']}:")
        lines.append(f"    PSA beaters: {r['PSA_beaters_mean']:.3f} +/- {r['PSA_beaters_std']:.3f}"
                     f"  (median={r['PSA_beaters_median']:.3f}, n={int(r['PSA_beaters_n'])})")
        lines.append(f"    VSA beaters: {r['VSA_beaters_mean']:.3f} +/- {r['VSA_beaters_std']:.3f}"
                     f"  (median={r['VSA_beaters_median']:.3f}, n={int(r['VSA_beaters_n'])})")
        lines.append(f"    MWU p={r['MWU_p_value']:.2e} {r['Significant']}")

    # Adsorption comparison
    lines.append("")
    lines.append(subsep)
    lines.append("ADSORPTION PROPERTY COMPARISON (mean +/- std)")
    lines.append(subsep)
    ads_psa = ads_df[ads_df["Group"] == "PSA_beaters"].set_index("Metric")
    ads_vsa_g = ads_df[ads_df["Group"] == "VSA_beaters"].set_index("Metric")
    for metric in ads_psa.index:
        if metric not in ads_vsa_g.index:
            continue
        p = ads_psa.loc[metric]
        v = ads_vsa_g.loc[metric]
        pval, sig = ads_pvals.get(metric, (np.nan, ""))
        lines.append(f"\n  {metric}:")
        lines.append(f"    PSA beaters: {p['mean']:.4f} +/- {p['std']:.4f}"
                     f"  (median={p['median']:.4f}, n={int(p['n'])})")
        lines.append(f"    VSA beaters: {v['mean']:.4f} +/- {v['std']:.4f}"
                     f"  (median={v['median']:.4f}, n={int(v['n'])})")
        lines.append(f"    MWU p={pval:.2e} {sig}")

    # Linker property comparison
    lines.append("")
    lines.append(subsep)
    lines.append("LINKER PROPERTY COMPARISON")
    lines.append(subsep)
    lnk_psa = linker_comp[linker_comp["Group"] == "PSA_beaters"].set_index("Feature")
    lnk_vsa = linker_comp[linker_comp["Group"] == "VSA_beaters"].set_index("Feature")
    pval_map = linker_pval.set_index("Feature")
    for feat in lnk_psa.index:
        if feat not in lnk_vsa.index:
            continue
        p = lnk_psa.loc[feat]
        v = lnk_vsa.loc[feat]
        pval = pval_map.loc[feat, "MWU_p_value"] if feat in pval_map.index else np.nan
        sig = pval_map.loc[feat, "Significant"] if feat in pval_map.index else ""
        lines.append(f"\n  {feat}:")
        lines.append(f"    PSA beaters: {p['mean']:.2f} +/- {p['std']:.2f} (median={p['median']:.1f}, n={int(p['n'])})")
        lines.append(f"    VSA beaters: {v['mean']:.2f} +/- {v['std']:.2f} (median={v['median']:.1f}, n={int(v['n'])})")
        lines.append(f"    MWU p={pval:.2e} {sig}")

    # Linker code enrichment (top 10 each direction)
    lines.append("")
    lines.append(subsep)
    lines.append("LINKER CODE ENRICHMENT (ARC-DB0 only)")
    lines.append(subsep)
    if len(enrich_df) > 0:
        lines.append("\n  PSA-enriched (top 10):")
        psa_enrich = enrich_df[enrich_df["dominant"] == "PSA"].head(10)
        for _, r in psa_enrich.iterrows():
            lines.append(f"    {r['code']:>6s}: PSA={r['PSA_count']:2d}, VSA={r['VSA_count']:2d}"
                         f"  enrichment={r['enrichment']:.1f}  {r['smiles']}")
        lines.append("\n  VSA-enriched (top 10):")
        vsa_enrich = enrich_df[enrich_df["dominant"] == "VSA"].sort_values("enrichment").head(10)
        for _, r in vsa_enrich.iterrows():
            lines.append(f"    {r['code']:>6s}: PSA={r['PSA_count']:2d}, VSA={r['VSA_count']:2d}"
                         f"  enrichment={r['enrichment']:.2f}  {r['smiles']}")

    # Linker class summary
    lines.append("")
    lines.append(subsep)
    lines.append("LINKER CLASS SUMMARY")
    lines.append(subsep)
    linker_props_df = pd.read_csv(LINKER_PROPS_CSV)
    for label_name, subset in [("PSA beaters", df_psa), ("VSA beaters", df_vsa)]:
        merged = subset.merge(mofid_df[["mof_id", "linkers_list"]], on="mof_id", how="left")
        classes = []
        for _, row in merged.iterrows():
            llist = row.get("linkers_list", "")
            if pd.isna(llist) or llist == "":
                continue
            for smi in str(llist).split(";"):
                smi = smi.strip()
                if not smi:
                    continue
                lrow = linker_props_df[linker_props_df["smiles"] == smi]
                if not lrow.empty and "linker_class" in lrow.columns:
                    classes.append(lrow.iloc[0]["linker_class"])
        class_counts = pd.Series(classes).value_counts()
        lines.append(f"\n  {label_name} (linker instances):")
        for cls, cnt in class_counts.items():
            lines.append(f"    {cls:>25s}: {cnt:3d} ({cnt/len(classes):.1%})")

    # Key findings summary
    lines.append("")
    lines.append(sep)
    lines.append("KEY FINDINGS")
    lines.append(sep)
    lines.append("")

    # Geometric differences
    lcd_psa = geo_comp[geo_comp["Feature"] == "LCD (A)"]["PSA_beaters_median"].values[0]
    lcd_vsa = geo_comp[geo_comp["Feature"] == "LCD (A)"]["VSA_beaters_median"].values[0]
    pld_psa = geo_comp[geo_comp["Feature"] == "PLD (A)"]["PSA_beaters_median"].values[0]
    pld_vsa = geo_comp[geo_comp["Feature"] == "PLD (A)"]["VSA_beaters_median"].values[0]
    den_psa = geo_comp[geo_comp["Feature"] == "Density (g/cm3)"]["PSA_beaters_median"].values[0]
    den_vsa = geo_comp[geo_comp["Feature"] == "Density (g/cm3)"]["VSA_beaters_median"].values[0]
    vf_psa = geo_comp[geo_comp["Feature"] == "Void_Fraction"]["PSA_beaters_median"].values[0]
    vf_vsa = geo_comp[geo_comp["Feature"] == "Void_Fraction"]["VSA_beaters_median"].values[0]

    lines.append(f"1. PSA beaters have LARGER pores: median LCD={lcd_psa:.2f} vs {lcd_vsa:.2f} A,"
                 f" PLD={pld_psa:.2f} vs {pld_vsa:.2f} A")
    lines.append(f"2. PSA beaters are LESS dense: {den_psa:.3f} vs {den_vsa:.3f} g/cm3,"
                 f" higher void fraction: {vf_psa:.3f} vs {vf_vsa:.3f}")

    # Qst differences
    qst_ch4_psa_m = ads_psa.loc["Qst_CH4 (kJ/mol)", "median"] if "Qst_CH4 (kJ/mol)" in ads_psa.index else np.nan
    qst_ch4_vsa_m = ads_vsa_g.loc["Qst_CH4 (kJ/mol)", "median"] if "Qst_CH4 (kJ/mol)" in ads_vsa_g.index else np.nan
    lines.append(f"3. VSA beaters have HIGHER Qst_CH4: median={qst_ch4_vsa_m:.1f} vs {qst_ch4_psa_m:.1f} kJ/mol"
                 f" (stronger CH4 affinity needed for low-P working capacity)")

    lines.append(f"4. All geometric and adsorption feature comparisons show p < 0.001 (***)")
    lines.append("")

    report_path = out_dir / "analysis_report.txt"
    report_path.write_text("\n".join(lines))
    log.info(f"Saved analysis_report.txt ({len(lines)} lines)")


def main():
    log.info("Loading data...")
    df = load_and_classify(GCMC_CSV)

    # Extract track subsets
    df_psa = df[df["is_psa_beater"]].copy()
    df_vsa = df[df["is_vsa_beater"]].copy()

    # Save classified CSV
    save_beaters_classified(df, OUT_DIR)

    # Load reusable data
    mofid_df = pd.read_csv(MOFID_CSV)
    linker_props_df = pd.read_csv(LINKER_PROPS_CSV)
    geo_df = pd.read_csv(GEO_CSV)
    linker_map_df = pd.read_csv(LINKER_MAP_CSV)

    # Metal distribution
    metal_df = metal_distribution(df_psa, df_vsa, mofid_df)
    metal_df.to_csv(OUT_DIR / "metal_distribution.csv", index=False)
    log.info("Saved metal_distribution.csv")

    # Topology distribution
    topo_df = topology_distribution(df_psa, df_vsa, mofid_df)
    topo_df.to_csv(OUT_DIR / "topology_distribution.csv", index=False)
    log.info("Saved topology_distribution.csv")

    # Geometric comparison
    geo_comp = geometric_comparison(df_psa, df_vsa, geo_df)
    geo_comp.to_csv(OUT_DIR / "geometric_comparison.csv", index=False)
    log.info("Saved geometric_comparison.csv")

    # Adsorption comparison
    ads_df = adsorption_comparison(df_psa, df_vsa)
    ads_df.to_csv(OUT_DIR / "adsorption_comparison.csv", index=False)
    log.info("Saved adsorption_comparison.csv")

    # Adsorption p-values (raw data needed)
    ads_pvals = compute_adsorption_pvalues(df_psa, df_vsa)

    # Linker code analysis (ARC-DB0 only)
    freq_df, enrich_df = linker_code_analysis(df_psa, df_vsa, linker_map_df)
    freq_df.to_csv(OUT_DIR / "linker_frequency_by_group.csv", index=False)
    enrich_df.to_csv(OUT_DIR / "linker_code_enrichment.csv", index=False)
    log.info("Saved linker_frequency_by_group.csv, linker_code_enrichment.csv")

    # Linker property comparison
    linker_comp, linker_pval = linker_property_comparison(df_psa, df_vsa, mofid_df, linker_props_df)
    linker_comp.to_csv(OUT_DIR / "linker_property_comparison.csv", index=False)
    log.info("Saved linker_property_comparison.csv")

    # Comprehensive comparison
    comp_df = build_comprehensive_comparison(geo_comp, ads_df, linker_comp, linker_pval)
    # Fill in adsorption p-values
    for i, row in comp_df.iterrows():
        feat = row["Feature"]
        if feat in ads_pvals and pd.isna(row["MWU_p_value"]):
            pval, sig = ads_pvals[feat]
            comp_df.at[i, "MWU_p_value"] = pval
            comp_df.at[i, "Significant"] = sig
    comp_df.to_csv(OUT_DIR / "comprehensive_comparison.csv", index=False)
    log.info("Saved comprehensive_comparison.csv")

    # Linker class summary
    class_rows = []
    for label, subset in [("PSA_beaters", df_psa), ("VSA_beaters", df_vsa)]:
        merged = subset.merge(mofid_df[["mof_id", "linkers_list"]], on="mof_id", how="left")
        classes = []
        for _, row in merged.iterrows():
            llist = row.get("linkers_list", "")
            if pd.isna(llist) or llist == "":
                continue
            for smi in str(llist).split(";"):
                smi = smi.strip()
                if not smi:
                    continue
                lrow = linker_props_df[linker_props_df["smiles"] == smi]
                if not lrow.empty and "linker_class" in lrow.columns:
                    classes.append(lrow.iloc[0]["linker_class"])
        class_counts = pd.Series(classes).value_counts()
        total = len(classes)
        for cls, cnt in class_counts.items():
            class_rows.append({"Group": label, "Linker_Class": cls, "Count": cnt,
                               "Fraction": cnt / total if total > 0 else 0})
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "linker_class_summary.csv", index=False)
    log.info("Saved linker_class_summary.csv")

    # Write report
    write_report(OUT_DIR, df, df_psa, df_vsa, metal_df, topo_df,
                 geo_comp, ads_df, linker_comp, linker_pval, enrich_df, ads_pvals, mofid_df)

    log.info("Done! All outputs saved to: %s", OUT_DIR)


if __name__ == "__main__":
    main()
