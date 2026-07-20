#!/usr/bin/env python
"""Compare Cluster 8 with other benchmark-beating MOFs using existing data.

The analysis is process-specific and uses the exact PSA (n=53) and VSA (n=66)
benchmark-beating sets exported by ``run_new_top10_pipeline.py``. It reports
median/IQR summaries for pore and adsorption metrics and categorical counts for
metals, MOFid nodes, and topologies. No simulation or descriptor calculation is
performed.

Usage:
    conda run -n alignn_env python src/analysis/analyze_cluster8_beaters.py
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "results" / "alignn" / "model_ep150"
PROCESS_DIR = MODEL_DIR / "process_candidates"
STRUCTURAL_DIR = MODEL_DIR / "structural_analysis" / "psa_vs_vsa_beaters"
ZEO_CSV = REPO_ROOT / "data" / "processed" / "RAC_and_zeo_features_deduplicated.csv"
MOFID_CSV = STRUCTURAL_DIR / "mofid_results.csv"
OUTPUT_DIR = MODEL_DIR / "structural_analysis" / "cluster8_beaters"
ATC_CU_ID = "CoRE-2020[Cu][pts]3[ASR]1"

GEO_FEATURES = {
    "Df": ("PLD", "A"),
    "Di": ("LCD", "A"),
    "GSA": ("gravimetric_surface_area", "m2/g"),
    "VPOV": ("pore_volume", "cm3/cm3"),
    "POAV_vol_frac": ("void_fraction", "dimensionless"),
    "rho": ("framework_density", "g/cm3"),
}

PROCESS_FILES = {
    "PSA": ("psa_beaters.csv", "PSA_WC_CH4", "PSA_alpha_CH4_N2", "PSA_API_CH4"),
    "VSA": ("vsa_beaters.csv", "VSA_WC_CH4", "VSA_alpha_CH4_N2", "VSA_API_CH4"),
}


def load_beater_data() -> pd.DataFrame:
    """Load and merge process, geometric, and existing MOFid records."""
    frames = []
    for process, (filename, wc_col, alpha_col, api_col) in PROCESS_FILES.items():
        frame = pd.read_csv(PROCESS_DIR / filename).rename(columns={"CifId": "mof_id"})
        frame = frame[frame["mof_id"] != ATC_CU_ID].copy()
        frame["process"] = process
        frame = frame.rename(columns={
            wc_col: "working_capacity",
            alpha_col: "selectivity",
            api_col: "API",
        })
        frames.append(frame)
    beaters = pd.concat(frames, ignore_index=True)

    expected = {"PSA": 53, "VSA": 66}
    actual = beaters.groupby("process").size().to_dict()
    if actual != expected:
        raise ValueError(f"Unexpected benchmark-beater counts: {actual}; expected {expected}")

    geo = pd.read_csv(ZEO_CSV, usecols=["name", *GEO_FEATURES]).rename(
        columns={"name": "mof_id"}
    )
    mofid = pd.read_csv(
        MOFID_CSV,
        usecols=["mof_id", "smiles_nodes", "topology", "metals", "status"],
    )
    merged = beaters.merge(geo, on="mof_id", how="left", validate="many_to_one")
    merged = merged.merge(mofid, on="mof_id", how="left", validate="many_to_one")

    required = [*GEO_FEATURES, "working_capacity", "selectivity", "API"]
    missing = merged[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing required numeric values:\n{missing[missing > 0]}")
    merged["cluster_group"] = merged["Cluster"].eq(8).map(
        {True: "Cluster 8", False: "Other clusters"}
    )
    return merged


def summarize_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """Return process-specific median and IQR for Cluster 8 and all others."""
    feature_meta = {
        **GEO_FEATURES,
        "working_capacity": ("working_capacity", "mol/kg"),
        "selectivity": ("selectivity", "dimensionless"),
        "API": ("API", "dimensionless"),
    }
    rows = []
    for process, process_data in data.groupby("process", sort=False):
        for column, (metric, unit) in feature_meta.items():
            stats = {}
            for group, group_data in process_data.groupby("cluster_group", sort=False):
                values = group_data[column]
                stats[group] = {
                    "n": int(len(values)),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                }
            c8 = stats["Cluster 8"]
            other = stats["Other clusters"]
            rows.append({
                "process": process,
                "metric": metric,
                "unit": unit,
                "cluster8_n": c8["n"],
                "cluster8_median": c8["median"],
                "cluster8_q1": c8["q1"],
                "cluster8_q3": c8["q3"],
                "other_n": other["n"],
                "other_median": other["median"],
                "other_q1": other["q1"],
                "other_q3": other["q3"],
                "median_difference_cluster8_minus_other": c8["median"] - other["median"],
            })
    return pd.DataFrame(rows)


def summarize_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """Count exact metal, node, and topology labels within each comparison group."""
    rows = []
    categories = {
        "metal": "metals",
        "node": "smiles_nodes",
        "topology": "topology",
    }
    for (process, group), group_data in data.groupby(
        ["process", "cluster_group"], sort=False
    ):
        for category, column in categories.items():
            values = group_data[["mof_id", column]].dropna().copy()
            if category == "metal":
                values[column] = values[column].astype(str).str.split(";")
                values = values.explode(column)
            counts = values[column].value_counts()
            denominator = int(len(group_data))
            for label, count in counts.items():
                rows.append({
                    "process": process,
                    "cluster_group": group,
                    "category": category,
                    "label": label,
                    "count": int(count),
                    "group_n": denominator,
                    "fraction": float(count / denominator),
                })
    return pd.DataFrame(rows)


def write_report(numeric: pd.DataFrame, categorical: pd.DataFrame) -> None:
    """Write a compact, auditable Markdown summary without causal claims."""
    lines = [
        "# Cluster 8 Benchmark-Beater Comparison",
        "",
        "This report compares Cluster 8 with all other occupied clusters within the exact PSA (n=53) and VSA (n=66) benchmark-beating sets. Values are medians with interquartile ranges; associations are descriptive and do not imply causality.",
        "",
        "## Numeric summary",
        "",
        numeric.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Leading categorical labels",
        "",
    ]
    for process in ["PSA", "VSA"]:
        for group in ["Cluster 8", "Other clusters"]:
            lines.extend([f"### {process}: {group}", ""])
            subset = categorical[
                (categorical["process"] == process)
                & (categorical["cluster_group"] == group)
            ]
            top = (
                subset.sort_values(["category", "count", "label"], ascending=[True, False, True])
                .groupby("category", sort=False)
                .head(5)
            )
            lines.extend([top.to_markdown(index=False, floatfmt=".3f"), ""])
    (OUTPUT_DIR / "cluster8_beater_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_beater_data()
    numeric = summarize_numeric(data)
    categorical = summarize_categorical(data)

    data.to_csv(OUTPUT_DIR / "cluster8_beater_details.csv", index=False)
    numeric.to_csv(OUTPUT_DIR / "cluster8_numeric_summary.csv", index=False)
    categorical.to_csv(OUTPUT_DIR / "cluster8_categorical_summary.csv", index=False)
    write_report(numeric, categorical)

    counts = data.groupby(["process", "cluster_group"]).size()
    print(counts.to_string())
    print(numeric.to_string(index=False))
    print(f"Saved analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
