#!/usr/bin/env python
"""Compute linker code enrichment between PSA-only and VSA-only groups (ARC-DB0 MOFs)."""

import pandas as pd
import sys
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path("/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150/structural_analysis/psa_vs_vsa_beaters")


def main():
    naming = pd.read_csv(OUTPUT_DIR / "naming_features.csv")
    arc = naming[naming["db_source"] == "ARC-DB0"].copy()

    group_codes = {}
    for grp in ["PSA-only", "VSA-only", "Both"]:
        codes = set()
        for c in arc[arc["beater_group"] == grp]["linker_codes"].dropna():
            if c:
                codes.update(str(c).split(";"))
        group_codes[grp] = codes

    psa_codes = group_codes["PSA-only"]
    vsa_codes = group_codes["VSA-only"]

    print("Unique linker codes:")
    print(f"  PSA-only: {len(psa_codes)}")
    print(f"  VSA-only: {len(vsa_codes)}")
    print(f"  Intersection: {len(psa_codes & vsa_codes)}")
    print(f"  PSA exclusive: {sorted(psa_codes - vsa_codes)}")
    print(f"  VSA exclusive: {sorted(vsa_codes - psa_codes)}")
    print(f"  Shared: {sorted(psa_codes & vsa_codes)}")

    psa_subset = arc[arc["beater_group"] == "PSA-only"]
    vsa_subset = arc[arc["beater_group"] == "VSA-only"]
    psa_counter = Counter()
    vsa_counter = Counter()
    for _, row in psa_subset.iterrows():
        lc = row.get("linker_codes")
        if lc and not pd.isna(lc):
            for c in str(lc).split(";"):
                psa_counter[c] += 1
    for _, row in vsa_subset.iterrows():
        lc = row.get("linker_codes")
        if lc and not pd.isna(lc):
            for c in str(lc).split(";"):
                vsa_counter[c] += 1

    all_codes = set(psa_counter.keys()) | set(vsa_counter.keys())
    enrich = []
    for code in all_codes:
        pn = psa_counter.get(code, 0)
        vn = vsa_counter.get(code, 0)
        pf = pn / len(psa_subset)
        vf = vn / len(vsa_subset)
        enrichment = (pf + 0.01) / (vf + 0.01)
        dominant = "PSA" if enrichment > 1.5 else ("VSA" if enrichment < 0.67 else "neutral")
        enrich.append(dict(
            code=code, PSA_count=pn, VSA_count=vn,
            PSA_frac=pf, VSA_frac=vf, enrichment=enrichment, dominant=dominant
        ))

    enrich_df = pd.DataFrame(enrich).sort_values("enrichment", ascending=False)
    print("\n=== Linker Code Enrichment (total count >= 3) ===")
    for _, r in enrich_df.iterrows():
        if r["PSA_count"] + r["VSA_count"] >= 3:
            print(
                "  %6s  PSA=%2d(%4.1f%%)  VSA=%2d(%4.1f%%)  enrich=%.2f  %s"
                % (r["code"], r["PSA_count"], r["PSA_frac"] * 100,
                   r["VSA_count"], r["VSA_frac"] * 100, r["enrichment"], r["dominant"])
            )

    enrich_df.to_csv(OUTPUT_DIR / "linker_code_enrichment.csv", index=False)
    print("\nSaved to linker_code_enrichment.csv")


if __name__ == "__main__":
    main()
