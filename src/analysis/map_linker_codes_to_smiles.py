#!/usr/bin/env python
"""Map ARC-DB0 linker codes (o14, o156, etc.) to their actual SMILES from MOFid results.

For each linker code, find all MOFs using that code, run MOFid, and correlate the
linker SMILES with the linker position in the name to identify which SMILES corresponds
to which code.
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_DIR = Path("/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150/structural_analysis/psa_vs_vsa_beaters")


def main():
    naming = pd.read_csv(OUTPUT_DIR / "naming_features.csv")
    mofid = pd.read_csv(OUTPUT_DIR / "mofid_results.csv")

    # Only ARC-DB0 with successful MOFid
    arc = naming[naming["db_source"] == "ARC-DB0"].copy()
    mofid_success = mofid[mofid["status"] == "success"].copy()

    merged = arc.merge(mofid_success[["mof_id", "smiles_linkers", "linkers_list", "topology"]], on="mof_id", how="inner")

    # For ARC-DB0 with exactly 2 linker codes and 2 MOFid linkers, we can map them
    # But the mapping isn't straightforward since MOFid may reorder linkers
    # Instead: find all MOFs sharing a linker code, collect their MOFid linkers
    # and find the intersection (common linker) -> that's the one from the shared code

    # Strategy: for each linker code, collect all MOFid SMILES sets from MOFs containing that code
    # The SMILES that appears in ALL or MOST of those sets is likely the one for that code

    code_to_smiles = defaultdict(Counter)

    for _, row in merged.iterrows():
        linker_codes = [c.strip() for c in str(row["linker_codes"]).split(";") if c.strip()] if row.get("linker_codes") and not pd.isna(row["linker_codes"]) else []
        linkers_list = [s.strip() for s in str(row["linkers_list"]).split(";") if s.strip()] if row.get("linkers_list") and not pd.isna(row["linkers_list"]) else []

        # Associate each code with all linkers from this MOF
        for code in linker_codes:
            for smi in linkers_list:
                code_to_smiles[code][smi] += 1

    # For each code, the most frequent SMILES across all MOFs with that code is the best match
    # But we need to disambiguate when a MOF has 2 codes and 2 linkers

    # Refined approach: for single-linker-code MOFs, mapping is unambiguous
    single_code_map = {}
    for _, row in merged.iterrows():
        linker_codes = [c.strip() for c in str(row["linker_codes"]).split(";") if c.strip()] if row.get("linker_codes") and not pd.isna(row["linker_codes"]) else []
        linkers_list = [s.strip() for s in str(row["linkers_list"]).split(";") if s.strip()] if row.get("linkers_list") and not pd.isna(row["linkers_list"]) else []

        if len(linker_codes) == 1 and len(linkers_list) == 1:
            code = linker_codes[0]
            smi = linkers_list[0]
            if code not in single_code_map:
                single_code_map[code] = Counter()
            single_code_map[code][smi] += 1

    # For two-code MOFs: if one code is already mapped, the other linker is for the other code
    for _, row in merged.iterrows():
        linker_codes = [c.strip() for c in str(row["linker_codes"]).split(";") if c.strip()] if row.get("linker_codes") and not pd.isna(row["linker_codes"]) else []
        linkers_list = [s.strip() for s in str(row["linkers_list"]).split(";") if s.strip()] if row.get("linkers_list") and not pd.isna(row["linkers_list"]) else []

        if len(linker_codes) == 2 and len(linkers_list) == 2:
            for i, code in enumerate(linker_codes):
                other_code = linker_codes[1 - i]
                if other_code in single_code_map:
                    # The most common SMILES for the other code
                    other_smi = single_code_map[other_code].most_common(1)[0][0] if single_code_map[other_code] else None
                    remaining_smiles = [s for s in linkers_list if s != other_smi]
                    if remaining_smiles:
                        if code not in single_code_map:
                            single_code_map[code] = Counter()
                        single_code_map[code][remaining_smiles[0]] += 1

    # Use frequency-based mapping for remaining
    # Combine both approaches
    final_map = {}
    for code in sorted(code_to_smiles.keys()):
        if code in single_code_map and single_code_map[code]:
            best_smi = single_code_map[code].most_common(1)[0][0]
            confidence = single_code_map[code][best_smi]
        else:
            best_smi = code_to_smiles[code].most_common(1)[0][0]
            confidence = code_to_smiles[code][best_smi]
        final_map[code] = (best_smi, confidence)

    # Print mapping
    print("=== ARC-DB0 Linker Code -> SMILES Mapping ===")
    print()
    for code in sorted(final_map.keys(), key=lambda x: int(re.sub(r'\D', '', x) or 0)):
        smi, conf = final_map[code]
        print(f"  {code:>6s} -> {smi:<60s}  (confidence={conf})")

    # Save
    rows = []
    for code, (smi, conf) in final_map.items():
        rows.append({"linker_code": code, "smiles": smi, "confidence": conf})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "linker_code_to_smiles_map.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR / 'linker_code_to_smiles_map.csv'}")

    # Now annotate the enrichment table
    enrich = pd.read_csv(OUTPUT_DIR / "linker_code_enrichment.csv")
    enrich["smiles"] = enrich["code"].map(lambda c: final_map.get(c, (None, 0))[0])
    enrich.to_csv(OUTPUT_DIR / "linker_code_enrichment.csv", index=False)
    print("Updated linker_code_enrichment.csv with SMILES column")


if __name__ == "__main__":
    main()
