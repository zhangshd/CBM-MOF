#!/usr/bin/env python
"""Compare key PSA-enriched vs VSA-enriched linkers in detail using RDKit."""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("/home/zhangsd/repos/CBM-MOF/results/alignn/model_ep150/structural_analysis/psa_vs_vsa_beaters")


def main():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        print("RDKit not available")
        return

    enrich = pd.read_csv(OUTPUT_DIR / "linker_code_enrichment.csv")

    # Key PSA-enriched and VSA-enriched linker codes (count>=3)
    psa_enriched = enrich[(enrich["dominant"] == "PSA") & (enrich["PSA_count"] + enrich["VSA_count"] >= 3)]
    vsa_enriched = enrich[(enrich["dominant"] == "VSA") & (enrich["PSA_count"] + enrich["VSA_count"] >= 3)]

    print("=== Key PSA-enriched Linkers ===")
    for _, row in psa_enriched.iterrows():
        code = row["code"]
        smi = row.get("smiles", "")
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    mol = None
        if mol is None:
            print(f"  {code}: SMILES={smi} (unparseable)")
            continue

        ha = mol.GetNumHeavyAtoms()
        ar = rdMolDescriptors.CalcNumAromaticRings(mol)
        nr = rdMolDescriptors.CalcNumRings(mol)
        mw = Descriptors.MolWt(mol)

        # Count carboxylates
        carb_pat = Chem.MolFromSmarts("[CX3](=O)[OX1,OX2H1]")
        n_carb = len(mol.GetSubstructMatches(carb_pat)) if carb_pat else 0

        # Count N atoms
        n_nitrogen = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")

        # Count S atoms
        n_sulfur = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "S")

        # Classify
        if n_carb >= 3:
            cls = "polycarboxylate"
        elif n_carb == 2:
            cls = "dicarboxylate"
        elif n_carb == 1:
            cls = "monocarboxylate"
        elif n_nitrogen > 0:
            cls = "N-donor"
        else:
            cls = "other"

        print(f"  {code:>6s}: heavy={ha:2d} arom_rings={ar} total_rings={nr} "
              f"MW={mw:6.1f} carb={n_carb} N={n_nitrogen} S={n_sulfur} "
              f"class={cls:15s} PSA={int(row['PSA_count'])} VSA={int(row['VSA_count'])}")
        print(f"         SMILES: {smi}")

    print("\n=== Key VSA-enriched Linkers ===")
    for _, row in vsa_enriched.iterrows():
        code = row["code"]
        smi = row.get("smiles", "")
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    mol = None
        if mol is None:
            print(f"  {code}: SMILES={smi} (unparseable)")
            continue

        ha = mol.GetNumHeavyAtoms()
        ar = rdMolDescriptors.CalcNumAromaticRings(mol)
        nr = rdMolDescriptors.CalcNumRings(mol)
        mw = Descriptors.MolWt(mol)
        carb_pat = Chem.MolFromSmarts("[CX3](=O)[OX1,OX2H1]")
        n_carb = len(mol.GetSubstructMatches(carb_pat)) if carb_pat else 0
        n_nitrogen = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
        n_sulfur = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "S")

        if n_carb >= 3:
            cls = "polycarboxylate"
        elif n_carb == 2:
            cls = "dicarboxylate"
        elif n_carb == 1:
            cls = "monocarboxylate"
        elif n_nitrogen > 0:
            cls = "N-donor"
        else:
            cls = "other"

        print(f"  {code:>6s}: heavy={ha:2d} arom_rings={ar} total_rings={nr} "
              f"MW={mw:6.1f} carb={n_carb} N={n_nitrogen} S={n_sulfur} "
              f"class={cls:15s} PSA={int(row['PSA_count'])} VSA={int(row['VSA_count'])}")
        print(f"         SMILES: {smi}")

    # Summary: average linker size for PSA-enriched vs VSA-enriched
    print("\n=== Summary: PSA-enriched vs VSA-enriched Linker Properties ===")
    for label, subset_df in [("PSA-enriched", psa_enriched), ("VSA-enriched", vsa_enriched)]:
        heavys, aroms, mws, carbs, nitrogens = [], [], [], [], []
        for _, row in subset_df.iterrows():
            smi = row.get("smiles", "")
            if not smi or pd.isna(smi):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                if mol:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        mol = None
            if mol is None:
                continue
            # Weight by usage count
            count = row["PSA_count"] + row["VSA_count"]
            heavys.extend([mol.GetNumHeavyAtoms()] * int(count))
            aroms.extend([rdMolDescriptors.CalcNumAromaticRings(mol)] * int(count))
            mws.extend([Descriptors.MolWt(mol)] * int(count))
            carb_pat = Chem.MolFromSmarts("[CX3](=O)[OX1,OX2H1]")
            n_carb = len(mol.GetSubstructMatches(carb_pat)) if carb_pat else 0
            carbs.extend([n_carb] * int(count))
            n_n = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
            nitrogens.extend([n_n] * int(count))

        import numpy as np
        print(f"\n  {label} (weighted by count):")
        print(f"    Heavy atoms: {np.mean(heavys):.1f} +/- {np.std(heavys):.1f}")
        print(f"    Aromatic rings: {np.mean(aroms):.1f} +/- {np.std(aroms):.1f}")
        print(f"    MW: {np.mean(mws):.1f} +/- {np.std(mws):.1f}")
        print(f"    Carboxylates: {np.mean(carbs):.1f} +/- {np.std(carbs):.1f}")
        print(f"    N atoms: {np.mean(nitrogens):.1f} +/- {np.std(nitrogens):.1f}")


if __name__ == "__main__":
    main()
