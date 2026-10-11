#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


WARHEADS = {
    "boronic_acid_or_boronate": [
        Chem.MolFromSmarts("[B]([O])([O])"),
        Chem.MolFromSmarts("[B]1[O][C][C][O]1"),
    ],
    "epoxyketone": [
        Chem.MolFromSmarts("C(=O)[C;R]1[O;R][C;R]1"),
        Chem.MolFromSmarts("C(=O)[C;R]1[C;R][O;R]1"),
    ],
}


def classify_warhead(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return "unresolved"
    labels = [
        label
        for label, patterns in WARHEADS.items()
        if any(pattern is not None and mol.HasSubstructMatch(pattern) for pattern in patterns)
    ]
    return ";".join(labels) if labels else "no_flagged_warhead"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source = root / "data" / "reference" / "chembl_psmb9_activities.json"
    out_dir = root / "results" / "drug_repurposing"
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = pd.DataFrame(json.loads(source.read_text())["activities"])
    quantitative = activities.loc[
        activities["standard_type"].eq("IC50")
        & activities["standard_units"].eq("nM")
        & activities["standard_value"].notna()
    ].copy()
    quantitative["standard_value_nM"] = pd.to_numeric(
        quantitative["standard_value"], errors="coerce"
    )
    quantitative = quantitative.dropna(subset=["standard_value_nM"])
    quantitative["warhead_class"] = quantitative["canonical_smiles"].map(classify_warhead)
    quantitative["molecule_label"] = quantitative["molecule_pref_name"].fillna(
        quantitative["molecule_chembl_id"]
    )

    keep = [
        "molecule_label",
        "molecule_pref_name",
        "molecule_chembl_id",
        "parent_molecule_chembl_id",
        "canonical_smiles",
        "warhead_class",
        "standard_relation",
        "standard_value_nM",
        "pchembl_value",
        "assay_chembl_id",
        "assay_description",
        "document_chembl_id",
        "document_journal",
        "document_year",
        "target_chembl_id",
        "target_pref_name",
        "target_organism",
    ]
    quantitative[keep].sort_values("standard_value_nM").to_csv(
        out_dir / "ChEMBL_PSMB9_quantitative_IC50_records.csv.gz",
        index=False,
        compression="gzip",
    )

    best = (
        quantitative.sort_values("standard_value_nM")
        .drop_duplicates("molecule_chembl_id")
        .loc[:, keep]
        .reset_index(drop=True)
    )
    best.insert(0, "potency_rank", range(1, len(best) + 1))
    best.to_csv(out_dir / "ChEMBL_PSMB9_best_IC50_per_molecule.csv", index=False)

    named = best.loc[best["molecule_pref_name"].notna()].copy()
    named.to_csv(out_dir / "ChEMBL_PSMB9_named_compounds.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "target_chembl_id": "CHEMBL1944495",
                "n_all_activity_records": len(activities),
                "n_unique_molecules_all_records": activities["molecule_chembl_id"].nunique(),
                "n_quantitative_ic50_nM_records": len(quantitative),
                "n_unique_quantitative_molecules": quantitative["molecule_chembl_id"].nunique(),
                "n_named_quantitative_molecules": named["molecule_chembl_id"].nunique(),
                "n_boronic_or_boronate_molecules": int(
                    best["warhead_class"].str.contains("boronic", na=False).sum()
                ),
                "n_epoxyketone_molecules": int(
                    best["warhead_class"].str.contains("epoxyketone", na=False).sum()
                ),
                "interpretation": (
                    "Standard noncovalent Vina docking is not an adequate validation method "
                    "for boronic-acid/boronate or epoxyketone active-site inhibitors."
                ),
            }
        ]
    )
    summary.to_csv(out_dir / "ChEMBL_PSMB9_summary.csv", index=False)


if __name__ == "__main__":
    main()
