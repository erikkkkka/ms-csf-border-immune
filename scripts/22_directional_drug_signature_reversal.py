#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, hypergeom
from statsmodels.stats.multitest import multipletests


LIBRARIES = {
    "NIBR_DRUGseq_2025": ("NIBR_DRUGseq_2025_up.gmt", "NIBR_DRUGseq_2025_down.gmt"),
    "Drug_Perturbations_from_GEO": (
        "Drug_Perturbations_from_GEO_up.gmt",
        "Drug_Perturbations_from_GEO_down.gmt",
    ),
    "Old_CMAP": ("Old_CMAP_up.gmt", "Old_CMAP_down.gmt"),
}


def read_gmt(path: Path) -> dict[str, set[str]]:
    terms = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or not fields[0].strip():
                continue
            terms[fields[0]] = {gene.upper() for gene in fields[2:] if gene}
    return terms


def canonical_drug(term: str, library: str) -> tuple[str, str, str]:
    if library == "NIBR_DRUGseq_2025":
        parts = term.split()
        cid = parts[0].upper()
        dose = parts[-1] if parts else ""
        core = parts[:-3] if len(parts) >= 4 and parts[-2].upper().startswith("BAID") else parts
        label_parts = core[1:] if core and core[0].upper().startswith("CID") else core
        label = " ".join(label_parts) or cid
        return cid, label, dose
    if library == "Drug_Perturbations_from_GEO":
        match = re.match(r"(.+?)\s+(DB\d+)\s+", term, flags=re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            return f"{label.lower()}|{match.group(2).upper()}", label, match.group(2).upper()
        label = term.split()[0]
        return label.lower(), label, ""
    base = re.sub(r"-\d+$", "", term).strip().lower()
    return base, base, ""


def enrich(query: set[str], gene_set: set[str], universe: set[str]) -> tuple[int, float, str]:
    query = query & universe
    gene_set = gene_set & universe
    overlap = sorted(query & gene_set)
    if not query or not gene_set:
        return 0, 1.0, ""
    p_value = hypergeom.sf(len(overlap) - 1, len(universe), len(gene_set), len(query))
    return len(overlap), float(p_value), ";".join(overlap)


def simes(values: pd.Series) -> float:
    p = np.sort(values.to_numpy(dtype=float))
    return float(min(1.0, np.min(p * len(p) / np.arange(1, len(p) + 1))))


def analyze_library(
    library: str,
    up_path: Path,
    down_path: Path,
    disease_up: set[str],
    disease_down: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drug_up = read_gmt(up_path)
    drug_down = read_gmt(down_path)
    paired_terms = sorted(set(drug_up) & set(drug_down))
    universe = set().union(*(drug_up[term] | drug_down[term] for term in paired_terms))
    rows = []
    for term in paired_terms:
        n_up_reversed, p_up_reversed, genes_up_reversed = enrich(
            disease_up, drug_down[term], universe
        )
        n_down_restored, p_down_restored, genes_down_restored = enrich(
            disease_down, drug_up[term], universe
        )
        combined_p = combine_pvalues([p_up_reversed, p_down_restored], method="fisher")[1]
        drug_key, drug_label, context = canonical_drug(term, library)
        rows.append({
            "library": library,
            "drug_key": drug_key,
            "drug_label": drug_label,
            "signature_term": term,
            "context": context,
            "universe_genes": len(universe),
            "disease_up_in_universe": len(disease_up & universe),
            "disease_down_in_universe": len(disease_down & universe),
            "n_disease_up_drug_down": n_up_reversed,
            "p_disease_up_drug_down": p_up_reversed,
            "genes_disease_up_drug_down": genes_up_reversed,
            "n_disease_down_drug_up": n_down_restored,
            "p_disease_down_drug_up": p_down_restored,
            "genes_disease_down_drug_up": genes_down_restored,
            "combined_p": combined_p,
        })
    signatures = pd.DataFrame(rows)
    signatures["combined_fdr"] = multipletests(signatures["combined_p"], method="fdr_bh")[1]
    signatures["bidirectional_min_overlap"] = signatures[[
        "n_disease_up_drug_down", "n_disease_down_drug_up"
    ]].min(axis=1)
    signatures["bidirectional_support"] = signatures["bidirectional_min_overlap"] >= 2
    signatures = signatures.sort_values(
        ["bidirectional_support", "combined_p", "bidirectional_min_overlap"],
        ascending=[False, True, False],
    )

    compound_rows = []
    for drug_key, group in signatures.groupby("drug_key", sort=False):
        supported = group.loc[group["bidirectional_support"]]
        test_group = supported if len(supported) else group
        best = test_group.sort_values("combined_p").iloc[0]
        compound_rows.append({
            "library": library,
            "drug_key": drug_key,
            "drug_label": best["drug_label"],
            "n_paired_signatures": len(group),
            "n_bidirectional_signatures": len(supported),
            "compound_simes_p": simes(test_group["combined_p"]),
            "best_signature_term": best["signature_term"],
            "best_signature_combined_p": best["combined_p"],
            "best_signature_combined_fdr": best["combined_fdr"],
            "best_up_reversal_overlap": best["n_disease_up_drug_down"],
            "best_down_restoration_overlap": best["n_disease_down_drug_up"],
            "best_up_reversal_genes": best["genes_disease_up_drug_down"],
            "best_down_restoration_genes": best["genes_disease_down_drug_up"],
        })
    compounds = pd.DataFrame(compound_rows)
    compounds["compound_fdr"] = multipletests(compounds["compound_simes_p"], method="fdr_bh")[1]
    compounds["bidirectional"] = compounds["n_bidirectional_signatures"] > 0
    compounds = compounds.sort_values(
        ["bidirectional", "compound_simes_p", "n_bidirectional_signatures"],
        ascending=[False, True, False],
    )
    return signatures, compounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    ref_dir = root / "data" / "reference" / "enrichr"
    out_dir = root / "results" / "drug_repurposing"
    out_dir.mkdir(parents=True, exist_ok=True)
    disease_up = {
        gene.strip().upper() for gene in (out_dir / "BAM_MS_consensus_up.txt").read_text().splitlines()
        if gene.strip()
    }
    disease_down = {
        gene.strip().upper() for gene in (out_dir / "BAM_MS_consensus_down.txt").read_text().splitlines()
        if gene.strip()
    }

    all_signatures = []
    all_compounds = []
    for library, (up_file, down_file) in LIBRARIES.items():
        signatures, compounds = analyze_library(
            library,
            ref_dir / up_file,
            ref_dir / down_file,
            disease_up,
            disease_down,
        )
        signatures.to_csv(out_dir / f"{library}_directional_reversal_signatures.csv.gz", index=False)
        compounds.to_csv(out_dir / f"{library}_directional_reversal_compounds.csv", index=False)
        all_signatures.append(signatures)
        all_compounds.append(compounds)

    pd.concat(all_signatures, ignore_index=True).to_csv(
        out_dir / "all_directional_reversal_signatures.csv.gz", index=False, compression="gzip"
    )
    pd.concat(all_compounds, ignore_index=True).to_csv(
        out_dir / "all_directional_reversal_compounds.csv", index=False
    )


if __name__ == "__main__":
    main()
