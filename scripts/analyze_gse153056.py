import argparse
import csv
import gzip
import json
from math import comb
from pathlib import Path
from statistics import mean

import numpy as np


MODULES = {
    "MHC-I": ("B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5"),
    "interferon": ("IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7"),
}
GENES = tuple(gene for module in MODULES.values() for gene in module) + ("IFNGR1", "CD274")
FEATURE_NAMES = {"TAP2.1": "TAP2"}
REPLICATES = ("rep1", "rep2", "rep3")
GUIDES = tuple(f"IFNGR1g{number}" for number in range(1, 5))


def load_annotations(path):
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        barcode_field = reader.fieldnames[0]
        selected = {}
        for row in reader:
            if row["con"] != "tx" or row["gene"] not in {"NT", "IFNGR1"}:
                continue
            guide = row["guide_ID"] if row["gene"] == "IFNGR1" else "NT"
            barcode = row[barcode_field]
            if barcode in selected or row["replicate"] not in REPLICATES or guide not in (*GUIDES, "NT"):
                raise ValueError(f"Unexpected annotation for {barcode}")
            selected[barcode] = (row["replicate"], guide, int(row["nCount_RNA"]))
    return selected


def load_counts(path, annotations):
    with gzip.open(path, "rb") as handle:
        barcodes = [field.strip(b'"').decode() for field in handle.readline().rstrip().split(b"\t")[1:]]
        if len(set(barcodes)) != len(barcodes) or not annotations.keys() <= set(barcodes):
            raise ValueError("RNA count columns and selected annotations do not map one to one")
        indices = np.array([i for i, barcode in enumerate(barcodes) if barcode in annotations])
        selected = [annotations[barcodes[i]] for i in indices]
        total = np.zeros(len(indices), dtype=np.int64)
        counts = {}
        for line in handle:
            name, values = line.split(b"\t", 1)
            feature = name.strip(b'"').decode()
            gene = FEATURE_NAMES.get(feature, feature)
            row = np.fromstring(values.decode(), sep="\t", dtype=np.int64)
            if row.size != len(barcodes) or np.any(row < 0):
                raise ValueError(f"Invalid count row: {gene}")
            matched = row[indices]
            total += matched
            if gene in GENES:
                if gene in counts:
                    raise ValueError(f"Duplicated panel gene: {gene}")
                counts[gene] = matched
    if set(counts) != set(GENES):
        raise ValueError(f"Missing genes: {set(GENES) - counts.keys()}")
    if np.any(total <= 0):
        raise ValueError("At least one selected cell has no RNA counts")
    difference = total - np.array([entry[2] for entry in selected])
    audit = {
        "rna_columns": len(barcodes),
        "selected_barcodes": len(selected),
        "cells_with_different_umi_total_from_metadata": int(np.count_nonzero(difference)),
        "maximum_absolute_umi_difference": int(np.abs(difference).max()),
    }
    return selected, total, counts, audit


def summarize_group(replicate, guide, selected, total, counts):
    group = np.array([
        rep == replicate and (assigned == guide or (guide == "IFNGR1" and assigned in GUIDES))
        for rep, assigned, _ in selected
    ])
    cell_count = int(group.sum())
    if not cell_count:
        raise ValueError(f"Missing {replicate} {guide}")
    total_umi = int(total[group].sum())
    gene_counts = {gene: int(counts[gene][group].sum()) for gene in GENES}
    expression = {gene: float(np.log2(1 + count * 1e6 / total_umi)) for gene, count in gene_counts.items()}
    scores = {name: mean(expression[gene] for gene in genes) for name, genes in MODULES.items()}
    return {
        "replicate": replicate,
        "group": guide,
        "n_cells": cell_count,
        "total_umi": total_umi,
        "gene_counts": gene_counts,
        "log2_cpm_plus_1": expression,
        "scores": scores,
    }


def contrasts_for_replicate(groups):
    control, *targeted = groups
    return [
        {
            "replicate": group["replicate"],
            "group": group["group"],
            "n_targeted": group["n_cells"],
            "n_non_targeting": control["n_cells"],
            "module_delta": {name: group["scores"][name] - control["scores"][name] for name in MODULES},
            "gene_delta": {
                gene: group["log2_cpm_plus_1"][gene] - control["log2_cpm_plus_1"][gene]
                for gene in GENES
            },
        }
        for group in targeted
    ]


def two_sided_sign_p(differences):
    positive = sum(delta > 0 for delta in differences)
    negative = sum(delta < 0 for delta in differences)
    n = positive + negative
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(positive, negative) + 1)) / 2**n)


def main():
    parser = argparse.ArgumentParser(description="Analyze the fixed CSF modules in pooled GSE153056 THP-1 data")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("rna_counts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    annotations = load_annotations(args.metadata)
    selected, total, counts, audit = load_counts(args.rna_counts, annotations)
    groups = []
    contrasts = []
    for replicate in REPLICATES:
        replicate_groups = [
            summarize_group(replicate, guide, selected, total, counts)
            for guide in ("NT", "IFNGR1", *GUIDES)
        ]
        groups.extend(replicate_groups)
        contrasts.extend(contrasts_for_replicate(replicate_groups))

    primary = [row["module_delta"]["MHC-I"] for row in contrasts if row["group"] == "IFNGR1"]
    without_tap2 = {
        row["replicate"]: mean(row["gene_delta"][gene] for gene in MODULES["MHC-I"] if gene != "TAP2")
        for row in contrasts
        if row["group"] == "IFNGR1"
    }
    equal_guide = {
        replicate: mean(
            row["module_delta"]["MHC-I"]
            for row in contrasts
            if row["replicate"] == replicate and row["group"] in GUIDES
        )
        for replicate in REPLICATES
    }
    output = {
        "audit": audit,
        "modules": MODULES,
        "feature_names": FEATURE_NAMES,
        "groups": groups,
        "contrasts": contrasts,
        "primary": {
            "paired_differences": primary,
            "mean_difference": mean(primary),
            "range": [min(primary), max(primary)],
            "two_sided_sign_p": two_sided_sign_p(primary),
            "equal_guide_paired_differences": equal_guide,
            "paired_differences_without_TAP2": without_tap2,
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"audit": audit, "primary": output["primary"]}, indent=2))


if __name__ == "__main__":
    main()
