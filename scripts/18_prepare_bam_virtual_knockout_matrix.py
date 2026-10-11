#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


FORCED_GENES = {
    "TAP1", "TAP2", "TAPBP", "PSME1", "PSME2", "PSMB8", "PSMB9", "PSMB10",
    "B2M", "HLA-A", "HLA-B", "HLA-C", "NLRC5", "STAT1", "STAT2", "IRF1",
    "IFNGR1", "IFNGR2", "IFNAR1", "IFNAR2", "JAK1", "JAK2", "TYK2",
    "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1", "LYVE1",
    "FOLR2", "DAB2", "MSR1", "CD163", "HAMP",
    "LTA4H",
}


def read_selected_sample(processed, accession, selected_ids):
    sample_dir = processed / accession
    matrix = sc.read_mtx(sample_dir / "matrix.mtx.gz").X.tocsc()
    barcodes = pd.read_csv(sample_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    features = pd.read_csv(sample_dir / "features.tsv.gz", sep="\t", header=None)
    cell_ids = pd.Index([f"{accession}:{barcode}" for barcode in barcodes])
    indices = cell_ids.get_indexer(selected_ids)
    if np.any(indices < 0):
        raise ValueError(f"Missing BAM cells in {accession}")
    return matrix[:, indices], features, cell_ids[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--gse", default="GSE138266")
    parser.add_argument("--n-hvg", type=int, default=1200)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    processed = root / f"data/processed/{args.gse}_decontx"
    annotations = pd.read_csv(
        root / f"results/annotation_cleaned/{args.gse}_clean_apc_cell_annotations.csv.gz",
        index_col=0,
    )
    selected = annotations.loc[
        (annotations["tissue"] == "CSF")
        & (annotations["disease_group"] == "MS")
        & (annotations["fine_apc_label"] == "BAM_like_candidate")
    ]

    matrices = []
    cell_ids = []
    reference = None
    for accession, frame in selected.groupby("sample_accession", observed=True, sort=True):
        matrix, features, ids = read_selected_sample(processed, accession, frame.index)
        if reference is None:
            reference = features
        elif not np.array_equal(reference.iloc[:, 0], features.iloc[:, 0]):
            raise ValueError(f"Feature order mismatch in {accession}")
        matrices.append(matrix)
        cell_ids.extend(ids)
    counts = sparse.hstack(matrices, format="csr").T
    gene_symbols = pd.Index(reference.iloc[:, 1].astype(str).to_numpy())
    keep = np.asarray((counts > 0).sum(axis=0)).ravel() >= 10
    keep &= ~gene_symbols.str.upper().str.startswith("MT-")
    keep &= ~gene_symbols.str.upper().str.match(r"^RP[SL]")
    filtered = ad.AnnData(counts[:, keep].copy())
    filtered.obs_names = pd.Index(cell_ids)
    filtered.var_names = gene_symbols[keep]
    filtered.var_names_make_unique()
    filtered.layers["counts"] = filtered.X.copy()
    sc.pp.normalize_total(filtered, target_sum=1e4)
    sc.pp.log1p(filtered)
    sc.pp.highly_variable_genes(filtered, n_top_genes=args.n_hvg, flavor="seurat")
    selected_genes = set(filtered.var_names[filtered.var["highly_variable"]])
    selected_genes.update(gene for gene in FORCED_GENES if gene in filtered.var_names)
    selected_mask = filtered.var_names.isin(selected_genes)
    output_counts = filtered.layers["counts"][:, selected_mask]
    output = pd.DataFrame.sparse.from_spmatrix(
        output_counts.T,
        index=filtered.var_names[selected_mask],
        columns=filtered.obs_names,
    )
    out_dir = root / "results/virtual_knockout"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.sparse.to_dense().to_csv(
        out_dir / f"{args.gse}_MS_CSF_BAM_scTenifold_input.csv.gz",
        compression="gzip",
    )
    pd.DataFrame({
        "gene": filtered.var_names[selected_mask],
        "forced_candidate": filtered.var_names[selected_mask].isin(FORCED_GENES),
        "n_detected_cells": np.asarray((output_counts > 0).sum(axis=0)).ravel(),
        "total_counts": np.asarray(output_counts.sum(axis=0)).ravel(),
    }).to_csv(out_dir / f"{args.gse}_MS_CSF_BAM_scTenifold_features.csv", index=False)
    selected.reset_index().groupby("sample_accession", observed=True).size().rename("n_bam_cells").to_csv(
        out_dir / f"{args.gse}_MS_CSF_BAM_scTenifold_sample_counts.csv"
    )


if __name__ == "__main__":
    main()
