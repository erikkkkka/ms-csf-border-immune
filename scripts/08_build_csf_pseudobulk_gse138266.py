#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import pandas as pd
import scanpy as sc


TARGET_BROAD = ["T", "B", "Plasma"]
TARGET_APC = ["BAM_like_candidate", "pDC", "cDC2", "cDC1", "classical_monocyte", "APC_ambiguous"]


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def read_sample(sample_dir, accession):
    adata = sc.read_mtx(sample_dir / "matrix.mtx.gz").T
    barcodes = pd.read_csv(sample_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    features = pd.read_csv(sample_dir / "features.tsv.gz", sep="\t", header=None)
    adata.obs_names = [f"{accession}:{x}" for x in barcodes]
    adata.var_names = features.iloc[:, 0].astype(str).to_numpy()
    return adata, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--gse", default="GSE138266")
    parser.add_argument("--tissue", default="CSF")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    gse = args.gse
    tissue = args.tissue
    processed = root / f"data/processed/{gse}_decontx"
    ann_dir = root / "results/annotation_cleaned"
    out_dir = root / "results/pseudobulk"
    out_dir.mkdir(parents=True, exist_ok=True)

    dedicated_meta = root / f"metadata/{gse}_sample_metadata.csv"
    if dedicated_meta.exists():
        meta = pd.read_csv(dedicated_meta)
    else:
        meta = pd.read_csv(root / "metadata/all_geo_sample_metadata.csv")
        meta = meta.loc[meta["gse"] == gse].copy()
    if gse == "GSE286068":
        meta = meta.loc[(meta["disease_group"] != "unknown") & ~meta["sample_title"].str.startswith("multiplexed")]
    meta = meta.loc[meta["tissue"] == tissue].sort_values("sample_accession").copy()
    global_cells = pd.read_csv(ann_dir / f"{gse}_clean_cell_annotations.csv.gz", index_col=0)
    apc_cells = pd.read_csv(ann_dir / f"{gse}_clean_apc_cell_annotations.csv.gz", index_col=0)
    apc_cells = apc_cells.join(global_cells[["broad_cell_type"]], how="left")
    apc_cells = apc_cells.loc[apc_cells["broad_cell_type"].isin(["Myeloid", "Dendritic"])]
    labels = pd.concat([
        global_cells.loc[global_cells["broad_cell_type"].isin(TARGET_BROAD), ["sample_accession", "broad_cell_type"]]
        .rename(columns={"broad_cell_type": "analysis_label"}),
        apc_cells.loc[apc_cells["fine_apc_label"].isin(TARGET_APC), ["sample_accession", "fine_apc_label"]]
        .rename(columns={"fine_apc_label": "analysis_label"}),
    ])
    if labels.index.has_duplicates:
        duplicated = labels.index[labels.index.duplicated()].unique()
        raise ValueError(f"Cells assigned to more than one target compartment: {len(duplicated)}")

    aggregates = {label: {} for label in TARGET_BROAD + TARGET_APC}
    cell_rows = []
    reference_features = None
    for row in meta.itertuples(index=False):
        adata, features = read_sample(processed / row.sample_accession, row.sample_accession)
        if reference_features is None:
            reference_features = features.copy()
        elif not np.array_equal(reference_features.iloc[:, 0].astype(str), features.iloc[:, 0].astype(str)):
            raise ValueError(f"Feature order differs in {row.sample_accession}")
        sample_labels = labels.loc[labels["sample_accession"] == row.sample_accession, "analysis_label"]
        sample_labels = sample_labels.reindex(adata.obs_names)
        for label in aggregates:
            mask = sample_labels.eq(label).fillna(False).to_numpy()
            aggregates[label][row.sample_accession] = np.asarray(adata.X[mask].sum(axis=0)).ravel().astype(np.int64)
            cell_rows.append({
                "analysis_label": label,
                "sample_accession": row.sample_accession,
                "subject_id": row.subject_id,
                "diagnosis": row.diagnosis,
                "disease_group": row.disease_group,
                "n_cells": int(mask.sum()),
                "library_size": int(aggregates[label][row.sample_accession].sum()),
            })

    feature_table = pd.DataFrame({
        "gene_id": reference_features.iloc[:, 0].astype(str),
        "gene_symbol": reference_features.iloc[:, 1].astype(str),
    })
    for label, sample_counts in aggregates.items():
        table = feature_table.copy()
        for accession in meta["sample_accession"]:
            table[accession] = sample_counts[accession]
        table.to_csv(out_dir / f"{gse}_{safe_name(tissue)}_{safe_name(label)}_counts.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cell_rows).to_csv(out_dir / f"{gse}_{safe_name(tissue)}_pseudobulk_sample_metadata.csv", index=False)


if __name__ == "__main__":
    main()
