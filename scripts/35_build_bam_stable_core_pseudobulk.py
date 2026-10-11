#!/usr/bin/env python3
"""Build raw-count CSF pseudobulk matrices for the strict BAM stable core."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def read_sample(sample_dir: Path, accession: str) -> tuple[sc.AnnData, pd.DataFrame]:
    adata = sc.read_mtx(sample_dir / "matrix.mtx.gz").T
    barcodes = pd.read_csv(sample_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    features = pd.read_csv(sample_dir / "features.tsv.gz", sep="\t", header=None)
    adata.obs_names = [f"{accession}:{barcode}" for barcode in barcodes]
    return adata, features


def load_metadata(root: Path, gse: str) -> pd.DataFrame:
    dedicated = root / "metadata" / f"{gse}_sample_metadata.csv"
    if dedicated.exists():
        metadata = pd.read_csv(dedicated)
    else:
        metadata = pd.read_csv(root / "metadata" / "all_geo_sample_metadata.csv")
        metadata = metadata[metadata["gse"].eq(gse)].copy()
    if gse == "GSE286068":
        metadata = metadata[
            metadata["disease_group"].ne("unknown")
            & ~metadata["sample_title"].str.startswith("multiplexed")
        ]
    return metadata[metadata["tissue"].eq("CSF")].sort_values("sample_accession").copy()


def build_cohort(root: Path, gse: str, output_dir: Path) -> None:
    processed = root / "data" / "processed" / f"{gse}_decontx"
    stability = pd.read_csv(
        root / "results" / "bam_robustness" / f"{gse}_BAM_cell_stability.csv.gz",
        index_col=0,
    )
    stable_ids = set(stability.index[stability["bam_stable_core"].astype(bool)])
    metadata = load_metadata(root, gse)
    sample_counts = {}
    sample_rows = []
    reference_features = None
    for row in metadata.itertuples(index=False):
        adata, features = read_sample(processed / row.sample_accession, row.sample_accession)
        if reference_features is None:
            reference_features = features.copy()
        elif not np.array_equal(
            reference_features.iloc[:, 0].astype(str), features.iloc[:, 0].astype(str)
        ):
            raise RuntimeError(f"Feature order differs in {row.sample_accession}")
        mask = np.fromiter(
            (cell_id in stable_ids for cell_id in adata.obs_names),
            dtype=bool,
            count=adata.n_obs,
        )
        counts = np.asarray(adata.X[mask].sum(axis=0)).ravel().astype(np.int64)
        sample_counts[row.sample_accession] = counts
        sample_rows.append(
            {
                "gse": gse,
                "analysis_label": "BAM_stable_core",
                "sample_accession": row.sample_accession,
                "subject_id": row.subject_id,
                "diagnosis": row.diagnosis,
                "disease_group": row.disease_group,
                "n_cells": int(mask.sum()),
                "library_size": int(counts.sum()),
            }
        )

    if reference_features is None:
        raise RuntimeError(f"No CSF samples found for {gse}")
    count_table = pd.DataFrame(
        {
            "gene_id": reference_features.iloc[:, 0].astype(str),
            "gene_symbol": reference_features.iloc[:, 1].astype(str),
        }
    )
    for accession in metadata["sample_accession"]:
        count_table[accession] = sample_counts[accession]
    count_table.to_csv(
        output_dir / f"{gse}_CSF_BAM_stable_core_counts.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(sample_rows).to_csv(
        output_dir / f"{gse}_CSF_BAM_stable_core_sample_metadata.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / "results" / "bam_robustness" / "pseudobulk"
    output_dir.mkdir(parents=True, exist_ok=True)
    for gse in ["GSE138266", "GSE194078", "GSE286068"]:
        build_cohort(root, gse, output_dir)
    summaries = []
    for path in sorted(output_dir.glob("*_sample_metadata.csv")):
        frame = pd.read_csv(path)
        summaries.append(
            frame.groupby(["gse", "disease_group"], observed=True)
            .agg(n_samples=("sample_accession", "size"), n_cells=("n_cells", "sum"), median_cells=("n_cells", "median"))
            .reset_index()
        )
    print(pd.concat(summaries, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
