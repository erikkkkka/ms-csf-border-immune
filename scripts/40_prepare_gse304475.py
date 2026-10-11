#!/usr/bin/env python3
"""Extract raw counts and sample metadata from the GSE304475 workbook."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(args.workbook, sheet_name="Sheet1")
    count_columns = [column for column in frame if column.startswith("count.")]
    expected = [
        "count.L_Con1", "count.L_Con2", "count.L_Con3", "count.L_Con4",
        "count.L_SC1", "count.L_SC2", "count.L_SC3", "count.L_SC4",
    ]
    if count_columns != expected:
        raise RuntimeError(f"Unexpected count columns: {count_columns}")
    counts = frame[["gene_id"] + count_columns].copy()
    counts = counts.loc[counts["gene_id"].notna()].copy()
    for column in count_columns:
        counts[column] = pd.to_numeric(counts[column], errors="raise")
    values = counts[count_columns].to_numpy(dtype=float)
    if np.any(values < 0) or not np.allclose(values, np.rint(values)):
        raise RuntimeError("Count matrix is not non-negative integer-valued")
    counts[count_columns] = np.rint(values).astype(np.int64)
    counts = counts.groupby("gene_id", as_index=False)[count_columns].sum()
    rename = {column: column.removeprefix("count.") for column in count_columns}
    counts = counts.rename(columns=rename)

    metadata = pd.DataFrame({
        "sample_id": list(rename.values()),
        "group": ["Vehicle"] * 4 + ["SC57461A"] * 4,
        "replicate": [1, 2, 3, 4, 1, 2, 3, 4],
        "cell_type": "sorted_large_intestinal_ILC2",
        "species": "Mus_musculus",
        "treatment_h": 16,
    })
    counts.to_csv(args.output_dir / "GSE304475_raw_counts.csv.gz", index=False)
    metadata.to_csv(args.output_dir / "GSE304475_sample_metadata.csv", index=False)
    manifest = pd.DataFrame([{
        "accession": "GSE304475",
        "source_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE304nnn/GSE304475/"
            "suppl/GSE304475_1_genes_fpkm_expression.xlsx"
        ),
        "source_file": args.workbook.name,
        "source_sha256": sha256(args.workbook),
        "n_genes": len(counts),
        "n_samples": len(metadata),
        "count_sum": int(counts[list(rename.values())].to_numpy().sum()),
        "analysis_contrast": "SC57461A_vs_Vehicle",
    }])
    manifest.to_csv(args.output_dir / "GSE304475_source_manifest.csv", index=False)
    print(manifest.to_string(index=False))
    print("\nLibrary sizes")
    print(counts[list(rename.values())].sum().to_string())


if __name__ == "__main__":
    main()
