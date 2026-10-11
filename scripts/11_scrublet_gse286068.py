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
import scrublet as scr


def find_file(raw_dir, accession, suffix):
    hits = sorted(raw_dir.glob(f"{accession}_*_{suffix}"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one {suffix} for {accession}; found {len(hits)}")
    return hits[0]


def read_counts(raw_dir, accession):
    matrix_file = find_file(raw_dir, accession, "matrix.mtx.gz")
    barcode_file = find_file(raw_dir, accession, "barcodes.tsv.gz")
    feature_file = find_file(raw_dir, accession, "features.tsv.gz")
    adata = sc.read_mtx(matrix_file).T
    barcodes = pd.read_csv(barcode_file, sep="\t", header=None)[0].astype(str)
    features = pd.read_csv(feature_file, sep="\t", header=None)
    gex = features.iloc[:, 2].astype(str).eq("Gene Expression").to_numpy() if features.shape[1] >= 3 else np.ones(len(features), dtype=bool)
    adata = adata[:, gex].copy()
    adata.obs_names = barcodes
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw_dir = root / "data/raw/GSE286068"
    qc_dir = root / "results/qc"
    metrics = pd.read_csv(qc_dir / "GSE286068_cell_calling_metrics.csv.gz")
    outputs = []
    summaries = []

    for accession, frame in metrics.groupby("sample_accession", sort=True):
        print(f"Scrublet: {accession} ({len(frame)} cells)", flush=True)
        adata = read_counts(raw_dir, accession)
        indexer = adata.obs_names.get_indexer(frame["barcode"].astype(str))
        if np.any(indexer < 0):
            raise ValueError(f"Missing called barcodes in {accession}")
        counts = adata.X[indexer].tocsr().astype(np.float64)
        n_cells = counts.shape[0]
        expected_rate = min(0.10, max(0.01, 0.008 * n_cells / 1000.0))
        detector = scr.Scrublet(counts, expected_doublet_rate=expected_rate, sim_doublet_ratio=2.0, random_state=12345)
        scores, _ = detector.scrub_doublets(
            min_counts=2,
            min_cells=3,
            min_gene_variability_pctl=85,
            n_prin_comps=min(30, max(5, n_cells - 2)),
            use_approx_neighbors=False,
            verbose=False,
        )
        if not np.all(np.isfinite(scores)) or np.unique(scores).size < min(10, max(5, n_cells // 20)):
            raise RuntimeError(f"Degenerate Scrublet scores for {accession}")
        n_calls = max(1, int(np.ceil(expected_rate * n_cells)))
        called_index = np.argsort(scores, kind="stable")[-n_calls:]
        predicted = np.zeros(n_cells, dtype=bool)
        predicted[called_index] = True
        threshold = float(scores[called_index].min())
        out = frame.copy()
        out["scrublet_score"] = scores
        out["scrublet_predicted_doublet"] = predicted.astype(bool)
        out["scrublet_threshold"] = threshold
        out["scrublet_expected_rate"] = expected_rate
        outputs.append(out)
        summaries.append({
            "sample_accession": accession,
            "n_pre_doublet_qc": n_cells,
            "scrublet_expected_rate": expected_rate,
            "scrublet_threshold": threshold,
            "n_scrublet_doublets": int(predicted.sum()),
            "pct_scrublet_doublets": float(predicted.mean() * 100),
            "scrublet_status": "exact_knn_top_expected_count",
        })

    pd.concat(outputs, ignore_index=True).to_csv(qc_dir / "GSE286068_scrublet_metrics.csv.gz", index=False, compression="gzip")
    pd.DataFrame(summaries).to_csv(qc_dir / "GSE286068_scrublet_summary.csv", index=False)


if __name__ == "__main__":
    main()
