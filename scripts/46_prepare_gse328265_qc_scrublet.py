#!/usr/bin/env python3
"""Build metadata and run cell QC/Scrublet for GSE327243 + GSE328264."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=6000)
    parser.add_argument("--max-mt", type=float, default=20.0)
    return parser.parse_args()


def metadata_from_path(path: Path) -> dict:
    accession = path.name.split("_", 1)[0]
    stem = path.name.removeprefix(f"{accession}_").removesuffix("_filtered_feature_bc_matrix.h5")
    stem = stem.removeprefix("PMS_")
    match = re.match(r"(?P<group>Ctrl|RMS|RRMS|PPMS|SPMS)_(?P<number>\d+)(?:_(?P<run>\d+))?$", stem)
    if not match:
        raise ValueError(f"Cannot parse sample name: {path.name}")
    source_group = match.group("group")
    group = "RRMS" if source_group in {"RMS", "RRMS"} else source_group
    number = match.group("number")
    subject_id = f"{group}_{number}"
    disease_group = "control" if group == "Ctrl" else group
    return {
        "gse": "GSE328265",
        "sample_accession": accession,
        "sample_title": stem,
        "subject_id": subject_id,
        "technical_run": match.group("run") or "1",
        "tissue": "CSF",
        "diagnosis": group,
        "disease_group": disease_group,
        "source_subseries": path.parent.name,
        "h5_path": str(path.resolve()),
    }


def scrublet_calls(counts, seed: int = 12345) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    n_cells = counts.shape[0]
    expected_rate = min(0.10, max(0.01, 0.008 * n_cells / 1000.0))
    if n_cells < 50:
        return (
            np.full(n_cells, np.nan),
            np.zeros(n_cells, dtype=bool),
            np.nan,
            expected_rate,
            "not_run_lt_50_cells",
        )
    detector = scr.Scrublet(
        counts.astype(np.float64),
        expected_doublet_rate=expected_rate,
        sim_doublet_ratio=2.0,
        random_state=seed,
    )
    n_components = min(30, max(2, n_cells - 2))
    scores, _ = detector.scrub_doublets(
        min_counts=2,
        min_cells=3,
        min_gene_variability_pctl=85,
        n_prin_comps=n_components,
        use_approx_neighbors=False,
        verbose=False,
    )
    if not np.all(np.isfinite(scores)) or np.unique(scores).size < 10:
        return (
            scores,
            np.zeros(n_cells, dtype=bool),
            np.nan,
            expected_rate,
            "degenerate_scores_no_calls",
        )
    n_calls = max(1, int(np.ceil(expected_rate * n_cells)))
    called = np.argsort(scores, kind="stable")[-n_calls:]
    predicted = np.zeros(n_cells, dtype=bool)
    predicted[called] = True
    threshold = float(scores[called].min())
    return scores, predicted, threshold, expected_rate, "exact_knn_top_expected_count"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dirs = [
        root / "data" / "external_validation" / "GSE327243",
        root / "data" / "external_validation" / "GSE328264",
    ]
    paths = sorted(path for directory in input_dirs for path in directory.glob("*_filtered_feature_bc_matrix.h5"))
    if len(paths) != 31:
        raise RuntimeError(f"Expected 31 h5 files, found {len(paths)}")
    metadata = pd.DataFrame([metadata_from_path(path) for path in paths])
    metadata_dir = root / "metadata"
    qc_dir = root / "results" / "qc"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_dir / "GSE328265_sample_metadata.csv", index=False)

    cell_tables = []
    sample_rows = []
    for row in metadata.itertuples(index=False):
        print(f"QC + Scrublet: {row.sample_accession} {row.sample_title}", flush=True)
        adata = sc.read_10x_h5(row.h5_path, gex_only=True)
        adata.var_names_make_unique()
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        qc_pass = (
            adata.obs["n_genes_by_counts"].between(args.min_genes, args.max_genes)
            & adata.obs["pct_counts_mt"].le(args.max_mt)
            & adata.obs["total_counts"].gt(0)
        )
        frame = adata.obs.loc[
            qc_pass, ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
        ].copy()
        frame.insert(0, "barcode", frame.index.astype(str))
        frame.insert(0, "cell_id", [f"{row.sample_accession}:{barcode}" for barcode in frame["barcode"]])
        frame.insert(0, "sample_accession", row.sample_accession)
        counts = adata[qc_pass].X.tocsr()
        scores, predicted, threshold, expected_rate, status = scrublet_calls(counts)
        frame["scrublet_score"] = scores
        frame["scrublet_predicted_doublet"] = predicted
        frame["scrublet_threshold"] = threshold
        frame["scrublet_expected_rate"] = expected_rate
        cell_tables.append(frame)
        sample_rows.append({
            "sample_accession": row.sample_accession,
            "subject_id": row.subject_id,
            "diagnosis": row.diagnosis,
            "n_filtered_input": adata.n_obs,
            "n_qc_pass": int(qc_pass.sum()),
            "n_scrublet_doublets": int(predicted.sum()),
            "pct_scrublet_doublets": float(predicted.mean() * 100) if len(predicted) else np.nan,
            "scrublet_status": status,
        })

    pd.concat(cell_tables, ignore_index=True).to_csv(
        qc_dir / "GSE328265_scrublet_metrics.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(sample_rows).to_csv(qc_dir / "GSE328265_scrublet_summary.csv", index=False)
    print(pd.DataFrame(sample_rows).groupby("diagnosis")[["n_filtered_input", "n_qc_pass", "n_scrublet_doublets"]].sum())


if __name__ == "__main__":
    main()
