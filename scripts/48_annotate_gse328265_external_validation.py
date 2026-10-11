#!/usr/bin/env python3
"""Blindly annotate cleaned GSE328265 cells and define a strict BAM core."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("CELLTYPIST_FOLDER", str(ROOT / ".celltypist_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import pandas as pd
import scanpy as sc


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def read_clean_sample(sample_dir: Path, row) -> sc.AnnData:
    adata = sc.read_mtx(sample_dir / "matrix.mtx.gz").T
    barcodes = pd.read_csv(sample_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    features = pd.read_csv(sample_dir / "features.tsv.gz", sep="\t", header=None)
    adata.obs_names = [f"{row.sample_accession}:{barcode}" for barcode in barcodes]
    adata.var_names = features.iloc[:, 1].astype(str).to_numpy()
    adata.var["gene_id"] = features.iloc[:, 0].astype(str).to_numpy()
    adata.var_names_make_unique()
    for key in [
        "gse", "sample_accession", "sample_title", "subject_id", "technical_run",
        "tissue", "diagnosis", "disease_group", "source_subseries",
    ]:
        adata.obs[key] = str(getattr(row, key))
    return adata


def cluster_embedding(adata: sc.AnnData, n_top_genes: int, n_pcs: int, key: str, helpers) -> sc.AnnData:
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_top_genes, adata.n_vars),
        flavor="seurat",
        batch_key="sample_accession",
    )
    analysis = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(analysis, zero_center=False, max_value=10)
    sc.tl.pca(
        analysis,
        n_comps=min(50, analysis.n_vars - 1),
        zero_center=False,
        svd_solver="arpack",
    )
    helpers.harmony_integrate(analysis, "sample_accession")
    sc.pp.neighbors(
        analysis,
        n_neighbors=20 if key.startswith("leiden_clean") else 15,
        n_pcs=min(n_pcs, analysis.obsm["X_pca_harmony"].shape[1]),
        use_rep="X_pca_harmony",
    )
    sc.tl.leiden(
        analysis,
        resolution=0.8,
        key_added=key,
        random_state=12345,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    return analysis


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    helpers = load_module(root / "scripts" / "06_recluster_refine_annotation_gse138266.py", "helpers")
    robustness = load_module(root / "scripts" / "34_bam_annotation_robustness.py", "robustness")
    metadata = pd.read_csv(root / "metadata" / "GSE328265_sample_metadata.csv")
    cleanup = pd.read_csv(
        root / "results" / "qc" / "GSE328265_cleanup_cell_metrics.csv.gz",
        index_col="cell_id",
    )
    processed = root / "data" / "processed" / "GSE328265_decontx"
    output_dir = root / "results" / "external_validation" / "GSE328265"
    output_dir.mkdir(parents=True, exist_ok=True)

    adatas = [read_clean_sample(processed / row.sample_accession, row) for row in metadata.itertuples(index=False)]
    full = sc.concat(adatas, join="inner", merge="same", index_unique=None)
    full.var_names_make_unique()
    full.obs = full.obs.join(
        cleanup[["scrublet_score", "scDblFinder_score", "decontX_contamination"]],
        how="left",
    )
    full.layers["counts"] = full.X.copy()
    sc.pp.calculate_qc_metrics(full, percent_top=None, log1p=False, inplace=True)
    post_keep = full.obs["total_counts"].gt(0) & full.obs["n_genes_by_counts"].ge(200)
    post_summary = (
        full.obs.assign(post_decont_qc_pass=post_keep)
        .groupby(["sample_accession", "subject_id", "diagnosis"], observed=True)
        .agg(n_postdoublet_input=("post_decont_qc_pass", "size"), n_postdecont_qc_pass=("post_decont_qc_pass", "sum"))
        .reset_index()
    )
    post_summary["n_postdecont_removed"] = post_summary["n_postdoublet_input"] - post_summary["n_postdecont_qc_pass"]
    post_summary.to_csv(output_dir / "GSE328265_postdecont_qc_summary.csv", index=False)
    full = full[post_keep].copy()

    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)
    helpers.score_sets(full, helpers.BROAD_MARKERS, "score_broad_")
    full.raw = full
    analysis = cluster_embedding(full, 3000, 30, "leiden_clean_0_8", helpers)
    full.obs["leiden_clean_0_8"] = analysis.obs["leiden_clean_0_8"].astype(str)
    helpers.run_celltypist(full, "leiden_clean_0_8")
    full.obs["broad_cell_type"] = full.obs["celltypist_label"].map(helpers.broad_from_celltypist)
    broad_score_columns = [column for column in full.obs if column.startswith("score_broad_")]
    fallback = full.obs["celltypist_conf"].lt(0.5) | full.obs["broad_cell_type"].isin(["Unknown", "Other"])
    best = full.obs[broad_score_columns].idxmax(axis=1).str.replace("score_broad_", "", regex=False)
    full.obs.loc[fallback, "broad_cell_type"] = best[fallback].replace({
        "Myeloid_APC": "Myeloid", "Dendritic": "Dendritic", "pDC": "Dendritic"
    })

    apc = full[full.obs["broad_cell_type"].isin(["Myeloid", "Dendritic"])].copy()
    apc_score_columns = helpers.score_sets(apc, helpers.APC_MARKERS, "score_apc_")
    apc.raw = apc
    apc_analysis = cluster_embedding(apc, 2500, 25, "leiden_apc_0_8", helpers)
    apc.obs["leiden_apc_0_8"] = apc_analysis.obs["leiden_apc_0_8"].astype(str)

    cluster_means = apc.obs.groupby("leiden_apc_0_8", observed=True)[apc_score_columns].mean()
    order = np.argsort(-cluster_means.to_numpy(), axis=1)
    top = cluster_means.columns.to_numpy()[order[:, 0]]
    margin = cluster_means.to_numpy()[np.arange(len(cluster_means)), order[:, 0]] - cluster_means.to_numpy()[
        np.arange(len(cluster_means)), order[:, 1]
    ]
    labels = pd.Series(top, index=cluster_means.index).str.replace("score_apc_", "", regex=False)
    cluster_sizes = apc.obs.groupby("leiden_apc_0_8", observed=True).size().reindex(cluster_means.index)
    labels[(margin < 0.10) | (cluster_sizes < 20)] = "APC_ambiguous"
    pdc_fraction = (
        apc.obs.assign(is_pdc=apc.obs["celltypist_label"].astype(str).eq("pDC"))
        .groupby("leiden_apc_0_8", observed=True)["is_pdc"]
        .mean()
        .reindex(cluster_means.index, fill_value=0)
    )
    labels[(labels == "pDC") & pdc_fraction.lt(0.25)] = "APC_ambiguous"
    apc.obs["fine_apc_label"] = apc.obs["leiden_apc_0_8"].map(labels).astype(str)

    scores, _, _ = robustness.deterministic_marker_scores(apc_analysis)
    apc_analysis.obs["fine_apc_label"] = apc.obs["fine_apc_label"]
    _, cluster_audit, membership = robustness.multiresolution_audit(apc_analysis, scores, "GSE328265")
    reference = apc.obs["fine_apc_label"].eq("BAM_like_candidate")
    competitor_columns = [column for column in scores if column != "BAM_like_candidate"]
    all_margin = scores["BAM_like_candidate"] - scores[competitor_columns].max(axis=1)
    stable = reference & membership["bam_resolution_count"].ge(4) & all_margin.gt(0)
    apc.obs["bam_resolution_count"] = membership["bam_resolution_count"].astype(int)
    apc.obs["bam_marker_margin_all"] = all_margin
    apc.obs["bam_stable_core"] = stable

    full.obs["fine_apc_label"] = "not_APC"
    full.obs.loc[apc.obs_names, "fine_apc_label"] = apc.obs["fine_apc_label"]
    full.obs["bam_stable_core"] = False
    full.obs.loc[apc.obs_names, "bam_stable_core"] = stable
    full.obs["bam_resolution_count"] = np.nan
    full.obs.loc[apc.obs_names, "bam_resolution_count"] = apc.obs["bam_resolution_count"]
    full.obs["bam_marker_margin_all"] = np.nan
    full.obs.loc[apc.obs_names, "bam_marker_margin_all"] = all_margin

    annotation_columns = [
        "sample_accession", "subject_id", "technical_run", "diagnosis", "disease_group",
        "broad_cell_type", "fine_apc_label", "bam_stable_core", "bam_resolution_count",
        "bam_marker_margin_all", "celltypist_label", "celltypist_conf",
        "scrublet_score", "scDblFinder_score", "decontX_contamination",
    ]
    full.obs[annotation_columns].to_csv(
        output_dir / "GSE328265_clean_cell_annotations.csv.gz", compression="gzip"
    )
    cluster_audit.to_csv(output_dir / "GSE328265_BAM_multiresolution_cluster_scores.csv.gz", index=False, compression="gzip")
    summary = pd.DataFrame([{
        "n_postdecont_cells": full.n_obs,
        "n_apc": apc.n_obs,
        "n_reference_BAM": int(reference.sum()),
        "n_stable_core_BAM": int(stable.sum()),
        "stable_core_recall": float(stable.sum() / reference.sum()) if reference.sum() else np.nan,
        "n_subjects": full.obs["subject_id"].nunique(),
    }])
    summary.to_csv(output_dir / "GSE328265_annotation_summary.csv", index=False)

    # Keep full-gene normalized raw data and decontaminated counts for downstream donor aggregation.
    full.write(output_dir / "GSE328265_clean_annotated.h5ad", compression="gzip")
    print(summary.to_string(index=False))
    print("\nFine APC labels")
    print(apc.obs["fine_apc_label"].value_counts().to_string())
    print("\nStable BAM by diagnosis")
    print(apc.obs.loc[stable, "diagnosis"].value_counts().to_string())


if __name__ == "__main__":
    main()
