#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("CELLTYPIST_FOLDER", str(ROOT / ".celltypist_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


MARKER_SETS = {
    "T_cells": ["CD3D", "CD3E", "TRAC", "IL7R", "LTB"],
    "CD4_T": ["CD4", "IL7R", "CCR7", "TCF7", "LTB"],
    "CD8_T": ["CD8A", "CD8B", "GZMK", "CCL5", "NKG7"],
    "Cytotoxic_T_NK": ["NKG7", "GNLY", "PRF1", "GZMB", "GZMH", "KLRD1"],
    "NK_cells": ["GNLY", "NKG7", "KLRD1", "FCGR3A", "TYROBP"],
    "B_cells": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "Plasma_cells": ["MZB1", "JCHAIN", "XBP1", "IGKC", "IGHG1"],
    "Monocytes": ["LYZ", "LST1", "S100A8", "S100A9", "FCN1", "CD14"],
    "Dendritic_cells": ["HLA-DRA", "CST3", "FCER1A", "CLEC10A", "ITGAX"],
    "ASDC": ["AXL", "SIGLEC6", "LILRA4", "PPP1R14A", "CD22"],
    "LAMP3_DC": ["LAMP3", "CCR7", "CCL19", "CCL22", "FSCN1"],
    "pDC": ["IL3RA", "LILRA4", "GZMB", "TCF4", "IRF7"],
    "BAM_like_myeloid": ["C1QA", "C1QB", "APOE", "TREM2", "MRC1", "CD163"],
}


def find_sample_files(raw_dir, accession):
    matrix = sorted(raw_dir.glob(f"{accession}_*_matrix.mtx.gz"))
    if not matrix:
        raise FileNotFoundError(f"No matrix file found for {accession}")
    matrix = matrix[0]
    prefix = str(matrix).removesuffix("_matrix.mtx.gz")
    barcodes = Path(prefix + "_barcodes.tsv.gz")
    genes = Path(prefix + "_genes.tsv.gz")
    features = Path(prefix + "_features.tsv.gz")
    feature_file = genes if genes.exists() else features
    if not barcodes.exists() or not feature_file.exists():
        raise FileNotFoundError(f"Incomplete 10x files for {accession}")
    return matrix, barcodes, feature_file


def read_one_sample(row, raw_dir):
    matrix, barcodes, features = find_sample_files(raw_dir, row["sample_accession"])
    adata = sc.read_mtx(matrix).T
    barcodes_df = pd.read_csv(barcodes, sep="\t", header=None)
    features_df = pd.read_csv(features, sep="\t", header=None)
    gene_ids = features_df.iloc[:, 0].astype(str).to_numpy()
    if features_df.shape[1] > 1:
        gene_symbols = features_df.iloc[:, 1].astype(str).to_numpy()
    else:
        gene_symbols = gene_ids
    adata.obs_names = [
        f"{row['sample_accession']}:{bc}" for bc in barcodes_df.iloc[:, 0].astype(str)
    ]
    adata.var_names = gene_symbols
    adata.var["gene_id"] = gene_ids
    adata.var_names_make_unique()
    for key in [
        "gse",
        "sample_accession",
        "sample_title",
        "subject_id",
        "tissue",
        "diagnosis",
        "disease_group",
    ]:
        adata.obs[key] = row[key]
    return adata


def summarize_by_sample(adata, group_key, out_path):
    metrics = [
        "n_barcodes_input",
        "n_cells_nonzero",
        "n_cells_qc_pass",
        "median_counts",
        "median_genes",
        "median_pct_mt",
        "pct_qc_pass_of_nonzero",
    ]
    records = []
    for sample, frame in adata.obs.groupby(group_key):
        nonzero = frame["nonzero_counts"]
        denominator = int(nonzero.sum())
        record = {
            "sample_accession": sample,
            "n_barcodes_input": int(frame.shape[0]),
            "n_cells_nonzero": denominator,
            "n_cells_qc_pass": int(frame["qc_pass"].sum()),
            "median_counts": float(frame.loc[nonzero, "total_counts"].median()),
            "median_genes": float(frame.loc[nonzero, "n_genes_by_counts"].median()),
            "median_pct_mt": float(frame.loc[nonzero, "pct_counts_mt"].median()),
            "pct_qc_pass_of_nonzero": float(frame["qc_pass"].sum() / max(denominator, 1) * 100),
        }
        first = frame.iloc[0]
        for key in ["sample_title", "subject_id", "tissue", "diagnosis", "disease_group"]:
            record[key] = first[key]
        records.append(record)
    out = pd.DataFrame(records)
    out = out[
        ["sample_accession", "sample_title", "subject_id", "tissue", "diagnosis", "disease_group"]
        + metrics
    ]
    out.sort_values(["tissue", "disease_group", "sample_title"]).to_csv(out_path, index=False)
    return out


def plot_qc(adata, out_path):
    obs = adata.obs.loc[adata.obs["nonzero_counts"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, ylabel in zip(
        axes,
        ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        ["UMIs", "Genes", "% mitochondrial"],
    ):
        obs.boxplot(column=metric, by="tissue", ax=ax, grid=False)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def score_marker_sets(adata):
    score_cols = []
    present = set(adata.var_names)
    for label, genes in MARKER_SETS.items():
        genes_present = [g for g in genes if g in present]
        score_name = f"score_{label}"
        if len(genes_present) >= 2:
            sc.tl.score_genes(adata, genes_present, score_name=score_name, use_raw=False)
        else:
            adata.obs[score_name] = np.nan
        score_cols.append(score_name)
    return score_cols


def run_celltypist(adata, leiden_key):
    try:
        import celltypist

        predictions = celltypist.annotate(
            adata,
            model="Immune_All_Low.pkl",
            majority_voting=True,
            over_clustering=leiden_key,
            p_thres=0.5,
        )
        ct = predictions.to_adata()
        adata.obs["celltypist_label"] = ct.obs["predicted_labels"].astype(str).to_numpy()
        adata.obs["celltypist_majority"] = ct.obs["majority_voting"].astype(str).to_numpy()
        adata.obs["celltypist_conf"] = pd.to_numeric(ct.obs["conf_score"], errors="coerce").to_numpy()
        adata.obs.loc[adata.obs["celltypist_conf"] < 0.5, "celltypist_majority"] = "Unknown"
        return True, ""
    except Exception as exc:
        adata.obs["celltypist_label"] = "not_run"
        adata.obs["celltypist_majority"] = "not_run"
        adata.obs["celltypist_conf"] = np.nan
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=6000)
    parser.add_argument("--max-mt", type=float, default=20.0)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--leiden-resolution", type=float, default=0.6)
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = root / "data/raw/GSE138266"
    meta = pd.read_csv(root / "metadata/GSE138266_sample_metadata.csv")
    meta = meta.sort_values("sample_accession")

    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=100, facecolor="white")
    adatas = [read_one_sample(row, raw_dir) for _, row in meta.iterrows()]
    adata = ad.concat(adatas, join="outer", merge="same", index_unique=None)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata.obs["nonzero_counts"] = adata.obs["total_counts"] > 0
    adata.obs["qc_pass"] = (
        (adata.obs["n_genes_by_counts"] >= args.min_genes)
        & (adata.obs["n_genes_by_counts"] <= args.max_genes)
        & (adata.obs["pct_counts_mt"] <= args.max_mt)
    )

    qc_dir = root / "results/qc"
    ann_dir = root / "results/annotation"
    qc_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    summarize_by_sample(adata, "sample_accession", qc_dir / "GSE138266_sample_qc_summary.csv")
    adata.obs.to_csv(qc_dir / "GSE138266_cell_qc_metrics.csv.gz", compression="gzip")
    plot_qc(adata, qc_dir / "GSE138266_qc_boxplot_by_tissue.png")

    adata = adata[adata.obs["qc_pass"]].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_top_genes,
        flavor="seurat",
        batch_key="sample_accession",
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    leiden_key = "leiden_0_6"
    sc.tl.leiden(adata, resolution=args.leiden_resolution, key_added=leiden_key)

    score_cols = score_marker_sets(adata)
    cluster_means = adata.obs.groupby(leiden_key, observed=True)[score_cols].mean()
    marker_labels = cluster_means.idxmax(axis=1).str.replace("^score_", "", regex=True)
    cluster_means.insert(0, "marker_based_label", marker_labels)
    cluster_means.to_csv(ann_dir / "GSE138266_cluster_marker_scores.csv")
    adata.obs["marker_based_label"] = adata.obs[leiden_key].map(marker_labels).astype(str)

    ct_ok, ct_error = run_celltypist(adata, leiden_key)
    adata.obs["initial_annotation"] = adata.obs["celltypist_majority"].astype(str)
    fallback = adata.obs["initial_annotation"].isin(["Unknown", "not_run", "nan"])
    adata.obs.loc[fallback, "initial_annotation"] = adata.obs.loc[fallback, "marker_based_label"]

    annotation_cols = [
        "gse",
        "sample_accession",
        "sample_title",
        "subject_id",
        "tissue",
        "diagnosis",
        "disease_group",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        leiden_key,
        "marker_based_label",
        "celltypist_label",
        "celltypist_majority",
        "celltypist_conf",
        "initial_annotation",
    ]
    adata.obs[annotation_cols].to_csv(
        ann_dir / "GSE138266_cell_initial_annotation.csv.gz", compression="gzip"
    )

    cluster_summary = (
        adata.obs.groupby([leiden_key, "initial_annotation"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values([leiden_key, "n_cells"], ascending=[True, False])
    )
    cluster_summary.to_csv(ann_dir / "GSE138266_cluster_annotation_summary.csv", index=False)

    sc.pl.umap(
        adata,
        color=["tissue", "disease_group", leiden_key, "initial_annotation"],
        ncols=2,
        frameon=False,
        show=False,
    )
    plt.savefig(ann_dir / "GSE138266_umap_initial_annotation.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    marker_groups = {
        k: [g for g in v if g in adata.raw.var_names]
        for k, v in MARKER_SETS.items()
        if any(g in adata.raw.var_names for g in v)
    }
    sc.pl.dotplot(
        adata,
        marker_groups,
        groupby="initial_annotation",
        standard_scale="var",
        show=False,
    )
    plt.savefig(ann_dir / "GSE138266_marker_dotplot_initial_annotation.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    adata.write(root / "results/GSE138266_qc_initial_annotation.h5ad", compression="gzip")
    with open(root / "logs/GSE138266_qc_annotation_run.log", "w", encoding="utf-8") as handle:
        handle.write(f"n_cells_qc_pass={adata.n_obs}\n")
        handle.write(f"n_hvg={adata.n_vars}\n")
        handle.write(f"celltypist_ok={ct_ok}\n")
        if ct_error:
            handle.write(f"celltypist_error={ct_error}\n")
    print(f"Saved filtered annotated object: {adata.n_obs} cells x {adata.n_vars} HVGs")
    print(f"CellTypist status: {ct_ok}")


if __name__ == "__main__":
    main()
