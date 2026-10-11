#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("CELLTYPIST_FOLDER", str(ROOT / ".celltypist_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


BROAD_MARKERS = {
    "T": ["CD3D", "CD3E", "TRAC", "LTB"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD19", "CD37"],
    "Plasma": ["MZB1", "JCHAIN", "XBP1", "TNFRSF17", "SDC1"],
    "NK": ["NKG7", "GNLY", "KLRD1", "PRF1", "FCGR3A"],
    "Myeloid_APC": ["LYZ", "LST1", "TYROBP", "FCER1G", "CTSS"],
    "Dendritic": ["HLA-DRA", "FCER1A", "CD1C", "CLEC10A", "CST3"],
    "pDC": ["GZMB", "JCHAIN", "LILRA4", "CLEC4C", "TCF4"],
}

APC_MARKERS = {
    "classical_monocyte": ["S100A8", "S100A9", "FCN1", "VCAN", "CCR2", "SELL", "CTSD"],
    "nonclassical_monocyte": ["FCGR3A", "MS4A7", "LILRB1", "IFITM3", "LST1", "LYZ"],
    "cDC1": ["CLEC9A", "XCR1", "BATF3", "CADM1", "WDFY4"],
    "cDC2": ["CD1C", "FCER1A", "CLEC10A", "CD1E", "CST3"],
    "DC3_moDC": ["CD1C", "CD14", "FCGR3A", "CD163", "S100A8", "CTSS"],
    "ASDC": ["AXL", "SIGLEC6", "PPP1R14A", "CD22", "LILRA4"],
    "pDC": ["GZMB", "JCHAIN", "LILRA4", "CLEC4C", "TCF4", "IL3RA"],
    "LAMP3_DC": ["LAMP3", "CCR7", "FSCN1", "CCL19", "CCL22"],
    "BAM_like_candidate": ["C1QA", "C1QB", "C1QC", "APOE", "TREM2", "LYVE1", "MRC1", "STAB1", "CH25H"],
}


def read_clean_sample(sample_dir, row):
    adata = sc.read_mtx(sample_dir / "matrix.mtx.gz").T
    barcodes = pd.read_csv(sample_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    features = pd.read_csv(sample_dir / "features.tsv.gz", sep="\t", header=None)
    adata.obs_names = [f"{row.sample_accession}:{x}" for x in barcodes]
    adata.var_names = features.iloc[:, 1].astype(str).to_numpy()
    adata.var["gene_id"] = features.iloc[:, 0].astype(str).to_numpy()
    adata.var_names_make_unique()
    for key in ["gse", "sample_accession", "sample_title", "subject_id", "tissue", "diagnosis", "disease_group"]:
        adata.obs[key] = getattr(row, key)
    return adata


def score_sets(adata, marker_sets, prefix):
    columns = []
    present = set(adata.var_names)
    for label, genes in marker_sets.items():
        genes = [g for g in genes if g in present]
        col = f"{prefix}{label}"
        if len(genes) >= 2:
            sc.tl.score_genes(adata, genes, score_name=col, use_raw=False)
        else:
            adata.obs[col] = np.nan
        columns.append(col)
    return columns


def harmony_integrate(adata, key, basis="X_pca", adjusted_basis="X_pca_harmony"):
    import harmonypy

    result = harmonypy.run_harmony(adata.obsm[basis], adata.obs, [key], random_state=12345)
    corrected = np.asarray(result.Z_corr)
    if corrected.shape[0] == adata.n_obs:
        pass
    elif corrected.shape[1] == adata.n_obs:
        corrected = corrected.T
    else:
        raise ValueError(f"Unexpected Harmony output shape: {corrected.shape}")
    adata.obsm[adjusted_basis] = corrected.astype(np.float32, copy=False)


def run_celltypist(full, cluster_key):
    import celltypist

    last_error = ""
    for model in ["Immune_All_High.pkl", "Immune_All_Low.pkl"]:
        try:
            pred = celltypist.annotate(
                full,
                model=model,
                majority_voting=True,
                over_clustering=cluster_key,
                p_thres=0.5,
            ).to_adata()
            full.obs["celltypist_label"] = pred.obs["predicted_labels"].astype(str).to_numpy()
            full.obs["celltypist_majority"] = pred.obs["majority_voting"].astype(str).to_numpy()
            full.obs["celltypist_conf"] = pd.to_numeric(pred.obs["conf_score"], errors="coerce").to_numpy()
            full.obs.loc[full.obs["celltypist_conf"] < 0.5, "celltypist_majority"] = "Unknown"
            full.uns["celltypist_model"] = model
            return
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"CellTypist failed: {last_error}")


def broad_from_celltypist(label):
    x = str(label).lower()
    if x == "unknown":
        return "Unknown"
    if "plasma" in x or "plasmablast" in x:
        return "Plasma"
    if "dendritic" in x or x.startswith("dc") or "pdc" in x or "asdc" in x:
        return "Dendritic"
    if "monocyte" in x or "macrophage" in x or "mono-mac" in x or x == "mnp" or "myelocyte" in x:
        return "Myeloid"
    if "b cell" in x or "b-cell" in x or x.startswith("b_"):
        return "B"
    if "nk" in x or "natural killer" in x or x == "ilc":
        return "NK"
    if "t cell" in x or "helper t" in x or "cytotoxic t" in x or x.startswith("t_"):
        return "T"
    if "eryth" in x:
        return "Erythroid"
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--gse", default="GSE138266")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    gse = args.gse
    processed = root / f"data/processed/{gse}_decontx"
    cleaned_dir = root / "results/cleaned"
    ann_dir = root / "results/annotation_cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    dedicated_meta = root / f"metadata/{gse}_sample_metadata.csv"
    if dedicated_meta.exists():
        meta = pd.read_csv(dedicated_meta)
    else:
        meta = pd.read_csv(root / "metadata/all_geo_sample_metadata.csv")
        meta = meta.loc[meta["gse"] == gse].copy()
    if gse == "GSE286068":
        meta = meta.loc[(meta["disease_group"] != "unknown") & ~meta["sample_title"].str.startswith("multiplexed")]
    meta = meta.sort_values("sample_accession")
    cleanup = pd.read_csv(root / f"results/qc/{gse}_cleanup_cell_metrics.csv.gz", index_col="cell_id")
    adatas = [read_clean_sample(processed / row.sample_accession, row) for row in meta.itertuples(index=False)]
    full = sc.concat(adatas, join="outer", merge="same", index_unique=None)
    full.var_names_make_unique()
    full.obs = full.obs.join(cleanup[["scrublet_score", "scDblFinder_score", "decontX_contamination"]], how="left")
    full.layers["counts"] = full.X.copy()
    full.var["mt"] = full.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(full, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    post_decont_keep = (full.obs["total_counts"] > 0) & (full.obs["n_genes_by_counts"] >= 200)
    post_decont_summary = (
        full.obs.assign(post_decont_qc_pass=post_decont_keep)
        .groupby("sample_accession", observed=True)["post_decont_qc_pass"]
        .agg(n_postdoublet_input="size", n_postdecont_qc_pass="sum")
        .reset_index()
    )
    post_decont_summary["n_postdecont_removed"] = (
        post_decont_summary["n_postdoublet_input"] - post_decont_summary["n_postdecont_qc_pass"]
    )
    post_decont_summary.to_csv(root / f"results/qc/{gse}_postdecont_qc_summary.csv", index=False)
    full = full[post_decont_keep].copy()

    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)
    score_sets(full, BROAD_MARKERS, "score_broad_")
    full.raw = full
    sc.pp.highly_variable_genes(full, n_top_genes=3000, flavor="seurat", batch_key="sample_accession")
    analysis = full[:, full.var["highly_variable"]].copy()
    sc.pp.scale(analysis, max_value=10)
    sc.tl.pca(analysis, n_comps=50, svd_solver="arpack")
    harmony_integrate(analysis, key="sample_accession", basis="X_pca", adjusted_basis="X_pca_harmony")
    sc.pp.neighbors(analysis, n_neighbors=20, n_pcs=30, use_rep="X_pca_harmony")
    sc.tl.umap(analysis, random_state=12345)
    cluster_key = "leiden_clean_0_8"
    sc.tl.leiden(analysis, resolution=0.8, key_added=cluster_key, random_state=12345)
    full.obs[cluster_key] = analysis.obs[cluster_key].astype(str)
    run_celltypist(full, cluster_key)
    full.obs["broad_cell_type"] = full.obs["celltypist_label"].map(broad_from_celltypist)

    broad_score_cols = [c for c in full.obs if c.startswith("score_broad_")]
    fallback = (full.obs["celltypist_conf"] < 0.5) | full.obs["broad_cell_type"].isin(["Unknown", "Other"])
    best_score = full.obs[broad_score_cols].idxmax(axis=1).str.replace("score_broad_", "", regex=False)
    fallback_map = {"Myeloid_APC": "Myeloid", "Dendritic": "Dendritic", "pDC": "Dendritic"}
    full.obs.loc[fallback, "broad_cell_type"] = best_score[fallback].replace(fallback_map)

    for col in [cluster_key, "celltypist_label", "celltypist_majority", "celltypist_conf", "broad_cell_type"]:
        analysis.obs[col] = full.obs[col]
    analysis.write(cleaned_dir / f"{gse}_clean_decontx_singlets.h5ad", compression="gzip")

    apc_mask = full.obs["broad_cell_type"].isin(["Myeloid", "Dendritic"])
    apc = full[apc_mask].copy()
    apc_score_cols = score_sets(apc, APC_MARKERS, "score_apc_")
    apc.raw = apc
    sc.pp.highly_variable_genes(apc, n_top_genes=min(2500, apc.n_vars), flavor="seurat", batch_key="sample_accession")
    apc_analysis = apc[:, apc.var["highly_variable"]].copy()
    sc.pp.scale(apc_analysis, max_value=10)
    sc.tl.pca(apc_analysis, n_comps=40, svd_solver="arpack")
    harmony_integrate(apc_analysis, key="sample_accession", basis="X_pca", adjusted_basis="X_pca_harmony")
    sc.pp.neighbors(apc_analysis, n_neighbors=15, n_pcs=25, use_rep="X_pca_harmony")
    sc.tl.umap(apc_analysis, random_state=12345)
    apc_key = "leiden_apc_0_8"
    sc.tl.leiden(apc_analysis, resolution=0.8, key_added=apc_key, random_state=12345)
    apc.obs[apc_key] = apc_analysis.obs[apc_key].astype(str)

    means = apc.obs.groupby(apc_key, observed=True)[apc_score_cols].mean()
    order = np.argsort(-means.to_numpy(), axis=1)
    top = means.columns.to_numpy()[order[:, 0]]
    margin = means.to_numpy()[np.arange(len(means)), order[:, 0]] - means.to_numpy()[np.arange(len(means)), order[:, 1]]
    labels = pd.Series(top, index=means.index).str.replace("score_apc_", "", regex=False)
    labels[margin < 0.10] = "APC_ambiguous"
    cluster_sizes = apc.obs.groupby(apc_key, observed=True).size()
    labels[cluster_sizes < 20] = "APC_ambiguous"
    pdc_fraction = (
        apc.obs.assign(is_celltypist_pdc=apc.obs["celltypist_label"].astype(str).eq("pDC"))
        .groupby(apc_key, observed=True)["is_celltypist_pdc"]
        .mean()
        .reindex(means.index, fill_value=0)
    )
    labels[(labels == "pDC") & (pdc_fraction < 0.25)] = "APC_ambiguous"
    means.insert(0, "fine_apc_label", labels)
    means.insert(1, "score_margin", margin)
    means.insert(2, "n_cells", cluster_sizes)
    means.insert(3, "celltypist_pdc_fraction", pdc_fraction)
    means.to_csv(ann_dir / f"{gse}_apc_cluster_marker_scores.csv")
    apc.obs["fine_apc_label"] = apc.obs[apc_key].map(labels).astype(str)

    for col in [apc_key, "fine_apc_label"] + apc_score_cols:
        apc_analysis.obs[col] = apc.obs[col]
    apc_analysis.write(cleaned_dir / f"{gse}_clean_apc_refined.h5ad", compression="gzip")

    full.obs[[
        "sample_accession", "subject_id", "tissue", "diagnosis", "disease_group", cluster_key,
        "celltypist_label", "celltypist_majority", "celltypist_conf", "broad_cell_type",
        "scrublet_score", "scDblFinder_score", "decontX_contamination"
    ]].to_csv(ann_dir / f"{gse}_clean_cell_annotations.csv.gz", compression="gzip")
    apc.obs[["sample_accession", "subject_id", "tissue", "diagnosis", "disease_group", apc_key, "fine_apc_label"] + apc_score_cols].to_csv(
        ann_dir / f"{gse}_clean_apc_cell_annotations.csv.gz", compression="gzip"
    )

    sc.pl.umap(
        analysis,
        color=["diagnosis" if gse == "GSE286068" else "tissue", "disease_group", cluster_key, "broad_cell_type"],
        ncols=2,
        frameon=False,
        show=False,
    )
    plt.savefig(ann_dir / f"{gse}_clean_umap_overview.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    sc.pl.umap(
        apc_analysis,
        color=["diagnosis" if gse == "GSE286068" else "tissue", "disease_group", apc_key, "fine_apc_label"],
        ncols=2,
        frameon=False,
        show=False,
    )
    plt.savefig(ann_dir / f"{gse}_clean_apc_umap.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    present_markers = {k: [g for g in v if g in apc_analysis.raw.var_names] for k, v in APC_MARKERS.items()}
    sc.pl.dotplot(
        apc_analysis,
        present_markers,
        groupby="fine_apc_label",
        use_raw=True,
        standard_scale="var",
        show=False,
    )
    plt.savefig(ann_dir / f"{gse}_clean_apc_marker_dotplot.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    summary = full.obs.groupby(["tissue", "disease_group", "broad_cell_type"], observed=True).size().reset_index(name="n_cells")
    summary.to_csv(ann_dir / f"{gse}_clean_broad_composition.csv", index=False)
    with open(root / f"logs/{gse}_clean_reannotation.log", "w", encoding="utf-8") as handle:
        handle.write(f"n_clean_singlets={full.n_obs}\n")
        handle.write(f"n_apc_candidates={apc.n_obs}\n")
        handle.write(f"celltypist_model={full.uns['celltypist_model']}\n")
        handle.write(f"celltypist_conf_lt_0.5={(full.obs['celltypist_conf'] < 0.5).mean():.6f}\n")


if __name__ == "__main__":
    main()
