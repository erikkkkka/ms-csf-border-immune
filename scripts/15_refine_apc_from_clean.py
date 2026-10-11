#!/usr/bin/env python3
import argparse
import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def load_pipeline_helpers(root):
    path = root / "scripts/06_recluster_refine_annotation_gse138266.py"
    spec = importlib.util.spec_from_file_location("clean_annotation_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--gse", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    gse = args.gse
    helpers = load_pipeline_helpers(root)
    cleaned_dir = root / "results/cleaned"
    ann_dir = root / "results/annotation_cleaned"

    integrated = sc.read_h5ad(cleaned_dir / f"{gse}_clean_decontx_singlets.h5ad")
    if integrated.raw is None:
        raise RuntimeError("Clean integrated object has no full-gene raw matrix")
    full = integrated.raw.to_adata()
    full.obs = integrated.obs.copy()

    apc = full[full.obs["broad_cell_type"].isin(["Myeloid", "Dendritic"])].copy()
    score_cols = helpers.score_sets(apc, helpers.APC_MARKERS, "score_apc_")
    apc.raw = apc
    sc.pp.highly_variable_genes(
        apc,
        n_top_genes=min(2500, apc.n_vars),
        flavor="seurat",
        batch_key="sample_accession",
    )
    analysis = apc[:, apc.var["highly_variable"]].copy()
    sc.pp.scale(analysis, max_value=10)
    sc.tl.pca(analysis, n_comps=40, svd_solver="arpack")
    helpers.harmony_integrate(analysis, "sample_accession")
    sc.pp.neighbors(analysis, n_neighbors=15, n_pcs=25, use_rep="X_pca_harmony")
    sc.tl.umap(analysis, random_state=12345)
    cluster_key = "leiden_apc_0_8"
    sc.tl.leiden(analysis, resolution=0.8, key_added=cluster_key, random_state=12345)
    apc.obs[cluster_key] = analysis.obs[cluster_key].astype(str)

    means = apc.obs.groupby(cluster_key, observed=True)[score_cols].mean()
    order = np.argsort(-means.to_numpy(), axis=1)
    top = means.columns.to_numpy()[order[:, 0]]
    margin = means.to_numpy()[np.arange(len(means)), order[:, 0]] - means.to_numpy()[
        np.arange(len(means)), order[:, 1]
    ]
    labels = pd.Series(top, index=means.index).str.replace("score_apc_", "", regex=False)
    sizes = apc.obs.groupby(cluster_key, observed=True).size().reindex(means.index)
    labels[(margin < 0.10) | (sizes < 20)] = "APC_ambiguous"
    pdc_fraction = (
        apc.obs.assign(is_celltypist_pdc=apc.obs["celltypist_label"].astype(str).eq("pDC"))
        .groupby(cluster_key, observed=True)["is_celltypist_pdc"]
        .mean()
        .reindex(means.index, fill_value=0)
    )
    labels[(labels == "pDC") & (pdc_fraction < 0.25)] = "APC_ambiguous"
    means.insert(0, "fine_apc_label", labels)
    means.insert(1, "score_margin", margin)
    means.insert(2, "n_cells", sizes)
    means.insert(3, "celltypist_pdc_fraction", pdc_fraction)
    means.to_csv(ann_dir / f"{gse}_apc_cluster_marker_scores.csv")
    apc.obs["fine_apc_label"] = apc.obs[cluster_key].map(labels).astype(str)

    for col in [cluster_key, "fine_apc_label"] + score_cols:
        analysis.obs[col] = apc.obs[col]
    analysis.write(cleaned_dir / f"{gse}_clean_apc_refined.h5ad", compression="gzip")
    apc.obs[[
        "sample_accession", "subject_id", "tissue", "diagnosis", "disease_group",
        cluster_key, "fine_apc_label",
    ] + score_cols].to_csv(
        ann_dir / f"{gse}_clean_apc_cell_annotations.csv.gz", compression="gzip"
    )

    color_by = "diagnosis" if gse == "GSE286068" else "tissue"
    sc.pl.umap(
        analysis,
        color=[color_by, "disease_group", cluster_key, "fine_apc_label"],
        ncols=2,
        frameon=False,
        show=False,
    )
    plt.savefig(ann_dir / f"{gse}_clean_apc_umap.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    present = {k: [gene for gene in genes if gene in analysis.raw.var_names] for k, genes in helpers.APC_MARKERS.items()}
    sc.pl.dotplot(
        analysis,
        present,
        groupby="fine_apc_label",
        use_raw=True,
        standard_scale="var",
        show=False,
    )
    plt.savefig(ann_dir / f"{gse}_clean_apc_marker_dotplot.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    with open(root / f"logs/{gse}_apc_refinement.log", "w", encoding="utf-8") as handle:
        handle.write(f"n_strict_apc={apc.n_obs}\n")
        handle.write("entry_rule=broad_cell_type_in_Myeloid_or_Dendritic\n")
        handle.write("pdc_rule=cluster_marker_score_plus_celltypist_pdc_fraction_ge_0.25\n")


if __name__ == "__main__":
    main()
