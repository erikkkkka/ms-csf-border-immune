#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, wilcoxon


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    keep = np.flatnonzero(np.isfinite(values))
    if len(keep) == 0:
        return result
    order = keep[np.argsort(values[keep])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def sample_composition(cells, label_col, meta, total_cells, compartment_name):
    labels = sorted(cells[label_col].dropna().astype(str).unique())
    counts = cells.groupby(["sample_accession", label_col], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(index=meta["sample_accession"], columns=labels, fill_value=0)
    long = counts.stack().rename("n_label").reset_index().rename(columns={label_col: "label", "level_1": "label"})
    long = long.merge(meta[["sample_accession", "subject_id", "tissue", "disease_group"]], on="sample_accession")
    long = long.merge(total_cells.rename("n_total_cells"), left_on="sample_accession", right_index=True)
    compartment_cells = cells.groupby("sample_accession", observed=True).size().reindex(meta["sample_accession"], fill_value=0)
    long = long.merge(compartment_cells.rename("n_compartment_cells"), left_on="sample_accession", right_index=True)
    long["prop_total"] = long["n_label"] / long["n_total_cells"]
    long["prop_compartment"] = long["n_label"] / long["n_compartment_cells"].replace(0, np.nan)
    long["compartment"] = compartment_name
    return long


def group_tests(compositions, case_label, control_label):
    rows = []
    metrics = ["prop_total"]
    if compositions["compartment"].iloc[0] == "APC":
        metrics.append("prop_compartment")
    for metric in metrics:
        for (tissue, label), frame in compositions.groupby(["tissue", "label"], observed=True):
            ms = frame.loc[frame["disease_group"] == case_label, metric].dropna().to_numpy()
            control = frame.loc[frame["disease_group"] == control_label, metric].dropna().to_numpy()
            if len(ms) == 0 or len(control) == 0:
                continue
            test = mannwhitneyu(ms, control, alternative="two-sided", method="auto")
            rows.append({
                "compartment": frame["compartment"].iloc[0],
                "metric": metric,
                "tissue": tissue,
                "label": label,
                "n_ms": len(ms),
                "n_control": len(control),
                "median_ms": np.median(ms),
                "median_control": np.median(control),
                "median_difference": np.median(ms) - np.median(control),
                "rank_biserial_ms_vs_control": 2 * test.statistic / (len(ms) * len(control)) - 1,
                "p_value": test.pvalue,
                "inferential_status": (
                    "adequate_minimum_group_size" if min(len(ms), len(control)) >= 4
                    else "descriptive_only_group_n_lt_4"
                ),
            })
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_adjust(out["p_value"])
    return out.sort_values(["fdr_bh", "p_value", "compartment", "metric"])


def paired_tissue_tests(compositions):
    rows = []
    metrics = ["prop_total"]
    if compositions["compartment"].iloc[0] == "APC":
        metrics.append("prop_compartment")
    for metric in metrics:
        for (disease, label), frame in compositions.groupby(["disease_group", "label"], observed=True):
            paired = frame.pivot_table(index="subject_id", columns="tissue", values=metric, aggfunc="first").dropna()
            if len(paired) < 3 or not {"CSF", "PBMC"}.issubset(paired.columns):
                continue
            diff = paired["CSF"] - paired["PBMC"]
            try:
                test = wilcoxon(paired["CSF"], paired["PBMC"], alternative="two-sided", method="auto")
                p_value = test.pvalue
            except ValueError:
                p_value = 1.0
            rows.append({
                "compartment": frame["compartment"].iloc[0],
                "metric": metric,
                "disease_group": disease,
                "label": label,
                "n_pairs": len(paired),
                "median_csf": paired["CSF"].median(),
                "median_pbmc": paired["PBMC"].median(),
                "median_paired_difference": diff.median(),
                "p_value": p_value,
            })
    out = pd.DataFrame(rows)
    out["fdr_bh"] = bh_adjust(out["p_value"])
    return out.sort_values(["fdr_bh", "p_value", "compartment", "metric"])


def detection_fraction(adata, rows, gene):
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0 or gene not in adata.raw.var_names:
        return np.nan
    column = adata.raw.X[rows, adata.raw.var_names.get_loc(gene)]
    return column.nnz / len(rows)


def marker_audit(root, apc_cells):
    genes = ["CD3D", "TRAC", "MS4A1", "CD79A", "LST1", "LYZ", "C1QA", "APOE", "MZB1", "JCHAIN"]
    initial = sc.read_h5ad(root / "results/GSE138266_qc_initial_annotation.h5ad", backed="r")
    clean = sc.read_h5ad(root / "results/cleaned/GSE138266_clean_decontx_singlets.h5ad", backed="r")
    clean_apc_ids = set(apc_cells.index[apc_cells["fine_apc_label"].astype(str) == "BAM_like_candidate"])
    groups = {
        "precleanup_plasma": (initial, np.flatnonzero(initial.obs["initial_annotation"].astype(str).str.contains("Plasma").to_numpy())),
        "postcleanup_plasma": (clean, np.flatnonzero((clean.obs["broad_cell_type"].astype(str) == "Plasma").to_numpy())),
        "precleanup_bam": (initial, np.flatnonzero((initial.obs["initial_annotation"].astype(str) == "BAM_like_myeloid").to_numpy())),
        "postcleanup_bam_candidate": (clean, np.flatnonzero(clean.obs_names.isin(clean_apc_ids))),
    }
    rows = []
    for group, (adata, indices) in groups.items():
        for gene in genes:
            rows.append({"group": group, "n_cells": len(indices), "gene": gene, "detection_fraction": detection_fraction(adata, indices, gene)})
    initial.file.close()
    clean.file.close()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--gse", default="GSE138266")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    gse = args.gse
    ann_dir = root / "results/annotation_cleaned"
    dedicated_meta = root / f"metadata/{gse}_sample_metadata.csv"
    if dedicated_meta.exists():
        meta = pd.read_csv(dedicated_meta)
    else:
        meta = pd.read_csv(root / "metadata/all_geo_sample_metadata.csv")
        meta = meta.loc[meta["gse"] == gse].copy()
    if gse == "GSE286068":
        meta = meta.loc[(meta["disease_group"] != "unknown") & ~meta["sample_title"].str.startswith("multiplexed")]
    meta = meta.sort_values("sample_accession")
    cells = pd.read_csv(ann_dir / f"{gse}_clean_cell_annotations.csv.gz", index_col=0)
    apc = pd.read_csv(ann_dir / f"{gse}_clean_apc_cell_annotations.csv.gz", index_col=0)
    total_cells = cells.groupby("sample_accession", observed=True).size()

    broad_comp = sample_composition(cells, "broad_cell_type", meta, total_cells, "Broad")
    apc_comp = sample_composition(apc, "fine_apc_label", meta, total_cells, "APC")
    broad_comp.to_csv(ann_dir / f"{gse}_clean_broad_sample_composition.csv", index=False)
    apc_comp.to_csv(ann_dir / f"{gse}_clean_apc_sample_composition.csv", index=False)
    control_label = "IIH_control" if gse == "GSE138266" else "non_inflammatory_control"
    pd.concat([group_tests(broad_comp, "MS", control_label), group_tests(apc_comp, "MS", control_label)], ignore_index=True).sort_values(
        ["fdr_bh", "p_value"]
    ).to_csv(ann_dir / f"{gse}_clean_composition_group_tests.csv", index=False)
    paired_meta = meta.groupby("subject_id", observed=True)["tissue"].nunique()
    if (paired_meta >= 2).any():
        pd.concat([paired_tissue_tests(broad_comp), paired_tissue_tests(apc_comp)], ignore_index=True).sort_values(
            ["fdr_bh", "p_value"]
        ).to_csv(ann_dir / f"{gse}_clean_paired_tissue_tests.csv", index=False)
    if gse == "GSE138266":
        marker_audit(root, apc).to_csv(ann_dir / f"{gse}_pre_post_marker_detection_audit.csv", index=False)


if __name__ == "__main__":
    main()
