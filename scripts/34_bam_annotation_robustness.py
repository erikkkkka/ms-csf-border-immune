#!/usr/bin/env python3
"""Audit BAM-like annotation stability across clustering and marker choices."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats


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

RESOLUTIONS = [0.4, 0.6, 0.8, 1.0, 1.2]
REFERENCE_LABEL = "BAM_like_candidate"
MONOCYTE_COMPETITORS = ["classical_monocyte", "nonclassical_monocyte", "DC3_moDC"]
CONTROL_GROUPS = {
    "GSE138266": "IIH_control",
    "GSE286068": "non_inflammatory_control",
}


def expression_matrix(adata: sc.AnnData, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    if adata.raw is None:
        raise RuntimeError("APC object has no full-gene raw matrix")
    present = [gene for gene in genes if gene in adata.raw.var_names]
    values = adata.raw[:, present].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32), present


def deterministic_marker_scores(
    adata: sc.AnnData,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    unique_genes = sorted({gene for genes in APC_MARKERS.values() for gene in genes})
    expression, present = expression_matrix(adata, unique_genes)
    means = expression.mean(axis=0)
    standard_deviations = expression.std(axis=0)
    standard_deviations[standard_deviations == 0] = 1
    z_scores = (expression - means) / standard_deviations
    gene_to_column = {gene: index for index, gene in enumerate(present)}
    module_scores = {}
    available = {}
    for label, genes in APC_MARKERS.items():
        available[label] = [gene for gene in genes if gene in gene_to_column]
        indices = [gene_to_column[gene] for gene in available[label]]
        module_scores[label] = z_scores[:, indices].mean(axis=1)
    scores = pd.DataFrame(module_scores, index=adata.obs_names)
    gene_z = pd.DataFrame(z_scores, index=adata.obs_names, columns=present)
    return scores, gene_z, available


def label_clusters(
    obs: pd.DataFrame,
    scores: pd.DataFrame,
    cluster_key: str,
    bam_score: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    working = scores.copy()
    if bam_score is not None:
        working[REFERENCE_LABEL] = bam_score
    cluster_scores = working.groupby(obs[cluster_key], observed=True).mean()
    values = cluster_scores.to_numpy()
    order = np.argsort(-values, axis=1)
    top = cluster_scores.columns.to_numpy()[order[:, 0]]
    margin = values[np.arange(len(values)), order[:, 0]] - values[
        np.arange(len(values)), order[:, 1]
    ]
    labels = pd.Series(top, index=cluster_scores.index, dtype="object")
    sizes = obs.groupby(cluster_key, observed=True).size().reindex(cluster_scores.index)
    labels[(margin < 0.10) | (sizes < 20)] = "APC_ambiguous"
    if "celltypist_label" in obs:
        pdc_fraction = (
            obs.assign(celltypist_pdc=obs["celltypist_label"].astype(str).eq("pDC"))
            .groupby(cluster_key, observed=True)["celltypist_pdc"]
            .mean()
            .reindex(cluster_scores.index, fill_value=0)
        )
        labels[(labels == "pDC") & (pdc_fraction < 0.25)] = "APC_ambiguous"
    else:
        pdc_fraction = pd.Series(np.nan, index=cluster_scores.index)
    audit = cluster_scores.copy()
    audit.insert(0, "label", labels)
    audit.insert(1, "score_margin", margin)
    audit.insert(2, "n_cells", sizes)
    audit.insert(3, "celltypist_pdc_fraction", pdc_fraction)
    cell_labels = obs[cluster_key].map(labels).astype(str)
    return cell_labels, audit


def overlap_metrics(reference: pd.Series, candidate: pd.Series) -> dict:
    reference = reference.to_numpy(dtype=bool)
    candidate = candidate.to_numpy(dtype=bool)
    intersection = np.sum(reference & candidate)
    union = np.sum(reference | candidate)
    return {
        "n_reference": int(reference.sum()),
        "n_candidate": int(candidate.sum()),
        "n_intersection": int(intersection),
        "precision_vs_reference": intersection / candidate.sum() if candidate.sum() else np.nan,
        "recall_vs_reference": intersection / reference.sum() if reference.sum() else np.nan,
        "jaccard": intersection / union if union else np.nan,
    }


def multiresolution_audit(
    adata: sc.AnnData, scores: pd.DataFrame, gse: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = adata.obs["fine_apc_label"].astype(str).eq(REFERENCE_LABEL)
    overlap_rows = []
    cluster_rows = []
    membership = pd.DataFrame(index=adata.obs_names)
    for resolution in RESOLUTIONS:
        key = f"leiden_robust_{str(resolution).replace('.', '_')}"
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            random_state=12345,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        labels, audit = label_clusters(adata.obs, scores, key)
        bam = labels.eq(REFERENCE_LABEL)
        membership[f"bam_res_{resolution}"] = bam.to_numpy()
        row = {"gse": gse, "resolution": resolution}
        row.update(overlap_metrics(reference, bam))
        overlap_rows.append(row)
        audit = audit.reset_index().rename(columns={key: "cluster"})
        audit.insert(0, "gse", gse)
        audit.insert(1, "resolution", resolution)
        cluster_rows.append(audit)
    membership["bam_resolution_count"] = membership.sum(axis=1)
    return pd.DataFrame(overlap_rows), pd.concat(cluster_rows, ignore_index=True), membership


def leave_one_out_audit(
    adata: sc.AnnData,
    scores: pd.DataFrame,
    gene_z: pd.DataFrame,
    available: dict[str, list[str]],
    gse: str,
) -> pd.DataFrame:
    reference = adata.obs["fine_apc_label"].astype(str).eq(REFERENCE_LABEL)
    rows = []
    genes = available[REFERENCE_LABEL]
    variants = [("none", genes)] + [(gene, [item for item in genes if item != gene]) for gene in genes]
    for omitted, retained in variants:
        bam_score = gene_z[retained].mean(axis=1)
        labels, _ = label_clusters(adata.obs, scores, "leiden_apc_0_8", bam_score=bam_score)
        bam = labels.eq(REFERENCE_LABEL)
        row = {"gse": gse, "omitted_marker": omitted, "n_markers_retained": len(retained)}
        row.update(overlap_metrics(reference, bam))
        rows.append(row)
    return pd.DataFrame(rows)


def marker_expression_audit(
    adata: sc.AnnData, cell_labels: pd.DataFrame, gse: str
) -> pd.DataFrame:
    genes = APC_MARKERS[REFERENCE_LABEL] + [
        "FOLR2", "DAB2", "MSR1", "CD163", "HAMP", "FCN1", "S100A8", "S100A9", "VCAN", "CCR2"
    ]
    expression, present = expression_matrix(adata, genes)
    frame = pd.DataFrame(expression, index=adata.obs_names, columns=present)
    rows = []
    definitions = {
        "reference_BAM": cell_labels["bam_reference"],
        "stable_core": cell_labels["bam_stable_core"],
        "classical_monocyte": adata.obs["fine_apc_label"].astype(str).eq("classical_monocyte"),
        "cDC2": adata.obs["fine_apc_label"].astype(str).eq("cDC2"),
    }
    for definition, mask in definitions.items():
        for gene in present:
            values = frame.loc[mask, gene]
            rows.append(
                {
                    "gse": gse,
                    "definition": definition,
                    "gene": gene,
                    "n_cells": len(values),
                    "mean_log1p_expression": values.mean(),
                    "pct_detected": 100 * values.gt(0).mean(),
                }
            )
    return pd.DataFrame(rows)


def sample_lta4h_scores(
    adata: sc.AnnData, cell_labels: pd.DataFrame, gse: str
) -> pd.DataFrame:
    expression, present = expression_matrix(adata, ["LTA4H"])
    if not present:
        raise RuntimeError(f"LTA4H is absent from {gse}")
    obs = adata.obs[["sample_accession", "subject_id", "tissue", "diagnosis", "disease_group"]].copy()
    obs["LTA4H_log1p_expression"] = expression[:, 0]
    obs["LTA4H_detected"] = expression[:, 0] > 0
    rows = []
    definitions = {
        "reference_BAM": cell_labels["bam_reference"],
        "stable_core": cell_labels["bam_stable_core"],
    }
    for definition, mask in definitions.items():
        selected = obs.loc[mask].copy()
        summary = (
            selected.groupby(
                ["sample_accession", "subject_id", "tissue", "diagnosis", "disease_group"],
                observed=True,
                dropna=False,
            )
            .agg(
                n_cells=("LTA4H_log1p_expression", "size"),
                mean_LTA4H_log1p=("LTA4H_log1p_expression", "mean"),
                pct_LTA4H_detected=("LTA4H_detected", "mean"),
            )
            .reset_index()
        )
        summary["pct_LTA4H_detected"] *= 100
        summary.insert(0, "gse", gse)
        summary.insert(1, "definition", definition)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def tissue_counts(adata: sc.AnnData, labels: pd.DataFrame, gse: str) -> pd.DataFrame:
    rows = []
    for definition, column in [
        ("reference_BAM", "bam_reference"),
        ("stable_core", "bam_stable_core"),
    ]:
        selected = adata.obs.loc[labels[column]]
        summary = (
            selected.groupby("tissue", observed=True)
            .agg(n_cells=("sample_accession", "size"), n_samples=("sample_accession", "nunique"))
            .reset_index()
        )
        summary.insert(0, "gse", gse)
        summary.insert(1, "definition", definition)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def disease_tests(sample_scores: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    rows = []
    for (gse, definition), frame in sample_scores.groupby(["gse", "definition"], observed=True):
        if gse not in CONTROL_GROUPS:
            continue
        frame = frame[(frame["tissue"] == "CSF") & (frame["n_cells"] >= min_cells)].copy()
        control_group = CONTROL_GROUPS[gse]
        for metric in ["mean_LTA4H_log1p", "pct_LTA4H_detected"]:
            case = frame.loc[frame["disease_group"].eq("MS"), metric].dropna()
            control = frame.loc[frame["disease_group"].eq(control_group), metric].dropna()
            if len(case) < 4 or len(control) < 4:
                continue
            estimate = case.mean() - control.mean()
            se = np.sqrt(case.var(ddof=1) / len(case) + control.var(ddof=1) / len(control))
            test = stats.ttest_ind(case, control, equal_var=False)
            rows.append(
                {
                    "gse": gse,
                    "definition": definition,
                    "metric": metric,
                    "n_ms": len(case),
                    "n_control": len(control),
                    "mean_ms": case.mean(),
                    "mean_control": control.mean(),
                    "mean_difference": estimate,
                    "standard_error": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                    "p_value": test.pvalue,
                    "test": "donor_level_welch_t",
                }
            )
    return pd.DataFrame(rows)


def meta_disease_tests(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (definition, metric), data in tests.groupby(["definition", "metric"], observed=True):
        if data["gse"].nunique() != 2:
            continue
        y = data["mean_difference"].to_numpy()
        variance = data["standard_error"].to_numpy() ** 2
        weights = 1 / variance
        fixed = np.sum(weights * y) / np.sum(weights)
        fixed_se = np.sqrt(1 / np.sum(weights))
        q = np.sum(weights * (y - fixed) ** 2)
        df = len(y) - 1
        c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
        random_weights = 1 / (variance + tau2)
        random = np.sum(random_weights * y) / np.sum(random_weights)
        random_se = np.sqrt(1 / np.sum(random_weights))
        for method, estimate, se in [
            ("inverse_variance_fixed", fixed, fixed_se),
            ("dersimonian_laird_random", random, random_se),
        ]:
            rows.append(
                {
                    "definition": definition,
                    "metric": metric,
                    "method": method,
                    "n_cohorts": len(data),
                    "estimate": estimate,
                    "standard_error": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                    "p_value": 2 * stats.norm.sf(abs(estimate / se)),
                    "cochran_q": q,
                    "q_p_value": stats.chi2.sf(q, df),
                    "i_squared_percent": max(0.0, (q - df) / q) * 100 if q > 0 else 0.0,
                    "tau_squared": tau2,
                }
            )
    return pd.DataFrame(rows)


def plot_retention(summary: pd.DataFrame, output: Path) -> None:
    cohorts = summary["gse"].tolist()
    retained = 100 * summary["n_stable_core"] / summary["n_reference_BAM"]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    bars = ax.bar(cohorts, retained, color=["#397367", "#5b7c99", "#b64b4b"], width=0.62)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in retained], padding=3)
    ax.set_ylabel("Reference BAM cells retained in stable core (%)")
    ax.set_ylim(0, max(105, retained.max() + 12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results" / "bam_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    sample_tables = []
    tissue_tables = []
    for gse in ["GSE138266", "GSE194078", "GSE286068"]:
        adata = sc.read_h5ad(root / "results" / "cleaned" / f"{gse}_clean_apc_refined.h5ad")
        scores, gene_z, available = deterministic_marker_scores(adata)
        overlap, clusters, membership = multiresolution_audit(adata, scores, gse)
        reference = adata.obs["fine_apc_label"].astype(str).eq(REFERENCE_LABEL)
        all_competitors = [column for column in scores if column != REFERENCE_LABEL]
        all_margin = scores[REFERENCE_LABEL] - scores[all_competitors].max(axis=1)
        mono_margin = scores[REFERENCE_LABEL] - scores[MONOCYTE_COMPETITORS].max(axis=1)
        labels = pd.DataFrame(index=adata.obs_names)
        labels["bam_reference"] = reference
        labels["bam_resolution_count"] = membership["bam_resolution_count"]
        labels["bam_marker_margin_all"] = all_margin
        labels["bam_marker_margin_monocyte"] = mono_margin
        labels["bam_monocyte_excluded"] = reference & mono_margin.gt(0)
        labels["bam_stable_core"] = (
            reference & membership["bam_resolution_count"].ge(4) & all_margin.gt(0)
        )
        labels.insert(0, "gse", gse)
        labels.to_csv(out_dir / f"{gse}_BAM_cell_stability.csv.gz", compression="gzip")
        overlap.to_csv(out_dir / f"{gse}_BAM_multiresolution_overlap.csv", index=False)
        clusters.to_csv(out_dir / f"{gse}_BAM_multiresolution_cluster_scores.csv.gz", index=False, compression="gzip")
        leave_one_out_audit(adata, scores, gene_z, available, gse).to_csv(
            out_dir / f"{gse}_BAM_marker_leave_one_out.csv", index=False
        )
        marker_expression_audit(adata, labels, gse).to_csv(
            out_dir / f"{gse}_BAM_marker_expression.csv", index=False
        )
        sample_tables.append(sample_lta4h_scores(adata, labels, gse))
        tissue_tables.append(tissue_counts(adata, labels, gse))
        summary_rows.append(
            {
                "gse": gse,
                "n_apc": adata.n_obs,
                "n_reference_BAM": int(reference.sum()),
                "n_monocyte_excluded": int(labels["bam_monocyte_excluded"].sum()),
                "n_stable_core": int(labels["bam_stable_core"].sum()),
                "stable_core_recall": labels["bam_stable_core"].sum() / reference.sum(),
                "median_resolution_count_reference_BAM": labels.loc[
                    reference, "bam_resolution_count"
                ].median(),
                "pct_reference_BAM_top_individual_module": 100 * all_margin.loc[reference].gt(0).mean(),
                "pct_reference_BAM_above_monocyte_modules": 100 * mono_margin.loc[reference].gt(0).mean(),
            }
        )
        del adata

    summary = pd.DataFrame(summary_rows)
    sample_scores = pd.concat(sample_tables, ignore_index=True)
    tissues = pd.concat(tissue_tables, ignore_index=True)
    tests = disease_tests(sample_scores, args.min_cells)
    meta = meta_disease_tests(tests)
    summary.to_csv(out_dir / "BAM_robustness_summary.csv", index=False)
    sample_scores.to_csv(out_dir / "BAM_LTA4H_sample_scores.csv", index=False)
    tissues.to_csv(out_dir / "BAM_tissue_restriction.csv", index=False)
    tests.to_csv(out_dir / "BAM_LTA4H_donor_disease_tests.csv", index=False)
    meta.to_csv(out_dir / "BAM_LTA4H_donor_disease_meta.csv", index=False)
    plot_retention(summary, out_dir / "BAM_stable_core_retention.png")

    print("BAM robustness summary")
    print(summary.to_string(index=False))
    print("\nTissue restriction")
    print(tissues.to_string(index=False))
    print("\nDonor-level LTA4H tests")
    print(tests.to_string(index=False))
    print("\nDonor-level meta-analysis")
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
