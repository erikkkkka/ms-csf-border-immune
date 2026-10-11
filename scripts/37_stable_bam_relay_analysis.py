#!/usr/bin/env python3
"""Retest cDC2-BAM relay using the strict BAM stable-core pseudobulk."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


MODULES = {
    "IFN_antiviral": [
        "IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2",
        "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1",
    ],
    "MHC_I_processing": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    ],
}


def load_module_scores(
    counts_path: Path,
    metadata_path: Path,
    label: str,
    min_cells: int,
) -> pd.DataFrame:
    counts = pd.read_csv(counts_path)
    metadata = pd.read_csv(metadata_path)
    if "analysis_label" in metadata:
        metadata = metadata[metadata["analysis_label"].eq(label)].copy()
    samples = metadata.loc[metadata["n_cells"].ge(min_cells), "sample_accession"].tolist()
    samples = [sample for sample in samples if sample in counts.columns]
    if len(samples) < 3:
        return pd.DataFrame(columns=["sample_accession"])
    gene_counts = counts.loc[counts["gene_symbol"].notna(), ["gene_symbol"] + samples]
    gene_counts = gene_counts.groupby("gene_symbol", observed=True)[samples].sum()
    library_sizes = counts[samples].sum(axis=0).astype(float)
    log_cpm = np.log2((gene_counts.T + 0.5).div(library_sizes + 1.0, axis=0) * 1e6)
    output = metadata.drop_duplicates("sample_accession").set_index("sample_accession").loc[
        samples, ["subject_id", "diagnosis", "disease_group", "n_cells"]
    ].reset_index()
    output = output.rename(columns={"n_cells": f"{label}_n_cells"})
    for module, genes in MODULES.items():
        present = [
            gene for gene in genes
            if gene in log_cpm.columns and (gene_counts.loc[gene] > 0).sum() >= 2
        ]
        matrix = log_cpm[present]
        standard_deviations = matrix.std(axis=0, ddof=0)
        informative = standard_deviations.index[standard_deviations.gt(0)]
        output[f"{label}_{module}_n_genes"] = len(informative)
        if len(informative) < 5:
            output[f"{label}_{module}"] = np.nan
            continue
        standardized = (
            matrix[informative] - matrix[informative].mean(axis=0)
        ) / standard_deviations[informative]
        output[f"{label}_{module}"] = standardized.mean(axis=1).to_numpy()
    return output


def residualize_ranks(values: np.ndarray, strata: pd.Series) -> np.ndarray:
    ranked = rankdata(values, method="average")
    design = pd.get_dummies(strata, drop_first=False, dtype=float).to_numpy()
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def pooled_test(
    table: pd.DataFrame,
    x_col: str,
    y_col: str,
    permutations: int,
    seed: int,
) -> dict:
    frame = table[["gse", "disease_group", x_col, y_col]].dropna().copy()
    frame["stratum"] = frame["gse"].astype(str) + ":" + frame["disease_group"].astype(str)
    x_residual = residualize_ranks(frame[x_col].to_numpy(), frame["stratum"])
    y_residual = residualize_ranks(frame[y_col].to_numpy(), frame["stratum"])
    observed = pearsonr(x_residual, y_residual).statistic
    rng = np.random.default_rng(seed)
    groups = [
        np.flatnonzero(frame["stratum"].to_numpy() == key)
        for key in frame["stratum"].unique()
    ]
    exceed = 0
    for _ in range(permutations):
        permuted = y_residual.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        exceed += abs(pearsonr(x_residual, permuted).statistic) >= abs(observed)
    return {
        "scope": "pooled_group_residualized",
        "gse": "pooled",
        "x": x_col,
        "y": y_col,
        "n_samples": len(frame),
        "correlation": observed,
        "p_value": (exceed + 1) / (permutations + 1),
        "method": f"rank_residual_pearson_block_permutation_{permutations}",
    }


def bh_adjust(values: pd.Series) -> np.ndarray:
    p_values = values.to_numpy()
    order = np.argsort(p_values)
    adjusted = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    return restored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    standard_dir = root / "results" / "pseudobulk"
    robust_dir = root / "results" / "bam_robustness"
    robust_pb = robust_dir / "pseudobulk"

    tables = []
    for gse in ["GSE138266", "GSE194078", "GSE286068"]:
        cdc2 = load_module_scores(
            standard_dir / f"{gse}_CSF_cDC2_counts.csv.gz",
            standard_dir / f"{gse}_CSF_pseudobulk_sample_metadata.csv",
            "cDC2",
            args.min_cells,
        )
        bam = load_module_scores(
            robust_pb / f"{gse}_CSF_BAM_stable_core_counts.csv.gz",
            robust_pb / f"{gse}_CSF_BAM_stable_core_sample_metadata.csv",
            "BAM_stable_core",
            args.min_cells,
        )
        bam = bam.drop(columns=["subject_id", "diagnosis", "disease_group"], errors="ignore")
        table = cdc2.merge(bam, on="sample_accession", how="inner")
        table.insert(0, "gse", gse)
        tables.append(table)
    scores = pd.concat(tables, ignore_index=True)
    scores.to_csv(robust_dir / "BAM_stable_core_relay_scores.csv", index=False)

    pairs = [
        ("cDC2_MHC_I_processing", "BAM_stable_core_MHC_I_processing"),
        ("cDC2_IFN_antiviral", "BAM_stable_core_MHC_I_processing"),
    ]
    rows = []
    for pair_index, (x_col, y_col) in enumerate(pairs):
        for gse, frame in scores.groupby("gse", observed=True):
            pair = frame[[x_col, y_col]].dropna()
            if len(pair) < 4:
                continue
            test = spearmanr(pair[x_col], pair[y_col])
            rows.append(
                {
                    "scope": "within_dataset",
                    "gse": gse,
                    "x": x_col,
                    "y": y_col,
                    "n_samples": len(pair),
                    "correlation": test.statistic,
                    "p_value": test.pvalue,
                    "method": "spearman",
                }
            )
        rows.append(
            pooled_test(
                scores,
                x_col,
                y_col,
                args.permutations,
                seed=20260810 + pair_index,
            )
        )
    results = pd.DataFrame(rows)
    results["fdr_bh"] = np.nan
    mask = results["scope"].eq("pooled_group_residualized")
    results.loc[mask, "fdr_bh"] = bh_adjust(results.loc[mask, "p_value"])
    results.to_csv(robust_dir / "BAM_stable_core_relay_correlations.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
