#!/usr/bin/env python3
"""Test whether the stable-BAM MHC-I relay is specific to cDC2."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests


MODULES = {
    "MHC_I_processing": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    ],
    "IFN_antiviral": [
        "IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2",
        "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1",
    ],
}

SOURCE_LABELS = ["cDC2", "classical_monocyte", "T", "B", "pDC", "cDC1"]
DATASETS = ["GSE138266", "GSE194078", "GSE286068"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def load_module_scores(
    counts_path: Path,
    metadata_path: Path,
    label: str,
    min_cells: int,
) -> pd.DataFrame:
    counts = pd.read_csv(counts_path)
    metadata = pd.read_csv(metadata_path)
    if "analysis_label" in metadata.columns:
        metadata = metadata.loc[metadata["analysis_label"].eq(label)].copy()
    metadata = metadata.drop_duplicates("sample_accession")
    samples = metadata.loc[metadata["n_cells"].ge(min_cells), "sample_accession"].tolist()
    samples = [sample for sample in samples if sample in counts.columns]
    if len(samples) < 3:
        return pd.DataFrame(columns=["sample_accession"])
    gene_counts = counts.loc[counts["gene_symbol"].notna(), ["gene_symbol"] + samples]
    gene_counts = gene_counts.groupby("gene_symbol", observed=True)[samples].sum()
    library_sizes = counts[samples].sum(axis=0).astype(float)
    log_cpm = np.log2((gene_counts.T + 0.5).div(library_sizes + 1.0, axis=0) * 1e6)
    output = metadata.set_index("sample_accession").loc[
        samples, ["subject_id", "diagnosis", "disease_group", "n_cells"]
    ].reset_index()
    output = output.rename(columns={"n_cells": f"{label}_n_cells"})
    for module_name, genes in MODULES.items():
        present = [
            gene for gene in genes
            if gene in log_cpm.columns and (gene_counts.loc[gene] > 0).sum() >= 2
        ]
        matrix = log_cpm[present]
        standard_deviations = matrix.std(axis=0, ddof=0)
        informative = standard_deviations.index[standard_deviations.gt(0)]
        output[f"{label}_{module_name}_n_genes"] = len(informative)
        if len(informative) < 5:
            output[f"{label}_{module_name}"] = np.nan
            continue
        standardized = (
            matrix[informative] - matrix[informative].mean(axis=0)
        ) / standard_deviations[informative]
        output[f"{label}_{module_name}"] = standardized.mean(axis=1).to_numpy()
    return output


def design_matrix(strata: pd.Series, covariates: list[np.ndarray] | None = None) -> np.ndarray:
    design = pd.get_dummies(strata, drop_first=False, dtype=float).to_numpy()
    for covariate in covariates or []:
        values = np.asarray(covariate, dtype=float)
        values = (values - values.mean()) / values.std(ddof=0)
        design = np.column_stack([design, values])
    return design


def residualize_ranks(
    values: np.ndarray,
    strata: pd.Series,
    covariates: list[np.ndarray] | None = None,
) -> np.ndarray:
    ranked = rankdata(values, method="average")
    design = design_matrix(strata, covariates)
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def block_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    strata: pd.Series,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    observed = float(pearsonr(x, y).statistic)
    rng = np.random.default_rng(seed)
    strata_values = strata.to_numpy()
    groups = [np.flatnonzero(strata_values == key) for key in pd.unique(strata_values)]
    exceed = 0
    for _ in range(permutations):
        permuted = y.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        exceed += abs(pearsonr(x, permuted).statistic) >= abs(observed)
    return observed, (exceed + 1) / (permutations + 1)


def pooled_test(
    frame: pd.DataFrame,
    source_label: str,
    source_module: str,
    permutations: int,
    seed: int,
) -> dict | None:
    x_col = f"{source_label}_{source_module}"
    source_cells = f"{source_label}_n_cells"
    y_col = "BAM_stable_core_MHC_I_processing"
    columns = [
        "gse", "disease_group", x_col, y_col, source_cells,
        "BAM_stable_core_n_cells",
    ]
    subset = frame[columns].dropna().copy()
    if len(subset) < 8:
        return None
    subset["stratum"] = subset["gse"].astype(str) + ":" + subset["disease_group"].astype(str)
    x_residual = residualize_ranks(
        subset[x_col].to_numpy(),
        subset["stratum"],
        [np.log1p(subset[source_cells].to_numpy())],
    )
    y_residual = residualize_ranks(
        subset[y_col].to_numpy(),
        subset["stratum"],
        [np.log1p(subset["BAM_stable_core_n_cells"].to_numpy())],
    )
    correlation, p_value = block_permutation_p(
        x_residual, y_residual, subset["stratum"], permutations, seed
    )
    return {
        "source_label": source_label,
        "source_module": source_module,
        "target": y_col,
        "n_samples": len(subset),
        "correlation": correlation,
        "permutation_p": p_value,
        "adjustment": "dataset:disease strata + source/BAM log cell counts",
    }


def partial_test(
    table: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_col: str,
    x_cells: str,
    y_cells: str,
    control_cells: str,
    permutations: int,
    seed: int,
) -> dict:
    columns = [
        "gse", "disease_group", x_col, y_col, control_col,
        x_cells, y_cells, control_cells,
    ]
    frame = table[columns].dropna().copy()
    frame["stratum"] = frame["gse"].astype(str) + ":" + frame["disease_group"].astype(str)
    x = residualize_ranks(
        frame[x_col].to_numpy(), frame["stratum"], [np.log1p(frame[x_cells])]
    )
    y = residualize_ranks(
        frame[y_col].to_numpy(), frame["stratum"], [np.log1p(frame[y_cells])]
    )
    control = residualize_ranks(
        frame[control_col].to_numpy(),
        frame["stratum"],
        [np.log1p(frame[control_cells])],
    )
    control_design = np.column_stack([np.ones(len(control)), control])
    x_partial = x - control_design @ np.linalg.lstsq(control_design, x, rcond=None)[0]
    y_partial = y - control_design @ np.linalg.lstsq(control_design, y, rcond=None)[0]
    correlation, p_value = block_permutation_p(
        x_partial, y_partial, frame["stratum"], permutations, seed
    )
    return {
        "predictor": x_col,
        "target": y_col,
        "conditioned_on": control_col,
        "n_samples": len(frame),
        "partial_correlation": correlation,
        "permutation_p": p_value,
        "adjustment": "dataset:disease strata + lineage-specific log cell counts",
    }


def write_report(
    path: Path,
    pooled: pd.DataFrame,
    partial: pd.DataFrame,
) -> None:
    ranked_mhc = pooled.loc[pooled["source_module"].eq("MHC_I_processing")].sort_values(
        "correlation", ascending=False
    )
    ranked_ifn = pooled.loc[pooled["source_module"].eq("IFN_antiviral")].sort_values(
        "correlation", ascending=False
    )
    cdc2_partial = partial.loc[
        partial["predictor"].eq("cDC2_MHC_I_processing")
        & partial["conditioned_on"].eq("classical_monocyte_MHC_I_processing")
    ].iloc[0]
    mono_partial = partial.loc[
        partial["predictor"].eq("classical_monocyte_MHC_I_processing")
        & partial["conditioned_on"].eq("cDC2_MHC_I_processing")
    ].iloc[0]
    lines = [
        "# Stable-BAM relay cell-type specificity audit",
        "",
        "## Question",
        "",
        "Does the donor-level association with strict-core BAM MHC-I processing "
        "specifically identify cDC2, or does it reflect a broader CSF immune state?",
        "",
        "## Pooled cell-count-adjusted results",
        "",
        "MHC-I source rankings:",
        "",
    ]
    for row in ranked_mhc.itertuples():
        lines.append(
            f"- {row.source_label}: r={row.correlation:.3f}, permutation "
            f"P={row.permutation_p:.4g}, FDR={row.fdr_bh:.4g}, n={row.n_samples}."
        )
    lines.extend(["", "IFN source rankings:", ""])
    for row in ranked_ifn.itertuples():
        lines.append(
            f"- {row.source_label}: r={row.correlation:.3f}, permutation "
            f"P={row.permutation_p:.4g}, FDR={row.fdr_bh:.4g}, n={row.n_samples}."
        )
    lines.extend([
        "",
        "## Conditional test",
        "",
        f"- cDC2 MHC-I versus BAM MHC-I after conditioning on classical-monocyte "
        f"MHC-I: partial r={cdc2_partial.partial_correlation:.3f}, permutation "
        f"P={cdc2_partial.permutation_p:.4g}, FDR={cdc2_partial.fdr_bh:.4g}.",
        f"- Classical-monocyte MHC-I versus BAM MHC-I after conditioning on cDC2 "
        f"MHC-I: partial r={mono_partial.partial_correlation:.3f}, permutation "
        f"P={mono_partial.permutation_p:.4g}, FDR={mono_partial.fdr_bh:.4g}.",
        "",
        "## Verdict",
        "",
        "The relay is not cDC2-specific. The defensible signal is a coordinated "
        "CSF myeloid/immune MHC-I state, with cDC2 IFN activity as one component. "
        "Use association/coordination language and do not use 'cDC2 licensing of BAM' "
        "without perturbational or spatial evidence.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    standard_dir = root / "results" / "pseudobulk"
    robust_pb_dir = root / "results" / "bam_robustness" / "pseudobulk"
    output_dir = root / "results" / "relay_specificity"
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    within_rows = []
    for gse in DATASETS:
        bam = load_module_scores(
            robust_pb_dir / f"{gse}_CSF_BAM_stable_core_counts.csv.gz",
            robust_pb_dir / f"{gse}_CSF_BAM_stable_core_sample_metadata.csv",
            "BAM_stable_core",
            args.min_cells,
        )
        table = bam.copy()
        for source_label in SOURCE_LABELS:
            source = load_module_scores(
                standard_dir / f"{gse}_CSF_{source_label}_counts.csv.gz",
                standard_dir / f"{gse}_CSF_pseudobulk_sample_metadata.csv",
                source_label,
                args.min_cells,
            )
            if source.empty:
                continue
            for source_module in MODULES:
                pair = bam[["sample_accession", "BAM_stable_core_MHC_I_processing"]].merge(
                    source[["sample_accession", f"{source_label}_{source_module}"]],
                    on="sample_accession",
                    how="inner",
                ).dropna()
                if len(pair) >= 4:
                    test = spearmanr(
                        pair[f"{source_label}_{source_module}"],
                        pair["BAM_stable_core_MHC_I_processing"],
                    )
                    within_rows.append({
                        "gse": gse,
                        "source_label": source_label,
                        "source_module": source_module,
                        "n_samples": len(pair),
                        "spearman_r": test.statistic,
                        "p_value": test.pvalue,
                    })
            source = source.drop(
                columns=["subject_id", "diagnosis", "disease_group"], errors="ignore"
            )
            table = table.merge(source, on="sample_accession", how="outer")
        table.insert(0, "gse", gse)
        tables.append(table)
    scores = pd.concat(tables, ignore_index=True)
    scores.to_csv(output_dir / "CSF_stable_BAM_celltype_module_scores.csv", index=False)
    pd.DataFrame(within_rows).to_csv(
        output_dir / "CSF_stable_BAM_celltype_within_dataset_correlations.csv", index=False
    )

    pooled_rows = []
    for label_index, source_label in enumerate(SOURCE_LABELS):
        for module_index, source_module in enumerate(MODULES):
            result = pooled_test(
                scores,
                source_label,
                source_module,
                args.permutations,
                args.seed + label_index * 10 + module_index,
            )
            if result is not None:
                pooled_rows.append(result)
    pooled = pd.DataFrame(pooled_rows)
    pooled["fdr_bh"] = multipletests(pooled["permutation_p"], method="fdr_bh")[1]
    pooled.to_csv(output_dir / "CSF_stable_BAM_celltype_specificity_tests.csv", index=False)

    y_col = "BAM_stable_core_MHC_I_processing"
    y_cells = "BAM_stable_core_n_cells"
    partial_specs = [
        (
            "cDC2_MHC_I_processing", "classical_monocyte_MHC_I_processing",
            "cDC2_n_cells", "classical_monocyte_n_cells",
        ),
        (
            "classical_monocyte_MHC_I_processing", "cDC2_MHC_I_processing",
            "classical_monocyte_n_cells", "cDC2_n_cells",
        ),
        (
            "cDC2_IFN_antiviral", "classical_monocyte_IFN_antiviral",
            "cDC2_n_cells", "classical_monocyte_n_cells",
        ),
        (
            "classical_monocyte_IFN_antiviral", "cDC2_IFN_antiviral",
            "classical_monocyte_n_cells", "cDC2_n_cells",
        ),
        (
            "cDC2_IFN_antiviral", "classical_monocyte_MHC_I_processing",
            "cDC2_n_cells", "classical_monocyte_n_cells",
        ),
    ]
    partial_rows = []
    for test_index, (predictor, control, predictor_cells, control_cells) in enumerate(partial_specs):
        partial_rows.append(partial_test(
            scores,
            predictor,
            y_col,
            control,
            predictor_cells,
            y_cells,
            control_cells,
            args.permutations,
            args.seed + 100 + test_index,
        ))
    partial = pd.DataFrame(partial_rows)
    partial["fdr_bh"] = multipletests(partial["permutation_p"], method="fdr_bh")[1]
    partial.to_csv(output_dir / "CSF_stable_BAM_conditional_relay_tests.csv", index=False)
    write_report(output_dir / "RELAY_CELLTYPE_SPECIFICITY_REPORT.md", pooled, partial)
    print(pooled.sort_values(["source_module", "correlation"], ascending=[True, False]).to_string(index=False))
    print("\nConditional tests")
    print(partial.to_string(index=False))


if __name__ == "__main__":
    main()
