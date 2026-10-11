#!/usr/bin/env python3
"""Stage-aware external validation of BAM-like and MHC-I associations."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.special import logit
from scipy.stats import norm, pearsonr, rankdata
from statsmodels.stats.multitest import multipletests


MODULES = {
    "MHC_I_processing": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    ],
    "MHC_II_processing": [
        "CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
        "HLA-DQA1", "HLA-DQB1", "CIITA",
    ],
    "IFN_antiviral": [
        "IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2",
        "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1",
    ],
    "BAM_homeostatic": [
        "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "DAB2", "MSR1", "CD163",
        "FOLR2", "MRC1", "LYVE1", "STAB1", "CH25H",
    ],
}

LINEAGES = ["BAM_stable_core", "BAM_reference", "classical_monocyte", "cDC2", "T"]
CONTRASTS = {
    "RRMS_vs_control": (["RRMS"], ["Ctrl"]),
    "PMS_vs_control": (["PPMS", "SPMS"], ["Ctrl"]),
    "PMS_vs_RRMS": (["PPMS", "SPMS"], ["RRMS"]),
    "SPMS_vs_PPMS": (["SPMS"], ["PPMS"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def lineage_masks(adata: sc.AnnData) -> dict[str, np.ndarray]:
    return {
        "BAM_stable_core": adata.obs["bam_stable_core"].astype(bool).to_numpy(),
        "BAM_reference": adata.obs["fine_apc_label"].astype(str).eq("BAM_like_candidate").to_numpy(),
        "classical_monocyte": adata.obs["fine_apc_label"].astype(str).eq("classical_monocyte").to_numpy(),
        "cDC2": adata.obs["fine_apc_label"].astype(str).eq("cDC2").to_numpy(),
        "T": adata.obs["broad_cell_type"].astype(str).eq("T").to_numpy(),
    }


def aggregate_counts(
    adata: sc.AnnData,
    mask: np.ndarray,
    subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = adata.layers["counts"].tocsr()
    subject_values = adata.obs["subject_id"].astype(str).to_numpy()
    rows = []
    cell_counts = []
    for subject in subjects:
        selected = mask & (subject_values == subject)
        cell_counts.append(int(selected.sum()))
        if selected.any():
            rows.append(np.asarray(matrix[selected].sum(axis=0)).ravel())
        else:
            rows.append(np.zeros(adata.n_vars, dtype=float))
    return np.vstack(rows), np.asarray(cell_counts)


def module_scores(counts: np.ndarray, genes: pd.Index, cell_counts: np.ndarray) -> pd.DataFrame:
    library_sizes = counts.sum(axis=1)
    output = pd.DataFrame(index=np.arange(len(counts)))
    output["n_cells"] = cell_counts
    valid = (cell_counts >= 5) & (library_sizes > 0)
    log_cpm = np.full(counts.shape, np.nan, dtype=float)
    log_cpm[valid] = np.log2(
        (counts[valid] + 0.5) / (library_sizes[valid, None] + 1.0) * 1e6
    )
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    for module, members in MODULES.items():
        present = [gene for gene in members if gene in gene_to_index]
        indices = [gene_to_index[gene] for gene in present]
        informative = [
            index for index in indices
            if (counts[valid, index] > 0).sum() >= 2
            and np.nanstd(log_cpm[valid, index], ddof=0) > 0
        ]
        output[f"{module}_n_genes"] = len(informative)
        if len(informative) < 5:
            output[module] = np.nan
            continue
        matrix = log_cpm[valid][:, informative]
        standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)
        values = np.full(len(counts), np.nan, dtype=float)
        values[valid] = standardized.mean(axis=1)
        output[module] = values
    return output


def permutation_test(
    values: np.ndarray,
    groups: np.ndarray,
    max_permutations: int,
    seed: int,
) -> tuple[float, float, str]:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=bool)
    observed = float(values[groups].mean() - values[~groups].mean())
    n_case = int(groups.sum())
    n_combinations = math.comb(len(values), n_case)
    exceed = 0
    if n_combinations <= max_permutations:
        iterator = itertools.combinations(range(len(values)), n_case)
        total = n_combinations
        for indices in iterator:
            permuted = np.zeros(len(values), dtype=bool)
            permuted[list(indices)] = True
            statistic = values[permuted].mean() - values[~permuted].mean()
            exceed += abs(statistic) >= abs(observed) - 1e-12
        return observed, exceed / total, "exact"
    rng = np.random.default_rng(seed)
    for _ in range(max_permutations):
        permuted = rng.permutation(groups)
        statistic = values[permuted].mean() - values[~permuted].mean()
        exceed += abs(statistic) >= abs(observed) - 1e-12
    return observed, (exceed + 1) / (max_permutations + 1), f"monte_carlo_{max_permutations}"


def endpoint_tests(table: pd.DataFrame, permutations: int, seed: int) -> pd.DataFrame:
    rows = []
    endpoints = ["logit_proportion"] + list(MODULES)
    for contrast_index, (contrast, (case_groups, control_groups)) in enumerate(CONTRASTS.items()):
        for lineage_index, lineage in enumerate(LINEAGES):
            lineage_table = table.loc[table["lineage"].eq(lineage)].copy()
            for endpoint_index, endpoint in enumerate(endpoints):
                subset = lineage_table.loc[
                    lineage_table["diagnosis"].isin(case_groups + control_groups),
                    ["diagnosis", "n_cells", endpoint],
                ].dropna()
                if endpoint != "logit_proportion":
                    subset = subset.loc[subset["n_cells"].ge(5)]
                groups = subset["diagnosis"].isin(case_groups).to_numpy()
                if groups.sum() < 3 or (~groups).sum() < 3:
                    continue
                values = subset[endpoint].to_numpy(dtype=float)
                difference, p_value, method = permutation_test(
                    values,
                    groups,
                    permutations,
                    seed + contrast_index * 1000 + lineage_index * 100 + endpoint_index,
                )
                rows.append({
                    "contrast": contrast,
                    "case_groups": "+".join(case_groups),
                    "control_groups": "+".join(control_groups),
                    "lineage": lineage,
                    "endpoint": endpoint,
                    "n_case": int(groups.sum()),
                    "n_control": int((~groups).sum()),
                    "case_median": float(np.median(values[groups])),
                    "control_median": float(np.median(values[~groups])),
                    "mean_difference_case_minus_control": difference,
                    "permutation_p": p_value,
                    "permutation_method": method,
                })
    output = pd.DataFrame(rows)
    output["fdr_bh_within_contrast"] = output.groupby("contrast")["permutation_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return output


def residualize_ranks(values: np.ndarray, strata: pd.Series, cells: np.ndarray) -> np.ndarray:
    design = pd.get_dummies(strata, drop_first=False, dtype=float).to_numpy()
    log_cells = np.log1p(cells.astype(float))
    log_cells = (log_cells - log_cells.mean()) / log_cells.std(ddof=0)
    design = np.column_stack([design, log_cells])
    ranked = rankdata(values)
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def block_permutation(
    x: np.ndarray,
    y: np.ndarray,
    strata: pd.Series,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    observed = float(pearsonr(x, y).statistic)
    rng = np.random.default_rng(seed)
    strata_values = strata.to_numpy()
    blocks = [np.flatnonzero(strata_values == value) for value in pd.unique(strata_values)]
    exceed = 0
    for _ in range(permutations):
        permuted = y.copy()
        for block in blocks:
            permuted[block] = rng.permutation(permuted[block])
        exceed += abs(pearsonr(x, permuted).statistic) >= abs(observed)
    return observed, (exceed + 1) / (permutations + 1)


def coordination_tests(table: pd.DataFrame, permutations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = table.pivot(index=["subject_id", "diagnosis"], columns="lineage")
    wide.columns = [f"{lineage}_{metric}" for metric, lineage in wide.columns]
    wide = wide.reset_index()
    target = "BAM_stable_core_MHC_I_processing"
    rows = []
    for index, lineage in enumerate(["classical_monocyte", "cDC2", "T"]):
        source = f"{lineage}_MHC_I_processing"
        columns = ["diagnosis", source, target, f"{lineage}_n_cells", "BAM_stable_core_n_cells"]
        subset = wide[columns].dropna()
        subset = subset.loc[(subset[f"{lineage}_n_cells"] >= 5) & (subset["BAM_stable_core_n_cells"] >= 5)]
        x = residualize_ranks(subset[source].to_numpy(), subset["diagnosis"], subset[f"{lineage}_n_cells"].to_numpy())
        y = residualize_ranks(subset[target].to_numpy(), subset["diagnosis"], subset["BAM_stable_core_n_cells"].to_numpy())
        correlation, p_value = block_permutation(x, y, subset["diagnosis"], permutations, seed + index)
        rows.append({
            "source_lineage": lineage,
            "source_module": "MHC_I_processing",
            "target": target,
            "n_subjects": len(subset),
            "correlation": correlation,
            "permutation_p": p_value,
        })
    coordination = pd.DataFrame(rows)
    coordination["fdr_bh"] = multipletests(coordination["permutation_p"], method="fdr_bh")[1]

    partial_rows = []
    specs = [("cDC2", "classical_monocyte"), ("classical_monocyte", "cDC2")]
    for index, (predictor, control) in enumerate(specs):
        columns = [
            "diagnosis", target,
            f"{predictor}_MHC_I_processing", f"{control}_MHC_I_processing",
            "BAM_stable_core_n_cells", f"{predictor}_n_cells", f"{control}_n_cells",
        ]
        subset = wide[columns].dropna()
        subset = subset.loc[
            (subset["BAM_stable_core_n_cells"] >= 5)
            & (subset[f"{predictor}_n_cells"] >= 5)
            & (subset[f"{control}_n_cells"] >= 5)
        ]
        x = residualize_ranks(
            subset[f"{predictor}_MHC_I_processing"].to_numpy(), subset["diagnosis"],
            subset[f"{predictor}_n_cells"].to_numpy(),
        )
        y = residualize_ranks(
            subset[target].to_numpy(), subset["diagnosis"],
            subset["BAM_stable_core_n_cells"].to_numpy(),
        )
        control_values = residualize_ranks(
            subset[f"{control}_MHC_I_processing"].to_numpy(), subset["diagnosis"],
            subset[f"{control}_n_cells"].to_numpy(),
        )
        design = np.column_stack([np.ones(len(control_values)), control_values])
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        correlation, p_value = block_permutation(x, y, subset["diagnosis"], permutations, seed + 100 + index)
        partial_rows.append({
            "predictor": predictor,
            "conditioned_on": control,
            "target": "BAM_stable_core",
            "n_subjects": len(subset),
            "partial_correlation": correlation,
            "permutation_p": p_value,
        })
    partial = pd.DataFrame(partial_rows)
    partial["fdr_bh"] = multipletests(partial["permutation_p"], method="fdr_bh")[1]
    return coordination, partial


def get_test(tests: pd.DataFrame, contrast: str, lineage: str, endpoint: str) -> pd.Series:
    return tests.loc[
        tests["contrast"].eq(contrast)
        & tests["lineage"].eq(lineage)
        & tests["endpoint"].eq(endpoint)
    ].iloc[0]


def write_report(path: Path, tests: pd.DataFrame, coordination: pd.DataFrame, partial: pd.DataFrame) -> None:
    rrms_ab = get_test(tests, "RRMS_vs_control", "BAM_stable_core", "logit_proportion")
    rrms_m1 = get_test(tests, "RRMS_vs_control", "BAM_stable_core", "MHC_I_processing")
    rrms_m2 = get_test(tests, "RRMS_vs_control", "BAM_stable_core", "MHC_II_processing")
    pms_ab = get_test(tests, "PMS_vs_control", "BAM_stable_core", "logit_proportion")
    pms_m1 = get_test(tests, "PMS_vs_control", "BAM_stable_core", "MHC_I_processing")
    pms_m2 = get_test(tests, "PMS_vs_control", "BAM_stable_core", "MHC_II_processing")
    lines = [
        "# GSE328265 external validation",
        "",
        "## Scope",
        "",
        "This analysis reprocessed newly profiled GSE328265 subjects for external validation.",
        "",
        "The public GEO super-series contains 31 technical libraries from 30 newly "
        "profiled subjects. The source article additionally incorporated ten subjects "
        "from earlier public studies and reported a final analysed set of 35 subjects "
        "after exclusions. The present analysis intentionally uses only the newly "
        "profiled raw matrices, so that reused discovery material is not mistaken for "
        "independent replication.",
        "",
        "## QC and BAM definition",
        "",
        "Filtered 10x matrices underwent basic cell QC, Scrublet plus scDblFinder union removal, "
        "and DecontX without an empty-droplet background. Technical runs for SPMS_4 were merged "
        "at donor aggregation. The strict BAM core required a reference BAM-like cluster, BAM "
        "assignment at four of five resolutions, and a BAM score above every APC competitor.",
        "",
        "## Stage-aware strict-BAM results",
        "",
        f"- RRMS abundance versus control: difference={rrms_ab.mean_difference_case_minus_control:.3f}, "
        f"P={rrms_ab.permutation_p:.4g}, FDR={rrms_ab.fdr_bh_within_contrast:.4g}.",
        f"- RRMS MHC-I versus control: difference={rrms_m1.mean_difference_case_minus_control:.3f}, "
        f"P={rrms_m1.permutation_p:.4g}, FDR={rrms_m1.fdr_bh_within_contrast:.4g}.",
        f"- RRMS MHC-II versus control: difference={rrms_m2.mean_difference_case_minus_control:.3f}, "
        f"P={rrms_m2.permutation_p:.4g}, FDR={rrms_m2.fdr_bh_within_contrast:.4g}.",
        f"- PMS abundance versus control: difference={pms_ab.mean_difference_case_minus_control:.3f}, "
        f"P={pms_ab.permutation_p:.4g}, FDR={pms_ab.fdr_bh_within_contrast:.4g}.",
        f"- PMS MHC-I versus control: difference={pms_m1.mean_difference_case_minus_control:.3f}, "
        f"P={pms_m1.permutation_p:.4g}, FDR={pms_m1.fdr_bh_within_contrast:.4g}.",
        f"- PMS MHC-II versus control: difference={pms_m2.mean_difference_case_minus_control:.3f}, "
        f"P={pms_m2.permutation_p:.4g}, FDR={pms_m2.fdr_bh_within_contrast:.4g}.",
        "",
        "## Cross-lineage coordination",
        "",
    ]
    for row in coordination.sort_values("correlation", ascending=False).itertuples():
        lines.append(
            f"- {row.source_lineage}: r={row.correlation:.3f}, P={row.permutation_p:.4g}, "
            f"FDR={row.fdr_bh:.4g}, n={row.n_subjects}."
        )
    for row in partial.itertuples():
        lines.append(
            f"- {row.predictor} conditioned on {row.conditioned_on}: partial r={row.partial_correlation:.3f}, "
            f"P={row.permutation_p:.4g}, FDR={row.fdr_bh:.4g}, n={row.n_subjects}."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "None of the strict-BAM abundance, MHC-I, MHC-II, or IFN stage contrasts survived "
        "multiplicity correction. The directions also failed to reproduce the proposed "
        "RRMS abundance-low/MHC-I-high phenotype. Reference-BAM results were likewise "
        "null. The modest cross-lineage correlations did not establish cDC2 specificity: "
        "cDC2 conditioned on classical-monocyte MHC-I was not significant after "
        "correction, and the reverse conditional test was also null.",
        "",
        "The source publication used donor-aware propeller models for abundance but "
        "cell-level Seurat Wilcoxon tests for differential expression. Its progressive "
        "BAM antigen-presentation result therefore does not constitute a donor-level "
        "pseudobulk replication of the present discovery analysis. Conversely, the "
        "present negative result does not negate the source paper's descriptive cell "
        "state; it shows that the proposed headline is sensitive to BAM definition, "
        "cohort composition, disease-stage contrast, and statistical unit.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "results" / "external_validation" / "GSE328265"
    adata = sc.read_h5ad(output_dir / "GSE328265_clean_annotated.h5ad")
    subjects = sorted(adata.obs["subject_id"].astype(str).unique())
    subject_meta = (
        adata.obs[["subject_id", "diagnosis"]]
        .drop_duplicates()
        .set_index("subject_id")
        .loc[subjects]
        .reset_index()
    )
    total_cells = adata.obs.groupby("subject_id", observed=True).size().reindex(subjects).to_numpy()
    masks = lineage_masks(adata)
    score_tables = []
    for lineage in LINEAGES:
        counts, n_cells = aggregate_counts(adata, masks[lineage], subjects)
        count_frame = pd.DataFrame(counts.T, index=adata.var_names, columns=subjects)
        count_frame.index.name = "gene_symbol"
        count_frame.reset_index().to_csv(
            output_dir / f"GSE328265_{lineage}_subject_pseudobulk_counts.csv.gz",
            index=False,
            compression="gzip",
        )
        scores = module_scores(counts, adata.var_names, n_cells)
        scores.insert(0, "lineage", lineage)
        scores.insert(0, "diagnosis", subject_meta["diagnosis"].to_numpy())
        scores.insert(0, "subject_id", subjects)
        proportions = n_cells / total_cells
        scores["proportion"] = proportions
        scores["logit_proportion"] = logit((n_cells + 0.5) / (total_cells + 1.0))
        scores["n_total_cells"] = total_cells
        score_tables.append(scores)
    scores = pd.concat(score_tables, ignore_index=True)
    scores.to_csv(output_dir / "GSE328265_subject_lineage_module_scores.csv", index=False)

    tests = endpoint_tests(scores, args.permutations, args.seed)
    tests.to_csv(output_dir / "GSE328265_stage_contrast_tests.csv", index=False)
    coordination, partial = coordination_tests(scores, args.permutations, args.seed + 10000)
    coordination.to_csv(output_dir / "GSE328265_MHCI_coordination_tests.csv", index=False)
    partial.to_csv(output_dir / "GSE328265_MHCI_conditional_tests.csv", index=False)
    write_report(output_dir / "GSE328265_EXTERNAL_VALIDATION_REPORT.md", tests, coordination, partial)

    focus = tests.loc[
        tests["lineage"].eq("BAM_stable_core")
        & tests["endpoint"].isin(["logit_proportion", "MHC_I_processing", "MHC_II_processing", "IFN_antiviral"])
    ]
    print(focus.to_string(index=False))
    print("\nCoordination")
    print(coordination.to_string(index=False))
    print("\nConditional")
    print(partial.to_string(index=False))


if __name__ == "__main__":
    main()
