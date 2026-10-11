#!/usr/bin/env python3
"""Reverse-translate author-defined meningeal LAM programs into human CSF."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import numpy as np
import pandas as pd
import patsy
import scanpy as sc
import statsmodels.api as sm
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


SEED = 12345
MIN_DONOR_CELLS = 20
INCOMPLETE_CUTOFF = 0.80
SOURCE_DOI = "10.1101/2025.08.15.670462"

AXIS2_GENES = {
    "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7",
}

SIGNATURES = {
    "author_human_LAM": {
        "source_context": "Human Xenium LAM signature stated in Methods",
        "mapping": "direct_human_symbol",
        "genes": [
            "ANXA1", "APOE", "APP", "CAPG", "CD14", "CD68", "CXCL14", "CXCR4",
            "FCER1G", "GPNMB", "HILPDA", "IDH1", "IL7R", "ITGB2", "MGST1", "PSEN2",
            "RFTN1", "TGFBI", "THBS1",
        ],
    },
    "mouse_LAM2_complement_APC": {
        "source_context": "Mouse LAM cluster 2 markers and trajectory genes stated in Results",
        "mapping": "curated_one_to_one_or_functional_MHCII_ortholog",
        "genes": [
            "APOE", "C1QA", "C1QB", "C1QC", "GPNMB", "SELENOP", "TREM2", "MERTK",
            "HLA-DRA", "CD74", "CIITA",
        ],
    },
    "mouse_LAM4_fibrotic": {
        "source_context": "Mouse LAM cluster 4 markers, wound-healing genes, and inferred drivers",
        "mapping": "curated_one_to_one_human_ortholog",
        "genes": [
            "ARG1", "CXCL2", "CLEC7A", "SPP1", "CCL24", "NOS2", "FN1", "TGFBI",
            "VEGFA", "ZEB2", "HIF1A",
        ],
    },
    "mouse_GMCSF_LAM_precursor": {
        "source_context": "Mouse cluster 0 LAM precursor genes stated in Results",
        "mapping": "curated_one_to_one_human_ortholog",
        "genes": ["CXCL10", "CXCL9", "TGM2", "TGFB3", "PLAC8", "IRF1", "CSF2RB"],
    },
}

DATASETS = {
    "GSE138266": {
        "path": "results/trajectory/continuum_objects/GSE138266_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "control": "IIH_control",
        "external": False,
    },
    "GSE194078": {
        "path": "results/trajectory/continuum_objects/GSE194078_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": None,
        "control": None,
        "external": False,
    },
    "GSE286068": {
        "path": "results/trajectory/continuum_objects/GSE286068_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "control": "non_inflammatory_control",
        "external": False,
    },
    "GSE328265": {
        "path": "results/external_validation/GSE328265_two_axis_projection/GSE328265_monocyte_BAM_two_axis.h5ad",
        "pt": "projected_differentiation",
        "activation": "external_activation_residual",
        "control": "control",
        "external": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def weighted_mean_sd(matrix: np.ndarray, subjects: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = subjects.value_counts()
    weights = subjects.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float, copy=True)
    weights /= weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(np.square(matrix - mean) * weights[:, None], axis=0)
    sd = np.sqrt(variance)
    sd[sd < 1e-8] = 1.0
    return mean, sd


def weighted_median(values: np.ndarray, subjects: pd.Series) -> float:
    values = np.asarray(values, dtype=float)
    counts = subjects.value_counts()
    weights = subjects.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float, copy=True)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(0.5, cumulative, values[order]))


def signature_ledger(root: Path) -> pd.DataFrame:
    dataset_genes = {}
    for dataset, config in DATASETS.items():
        backed = sc.read_h5ad(root / config["path"], backed="r")
        dataset_genes[dataset] = set(backed.var_names.astype(str))
        backed.file.close()
    rows = []
    for signature, details in SIGNATURES.items():
        for gene in details["genes"]:
            rows.append({
                "signature": signature,
                "human_gene": gene,
                "source_context": details["source_context"],
                "mapping_strategy": details["mapping"],
                "source_doi": SOURCE_DOI,
                "source_version": "bioRxiv v1, posted 2025-08-21",
                "source_raw_scrna_status": "not_public_as_of_analysis; promised_upon_publication",
                "used_to_define_axis2_activation": gene in AXIS2_GENES,
                "available_in_all_human_datasets": all(
                    gene in genes for genes in dataset_genes.values()
                ),
            })
    return pd.DataFrame(rows)


def add_signature_scores(adata: sc.AnnData) -> pd.DataFrame:
    genes = sorted({gene for details in SIGNATURES.values() for gene in details["genes"] if gene in adata.var_names})
    matrix = dense(adata[:, genes].X).astype(np.float64)
    mean, sd = weighted_mean_sd(matrix, adata.obs["subject_id"].astype(str))
    standardized = (matrix - mean) / sd
    lookup = {gene: index for index, gene in enumerate(genes)}
    availability = []
    for signature, details in SIGNATURES.items():
        available = [gene for gene in details["genes"] if gene in lookup]
        missing = [gene for gene in details["genes"] if gene not in lookup]
        score = standardized[:, [lookup[gene] for gene in available]].mean(axis=1)
        adata.obs[f"LAM_score_{signature}"] = score
        availability.append({
            "signature": signature,
            "n_requested": len(details["genes"]),
            "n_available": len(available),
            "available_genes": ";".join(available),
            "missing_genes": ";".join(missing),
        })
    return pd.DataFrame(availability)


def residualize_signature(
    adata: sc.AnnData,
    score_column: str,
    pt_column: str,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    frame = pd.DataFrame({
        "pt": adata.obs[pt_column].to_numpy(float),
        "log_counts": np.log1p(adata.obs["total_counts"].to_numpy(float)),
        "n_genes": adata.obs["n_genes_by_counts"].to_numpy(float),
    }, index=adata.obs_names)
    reference_subjects = adata.obs.loc[reference_mask, "subject_id"].astype(str)
    for column in ["log_counts", "n_genes"]:
        values = frame.loc[reference_mask, column].to_numpy(float)[:, None]
        mean, sd = weighted_mean_sd(values, reference_subjects)
        frame[f"{column}_z"] = (frame[column] - mean[0]) / sd[0]
    formula = (
        "bs(pt, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound=1) "
        "+ log_counts_z + n_genes_z"
    )
    design_reference = patsy.dmatrix(formula, frame.loc[reference_mask], return_type="dataframe")
    counts = reference_subjects.value_counts()
    weights = reference_subjects.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    outcome = adata.obs.loc[reference_mask, score_column].to_numpy(float)
    model = sm.WLS(outcome, design_reference, weights=weights).fit()
    design_all = patsy.build_design_matrices([design_reference.design_info], frame, return_type="dataframe")[0]
    residual = adata.obs[score_column].to_numpy(float) - np.asarray(model.predict(design_all), dtype=float)
    center = weighted_median(residual[reference_mask], reference_subjects)
    centered_reference = residual[reference_mask] - center
    _, scale = weighted_mean_sd(centered_reference[:, None], reference_subjects)
    return (residual - center) / scale[0], float(model.rsquared)


def donor_trajectory_correlations(
    adata: sc.AnnData,
    dataset: str,
    pt_column: str,
) -> pd.DataFrame:
    rows = []
    score_columns = [f"LAM_score_{signature}" for signature in SIGNATURES]
    for subject, frame in adata.obs.groupby("subject_id", observed=True):
        if len(frame) < MIN_DONOR_CELLS:
            continue
        pt = frame[pt_column].to_numpy(float)
        if np.ptp(pt) < 0.05:
            continue
        for signature, column in zip(SIGNATURES, score_columns):
            correlation = stats.spearmanr(pt, frame[column].to_numpy(float))
            rows.append({
                "dataset": dataset,
                "subject_id": str(subject),
                "signature": signature,
                "n_cells": len(frame),
                "spearman_with_differentiation": correlation.statistic,
                "p_value": correlation.pvalue,
            })
    return pd.DataFrame(rows)


def prepare_dataset(root: Path, dataset: str, config: dict) -> tuple[sc.AnnData, pd.DataFrame, pd.DataFrame]:
    source = sc.read_h5ad(root / config["path"])
    if not config["external"]:
        source = source[source.obs["tissue"].astype(str).str.upper().eq("CSF")].copy()
    else:
        source = source.copy()
    availability = add_signature_scores(source)
    availability.insert(0, "dataset", dataset)

    control_group = config["control"]
    if control_group is None:
        reference = np.ones(source.n_obs, dtype=bool)
        reference_kind = "all_cells_for_trajectory_only"
    else:
        reference = source.obs["disease_group"].astype(str).eq(control_group).to_numpy()
        reference_kind = "control_cells"
    diagnostics = []
    for signature in SIGNATURES:
        column = f"LAM_score_{signature}"
        residual, r_squared = residualize_signature(source, column, config["pt"], reference)
        source.obs[f"LAM_residual_{signature}"] = residual
        diagnostics.append({
            "dataset": dataset,
            "signature": signature,
            "reference_kind": reference_kind,
            "n_reference_cells": int(reference.sum()),
            "n_reference_subjects": int(source.obs.loc[reference, "subject_id"].nunique()),
            "baseline_r_squared": r_squared,
        })
    source.obs["LAM_residual_LAM4_minus_LAM2"] = (
        source.obs["LAM_residual_mouse_LAM4_fibrotic"]
        - source.obs["LAM_residual_mouse_LAM2_complement_APC"]
    )
    return source, availability, pd.DataFrame(diagnostics)


def donor_scores(adata: sc.AnnData, dataset: str, config: dict) -> pd.DataFrame:
    pt_column = config["pt"]
    activation_column = config["activation"]
    has_activation = activation_column is not None and activation_column in adata.obs
    if not has_activation:
        activation_threshold = np.nan
    elif config["external"]:
        activation_threshold = float(adata.uns["external_control_activation_q75"])
    elif config["control"] is not None:
        activation_threshold = float(adata.uns["two_axis_control_activation_q75"])
    else:
        activation_threshold = float(np.quantile(adata.obs[activation_column].to_numpy(float), 0.75))
    rows = []
    residual_columns = [f"LAM_residual_{signature}" for signature in SIGNATURES]
    residual_columns.append("LAM_residual_LAM4_minus_LAM2")
    for subject, frame in adata.obs.groupby("subject_id", observed=True):
        pt = frame[pt_column].to_numpy(float)
        activation = (
            frame[activation_column].to_numpy(float)
            if has_activation
            else np.full(len(frame), np.nan)
        )
        incomplete = pt < INCOMPLETE_CUTOFF
        high = incomplete & (activation > activation_threshold) if has_activation else np.zeros(len(frame), dtype=bool)
        other = incomplete & ~high if has_activation else np.zeros(len(frame), dtype=bool)
        base = {
            "dataset": dataset,
            "subject_id": str(subject),
            "diagnosis": str(frame["diagnosis"].iloc[0]),
            "disease_group": str(frame["disease_group"].iloc[0]),
            "n_cells": len(frame),
            "n_incomplete": int(incomplete.sum()),
            "n_decoupled": int(high.sum()),
            "n_other_incomplete": int(other.sum()),
        }
        if config["external"]:
            for column in ["age", "sex", "sample_type", "dmt_at_sampling", "publication_final_new"]:
                base[column] = frame[column].iloc[0]
        for column in residual_columns:
            residual = frame[column].to_numpy(float)
            key = column.replace("LAM_residual_", "")
            base[f"mean_residual_all__{key}"] = float(residual.mean())
            base[f"mean_residual_incomplete__{key}"] = float(residual[incomplete].mean()) if incomplete.any() else np.nan
            base[f"mean_residual_decoupled__{key}"] = float(residual[high].mean()) if high.any() else np.nan
            base[f"decoupled_excess__{key}"] = (
                float(residual[high].mean() - residual[other].mean())
                if high.sum() >= 5 and other.sum() >= 5
                else np.nan
            )
        rows.append(base)
    return pd.DataFrame(rows)


def permutation_test(case: np.ndarray, control: np.ndarray, seed: int) -> tuple[float, float, str]:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    pooled = np.concatenate([case, control])
    observed = float(case.mean() - control.mean())
    total = math.comb(len(pooled), len(case))
    if total <= 100000:
        differences = []
        for combination in itertools.combinations(np.arange(len(pooled)), len(case)):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(combination)] = True
            differences.append(pooled[mask].mean() - pooled[~mask].mean())
        differences = np.asarray(differences)
        p_value = float(np.mean(np.abs(differences) >= abs(observed) - 1e-12))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        exceedances = 0
        for _ in range(100000):
            permuted = rng.permutation(pooled)
            difference = permuted[: len(case)].mean() - permuted[len(case):].mean()
            exceedances += abs(difference) >= abs(observed) - 1e-12
        p_value = float((exceedances + 1) / 100001)
        method = "monte_carlo_100000"
    comparisons = case[:, None] - control[None, :]
    cliffs = float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)
    return p_value, cliffs, method


def sign_flip_test(values: np.ndarray, seed: int) -> tuple[float, str]:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    total = 2 ** len(values)
    if total <= 100000:
        means = []
        for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
            means.append(np.mean(values * np.asarray(signs)))
        p_value = float(np.mean(np.abs(means) >= observed - 1e-12))
        method = "exact_sign_flip"
    else:
        rng = np.random.default_rng(seed)
        random_signs = rng.choice([-1.0, 1.0], size=(100000, len(values)))
        means = np.mean(random_signs * values[None, :], axis=1)
        p_value = float((np.sum(np.abs(means) >= observed - 1e-12) + 1) / 100001)
        method = "monte_carlo_sign_flip_100000"
    return p_value, method


def trajectory_summary(correlations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, signature), frame in correlations.groupby(["dataset", "signature"], observed=True):
        values = frame["spearman_with_differentiation"].dropna().to_numpy(float)
        p_value, method = sign_flip_test(values, SEED)
        rows.append({
            "dataset": dataset,
            "signature": signature,
            "n_donors": len(values),
            "median_donor_spearman": float(np.median(values)),
            "positive_donor_fraction": float(np.mean(values > 0)),
            "sign_flip_p": p_value,
            "test_method": method,
        })
    summary = pd.DataFrame(rows)
    summary["fdr_bh_within_dataset"] = summary.groupby("dataset", observed=True)["sign_flip_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return summary


def decoupled_pair_tests(donors: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    endpoints = [f"decoupled_excess__{signature}" for signature in SIGNATURES]
    endpoints.append("decoupled_excess__LAM4_minus_LAM2")
    for dataset, frame in donors.groupby("dataset", observed=True):
        disease = frame.loc[
            frame["diagnosis"].ne("Ctrl")
            & frame["disease_group"].ne("IIH_control")
            & frame["disease_group"].ne("non_inflammatory_control")
        ]
        disease = disease.loc[disease["n_cells"].ge(MIN_DONOR_CELLS)]
        for endpoint in endpoints:
            values = disease[endpoint].dropna().to_numpy(float)
            if len(values) < 3:
                continue
            p_value, method = sign_flip_test(values, seed)
            rows.append({
                "dataset": dataset,
                "endpoint": endpoint,
                "n_donors": len(values),
                "mean_decoupled_excess": values.mean(),
                "median_decoupled_excess": np.median(values),
                "positive_fraction": np.mean(values > 0),
                "sign_flip_p": p_value,
                "test_method": method,
            })
    tests = pd.DataFrame(rows)
    tests["fdr_bh_within_dataset"] = tests.groupby("dataset", observed=True)["sign_flip_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return tests


def decoupled_meta(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, frame in tests.groupby("endpoint", observed=True):
        signs = np.sign(frame["mean_decoupled_excess"].to_numpy(float))
        p_values = np.clip(frame["sign_flip_p"].to_numpy(float), 1e-12, 1)
        z_scores = signs * stats.norm.isf(p_values / 2)
        weights = np.sqrt(frame["n_donors"].to_numpy(float))
        combined_z = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
        rows.append({
            "endpoint": endpoint,
            "n_cohorts": len(frame),
            "direction_concordant": bool(len(set(signs)) == 1),
            "mean_cohort_excess": frame["mean_decoupled_excess"].mean(),
            "combined_z": combined_z,
            "stouffer_p": float(2 * stats.norm.sf(abs(combined_z))),
        })
    meta = pd.DataFrame(rows)
    meta["fdr_bh"] = multipletests(meta["stouffer_p"], method="fdr_bh")[1]
    return meta.sort_values("stouffer_p")


def external_stage_tests(donors: pd.DataFrame, seed: int) -> pd.DataFrame:
    external = donors.loc[
        donors["dataset"].eq("GSE328265") & donors["n_cells"].ge(MIN_DONOR_CELLS)
    ].copy()
    contrasts = {
        "MS_vs_control": (["RRMS", "PPMS", "SPMS"], ["Ctrl"]),
        "PMS_vs_RRMS": (["PPMS", "SPMS"], ["RRMS"]),
        "SPMS_vs_PPMS": (["SPMS"], ["PPMS"]),
    }
    endpoints = []
    for signature in SIGNATURES:
        endpoints.extend([
            f"mean_residual_incomplete__{signature}",
            f"mean_residual_decoupled__{signature}",
            f"decoupled_excess__{signature}",
        ])
    endpoints.extend([
        "mean_residual_incomplete__LAM4_minus_LAM2",
        "mean_residual_decoupled__LAM4_minus_LAM2",
        "decoupled_excess__LAM4_minus_LAM2",
    ])
    rows = []
    for contrast, (case_groups, control_groups) in contrasts.items():
        for endpoint in endpoints:
            case = external.loc[external["diagnosis"].isin(case_groups), endpoint].dropna().to_numpy(float)
            control = external.loc[external["diagnosis"].isin(control_groups), endpoint].dropna().to_numpy(float)
            if len(case) < 2 or len(control) < 2:
                continue
            p_value, cliffs, method = permutation_test(case, control, seed)
            rows.append({
                "contrast": contrast,
                "endpoint": endpoint,
                "n_case": len(case),
                "n_control": len(control),
                "mean_case": case.mean(),
                "mean_control": control.mean(),
                "mean_difference_case_minus_control": case.mean() - control.mean(),
                "cliffs_delta": cliffs,
                "permutation_p": p_value,
                "permutation_method": method,
            })
    tests = pd.DataFrame(rows)
    tests["fdr_bh_within_contrast"] = tests.groupby("contrast", observed=True)["permutation_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return tests


def write_report(
    path: Path,
    ledger: pd.DataFrame,
    availability: pd.DataFrame,
    trajectory: pd.DataFrame,
    pair_tests: pd.DataFrame,
    meta: pd.DataFrame,
    stage_tests: pd.DataFrame,
) -> None:
    lines = [
        "# Cross-species meningeal LAM reverse translation",
        "",
        "## Source gate",
        "",
        f"The source is Villar-Vesga et al. ({SOURCE_DOI}). Its NOD-EAE scRNA-seq matrix and code were not "
        "public at the analysis date; the manuscript states that they will be released upon publication. "
        "This analysis therefore projects only gene programs explicitly stated by the authors.",
        "",
        "## Non-circularity audit",
        "",
        f"Exact overlap between the four source-defined LAM programs and the 16 genes used to define "
        f"the MHC-I/IFN activation axis: {int(ledger['used_to_define_axis2_activation'].sum())} genes. "
        "LAM enrichment is therefore not created by reusing genes from the state-defining activation score.",
        "",
        "## Signature coverage",
        "",
    ]
    for signature, frame in availability.groupby("signature", observed=True):
        lines.append(
            f"- {signature}: {int(frame['n_available'].min())}/{int(frame['n_requested'].max())} genes available "
            "in every human dataset."
        )
    lines.extend(["", "## Position along the human monocyte-to-BAM continuum", ""])
    for signature, frame in trajectory.groupby("signature", observed=True):
        values = ", ".join(
            f"{row.dataset} r={row.median_donor_spearman:.3f}" for row in frame.itertuples()
        )
        lines.append(f"- {signature}: {values}.")
    lines.extend(["", "## Decoupled-state excess after differentiation adjustment", ""])
    for row in meta.itertuples():
        lines.append(
            f"- {row.endpoint}: z={row.combined_z:.3f}, P={row.stouffer_p:.4g}, "
            f"FDR={row.fdr_bh:.4g}, direction concordant={row.direction_concordant}."
        )
    lines.extend(["", "## GSE328265 progressive-subtype tests", ""])
    subtype = stage_tests.loc[stage_tests["contrast"].eq("SPMS_vs_PPMS")].sort_values("permutation_p").head(8)
    for row in subtype.itertuples():
        lines.append(
            f"- {row.endpoint}: SPMS-minus-PPMS={row.mean_difference_case_minus_control:.3f}, "
            f"Cliff delta={row.cliffs_delta:.3f}, P={row.permutation_p:.4g}, "
            f"FDR={row.fdr_bh_within_contrast:.4g}."
        )
    lines.extend([
        "",
        "## Decision rule",
        "",
        "Positive LAM4-minus-LAM2 residual identifies a fibrotic/wound-healing branch beyond that expected "
        "from differentiation position. Positive precursor excess with absent terminal-LAM excess supports a "
        "stall before LAM acquisition. Positive terminal-LAM excess supports accelerated acquisition rather "
        "than a simple maturation block.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results" / "cross_species_lam"
    output.mkdir(parents=True, exist_ok=True)

    ledger = signature_ledger(root)
    ledger.to_csv(output / "meningeal_LAM_signature_ledger.csv", index=False)
    all_availability = []
    all_diagnostics = []
    all_correlations = []
    all_donors = []
    for dataset, config in DATASETS.items():
        print(f"LAM projection: {dataset}", flush=True)
        adata, availability, diagnostics = prepare_dataset(root, dataset, config)
        correlations = donor_trajectory_correlations(adata, dataset, config["pt"])
        donors = donor_scores(adata, dataset, config)
        score_columns = [
            column for column in adata.obs
            if column.startswith("LAM_score_") or column.startswith("LAM_residual_")
        ]
        cell_columns = ["subject_id", "diagnosis", "disease_group", config["pt"]]
        if config["activation"] is not None:
            cell_columns.append(config["activation"])
        cell_columns.extend(score_columns)
        adata.obs[cell_columns].to_csv(
            output / f"{dataset}_LAM_cell_scores.csv.gz", compression="gzip"
        )
        all_availability.append(availability)
        all_diagnostics.append(diagnostics)
        all_correlations.append(correlations)
        all_donors.append(donors)

    availability = pd.concat(all_availability, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    correlations = pd.concat(all_correlations, ignore_index=True)
    donors = pd.concat(all_donors, ignore_index=True)
    trajectory = trajectory_summary(correlations)
    pair_tests = decoupled_pair_tests(donors, args.seed)
    meta = decoupled_meta(pair_tests)
    stage_tests = external_stage_tests(donors, args.seed)

    availability.to_csv(output / "LAM_signature_availability.csv", index=False)
    diagnostics.to_csv(output / "LAM_control_manifold_diagnostics.csv", index=False)
    correlations.to_csv(output / "LAM_donor_trajectory_correlations.csv", index=False)
    trajectory.to_csv(output / "LAM_trajectory_summary.csv", index=False)
    donors.to_csv(output / "LAM_donor_scores.csv", index=False)
    pair_tests.to_csv(output / "LAM_decoupled_pair_tests.csv", index=False)
    meta.to_csv(output / "LAM_decoupled_meta.csv", index=False)
    stage_tests.to_csv(output / "GSE328265_LAM_stage_tests.csv", index=False)
    write_report(
        output / "CROSS_SPECIES_LAM_REPORT.md",
        ledger,
        availability,
        trajectory,
        pair_tests,
        meta,
        stage_tests,
    )
    print(meta.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
