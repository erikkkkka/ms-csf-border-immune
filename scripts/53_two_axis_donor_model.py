#!/usr/bin/env python3
"""Fit donor-level differentiation-by-activation models in MS CSF."""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import numpy as np
import pandas as pd
import patsy
import scanpy as sc
from scipy import sparse, stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


COHORTS = {
    "GSE138266": "IIH_control",
    "GSE286068": "non_inflammatory_control",
}

MODULES = {
    "MHC_I": ["B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5"],
    "IFN": ["IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7"],
    "MHC_II": ["CD74", "CIITA", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"],
}

TEST_METRICS = [
    "median_consensus_pseudotime",
    "median_BAM_macrostate_membership",
    "transitional_fraction",
    "terminal_fraction",
    "transition_entropy",
    "mean_activation_residual",
    "high_activation_fraction",
    "activated_transition_fraction",
    "decoupled_inflammatory_fraction",
    "continuous_decoupling_index",
    "strict_BAM_mean_activation_residual",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def z_module_scores(adata: sc.AnnData, cell_mask: np.ndarray) -> dict[str, np.ndarray]:
    genes = sorted({gene for values in MODULES.values() for gene in values if gene in adata.var_names})
    matrix = dense(adata[:, genes].X).astype(np.float64)
    reference = matrix[cell_mask]
    means = reference.mean(axis=0)
    standard_deviations = reference.std(axis=0)
    standard_deviations[standard_deviations == 0] = 1
    standardized = (matrix - means) / standard_deviations
    gene_to_column = {gene: index for index, gene in enumerate(genes)}
    scores = {}
    for module, requested in MODULES.items():
        available = [gene for gene in requested if gene in gene_to_column]
        indices = [gene_to_column[gene] for gene in available]
        scores[module] = standardized[:, indices].mean(axis=1)
        adata.uns[f"two_axis_{module}_genes"] = available
    return scores


def baseline_residual(
    outcome: np.ndarray,
    obs: pd.DataFrame,
    control_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    frame = pd.DataFrame({
        "pt": obs["consensus_pseudotime"].to_numpy(float),
        "log_counts": np.log1p(obs["total_counts"].to_numpy(float)),
        "n_genes": obs["n_genes_by_counts"].to_numpy(float),
    }, index=obs.index)
    for column in ["log_counts", "n_genes"]:
        mean = frame.loc[control_mask, column].mean()
        sd = frame.loc[control_mask, column].std(ddof=0)
        frame[f"{column}_z"] = (frame[column] - mean) / (sd if sd > 0 else 1)
    formula = (
        "bs(pt, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound=1) "
        "+ log_counts_z + n_genes_z"
    )
    control_design = patsy.dmatrix(formula, frame.loc[control_mask], return_type="dataframe")
    model = sm.OLS(outcome[control_mask], control_design).fit()
    all_design = patsy.build_design_matrices([control_design.design_info], frame, return_type="dataframe")[0]
    residual = outcome - model.predict(all_design)
    center = np.median(residual[control_mask])
    scale = np.std(residual[control_mask], ddof=1)
    standardized = (residual - center) / (scale if scale > 0 else 1)
    diagnostics = {
        "control_n_cells": int(control_mask.sum()),
        "control_r_squared": float(model.rsquared),
        "control_residual_center": float(center),
        "control_residual_sd": float(scale),
    }
    return standardized, diagnostics


def normalized_entropy(values: np.ndarray) -> float:
    counts = np.bincount(values, minlength=3).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(3))


def aggregate_donors(adata: sc.AnnData, dataset: str, control_group: str) -> pd.DataFrame:
    csf = adata.obs["tissue"].astype(str).str.upper().eq("CSF").to_numpy()
    scores = z_module_scores(adata, csf)
    for module, values in scores.items():
        adata.obs[f"two_axis_{module}_z"] = values
    control = csf & adata.obs["disease_group"].astype(str).eq(control_group).to_numpy()
    diagnostics = []
    residuals = []
    for module in ["MHC_I", "IFN"]:
        residual, diagnostic = baseline_residual(scores[module], adata.obs, control)
        adata.obs[f"two_axis_{module}_residual_z"] = residual
        diagnostic.update({"dataset": dataset, "module": module})
        diagnostics.append(diagnostic)
        residuals.append(residual)
    activation = np.mean(np.column_stack(residuals), axis=1)
    adata.obs["two_axis_activation_residual"] = activation
    activation_threshold = float(np.quantile(activation[control], 0.75))
    adata.uns["two_axis_control_activation_q75"] = activation_threshold
    adata.uns["two_axis_control_group"] = control_group

    rows = []
    csf_obs = adata.obs.loc[csf].copy()
    for subject, frame in csf_obs.groupby("subject_id", observed=True):
        pt = frame["consensus_pseudotime"].to_numpy(float)
        activation_values = frame["two_axis_activation_residual"].to_numpy(float)
        bins = np.where(pt < 0.4, 0, np.where(pt < 0.8, 1, 2))
        high_activation = activation_values > activation_threshold
        transitional = bins == 1
        strict_bam = frame["trajectory_role"].astype(str).eq("terminal_candidate").to_numpy()
        rows.append({
            "dataset": dataset,
            "subject_id": str(subject),
            "diagnosis": str(frame["diagnosis"].iloc[0]),
            "disease_group": str(frame["disease_group"].iloc[0]),
            "is_MS": str(frame["disease_group"].iloc[0]) == "MS",
            "n_CSF_continuum_cells": len(frame),
            "n_strict_BAM": int(strict_bam.sum()),
            "median_consensus_pseudotime": float(np.median(pt)),
            "median_BAM_macrostate_membership": float(np.median(frame["cellrank_BAM_macrostate_membership"])),
            "transitional_fraction": float(transitional.mean()),
            "terminal_fraction": float((bins == 2).mean()),
            "transition_entropy": normalized_entropy(bins),
            "mean_activation_residual": float(np.mean(activation_values)),
            "high_activation_fraction": float(high_activation.mean()),
            "activated_transition_fraction": float((transitional & high_activation).mean()),
            "decoupled_inflammatory_fraction": float(((bins < 2) & high_activation).mean()),
            "continuous_decoupling_index": float(np.mean(activation_values * (1 - pt))),
            "strict_BAM_mean_activation_residual": float(np.mean(activation_values[strict_bam])) if strict_bam.any() else np.nan,
        })
    output_path = ROOT / "results" / "trajectory" / "continuum_objects" / f"{dataset}_monocyte_BAM_continuum.h5ad"
    adata.write_h5ad(output_path, compression="gzip")
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def exact_permutation_test(case: np.ndarray, control: np.ndarray) -> tuple[float, float, float]:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    pooled = np.concatenate([case, control])
    n_case = len(case)
    observed = case.mean() - control.mean()
    differences = []
    all_indices = np.arange(len(pooled))
    for combination in itertools.combinations(all_indices, n_case):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(combination)] = True
        differences.append(pooled[mask].mean() - pooled[~mask].mean())
    differences = np.asarray(differences)
    p_value = float(np.mean(np.abs(differences) >= abs(observed) - 1e-12))
    comparisons = case[:, None] - control[None, :]
    cliffs_delta = float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)
    pooled_sd = np.sqrt(((len(case) - 1) * case.var(ddof=1) + (len(control) - 1) * control.var(ddof=1)) / (len(pooled) - 2))
    standardized = float(observed / pooled_sd) if pooled_sd > 0 else np.nan
    return p_value, cliffs_delta, standardized


def disease_tests(donors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in donors.groupby("dataset", observed=True):
        frame = frame.loc[frame["n_CSF_continuum_cells"] >= 20]
        for metric in TEST_METRICS:
            case = frame.loc[frame["is_MS"], metric].dropna().to_numpy(float)
            control = frame.loc[~frame["is_MS"], metric].dropna().to_numpy(float)
            p_value, cliffs_delta, standardized = exact_permutation_test(case, control)
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "n_MS": len(case),
                "n_control": len(control),
                "mean_MS": case.mean(),
                "mean_control": control.mean(),
                "mean_difference_MS_minus_control": case.mean() - control.mean(),
                "standardized_mean_difference": standardized,
                "cliffs_delta": cliffs_delta,
                "exact_permutation_p": p_value,
            })
    tests = pd.DataFrame(rows)
    tests["fdr_bh_within_dataset"] = tests.groupby("dataset", observed=True)["exact_permutation_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return tests


def meta_tests(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, frame in tests.groupby("metric", observed=True):
        if frame["dataset"].nunique() != len(COHORTS):
            continue
        signs = np.sign(frame["mean_difference_MS_minus_control"].to_numpy(float))
        p_values = np.clip(frame["exact_permutation_p"].to_numpy(float), 1e-12, 1)
        z_scores = signs * stats.norm.isf(p_values / 2)
        weights = np.sqrt((frame["n_MS"] + frame["n_control"]).to_numpy(float))
        combined_z = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
        rows.append({
            "metric": metric,
            "n_cohorts": len(frame),
            "direction_concordant": bool(len(set(signs)) == 1),
            "combined_z": combined_z,
            "stouffer_p": float(2 * stats.norm.sf(abs(combined_z))),
            "mean_standardized_effect": frame["standardized_mean_difference"].mean(),
        })
    meta = pd.DataFrame(rows)
    meta["fdr_bh"] = multipletests(meta["stouffer_p"], method="fdr_bh")[1]
    return meta.sort_values("stouffer_p")


def residual_axis_correlation(donors: pd.DataFrame) -> pd.DataFrame:
    frame = donors.loc[donors["n_CSF_continuum_cells"] >= 20].copy()
    design = patsy.dmatrix("0 + C(dataset):C(disease_group)", frame, return_type="dataframe")
    x = frame["median_consensus_pseudotime"].to_numpy(float)
    y = frame["mean_activation_residual"].to_numpy(float)
    x_residual = sm.OLS(x, design).fit().resid
    y_residual = sm.OLS(y, design).fit().resid
    correlation = stats.spearmanr(x_residual, y_residual)
    return pd.DataFrame([{
        "n_donors": len(frame),
        "strata_adjusted_spearman": correlation.statistic,
        "p_value": correlation.pvalue,
    }])


def write_report(
    path: Path,
    diagnostics: pd.DataFrame,
    tests: pd.DataFrame,
    meta: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    lines = [
        "# Differentiation-by-activation donor model",
        "",
        "## Definition",
        "",
        "Axis 1 is consensus monocyte-to-BAM pseudotime. Axis 2 is the mean standardized "
        "residual of MHC-I and IFN modules after fitting their control-cell relationship "
        "with pseudotime, total counts, and detected genes. All disease tests use donor "
        "summaries and exact label permutations.",
        "",
        "## Control-manifold fits",
        "",
    ]
    for row in diagnostics.itertuples():
        lines.append(
            f"- {row.dataset} {row.module}: {row.control_n_cells} control cells; "
            f"baseline R2={row.control_r_squared:.3f}."
        )
    lines.extend(["", "## Cross-cohort results", ""])
    for row in meta.itertuples():
        lines.append(
            f"- {row.metric}: z={row.combined_z:.3f}, P={row.stouffer_p:.4g}, "
            f"FDR={row.fdr_bh:.4g}, direction concordant={row.direction_concordant}."
        )
    corr = correlation.iloc[0]
    lines.extend([
        "",
        "## Axis orthogonality",
        "",
        f"Across {int(corr.n_donors)} donors, after dataset:disease-stratum adjustment, "
        f"Axis 1 versus Axis 2 Spearman r={corr.strata_adjusted_spearman:.3f}, "
        f"P={corr.p_value:.4g}.",
        "",
        "## Interpretation",
        "",
        "The transition and activation endpoints must be interpreted together. A disease "
        "effect on activation residual without a shift in median pseudotime supports an "
        "orthogonal inflammatory field; enrichment of activated transitional cells supports "
        "a disease-associated stall or altered occupancy along the border-adaptation process.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    global ROOT
    ROOT = root
    donor_frames = []
    diagnostic_frames = []
    for dataset, control_group in COHORTS.items():
        print(f"Two-axis model: {dataset}", flush=True)
        path = root / "results" / "trajectory" / "continuum_objects" / f"{dataset}_monocyte_BAM_continuum.h5ad"
        adata = sc.read_h5ad(path)
        donors, diagnostics = aggregate_donors(adata, dataset, control_group)
        donor_frames.append(donors)
        diagnostic_frames.append(diagnostics)
    donors = pd.concat(donor_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    tests = disease_tests(donors)
    meta = meta_tests(tests)
    correlation = residual_axis_correlation(donors)
    output_dir = root / "results" / "two_axis"
    output_dir.mkdir(parents=True, exist_ok=True)
    donors.to_csv(output_dir / "two_axis_donor_scores.csv", index=False)
    diagnostics.to_csv(output_dir / "control_manifold_fit_diagnostics.csv", index=False)
    tests.to_csv(output_dir / "two_axis_disease_tests.csv", index=False)
    meta.to_csv(output_dir / "two_axis_meta_tests.csv", index=False)
    correlation.to_csv(output_dir / "two_axis_orthogonality.csv", index=False)
    write_report(output_dir / "TWO_AXIS_DONOR_MODEL_REPORT.md", diagnostics, tests, meta, correlation)
    print(meta.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
