#!/usr/bin/env python3
"""Project the frozen monocyte-to-BAM program into GSE328265."""

from __future__ import annotations

import argparse
import itertools
import os
import re
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


DISCOVERY = ["GSE138266", "GSE194078", "GSE286068"]
PROGRAM_SIZE_PER_DIRECTION = 40
MIN_DIRECTIONAL_CORRELATION = 0.15
MIN_CELLS_PER_DONOR = 20
INCOMPLETE_ADAPTATION_CUTOFF = 0.80
SEED = 12345

ACTIVATION_MODULES = {
    "MHC_I": ["B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5"],
    "IFN": ["IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7"],
}

ANNOTATION_MARKERS = {
    "classical_monocyte": ["S100A8", "S100A9", "FCN1", "VCAN", "CCR2", "SELL", "CTSD"],
    "nonclassical_monocyte": ["FCGR3A", "MS4A7", "LILRB1", "IFITM3", "LST1", "LYZ"],
    "cDC1": ["CLEC9A", "XCR1", "BATF3", "CADM1", "WDFY4"],
    "cDC2": ["CD1C", "FCER1A", "CLEC10A", "CD1E", "CST3"],
    "DC3_moDC": ["CD1C", "CD14", "FCGR3A", "CD163", "S100A8", "CTSS"],
    "pDC": ["GZMB", "JCHAIN", "LILRA4", "CLEC4C", "TCF4", "IL3RA"],
    "BAM": ["C1QA", "C1QB", "C1QC", "APOE", "TREM2", "LYVE1", "MRC1", "STAB1", "CH25H"],
}

STRESS_AND_CYCLE = {
    "FOS", "FOSB", "JUN", "JUNB", "JUND", "DUSP1", "IER2", "IER3", "EGR1", "EGR2",
    "HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1", "HSPB1", "MKI67", "TOP2A", "TUBA1B",
    "XIST", "RPS4Y1", "KDM5D", "DDX3Y", "EIF1AY", "UTY",
}

DONOR_METRICS = [
    "median_projected_differentiation",
    "terminal_score_fraction",
    "strict_BAM_fraction",
    "mean_activation_residual",
    "continuous_decoupling_index",
    "decoupled_inflammatory_fraction",
]

CONTRASTS = {
    "MS_vs_control": (["RRMS", "PPMS", "SPMS"], ["Ctrl"]),
    "RRMS_vs_control": (["RRMS"], ["Ctrl"]),
    "PMS_vs_control": (["PPMS", "SPMS"], ["Ctrl"]),
    "PMS_vs_RRMS": (["PPMS", "SPMS"], ["RRMS"]),
    "SPMS_vs_RRMS": (["SPMS"], ["RRMS"]),
    "PPMS_vs_RRMS": (["PPMS"], ["RRMS"]),
    "SPMS_vs_PPMS": (["SPMS"], ["PPMS"]),
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


def gene_is_eligible(gene: str) -> bool:
    if re.match(r"^(MT-|RPS|RPL|HLA-|IG[HKL]|TR[ABDG])", gene):
        return False
    excluded = STRESS_AND_CYCLE | {
        value
        for collection in list(ACTIVATION_MODULES.values()) + list(ANNOTATION_MARKERS.values())
        for value in collection
    }
    return gene not in excluded


def weighted_mean_sd(matrix: np.ndarray, subjects: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = subjects.value_counts()
    weights = subjects.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float, copy=True)
    weights /= weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(np.square(matrix - mean) * weights[:, None], axis=0)
    sd = np.sqrt(variance)
    sd[sd < 1e-8] = 1.0
    return mean, sd


def weighted_quantile(values: np.ndarray, subjects: pd.Series, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    counts = subjects.value_counts()
    weights = subjects.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=float, copy=True)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, values[order]))


def rank_correlations(matrix: np.ndarray, pseudotime: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 4 or np.unique(pseudotime).size < 3:
        return np.full(matrix.shape[1], np.nan)
    x_rank = stats.rankdata(matrix, axis=0)
    y_rank = stats.rankdata(pseudotime)
    x_centered = x_rank - x_rank.mean(axis=0)
    y_centered = y_rank - y_rank.mean()
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    result = np.divide(
        np.sum(x_centered * y_centered[:, None], axis=0),
        denominator,
        out=np.zeros(matrix.shape[1], dtype=float),
        where=denominator > 0,
    )
    return result


def donor_bin_correlations(
    adata: sc.AnnData,
    genes: list[str],
    dataset: str,
) -> pd.DataFrame:
    csf = adata.obs["tissue"].astype(str).str.upper().eq("CSF")
    adata = adata[csf, genes]
    rows = []
    for subject, positions in adata.obs.groupby("subject_id", observed=True).indices.items():
        positions = np.asarray(positions)
        if len(positions) < MIN_CELLS_PER_DONOR:
            continue
        pt = adata.obs.iloc[positions]["consensus_pseudotime"].to_numpy(float)
        n_bins = min(6, max(4, len(positions) // 15))
        labels = pd.qcut(pt, q=n_bins, labels=False, duplicates="drop")
        labels = np.asarray(labels)
        unique = np.unique(labels)
        if len(unique) < 4:
            continue
        matrix = adata.X[positions]
        binned_expression = []
        binned_pt = []
        for label in unique:
            selected = labels == label
            if selected.sum() < 3:
                continue
            binned_expression.append(np.asarray(matrix[selected].mean(axis=0)).ravel())
            binned_pt.append(np.median(pt[selected]))
        if len(binned_pt) < 4:
            continue
        correlations = rank_correlations(np.vstack(binned_expression), np.asarray(binned_pt))
        rows.append(pd.DataFrame({
            "dataset": dataset,
            "subject_id": str(subject),
            "gene": genes,
            "within_donor_spearman": correlations,
            "n_bins": len(binned_pt),
        }))
    if not rows:
        raise RuntimeError(f"{dataset}: no donors support trajectory-gene estimation")
    return pd.concat(rows, ignore_index=True)


def derive_program_statistics(root: Path) -> tuple[pd.DataFrame, dict[str, sc.AnnData]]:
    objects = {}
    shared = None
    for dataset in DISCOVERY:
        path = root / "results" / "trajectory" / "continuum_objects" / f"{dataset}_monocyte_BAM_continuum.h5ad"
        adata = sc.read_h5ad(path)
        objects[dataset] = adata
        genes = {gene for gene in adata.var_names.astype(str) if gene_is_eligible(gene)}
        shared = genes if shared is None else shared & genes
    external_path = root / "results" / "external_validation" / "GSE328265" / "GSE328265_clean_annotated.h5ad"
    external_genes = set(sc.read_h5ad(external_path, backed="r").var_names.astype(str))
    shared = sorted(shared & external_genes)

    correlation_frames = []
    prevalence_frames = []
    for dataset, adata in objects.items():
        csf = adata.obs["tissue"].astype(str).str.upper().eq("CSF").to_numpy()
        matrix = adata[csf, shared].X
        prevalence = np.asarray((matrix > 0).mean(axis=0)).ravel()
        prevalence_frames.append(pd.DataFrame({
            "dataset": dataset,
            "gene": shared,
            "prevalence": prevalence,
        }))
        correlation_frames.append(donor_bin_correlations(adata, shared, dataset))

    donor_correlations = pd.concat(correlation_frames, ignore_index=True)
    summaries = (
        donor_correlations.groupby(["dataset", "gene"], observed=True)
        .agg(
            median_within_donor_spearman=("within_donor_spearman", "median"),
            n_donors=("subject_id", "nunique"),
            positive_donor_fraction=("within_donor_spearman", lambda values: np.mean(np.asarray(values) > 0)),
        )
        .reset_index()
    )
    prevalence = pd.concat(prevalence_frames, ignore_index=True)
    summaries = summaries.merge(prevalence, on=["dataset", "gene"], how="left")
    wide_r = summaries.pivot(index="gene", columns="dataset", values="median_within_donor_spearman")
    wide_prev = summaries.pivot(index="gene", columns="dataset", values="prevalence")
    statistics = pd.DataFrame(index=wide_r.index)
    for dataset in DISCOVERY:
        statistics[f"r_{dataset}"] = wide_r[dataset]
        statistics[f"prevalence_{dataset}"] = wide_prev[dataset]
    r_values = statistics[[f"r_{dataset}" for dataset in DISCOVERY]].to_numpy(float)
    statistics["direction_concordant"] = (np.all(r_values > 0, axis=1) | np.all(r_values < 0, axis=1))
    statistics["direction"] = np.where(np.all(r_values > 0, axis=1), "positive", np.where(np.all(r_values < 0, axis=1), "negative", "discordant"))
    statistics["median_r"] = np.median(r_values, axis=1)
    statistics["minimum_directional_abs_r"] = np.min(np.abs(r_values), axis=1)
    statistics["minimum_prevalence"] = statistics[[f"prevalence_{dataset}" for dataset in DISCOVERY]].min(axis=1)
    statistics = statistics.reset_index()
    return statistics, objects


def select_program(statistics: pd.DataFrame, training_datasets: list[str]) -> pd.DataFrame:
    columns = [f"r_{dataset}" for dataset in training_datasets]
    prevalence_columns = [f"prevalence_{dataset}" for dataset in training_datasets]
    values = statistics[columns].to_numpy(float)
    direction = np.where(np.all(values > 0, axis=1), "positive", np.where(np.all(values < 0, axis=1), "negative", "discordant"))
    minimum_abs = np.min(np.abs(values), axis=1)
    median_r = np.median(values, axis=1)
    training_minimum_prevalence = statistics[prevalence_columns].min(axis=1).to_numpy(float)
    selection_mask = (
        (direction != "discordant")
        & (minimum_abs >= MIN_DIRECTIONAL_CORRELATION)
        & (training_minimum_prevalence >= 0.03)
    )
    eligible = statistics.loc[
        selection_mask
    ].copy()
    eligible["training_direction"] = direction[selection_mask]
    eligible["training_min_abs_r"] = minimum_abs[selection_mask]
    eligible["training_median_r"] = median_r[selection_mask]
    eligible["training_minimum_prevalence"] = training_minimum_prevalence[selection_mask]
    selected = []
    for sign in ["positive", "negative"]:
        frame = eligible.loc[eligible["training_direction"].eq(sign)].sort_values(
            ["training_min_abs_r", "training_minimum_prevalence"], ascending=False
        ).head(PROGRAM_SIZE_PER_DIRECTION)
        selected.append(frame)
    result = pd.concat(selected, ignore_index=True)
    if result.groupby("training_direction").size().min() < 15:
        raise RuntimeError("Too few stable genes in one trajectory-program direction")
    return result


def program_score(
    adata: sc.AnnData,
    program: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    genes = program["gene"].tolist()
    matrix = dense(adata[mask, genes].X).astype(np.float64)
    subjects = adata.obs.loc[mask, "subject_id"].astype(str)
    mean, sd = weighted_mean_sd(matrix, subjects)
    standardized = (matrix - mean) / sd
    directions = program["training_direction"].to_numpy()
    magnitudes = program["training_min_abs_r"].to_numpy(float)
    scores = np.zeros(matrix.shape[0], dtype=float)
    for direction, multiplier in [("positive", 1.0), ("negative", -1.0)]:
        selected = directions == direction
        weights = magnitudes[selected]
        weights /= weights.sum()
        scores += multiplier * np.sum(standardized[:, selected] * weights[None, :], axis=1)
    percentiles = (stats.rankdata(scores, method="average") - 0.5) / len(scores)
    return scores, percentiles


def state_auc(obs: pd.DataFrame, score: np.ndarray) -> float:
    root = obs["transition_state"].astype(str).str.contains("classical_monocyte").to_numpy()
    terminal = obs["transition_state"].astype(str).eq("CSF_BAM_terminal").to_numpy()
    if not root.any() or not terminal.any():
        return np.nan
    return float(stats.mannwhitneyu(score[terminal], score[root]).statistic / (terminal.sum() * root.sum()))


def validate_program(
    statistics: pd.DataFrame,
    objects: dict[str, sc.AnnData],
) -> tuple[pd.DataFrame, float]:
    rows = []
    terminal_cutpoints = []
    for holdout in DISCOVERY:
        training = [dataset for dataset in DISCOVERY if dataset != holdout]
        program = select_program(statistics, training)
        adata = objects[holdout]
        mask = adata.obs["tissue"].astype(str).str.upper().eq("CSF").to_numpy()
        raw_score, score = program_score(adata, program, mask)
        obs = adata.obs.loc[mask]
        pt = obs["consensus_pseudotime"].to_numpy(float)
        rows.append({
            "validation": "leave_one_cohort_out",
            "holdout_dataset": holdout,
            "training_datasets": "+".join(training),
            "n_positive_genes": int(program["training_direction"].eq("positive").sum()),
            "n_negative_genes": int(program["training_direction"].eq("negative").sum()),
            "spearman_projected_vs_consensus": stats.spearmanr(score, pt).statistic,
            "terminal_vs_root_auc": state_auc(obs, score),
        })

    final_program = select_program(statistics, DISCOVERY)
    for dataset, adata in objects.items():
        mask = adata.obs["tissue"].astype(str).str.upper().eq("CSF").to_numpy()
        _, score = program_score(adata, final_program, mask)
        obs = adata.obs.loc[mask]
        root = obs["transition_state"].astype(str).str.contains("classical_monocyte").to_numpy()
        terminal = obs["transition_state"].astype(str).eq("CSF_BAM_terminal").to_numpy()
        labels = np.concatenate([np.zeros(root.sum()), np.ones(terminal.sum())])
        values = np.concatenate([score[root], score[terminal]])
        fpr, tpr, thresholds = roc_curve(labels, values)
        terminal_cutpoints.append(float(thresholds[np.argmax(tpr - fpr)]))
        rows.append({
            "validation": "final_program_internal_audit",
            "holdout_dataset": dataset,
            "training_datasets": "+".join(DISCOVERY),
            "n_positive_genes": int(final_program["training_direction"].eq("positive").sum()),
            "n_negative_genes": int(final_program["training_direction"].eq("negative").sum()),
            "spearman_projected_vs_consensus": stats.spearmanr(
                score, obs["consensus_pseudotime"].to_numpy(float)
            ).statistic,
            "terminal_vs_root_auc": state_auc(obs, score),
        })
    return pd.DataFrame(rows), float(np.median(terminal_cutpoints))


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]
    distinct = np.r_[True, np.diff(scores) != 0]
    tp = np.cumsum(labels == 1)[distinct]
    fp = np.cumsum(labels == 0)[distinct]
    thresholds = scores[distinct]
    tpr = tp / max(1, np.sum(labels == 1))
    fpr = fp / max(1, np.sum(labels == 0))
    return fpr, tpr, thresholds


def read_clinical_metadata(root: Path) -> pd.DataFrame:
    path = root / "data" / "external_validation" / "GSE328264" / "12974_2026_3861_MOESM1_ESM.xlsx"
    frame = pd.read_excel(path, sheet_name="final scRNA-seq", header=1)
    frame = frame.loc[frame["cohort"].notna()].copy()
    frame["subject_id"] = frame["sample"].astype(str).str.replace(r"^RMS_", "RRMS_", regex=True)
    frame = frame.drop_duplicates("subject_id", keep="first")
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["sex"] = frame["sex"].astype(str).str.lower()
    frame["sample_type"] = frame["sample type"].astype(str).str.lower()
    frame["dmt_at_sampling"] = pd.to_numeric(frame["DMT at sampling"], errors="coerce")
    frame.loc[frame["group"].eq("Ctrl"), "dmt_at_sampling"] = 0
    return frame[["subject_id", "age", "sex", "sample_type", "dmt_at_sampling"]]


def assign_external_states(obs: pd.DataFrame) -> pd.Series:
    fine = obs["fine_apc_label"].astype(str)
    stable = obs["bam_stable_core"].astype(bool)
    state = pd.Series("off_axis", index=obs.index, dtype="object")
    state[fine.eq("classical_monocyte")] = "CSF_classical_monocyte"
    state[fine.eq("APC_ambiguous")] = "CSF_APC_ambiguous"
    state[fine.eq("BAM_like_candidate") & ~stable] = "CSF_BAM_intermediate"
    state[stable] = "CSF_BAM_terminal"
    return state


def module_scores_control_standardized(
    adata: sc.AnnData,
    control_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    genes = sorted({gene for values in ACTIVATION_MODULES.values() for gene in values if gene in adata.var_names})
    matrix = dense(adata[:, genes].X).astype(np.float64)
    control_subjects = adata.obs.loc[control_mask, "subject_id"].astype(str)
    mean, sd = weighted_mean_sd(matrix[control_mask], control_subjects)
    standardized = (matrix - mean) / sd
    lookup = {gene: index for index, gene in enumerate(genes)}
    scores = {}
    for module, requested in ACTIVATION_MODULES.items():
        available = [gene for gene in requested if gene in lookup]
        scores[module] = standardized[:, [lookup[gene] for gene in available]].mean(axis=1)
        adata.uns[f"external_two_axis_{module}_genes"] = available
    return scores


def baseline_residual(
    outcome: np.ndarray,
    obs: pd.DataFrame,
    control_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    frame = pd.DataFrame({
        "pt": obs["projected_differentiation"].to_numpy(float),
        "log_counts": np.log1p(obs["total_counts"].to_numpy(float)),
        "n_genes": obs["n_genes_by_counts"].to_numpy(float),
    }, index=obs.index)
    for column in ["log_counts", "n_genes"]:
        subjects = obs.loc[control_mask, "subject_id"].astype(str)
        values = frame.loc[control_mask, column].to_numpy(float)[:, None]
        mean, sd = weighted_mean_sd(values, subjects)
        frame[f"{column}_z"] = (frame[column] - mean[0]) / sd[0]
    formula = (
        "bs(pt, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound=1) "
        "+ log_counts_z + n_genes_z"
    )
    control_design = patsy.dmatrix(formula, frame.loc[control_mask], return_type="dataframe")
    subjects = obs.loc[control_mask, "subject_id"].astype(str)
    counts = subjects.value_counts()
    weights = subjects.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    model = sm.WLS(outcome[control_mask], control_design, weights=weights).fit()
    all_design = patsy.build_design_matrices([control_design.design_info], frame, return_type="dataframe")[0]
    residual = np.asarray(outcome, dtype=float) - np.asarray(model.predict(all_design), dtype=float)
    center = weighted_quantile(residual[control_mask], subjects, 0.5)
    centered = residual[control_mask] - center
    _, scale_array = weighted_mean_sd(centered[:, None], subjects)
    scale = scale_array[0]
    standardized = (residual - center) / scale
    return standardized, {
        "control_n_cells": int(control_mask.sum()),
        "control_n_subjects": int(subjects.nunique()),
        "control_r_squared": float(model.rsquared),
        "control_residual_center": float(center),
        "control_residual_sd": float(scale),
    }


def prepare_external(
    root: Path,
    program: pd.DataFrame,
    terminal_cutpoint: float,
) -> tuple[sc.AnnData, pd.DataFrame, pd.DataFrame]:
    path = root / "results" / "external_validation" / "GSE328265" / "GSE328265_clean_annotated.h5ad"
    source = sc.read_h5ad(path)
    source.obs["transition_state"] = assign_external_states(source.obs)
    eligible = source.obs["transition_state"].ne("off_axis").to_numpy()
    adata = source[eligible].copy()
    adata.obs["transition_state"] = source.obs.loc[eligible, "transition_state"].to_numpy()
    clinical = read_clinical_metadata(root)
    clinical_subjects = set(clinical["subject_id"])
    adata.obs["publication_final_new"] = adata.obs["subject_id"].astype(str).isin(clinical_subjects)
    clinical_lookup = clinical.set_index("subject_id")
    for column in ["age", "sex", "sample_type", "dmt_at_sampling"]:
        adata.obs[column] = adata.obs["subject_id"].astype(str).map(clinical_lookup[column])
    sample_type_fallback = pd.Series(
        np.where(adata.obs["source_subseries"].astype(str).eq("GSE328264"), "fresh", "frozen"),
        index=adata.obs_names,
    )
    adata.obs["sample_type"] = adata.obs["sample_type"].fillna(sample_type_fallback)

    raw_score, projected = program_score(adata, program, np.ones(adata.n_obs, dtype=bool))
    adata.obs["trajectory_program_raw"] = raw_score
    adata.obs["projected_differentiation"] = projected
    adata.uns["external_terminal_score_cutpoint"] = terminal_cutpoint
    state_order = {
        "CSF_classical_monocyte": 0,
        "CSF_APC_ambiguous": 1,
        "CSF_BAM_intermediate": 2,
        "CSF_BAM_terminal": 3,
    }
    ordered = adata.obs["transition_state"].map(state_order).to_numpy(float)
    root = adata.obs["transition_state"].eq("CSF_classical_monocyte").to_numpy()
    terminal = adata.obs["transition_state"].eq("CSF_BAM_terminal").to_numpy()
    topology = pd.DataFrame([{
        "n_cells": adata.n_obs,
        "n_subjects": adata.obs["subject_id"].nunique(),
        "n_root": int(root.sum()),
        "n_terminal": int(terminal.sum()),
        "state_order_spearman": stats.spearmanr(projected, ordered).statistic,
        "terminal_vs_root_auc": float(
            stats.mannwhitneyu(projected[terminal], projected[root]).statistic / (terminal.sum() * root.sum())
        ),
        "terminal_score_cutpoint_from_discovery": terminal_cutpoint,
    }])

    control = adata.obs["diagnosis"].astype(str).eq("Ctrl").to_numpy()
    module_scores = module_scores_control_standardized(adata, control)
    diagnostics = []
    residuals = []
    for module in ["MHC_I", "IFN"]:
        adata.obs[f"external_{module}_z"] = module_scores[module]
        residual, diagnostic = baseline_residual(module_scores[module], adata.obs, control)
        adata.obs[f"external_{module}_residual_z"] = residual
        diagnostic["module"] = module
        diagnostics.append(diagnostic)
        residuals.append(residual)
    activation = np.mean(np.column_stack(residuals), axis=1)
    adata.obs["external_activation_residual"] = activation
    activation_threshold = weighted_quantile(
        activation[control], adata.obs.loc[control, "subject_id"].astype(str), 0.75
    )
    adata.uns["external_control_activation_q75"] = activation_threshold
    adata.obs["external_continuous_decoupling"] = activation * (1 - projected)
    adata.obs["external_decoupled_inflammatory"] = (
        (activation > activation_threshold) & (projected < INCOMPLETE_ADAPTATION_CUTOFF)
    )
    adata.obs["external_terminal_score_state"] = projected >= terminal_cutpoint
    adata.uns["trajectory_program_positive_genes"] = program.loc[
        program["training_direction"].eq("positive"), "gene"
    ].tolist()
    adata.uns["trajectory_program_negative_genes"] = program.loc[
        program["training_direction"].eq("negative"), "gene"
    ].tolist()
    return adata, topology, pd.DataFrame(diagnostics)


def aggregate_donors(adata: sc.AnnData) -> pd.DataFrame:
    rows = []
    for subject, frame in adata.obs.groupby("subject_id", observed=True):
        pt = frame["projected_differentiation"].to_numpy(float)
        activation = frame["external_activation_residual"].to_numpy(float)
        rows.append({
            "subject_id": str(subject),
            "diagnosis": str(frame["diagnosis"].iloc[0]),
            "disease_group": str(frame["disease_group"].iloc[0]),
            "publication_final_new": bool(frame["publication_final_new"].iloc[0]),
            "age": pd.to_numeric(frame["age"].iloc[0], errors="coerce"),
            "sex": str(frame["sex"].iloc[0]),
            "sample_type": str(frame["sample_type"].iloc[0]),
            "dmt_at_sampling": pd.to_numeric(frame["dmt_at_sampling"].iloc[0], errors="coerce"),
            "n_continuum_cells": len(frame),
            "n_strict_BAM": int(frame["bam_stable_core"].astype(bool).sum()),
            "median_projected_differentiation": float(np.median(pt)),
            "terminal_score_fraction": float(frame["external_terminal_score_state"].mean()),
            "strict_BAM_fraction": float(frame["bam_stable_core"].astype(bool).mean()),
            "mean_activation_residual": float(np.mean(activation)),
            "continuous_decoupling_index": float(np.mean(frame["external_continuous_decoupling"])),
            "decoupled_inflammatory_fraction": float(frame["external_decoupled_inflammatory"].mean()),
        })
    return pd.DataFrame(rows)


def permutation_test(case: np.ndarray, control: np.ndarray, seed: int) -> tuple[float, float, str]:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    pooled = np.concatenate([case, control])
    n_case = len(case)
    observed = float(case.mean() - control.mean())
    total = np.math.comb(len(pooled), n_case) if hasattr(np, "math") else __import__("math").comb(len(pooled), n_case)
    if total <= 100000:
        differences = []
        indices = np.arange(len(pooled))
        for combination in itertools.combinations(indices, n_case):
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
            difference = permuted[:n_case].mean() - permuted[n_case:].mean()
            exceedances += abs(difference) >= abs(observed) - 1e-12
        p_value = float((exceedances + 1) / 100001)
        method = "monte_carlo_100000"
    comparisons = case[:, None] - control[None, :]
    cliffs = float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)
    return p_value, cliffs, method


def stage_tests(donors: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    analysis_sets = {
        "all_new_QC": donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR),
        "publication_final_new": donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR) & donors["publication_final_new"],
        "frozen_only": donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR) & donors["sample_type"].eq("frozen"),
    }
    for analysis_set, mask in analysis_sets.items():
        frame = donors.loc[mask].copy()
        for contrast, (case_groups, control_groups) in CONTRASTS.items():
            for metric in DONOR_METRICS:
                case = frame.loc[frame["diagnosis"].isin(case_groups), metric].dropna().to_numpy(float)
                control = frame.loc[frame["diagnosis"].isin(control_groups), metric].dropna().to_numpy(float)
                if len(case) < 2 or len(control) < 2:
                    continue
                p_value, cliffs, method = permutation_test(case, control, seed)
                rows.append({
                    "analysis_set": analysis_set,
                    "contrast": contrast,
                    "case_groups": "+".join(case_groups),
                    "control_groups": "+".join(control_groups),
                    "metric": metric,
                    "n_case": len(case),
                    "n_control": len(control),
                    "mean_case": case.mean(),
                    "mean_control": control.mean(),
                    "mean_difference_case_minus_control": case.mean() - control.mean(),
                    "standardized_mean_difference": (
                        (case.mean() - control.mean())
                        / np.sqrt(
                            ((len(case) - 1) * case.var(ddof=1) + (len(control) - 1) * control.var(ddof=1))
                            / (len(case) + len(control) - 2)
                        )
                        if len(case) + len(control) > 2
                        and np.sqrt(
                            ((len(case) - 1) * case.var(ddof=1) + (len(control) - 1) * control.var(ddof=1))
                            / (len(case) + len(control) - 2)
                        ) > 0
                        else np.nan
                    ),
                    "cliffs_delta": cliffs,
                    "permutation_p": p_value,
                    "permutation_method": method,
                })
    tests = pd.DataFrame(rows)
    tests["fdr_bh_within_analysis_contrast"] = tests.groupby(
        ["analysis_set", "contrast"], observed=True
    )["permutation_p"].transform(lambda values: multipletests(values, method="fdr_bh")[1])
    return tests


def covariate_models(donors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = {
        "all_new_age_sex": (
            donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR) & donors["age"].notna(),
            "C(diagnosis, Treatment(reference='Ctrl')) + age + C(sex)",
        ),
        "publication_final_age_sex": (
            donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR)
            & donors["publication_final_new"]
            & donors["age"].notna(),
            "C(diagnosis, Treatment(reference='Ctrl')) + age + C(sex)",
        ),
        "frozen_age_sex": (
            donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR)
            & donors["sample_type"].eq("frozen")
            & donors["age"].notna(),
            "C(diagnosis, Treatment(reference='Ctrl')) + age + C(sex)",
        ),
        "MS_only_age_sex_DMT": (
            donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR)
            & donors["diagnosis"].isin(["RRMS", "PPMS", "SPMS"])
            & donors["age"].notna()
            & donors["dmt_at_sampling"].notna(),
            "C(diagnosis, Treatment(reference='RRMS')) + age + C(sex) + dmt_at_sampling",
        ),
        "MS_only_PPMS_reference_age_sex_DMT": (
            donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR)
            & donors["diagnosis"].isin(["RRMS", "PPMS", "SPMS"])
            & donors["age"].notna()
            & donors["dmt_at_sampling"].notna(),
            "C(diagnosis, Treatment(reference='PPMS')) + age + C(sex) + dmt_at_sampling",
        ),
        "frozen_MS_PPMS_reference_age_sex_DMT": (
            donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR)
            & donors["diagnosis"].isin(["RRMS", "PPMS", "SPMS"])
            & donors["sample_type"].eq("frozen")
            & donors["age"].notna()
            & donors["dmt_at_sampling"].notna(),
            "C(diagnosis, Treatment(reference='PPMS')) + age + C(sex) + dmt_at_sampling",
        ),
    }
    for specification, (mask, predictors) in specifications.items():
        frame = donors.loc[mask].copy()
        if len(frame) < 8:
            continue
        for metric in DONOR_METRICS:
            model_frame = frame[[metric, "diagnosis", "age", "sex", "dmt_at_sampling"]].copy()
            formula = f"{metric} ~ {predictors}"
            try:
                outcome, design = patsy.dmatrices(formula, model_frame, return_type="dataframe")
                model = sm.OLS(outcome, design).fit(cov_type="HC3")
            except (ValueError, np.linalg.LinAlgError):
                continue
            for term in design.columns:
                if not term.startswith("C(diagnosis"):
                    continue
                rows.append({
                    "specification": specification,
                    "metric": metric,
                    "term": term,
                    "n_donors": int(model.nobs),
                    "coefficient": float(model.params[term]),
                    "hc3_se": float(model.bse[term]),
                    "hc3_p": float(model.pvalues[term]),
                    "ci95_low": float(model.conf_int().loc[term, 0]),
                    "ci95_high": float(model.conf_int().loc[term, 1]),
                    "model_r_squared": float(model.rsquared),
                })
    models = pd.DataFrame(rows)
    if not models.empty:
        models["fdr_bh_within_specification"] = models.groupby("specification", observed=True)["hc3_p"].transform(
            lambda values: multipletests(values, method="fdr_bh")[1]
        )
    return models


def threshold_sensitivity(adata: sc.AnnData, donors: pd.DataFrame, seed: int) -> pd.DataFrame:
    control = adata.obs["diagnosis"].astype(str).eq("Ctrl")
    eligible_subjects = set(donors.loc[donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR), "subject_id"])
    rows = []
    for activation_quantile in [0.67, 0.75, 0.80, 0.90]:
        threshold = weighted_quantile(
            adata.obs.loc[control, "external_activation_residual"].to_numpy(float),
            adata.obs.loc[control, "subject_id"].astype(str),
            activation_quantile,
        )
        for differentiation_cutoff in [0.60, 0.70, 0.80, 0.90]:
            cell_state = (
                adata.obs["external_activation_residual"].to_numpy(float) > threshold
            ) & (
                adata.obs["projected_differentiation"].to_numpy(float) < differentiation_cutoff
            )
            frame = adata.obs[["subject_id", "diagnosis"]].copy()
            frame["state"] = cell_state
            subject_scores = frame.groupby(["subject_id", "diagnosis"], observed=True)["state"].mean().reset_index()
            subject_scores = subject_scores.loc[subject_scores["subject_id"].astype(str).isin(eligible_subjects)]
            for contrast in ["MS_vs_control", "PMS_vs_RRMS"]:
                case_groups, control_groups = CONTRASTS[contrast]
                case = subject_scores.loc[subject_scores["diagnosis"].isin(case_groups), "state"].to_numpy(float)
                comparator = subject_scores.loc[subject_scores["diagnosis"].isin(control_groups), "state"].to_numpy(float)
                p_value, cliffs, method = permutation_test(case, comparator, seed)
                rows.append({
                    "activation_control_quantile": activation_quantile,
                    "differentiation_cutoff": differentiation_cutoff,
                    "contrast": contrast,
                    "n_case": len(case),
                    "n_control": len(comparator),
                    "mean_difference_case_minus_control": case.mean() - comparator.mean(),
                    "cliffs_delta": cliffs,
                    "permutation_p": p_value,
                    "permutation_method": method,
                })
    sensitivity = pd.DataFrame(rows)
    sensitivity["fdr_bh_within_contrast"] = sensitivity.groupby("contrast", observed=True)["permutation_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return sensitivity


def donor_jackknife(donors: pd.DataFrame) -> pd.DataFrame:
    frame = donors.loc[donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR)].copy()
    rows = []
    for contrast in ["MS_vs_control", "PMS_vs_RRMS", "SPMS_vs_RRMS"]:
        case_groups, control_groups = CONTRASTS[contrast]
        relevant = frame.loc[frame["diagnosis"].isin(case_groups + control_groups)]
        for metric in ["continuous_decoupling_index", "decoupled_inflammatory_fraction"]:
            for omitted in relevant["subject_id"]:
                subset = relevant.loc[relevant["subject_id"].ne(omitted)]
                case = subset.loc[subset["diagnosis"].isin(case_groups), metric]
                control = subset.loc[subset["diagnosis"].isin(control_groups), metric]
                rows.append({
                    "contrast": contrast,
                    "metric": metric,
                    "omitted_subject": omitted,
                    "mean_difference_case_minus_control": case.mean() - control.mean(),
                })
    return pd.DataFrame(rows)


def three_cohort_meta(root: Path, external_tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = pd.read_csv(root / "results" / "two_axis" / "two_axis_disease_tests.csv")
    endpoints = ["continuous_decoupling_index", "decoupled_inflammatory_fraction"]
    rows = []
    for endpoint in endpoints:
        source = discovery.loc[discovery["metric"].eq(endpoint)]
        for row in source.itertuples():
            rows.append({
                "endpoint": endpoint,
                "cohort": row.dataset,
                "n_case": row.n_MS,
                "n_control": row.n_control,
                "effect": row.mean_difference_MS_minus_control,
                "standardized_effect": row.standardized_mean_difference,
                "p_value": row.exact_permutation_p,
                "source": "discovery",
            })
        external = external_tests.loc[
            external_tests["analysis_set"].eq("all_new_QC")
            & external_tests["contrast"].eq("MS_vs_control")
            & external_tests["metric"].eq(endpoint)
        ].iloc[0]
        rows.append({
            "endpoint": endpoint,
            "cohort": "GSE328265_new_only",
            "n_case": external.n_case,
            "n_control": external.n_control,
            "effect": external.mean_difference_case_minus_control,
            "standardized_effect": external.standardized_mean_difference,
            "p_value": external.permutation_p,
            "source": "external_projection",
        })
    cohort_results = pd.DataFrame(rows)
    meta_rows = []
    for endpoint, frame in cohort_results.groupby("endpoint", observed=True):
        signs = np.sign(frame["effect"].to_numpy(float))
        p_values = np.clip(frame["p_value"].to_numpy(float), 1e-12, 1)
        z_scores = signs * stats.norm.isf(p_values / 2)
        weights = np.sqrt((frame["n_case"] + frame["n_control"]).to_numpy(float))
        combined_z = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
        meta_rows.append({
            "endpoint": endpoint,
            "n_cohorts": len(frame),
            "direction_concordant": bool(len(set(signs)) == 1),
            "combined_z": combined_z,
            "stouffer_p": float(2 * stats.norm.sf(abs(combined_z))),
            "mean_standardized_effect": frame["standardized_effect"].mean(),
        })
    meta = pd.DataFrame(meta_rows)
    meta["fdr_bh"] = multipletests(meta["stouffer_p"], method="fdr_bh")[1]
    return cohort_results, meta.sort_values("stouffer_p")


def write_report(
    path: Path,
    program: pd.DataFrame,
    validation: pd.DataFrame,
    topology: pd.DataFrame,
    diagnostics: pd.DataFrame,
    donors: pd.DataFrame,
    tests: pd.DataFrame,
    covariates: pd.DataFrame,
    sensitivity: pd.DataFrame,
    jackknife: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    topo = topology.iloc[0]
    primary = tests.loc[
        tests["analysis_set"].eq("all_new_QC")
        & tests["metric"].isin(["continuous_decoupling_index", "decoupled_inflammatory_fraction"])
    ].sort_values(["contrast", "metric"])
    lines = [
        "# GSE328265 frozen-program two-axis projection",
        "",
        "## Frozen projection design",
        "",
        f"The differentiation program contains {program['training_direction'].eq('positive').sum()} positive and "
        f"{program['training_direction'].eq('negative').sum()} negative genes concordant across all three discovery cohorts. "
        "MHC-I, IFN, HLA, technical, stress, sex, cell-cycle, and annotation-marker genes were excluded before selection.",
        "",
        "The external cohort was not used to choose genes or directions. Its control donors define only the expected "
        "activation-versus-differentiation manifold and the control activation threshold.",
        f"The binary joint-state endpoint keeps the discovery definition of incomplete adaptation "
        f"(projected differentiation < {INCOMPLETE_ADAPTATION_CUTOFF:.2f}); the continuous endpoint is threshold-free.",
        "",
        "## Projection audits",
        "",
    ]
    for row in validation.loc[validation["validation"].eq("leave_one_cohort_out")].itertuples():
        lines.append(
            f"- Hold out {row.holdout_dataset}: projected-versus-consensus r={row.spearman_projected_vs_consensus:.3f}; "
            f"terminal-versus-root AUC={row.terminal_vs_root_auc:.3f}."
        )
    lines.extend([
        f"- GSE328265: {int(topo.n_cells)} continuum cells from {int(topo.n_subjects)} subjects; "
        f"state-order r={topo.state_order_spearman:.3f}; terminal-versus-root AUC={topo.terminal_vs_root_auc:.3f}.",
        "",
        "## Control-manifold fits",
        "",
    ])
    for row in diagnostics.itertuples():
        lines.append(
            f"- {row.module}: {row.control_n_cells} cells from {row.control_n_subjects} control subjects; "
            f"weighted baseline R2={row.control_r_squared:.3f}."
        )
    lines.extend(["", "## Donor-level stage tests", ""])
    for row in primary.itertuples():
        lines.append(
            f"- {row.contrast}, {row.metric}: difference={row.mean_difference_case_minus_control:.4f}, "
            f"P={row.permutation_p:.4g}, FDR={row.fdr_bh_within_analysis_contrast:.4g}, "
            f"n={row.n_case} versus {row.n_control}."
        )
    lines.extend(["", "## Three-cohort synthesis", ""])
    for row in meta.itertuples():
        lines.append(
            f"- {row.endpoint}: {row.n_cohorts} cohorts, direction concordant={row.direction_concordant}; "
            f"Stouffer z={row.combined_z:.3f}, P={row.stouffer_p:.4g}, FDR={row.fdr_bh:.4g}."
        )
    subtype = tests.loc[
        tests["analysis_set"].eq("all_new_QC")
        & tests["contrast"].eq("SPMS_vs_PPMS")
        & tests["metric"].isin(["median_projected_differentiation", "terminal_score_fraction"])
    ].sort_values("metric")
    lines.extend(["", "## Progressive-subtype branch", ""])
    for row in subtype.itertuples():
        lines.append(
            f"- {row.metric}: SPMS-minus-PPMS={row.mean_difference_case_minus_control:.4f}, "
            f"Cliff delta={row.cliffs_delta:.3f}, P={row.permutation_p:.4g}, "
            f"FDR={row.fdr_bh_within_analysis_contrast:.4g}."
        )
    eligible = donors.loc[donors["n_continuum_cells"].ge(MIN_CELLS_PER_DONOR)]
    lines.extend([
        "",
        "## Robustness layers",
        "",
        f"- Donor eligibility: {len(eligible)} subjects with at least {MIN_CELLS_PER_DONOR} continuum cells; "
        f"{eligible['publication_final_new'].sum()} also belong to the publication-final newly profiled cohort.",
        f"- Threshold grid: {(sensitivity['mean_difference_case_minus_control'] > 0).sum()} of {len(sensitivity)} "
        "contrast-threshold combinations have positive case-minus-control direction.",
        f"- Donor jackknife: {(jackknife['mean_difference_case_minus_control'] > 0).sum()} of {len(jackknife)} "
        "leave-one-donor effects retain a positive direction.",
        "- Age/sex, DMT, publication-eligibility, and frozen-only estimates are provided as separate tables rather "
        "than being folded into the primary exact permutation test.",
        "",
        "## Interpretation rule",
        "",
        "The external endpoint is the joint state, not BAM abundance or antigen-presentation expression alone. "
        "A positive continuous decoupling effect identifies activation that is concentrated toward the incompletely "
        "adapted side of the frozen monocyte-to-border-macrophage program.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results" / "external_validation" / "GSE328265_two_axis_projection"
    output.mkdir(parents=True, exist_ok=True)

    print("Deriving cross-cohort trajectory program", flush=True)
    statistics, objects = derive_program_statistics(root)
    program = select_program(statistics, DISCOVERY)
    validation, terminal_cutpoint = validate_program(statistics, objects)
    statistics["selected_final_program"] = statistics["gene"].isin(program["gene"])
    statistics.to_csv(output / "trajectory_program_gene_statistics.csv", index=False)
    program.to_csv(output / "frozen_trajectory_program.csv", index=False)
    validation.to_csv(output / "trajectory_program_validation.csv", index=False)

    print("Projecting GSE328265", flush=True)
    external, topology, diagnostics = prepare_external(root, program, terminal_cutpoint)
    donors = aggregate_donors(external)
    tests = stage_tests(donors, args.seed)
    covariates = covariate_models(donors)
    sensitivity = threshold_sensitivity(external, donors, args.seed)
    jackknife = donor_jackknife(donors)
    cohort_meta_inputs, meta = three_cohort_meta(root, tests)

    topology.to_csv(output / "external_projection_topology.csv", index=False)
    diagnostics.to_csv(output / "external_control_manifold_diagnostics.csv", index=False)
    donors.to_csv(output / "GSE328265_two_axis_donor_scores.csv", index=False)
    tests.to_csv(output / "GSE328265_two_axis_stage_tests.csv", index=False)
    covariates.to_csv(output / "GSE328265_two_axis_covariate_models.csv", index=False)
    sensitivity.to_csv(output / "GSE328265_decoupling_threshold_sensitivity.csv", index=False)
    jackknife.to_csv(output / "GSE328265_two_axis_donor_jackknife.csv", index=False)
    cohort_meta_inputs.to_csv(output / "three_cohort_two_axis_meta_inputs.csv", index=False)
    meta.to_csv(output / "three_cohort_two_axis_meta.csv", index=False)
    external.obs.to_csv(output / "GSE328265_two_axis_cell_scores.csv.gz", compression="gzip")
    external.write_h5ad(output / "GSE328265_monocyte_BAM_two_axis.h5ad", compression="gzip")
    write_report(
        output / "GSE328265_TWO_AXIS_PROJECTION_REPORT.md",
        program,
        validation,
        topology,
        diagnostics,
        donors,
        tests,
        covariates,
        sensitivity,
        jackknife,
        meta,
    )
    print(tests.loc[
        tests["analysis_set"].eq("all_new_QC")
        & tests["metric"].isin(["continuous_decoupling_index", "decoupled_inflammatory_fraction"])
    ].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
