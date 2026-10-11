#!/usr/bin/env python3
"""Infer donor-level T-cell niche to myeloid-state regulatory chains."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import decoupler as dc
import liana as li
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


SEED = 12345
MIN_STATE_CELLS = 5
PERMUTATIONS = 20000

DATASETS = {
    "GSE138266": {
        "continuum": "results/trajectory/continuum_objects/GSE138266_monocyte_BAM_continuum.h5ad",
        "full": "results/cleaned/GSE138266_clean_decontx_singlets.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
    },
    "GSE194078": {
        "continuum": "results/trajectory/continuum_objects/GSE194078_monocyte_BAM_continuum.h5ad",
        "full": "results/cleaned/GSE194078_clean_decontx_singlets.h5ad",
        "pt": "consensus_pseudotime",
        "activation": None,
        "threshold": None,
    },
    "GSE286068": {
        "continuum": "results/trajectory/continuum_objects/GSE286068_monocyte_BAM_continuum.h5ad",
        "full": "results/cleaned/GSE286068_clean_decontx_singlets.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
    },
    "GSE328265": {
        "continuum": "results/external_validation/GSE328265_two_axis_projection/GSE328265_monocyte_BAM_two_axis.h5ad",
        "full": "results/external_validation/GSE328265/GSE328265_clean_annotated.h5ad",
        "pt": "projected_differentiation",
        "activation": "external_activation_residual",
        "threshold": "external_control_activation_q75",
    },
}

AXIS2_GENES = {
    "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7",
}

LAM_GENES = {
    "ANXA1", "APOE", "APP", "CAPG", "CD14", "CD68", "CXCL14", "CXCR4",
    "FCER1G", "GPNMB", "HILPDA", "IDH1", "IL7R", "ITGB2", "MGST1", "PSEN2",
    "RFTN1", "TGFBI", "THBS1", "C1QA", "C1QB", "C1QC", "SELENOP", "TREM2",
    "MERTK", "HLA-DRA", "CD74", "CIITA", "ARG1", "CXCL2", "CLEC7A", "SPP1",
    "CCL24", "NOS2", "FN1", "VEGFA", "ZEB2", "HIF1A", "CXCL10", "CXCL9",
    "TGM2", "TGFB3", "PLAC8", "IRF1", "CSF2RB",
}

REGULATORS = {
    "IRF1": "KO", "STAT1": "KO", "STAT3": "KO", "STAT5A": "KO", "STAT5B": "KO",
    "NFKB1": "KO", "NFKB2": "KO", "RELA": "KO", "RELB": "KO", "CEBPB": "KO",
    "BHLHE40": "KO", "HIF1A": "KO", "ZEB2": "KO",
    "SPI1": "OE", "MAF": "OE", "MAFB": "OE", "PPARG": "OE", "NR1H3": "OE",
    "TFEB": "OE", "BHLHE41": "OE",
}

CIRCUITS = {
    "CSF2": {
        "ligand_components": ["CSF2"],
        "receptor_components": ["CSF2RA", "CSF2RB"],
        "liana_ligand": "CSF2",
        "liana_receptor": "CSF2RA_CSF2RB",
        "regulators": ["STAT5A", "STAT5B", "STAT3", "IRF1", "NFKB1", "RELA"],
    },
    "IFNG": {
        "ligand_components": ["IFNG"],
        "receptor_components": ["IFNGR1", "IFNGR2"],
        "liana_ligand": "IFNG",
        "liana_receptor": "IFNGR1_IFNGR2",
        "regulators": ["STAT1", "IRF1"],
    },
    "TNF": {
        "ligand_components": ["TNF"],
        "receptor_components": ["TNFRSF1A", "TNFRSF1B"],
        "liana_ligand": "TNF",
        "liana_receptor": "TNFRSF1A",
        "regulators": ["NFKB1", "NFKB2", "RELA", "RELB", "CEBPB"],
    },
    "LTA_LTB": {
        "ligand_components": ["LTA", "LTB"],
        "receptor_components": ["LTBR"],
        "liana_ligand": "LTA_LTB",
        "liana_receptor": "LTBR",
        "regulators": ["NFKB2", "RELB"],
    },
    "CCL5": {
        "ligand_components": ["CCL5"],
        "receptor_components": ["CCR1", "CCR5"],
        "liana_ligand": "CCL5",
        "liana_receptor": "CCR1",
        "regulators": ["NFKB1", "RELA", "CEBPB"],
    },
}

LAM_OUTCOMES = [
    "decoupled_excess__mouse_GMCSF_LAM_precursor",
    "decoupled_excess__mouse_LAM2_complement_APC",
    "decoupled_excess__mouse_LAM4_fibrotic",
    "decoupled_excess__LAM4_minus_LAM2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def rank_normal(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="average")
    quantiles = (ranked - 0.5) / len(values)
    return pd.Series(stats.norm.ppf(quantiles), index=values.index)


def standardize(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-8:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / sd


def disease_class(values: pd.Series) -> pd.Series:
    controls = {"IIH_control", "non_inflammatory_control", "control"}
    output = pd.Series("other_inflammatory", index=values.index, dtype="object")
    output.loc[values.astype(str).isin(controls)] = "control"
    output.loc[values.astype(str).isin({"MS", "RRMS", "PPMS", "SPMS"})] = "MS"
    return output


def raw_gene_slice(adata: sc.AnnData, mask: np.ndarray, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    source = adata.raw if adata.raw is not None else adata
    available = [gene for gene in genes if gene in source.var_names]
    cell_indices = np.flatnonzero(mask)
    gene_indices = source.var_names.get_indexer(available)
    matrix = source.X[cell_indices, :]
    matrix = matrix[:, gene_indices]
    return dense(matrix), available


def build_lr_ledger() -> pd.DataFrame:
    consensus = li.rs.select_resource("consensus")
    rows = []
    for circuit, details in CIRCUITS.items():
        exact = (
            consensus["ligand"].astype(str).eq(details["liana_ligand"])
            & consensus["receptor"].astype(str).eq(details["liana_receptor"])
        )
        rows.append({
            "circuit": circuit,
            "ligand_components": ";".join(details["ligand_components"]),
            "receptor_components": ";".join(details["receptor_components"]),
            "candidate_regulators": ";".join(details["regulators"]),
            "liana_resource": "consensus",
            "liana_ligand_key": details["liana_ligand"],
            "liana_receptor_key": details["liana_receptor"],
            "exact_pair_in_liana_consensus": bool(exact.any()),
        })
    return pd.DataFrame(rows)


def build_sender_scores(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = sorted({gene for details in CIRCUITS.values() for gene in details["ligand_components"]})
    gene_rows = []
    score_tables = []
    for dataset, config in DATASETS.items():
        adata = sc.read_h5ad(root / config["full"], backed="r")
        t_mask = (
            adata.obs["broad_cell_type"].astype(str).eq("T").to_numpy()
            & adata.obs["tissue"].astype(str).str.upper().eq("CSF").to_numpy()
        )
        matrix, available = raw_gene_slice(adata, t_mask, genes)
        obs = adata.obs.loc[t_mask, ["subject_id", "diagnosis", "disease_group"]].copy()
        obs["subject_id"] = obs["subject_id"].astype(str)
        lookup = {gene: index for index, gene in enumerate(available)}
        donor_rows = []
        for subject, indices in obs.groupby("subject_id", observed=True).indices.items():
            positions = np.asarray(indices, dtype=int)
            row = {
                "dataset": dataset,
                "subject_id": subject,
                "diagnosis": str(obs.iloc[positions[0]]["diagnosis"]),
                "disease_group": str(obs.iloc[positions[0]]["disease_group"]),
                "n_CSF_T_cells": len(positions),
            }
            for gene in genes:
                values = matrix[positions, lookup[gene]] if gene in lookup else np.full(len(positions), np.nan)
                mean = float(np.nanmean(values))
                detected = float(np.nanmean(values > 0))
                row[f"mean__{gene}"] = mean
                row[f"detected_fraction__{gene}"] = detected
                gene_rows.append({
                    "dataset": dataset,
                    "subject_id": subject,
                    "gene": gene,
                    "n_CSF_T_cells": len(positions),
                    "mean_log_expression": mean,
                    "detected_fraction": detected,
                })
            donor_rows.append(row)
        donor = pd.DataFrame(donor_rows)
        for circuit, details in CIRCUITS.items():
            expression_columns = [f"mean__{gene}" for gene in details["ligand_components"]]
            detection_columns = [f"detected_fraction__{gene}" for gene in details["ligand_components"]]
            donor[f"sender_mean__{circuit}"] = donor[expression_columns].min(axis=1)
            donor[f"sender_detection__{circuit}"] = donor[detection_columns].min(axis=1)
            donor[f"sender_score__{circuit}"] = rank_normal(donor[f"sender_mean__{circuit}"])
        core = ["CSF2", "IFNG", "TNF", "LTA_LTB"]
        donor["sender_score__cytokine_ensemble"] = donor[
            [f"sender_score__{circuit}" for circuit in core]
        ].mean(axis=1)
        donor["sender_score__all_ligand_ensemble"] = donor[
            [f"sender_score__{circuit}" for circuit in CIRCUITS]
        ].mean(axis=1)
        donor["disease_class"] = disease_class(donor["disease_group"])
        score_tables.append(donor)
        adata.file.close()
    return pd.concat(score_tables, ignore_index=True), pd.DataFrame(gene_rows)


def network_and_audit(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.read_csv(root / "data/reference/dorothea_human_ABC.csv.gz")
    original = original.loc[original["source"].isin(REGULATORS)].copy()
    excluded = AXIS2_GENES | LAM_GENES
    rows = []
    for regulator in REGULATORS:
        frame = original.loc[original["source"].eq(regulator)]
        rows.append({
            "regulator": regulator,
            "recommended_perturbation": REGULATORS[regulator],
            "n_targets_original": len(frame),
            "n_axis2_targets_removed": int(frame["target"].isin(AXIS2_GENES).sum()),
            "n_LAM_targets_removed": int(frame["target"].isin(LAM_GENES).sum()),
            "n_targets_after_exclusion": int((~frame["target"].isin(excluded)).sum()),
            "activity_status": "DoRothEA_ULM" if len(frame) >= 5 else "expression_proxy_only",
        })
    filtered = original.loc[~original["target"].isin(excluded)].copy()
    retained = filtered.groupby("source", observed=True).size()
    filtered = filtered.loc[filtered["source"].map(retained).ge(5)].copy()
    return filtered[["source", "target", "weight"]], pd.DataFrame(rows)


def state_masks(obs: pd.DataFrame, pt: str, activation: str | None, threshold: float | None) -> dict[str, np.ndarray]:
    pseudotime = obs[pt].to_numpy(float)
    masks = {
        "early": pseudotime < 0.4,
        "transitional": (pseudotime >= 0.4) & (pseudotime < 0.8),
        "terminal": pseudotime >= 0.8,
    }
    if activation is not None:
        high = obs[activation].to_numpy(float) > float(threshold)
        incomplete = pseudotime < 0.8
        masks["other_incomplete"] = incomplete & ~high
        masks["decoupled"] = incomplete & high
    return masks


def aggregate_receiver_dataset(
    root: Path,
    dataset: str,
    config: dict,
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adata = sc.read_h5ad(root / config["continuum"])
    if "tissue" in adata.obs:
        adata = adata[adata.obs["tissue"].astype(str).str.upper().eq("CSF")].copy()
    threshold = adata.uns[config["threshold"]] if config["threshold"] is not None else None
    masks = state_masks(adata.obs, config["pt"], config["activation"], threshold)
    subjects = adata.obs["subject_id"].astype(str).to_numpy()
    matrix = adata.X
    sample_rows = []
    expression_rows = []
    for subject in sorted(np.unique(subjects)):
        subject_mask = subjects == subject
        subject_obs = adata.obs.loc[subject_mask]
        for state, state_mask in masks.items():
            selected = subject_mask & state_mask
            if selected.sum() < MIN_STATE_CELLS:
                continue
            sample_id = f"{dataset}::{subject}::{state}"
            sample_rows.append({
                "sample_id": sample_id,
                "dataset": dataset,
                "subject_id": subject,
                "diagnosis": str(subject_obs["diagnosis"].iloc[0]),
                "disease_group": str(subject_obs["disease_group"].iloc[0]),
                "state": state,
                "n_cells": int(selected.sum()),
                "mean_pseudotime": float(adata.obs.loc[selected, config["pt"]].mean()),
            })
            expression_rows.append(np.asarray(matrix[selected].mean(axis=0)).ravel())
    metadata = pd.DataFrame(sample_rows).set_index("sample_id")
    expression = pd.DataFrame(np.vstack(expression_rows), index=metadata.index, columns=adata.var_names)

    activity_network = network.loc[network["target"].isin(expression.columns)].copy()
    scores, _ = dc.mt.ulm(expression, activity_network, tmin=5)
    scores = scores.reindex(metadata.index)
    score_columns = {}
    for regulator in REGULATORS:
        if regulator in scores:
            score_columns[f"activity__{regulator}"] = standardize(scores[regulator])
        if regulator in expression:
            score_columns[f"expression__{regulator}"] = standardize(expression[regulator])
    features = pd.DataFrame(score_columns, index=metadata.index)

    receptor_rows = []
    for circuit, details in CIRCUITS.items():
        available = [gene for gene in details["receptor_components"] if gene in expression]
        if len(available) != len(details["receptor_components"]):
            receptor = pd.Series(np.nan, index=expression.index)
        else:
            receptor = expression[available].min(axis=1)
        features[f"receptor__{circuit}"] = standardize(receptor) if receptor.notna().all() else receptor
        receptor_rows.append({
            "dataset": dataset,
            "circuit": circuit,
            "requested_receptors": ";".join(details["receptor_components"]),
            "available_receptors": ";".join(available),
            "all_components_available": len(available) == len(details["receptor_components"]),
        })

    output = metadata.join(features).reset_index()
    output["disease_class"] = disease_class(output["disease_group"])
    feature_columns = [column for column in output if "__" in column]
    long = output.melt(
        id_vars=[
            "sample_id", "dataset", "subject_id", "diagnosis", "disease_group",
            "disease_class", "state", "n_cells", "mean_pseudotime",
        ],
        value_vars=feature_columns,
        var_name="feature",
        value_name="value",
    )
    long[["feature_type", "feature_name"]] = long["feature"].str.split("__", n=1, expand=True)
    del adata
    return output, long, pd.DataFrame(receptor_rows)


def sign_flip_test(values: np.ndarray, seed: int) -> tuple[float, str]:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    total = 2 ** len(values)
    if total <= 100000:
        means = [np.mean(values * np.asarray(signs)) for signs in itertools.product([-1.0, 1.0], repeat=len(values))]
        return float(np.mean(np.abs(means) >= observed - 1e-12)), "exact_sign_flip"
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(100000, len(values)))
    means = np.mean(signs * values[None, :], axis=1)
    return float((np.sum(np.abs(means) >= observed - 1e-12) + 1) / 100001), "mc_sign_flip_100000"


def state_deltas(state_long: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrasts = {
        "transitional_minus_early": ("transitional", "early"),
        "terminal_minus_transitional": ("terminal", "transitional"),
        "terminal_minus_early": ("terminal", "early"),
        "decoupled_minus_other_incomplete": ("decoupled", "other_incomplete"),
    }
    rows = []
    keys = ["dataset", "subject_id", "diagnosis", "disease_group", "disease_class", "feature_type", "feature_name"]
    for key, frame in state_long.groupby(keys, observed=True):
        values = frame.drop_duplicates("state").set_index("state")["value"]
        cell_counts = frame.drop_duplicates("state").set_index("state")["n_cells"]
        for contrast, (case_state, control_state) in contrasts.items():
            if case_state not in values or control_state not in values:
                continue
            rows.append({
                **dict(zip(keys, key)),
                "contrast": contrast,
                "delta": float(values[case_state] - values[control_state]),
                "n_case_state_cells": int(cell_counts[case_state]),
                "n_control_state_cells": int(cell_counts[control_state]),
            })
    deltas = pd.DataFrame(rows)
    test_rows = []
    for key, frame in deltas.groupby(["dataset", "contrast", "feature_type", "feature_name"], observed=True):
        values = frame["delta"].dropna().to_numpy(float)
        if len(values) < 3:
            continue
        p_value, method = sign_flip_test(values, seed)
        test_rows.append({
            "dataset": key[0], "contrast": key[1], "feature_type": key[2], "feature_name": key[3],
            "n_donors": len(values), "mean_delta": values.mean(), "median_delta": np.median(values),
            "standardized_effect": values.mean() / (values.std(ddof=1) if len(values) > 1 and values.std(ddof=1) > 0 else 1),
            "positive_fraction": np.mean(values > 0), "sign_flip_p": p_value, "test_method": method,
        })
    tests = pd.DataFrame(test_rows)
    tests["fdr_bh_within_dataset_contrast"] = tests.groupby(
        ["dataset", "contrast"], observed=True
    )["sign_flip_p"].transform(lambda values: multipletests(values, method="fdr_bh")[1])

    meta_rows = []
    for key, frame in tests.groupby(["contrast", "feature_type", "feature_name"], observed=True):
        signs = np.sign(frame["mean_delta"].to_numpy(float))
        p_values = np.clip(frame["sign_flip_p"].to_numpy(float), 1e-12, 1)
        z_scores = signs * stats.norm.isf(p_values / 2)
        weights = np.sqrt(frame["n_donors"].to_numpy(float))
        combined_z = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
        meta_rows.append({
            "contrast": key[0], "feature_type": key[1], "feature_name": key[2],
            "n_cohorts": len(frame), "n_donors": int(frame["n_donors"].sum()),
            "direction_concordant": bool(len(set(signs)) == 1),
            "mean_standardized_effect": frame["standardized_effect"].mean(),
            "combined_z": combined_z, "stouffer_p": float(2 * stats.norm.sf(abs(combined_z))),
        })
    meta = pd.DataFrame(meta_rows)
    meta["fdr_bh_within_contrast"] = meta.groupby("contrast", observed=True)["stouffer_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    return deltas, tests, meta


def residualize_ranks(frame: pd.DataFrame, column: str) -> np.ndarray:
    ranked = stats.rankdata(frame[column].to_numpy(float), method="average")
    design = pd.get_dummies(frame["stratum"], drop_first=False, dtype=float).to_numpy()
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def pooled_partial_test(frame: pd.DataFrame, x: str, y: str, permutations: int, seed: int) -> tuple[float, float]:
    use = frame[["dataset", "disease_class", x, y]].dropna().copy()
    use["stratum"] = use["dataset"].astype(str) + ":" + use["disease_class"].astype(str)
    x_residual = residualize_ranks(use, x)
    y_residual = residualize_ranks(use, y)
    observed = float(stats.pearsonr(x_residual, y_residual).statistic)
    groups = [np.flatnonzero(use["stratum"].to_numpy() == value) for value in use["stratum"].unique()]
    denominator = np.sqrt(np.sum(x_residual**2) * np.sum(y_residual**2))
    rng = np.random.default_rng(seed)
    exceed = 0
    batch_size = 2000
    for start in range(0, permutations, batch_size):
        size = min(batch_size, permutations - start)
        permuted = np.tile(y_residual, (size, 1))
        for indices in groups:
            if len(indices) < 2:
                continue
            orders = np.argsort(rng.random((size, len(indices))), axis=1)
            permuted[:, indices] = y_residual[indices][orders]
        correlations = permuted @ x_residual / denominator
        exceed += int(np.sum(np.abs(correlations) >= abs(observed) - 1e-12))
    return observed, float((exceed + 1) / (permutations + 1))


def association_table(
    frame: pd.DataFrame,
    x_columns: list[str],
    y_columns: list[str],
    permutations: int,
    seed: int,
    family: str,
    prespecified_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    counter = 0
    pairs = prespecified_pairs if prespecified_pairs is not None else list(itertools.product(x_columns, y_columns))
    for x, y in pairs:
        subset = frame[["dataset", "disease_class", x, y]].dropna()
        if len(subset) < 12 or subset["dataset"].nunique() < 2:
            continue
        correlation, p_value = pooled_partial_test(subset, x, y, permutations, seed + counter)
        rows.append({
            "family": family, "scope": "pooled_dataset_disease_residualized",
            "x": x, "y": y, "n_donors": len(subset), "n_cohorts": subset["dataset"].nunique(),
            "correlation": correlation, "permutation_p": p_value,
            "method": f"partial_rank_block_permutation_{permutations}",
        })
        for dataset, dataset_frame in subset.groupby("dataset", observed=True):
            if len(dataset_frame) < 5 or dataset_frame[x].nunique() < 3 or dataset_frame[y].nunique() < 3:
                continue
            result = stats.spearmanr(dataset_frame[x], dataset_frame[y])
            rows.append({
                "family": family, "scope": "within_dataset", "dataset": dataset,
                "x": x, "y": y, "n_donors": len(dataset_frame), "n_cohorts": 1,
                "correlation": result.statistic, "permutation_p": result.pvalue, "method": "spearman",
            })
        counter += 1
    output = pd.DataFrame(rows)
    if len(output):
        pooled = output["scope"].eq("pooled_dataset_disease_residualized")
        output["fdr_bh_within_family"] = np.nan
        output.loc[pooled, "fdr_bh_within_family"] = multipletests(
            output.loc[pooled, "permutation_p"], method="fdr_bh"
        )[1]
    return output


def build_candidate_ranking(
    meta: pd.DataFrame,
    delta_outcome_associations: pd.DataFrame,
    sender_tf_associations: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    dec = meta.loc[meta["contrast"].eq("decoupled_minus_other_incomplete")].copy()
    term = meta.loc[meta["contrast"].eq("terminal_minus_early")].copy()
    pooled_delta = delta_outcome_associations.loc[
        delta_outcome_associations["scope"].eq("pooled_dataset_disease_residualized")
        & delta_outcome_associations["y"].eq("decoupled_excess__mouse_GMCSF_LAM_precursor")
    ].copy()
    pooled_sender = sender_tf_associations.loc[
        sender_tf_associations["scope"].eq("pooled_dataset_disease_residualized")
    ].copy()
    rows = []
    for regulator, perturbation in REGULATORS.items():
        preferred_type = "activity" if (
            (audit["regulator"].eq(regulator) & audit["activity_status"].eq("DoRothEA_ULM")).any()
        ) else "expression"
        dec_row = dec.loc[(dec["feature_type"].eq(preferred_type)) & dec["feature_name"].eq(regulator)]
        term_row = term.loc[(term["feature_type"].eq(preferred_type)) & term["feature_name"].eq(regulator)]
        feature = f"{preferred_type}__{regulator}"
        precursor = pooled_delta.loc[pooled_delta["x"].eq(feature)]
        canonical = [circuit for circuit, details in CIRCUITS.items() if regulator in details["regulators"]]
        sender = pooled_sender.loc[
            pooled_sender["y"].eq(feature)
            & pooled_sender["x"].isin([f"sender_score__{circuit}" for circuit in canonical])
        ].sort_values("correlation", ascending=False)
        dec_z = float(dec_row["combined_z"].iloc[0]) if len(dec_row) else np.nan
        dec_p = float(dec_row["stouffer_p"].iloc[0]) if len(dec_row) else np.nan
        dec_fdr = float(dec_row["fdr_bh_within_contrast"].iloc[0]) if len(dec_row) else np.nan
        term_effect = float(term_row["mean_standardized_effect"].iloc[0]) if len(term_row) else np.nan
        precursor_r = float(precursor["correlation"].iloc[0]) if len(precursor) else np.nan
        precursor_p = float(precursor["permutation_p"].iloc[0]) if len(precursor) else np.nan
        sender_r = float(sender["correlation"].iloc[0]) if len(sender) else np.nan
        sender_p = float(sender["permutation_p"].iloc[0]) if len(sender) else np.nan
        best_ligand = sender["x"].iloc[0].replace("sender_score__", "") if len(sender) else ""
        if perturbation == "KO":
            direction_score = max(0.0, dec_z if np.isfinite(dec_z) else 0.0)
            adaptation_score = max(0.0, -(term_effect if np.isfinite(term_effect) else 0.0))
            precursor_score = max(0.0, precursor_r if np.isfinite(precursor_r) else 0.0)
            direction_consistent = np.isfinite(dec_z) and dec_z > 0
            if direction_consistent and np.isfinite(dec_fdr) and dec_fdr < 0.05:
                evidence_tier = "A"
            elif direction_consistent and (
                (np.isfinite(dec_p) and dec_p < 0.10)
                or (np.isfinite(precursor_p) and precursor_p < 0.10 and precursor_r > 0)
            ):
                evidence_tier = "B"
            else:
                evidence_tier = "C"
        else:
            direction_score = max(0.0, -(dec_z if np.isfinite(dec_z) else 0.0))
            adaptation_score = max(0.0, term_effect if np.isfinite(term_effect) else 0.0)
            precursor_score = max(0.0, -(precursor_r if np.isfinite(precursor_r) else 0.0))
            direction_consistent = (
                np.isfinite(dec_z) and dec_z < 0
                and np.isfinite(term_effect) and term_effect > 0
            )
            term_fdr = float(term_row["fdr_bh_within_contrast"].iloc[0]) if len(term_row) else np.nan
            if direction_consistent and np.isfinite(term_fdr) and term_fdr < 0.05:
                evidence_tier = "A"
            elif direction_consistent and (
                (np.isfinite(dec_p) and dec_p < 0.10)
                or (np.isfinite(term_fdr) and term_fdr < 0.10)
                or (np.isfinite(precursor_p) and precursor_p < 0.10 and precursor_r < 0)
            ):
                evidence_tier = "B"
            else:
                evidence_tier = "C"
        sender_score = max(0.0, sender_r if np.isfinite(sender_r) else 0.0)
        rank_score = direction_score + adaptation_score + 2 * precursor_score + sender_score
        rows.append({
            "regulator": regulator, "recommended_perturbation": perturbation,
            "evidence_mode": preferred_type, "canonical_ligands": ";".join(canonical),
            "best_observed_canonical_ligand": best_ligand,
            "decoupled_combined_z": dec_z, "decoupled_p": dec_p, "decoupled_fdr": dec_fdr,
            "terminal_minus_early_standardized_effect": term_effect,
            "precursor_excess_partial_r": precursor_r, "precursor_excess_p": precursor_p,
            "best_sender_TF_partial_r": sender_r, "best_sender_TF_p": sender_p,
            "direction_consistent_with_perturbation": bool(direction_consistent),
            "rank_score": rank_score, "evidence_tier": evidence_tier,
        })
    return pd.DataFrame(rows).sort_values(["evidence_tier", "rank_score"], ascending=[True, False])


def build_circuit_ranking(
    meta: pd.DataFrame,
    sender_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    dec = meta.loc[meta["contrast"].eq("decoupled_minus_other_incomplete")].copy()
    sender = sender_outcomes.loc[
        sender_outcomes["scope"].eq("pooled_dataset_disease_residualized")
        & sender_outcomes["y"].eq("decoupled_excess__mouse_GMCSF_LAM_precursor")
    ]
    rows = []
    for circuit, details in CIRCUITS.items():
        receptor = dec.loc[
            dec["feature_type"].eq("receptor") & dec["feature_name"].eq(circuit)
        ]
        regulators = dec.loc[
            dec["feature_type"].eq("activity")
            & dec["feature_name"].isin(details["regulators"])
        ].sort_values("stouffer_p")
        sender_row = sender.loc[sender["x"].eq(f"sender_score__{circuit}")]
        receptor_z = float(receptor["combined_z"].iloc[0]) if len(receptor) else np.nan
        receptor_p = float(receptor["stouffer_p"].iloc[0]) if len(receptor) else np.nan
        receptor_fdr = float(receptor["fdr_bh_within_contrast"].iloc[0]) if len(receptor) else np.nan
        regulator = str(regulators["feature_name"].iloc[0]) if len(regulators) else ""
        regulator_z = float(regulators["combined_z"].iloc[0]) if len(regulators) else np.nan
        regulator_p = float(regulators["stouffer_p"].iloc[0]) if len(regulators) else np.nan
        regulator_fdr = float(regulators["fdr_bh_within_contrast"].iloc[0]) if len(regulators) else np.nan
        sender_r = float(sender_row["correlation"].iloc[0]) if len(sender_row) else np.nan
        sender_p = float(sender_row["permutation_p"].iloc[0]) if len(sender_row) else np.nan
        receiver_score = max(0.0, receptor_z if np.isfinite(receptor_z) else 0.0) + max(
            0.0, regulator_z if np.isfinite(regulator_z) else 0.0
        )
        if (
            np.isfinite(receptor_fdr) and receptor_fdr < 0.05 and receptor_z > 0
            and np.isfinite(regulator_fdr) and regulator_fdr < 0.05 and regulator_z > 0
        ):
            tier = "A"
        elif (
            (np.isfinite(receptor_fdr) and receptor_fdr < 0.05 and receptor_z > 0)
            or (np.isfinite(regulator_fdr) and regulator_fdr < 0.05 and regulator_z > 0)
        ):
            tier = "B"
        else:
            tier = "C"
        rows.append({
            "circuit": circuit, "receptor_combined_z": receptor_z,
            "receptor_p": receptor_p, "receptor_fdr": receptor_fdr,
            "best_downstream_regulator": regulator, "regulator_combined_z": regulator_z,
            "regulator_p": regulator_p, "regulator_fdr": regulator_fdr,
            "sender_precursor_partial_r": sender_r, "sender_precursor_p": sender_p,
            "receiver_chain_score": receiver_score, "receiver_evidence_tier": tier,
        })
    return pd.DataFrame(rows).sort_values(
        ["receiver_evidence_tier", "receiver_chain_score"], ascending=[True, False]
    )


def write_report(
    path: Path,
    lr_ledger: pd.DataFrame,
    audit: pd.DataFrame,
    sender: pd.DataFrame,
    meta: pd.DataFrame,
    sender_outcomes: pd.DataFrame,
    circuits: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    lines = [
        "# Donor-level T-cell niche and regulator inference",
        "",
        "## Design",
        "",
        "CSF T-cell ligand scores were paired to the same donor's monocyte-to-BAM continuum. "
        "LIANA consensus was used only to gate canonical ligand-receptor pairs. Receiver TF "
        "activities were inferred with DoRothEA ABC and ULM after removing every target gene "
        "used in the MHC-I/IFN axis or any projected LAM program.",
        "",
        "## Circularity and resource audit",
        "",
        f"- LIANA consensus supports {int(lr_ledger['exact_pair_in_liana_consensus'].sum())}/{len(lr_ledger)} "
        "prespecified circuits.",
        f"- {int(audit['n_axis2_targets_removed'].sum())} Axis-2 target edges and "
        f"{int(audit['n_LAM_targets_removed'].sum())} LAM-program target edges were removed before ULM.",
        f"- {(audit['activity_status'] == 'DoRothEA_ULM').sum()} regulators retained a curated activity model; "
        f"{(audit['activity_status'] != 'DoRothEA_ULM').sum()} are expression-only proxies.",
        "",
        "## Sender measurability",
        "",
    ]
    for circuit in CIRCUITS:
        values = sender.groupby("dataset", observed=True)[f"sender_detection__{circuit}"].mean()
        lines.append(f"- {circuit}: mean donor detection fraction by cohort " + ", ".join(
            f"{dataset}={value:.4f}" for dataset, value in values.items()
        ) + ".")
    lines.extend(["", "## Decoupled-state regulator activity", ""])
    dec = meta.loc[
        meta["contrast"].eq("decoupled_minus_other_incomplete")
        & meta["feature_type"].eq("activity")
    ].sort_values("stouffer_p").head(10)
    for row in dec.itertuples():
        lines.append(
            f"- {row.feature_name}: z={row.combined_z:.3f}, P={row.stouffer_p:.4g}, "
            f"FDR={row.fdr_bh_within_contrast:.4g}, concordant={row.direction_concordant}."
        )
    lines.extend(["", "## T-cell ligand association with GM-CSF precursor excess", ""])
    pooled = sender_outcomes.loc[
        sender_outcomes["scope"].eq("pooled_dataset_disease_residualized")
        & sender_outcomes["y"].eq("decoupled_excess__mouse_GMCSF_LAM_precursor")
    ].sort_values("permutation_p")
    for row in pooled.itertuples():
        lines.append(
            f"- {row.x}: partial rank r={row.correlation:.3f}, P={row.permutation_p:.4g}, "
            f"FDR={row.fdr_bh_within_family:.4g}, n={row.n_donors}."
        )
    lines.extend(["", "## Receiver-side circuit ranking", ""])
    for row in circuits.itertuples():
        lines.append(
            f"- {row.circuit}: tier {row.receiver_evidence_tier}, receptor z={row.receptor_combined_z:.3f} "
            f"(FDR={row.receptor_fdr:.4g}), best regulator={row.best_downstream_regulator} "
            f"z={row.regulator_combined_z:.3f} (FDR={row.regulator_fdr:.4g}); "
            f"sender-to-precursor r={row.sender_precursor_partial_r:.3f}."
        )
    lines.extend(["", "## Directional perturbation shortlist", ""])
    for row in ranking.head(10).itertuples():
        lines.append(
            f"- {row.regulator} {row.recommended_perturbation}: tier {row.evidence_tier}, "
            f"mode={row.evidence_mode}, rank={row.rank_score:.3f}, "
            f"best canonical ligand={row.best_observed_canonical_ligand or 'none'}."
        )
    lines.extend([
        "",
        "## Interpretation gate",
        "",
        "A ligand is not called causal from expression correlation alone. Advancement to virtual "
        "perturbation requires a resource-supported receptor, a directionally compatible regulator, "
        "and a donor-level relation to precursor excess or terminal adaptation. KO candidates should "
        "reduce precursor/Axis-2 activation; OE candidates should restore terminal LAM2 adaptation.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results" / "niche_regulator"
    output.mkdir(parents=True, exist_ok=True)

    lr_ledger = build_lr_ledger()
    lr_ledger.to_csv(output / "LIANA_consensus_canonical_circuit_ledger.csv", index=False)
    sender, sender_genes = build_sender_scores(root)
    sender.to_csv(output / "CSF_T_sender_donor_scores.csv", index=False)
    sender_genes.to_csv(output / "CSF_T_sender_gene_measurability.csv", index=False)

    network, audit = network_and_audit(root)
    network.to_csv(output / "DoRothEA_axis2_LAM_excluded_network.csv.gz", index=False, compression="gzip")
    audit.to_csv(output / "regulator_noncircularity_audit.csv", index=False)

    state_tables = []
    state_long_tables = []
    receptor_tables = []
    for dataset, config in DATASETS.items():
        print(f"Receiver state aggregation + ULM: {dataset}", flush=True)
        state, state_long, receptor = aggregate_receiver_dataset(root, dataset, config, network)
        state_tables.append(state)
        state_long_tables.append(state_long)
        receptor_tables.append(receptor)
    state = pd.concat(state_tables, ignore_index=True)
    state_long = pd.concat(state_long_tables, ignore_index=True)
    receptors = pd.concat(receptor_tables, ignore_index=True)
    state.to_csv(output / "receiver_state_regulator_receptor_scores.csv.gz", index=False, compression="gzip")
    state_long.to_csv(output / "receiver_state_features_long.csv.gz", index=False, compression="gzip")
    receptors.to_csv(output / "receiver_receptor_availability.csv", index=False)

    deltas, delta_tests, delta_meta = state_deltas(state_long, args.seed)
    deltas.to_csv(output / "receiver_donor_state_deltas.csv.gz", index=False, compression="gzip")
    delta_tests.to_csv(output / "receiver_state_delta_tests.csv", index=False)
    delta_meta.to_csv(output / "receiver_state_delta_meta.csv", index=False)

    lam = pd.read_csv(root / "results/cross_species_lam/LAM_donor_scores.csv")
    paired = sender.merge(lam[["dataset", "subject_id", *LAM_OUTCOMES]], on=["dataset", "subject_id"], how="inner")
    sender_columns = [column for column in sender if column.startswith("sender_score__")]
    sender_outcomes = association_table(
        paired, sender_columns, LAM_OUTCOMES, args.permutations, args.seed,
        "sender_to_LAM_outcome",
    )
    sender_outcomes.to_csv(output / "sender_to_LAM_outcome_associations.csv", index=False)

    decoupled = deltas.loc[
        deltas["contrast"].eq("decoupled_minus_other_incomplete")
        & deltas["feature_type"].isin(["activity", "expression"])
    ].copy()
    wide_delta = decoupled.pivot_table(
        index=["dataset", "subject_id", "disease_class"],
        columns=["feature_type", "feature_name"], values="delta",
    )
    wide_delta.columns = [f"{kind}__{name}" for kind, name in wide_delta.columns]
    wide_delta = wide_delta.reset_index()
    sender_delta = sender.merge(wide_delta, on=["dataset", "subject_id", "disease_class"], how="inner")
    delta_features = [column for column in wide_delta if "__" in column]
    preferred_delta_features = []
    for row in audit.itertuples():
        mode = "activity" if row.activity_status == "DoRothEA_ULM" else "expression"
        feature = f"{mode}__{row.regulator}"
        if feature in delta_features:
            preferred_delta_features.append(feature)
    canonical_pairs = []
    for circuit, details in CIRCUITS.items():
        for regulator in details["regulators"]:
            candidates = [
                feature for feature in preferred_delta_features
                if feature.endswith(f"__{regulator}")
            ]
            canonical_pairs.extend(
                (f"sender_score__{circuit}", feature) for feature in candidates
            )
    sender_tf = association_table(
        sender_delta, sender_columns, preferred_delta_features, args.permutations, args.seed + 1000,
        "sender_to_regulator_delta", prespecified_pairs=canonical_pairs,
    )
    sender_tf.to_csv(output / "sender_to_regulator_delta_associations.csv", index=False)

    delta_outcome = wide_delta.merge(
        lam[["dataset", "subject_id", *LAM_OUTCOMES]], on=["dataset", "subject_id"], how="inner"
    )
    delta_outcomes = association_table(
        delta_outcome, preferred_delta_features, LAM_OUTCOMES, args.permutations, args.seed + 2000,
        "regulator_delta_to_LAM_outcome",
    )
    delta_outcomes.to_csv(output / "regulator_delta_to_LAM_outcome_associations.csv", index=False)

    ranking = build_candidate_ranking(delta_meta, delta_outcomes, sender_tf, audit)
    ranking.to_csv(output / "directional_regulator_candidate_ranking.csv", index=False)
    circuits = build_circuit_ranking(delta_meta, sender_outcomes)
    circuits.to_csv(output / "receiver_side_circuit_ranking.csv", index=False)
    write_report(
        output / "NICHE_REGULATOR_INFERENCE_REPORT.md",
        lr_ledger, audit, sender, delta_meta, sender_outcomes, circuits, ranking,
    )
    print(ranking.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
