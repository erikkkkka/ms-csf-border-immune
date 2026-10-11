#!/usr/bin/env python3
"""Run cohort-specific CellOracle perturbations on the CSF monocyte-BAM continuum."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HOME", str(ROOT))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / ".config"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import celloracle as co
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SEED = 12345
MS_LABELS = {"MS", "RRMS", "PPMS", "SPMS"}

DATASETS = {
    "GSE138266": {
        "path": "results/virtual_perturbation_celloracle/compatible_inputs/GSE138266_MS_CSF_continuum_minimal.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
    },
    "GSE286068": {
        "path": "results/virtual_perturbation_celloracle/compatible_inputs/GSE286068_MS_CSF_continuum_minimal.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
    },
    "GSE328265": {
        "path": "results/virtual_perturbation_celloracle/compatible_inputs/GSE328265_MS_CSF_continuum_minimal.h5ad",
        "pt": "projected_differentiation",
        "activation": "external_activation_residual",
        "threshold": "external_control_activation_q75",
    },
}

CANDIDATES = {
    "IRF1": "KO",
    "STAT1": "KO",
    "PPARG": "OE",
    "NR1H3": "OE",
    "MAFB": "OE",
}

MODULES = {
    "axis2": [
        "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5",
        "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7",
    ],
    "gmcsf_lam_precursor": ["CXCL10", "CXCL9", "TGM2", "TGFB3", "PLAC8", "IRF1", "CSF2RB"],
    "lam2_complement_apc": [
        "APOE", "C1QA", "C1QB", "C1QC", "GPNMB", "SELENOP", "TREM2", "MERTK",
        "HLA-DRA", "CD74", "CIITA",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-hvgs", type=int, default=500)
    parser.add_argument("--candidate-targets", type=int, default=60)
    parser.add_argument("--max-genes", type=int, default=1400)
    parser.add_argument("--cap-per-donor-state", type=int, default=150)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--randomized-grns", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def gene_moments(matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        prevalence = np.asarray((matrix > 0).mean(axis=0)).ravel()
    else:
        values = np.asarray(matrix, dtype=float)
        mean = values.mean(axis=0)
        mean_sq = np.square(values).mean(axis=0)
        prevalence = (values > 0).mean(axis=0)
    variance = np.maximum(mean_sq - np.square(mean), 0)
    return mean, variance, prevalence


def assign_analysis_state(adata: sc.AnnData, config: dict) -> None:
    pt = adata.obs[config["pt"]].to_numpy(float)
    activation = adata.obs[config["activation"]].to_numpy(float)
    threshold = float(adata.uns[config["threshold"]])
    incomplete = pt < 0.80
    high = activation > threshold
    state = np.full(adata.n_obs, "terminal", dtype=object)
    state[incomplete & ~high] = "other_incomplete"
    state[incomplete & high] = "decoupled"
    adata.obs["analysis_state"] = pd.Categorical(
        state,
        categories=["other_incomplete", "decoupled", "terminal"],
    )
    adata.obs["analysis_pseudotime"] = pt
    adata.obs["analysis_activation"] = activation


def donor_state_cap(adata: sc.AnnData, cap: int, seed: int) -> sc.AnnData:
    rng = np.random.default_rng(seed)
    selected = []
    frame = adata.obs[["subject_id", "analysis_state"]].copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["analysis_state"] = frame["analysis_state"].astype(str)
    for _, positions in frame.groupby(["subject_id", "analysis_state"], observed=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        if len(positions) > cap:
            positions = np.sort(rng.choice(positions, size=cap, replace=False))
        selected.extend(positions.tolist())
    return adata[np.sort(np.asarray(selected, dtype=int))].copy()


def load_network(root: Path) -> pd.DataFrame:
    network = pd.read_csv(root / "data/reference/dorothea_human_ABC.csv.gz")
    network = network.loc[network["confidence"].isin(["A", "B", "C"])].copy()
    network["source"] = network["source"].astype(str)
    network["target"] = network["target"].astype(str)
    return network.drop_duplicates(["source", "target"])


def load_trajectory_program(root: Path) -> dict[str, float]:
    frame = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/frozen_trajectory_program.csv"
    )
    sign = frame["training_direction"].map({"positive": 1.0, "negative": -1.0})
    return dict(zip(frame["gene"].astype(str), sign.astype(float)))


def select_genes(
    adata: sc.AnnData,
    network: pd.DataFrame,
    trajectory: dict[str, float],
    n_hvgs: int,
    candidate_targets: int,
    max_genes: int,
) -> tuple[list[str], pd.DataFrame]:
    genes = adata.var_names.astype(str)
    _, variance, prevalence = gene_moments(adata.X)
    metrics = pd.DataFrame({
        "gene": genes,
        "variance": variance,
        "prevalence": prevalence,
    }).set_index("gene")
    eligible = metrics.index[
        (metrics["prevalence"] >= 0.01)
        & (metrics["variance"] > 1e-8)
        & ~metrics.index.str.startswith("MT-")
        & ~metrics.index.str.match(r"^RP[SL][0-9]")
    ]
    eligible_metrics = metrics.loc[eligible].copy()
    eligible_metrics["selection_score"] = (
        eligible_metrics["variance"] * np.sqrt(eligible_metrics["prevalence"])
    )
    variable = list(eligible_metrics.nlargest(n_hvgs, "selection_score").index)

    mandatory = set(CANDIDATES)
    mandatory.update(gene for module in MODULES.values() for gene in module)
    mandatory.update(trajectory)
    mandatory &= set(genes)

    source_genes = set(network["source"]) & set(eligible)
    candidate_target_genes = set()
    for candidate in CANDIDATES:
        targets = set(network.loc[network["source"].eq(candidate), "target"]) & set(eligible)
        if targets:
            ranked = metrics.loc[sorted(targets)].copy()
            ranked["selection_score"] = ranked["variance"] * np.sqrt(ranked["prevalence"])
            candidate_target_genes.update(ranked.nlargest(candidate_targets, "selection_score").index)

    selected = set(variable) | mandatory | source_genes | candidate_target_genes
    if len(selected) > max_genes:
        protected = mandatory | source_genes | candidate_target_genes
        remaining_slots = max(0, max_genes - len(protected))
        variable_keep = [gene for gene in variable if gene not in protected][:remaining_slots]
        selected = protected | set(variable_keep)
    selected = sorted(selected)

    ledger = metrics.loc[selected].reset_index()
    ledger["mandatory_endpoint_or_candidate"] = ledger["gene"].isin(mandatory)
    ledger["dorothea_source_tf"] = ledger["gene"].isin(source_genes)
    ledger["candidate_high_information_target"] = ledger["gene"].isin(candidate_target_genes)
    ledger["top_variable"] = ledger["gene"].isin(variable)
    return selected, ledger


def build_tfdict(network: pd.DataFrame, selected: list[str]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    selected_set = set(selected)
    edges = network.loc[
        network["source"].isin(selected_set)
        & network["target"].isin(selected_set)
        & ~network["source"].eq(network["target"])
    ].copy()
    tf_dict = (
        edges.groupby("target", observed=True)["source"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    return tf_dict, edges


def endpoint_definitions(trajectory: dict[str, float]) -> dict[str, dict[str, float]]:
    endpoints = {
        name: {gene: 1.0 for gene in genes}
        for name, genes in MODULES.items()
    }
    endpoints["trajectory_signed"] = trajectory.copy()
    return endpoints


def iter_simulation_settings(oracle: co.Oracle, n_randomized: int, seed: int):
    yield "observed_GRN", 1, False
    yield "observed_GRN", 3, False
    for random_index in range(n_randomized):
        random_seed = seed + random_index + 1
        oracle.calculate_randomized_coef_table(random_seed=random_seed)
        yield f"target_shuffled_GRN_seed_{random_seed}", 3, True


def perturbation_value(
    oracle: co.Oracle,
    candidate: str,
    perturbation: str,
) -> tuple[float, dict]:
    index = oracle.adata.var_names.get_loc(candidate)
    values = dense(oracle.adata.layers["imputed_count"][:, index]).ravel().astype(float)
    terminal = oracle.adata.obs["analysis_state"].astype(str).eq("terminal").to_numpy()
    terminal_positive = values[terminal & (values > 0)]
    fallback = "terminal_positive_q90"
    calibration = terminal_positive
    if len(calibration) < 3:
        calibration = values[values > 0]
        fallback = "all_positive_q90"
    if perturbation == "KO":
        value = 0.0
        fallback = "fixed_zero"
    elif len(calibration):
        value = float(np.quantile(calibration, 0.90))
    else:
        value = 0.0
        fallback = "not_measurable_zero"
    details = {
        "candidate": candidate,
        "perturbation": perturbation,
        "perturbation_value": value,
        "calibration_rule": fallback,
        "raw_imputed_detection_fraction": float(np.mean(values > 0)),
        "terminal_imputed_detection_fraction": float(np.mean(values[terminal] > 0)) if terminal.any() else np.nan,
        "imputed_median": float(np.median(values)),
        "imputed_q90": float(np.quantile(values, 0.90)),
        "imputed_max": float(np.max(values)),
    }
    return value, details


def score_delta(
    delta: np.ndarray,
    imputed: np.ndarray,
    var_names: pd.Index,
    weights: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    available = [gene for gene in weights if gene in var_names]
    if not available:
        return np.full(delta.shape[0], np.nan), []
    indices = var_names.get_indexer(available)
    sd = imputed[:, indices].std(axis=0, ddof=0)
    sd[sd < 1e-6] = 1.0
    signs = np.asarray([weights[gene] for gene in available], dtype=float)
    values = delta[:, indices] / sd[None, :]
    return np.mean(values * signs[None, :], axis=1), available


def aggregate_donor_states(
    oracle: co.Oracle,
    cell_effect: np.ndarray,
    dataset: str,
    candidate: str,
    perturbation: str,
    propagation: int,
    endpoint: str,
    endpoint_variant: str,
    n_endpoint_genes: int,
    network_model: str,
) -> list[dict]:
    obs = oracle.adata.obs
    subjects = obs["subject_id"].astype(str).to_numpy()
    state = obs["analysis_state"].astype(str).to_numpy()
    state_masks = {
        "decoupled": state == "decoupled",
        "other_incomplete": state == "other_incomplete",
        "incomplete_all": state != "terminal",
        "terminal": state == "terminal",
        "all": np.ones(len(state), dtype=bool),
    }
    rows = []
    for subject in sorted(set(subjects)):
        subject_mask = subjects == subject
        for state_name, state_mask in state_masks.items():
            selected = subject_mask & state_mask & np.isfinite(cell_effect)
            if not selected.any():
                continue
            rows.append({
                "dataset": dataset,
                "subject_id": subject,
                "candidate": candidate,
                "perturbation": perturbation,
                "n_propagation": propagation,
                "endpoint": endpoint,
                "endpoint_variant": endpoint_variant,
                "network_model": network_model,
                "state": state_name,
                "n_endpoint_genes": n_endpoint_genes,
                "n_cells": int(selected.sum()),
                "mean_standardized_delta": float(np.mean(cell_effect[selected])),
                "median_standardized_delta": float(np.median(cell_effect[selected])),
            })
    return rows


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    dataset = args.dataset
    config = DATASETS[dataset]
    output = root / "results/virtual_perturbation_celloracle" / dataset
    output.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(root / config["path"])
    mask = adata.obs["disease_group"].astype(str).isin(MS_LABELS).to_numpy()
    if "tissue" in adata.obs:
        mask &= adata.obs["tissue"].astype(str).str.upper().eq("CSF").to_numpy()
    adata = adata[mask].copy()
    assign_analysis_state(adata, config)
    adata = donor_state_cap(adata, args.cap_per_donor_state, args.seed)

    network = load_network(root)
    trajectory = load_trajectory_program(root)
    selected, selection_ledger = select_genes(
        adata,
        network,
        trajectory,
        args.n_hvgs,
        args.candidate_targets,
        args.max_genes,
    )
    selection_ledger.to_csv(output / "selected_gene_ledger.csv", index=False)
    work = adata[:, selected].copy()
    work.obs["analysis_state"] = work.obs["analysis_state"].astype("category")
    # CellOracle 0.20.0's variability QC reads this layer although its import
    # method no longer creates it. The simulation itself uses normalized_count.
    work.layers["raw_count"] = work.X.copy()
    tf_dict, base_edges = build_tfdict(network, selected)
    base_edges.to_csv(output / "dorothea_base_grn_edges.csv.gz", index=False, compression="gzip")

    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(
        work,
        cluster_column_name="analysis_state",
        embedding_name="X_umap",
        test_mode=True,
    )
    oracle.high_var_genes = oracle.adata.var_names.to_numpy(copy=True)
    oracle.adata.var["symbol"] = oracle.adata.var_names.astype(str)
    oracle.adata.var["isin_top1000_var_mean_genes"] = True
    oracle.import_TF_data(TFdict=tf_dict)
    n_components = min(50, oracle.adata.n_obs - 1, oracle.adata.n_vars - 1)
    oracle.perform_PCA(n_components=n_components)
    k = min(50, max(15, int(round(np.sqrt(oracle.adata.n_obs)))))
    k = min(k, oracle.adata.n_obs - 1)
    oracle.knn_imputation(
        k=k,
        n_pca_dims=min(30, n_components),
        balanced=False,
        n_jobs=args.threads,
    )
    oracle.fit_GRN_for_simulation(
        GRN_unit="whole",
        alpha=args.alpha,
        use_cluster_specific_TFdict=False,
        verbose_level=1,
    )

    coefficient = oracle.coef_matrix
    candidate_edge_rows = []
    for candidate in CANDIDATES:
        if candidate not in coefficient.index:
            continue
        values = coefficient.loc[candidate]
        for target, value in values.loc[np.abs(values) > 1e-10].items():
            candidate_edge_rows.append({
                "dataset": dataset,
                "source": candidate,
                "target": target,
                "ridge_coefficient": float(value),
            })
    pd.DataFrame(candidate_edge_rows).to_csv(output / "inferred_candidate_outgoing_edges.csv", index=False)

    endpoints = endpoint_definitions(trajectory)
    imputed = dense(oracle.adata.layers["imputed_count"]).astype(np.float64)
    donor_rows = []
    endpoint_rows = []
    status_rows = []
    direct_targets = {
        candidate: set(network.loc[network["source"].eq(candidate), "target"])
        for candidate in CANDIDATES
    }

    active = set(getattr(oracle, "active_regulatory_genes", []))
    for candidate, perturbation in CANDIDATES.items():
        value, status = perturbation_value(oracle, candidate, perturbation)
        status.update({
            "dataset": dataset,
            "candidate_in_selected_genes": candidate in oracle.adata.var_names,
            "candidate_in_base_grn_regulators": candidate in oracle.all_regulatory_genes_in_TFdict,
            "candidate_active_in_fitted_grn": candidate in active,
            "n_inferred_outgoing_edges": int(
                np.sum(np.abs(coefficient.loc[candidate].to_numpy(float)) > 1e-10)
            ) if candidate in coefficient.index else 0,
        })
        if candidate not in active or (perturbation == "OE" and value <= 0):
            status["simulation_status"] = "not_estimable"
            status_rows.append(status)
            continue

        status["simulation_status"] = "completed"
        status_rows.append(status)
        for network_model, propagation, randomized in iter_simulation_settings(
            oracle, args.randomized_grns, args.seed
        ):
            oracle.simulate_shift(
                perturb_condition={candidate: value},
                GRN_unit="whole",
                n_propagation=propagation,
                ignore_warning=True,
                clip_delta_X=True,
                use_randomized_GRN=randomized,
            )
            delta = dense(oracle.adata.layers["delta_X"]).astype(np.float64)
            for endpoint, weights in endpoints.items():
                variants = {
                    "full": weights,
                    "heldout_direct_targets": {
                        gene: weight
                        for gene, weight in weights.items()
                        if gene not in direct_targets[candidate] and gene != candidate
                    },
                }
                for endpoint_variant, variant_weights in variants.items():
                    cell_effect, available = score_delta(
                        delta,
                        imputed,
                        oracle.adata.var_names,
                        variant_weights,
                    )
                    removed = sorted(
                        (set(weights) & (direct_targets[candidate] | {candidate}))
                        & set(oracle.adata.var_names)
                    )
                    endpoint_rows.append({
                        "dataset": dataset,
                        "candidate": candidate,
                        "perturbation": perturbation,
                        "n_propagation": propagation,
                        "endpoint": endpoint,
                        "endpoint_variant": endpoint_variant,
                        "network_model": network_model,
                        "n_original_genes": len(weights),
                        "n_available_genes": len(available),
                        "available_genes": ";".join(available),
                        "n_direct_targets_removed": len(removed) if endpoint_variant != "full" else 0,
                        "direct_targets_removed": ";".join(removed) if endpoint_variant != "full" else "",
                    })
                    donor_rows.extend(aggregate_donor_states(
                        oracle,
                        cell_effect,
                        dataset,
                        candidate,
                        perturbation,
                        propagation,
                        endpoint,
                        endpoint_variant,
                        len(available),
                        network_model,
                    ))

    pd.DataFrame(status_rows).to_csv(output / "perturbation_status.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(output / "endpoint_holdout_ledger.csv", index=False)
    pd.DataFrame(donor_rows).to_csv(output / "donor_state_perturbation_effects.csv", index=False)

    input_ledger = {
        "dataset": dataset,
        "n_cells": int(oracle.adata.n_obs),
        "n_donors": int(oracle.adata.obs["subject_id"].astype(str).nunique()),
        "n_genes": int(oracle.adata.n_vars),
        "n_base_grn_targets": len(tf_dict),
        "n_base_grn_edges": len(base_edges),
        "n_active_regulators": len(active),
        "knn_k": k,
        "n_pcs": n_components,
        "ridge_alpha": args.alpha,
        "celloracle_version": co.__version__,
        "random_seed": args.seed,
        "n_target_shuffled_GRNs": args.randomized_grns,
        "cap_per_donor_state": args.cap_per_donor_state,
        "state_counts": oracle.adata.obs["analysis_state"].astype(str).value_counts().to_dict(),
    }
    with (output / "run_ledger.json").open("w") as handle:
        json.dump(input_ledger, handle, indent=2)
    print(json.dumps(input_ledger, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
