#!/usr/bin/env python3
"""Test compensatory co-perturbations around the STAT1 inflammatory-lock anchor."""

from __future__ import annotations

import argparse
import importlib.util
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


def load_single_module():
    path = ROOT / "scripts/57_run_celloracle_directional_perturbation.py"
    spec = importlib.util.spec_from_file_location("single_perturbation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


single = load_single_module()

INTERVENTIONS = {
    "STAT1_KO__PPARG_OE": {"STAT1": "KO", "PPARG": "OE"},
    "STAT1_KO__MAF_OE": {"STAT1": "KO", "MAF": "OE"},
    "STAT1_KO__MAFB_OE": {"STAT1": "KO", "MAFB": "OE"},
    "STAT1_KO__PPARG_OE__MAFB_OE": {"STAT1": "KO", "PPARG": "OE", "MAFB": "OE"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(single.DATASETS), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-hvgs", type=int, default=700)
    parser.add_argument("--candidate-targets", type=int, default=75)
    parser.add_argument("--max-genes", type=int, default=1600)
    parser.add_argument("--cap-per-donor-state", type=int, default=150)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--randomized-grns", type=int, default=50)
    parser.add_argument("--seed", type=int, default=single.SEED)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    dataset = args.dataset
    config = single.DATASETS[dataset]
    output = root / "results/virtual_perturbation_celloracle_combinations" / dataset
    output.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(root / config["path"])
    single.assign_analysis_state(adata, config)
    adata = single.donor_state_cap(adata, args.cap_per_donor_state, args.seed)

    network = single.load_network(root)
    trajectory = single.load_trajectory_program(root)
    selected, selection_ledger = single.select_genes(
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
    work.layers["raw_count"] = work.X.copy()
    tf_dict, base_edges = single.build_tfdict(network, selected)
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

    endpoints = single.endpoint_definitions(trajectory)
    imputed = single.dense(oracle.adata.layers["imputed_count"]).astype(np.float64)
    active = set(getattr(oracle, "active_regulatory_genes", []))
    direct_targets_by_gene = {
        gene: set(network.loc[network["source"].eq(gene), "target"])
        for components in INTERVENTIONS.values()
        for gene in components
    }

    donor_rows = []
    endpoint_rows = []
    status_rows = []
    component_rows = []
    for intervention, components in INTERVENTIONS.items():
        perturb_condition = {}
        all_estimable = True
        for gene, perturbation in components.items():
            value, details = single.perturbation_value(oracle, gene, perturbation)
            details.update({"dataset": dataset, "intervention": intervention})
            component_rows.append(details)
            perturb_condition[gene] = value
            if gene not in active or (perturbation == "OE" and value <= 0):
                all_estimable = False

        union_direct_targets = set().union(
            *(direct_targets_by_gene[gene] for gene in components)
        )
        status_rows.append({
            "dataset": dataset,
            "intervention": intervention,
            "components": ";".join(f"{gene}:{perturbation}" for gene, perturbation in components.items()),
            "perturb_condition": json.dumps(perturb_condition, sort_keys=True),
            "all_components_active_in_fitted_grn": all_estimable,
            "n_union_direct_targets": len(union_direct_targets),
            "simulation_status": "completed" if all_estimable else "not_estimable",
        })
        if not all_estimable:
            continue

        for network_model, propagation, randomized in single.iter_simulation_settings(
            oracle, args.randomized_grns, args.seed
        ):
            oracle.simulate_shift(
                perturb_condition=perturb_condition,
                GRN_unit="whole",
                n_propagation=propagation,
                ignore_warning=True,
                clip_delta_X=True,
                use_randomized_GRN=randomized,
            )
            delta = single.dense(oracle.adata.layers["delta_X"]).astype(np.float64)
            for endpoint, weights in endpoints.items():
                variants = {
                    "full": weights,
                    "heldout_direct_targets": {
                        gene: weight
                        for gene, weight in weights.items()
                        if gene not in union_direct_targets and gene not in components
                    },
                }
                for endpoint_variant, variant_weights in variants.items():
                    cell_effect, available = single.score_delta(
                        delta,
                        imputed,
                        oracle.adata.var_names,
                        variant_weights,
                    )
                    removed = sorted(
                        (set(weights) & (union_direct_targets | set(components)))
                        & set(oracle.adata.var_names)
                    )
                    endpoint_rows.append({
                        "dataset": dataset,
                        "intervention": intervention,
                        "n_propagation": propagation,
                        "endpoint": endpoint,
                        "endpoint_variant": endpoint_variant,
                        "network_model": network_model,
                        "n_original_genes": len(weights),
                        "n_available_genes": len(available),
                        "available_genes": ";".join(available),
                        "n_union_direct_targets_removed": len(removed) if endpoint_variant != "full" else 0,
                        "union_direct_targets_removed": ";".join(removed) if endpoint_variant != "full" else "",
                    })
                    donor_rows.extend(single.aggregate_donor_states(
                        oracle,
                        cell_effect,
                        dataset,
                        intervention,
                        "+".join(components.values()),
                        propagation,
                        endpoint,
                        endpoint_variant,
                        len(available),
                        network_model,
                    ))

    donor = pd.DataFrame(donor_rows).rename(columns={"candidate": "intervention"})
    donor.to_csv(output / "donor_state_combination_effects.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(output / "combination_endpoint_holdout_ledger.csv", index=False)
    pd.DataFrame(status_rows).to_csv(output / "combination_status.csv", index=False)
    pd.DataFrame(component_rows).to_csv(output / "combination_component_calibration.csv", index=False)

    ledger = {
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
        "interventions": INTERVENTIONS,
    }
    (output / "run_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
