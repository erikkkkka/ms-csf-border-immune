#!/usr/bin/env python3
"""Build donor-paired inflammatory-lock and border-adaptation drug signatures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HOME", str(ROOT))
os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / ".config"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


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

AXIS2 = {
    "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7",
}
LAM2 = {
    "APOE", "C1QA", "C1QB", "C1QC", "GPNMB", "SELENOP", "TREM2", "MERTK",
    "HLA-DRA", "CD74", "CIITA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-per-direction", type=int, default=150)
    parser.add_argument("--min-cells-per-state", type=int, default=5)
    return parser.parse_args()


def dense_mean(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel()
    return np.asarray(matrix, dtype=float).mean(axis=0)


def assign_state(adata: sc.AnnData, config: dict) -> np.ndarray:
    pt = adata.obs[config["pt"]].to_numpy(float)
    activation = adata.obs[config["activation"]].to_numpy(float)
    incomplete = pt < 0.80
    high = activation > float(adata.uns[config["threshold"]])
    state = np.full(adata.n_obs, "terminal", dtype=object)
    state[incomplete & ~high] = "other_incomplete"
    state[incomplete & high] = "decoupled"
    return state


def donor_paired_deltas(
    adata: sc.AnnData,
    config: dict,
    min_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = assign_state(adata, config)
    subjects = adata.obs["subject_id"].astype(str).to_numpy()
    delta_rows: list[np.ndarray] = []
    ledger_rows: list[dict] = []
    for subject in sorted(set(subjects)):
        donor = subjects == subject
        decoupled = donor & (state == "decoupled")
        other = donor & (state == "other_incomplete")
        eligible = decoupled.sum() >= min_cells and other.sum() >= min_cells
        ledger_rows.append({
            "subject_id": subject,
            "n_decoupled": int(decoupled.sum()),
            "n_other_incomplete": int(other.sum()),
            "eligible_pair": bool(eligible),
        })
        if eligible:
            delta_rows.append(dense_mean(adata.X[decoupled]) - dense_mean(adata.X[other]))
    if not delta_rows:
        raise RuntimeError("No eligible donor-state pairs")
    delta = pd.DataFrame(
        np.vstack(delta_rows),
        columns=adata.var_names.astype(str).str.upper(),
    )
    return delta, pd.DataFrame(ledger_rows)


def cohort_statistics(delta: pd.DataFrame, dataset: str) -> pd.DataFrame:
    values = delta.to_numpy(float)
    n = values.shape[0]
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    se = sd / np.sqrt(n)
    t_value = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
    p_value = 2 * stats.t.sf(np.abs(t_value), df=n - 1)
    standardized = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    return pd.DataFrame({
        "gene": delta.columns,
        f"{dataset}_n_pairs": n,
        f"{dataset}_mean_delta": mean,
        f"{dataset}_standardized_effect": standardized,
        f"{dataset}_p": p_value,
    })


def signed_z(p_value: np.ndarray, effect: np.ndarray) -> np.ndarray:
    p_value = np.clip(p_value, np.finfo(float).tiny, 1.0)
    return np.sign(effect) * stats.norm.isf(p_value / 2)


def build_meta(tables: list[pd.DataFrame]) -> pd.DataFrame:
    meta = tables[0]
    for table in tables[1:]:
        meta = meta.merge(table, on="gene", how="inner")
    z_values = []
    weights = []
    effects = []
    for dataset in DATASETS:
        effect = meta[f"{dataset}_mean_delta"].to_numpy(float)
        p_value = meta[f"{dataset}_p"].to_numpy(float)
        z_values.append(signed_z(p_value, effect))
        weights.append(np.sqrt(float(meta[f"{dataset}_n_pairs"].iloc[0])))
        effects.append(effect)
    z_matrix = np.vstack(z_values)
    weight = np.asarray(weights, dtype=float)[:, None]
    effect_matrix = np.vstack(effects)
    meta["meta_z"] = (z_matrix * weight).sum(axis=0) / np.sqrt(np.square(weight).sum())
    meta["meta_p"] = 2 * stats.norm.sf(np.abs(meta["meta_z"]))
    meta["meta_fdr"] = multipletests(meta["meta_p"], method="fdr_bh")[1]
    signs = np.sign(effect_matrix)
    meta["direction_concordant"] = np.all(signs == signs[0], axis=0) & np.all(signs != 0, axis=0)
    meta["minimum_abs_standardized_effect"] = np.min(np.vstack([
        np.abs(meta[f"{dataset}_standardized_effect"].to_numpy(float))
        for dataset in DATASETS
    ]), axis=0)
    return meta.sort_values("meta_p").reset_index(drop=True)


def write_gene_list(path: Path, genes: pd.Series | list[str]) -> None:
    path.write_text("\n".join(map(str, genes)) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/transition_aware_drug_screen/signatures"
    output.mkdir(parents=True, exist_ok=True)

    cohort_tables = []
    donor_ledgers = []
    for dataset, config in DATASETS.items():
        adata = sc.read_h5ad(root / config["path"])
        delta, ledger = donor_paired_deltas(adata, config, args.min_cells_per_state)
        ledger.insert(0, "dataset", dataset)
        donor_ledgers.append(ledger)
        delta.insert(0, "subject_pair_index", [f"{dataset}_{i + 1}" for i in range(len(delta))])
        delta.to_csv(output / f"{dataset}_donor_paired_state_deltas.csv.gz", index=False)
        delta = delta.drop(columns="subject_pair_index")
        cohort = cohort_statistics(delta, dataset)
        cohort.to_csv(output / f"{dataset}_gene_statistics.csv.gz", index=False)
        cohort_tables.append(cohort)

    donor_ledger = pd.concat(donor_ledgers, ignore_index=True)
    donor_ledger.to_csv(output / "donor_pair_eligibility.csv", index=False)
    meta = build_meta(cohort_tables)

    trajectory = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/frozen_trajectory_program.csv"
    )
    trajectory["gene"] = trajectory["gene"].astype(str).str.upper()
    trajectory_positive = set(trajectory.loc[trajectory["training_direction"].eq("positive"), "gene"])
    trajectory_negative = set(trajectory.loc[trajectory["training_direction"].eq("negative"), "gene"])
    held_out = AXIS2 | LAM2 | trajectory_positive | trajectory_negative
    meta["held_out_endpoint_gene"] = meta["gene"].isin(held_out)
    meta["selection_eligible"] = meta["direction_concordant"] & ~meta["held_out_endpoint_gene"]

    up = meta.loc[meta["selection_eligible"] & meta["meta_z"].gt(0)].nlargest(
        args.n_per_direction, "meta_z"
    ).copy()
    down = meta.loc[meta["selection_eligible"] & meta["meta_z"].lt(0)].nsmallest(
        args.n_per_direction, "meta_z"
    ).copy()
    up["signature_direction"] = "lock_high"
    down["signature_direction"] = "lock_low"
    selected = pd.concat([up, down], ignore_index=True)
    meta.to_csv(output / "inflammatory_activation_gene_meta_analysis.csv.gz", index=False)
    selected.to_csv(output / "inflammatory_activation_signature.csv", index=False)

    write_gene_list(output / "lock_high.txt", up["gene"])
    write_gene_list(output / "lock_low.txt", down["gene"])
    write_gene_list(output / "axis2.txt", sorted(AXIS2))
    write_gene_list(output / "lam2_complement_apc.txt", sorted(LAM2))
    write_gene_list(output / "trajectory_positive.txt", sorted(trajectory_positive))
    write_gene_list(output / "trajectory_negative.txt", sorted(trajectory_negative))

    summary = pd.DataFrame([{
        "n_cohorts": len(DATASETS),
        "n_eligible_donor_pairs": int(donor_ledger["eligible_pair"].sum()),
        "n_common_genes": len(meta),
        "n_direction_concordant": int(meta["direction_concordant"].sum()),
        "n_held_out_endpoint_genes": int(meta["held_out_endpoint_gene"].sum()),
        "n_lock_high": len(up),
        "n_lock_low": len(down),
        "n_axis2": len(AXIS2),
        "n_lam2": len(LAM2),
        "n_trajectory_positive": len(trajectory_positive),
        "n_trajectory_negative": len(trajectory_negative),
        "primary_selection_rule": (
            "direction concordant across three MS CSF cohorts; endpoint genes held out; "
            f"top {args.n_per_direction} per direction by weighted Stouffer meta-z"
        ),
    }])
    summary.to_csv(output / "signature_build_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print("\nLock-high leaders", flush=True)
    print(up[["gene", "meta_z", "meta_fdr", "minimum_abs_standardized_effect"]].head(20).to_string(index=False), flush=True)
    print("\nLock-low leaders", flush=True)
    print(down[["gene", "meta_z", "meta_fdr", "minimum_abs_standardized_effect"]].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
