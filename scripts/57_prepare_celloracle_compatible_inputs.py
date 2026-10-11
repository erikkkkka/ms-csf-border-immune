#!/usr/bin/env python3
"""Export minimal h5ad inputs readable by CellOracle's pinned AnnData stack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


MS_LABELS = {"MS", "RRMS", "PPMS", "SPMS"}

DATASETS = {
    "GSE138266": {
        "path": "results/trajectory/continuum_objects/GSE138266_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
    },
    "GSE286068": {
        "path": "results/trajectory/continuum_objects/GSE286068_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
    },
    "GSE328265": {
        "path": "results/external_validation/GSE328265_two_axis_projection/GSE328265_monocyte_BAM_two_axis.h5ad",
        "pt": "projected_differentiation",
        "activation": "external_activation_residual",
        "threshold": "external_control_activation_q75",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    return parser.parse_args()


def prepare_dataset(root: Path, dataset: str) -> dict:
    config = DATASETS[dataset]
    source_path = root / config["path"]
    source = sc.read_h5ad(source_path)
    mask = source.obs["disease_group"].astype(str).isin(MS_LABELS).to_numpy()
    if "tissue" in source.obs:
        mask &= source.obs["tissue"].astype(str).str.upper().eq("CSF").to_numpy()
    source = source[mask].copy()

    required_obs = [
        "subject_id", "diagnosis", "disease_group", "tissue", "transition_state",
        config["pt"], config["activation"],
    ]
    obs = source.obs[[column for column in required_obs if column in source.obs]].copy()
    for column in obs.select_dtypes(include=["string", "object"]).columns:
        obs[column] = pd.Categorical(obs[column].astype(str))
    obs.index = pd.Index(obs.index.astype(str), dtype=object)
    var = pd.DataFrame(index=pd.Index(source.var_names.astype(str), dtype=object))

    if "X_umap" in source.obsm:
        embedding = np.asarray(source.obsm["X_umap"], dtype=np.float32)
        embedding_source = "source_X_umap"
    else:
        embedding = source.obs[[config["pt"], config["activation"]]].to_numpy(np.float32)
        embedding = (embedding - embedding.mean(axis=0)) / np.maximum(embedding.std(axis=0), 1e-6)
        embedding_source = f"standardized_{config['pt']}_and_{config['activation']}"

    output = ad.AnnData(
        X=source.X.copy(),
        obs=obs,
        var=var,
        obsm={"X_umap": embedding},
    )
    output.uns[config["threshold"]] = float(source.uns[config["threshold"]])
    output.uns["dataset"] = dataset
    output.uns["matrix_semantics"] = "unchanged full-gene log-normalized expression from source continuum"

    output_dir = root / "results/virtual_perturbation_celloracle/compatible_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_MS_CSF_continuum_minimal.h5ad"
    output.write_h5ad(output_path, compression="gzip")

    ledger = {
        "dataset": dataset,
        "source_path": str(source_path.relative_to(root)),
        "output_path": str(output_path.relative_to(root)),
        "n_cells": int(output.n_obs),
        "n_genes": int(output.n_vars),
        "n_donors": int(output.obs["subject_id"].astype(str).nunique()),
        "threshold_key": config["threshold"],
        "threshold_value": float(output.uns[config["threshold"]]),
        "embedding_source": embedding_source,
        "expression_checksum_sum": float(output.X.sum()),
        "expression_checksum_squared_sum": float(output.X.multiply(output.X).sum())
        if hasattr(output.X, "multiply") else float(np.square(output.X).sum()),
    }
    del source, output
    return ledger


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows = [prepare_dataset(root, dataset) for dataset in args.datasets]
    ledger = pd.DataFrame(rows)
    output = root / "results/virtual_perturbation_celloracle/compatible_inputs"
    ledger.to_csv(output / "compatible_input_ledger.csv", index=False)
    with (output / "compatible_input_ledger.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    print(ledger.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
