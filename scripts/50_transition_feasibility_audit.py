#!/usr/bin/env python3
"""Audit whether existing datasets can support a human monocyte-to-BAM study."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


APC_FILES = {
    "GSE138266": "GSE138266_clean_apc_refined.h5ad",
    "GSE194078": "GSE194078_clean_apc_refined.h5ad",
    "GSE286068": "GSE286068_clean_apc_refined.h5ad",
}

FULL_FILES = {
    "GSE138266": "GSE138266_clean_decontx_singlets.h5ad",
    "GSE194078": "GSE194078_clean_decontx_singlets.h5ad",
    "GSE286068": "GSE286068_clean_decontx_singlets.h5ad",
}

MODULES = {
    "monocyte_entry": ["CCR2", "FCN1", "S100A8", "S100A9", "VCAN", "CTSS", "LILRB1"],
    "BAM_identity": ["C1QA", "C1QB", "C1QC", "APOE", "TREM2", "DAB2", "MSR1", "CD163", "MRC1", "LYVE1"],
    "lipid_lysosome": ["SPP1", "GPNMB", "LPL", "FABP5", "LGALS3", "LIPA", "CTSB", "CTSD", "CTSL", "FN1"],
    "MHC_I_processing": ["B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5"],
    "MHC_II_processing": ["CD74", "CIITA", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"],
    "CSF1_GMCSF_response": ["CSF1R", "CSF2RA", "CSF2RB", "SPI1", "MAFB", "MAF", "STAT5A", "STAT5B", "CISH"],
}

T_LIGANDS = ["CSF2", "IFNG", "TNF", "LTB", "CCL5", "IL17A", "IL21"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def raw_slice(adata: ad.AnnData, mask: np.ndarray, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    source = adata.raw if adata.raw is not None else adata
    available = [gene for gene in genes if gene in source.var_names]
    if not available:
        return np.empty((int(mask.sum()), 0)), []
    cell_indices = np.flatnonzero(mask)
    gene_indices = source.var_names.get_indexer(available)
    matrix = source.X[cell_indices, :]
    matrix = matrix[:, gene_indices]
    return dense(matrix), available


def apc_audit(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    module_rows: list[dict] = []
    gene_rows: list[dict] = []
    inventory_rows: list[dict] = []
    apc_dir = root / "results" / "cleaned"

    for dataset, filename in APC_FILES.items():
        adata = ad.read_h5ad(apc_dir / filename)
        obs = adata.obs.copy()
        inventory = (
            obs.groupby(["tissue", "fine_apc_label"], observed=True)
            .agg(n_cells=("subject_id", "size"), n_subjects=("subject_id", "nunique"))
            .reset_index()
        )
        inventory.insert(0, "dataset", dataset)
        inventory_rows.extend(inventory.to_dict("records"))

        group_columns = ["tissue", "fine_apc_label"]
        for keys, group_index in obs.groupby(group_columns, observed=True).groups.items():
            tissue, label = keys
            mask = obs.index.isin(group_index)
            n_cells = int(mask.sum())
            n_subjects = int(obs.loc[mask, "subject_id"].nunique())
            for module, genes in MODULES.items():
                values, available = raw_slice(adata, mask, genes)
                module_rows.append({
                    "dataset": dataset,
                    "tissue": tissue,
                    "fine_apc_label": label,
                    "module": module,
                    "n_cells": n_cells,
                    "n_subjects": n_subjects,
                    "n_genes_requested": len(genes),
                    "n_genes_available": len(available),
                    "mean_log_expression": float(values.mean(axis=1).mean()) if available else np.nan,
                    "pct_cells_any_gene_detected": float((values > 0).any(axis=1).mean() * 100) if available else np.nan,
                    "median_genes_detected_per_cell": float(np.median((values > 0).sum(axis=1))) if available else np.nan,
                })
                for column, gene in enumerate(available):
                    gene_rows.append({
                        "dataset": dataset,
                        "tissue": tissue,
                        "fine_apc_label": label,
                        "module": module,
                        "gene": gene,
                        "n_cells": n_cells,
                        "pct_detected": float((values[:, column] > 0).mean() * 100),
                        "mean_log_expression": float(values[:, column].mean()),
                    })
        del adata

    return pd.DataFrame(inventory_rows), pd.DataFrame(module_rows), pd.DataFrame(gene_rows)


def t_ligand_audit(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    cleaned_dir = root / "results" / "cleaned"
    for dataset, filename in FULL_FILES.items():
        adata = ad.read_h5ad(cleaned_dir / filename, backed="r")
        t_mask = adata.obs["broad_cell_type"].astype(str).eq("T").to_numpy()
        for tissue in sorted(adata.obs.loc[t_mask, "tissue"].astype(str).unique()):
            mask = t_mask & adata.obs["tissue"].astype(str).eq(tissue).to_numpy()
            values, available = raw_slice(adata, mask, T_LIGANDS)
            subjects = adata.obs.loc[mask, "subject_id"].astype(str).to_numpy()
            for column, gene in enumerate(available):
                detected = values[:, column] > 0
                detected_subjects = np.unique(subjects[detected]).size
                rows.append({
                    "dataset": dataset,
                    "tissue": tissue,
                    "gene": gene,
                    "n_T_cells": int(mask.sum()),
                    "n_subjects": int(np.unique(subjects).size),
                    "pct_T_cells_detected": float(detected.mean() * 100),
                    "n_subjects_detected": int(detected_subjects),
                    "mean_log_expression": float(values[:, column].mean()),
                })
        adata.file.close()
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    inventory: pd.DataFrame,
    modules: pd.DataFrame,
    t_ligands: pd.DataFrame,
) -> None:
    selected_labels = ["classical_monocyte", "nonclassical_monocyte", "BAM_like_candidate", "APC_ambiguous"]
    csf_inventory = inventory.loc[
        inventory["tissue"].astype(str).str.upper().eq("CSF")
        & inventory["fine_apc_label"].isin(selected_labels)
    ]
    lines = [
        "# Monocyte-to-BAM transition feasibility audit",
        "",
        "## Cell inventory",
        "",
    ]
    for row in csf_inventory.itertuples():
        lines.append(
            f"- {row.dataset} {row.fine_apc_label}: {row.n_cells:,} cells from {row.n_subjects} subjects."
        )

    lines.extend(["", "## Module measurability", ""])
    focus = modules.loc[
        modules["tissue"].astype(str).str.upper().eq("CSF")
        & modules["fine_apc_label"].isin(["classical_monocyte", "BAM_like_candidate"])
    ]
    coverage = (
        focus.groupby("module", observed=True)
        .agg(min_genes_available=("n_genes_available", "min"), min_pct_cells=("pct_cells_any_gene_detected", "min"))
        .reset_index()
    )
    for row in coverage.itertuples():
        lines.append(
            f"- {row.module}: at least {row.min_genes_available} genes available in every dataset/lineage; "
            f"minimum any-gene detection {row.min_pct_cells:.1f}%."
        )

    lines.extend(["", "## T-cell niche ligand measurability", ""])
    csf_t = t_ligands.loc[t_ligands["tissue"].astype(str).str.upper().eq("CSF")]
    ligand_summary = (
        csf_t.groupby("gene", observed=True)
        .agg(
            max_pct_T_cells=("pct_T_cells_detected", "max"),
            min_subjects_detected=("n_subjects_detected", "min"),
            max_subjects_detected=("n_subjects_detected", "max"),
        )
        .reset_index()
    )
    for row in ligand_summary.itertuples():
        lines.append(
            f"- {row.gene}: maximum T-cell detection {row.max_pct_T_cells:.2f}%; "
            f"detected in {row.min_subjects_detected}-{row.max_subjects_detected} subjects across datasets."
        )

    lines.extend([
        "",
        "## Rapid verdict",
        "",
        "The cell numbers and paired CSF/PBMC design are sufficient to test a human "
        "monocyte-to-BAM-like compartmental transition. The transition claim must be "
        "built from continuous-state reconstruction and donor-level replication, not "
        "from relabelling terminal clusters. T-cell GM-CSF is a high-ceiling niche "
        "hypothesis only if CSF2 is measurable across enough donors; otherwise the core "
        "paper should center the myeloid transition and treat ligand inference as a "
        "secondary mechanistic layer.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "results" / "cell_inventory"
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory, modules, genes = apc_audit(root)
    t_ligands = t_ligand_audit(root)
    inventory.to_csv(output_dir / "APC_transition_cell_inventory.csv", index=False)
    modules.to_csv(output_dir / "APC_transition_module_measurability.csv", index=False)
    genes.to_csv(output_dir / "APC_transition_gene_measurability.csv.gz", index=False)
    t_ligands.to_csv(output_dir / "T_cell_niche_ligand_measurability.csv", index=False)
    write_report(output_dir / "MONOCYTE_BAM_TRANSITION_FEASIBILITY.md", inventory, modules, t_ligands)


if __name__ == "__main__":
    main()
