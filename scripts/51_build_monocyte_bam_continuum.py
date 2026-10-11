#!/usr/bin/env python3
"""Build donor-balanced monocyte-to-BAM continuum objects for each cohort."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats


DATASETS = ["GSE138266", "GSE194078", "GSE286068"]

MODULES = {
    "monocyte_entry": ["CCR2", "FCN1", "S100A8", "S100A9", "VCAN", "CTSS", "LILRB1"],
    "BAM_identity": ["C1QA", "C1QB", "C1QC", "APOE", "TREM2", "DAB2", "MSR1", "CD163", "MRC1", "LYVE1"],
    "lipid_lysosome": ["SPP1", "GPNMB", "LPL", "FABP5", "LGALS3", "LIPA", "CTSB", "CTSD", "CTSL", "FN1"],
    "MHC_I_processing": ["B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5"],
    "MHC_II_processing": ["CD74", "CIITA", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"],
    "CSF1_GMCSF_response": ["CSF1R", "CSF2RA", "CSF2RB", "SPI1", "MAFB", "MAF", "STAT5A", "STAT5B", "CISH"],
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

STATE_ORDER = {
    "PBMC_classical_monocyte": 0,
    "PBMC_nonclassical_monocyte": 1,
    "CSF_classical_monocyte": 2,
    "CSF_nonclassical_monocyte": 3,
    "CSF_APC_ambiguous": 4,
    "CSF_BAM_intermediate": 5,
    "CSF_BAM_terminal": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cap-per-donor-state", type=int, default=500)
    parser.add_argument("--n-hvgs", type=int, default=2500)
    parser.add_argument("--n-pcs", type=int, default=40)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def load_labels(root: Path, dataset: str, obs_names: pd.Index) -> pd.DataFrame:
    path = root / "results" / "bam_robustness" / f"{dataset}_BAM_cell_stability.csv.gz"
    labels = pd.read_csv(path, index_col=0)
    if not obs_names.isin(labels.index).all():
        missing = obs_names[~obs_names.isin(labels.index)]
        raise RuntimeError(f"{dataset}: {len(missing)} APC cells lack BAM stability labels")
    labels = labels.reindex(obs_names)
    for column in ["bam_reference", "bam_monocyte_excluded", "bam_stable_core"]:
        labels[column] = boolean_series(labels[column])
    return labels


def assign_transition_state(obs: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    tissue = obs["tissue"].astype(str).str.upper()
    cell_type = obs["fine_apc_label"].astype(str)
    state = pd.Series("off_axis", index=obs.index, dtype="object")
    state[(tissue == "PBMC") & cell_type.eq("classical_monocyte")] = "PBMC_classical_monocyte"
    state[(tissue == "PBMC") & cell_type.eq("nonclassical_monocyte")] = "PBMC_nonclassical_monocyte"
    state[(tissue == "CSF") & cell_type.eq("classical_monocyte")] = "CSF_classical_monocyte"
    state[(tissue == "CSF") & cell_type.eq("nonclassical_monocyte")] = "CSF_nonclassical_monocyte"
    state[(tissue == "CSF") & cell_type.eq("APC_ambiguous")] = "CSF_APC_ambiguous"
    state[(tissue == "CSF") & labels["bam_reference"] & ~labels["bam_stable_core"]] = "CSF_BAM_intermediate"
    state[(tissue == "CSF") & labels["bam_stable_core"]] = "CSF_BAM_terminal"
    return state


def donor_balanced_indices(obs: pd.DataFrame, cap: int, seed: int) -> pd.Index:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for _, frame in obs.groupby(["subject_id", "transition_state"], observed=True, sort=True):
        names = frame.index.to_numpy()
        if len(names) > cap:
            names = rng.choice(names, size=cap, replace=False)
        selected.extend(names.tolist())
    return pd.Index(selected)


def copy_raw_subset(adata: ad.AnnData, selected: pd.Index) -> ad.AnnData:
    if adata.raw is None:
        raise RuntimeError("APC object lacks its full-gene log-normalized raw matrix")
    positions = adata.obs_names.get_indexer(selected)
    if np.any(positions < 0):
        raise RuntimeError("Selected cells are not present in the APC object")
    matrix = adata.raw.X[positions, :]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    matrix = matrix.copy()
    obs = adata.obs.iloc[positions].copy()
    var = adata.raw.var.copy()
    return ad.AnnData(X=matrix, obs=obs, var=var)


def add_module_scores(adata: ad.AnnData) -> None:
    for name, genes in MODULES.items():
        available = [gene for gene in genes if gene in adata.var_names]
        matrix = adata[:, available].X
        score = np.asarray(matrix.mean(axis=1)).ravel()
        adata.obs[f"module_{name}"] = score
        adata.uns[f"module_genes_{name}"] = available


def technical_gene_mask(var_names: pd.Index) -> np.ndarray:
    names = var_names.astype(str)
    return ~names.str.match(r"^(MT-|RPS|RPL|HBA|HBB|IG[HKL]|TR[ABDG])")


def select_hvgs(adata: ad.AnnData, n_top: int, exclude_markers: bool) -> list[str]:
    temporary = adata.copy()
    sc.pp.highly_variable_genes(temporary, flavor="seurat", n_top_genes=n_top, inplace=True)
    genes = temporary.var_names[temporary.var["highly_variable"].to_numpy()].tolist()
    allowed = set(adata.var_names[technical_gene_mask(adata.var_names)])
    genes = [gene for gene in genes if gene in allowed]
    if exclude_markers:
        excluded = {
            gene
            for collection in list(MODULES.values()) + list(ANNOTATION_MARKERS.values())
            for gene in collection
        }
        genes = [gene for gene in genes if gene not in excluded]
    del temporary
    if len(genes) < 500:
        raise RuntimeError(f"Only {len(genes)} usable HVGs remain")
    return genes


def compute_pca(adata: ad.AnnData, genes: list[str], n_pcs: int, key: str) -> None:
    work = adata[:, genes].copy()
    sc.pp.scale(work, max_value=10)
    components = min(n_pcs, work.n_obs - 1, work.n_vars - 1)
    sc.tl.pca(work, n_comps=components, svd_solver="arpack", random_state=12345)
    adata.obsm[key] = work.obsm["X_pca"].copy()
    adata.uns[f"{key}_variance_ratio"] = work.uns["pca"]["variance_ratio"].copy()
    del work


def choose_root(adata: ad.AnnData, pca_key: str) -> str:
    preferred = adata.obs["transition_state"].astype(str).eq("PBMC_classical_monocyte")
    if not preferred.any():
        preferred = adata.obs["transition_state"].astype(str).eq("CSF_classical_monocyte")
    if not preferred.any():
        raise RuntimeError("No monocyte root candidates")
    candidates = np.flatnonzero(preferred.to_numpy())
    coordinates = adata.obsm[pca_key][candidates, : min(20, adata.obsm[pca_key].shape[1])]
    centroid = np.median(coordinates, axis=0)
    medoid = candidates[np.argmin(np.square(coordinates - centroid).sum(axis=1))]
    return str(adata.obs_names[medoid])


def run_graph(
    adata: ad.AnnData,
    pca_key: str,
    suffix: str,
    root_cell: str,
    n_neighbors: int,
    seed: int,
) -> None:
    neighbors_key = f"neighbors_{suffix}" if suffix else None
    neighbor_kwargs = {
        "n_neighbors": min(n_neighbors, adata.n_obs - 1),
        "n_pcs": min(30, adata.obsm[pca_key].shape[1]),
        "use_rep": pca_key,
        "random_state": seed,
    }
    if neighbors_key is not None:
        neighbor_kwargs["key_added"] = neighbors_key
    sc.pp.neighbors(adata, **neighbor_kwargs)
    sc.tl.umap(adata, neighbors_key=neighbors_key, random_state=seed)
    adata.obsm[f"X_umap{('_' + suffix) if suffix else ''}"] = adata.obsm["X_umap"].copy()
    sc.tl.diffmap(adata, neighbors_key=neighbors_key, n_comps=15, random_state=seed)
    adata.uns["iroot"] = int(adata.obs_names.get_loc(root_cell))
    sc.tl.dpt(adata, neighbors_key=neighbors_key, n_dcs=10)
    adata.obs[f"dpt_pseudotime{('_' + suffix) if suffix else ''}"] = adata.obs["dpt_pseudotime"].copy()
    adata.obsm[f"X_diffmap{('_' + suffix) if suffix else ''}"] = adata.obsm["X_diffmap"].copy()
    if not suffix:
        sc.tl.leiden(
            adata,
            neighbors_key=neighbors_key,
            resolution=0.6,
            key_added="leiden_continuum_0_6",
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=seed,
        )
        sc.tl.paga(adata, groups="transition_state", neighbors_key=neighbors_key)


def state_summary(adata: ad.AnnData, dataset: str) -> pd.DataFrame:
    columns = [
        "dpt_pseudotime",
        "dpt_pseudotime_marker_excluded",
        *[f"module_{name}" for name in MODULES],
    ]
    frame = adata.obs.reset_index(names="cell_id")
    summary = (
        frame.groupby(["transition_state", "tissue"], observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            n_subjects=("subject_id", "nunique"),
            **{f"median_{column}": (column, "median") for column in columns},
        )
        .reset_index()
    )
    summary.insert(0, "dataset", dataset)
    return summary


def topology_metrics(adata: ad.AnnData, dataset: str, root_cell: str) -> pd.DataFrame:
    primary = adata.obs["dpt_pseudotime"].to_numpy(float)
    sensitivity = adata.obs["dpt_pseudotime_marker_excluded"].to_numpy(float)
    order = adata.obs["state_order"].to_numpy(float)
    root = adata.obs["transition_state"].astype(str).str.endswith("classical_monocyte").to_numpy()
    if (adata.obs["transition_state"].astype(str) == "PBMC_classical_monocyte").any():
        root = adata.obs["transition_state"].astype(str).eq("PBMC_classical_monocyte").to_numpy()
    terminal = adata.obs["transition_state"].astype(str).eq("CSF_BAM_terminal").to_numpy()
    auc_primary = stats.mannwhitneyu(primary[terminal], primary[root], alternative="two-sided").statistic / (
        terminal.sum() * root.sum()
    )
    auc_sensitivity = stats.mannwhitneyu(
        sensitivity[terminal], sensitivity[root], alternative="two-sided"
    ).statistic / (terminal.sum() * root.sum())
    rows = [
        {
            "dataset": dataset,
            "root_cell": root_cell,
            "n_cells": adata.n_obs,
            "n_subjects": adata.obs["subject_id"].nunique(),
            "n_root": int(root.sum()),
            "n_terminal": int(terminal.sum()),
            "primary_marker_excluded_spearman": stats.spearmanr(primary, sensitivity).statistic,
            "primary_state_order_spearman": stats.spearmanr(primary, order).statistic,
            "marker_excluded_state_order_spearman": stats.spearmanr(sensitivity, order).statistic,
            "primary_terminal_vs_root_auc": auc_primary,
            "marker_excluded_terminal_vs_root_auc": auc_sensitivity,
            "primary_monocyte_entry_spearman": stats.spearmanr(
                primary, adata.obs["module_monocyte_entry"]
            ).statistic,
            "primary_BAM_identity_spearman": stats.spearmanr(
                primary, adata.obs["module_BAM_identity"]
            ).statistic,
        }
    ]
    return pd.DataFrame(rows)


def build_dataset(root: Path, dataset: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_path = root / "results" / "cleaned" / f"{dataset}_clean_apc_refined.h5ad"
    source = sc.read_h5ad(input_path)
    labels = load_labels(root, dataset, source.obs_names)
    source.obs["bam_reference"] = labels["bam_reference"].to_numpy()
    source.obs["bam_stable_core"] = labels["bam_stable_core"].to_numpy()
    source.obs["bam_resolution_count"] = labels["bam_resolution_count"].to_numpy()
    source.obs["bam_marker_margin_all"] = labels["bam_marker_margin_all"].to_numpy()
    source.obs["transition_state"] = assign_transition_state(source.obs, labels)
    eligible = source.obs["transition_state"].ne("off_axis")
    eligible_obs = source.obs.loc[eligible].copy()
    before = (
        eligible_obs.groupby(["subject_id", "transition_state"], observed=True)
        .size()
        .rename("n_before_balance")
        .reset_index()
    )
    selected = donor_balanced_indices(eligible_obs, args.cap_per_donor_state, args.seed)
    continuum = copy_raw_subset(source, selected)
    continuum.obs["transition_state"] = source.obs.loc[selected, "transition_state"].astype(str).to_numpy()
    continuum.obs["transition_state"] = pd.Categorical(
        continuum.obs["transition_state"],
        categories=[state for state in STATE_ORDER if state in continuum.obs["transition_state"].unique()],
        ordered=True,
    )
    continuum.obs["state_order"] = continuum.obs["transition_state"].astype(str).map(STATE_ORDER).astype(int)
    continuum.obs["trajectory_role"] = "intermediate"
    root_name = "PBMC_classical_monocyte" if dataset in ["GSE138266", "GSE194078"] else "CSF_classical_monocyte"
    continuum.obs.loc[continuum.obs["transition_state"].astype(str).eq(root_name), "trajectory_role"] = "root_candidate"
    continuum.obs.loc[
        continuum.obs["transition_state"].astype(str).eq("CSF_BAM_terminal"), "trajectory_role"
    ] = "terminal_candidate"
    continuum.obs["bam_reference"] = source.obs.loc[selected, "bam_reference"].to_numpy(bool)
    continuum.obs["bam_stable_core"] = source.obs.loc[selected, "bam_stable_core"].to_numpy(bool)
    continuum.obs["bam_resolution_count"] = source.obs.loc[selected, "bam_resolution_count"].to_numpy(int)
    continuum.obs["bam_marker_margin_all"] = source.obs.loc[selected, "bam_marker_margin_all"].to_numpy(float)
    continuum.uns["dataset"] = dataset
    continuum.uns["matrix_semantics"] = "full-gene log-normalized expression copied from APC raw"
    continuum.uns["sampling_cap_per_donor_state"] = args.cap_per_donor_state
    continuum.uns["random_seed"] = args.seed

    add_module_scores(continuum)
    primary_hvgs = select_hvgs(continuum, args.n_hvgs, exclude_markers=False)
    sensitivity_hvgs = select_hvgs(continuum, args.n_hvgs, exclude_markers=True)
    continuum.var["highly_variable_primary"] = continuum.var_names.isin(primary_hvgs)
    continuum.var["highly_variable_marker_excluded"] = continuum.var_names.isin(sensitivity_hvgs)
    compute_pca(continuum, primary_hvgs, args.n_pcs, "X_pca")
    compute_pca(continuum, sensitivity_hvgs, args.n_pcs, "X_pca_marker_excluded")
    root_cell = choose_root(continuum, "X_pca")
    continuum.uns["root_cell"] = root_cell
    run_graph(continuum, "X_pca", "", root_cell, args.n_neighbors, args.seed)
    primary_umap = continuum.obsm["X_umap"].copy()
    primary_diffmap = continuum.obsm["X_diffmap"].copy()
    primary_dpt = continuum.obs["dpt_pseudotime"].copy()
    run_graph(
        continuum,
        "X_pca_marker_excluded",
        "marker_excluded",
        root_cell,
        args.n_neighbors,
        args.seed,
    )
    continuum.obsm["X_umap"] = primary_umap
    continuum.obsm["X_diffmap"] = primary_diffmap
    continuum.obs["dpt_pseudotime"] = primary_dpt

    output_dir = root / "results" / "trajectory" / "continuum_objects"
    output_dir.mkdir(parents=True, exist_ok=True)
    continuum.write_h5ad(output_dir / f"{dataset}_monocyte_BAM_continuum.h5ad", compression="gzip")

    after = (
        continuum.obs.reset_index(names="cell_id")
        .groupby(["subject_id", "transition_state"], observed=True)
        .size()
        .rename("n_after_balance")
        .reset_index()
    )
    balance = before.merge(after, on=["subject_id", "transition_state"], how="left").fillna({"n_after_balance": 0})
    balance.insert(0, "dataset", dataset)
    summary = state_summary(continuum, dataset)
    metrics = topology_metrics(continuum, dataset, root_cell)
    del source, continuum
    return balance, summary, metrics


def write_report(path: Path, metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Monocyte-to-BAM continuum build report",
        "",
        "## Design",
        "",
        "Each cohort was processed independently. PBMC classical monocytes anchor the root in "
        "the paired cohorts; CSF classical monocytes anchor GSE286068. Reference BAM cells not "
        "meeting the strict-core definition remain as possible intermediates, while strict-core "
        "CSF BAM cells define terminal candidates. Sampling was capped within donor-state strata.",
        "",
        "Two graphs were constructed: a primary log-normalized HVG graph and a sensitivity graph "
        "that excludes all annotation markers and prespecified mechanism-module genes.",
        "",
        "## Topology quality control",
        "",
    ]
    for row in metrics.itertuples():
        lines.extend([
            f"### {row.dataset}",
            "",
            f"- {row.n_cells:,} balanced cells from {row.n_subjects} subjects; "
            f"{row.n_root:,} root and {row.n_terminal:,} terminal candidates.",
            f"- Primary versus marker-excluded DPT: Spearman r={row.primary_marker_excluded_spearman:.3f}.",
            f"- Ordered-state association: primary r={row.primary_state_order_spearman:.3f}; "
            f"marker-excluded r={row.marker_excluded_state_order_spearman:.3f}.",
            f"- Terminal-versus-root ordering AUC: primary={row.primary_terminal_vs_root_auc:.3f}; "
            f"marker-excluded={row.marker_excluded_terminal_vs_root_auc:.3f}.",
            f"- Primary DPT correlation with monocyte-entry={row.primary_monocyte_entry_spearman:.3f}; "
            f"with BAM identity={row.primary_BAM_identity_spearman:.3f}.",
            "",
        ])
    lines.extend([
        "## Interpretation gate",
        "",
        "A high root-to-terminal AUC alone is not evidence of ontogeny because endpoint labels "
        "help define the biological anchors. Advancement to consensus trajectory requires the "
        "marker-excluded graph to preserve ordering, intermediate states to occupy non-terminal "
        "pseudotime, and leave-one-donor-out analyses to retain the same topology.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "results" / "trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)
    balances = []
    summaries = []
    metrics = []
    for dataset in DATASETS:
        print(f"Building {dataset} continuum", flush=True)
        balance, summary, metric = build_dataset(root, dataset, args)
        balances.append(balance)
        summaries.append(summary)
        metrics.append(metric)
    balance = pd.concat(balances, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    metric = pd.concat(metrics, ignore_index=True)
    balance.to_csv(output_dir / "continuum_sampling_ledger.csv", index=False)
    summary.to_csv(output_dir / "continuum_state_summary.csv", index=False)
    metric.to_csv(output_dir / "continuum_topology_qc.csv", index=False)
    write_report(output_dir / "CONTINUUM_BUILD_REPORT.md", metric, summary)
    print(metric.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
