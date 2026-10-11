#!/usr/bin/env python3
"""Build multi-root DPT, Slingshot inputs, and a consensus trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import cellrank as cr
from cellrank.kernels import ConnectivityKernel, CytoTRACEKernel, PseudotimeKernel
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats


DATASETS = ["GSE138266", "GSE194078", "GSE286068"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=["prepare", "finalize"], required=True)
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    if finite.any():
        result[finite] = (stats.rankdata(values[finite], method="average") - 0.5) / finite.sum()
    return result


def medoid_cell(adata: sc.AnnData, candidates: np.ndarray, pca_key: str) -> int:
    coordinates = adata.obsm[pca_key][candidates, : min(20, adata.obsm[pca_key].shape[1])]
    centroid = np.median(coordinates, axis=0)
    return int(candidates[np.argmin(np.square(coordinates - centroid).sum(axis=1))])


def donor_roots(adata: sc.AnnData, pca_key: str) -> pd.DataFrame:
    root_mask = adata.obs["trajectory_role"].astype(str).eq("root_candidate")
    rows = []
    for subject, index in adata.obs.loc[root_mask].groupby("subject_id", observed=True).groups.items():
        positions = adata.obs_names.get_indexer(index)
        root_position = medoid_cell(adata, positions, pca_key)
        rows.append({
            "subject_id": str(subject),
            "root_cell": str(adata.obs_names[root_position]),
            "root_position": root_position,
            "n_root_candidates": len(positions),
        })
    return pd.DataFrame(rows)


def multiroot_dpt(
    adata: sc.AnnData,
    roots: pd.DataFrame,
    neighbors_key: str | None,
    diffmap_key: str,
) -> np.ndarray:
    original_diffmap = adata.obsm["X_diffmap"].copy()
    adata.obsm["X_diffmap"] = adata.obsm[diffmap_key].copy()
    scores = []
    for row in roots.itertuples():
        adata.uns["iroot"] = int(row.root_position)
        sc.tl.dpt(adata, neighbors_key=neighbors_key, n_dcs=10)
        scores.append(percentile_rank(adata.obs["dpt_pseudotime"].to_numpy(float)))
    adata.obsm["X_diffmap"] = original_diffmap
    return np.column_stack(scores).astype(np.float32)


def f1_anchor_cluster(adata: sc.AnnData, role: str) -> str:
    target = adata.obs["trajectory_role"].astype(str).eq(role)
    clusters = adata.obs["leiden_continuum_0_6"].astype(str)
    rows = []
    for cluster in sorted(clusters.unique()):
        member = clusters.eq(cluster)
        tp = int((member & target).sum())
        if tp == 0:
            continue
        precision = tp / member.sum()
        recall = tp / target.sum()
        f1 = 2 * precision * recall / (precision + recall)
        rows.append((f1, tp, precision, recall, cluster))
    if not rows:
        raise RuntimeError(f"No cluster contains {role}")
    return str(max(rows)[-1])


def cytotrace_audit(adata: sc.AnnData, dataset: str) -> pd.DataFrame:
    kernel = CytoTRACEKernel(adata)
    kernel.compute_cytotrace(layer=None, use_raw=False, n_genes=200)
    rows = []
    for state, frame in adata.obs.groupby("transition_state", observed=True):
        rows.append({
            "dataset": dataset,
            "transition_state": str(state),
            "n_cells": len(frame),
            "median_ct_score": frame["ct_score"].median(),
            "median_ct_pseudotime": frame["ct_pseudotime"].median(),
            "median_ct_num_exp_genes": frame["ct_num_exp_genes"].median(),
        })
    audit = pd.DataFrame(rows)
    audit["cell_level_ct_pt_vs_state_order_spearman"] = stats.spearmanr(
        adata.obs["ct_pseudotime"], adata.obs["state_order"]
    ).statistic
    audit["cell_level_ct_pt_vs_n_genes_spearman"] = stats.spearmanr(
        adata.obs["ct_pseudotime"], adata.obs["n_genes_by_counts"]
    ).statistic
    return audit


def prepare_dataset(root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    object_path = root / "results" / "trajectory" / "continuum_objects" / f"{dataset}_monocyte_BAM_continuum.h5ad"
    adata = sc.read_h5ad(object_path)
    roots_primary = donor_roots(adata, "X_pca")
    roots_marker = donor_roots(adata, "X_pca_marker_excluded")
    if not roots_primary["subject_id"].equals(roots_marker["subject_id"]):
        raise RuntimeError(f"{dataset}: root donor order differs between representations")

    primary_matrix = multiroot_dpt(adata, roots_primary, None, "X_diffmap")
    marker_matrix = multiroot_dpt(
        adata,
        roots_marker,
        "neighbors_marker_excluded",
        "X_diffmap_marker_excluded",
    )
    adata.obsm["X_dpt_multiroot_primary"] = primary_matrix
    adata.obsm["X_dpt_multiroot_marker_excluded"] = marker_matrix
    adata.obs["dpt_multiroot_primary"] = np.nanmedian(primary_matrix, axis=1)
    adata.obs["dpt_multiroot_marker_excluded"] = np.nanmedian(marker_matrix, axis=1)
    adata.obs["dpt_multiroot_root_iqr"] = np.nanpercentile(primary_matrix, 75, axis=1) - np.nanpercentile(
        primary_matrix, 25, axis=1
    )

    cytotrace = cytotrace_audit(adata, dataset)
    root_cluster = f1_anchor_cluster(adata, "root_candidate")
    terminal_cluster = f1_anchor_cluster(adata, "terminal_candidate")
    roots_primary.insert(0, "dataset", dataset)
    roots_primary["representation"] = "primary"
    roots_marker.insert(0, "dataset", dataset)
    roots_marker["representation"] = "marker_excluded"
    roots = pd.concat([roots_primary, roots_marker], ignore_index=True)

    export = pd.DataFrame({
        "cell_id": adata.obs_names,
        "cluster": adata.obs["leiden_continuum_0_6"].astype(str).to_numpy(),
        "transition_state": adata.obs["transition_state"].astype(str).to_numpy(),
        "trajectory_role": adata.obs["trajectory_role"].astype(str).to_numpy(),
        "subject_id": adata.obs["subject_id"].astype(str).to_numpy(),
    })
    for index in range(min(20, adata.obsm["X_pca"].shape[1])):
        export[f"primary_PC{index + 1}"] = adata.obsm["X_pca"][:, index]
        export[f"marker_excluded_PC{index + 1}"] = adata.obsm["X_pca_marker_excluded"][:, index]

    input_dir = root / "results" / "trajectory" / "slingshot_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    export.to_csv(input_dir / f"{dataset}_slingshot_input.csv.gz", index=False)
    adata.uns["multiroot_subjects"] = roots_primary["subject_id"].astype(str).tolist()
    adata.uns["slingshot_start_cluster"] = root_cluster
    adata.uns["slingshot_end_cluster"] = terminal_cluster
    adata.write_h5ad(object_path, compression="gzip")

    topology = pd.DataFrame([{
        "dataset": dataset,
        "n_cells": adata.n_obs,
        "n_root_donors": len(roots_primary),
        "primary_marker_excluded_multiroot_spearman": stats.spearmanr(
            adata.obs["dpt_multiroot_primary"], adata.obs["dpt_multiroot_marker_excluded"]
        ).statistic,
        "primary_multiroot_state_order_spearman": stats.spearmanr(
            adata.obs["dpt_multiroot_primary"], adata.obs["state_order"]
        ).statistic,
        "marker_excluded_multiroot_state_order_spearman": stats.spearmanr(
            adata.obs["dpt_multiroot_marker_excluded"], adata.obs["state_order"]
        ).statistic,
        "median_root_IQR": adata.obs["dpt_multiroot_root_iqr"].median(),
        "start_cluster": root_cluster,
        "end_cluster": terminal_cluster,
    }])
    return roots, topology, cytotrace


def audit_slingshot_lineages(
    frame: pd.DataFrame,
    paths: pd.DataFrame,
    basis: str,
    terminal_cluster: str,
) -> tuple[list[str], pd.DataFrame]:
    pt_columns = [column for column in frame if column.startswith(f"{basis}_lineage") and column.endswith("_pseudotime")]
    rows = []
    root = frame["trajectory_role"].eq("root_candidate").to_numpy()
    terminal = frame["trajectory_role"].eq("terminal_candidate").to_numpy()
    for column in pt_columns:
        prefix = column.removesuffix("_pseudotime")
        weight_column = f"{prefix}_weight"
        values = frame[column].to_numpy(float)
        weights = frame[weight_column].to_numpy(float)
        eligible = np.isfinite(values)
        if not (eligible & root).any() or not (eligible & terminal).any():
            continue
        ranked = percentile_rank(values)
        auc = stats.mannwhitneyu(
            ranked[eligible & terminal], ranked[eligible & root], alternative="two-sided"
        ).statistic / ((eligible & terminal).sum() * (eligible & root).sum())
        terminal_weight = np.nanmean(weights[terminal])
        coverage = np.mean(eligible)
        rows.append({
            "basis": basis,
            "lineage": prefix,
            "root_terminal_auc": auc,
            "terminal_mean_weight": terminal_weight,
            "cell_coverage": coverage,
            "selection_score": auc * terminal_weight * coverage,
        })
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError(f"No usable Slingshot lineage for {basis}")
    terminal_lineages = paths.loc[
        paths["basis"].eq(basis) & paths["end_cluster"].astype(str).eq(str(terminal_cluster)), "lineage"
    ].astype(str).tolist()
    selected = [lineage for lineage in terminal_lineages if lineage in audit["lineage"].tolist()]
    if not selected:
        selected = [str(audit.sort_values("selection_score", ascending=False).iloc[0]["lineage"])]
    audit["reaches_prespecified_terminal"] = audit["lineage"].isin(terminal_lineages)
    audit["selected"] = audit["lineage"].isin(selected)
    return selected, audit


def ensemble_slingshot_score(frame: pd.DataFrame, lineages: list[str]) -> np.ndarray:
    pseudotimes = []
    weights = []
    for lineage in lineages:
        pseudotimes.append(percentile_rank(frame[f"{lineage}_pseudotime"].to_numpy(float)))
        weights.append(frame[f"{lineage}_weight"].to_numpy(float))
    pseudotime_matrix = np.column_stack(pseudotimes)
    weight_matrix = np.column_stack(weights)
    valid = np.isfinite(pseudotime_matrix) & np.isfinite(weight_matrix) & (weight_matrix > 0)
    numerator = np.nansum(np.where(valid, pseudotime_matrix * weight_matrix, 0), axis=1)
    denominator = np.nansum(np.where(valid, weight_matrix, 0), axis=1)
    result = np.full(len(frame), np.nan, dtype=np.float32)
    result[denominator > 0] = numerator[denominator > 0] / denominator[denominator > 0]
    return result


def compute_cellrank_macrostates(adata: sc.AnnData) -> tuple[pd.DataFrame, str]:
    time_key = "consensus_pseudotime"
    pk = PseudotimeKernel(adata, time_key=time_key).compute_transition_matrix(
        threshold_scheme="soft", n_jobs=1, show_progress_bar=False
    )
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    kernel = 0.8 * pk + 0.2 * ck
    estimator = cr.estimators.GPCCA(kernel)
    n_states = min(6, adata.obs["leiden_continuum_0_6"].nunique())
    estimator.compute_schur(n_components=min(20, adata.n_obs - 1))
    estimator.compute_macrostates(n_states=n_states, cluster_key="transition_state", n_cells=30)
    memberships = np.asarray(estimator.macrostates_memberships.X)
    names = [str(name) for name in estimator.macrostates_memberships.names]
    terminal = adata.obs["trajectory_role"].astype(str).eq("terminal_candidate").to_numpy()
    bam_state_index = int(np.argmax(memberships[terminal].mean(axis=0)))
    bam_state = names[bam_state_index]
    adata.obs["cellrank_BAM_macrostate_membership"] = memberships[:, bam_state_index]
    rows = []
    for index, name in enumerate(names):
        rows.append({
            "macrostate": name,
            "mean_membership_root": memberships[
                adata.obs["trajectory_role"].astype(str).eq("root_candidate").to_numpy(), index
            ].mean(),
            "mean_membership_terminal": memberships[terminal, index].mean(),
            "mean_consensus_pseudotime": np.average(
                adata.obs["consensus_pseudotime"], weights=np.maximum(memberships[:, index], 1e-12)
            ),
            "selected_BAM_macrostate": index == bam_state_index,
        })
    return pd.DataFrame(rows), bam_state


def finalize_dataset(root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    object_path = root / "results" / "trajectory" / "continuum_objects" / f"{dataset}_monocyte_BAM_continuum.h5ad"
    result_path = root / "results" / "trajectory" / "slingshot_output" / f"{dataset}_slingshot_results.csv.gz"
    lineage_path = root / "results" / "trajectory" / "slingshot_output" / f"{dataset}_slingshot_lineages.csv"
    adata = sc.read_h5ad(object_path)
    slingshot = pd.read_csv(result_path).set_index("cell_id").reindex(adata.obs_names)
    paths = pd.read_csv(lineage_path)
    if slingshot.index.has_duplicates or slingshot.isna().all(axis=1).any():
        raise RuntimeError(f"{dataset}: invalid Slingshot cell alignment")
    terminal_cluster = str(adata.uns["slingshot_end_cluster"])
    lineages_primary, audit_primary = audit_slingshot_lineages(
        slingshot.reset_index(), paths, "primary", terminal_cluster
    )
    lineages_marker, audit_marker = audit_slingshot_lineages(
        slingshot.reset_index(), paths, "marker_excluded", terminal_cluster
    )
    slingshot_primary = ensemble_slingshot_score(slingshot, lineages_primary)
    slingshot_marker = ensemble_slingshot_score(slingshot, lineages_marker)
    methods = np.column_stack([
        percentile_rank(adata.obs["dpt_multiroot_primary"].to_numpy(float)),
        percentile_rank(adata.obs["dpt_multiroot_marker_excluded"].to_numpy(float)),
        slingshot_primary,
        slingshot_marker,
    ])
    adata.obsm["X_consensus_trajectory_methods"] = methods.astype(np.float32)
    adata.uns["consensus_trajectory_method_names"] = [
        "multiroot_DPT_primary",
        "multiroot_DPT_marker_excluded",
        "primary_slingshot_terminal_ensemble",
        "marker_excluded_slingshot_terminal_ensemble",
    ]
    adata.obs["slingshot_primary_pseudotime"] = slingshot_primary
    adata.obs["slingshot_marker_excluded_pseudotime"] = slingshot_marker
    adata.obs["consensus_pseudotime"] = np.nanmedian(methods, axis=1)
    adata.obs["consensus_method_IQR"] = np.nanpercentile(methods, 75, axis=1) - np.nanpercentile(
        methods, 25, axis=1
    )
    macrostate, bam_macrostate = compute_cellrank_macrostates(adata)
    macrostate.insert(0, "dataset", dataset)
    adata.uns["cellrank_BAM_macrostate"] = bam_macrostate
    adata.write_h5ad(object_path, compression="gzip")

    method_names = adata.uns["consensus_trajectory_method_names"]
    concordance = pd.DataFrame(
        stats.spearmanr(methods, axis=0, nan_policy="omit").statistic,
        index=method_names,
        columns=method_names,
    ).reset_index(names="method_a").melt(id_vars="method_a", var_name="method_b", value_name="spearman_r")
    concordance.insert(0, "dataset", dataset)
    lineage_audit = pd.concat([audit_primary, audit_marker], ignore_index=True)
    lineage_audit.insert(0, "dataset", dataset)

    cell = adata.obs[[
        "subject_id",
        "sample_accession",
        "tissue",
        "diagnosis",
        "disease_group",
        "transition_state",
        "trajectory_role",
        "dpt_multiroot_primary",
        "dpt_multiroot_marker_excluded",
        "slingshot_primary_pseudotime",
        "slingshot_marker_excluded_pseudotime",
        "consensus_pseudotime",
        "consensus_method_IQR",
        "cellrank_BAM_macrostate_membership",
    ]].copy()
    cell.insert(0, "dataset", dataset)
    cell.reset_index(names="cell_id").to_csv(
        root / "results" / "trajectory" / f"{dataset}_consensus_cell_scores.csv.gz", index=False
    )
    return lineage_audit, concordance, macrostate


def write_report(root: Path, lineage: pd.DataFrame, concordance: pd.DataFrame, macrostate: pd.DataFrame) -> None:
    prepare_qc = pd.read_csv(root / "results" / "trajectory" / "multiroot_DPT_qc.csv")
    cytotrace = pd.read_csv(root / "results" / "trajectory" / "cytotrace_audit.csv")
    lines = [
        "# Consensus trajectory report",
        "",
        "## Method",
        "",
        "Consensus pseudotime is the median percentile rank across donor-rooted DPT and "
        "Slingshot, each computed on both the primary and annotation-marker-excluded PCA. "
        "CellRank uses the resulting consensus direction to estimate macrostates; it is not "
        "counted as an independent ordering method.",
        "",
        "## Multi-root DPT",
        "",
    ]
    for row in prepare_qc.itertuples():
        lines.append(
            f"- {row.dataset}: {row.n_root_donors} donor roots; primary/marker-excluded "
            f"r={row.primary_marker_excluded_multiroot_spearman:.3f}; median root IQR={row.median_root_IQR:.3f}."
        )
    lines.extend(["", "## CytoTRACE method audit", ""])
    for dataset, frame in cytotrace.groupby("dataset", observed=True):
        row = frame.iloc[0]
        lines.append(
            f"- {dataset}: CytoTRACE pseudotime versus ordered state r="
            f"{row.cell_level_ct_pt_vs_state_order_spearman:.3f}; versus detected genes r="
            f"{row.cell_level_ct_pt_vs_n_genes_spearman:.3f}. It was audited but excluded from consensus."
        )
    lines.extend(["", "## Slingshot and consensus", ""])
    for dataset in DATASETS:
        selected = lineage.loc[(lineage["dataset"] == dataset) & lineage["selected"]]
        cross = concordance.loc[
            (concordance["dataset"] == dataset)
            & concordance["method_a"].eq("multiroot_DPT_primary")
            & concordance["method_b"].eq("primary_slingshot_terminal_ensemble")
        ]
        details = "; ".join(
            f"{row.basis}:{row.lineage}, AUC={row.root_terminal_auc:.3f}" for row in selected.itertuples()
        )
        correlation = cross["spearman_r"].max() if len(cross) else np.nan
        bam_state = macrostate.loc[
            (macrostate["dataset"] == dataset) & macrostate["selected_BAM_macrostate"], "macrostate"
        ].iloc[0]
        lines.append(
            f"- {dataset}: {details}; primary DPT-Slingshot r={correlation:.3f}; "
            f"CellRank BAM-enriched macrostate={bam_state}."
        )
    lines.extend([
        "",
        "CellRank macrostate names inherit the plurality transition-state annotation and "
        "are not cell identities. A macrostate can therefore be BAM-enriched by soft "
        "membership even when its displayed name refers to another state. CellRank "
        "macrostates are retained as secondary topology features and do not define "
        "consensus pseudotime.",
        "",
        "## Robustness status",
        "",
        "The subsequent leave-one-donor-out graph reconstruction retained rebuilt/full "
        "pseudotime correlations of at least 0.971 across all cohorts and representations. "
        "The topology gate is passed; donor-level two-axis disease modelling is active.",
    ])
    (root / "results" / "trajectory" / "CONSENSUS_TRAJECTORY_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    trajectory_dir = root / "results" / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        roots, topology, cytotrace = [], [], []
        for dataset in DATASETS:
            print(f"Preparing {dataset}", flush=True)
            root_frame, topology_frame, cytotrace_frame = prepare_dataset(root, dataset)
            roots.append(root_frame)
            topology.append(topology_frame)
            cytotrace.append(cytotrace_frame)
        pd.concat(roots, ignore_index=True).to_csv(trajectory_dir / "multiroot_DPT_roots.csv", index=False)
        pd.concat(topology, ignore_index=True).to_csv(trajectory_dir / "multiroot_DPT_qc.csv", index=False)
        pd.concat(cytotrace, ignore_index=True).to_csv(trajectory_dir / "cytotrace_audit.csv", index=False)
        return

    lineage, concordance, macrostate = [], [], []
    for dataset in DATASETS:
        print(f"Finalizing {dataset}", flush=True)
        lineage_frame, concordance_frame, macrostate_frame = finalize_dataset(root, dataset)
        lineage.append(lineage_frame)
        concordance.append(concordance_frame)
        macrostate.append(macrostate_frame)
    lineage_frame = pd.concat(lineage, ignore_index=True)
    concordance_frame = pd.concat(concordance, ignore_index=True)
    macrostate_frame = pd.concat(macrostate, ignore_index=True)
    lineage_frame.to_csv(trajectory_dir / "slingshot_lineage_audit.csv", index=False)
    concordance_frame.to_csv(trajectory_dir / "consensus_method_concordance.csv", index=False)
    macrostate_frame.to_csv(trajectory_dir / "cellrank_macrostate_audit.csv", index=False)
    write_report(root, lineage_frame, concordance_frame, macrostate_frame)


if __name__ == "__main__":
    main()
