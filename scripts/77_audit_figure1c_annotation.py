#!/usr/bin/env python3
"""Audit and revise the biological annotation used in Figure 1c."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_score
from umap import UMAP


ROOT = Path(__file__).resolve().parents[1]
MM = 1 / 25.4

COLORS = {
    "PBMC classical monocyte root": "#AEB6BC",
    "CSF classical monocyte entry": "#4E79A7",
    "CSF C1Q+ monocyte-like transition": "#D99A3D",
    "C1Q/APOE-high BAM-like state": "#3F8C7A",
    "PBMC nonclassical off-branch": "#D8DDE1",
    "Excluded ambiguous APC": "#ECEFF1",
}

SHORT_LABELS = {
    "PBMC classical monocyte root": "PBMC cMono root",
    "CSF classical monocyte entry": "CSF cMono entry",
    "CSF C1Q+ monocyte-like transition": "CSF C1Q+ transition",
    "C1Q/APOE-high BAM-like state": "C1Q/APOE-high BAM-like",
}

MARKERS = [
    "S100A8", "FCN1", "VCAN", "FCGR3A", "MS4A7", "C1QA", "C1QC",
    "APOE", "TREM2", "DAB2", "MRC1",
]


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, np.bytes_)) else value
        for value in values
    ], dtype=object)


def read_encoded(group: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(group, h5py.Dataset):
        return decode(group[:]) if group.dtype.kind in {"S", "O", "U"} else group[:]
    if group.attrs.get("encoding-type") == "categorical":
        categories = decode(group["categories"][:])
        codes = group["codes"][:]
        return np.asarray([
            categories[code] if code >= 0 else np.nan for code in codes
        ], dtype=object)
    if "values" in group:
        values = decode(group["values"][:])
        if "mask" in group:
            values = values.astype(object)
            values[group["mask"][:]] = np.nan
        return values
    raise ValueError(f"Unsupported h5ad encoding at {group.name}")


def load_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    obs_columns = [
        "transition_state", "fine_apc_label", "tissue", "subject_id",
        "leiden_continuum_0_6", "consensus_pseudotime", "bam_reference",
        "bam_stable_core", "bam_resolution_count", "bam_marker_margin_all",
        "module_monocyte_entry", "module_BAM_identity",
        "module_CSF1_GMCSF_response", "module_MHC_II_processing",
    ]
    with h5py.File(path, "r") as handle:
        obs = pd.DataFrame({
            column: read_encoded(handle["obs"][column]) for column in obs_columns
        })
        obs.insert(0, "cell_id", read_encoded(handle["obs"]["_index"]))
        current_umap = handle["obsm/X_umap"][:]
        pca = handle["obsm/X_pca"][:]
        genes = read_encoded(handle["var"]["_index"])
        shape = tuple(handle["X"].attrs["shape"])
        matrix = sparse.csr_matrix((
            handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]
        ), shape=shape)

    gene_index = {gene: index for index, gene in enumerate(genes)}
    available = [gene for gene in MARKERS if gene in gene_index]
    expression = pd.DataFrame(
        matrix[:, [gene_index[gene] for gene in available]].toarray(),
        columns=available,
    )
    for column in [
        "consensus_pseudotime", "bam_resolution_count", "bam_marker_margin_all",
        "module_monocyte_entry", "module_BAM_identity",
        "module_CSF1_GMCSF_response", "module_MHC_II_processing",
    ]:
        obs[column] = pd.to_numeric(obs[column], errors="coerce")
    obs["bam_reference"] = obs["bam_reference"].astype(bool)
    obs["bam_stable_core"] = obs["bam_stable_core"].astype(bool)
    return obs, current_umap, pca, matrix, expression


def assign_revised_labels(obs: pd.DataFrame) -> pd.Series:
    state = obs["transition_state"].astype(str)
    revised = pd.Series("Excluded ambiguous APC", index=obs.index, dtype="object")
    revised[state.eq("PBMC_classical_monocyte")] = "PBMC classical monocyte root"
    revised[state.eq("PBMC_nonclassical_monocyte")] = "PBMC nonclassical off-branch"
    revised[state.eq("CSF_classical_monocyte")] = "CSF classical monocyte entry"
    revised[state.eq("CSF_nonclassical_monocyte")] = "CSF C1Q+ monocyte-like transition"
    revised[obs["bam_reference"]] = "C1Q/APOE-high BAM-like state"
    return revised


def safe_silhouette(pca: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> float:
    if mask.sum() < 3 or len(np.unique(labels[mask])) < 2:
        return np.nan
    return float(silhouette_score(pca[mask, :30], labels[mask], metric="euclidean"))


def audit_metrics(obs: pd.DataFrame, pca: np.ndarray) -> dict[str, object]:
    bam = obs["bam_reference"].to_numpy(bool)
    stable = obs["bam_stable_core"].to_numpy(bool)
    state = obs["transition_state"].astype(str)
    stable_cluster = obs.loc[stable, "leiden_continuum_0_6"].mode().iloc[0]
    noncore = bam & ~stable
    noncore_cluster = obs.loc[noncore, "leiden_continuum_0_6"].mode().iloc[0]

    current_bam_labels = np.where(stable, "strict_core", "reference_noncore")
    transition_bam = state.isin(["CSF_nonclassical_monocyte"]).to_numpy() | bam
    transition_bam_labels = np.where(bam, "BAM_like", "CSF_C1Q_transition")
    entry_transition = state.isin([
        "CSF_classical_monocyte", "CSF_nonclassical_monocyte"
    ]).to_numpy()
    entry_transition_labels = state.to_numpy()
    return {
        "n_cells": int(len(obs)),
        "n_reference_BAM": int(bam.sum()),
        "n_strict_BAM_core": int(stable.sum()),
        "n_reference_BAM_noncore": int(noncore.sum()),
        "strict_core_fraction_of_reference_BAM": float(stable.sum() / bam.sum()),
        "strict_core_majority_leiden_cluster": str(stable_cluster),
        "strict_core_in_majority_cluster_fraction": float(
            (obs.loc[stable, "leiden_continuum_0_6"] == stable_cluster).mean()
        ),
        "noncore_majority_leiden_cluster": str(noncore_cluster),
        "noncore_in_majority_cluster_fraction": float(
            (obs.loc[noncore, "leiden_continuum_0_6"] == noncore_cluster).mean()
        ),
        "silhouette_strict_core_vs_reference_noncore": safe_silhouette(
            pca, bam, current_bam_labels
        ),
        "silhouette_CSF_C1Q_transition_vs_BAM_like": safe_silhouette(
            pca, transition_bam, transition_bam_labels
        ),
        "silhouette_CSF_classical_vs_C1Q_transition": safe_silhouette(
            pca, entry_transition, entry_transition_labels
        ),
        "n_CSF_C1Q_monocyte_like_transition": int(
            state.eq("CSF_nonclassical_monocyte").sum()
        ),
    }


def summarize_states(obs: pd.DataFrame, expression: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([obs, expression], axis=1)
    rows = []
    for label, group in frame.groupby("revised_annotation", sort=False):
        row = {
            "revised_annotation": label,
            "n_cells": len(group),
            "n_donors": group["subject_id"].nunique(),
            "median_consensus_pseudotime": group["consensus_pseudotime"].median(),
            "median_monocyte_entry": group["module_monocyte_entry"].median(),
            "median_BAM_identity": group["module_BAM_identity"].median(),
            "median_CSF1_GMCSF_response": group["module_CSF1_GMCSF_response"].median(),
            "median_MHC_II_processing": group["module_MHC_II_processing"].median(),
            "reference_BAM_fraction": group["bam_reference"].mean(),
            "strict_BAM_core_fraction": group["bam_stable_core"].mean(),
        }
        for gene in expression.columns:
            row[f"mean_{gene}"] = group[gene].mean()
            row[f"pct_{gene}_detected"] = 100 * group[gene].gt(0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_embedding(
    ax: plt.Axes,
    coordinates: np.ndarray,
    labels: pd.Series,
    order: list[str],
    title: str,
    direct_labels: bool = True,
) -> None:
    for label in order:
        mask = labels.eq(label).to_numpy()
        if not mask.any():
            continue
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=2.0,
            color=COLORS[label], alpha=0.48, linewidth=0, rasterized=True,
        )
        if direct_labels and label in SHORT_LABELS:
            center = np.nanmedian(coordinates[mask], axis=0)
            ax.text(
                center[0], center[1], SHORT_LABELS[label], fontsize=5.4,
                fontweight="bold", ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.8),
            )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontweight="bold")


def make_figure(
    obs: pd.DataFrame,
    current_umap: np.ndarray,
    branch_umap: np.ndarray,
    branch_mask: np.ndarray,
    expression: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(183 * MM, 124 * MM))
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.0, 0.9], hspace=0.40, wspace=0.9,
        left=0.055, right=0.985, bottom=0.12, top=0.95,
    )
    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:8])
    ax_c = fig.add_subplot(gs[0, 8:12])
    ax_d = fig.add_subplot(gs[1, 0:12])

    current_map = {
        "PBMC_classical_monocyte": "PBMC classical monocyte root",
        "PBMC_nonclassical_monocyte": "PBMC nonclassical off-branch",
        "CSF_classical_monocyte": "CSF classical monocyte entry",
        "CSF_nonclassical_monocyte": "CSF C1Q+ monocyte-like transition",
        "CSF_BAM_intermediate": "CSF C1Q+ monocyte-like transition",
        "CSF_BAM_terminal": "C1Q/APOE-high BAM-like state",
        "CSF_APC_ambiguous": "Excluded ambiguous APC",
    }
    current_labels = obs["transition_state"].astype(str).map(current_map)
    current_order = list(COLORS)
    plot_embedding(
        ax_a, current_umap, current_labels, current_order,
        "Current display reveals a confidence split", direct_labels=False,
    )
    ax_a.text(
        0.02, 0.02,
        "current 'intermediate' = reference BAM outside strict core",
        transform=ax_a.transAxes, fontsize=5.1, color="#7C858D",
    )

    branch_labels = obs.loc[branch_mask, "revised_annotation"].reset_index(drop=True)
    branch_order = [
        "PBMC classical monocyte root",
        "CSF classical monocyte entry",
        "CSF C1Q+ monocyte-like transition",
        "C1Q/APOE-high BAM-like state",
    ]
    plot_embedding(
        ax_b, branch_umap, branch_labels, branch_order,
        "Re-embedded root-to-BAM branch",
    )

    scatter = ax_c.scatter(
        branch_umap[:, 0], branch_umap[:, 1],
        c=obs.loc[branch_mask, "consensus_pseudotime"], cmap="viridis",
        s=2.0, alpha=0.55, linewidth=0, rasterized=True,
    )
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.set_title("Continuous consensus pseudotime", loc="left", fontweight="bold")
    cbar = fig.colorbar(scatter, ax=ax_c, fraction=0.035, pad=0.02)
    cbar.set_label("Consensus pseudotime", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    dot_labels = branch_order
    dot_frame = pd.concat([
        obs.loc[branch_mask, ["revised_annotation"]].reset_index(drop=True),
        expression.loc[branch_mask].reset_index(drop=True),
    ], axis=1)
    mean = dot_frame.groupby("revised_annotation")[MARKERS].mean().reindex(dot_labels)
    detected = (
        dot_frame.assign(**{gene: dot_frame[gene].gt(0) for gene in MARKERS})
        .groupby("revised_annotation")[MARKERS].mean().reindex(dot_labels) * 100
    )
    x = np.arange(len(MARKERS))
    y = np.arange(len(dot_labels))[::-1]
    for yi, label in zip(y, dot_labels):
        values = mean.loc[label].to_numpy(float)
        sizes = 7 + 55 * detected.loc[label].to_numpy(float) / 100
        ax_d.scatter(x, np.repeat(yi, len(x)), c=values, s=sizes, cmap="Blues", vmin=0, vmax=mean.to_numpy().max(), edgecolor="#667078", linewidth=0.25)
    ax_d.set_xticks(x, MARKERS, rotation=45, ha="right")
    ax_d.set_yticks(y, [SHORT_LABELS[label] for label in dot_labels])
    ax_d.set_xlim(-0.6, len(MARKERS) - 0.4)
    ax_d.set_title("Marker evidence for revised lineage-state names", loc="left", fontweight="bold")
    ax_d.set_xlabel("Color: mean log-normalized expression; size: fraction detected")
    ax_d.spines["left"].set_visible(False)
    ax_d.tick_params(axis="y", length=0)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold")

    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(output / f"Figure1c_annotation_audit.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    configure_style()
    output = ROOT / "results/figure1c_annotation_audit"
    output.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    summary_rows = []
    diagnostic = None
    for dataset in ["GSE138266", "GSE194078", "GSE286068"]:
        input_path = (
            ROOT / "results/trajectory/continuum_objects"
            / f"{dataset}_monocyte_BAM_continuum.h5ad"
        )
        dataset_obs, dataset_umap, dataset_pca, _, dataset_expression = load_data(input_path)
        dataset_obs["revised_annotation"] = assign_revised_labels(dataset_obs)
        dataset_metrics = audit_metrics(dataset_obs, dataset_pca)
        dataset_metrics["dataset"] = dataset
        metric_rows.append(dataset_metrics)
        dataset_summary = summarize_states(dataset_obs, dataset_expression)
        dataset_summary.insert(0, "dataset", dataset)
        summary_rows.append(dataset_summary)
        if dataset == "GSE194078":
            diagnostic = (
                dataset_obs, dataset_umap, dataset_pca, dataset_expression,
                dataset_metrics, dataset_summary,
            )

    if diagnostic is None:
        raise RuntimeError("GSE194078 diagnostic object was not loaded")
    obs, current_umap, pca, expression, metrics, summary = diagnostic
    branch_labels = {
        "PBMC classical monocyte root",
        "CSF classical monocyte entry",
        "CSF C1Q+ monocyte-like transition",
        "C1Q/APOE-high BAM-like state",
    }
    branch_mask = obs["revised_annotation"].isin(branch_labels).to_numpy()
    branch_umap = UMAP(
        n_neighbors=30, min_dist=0.35, metric="cosine", random_state=12345,
        n_jobs=1,
    ).fit_transform(pca[branch_mask, :30])

    obs["current_UMAP1"] = current_umap[:, 0]
    obs["current_UMAP2"] = current_umap[:, 1]
    obs["branch_UMAP1"] = np.nan
    obs["branch_UMAP2"] = np.nan
    obs.loc[branch_mask, "branch_UMAP1"] = branch_umap[:, 0]
    obs.loc[branch_mask, "branch_UMAP2"] = branch_umap[:, 1]

    composition = pd.crosstab(
        obs["leiden_continuum_0_6"], obs["revised_annotation"], normalize="index"
    ).reset_index()
    obs.to_csv(output / "GSE194078_Figure1c_revised_cell_annotations.csv.gz", index=False, compression="gzip")
    summary.to_csv(output / "GSE194078_Figure1c_revised_state_summary.csv", index=False)
    composition.to_csv(output / "GSE194078_Figure1c_leiden_composition.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(
        output / "Figure1c_multicohort_annotation_metrics.csv", index=False
    )
    pd.concat(summary_rows, ignore_index=True).to_csv(
        output / "Figure1c_multicohort_revised_state_summary.csv", index=False
    )
    (output / "GSE194078_Figure1c_annotation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(obs, current_umap, branch_umap, branch_mask, expression, output)
    print(json.dumps(metrics, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
