#!/usr/bin/env python3
"""Generate submission-grade Extended Data figures from frozen result tables."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MM = 1 / 25.4
EXPORT_FORMATS = ["svg", "pdf", "png", "tiff"]
LAYOUT_QA_DIR = None
LAYOUT = importlib.import_module("75_generate_manuscript_figures")

COLORS = {
    "ink": "#2B2D2F", "grey": "#7C858D", "light": "#D8DDE1", "pale": "#F2F4F5",
    "ms": "#B94B52", "entry": "#4E79A7", "transition": "#D99A3D",
    "terminal": "#3F8C7A", "probe": "#8064A2", "harm": "#B94B52",
}
DATASET_COLORS = {
    "GSE138266": "#4E79A7", "GSE194078": "#D99A3D",
    "GSE286068": "#3F8C7A", "GSE328265": "#8064A2",
}
DATASETS = list(DATASET_COLORS)

FIGURES = {
    1: ("Extended_Data_Figure_1_preprocessing_QC", 120),
    2: ("Extended_Data_Figure_2_annotation_robustness", 120),
    3: ("Extended_Data_Figure_3_trajectory_robustness", 125),
    4: ("Extended_Data_Figure_4_two_axis_sensitivity", 125),
    5: ("Extended_Data_Figure_5_cross_species_LAM_audit", 125),
    6: ("Extended_Data_Figure_6_receiver_sender_boundary", 125),
    7: ("Extended_Data_Figure_7_in_vivo_CRISPR_robustness", 130),
    8: ("Extended_Data_Figure_8_virtual_perturbation_audit", 130),
    9: ("Extended_Data_Figure_9_drug_target_structure_audit", 140),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--figures", nargs="*", type=int, default=list(FIGURES))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=EXPORT_FORMATS, default=EXPORT_FORMATS)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.titlesize": 7.4, "axes.labelsize": 7,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.8,
        "axes.linewidth": 0.65, "axes.spines.right": False, "axes.spines.top": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel(ax: plt.Axes, label: str, x: float = -0.13) -> None:
    ax.set_label(label)
    ax.text(x, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def clean(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(colors=COLORS["ink"])


def export(frame: pd.DataFrame, source: Path, name: str) -> None:
    frame.to_csv(source / f"{name}.csv", index=False)


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    LAYOUT.prepare_layout(fig)
    if LAYOUT_QA_DIR is not None:
        LAYOUT_QA_DIR.mkdir(parents=True, exist_ok=True)
        (LAYOUT_QA_DIR / f"{stem}.json").write_text(json.dumps(LAYOUT.layout_metrics(fig), indent=2))
    for extension in EXPORT_FORMATS:
        options = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        if extension == "png":
            options["dpi"] = 300
        if extension == "pdf":
            options["dpi"] = 600
        fig.savefig(output / f"{stem}.{extension}", **options)
    plt.close(fig)


def grid_figure(height_mm: int) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, (height_mm + 15) * MM))
    fig.subplots_adjust(left=0.165, right=0.965, bottom=0.155, top=0.895, wspace=0.68, hspace=0.82)
    return fig, axes.ravel()


def jitter(n: int, center: float, width: float = 0.09, seed: int = 20260903) -> np.ndarray:
    return center + np.random.default_rng(seed + n + int(center * 19)).uniform(-width, width, n)


def ed1(root: Path, output: Path, source: Path) -> None:
    frames = []
    for dataset in DATASETS:
        frame = pd.read_csv(root / f"results/qc/{dataset}_cleanup_sample_summary.csv")
        frame["dataset"] = dataset
        frame["n_input_qc"] = frame.get("n_pre_doublet_qc", frame.get("n_qc_pass"))
        frames.append(frame)
    qc = pd.concat(frames, ignore_index=True)
    fig, ax = grid_figure(120)

    totals = qc.groupby("dataset", observed=True).agg(
        n_input_qc=("n_input_qc", "sum"), n_singlets=("n_singlets", "sum"), n_samples=("sample_accession", "nunique")
    ).reindex(DATASETS).reset_index()
    totals["statistical_unit"] = "sample aggregated within cohort"
    export(totals, source, "EDFig1a_cell_retention")
    x = np.arange(4)
    ax[0].bar(x - 0.17, totals.n_input_qc, 0.34, color=COLORS["light"], label="QC-pass input")
    ax[0].bar(x + 0.17, totals.n_singlets, 0.34, color=[DATASET_COLORS[d] for d in DATASETS], label="Retained singlets")
    ax[0].set_xticks(x, [d.replace("GSE", "") for d in DATASETS], rotation=25, ha="right")
    ax[0].set_ylabel("Cells")
    ax[0].set_title("Cell retention across four cohorts", loc="left", fontweight="bold")
    ax[0].legend()
    panel(ax[0], "a")

    doublet = qc[["dataset", "sample_accession", "pct_doublet_union"]].copy()
    doublet["statistical_unit"] = "sample"
    export(doublet, source, "EDFig1b_doublet_union_rates")
    for i, dataset in enumerate(DATASETS):
        values = doublet.loc[doublet.dataset.eq(dataset), "pct_doublet_union"].dropna().to_numpy()
        ax[1].scatter(jitter(len(values), i), values, s=14, color=DATASET_COLORS[dataset], alpha=0.8)
        ax[1].plot([i - 0.18, i + 0.18], [np.median(values)] * 2, color=COLORS["ink"], lw=1.1)
    ax[1].set_xticks(x, [d.replace("GSE", "") for d in DATASETS], rotation=25, ha="right")
    ax[1].set_ylabel("Union doublet rate (%)")
    ax[1].set_title("Independent-caller union by sample", loc="left", fontweight="bold")
    panel(ax[1], "b")

    contamination = qc[["dataset", "sample_accession", "median_decontX_contamination", "decontX_status"]].copy()
    contamination["statistical_unit"] = "sample"
    export(contamination, source, "EDFig1c_decontX_contamination")
    for i, dataset in enumerate(DATASETS):
        values = contamination.loc[contamination.dataset.eq(dataset), "median_decontX_contamination"].dropna().to_numpy()
        ax[2].scatter(jitter(len(values), i), 100 * values, s=14, color=DATASET_COLORS[dataset], alpha=0.8)
        ax[2].plot([i - 0.18, i + 0.18], [100 * np.median(values)] * 2, color=COLORS["ink"], lw=1.1)
    ax[2].set_xticks(x, [d.replace("GSE", "") for d in DATASETS], rotation=25, ha="right")
    ax[2].set_ylabel("Median DecontX contamination (%)")
    ax[2].set_title("Ambient-RNA estimates remain bounded", loc="left", fontweight="bold")
    panel(ax[2], "c")

    caller_rows = []
    for row in qc.itertuples():
        denominator = float(row.n_input_qc)
        for caller, count in [
            ("Scrublet", row.n_scrublet_doublets), ("scDblFinder", row.n_scDblFinder_doublets),
            ("Union", row.n_doublet_union),
        ]:
            caller_rows.append({"dataset": row.dataset, "sample_accession": row.sample_accession, "caller": caller, "pct_flagged": 100 * count / denominator})
    caller = pd.DataFrame(caller_rows)
    export(caller, source, "EDFig1d_doublet_caller_comparison")
    medians = caller.groupby(["dataset", "caller"], observed=True).pct_flagged.median().unstack().reindex(DATASETS)
    cx = np.arange(4)
    for j, method in enumerate(["Scrublet", "scDblFinder", "Union"]):
        ax[3].bar(cx + (j - 1) * 0.22, medians[method], width=0.21, color=[COLORS["entry"], COLORS["transition"], COLORS["ms"]][j], label=method)
    ax[3].set_xticks(cx, [d.replace("GSE", "") for d in DATASETS], rotation=25, ha="right")
    ax[3].set_ylabel("Median cells flagged (%)")
    ax[3].set_title("Caller agreement and conservative union", loc="left", fontweight="bold")
    ax[3].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.30), borderaxespad=0)
    panel(ax[3], "d")
    for axis in ax: clean(axis)
    save(fig, output, FIGURES[1][0])


def ed2(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(120)
    summary = pd.read_csv(root / "results/bam_robustness/BAM_robustness_summary.csv")
    summary["annotation_scope"] = "BAM-like confidence-supported annotation"
    export(summary, source, "EDFig2a_stable_core_recall")
    order = summary.gse.tolist()
    x = np.arange(len(order))
    ax[0].bar(x, 100 * summary.stable_core_recall, color=[DATASET_COLORS[d] for d in order])
    ax[0].set_ylim(0, 105)
    ax[0].set_xticks(x, [d.replace("GSE", "") for d in order])
    ax[0].set_ylabel("Stable-core recall (%)")
    ax[0].set_title("High-confidence BAM-like core retention", loc="left", fontweight="bold")
    panel(ax[0], "a")

    loo = []
    resolution = []
    for dataset in order:
        q = pd.read_csv(root / f"results/bam_robustness/{dataset}_BAM_marker_leave_one_out.csv")
        q["dataset"] = dataset
        loo.append(q)
        q = pd.read_csv(root / f"results/bam_robustness/{dataset}_BAM_multiresolution_overlap.csv")
        q["dataset"] = dataset
        resolution.append(q)
    loo = pd.concat(loo, ignore_index=True)
    resolution = pd.concat(resolution, ignore_index=True)
    export(loo, source, "EDFig2b_marker_leave_one_out")
    for i, dataset in enumerate(order):
        values = loo.loc[loo.dataset.eq(dataset) & loo.omitted_marker.ne("none"), "jaccard"].to_numpy()
        ax[1].scatter(jitter(len(values), i), values, color=DATASET_COLORS[dataset], s=16)
        ax[1].plot([i - .18, i + .18], [np.median(values)] * 2, color=COLORS["ink"], lw=1)
    ax[1].set_ylim(0, 1.03)
    ax[1].set_xticks(x, [d.replace("GSE", "") for d in order])
    ax[1].set_ylabel("Jaccard versus full marker set")
    ax[1].set_title("Marker leave-one-out robustness", loc="left", fontweight="bold")
    panel(ax[1], "b")

    export(resolution, source, "EDFig2c_multiresolution_overlap")
    for dataset in order:
        q = resolution.loc[resolution.dataset.eq(dataset)].sort_values("resolution")
        ax[2].plot(q.resolution, q.jaccard, marker="o", ms=3.5, color=DATASET_COLORS[dataset], label=dataset.replace("GSE", ""))
    ax[2].set_ylim(0, 1.03)
    ax[2].set_xlabel("Leiden resolution")
    ax[2].set_ylabel("Jaccard versus reference BAM-like set")
    ax[2].set_title("Clustering-resolution sensitivity", loc="left", fontweight="bold")
    ax[2].legend(ncol=3)
    panel(ax[2], "c")

    silhouette = pd.read_csv(root / "results/figure1c_annotation_audit/Figure1c_multicohort_annotation_metrics.csv")
    silhouette["annotation_scope"] = "confidence boundary, not discrete cell type"
    export(silhouette, source, "EDFig2d_annotation_silhouette_audit")
    metrics = [
        ("silhouette_strict_core_vs_reference_noncore", "Core vs\nnon-core"),
        ("silhouette_CSF_C1Q_transition_vs_BAM_like", "C1Q transition\nvs BAM-like"),
        ("silhouette_CSF_classical_vs_C1Q_transition", "CSF cMono\nvs C1Q transition"),
    ]
    for i, (column, label) in enumerate(metrics):
        for j, row in silhouette.iterrows():
            ax[3].scatter(i + (j - 1) * 0.10, row[column], s=25, color=DATASET_COLORS[row.dataset])
    ax[3].axhline(0, color=COLORS["grey"], lw=0.7)
    ax[3].set_xticks(range(3), [m[1] for m in metrics])
    ax[3].set_xlim(-0.4, 2.4)
    ax[3].set_ylabel("30-PC silhouette")
    ax[3].set_title("Confidence labels are not clean clusters", loc="left", fontweight="bold")
    panel(ax[3], "d")
    for axis in ax: clean(axis)
    save(fig, output, FIGURES[2][0])


def ed3(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(125)
    concordance = pd.read_csv(root / "results/trajectory/consensus_method_concordance.csv")
    concordance["statistical_scope"] = "cell-level rank concordance within cohort"
    export(concordance, source, "EDFig3a_method_concordance")
    methods = list(concordance.method_a.drop_duplicates())
    short = {m: m.replace("multiroot_DPT_", "DPT ").replace("primary_slingshot_terminal_ensemble", "Slingshot primary").replace("marker_excluded_slingshot_terminal_ensemble", "Slingshot held-out").replace("primary", "primary").replace("marker_excluded", "held-out") for m in methods}
    median = concordance.groupby(["method_a", "method_b"], observed=True).spearman_r.median().unstack().reindex(index=methods, columns=methods)
    image = ax[0].imshow(median, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax[0].set_xticks(range(len(methods)), [short[m] for m in methods], rotation=35, ha="right")
    ax[0].set_yticks(range(len(methods)), [short[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax[0].text(j, i, f"{median.iloc[i, j]:.2f}", ha="center", va="center", fontsize=4.8,
                       color="white" if median.iloc[i, j] > 0.75 else COLORS["ink"])
    fig.colorbar(image, ax=ax[0], fraction=.035, pad=.02).set_label("Median Spearman r")
    ax[0].set_title("Alternative trajectory representations agree", loc="left", fontweight="bold")
    panel(ax[0], "a", x=-0.23)

    topology = pd.read_csv(root / "results/trajectory/continuum_topology_qc.csv")
    topology["endpoint_scope"] = "high-confidence BAM-like endpoint versus monocyte root"
    export(topology, source, "EDFig3b_endpoint_topology")
    x = np.arange(len(topology))
    ax[1].bar(x - .16, topology.primary_terminal_vs_root_auc, .32, color=COLORS["entry"], label="Primary")
    ax[1].bar(x + .16, topology.marker_excluded_terminal_vs_root_auc, .32, color=COLORS["terminal"], label="Marker-held-out")
    ax[1].axhline(.5, color=COLORS["grey"], ls="--", lw=.7)
    ax[1].set_ylim(.45, 1.02)
    ax[1].set_xticks(x, topology.dataset.str.replace("GSE", ""))
    ax[1].set_ylabel("Endpoint-versus-root AUC")
    ax[1].set_title("Topology persists after marker holdout", loc="left", fontweight="bold")
    ax[1].legend()
    panel(ax[1], "b")

    jackknife = pd.read_csv(root / "results/trajectory/trajectory_donor_jackknife.csv")
    jackknife["statistical_unit"] = "held-out donor reconstruction"
    export(jackknife, source, "EDFig3c_donor_jackknife")
    positions, labels = [], []
    p = 0
    for dataset in topology.dataset:
        for representation, color in [("primary", COLORS["entry"]), ("marker_excluded", COLORS["terminal"])]:
            values = jackknife.loc[jackknife.dataset.eq(dataset) & jackknife.representation.eq(representation), "rebuilt_vs_full_spearman"].to_numpy()
            ax[2].scatter(jitter(len(values), p, width=.07), values, s=10, color=color, alpha=.65)
            ax[2].plot([p-.16, p+.16], [np.median(values)]*2, color=COLORS["ink"], lw=1)
            positions.append(p); labels.append(f"{dataset.replace('GSE','')}\n{'primary' if representation=='primary' else 'held-out'}")
            p += 1
        p += .35
    ax[2].set_xticks(positions, labels, rotation=25, ha="right")
    ax[2].set_ylabel("Rebuilt-versus-full Spearman r")
    ax[2].set_title("Leave-one-donor reconstruction", loc="left", fontweight="bold")
    panel(ax[2], "c")

    multiroot = pd.read_csv(root / "results/trajectory/multiroot_DPT_qc.csv")
    multiroot["root_definition"] = "multi-donor monocyte root"
    export(multiroot, source, "EDFig3d_multiroot_audit")
    x = np.arange(len(multiroot))
    ax[3].bar(x - .17, multiroot.primary_multiroot_state_order_spearman, .34, color=COLORS["entry"], label="State-order r")
    ax[3].bar(x + .17, multiroot.primary_marker_excluded_multiroot_spearman, .34, color=COLORS["terminal"], label="Marker-held-out r")
    ax[3].set_ylim(0, 1.05)
    ax[3].set_xticks(x, multiroot.dataset.str.replace("GSE", ""))
    ax[3].set_ylabel("Spearman r")
    ax[3].set_title("Multi-donor root sensitivity", loc="left", fontweight="bold")
    ax[3].legend(loc="upper left", bbox_to_anchor=(0, -0.26), borderaxespad=0, fontsize=5.4)
    panel(ax[3], "d")
    for axis in ax: clean(axis)
    save(fig, output, FIGURES[3][0])


def ed4(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(125)
    fit = pd.read_csv(root / "results/two_axis/control_manifold_fit_diagnostics.csv")
    fit["fit_scope"] = "control-cell differentiation model"
    export(fit, source, "EDFig4a_control_manifold_fit")
    fit["label"] = fit.dataset.str.replace("GSE", "") + " | " + fit.module
    ax[0].barh(np.arange(len(fit)), fit.control_r_squared, color=[DATASET_COLORS[d] for d in fit.dataset])
    ax[0].set_yticks(np.arange(len(fit)), fit.label)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Control-manifold R²")
    ax[0].set_title("Activation residual estimated in controls", loc="left", fontweight="bold")
    panel(ax[0], "a", x=-0.25)

    orth = pd.read_csv(root / "results/two_axis/two_axis_orthogonality.csv")
    orth["statistical_unit"] = "donor"
    export(orth, source, "EDFig4b_axis_orthogonality")
    value = float(orth.strata_adjusted_spearman.iloc[0])
    ax[1].bar([0], [value], width=.45, color=COLORS["grey"])
    ax[1].axhline(0, color=COLORS["ink"], lw=.7)
    ax[1].set_xlim(-.7, .7); ax[1].set_ylim(-1, 1)
    ax[1].set_xticks([0], ["Adjusted donor r"])
    ax[1].set_ylabel("Strata-adjusted Spearman r")
    ax[1].set_title("Differentiation and inflammation remain separable", loc="left", fontweight="bold")
    panel(ax[1], "b")

    thresholds = pd.read_csv(root / "results/two_axis/decoupling_threshold_meta_grid.csv")
    thresholds["sensitivity_scope"] = "prespecified pseudotime and control-activation cutoffs"
    export(thresholds, source, "EDFig4c_threshold_grid")
    matrix = thresholds.pivot(index="pseudotime_cutoff", columns="activation_control_quantile", values="combined_z").sort_index(ascending=False)
    image = ax[2].imshow(matrix, cmap="Reds", vmin=0, vmax=max(3, matrix.to_numpy().max()), aspect="auto")
    ax[2].set_xticks(range(matrix.shape[1]), [f"q{q:.2f}" for q in matrix.columns])
    ax[2].set_yticks(range(matrix.shape[0]), [f"pt<{p:.2f}" for p in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax[2].text(j, i, f"{matrix.iloc[i,j]:.1f}", ha="center", va="center", fontsize=5.0)
    fig.colorbar(image, ax=ax[2], fraction=.035, pad=.02).set_label("Meta z")
    ax[2].set_title("MS decoupling persists across cutoff grid", loc="left", fontweight="bold")
    panel(ax[2], "c", x=-0.20)

    disease = pd.read_csv(root / "results/two_axis/two_axis_disease_tests.csv")
    selected = disease.loc[disease.metric.isin([
        "continuous_decoupling_index", "decoupled_inflammatory_fraction",
        "mean_activation_residual", "median_consensus_pseudotime",
    ])].copy()
    selected["statistical_unit"] = "donor"
    export(selected, source, "EDFig4d_discovery_disease_effects")
    labels = {
        "continuous_decoupling_index": "Continuous decoupling", "decoupled_inflammatory_fraction": "Decoupled fraction",
        "mean_activation_residual": "Activation residual", "median_consensus_pseudotime": "Differentiation",
    }
    metrics = list(labels)
    for j, dataset in enumerate(sorted(selected.dataset.unique())):
        q = selected.loc[selected.dataset.eq(dataset)].set_index("metric").reindex(metrics)
        ax[3].scatter(q.standardized_mean_difference, np.arange(len(metrics)) + (j-.5)*.12, s=24, color=DATASET_COLORS[dataset], label=dataset.replace("GSE", ""))
    ax[3].axvline(0, color=COLORS["grey"], ls="--", lw=.7)
    ax[3].set_yticks(range(len(metrics)), [labels[m] for m in metrics])
    ax[3].invert_yaxis()
    ax[3].set_xlabel("Standardized MS-control difference")
    ax[3].set_title("Discovery-cohort two-axis disease effects", loc="left", fontweight="bold")
    ax[3].legend()
    panel(ax[3], "d", x=-0.24)
    for axis in ax: clean(axis)
    save(fig, output, FIGURES[4][0])


def ed5(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(125)
    fig.subplots_adjust(left=0.20, wspace=0.85)
    availability = pd.read_csv(root / "results/cross_species_lam/LAM_signature_availability.csv")
    availability["evidence_scope"] = "gene-program availability, not cell identity"
    export(availability, source, "EDFig5a_signature_availability")
    availability["fraction"] = availability.n_available / availability.n_requested
    sigs = list(availability.signature.drop_duplicates())
    matrix = availability.pivot(index="signature", columns="dataset", values="fraction").reindex(index=sigs, columns=DATASETS)
    image = ax[0].imshow(matrix, vmin=.7, vmax=1, cmap="YlGn", aspect="auto")
    ax[0].set_xticks(range(4), [d.replace("GSE", "") for d in DATASETS], rotation=25, ha="right")
    ax[0].set_yticks(range(len(sigs)), [s.replace("mouse_", "").replace("_", " ") for s in sigs])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax[0].text(j, i, f"{100*matrix.iloc[i,j]:.0f}%", ha="center", va="center", fontsize=4.8,
                       color="white" if matrix.iloc[i,j] >= 0.90 else COLORS["ink"])
    fig.colorbar(image, ax=ax[0], fraction=.035, pad=.02).set_label("Available fraction")
    ax[0].set_title("Cross-species programs are measurable", loc="left", fontweight="bold")
    panel(ax[0], "a", x=-0.23)

    fit = pd.read_csv(root / "results/cross_species_lam/LAM_control_manifold_diagnostics.csv")
    fit["model_scope"] = "differentiation adjustment"
    export(fit, source, "EDFig5b_LAM_control_fit")
    for i, signature in enumerate(sigs):
        q = fit.loc[fit.signature.eq(signature)].set_index("dataset").reindex(DATASETS)
        ax[1].plot(range(4), q.baseline_r_squared, marker="o", ms=3.5, label=signature.replace("mouse_", "").replace("_", " "))
    ax[1].set_xticks(range(4), [d.replace("GSE", "") for d in DATASETS], rotation=25, ha="right")
    ax[1].set_ylabel("Control-manifold R²")
    ax[1].set_title("Program scores are adjusted for differentiation", loc="left", fontweight="bold")
    ax[1].legend(fontsize=4.8, loc="upper left", bbox_to_anchor=(0, -0.30), borderaxespad=0)
    panel(ax[1], "b")

    correlations = pd.read_csv(root / "results/cross_species_lam/LAM_donor_trajectory_correlations.csv")
    correlations["statistical_unit"] = "donor"
    export(correlations, source, "EDFig5c_donor_trajectory_correlations")
    for i, signature in enumerate(sigs):
        values = correlations.loc[correlations.signature.eq(signature), "spearman_with_differentiation"].dropna().to_numpy()
        ax[2].scatter(jitter(len(values), i), values, s=9, alpha=.55, color=[COLORS["grey"], COLORS["terminal"], COLORS["transition"], COLORS["probe"]][i])
        ax[2].plot([i-.18, i+.18], [np.median(values)]*2, color=COLORS["ink"], lw=1)
    ax[2].axhline(0, color=COLORS["grey"], ls="--", lw=.7)
    ax[2].set_xticks(range(len(sigs)), [s.replace("mouse_", "").replace("_", " ") for s in sigs], rotation=25, ha="right")
    ax[2].set_ylabel("Within-donor Spearman r")
    ax[2].set_title("LAM programs map differently along trajectory", loc="left", fontweight="bold")
    panel(ax[2], "c")

    pair = pd.read_csv(root / "results/cross_species_lam/LAM_decoupled_pair_tests.csv")
    pair["program"] = pair.endpoint.str.replace("decoupled_excess__", "", regex=False)
    pair["evidence_scope"] = "differentiation-adjusted program excess"
    export(pair, source, "EDFig5d_decoupled_program_excess")
    programs = list(pair.program.drop_duplicates())
    for j, dataset in enumerate([d for d in DATASETS if d in pair.dataset.unique()]):
        q = pair.loc[pair.dataset.eq(dataset)].set_index("program").reindex(programs)
        ax[3].scatter(q.mean_decoupled_excess, np.arange(len(programs)) + (j-1)*.12, s=22, color=DATASET_COLORS[dataset], label=dataset.replace("GSE", ""))
    ax[3].axvline(0, color=COLORS["grey"], ls="--", lw=.7)
    ax[3].set_yticks(range(len(programs)), [p.replace("mouse_", "").replace("_", " ") for p in programs])
    ax[3].invert_yaxis()
    ax[3].set_xlabel("Decoupled minus other-incomplete excess")
    ax[3].set_title("Precursor excess is not terminal LAM identity", loc="left", fontweight="bold")
    ax[3].legend(ncol=3, fontsize=4.8, loc="upper left", bbox_to_anchor=(0,-0.32), borderaxespad=0)
    panel(ax[3], "d", x=-0.24)
    for axis in ax: clean(axis)
    save(fig, output, FIGURES[5][0])


def ed6(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(125)
    receptors = pd.read_csv(root / "results/niche_regulator/receiver_receptor_availability.csv")
    receptors["availability_fraction"] = receptors.available_receptors.str.count(";").add(1) / receptors.requested_receptors.str.count(";").add(1)
    receptors["evidence_scope"] = "receiver-component expression availability"
    export(receptors, source, "EDFig6a_receptor_availability")
    circuits = list(receptors.circuit.drop_duplicates())
    matrix = receptors.pivot(index="circuit", columns="dataset", values="availability_fraction").reindex(index=circuits, columns=DATASETS)
    ax[0].imshow(matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax[0].set_xticks(range(4), [d.replace("GSE", "") for d in DATASETS], rotation=25, ha="right")
    ax[0].set_yticks(range(len(circuits)), circuits)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax[0].text(j, i, f"{matrix.iloc[i,j]:.0%}", ha="center", va="center", fontsize=5,
                       color="white" if matrix.iloc[i,j] >= 0.90 else COLORS["ink"])
    ax[0].set_title("Receiver complexes are measurable", loc="left", fontweight="bold")
    panel(ax[0], "a", x=-0.20)

    audit = pd.read_csv(root / "results/niche_regulator/regulator_noncircularity_audit.csv")
    audit["holdout_scope"] = "Axis 2 and LAM targets removed before activity scoring"
    export(audit, source, "EDFig6b_regulator_target_holdout")
    q = audit.sort_values("n_targets_after_exclusion").tail(10)
    removed = q.n_axis2_targets_removed + q.n_LAM_targets_removed
    y = np.arange(len(q))
    ax[1].barh(y, q.n_targets_after_exclusion, color=COLORS["entry"], label="Retained")
    ax[1].barh(y, removed, left=q.n_targets_after_exclusion, color=COLORS["light"], label="Endpoint targets removed")
    ax[1].set_yticks(y, q.regulator)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("DoRothEA targets (log scale)")
    ax[1].set_title("Regulator activity survives endpoint holdout", loc="left", fontweight="bold")
    ax[1].legend()
    panel(ax[1], "b", x=-0.21)

    sender = pd.read_csv(root / "results/niche_regulator/sender_to_LAM_outcome_associations.csv")
    sender = sender.loc[
        sender.scope.eq("pooled_dataset_disease_residualized")
        & sender.y.eq("decoupled_excess__mouse_GMCSF_LAM_precursor")
    ].copy()
    sender["sender_display"] = sender.x.str.replace("sender_score__", "", regex=False)
    sender["evidence_scope"] = "pooled donor-level partial rank association"
    export(sender, source, "EDFig6c_sender_negative_control")
    sender = sender.sort_values("correlation")
    y = np.arange(len(sender))
    ax[2].scatter(sender.correlation, y, s=24, color=COLORS["grey"])
    ax[2].axvline(0, color=COLORS["ink"], ls="--", lw=.7)
    ax[2].set_yticks(y, sender.sender_display)
    ax[2].set_xlabel("Partial rank correlation")
    ax[2].set_title("Measured T-cell ligand abundance is negative", loc="left", fontweight="bold")
    panel(ax[2], "c", x=-0.24)

    regulator = pd.read_csv(root / "results/niche_regulator/regulator_delta_to_LAM_outcome_associations.csv")
    regulator = regulator.loc[
        regulator.scope.eq("pooled_dataset_disease_residualized")
        & regulator.y.eq("decoupled_excess__mouse_GMCSF_LAM_precursor")
    ].copy()
    regulator["regulator_display"] = regulator.x.str.replace("activity__", "", regex=False).str.replace("expression__", "", regex=False)
    regulator["evidence_scope"] = "pooled donor-level partial rank association"
    regulator = regulator.reindex(regulator.correlation.abs().sort_values(ascending=False).index).head(10)
    export(regulator, source, "EDFig6d_regulator_program_associations")
    regulator = regulator.sort_values("correlation")
    y = np.arange(len(regulator))
    ax[3].scatter(regulator.correlation, y, s=24, color=[COLORS["ms"] if q < .1 else COLORS["grey"] for q in regulator.fdr_bh_within_family])
    ax[3].axvline(0, color=COLORS["ink"], ls="--", lw=.7)
    ax[3].set_yticks(y, regulator.regulator_display)
    ax[3].set_xlabel("Partial rank correlation")
    ax[3].set_title("Regulator-to-precursor associations", loc="left", fontweight="bold")
    panel(ax[3], "d", x=-0.24)
    for axis in ax: clean(axis)
    save(fig, output, FIGURES[6][0])


def ed7(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(130)
    fig.subplots_adjust(left=0.155, wspace=0.58)
    source_qc = pd.read_csv(root / "results/in_vivo_crispr_bridge/external_signature_source_qc.csv")
    source_qc["source_scope"] = "independent Hoxb8FL and chimera EAE macrophage KO models"
    export(source_qc, source, "EDFig7a_source_model_concordance")
    order = ["Ifngr1", "Tnfrsf1a", "Csf2ra", "Tgfbr1"]
    display = {"Ifngr1":"IFNGR1", "Tnfrsf1a":"TNFR1", "Csf2ra":"CSF2RA", "Tgfbr1":"TGFBR1"}
    q = source_qc.set_index("perturbation").reindex(order)
    x = np.arange(4)
    ax[0].bar(x-.17, q.pearson_hoxb8_chimera, .34, color=COLORS["entry"], label="Pearson")
    ax[0].bar(x+.17, q.spearman_hoxb8_chimera, .34, color=COLORS["terminal"], label="Spearman")
    ax[0].set_xticks(x, [display[k] for k in order], rotation=20, ha="right")
    ax[0].set_ylim(0, .8)
    ax[0].set_ylabel("Cross-model concordance")
    ax[0].set_title("Independent KO systems agree", loc="left", fontweight="bold")
    ax[0].legend()
    panel(ax[0], "a")

    coverage = q.reset_index().copy()
    coverage["published_mapping_fraction"] = coverage.n_published_one_to_one_mapped / coverage.n_published_response_genes
    coverage["direct_mapping_fraction"] = coverage.n_direct_one_to_one_mapped / coverage.n_direct_genes
    coverage["mapping_scope"] = "one-to-one mouse-human orthologs"
    export(coverage, source, "EDFig7b_ortholog_coverage")
    ax[1].bar(x-.17, 100*coverage.published_mapping_fraction, .34, color=COLORS["ms"], label="Published response")
    ax[1].bar(x+.17, 100*coverage.direct_mapping_fraction, .34, color=COLORS["entry"], label="Direct KO all-gene")
    ax[1].set_xticks(x, [display[k] for k in order], rotation=20, ha="right")
    ax[1].set_ylim(0, 105)
    ax[1].set_ylabel("One-to-one ortholog coverage (%)")
    ax[1].set_title("Signatures remain powered after mapping", loc="left", fontweight="bold")
    ax[1].legend(loc="upper left", bbox_to_anchor=(0, -0.30), borderaxespad=0)
    panel(ax[1], "b")

    tests = pd.read_csv(root / "results/in_vivo_crispr_bridge/cohort_bridge_tests.csv")
    tests = tests.loc[
        tests.perturbation.eq("Ifngr1")
        & tests.endpoint.isin(["decoupled_minus_other_incomplete", "axis2_spearman"])
        & tests.dataset.isin(["GSE138266", "GSE286068", "GSE328265"])
    ].copy()
    tests["signature_display"] = tests.signature_type.map({"published_response":"Response", "direct_ko_consensus":"Direct KO-like"})
    tests["statistical_unit"] = "donor"
    export(tests, source, "EDFig7c_IFNGR1_cohort_effects")
    categories = [(d,e) for e in ["decoupled_minus_other_incomplete","axis2_spearman"] for d in ["GSE138266","GSE286068","GSE328265"]]
    y = np.arange(len(categories))
    for sig, offset, color, marker in [("published_response",.08,COLORS["ms"],"o"),("direct_ko_consensus",-.08,COLORS["entry"],"s")]:
        vals=[]
        for dataset, endpoint in categories:
            vals.append(tests.loc[tests.signature_type.eq(sig)&tests.dataset.eq(dataset)&tests.endpoint.eq(endpoint),"standardized_effect"].iloc[0])
        ax[2].scatter(vals, y+offset, s=22, color=color, marker=marker, label="Response" if sig.startswith("published") else "Direct KO-like")
    ax[2].axvline(0, color=COLORS["grey"], ls="--", lw=.7)
    ax[2].set_yticks(y, [f"{d.replace('GSE','')} | {'decoupled' if e.startswith('decoupled') else 'Axis 2'}" for d,e in categories])
    ax[2].invert_yaxis()
    ax[2].set_xlabel("Donor-standardized effect")
    ax[2].set_title("IFNGR1 direction reverses in every cohort", loc="left", fontweight="bold")
    ax[2].legend(ncol=2, loc="upper left", bbox_to_anchor=(0,-0.36), borderaxespad=0)
    panel(ax[2], "c", x=-0.24)

    loo = pd.read_csv(root / "results/in_vivo_crispr_bridge/leave_one_cohort_out_audit.csv")
    loo = loo.loc[
        loo.perturbation.eq("Ifngr1")
        & loo.endpoint.isin(["decoupled_minus_other_incomplete", "axis2_spearman", "pseudotime_spearman"])
    ].copy()
    loo["audit_scope"] = "post hoc leave-one-cohort-out robustness audit"
    export(loo, source, "EDFig7d_leave_one_cohort_out")
    groups = [
        ("published_response","decoupled_minus_other_incomplete","Response | decoupled"),
        ("published_response","axis2_spearman","Response | Axis 2"),
        ("direct_ko_consensus","decoupled_minus_other_incomplete","KO-like | decoupled"),
        ("direct_ko_consensus","axis2_spearman","KO-like | Axis 2"),
        ("direct_ko_consensus","pseudotime_spearman","KO-like | differentiation"),
    ]
    for i,(sig,endpoint,label) in enumerate(groups):
        vals=loo.loc[loo.signature_type.eq(sig)&loo.endpoint.eq(endpoint),"combined_z"].to_numpy()
        ax[3].scatter(vals, np.repeat(i,len(vals)), s=19, color=COLORS["ms"] if sig.startswith("published") else COLORS["entry"])
        ax[3].plot([vals.min(),vals.max()],[i,i],color=COLORS["grey"],lw=.8,zorder=0)
    ax[3].axvline(0,color=COLORS["ink"],ls="--",lw=.7)
    ax[3].set_yticks(range(len(groups)),[g[2] for g in groups])
    ax[3].invert_yaxis()
    ax[3].set_xlabel("Leave-one-cohort-out Stouffer z")
    ax[3].set_title("No cohort exclusion reverses IFNGR1", loc="left", fontweight="bold")
    panel(ax[3], "d", x=-0.25)
    for axis in ax: clean(axis)
    save(fig, output, FIGURES[7][0])


def ed8(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(130)
    fig.subplots_adjust(left=0.135, wspace=0.54)
    primary = pd.read_csv(root / "results/virtual_perturbation_celloracle/primary_endpoint_meta_analysis.csv")
    primary["evidence_scope"] = "virtual perturbation in fitted whole-cohort GRNs; effects aggregated by donor"
    export(primary, source, "EDFig8a_observed_vs_network_specific")
    for row in primary.itertuples():
        color = COLORS["ms"] if row.candidate in {"IRF1","STAT1"} else COLORS["terminal"]
        ax[0].scatter(row.combined_expected_z,row.combined_network_specific_z,s=25,color=color,edgecolor="white",linewidth=.3)
        if row.endpoint=="axis2": ax[0].text(row.combined_expected_z+.08,row.combined_network_specific_z+.08,row.candidate,fontsize=5)
    ax[0].axhline(0,color=COLORS["grey"],lw=.7); ax[0].axvline(0,color=COLORS["grey"],lw=.7)
    ax[0].set_xlabel("Observed-GRN expected-direction z")
    ax[0].set_ylabel("Specificity versus shuffled-GRN z")
    ax[0].set_title("Biological direction and network specificity differ",loc="left",fontweight="bold")
    panel(ax[0],"a",x=-.19)

    shuffled = pd.read_csv(root / "results/virtual_perturbation_celloracle/shuffled_GRN_empirical_tests.csv")
    shuffled = shuffled.loc[
        shuffled.candidate.isin(["IRF1","STAT1"])&shuffled.endpoint.eq("axis2")&shuffled.state.eq("decoupled")
    ].copy()
    shuffled["evidence_scope"] = "50 target-shuffled GRNs per cohort"
    export(shuffled,source,"EDFig8b_shuffled_network_intervals")
    shuffled["label"] = shuffled.dataset.str.replace("GSE","")+" | "+shuffled.candidate
    y=np.arange(len(shuffled))
    ax[1].hlines(y,shuffled.shuffled_q025,shuffled.shuffled_q975,color=COLORS["light"],lw=4,label="Shuffled 95% interval")
    ax[1].scatter(shuffled.observed_mean_effect,y,s=22,color=COLORS["ms"],label="Observed GRN",zorder=3)
    ax[1].axvline(0,color=COLORS["grey"],ls="--",lw=.7)
    ax[1].set_yticks(y,shuffled.label); ax[1].invert_yaxis()
    ax[1].set_xlabel("Axis 2 effect")
    ax[1].set_title("Observed effects versus target-shuffled nulls",loc="left",fontweight="bold")
    ax[1].legend()
    panel(ax[1],"b",x=-.24)

    harm=pd.read_csv(root/"results/virtual_perturbation_celloracle/harm_veto_meta_analysis.csv")
    harm["evidence_scope"]="prespecified harmful-direction test"
    export(harm,source,"EDFig8c_all_harm_vetoes")
    harm=harm.sort_values("combined_harmful_z")
    veto_labels = {
        "lam2_complement_apc": "LAM2", "trajectory_signed": "trajectory", "axis2": "Axis 2",
    }
    harm["label"] = harm.candidate + " | " + harm.veto_endpoint.map(veto_labels)
    y=np.arange(len(harm))
    ax[2].barh(y,harm.combined_harmful_z,color=[COLORS["harm"] if q<.1 else COLORS["light"] for q in harm.fdr_bh_veto_endpoints])
    ax[2].axvline(0,color=COLORS["grey"],lw=.7)
    ax[2].set_yticks(y,harm.label); ax[2].set_xlabel("Harmful-direction meta z")
    ax[2].set_title("Full prespecified harm-veto family",loc="left",fontweight="bold")
    panel(ax[2],"c",x=-.24)

    combo=pd.read_csv(root/"results/virtual_perturbation_celloracle_combinations/combination_criterion_meta_analysis.csv")
    combo=combo.loc[combo.criterion.eq("positive_lam2_rescue_vs_STAT1_KO")].copy()
    combo["intervention_display"]=combo.intervention.str.replace("STAT1_KO__","",regex=False).str.replace("_OE"," OE",regex=False).str.replace("__"," + ",regex=False)
    combo["evidence_scope"]="virtual additive rescue versus STAT1 KO"
    export(combo,source,"EDFig8d_compensatory_rescue")
    combo=combo.sort_values("combined_expected_z")
    y=np.arange(len(combo))
    ax[3].scatter(combo.combined_expected_z,y,s=28,color=COLORS["grey"],edgecolor=COLORS["ink"],linewidth=.3)
    ax[3].axvline(0,color=COLORS["grey"],ls="--",lw=.7)
    ax[3].set_yticks(y,combo.intervention_display)
    ax[3].set_xlabel("LAM2 rescue expected-direction z")
    ax[3].set_title("Adaptation-factor OE does not rescue STAT1 KO",loc="left",fontweight="bold")
    panel(ax[3],"d",x=-.25)
    for axis in ax: clean(axis)
    save(fig,output,FIGURES[8][0])


def ed9(root: Path, output: Path, source: Path) -> None:
    fig, ax = grid_figure(140)
    profiles=pd.read_csv(root/"results/transition_aware_drug_screen/nibr_full_profiles/NIBR_transition_candidate_profile_scores.csv")
    profiles["evidence_scope"]="24-hour U2OS perturbation profile"
    export(profiles,source,"EDFig9a_complete_NIBR_screen")
    colors=np.where(profiles.profile_advance,COLORS["probe"],np.where(profiles.adaptation_harm_veto,COLORS["harm"],COLORS["light"]))
    ax[0].scatter(profiles.lock_weighted_reversal_score,np.minimum(profiles.lam2_adaptation_score,profiles.trajectory_adaptation_score),s=18,c=colors,alpha=.85,edgecolor="white",linewidth=.25)
    hit=profiles.loc[profiles.profile_advance]
    if len(hit):
        row=hit.iloc[0]; ax[0].text(row.lock_weighted_reversal_score+.01,min(row.lam2_adaptation_score,row.trajectory_adaptation_score)+.01,f"{row.drug} {row.dose_um:g} uM",fontsize=5.3)
    ax[0].axhline(0,color=COLORS["grey"],ls="--",lw=.7); ax[0].axvline(0,color=COLORS["grey"],ls="--",lw=.7)
    ax[0].set_xlabel("Inflammatory-lock reversal")
    ax[0].set_ylabel("Minimum adaptation score")
    ax[0].set_title("Complete 76-profile phenotype screen",loc="left",fontweight="bold")
    panel(ax[0],"a")

    genetic=pd.read_csv(root/"results/transition_aware_drug_screen/genetic_deconvolution/genetic_target_cross_library_decisions.csv")
    genetic["evidence_scope"]="cross-library loss-of-function direction audit"
    export(genetic,source,"EDFig9b_genetic_target_deconvolution")
    genetic=genetic.sort_values(["n_libraries_advance","n_libraries_positive_mean_direction"],ascending=False)
    color_map={"advance_primary_genetic_support":COLORS["terminal"],"secondary_only_requires_primary_replication":COLORS["transition"],"no_genetic_support":COLORS["light"]}
    y=np.arange(len(genetic))
    ax[1].barh(y,genetic.n_libraries_positive_mean_direction,color=[color_map[d] for d in genetic.decision])
    ax[1].set_yticks(y,genetic.target); ax[1].invert_yaxis(); ax[1].set_xlim(0,3.2)
    ax[1].set_xlabel("Libraries with positive mean direction")
    ax[1].set_title("PDPK1 lacks advancing genetic support",loc="left",fontweight="bold")
    panel(ax[1],"b",x=-.20)

    neighbors=pd.read_csv(root/"results/transition_aware_drug_screen/mechanism_deconvolution/BX912_compound_level_directional_neighbors.csv")
    neighbors=neighbors.sort_values("compound_rank").head(100).copy()
    neighbors["pdpkl_annotated"] = neighbors.drh_target.fillna("").str.contains("PDPK1",case=False)
    neighbors["evidence_scope"]="unbiased compound-level directional neighborhood"
    export(neighbors,source,"EDFig9c_unbiased_neighbors")
    ax[2].scatter(neighbors.compound_rank,neighbors.directional_cosine,s=15,color=np.where(neighbors.pdpkl_annotated,COLORS["ms"],COLORS["grey"]),alpha=.75)
    ax[2].set_xlabel("Compound-level directional rank")
    ax[2].set_ylabel("Directional cosine to BX-912")
    ax[2].set_title("Top neighborhood lacks a PDPK1 cluster",loc="left",fontweight="bold")
    panel(ax[2],"c")

    redock=pd.read_csv(root/"results/docking/1Z5M_PDPK1_BX320_site/1Z5M_BX320_redocking_validation.csv")
    bx=pd.read_csv(root/"results/docking/1Z5M_PDPK1_BX912/BX912_pose_robustness.csv")
    docking=pd.concat([
        pd.DataFrame({"analysis":"BX-320 redocking","replicate":redock.pose_rank,"distance_A":redock.symmetry_aware_heavy_atom_rmsd_A,"criterion":"pose RMSD <2 A"}),
        pd.DataFrame({"analysis":"BX-912 pose recovery","replicate":np.arange(1,len(bx)+1),"distance_A":bx.minimum_ligand_nitrogen_to_ALA162_backbone_A,"criterion":"hinge distance <3.5 A"}),
    ],ignore_index=True)
    docking["evidence_scope"]="validated-site docking quality control"
    export(docking,source,"EDFig9d_structure_quality_control")
    for i,(name,color,threshold) in enumerate([("BX-320 redocking",COLORS["terminal"],2.0),("BX-912 pose recovery",COLORS["probe"],3.5)]):
        values=docking.loc[docking.analysis.eq(name),"distance_A"].dropna().to_numpy()
        ax[3].scatter(jitter(len(values),i),values,s=20,color=color,alpha=.8)
        ax[3].plot([i-.2,i+.2],[threshold,threshold],color=COLORS["ms"],ls="--",lw=1)
    ax[3].set_xticks([0,1],["BX-320\nredocking RMSD","BX-912\nhinge distance"])
    ax[3].set_ylabel("Distance (A)")
    ax[3].set_title("Control redocking passes; BX-912 pose does not",loc="left",fontweight="bold")
    panel(ax[3],"d")
    for axis in ax: clean(axis)
    save(fig,output,FIGURES[9][0])


def contact_sheet(output: Path) -> None:
    images=[]
    for number,(stem,_) in FIGURES.items():
        image=Image.open(output/f"{stem}.png").convert("RGB")
        image.thumbnail((700,500))
        images.append((number,image.copy()))
    row_height = 535
    canvas=Image.new("RGB",(1440,row_height * 5),"white")
    draw=ImageDraw.Draw(canvas)
    for idx,(number,image) in enumerate(images):
        row,col=divmod(idx,2)
        x,y=col*720,row*row_height
        canvas.paste(image,(x,y+18))
        draw.text((x+8,y+2),f"Extended Data Figure {number}",fill="black")
    canvas.save(output/"Extended_Data_contact_sheet.png",dpi=(150,150))


def main() -> None:
    global EXPORT_FORMATS, LAYOUT_QA_DIR
    args=parse_args(); root=args.root.resolve()
    output=args.output_dir or root/"figures_extended/exports"
    source=args.source_dir or root/"figures_extended/source_data"
    EXPORT_FORMATS=args.formats; LAYOUT_QA_DIR=args.qa_dir
    source.mkdir(parents=True,exist_ok=True); configure_style()
    functions={1:ed1,2:ed2,3:ed3,4:ed4,5:ed5,6:ed6,7:ed7,8:ed8,9:ed9}
    for number in args.figures:
        functions[number](root,output,source)
        print(f"Generated Extended Data Figure {number}",flush=True)
    if set(args.figures)==set(FIGURES) and "png" in EXPORT_FORMATS: contact_sheet(output)


if __name__=="__main__":
    main()
