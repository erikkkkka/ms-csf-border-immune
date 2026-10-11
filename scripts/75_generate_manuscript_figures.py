#!/usr/bin/env python3
"""Generate the five frozen-evidence manuscript figures using Python only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MM = 1 / 25.4
EXPORT_FORMATS = ["svg", "pdf", "png", "tiff"]
LAYOUT_QA_DIR = None

COLORS = {
    "ink": "#2B2D2F",
    "grey": "#7C858D",
    "light_grey": "#D8DDE1",
    "pale_grey": "#F2F4F5",
    "control": "#78838D",
    "ms": "#B94B52",
    "ms_light": "#E9B8BA",
    "entry": "#4E79A7",
    "entry_light": "#C7D7E8",
    "transition": "#D99A3D",
    "transition_light": "#F1D7AA",
    "terminal": "#3F8C7A",
    "terminal_light": "#B9D8CF",
    "probe": "#8064A2",
    "probe_light": "#D7CBE3",
    "harm": "#B94B52",
    "safe": "#3F8C7A",
}

STATE_COLORS = {
    "PBMC_classical_monocyte": "#C9D0D5",
    "PBMC_nonclassical_monocyte": "#9DA8B0",
    "CSF_classical_monocyte": COLORS["entry"],
    "CSF_nonclassical_monocyte": "#79A6CF",
    "CSF_APC_ambiguous": "#B9B3A8",
    "CSF_BAM_intermediate": COLORS["transition"],
    "CSF_BAM_terminal": COLORS["terminal"],
}

STATE_LABELS = {
    "PBMC_classical_monocyte": "PBMC cMono",
    "PBMC_nonclassical_monocyte": "PBMC ncMono",
    "CSF_classical_monocyte": "CSF cMono",
    "CSF_nonclassical_monocyte": "CSF ncMono",
    "CSF_APC_ambiguous": "APC ambiguous",
    "CSF_BAM_intermediate": "BAM-like intermediate",
    "CSF_BAM_terminal": "BAM-like terminal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--figures", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--qa-dir", type=Path)
    parser.add_argument("--formats", nargs="+", choices=EXPORT_FORMATS, default=EXPORT_FORMATS)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.65,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray([
            value.decode() if isinstance(value, (bytes, np.bytes_)) else value
            for value in values
        ], dtype=object)
    return values


def read_h5_obs(
    path: Path, columns: Iterable[str], obsm_key: str | None = None
) -> tuple[pd.DataFrame, np.ndarray | None]:
    data = {}
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        for column in columns:
            obj = obs[column]
            if isinstance(obj, h5py.Dataset):
                data[column] = decode(obj[:])
            elif obj.attrs.get("encoding-type") == "categorical":
                categories = decode(obj["categories"][:])
                codes = obj["codes"][:]
                data[column] = np.asarray([
                    categories[code] if code >= 0 else np.nan for code in codes
                ], dtype=object)
            elif "values" in obj:
                values = decode(obj["values"][:])
                if "mask" in obj:
                    values = values.astype(object)
                    values[obj["mask"][:]] = np.nan
                data[column] = values
            else:
                raise ValueError(f"Unsupported h5ad obs encoding: {column}")
        coords = handle["obsm"][obsm_key][:] if obsm_key else None
    return pd.DataFrame(data), coords


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.set_label(label)
    ax.text(
        x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold",
        ha="left", va="bottom", color=COLORS["ink"], clip_on=False,
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(colors=COLORS["ink"])
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    ax.title.set_color(COLORS["ink"])


def draw_box(
    ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
    text: str, facecolor: str, edgecolor: str = COLORS["ink"],
    fontsize: float = 6.2, weight: str = "normal",
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.65, facecolor=facecolor, edgecolor=edgecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text, transform=ax.transAxes,
        ha="center", va="center", fontsize=fontsize, fontweight=weight,
        color=COLORS["ink"], linespacing=1.2,
    )
    return box


def arrow_axes(
    ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
    color: str = COLORS["grey"], lw: float = 0.8,
) -> None:
    ax.annotate(
        "", xy=end, xytext=start, xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=7),
    )


def export_source(frame: pd.DataFrame, source_dir: Path, name: str) -> None:
    frame.to_csv(source_dir / f"{name}.csv", index=False)


def wrap_artist_text(artist, renderer, max_width: float) -> None:
    lines = []
    for paragraph in artist.get_text().split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            width = renderer.get_text_width_height_descent(
                candidate, artist.get_fontproperties(), ismath=False,
            )[0]
            if line and width > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    artist.set_text("\n".join(lines))


def prepare_layout(fig: plt.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for title in (ax.title, ax._left_title, ax._right_title):
            wrap_artist_text(title, renderer, ax.bbox.width)
        wrap_artist_text(ax.xaxis.label, renderer, ax.bbox.width * 1.10)
    fig.canvas.draw()
    title_groups = {}
    for ax in fig.axes:
        titles = [t for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
        if titles:
            row = round(ax.bbox.y1 / fig.bbox.height, 3)
            title_groups[row] = max(title_groups.get(row, 0), *(t.get_window_extent(renderer).height for t in titles))
    for ax in fig.axes:
        titles = [t for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
        if not titles:
            continue
        row = round(ax.bbox.y1 / fig.bbox.height, 3)
        title_top = ax.bbox.y1 + 6 * fig.dpi / 72 + title_groups[row]
        label_top = ax.transAxes.inverted().transform((0, title_top))[1]
        for loc, title in zip(("center", "left", "right"), (ax.title, ax._left_title, ax._right_title)):
            if title.get_text():
                fontproperties = title.get_fontproperties().copy()
                ax.set_title(title.get_text(), loc=loc, y=label_top, pad=0,
                             va="top", fontproperties=fontproperties)
        for artist in ax.texts:
            if artist.get_text() == ax.get_label():
                artist.set_position((artist.get_position()[0], label_top))
                artist.set_verticalalignment("top")
    fig.canvas.draw()


def layout_metrics(fig: plt.Figure) -> dict:
    renderer = fig.canvas.get_renderer()
    rows = []
    for ax in fig.axes:
        xticks = [t for loc, t in zip(ax.get_xticks(), ax.get_xticklabels())
                  if min(ax.get_xlim()) <= loc <= max(ax.get_xlim())]
        yticks = [t for loc, t in zip(ax.get_yticks(), ax.get_yticklabels())
                  if min(ax.get_ylim()) <= loc <= max(ax.get_ylim())]
        artists = [
            *ax.texts, *xticks, *yticks,
            ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title,
        ]
        legend = ax.get_legend()
        if legend is not None:
            artists.extend(legend.get_texts())
        for artist in artists:
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            bbox = artist.get_window_extent(renderer)
            rows.append({
                "panel": ax.get_label(), "text": artist.get_text(),
                "bbox_fraction": [bbox.x0 / fig.bbox.width, bbox.y0 / fig.bbox.height,
                                  bbox.x1 / fig.bbox.width, bbox.y1 / fig.bbox.height],
            })
    outside = [r for r in rows if any(v < -0.002 or v > 1.002 for v in r["bbox_fraction"])]
    return {"text_artists": rows, "outside_canvas": outside}


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_layout(fig)
    if LAYOUT_QA_DIR is not None:
        LAYOUT_QA_DIR.mkdir(parents=True, exist_ok=True)
        (LAYOUT_QA_DIR / f"{name}.json").write_text(json.dumps(layout_metrics(fig), indent=2))
    for extension in EXPORT_FORMATS:
        options = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        if extension == "png":
            options["dpi"] = 300
        if extension == "pdf":
            options["dpi"] = 600
        fig.savefig(output_dir / f"{name}.{extension}", **options)
    plt.close(fig)


def jitter(n: int, width: float = 0.08, seed: int = 20260821) -> np.ndarray:
    return np.random.default_rng(seed + n).uniform(-width, width, n)


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.quantile(values, [0.25, 0.5, 0.75]))


def figure_1(root: Path, output: Path, source: Path) -> None:
    topology = pd.read_csv(root / "results/trajectory/continuum_topology_qc.csv")
    inventory = pd.read_csv(root / "results/cell_inventory/APC_transition_cell_inventory.csv")
    summary = pd.read_csv(root / "results/trajectory/continuum_state_summary.csv")
    jackknife = pd.read_csv(root / "results/trajectory/trajectory_donor_jackknife.csv")

    fig = plt.figure(figsize=(183 * MM, 140 * MM))
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1.08, 1.0], hspace=0.38, wspace=1.0,
        left=0.065, right=0.955, bottom=0.13, top=0.91,
    )
    ax_a = fig.add_subplot(gs[0, 0:5])
    ax_b = fig.add_subplot(gs[0, 5:8])
    ax_c = fig.add_subplot(gs[0, 8:12])
    ax_d = fig.add_subplot(gs[1, 0:7])
    ax_e = fig.add_subplot(gs[1, 8:12])

    ax_a.set_axis_off()
    add_panel_label(ax_a, "a", x=-0.10, y=1.01)
    ax_a.set_title("Study design", loc="left", fontweight="bold")
    cohort_info = topology.set_index("dataset")
    boxes = [
        ("GSE138266", "PBMC + CSF"),
        ("GSE194078", "PBMC + CSF"),
        ("GSE286068", "CSF"),
    ]
    for index, (dataset, role) in enumerate(boxes):
        row = cohort_info.loc[dataset]
        x = 0.01 + index * 0.33
        text = f"{dataset}\n{role}"
        draw_box(ax_a, (x, 0.63), 0.29, 0.24, text, "#EEF2F5", fontsize=5.7)
        arrow_axes(ax_a, (x + 0.145, 0.63), (0.50, 0.48))
    draw_box(
        ax_a, (0.27, 0.30), 0.46, 0.18,
        "donor-aware\nconsensus trajectory",
        COLORS["entry_light"], fontsize=6.2, weight="bold",
    )
    arrow_axes(ax_a, (0.50, 0.30), (0.50, 0.20), COLORS["entry"])
    draw_box(
        ax_a, (0.02, 0.03), 0.91, 0.16,
        "PBMC monocyte  ->  CSF monocyte\n->  C1Q/APOE-high BAM-like pole",
        COLORS["terminal_light"], fontsize=5.5,
    )

    paired = inventory[inventory.dataset.isin(["GSE138266", "GSE194078"])].copy()
    totals = paired.groupby(["dataset", "tissue"], as_index=False).n_cells.sum()
    bam = paired[paired.fine_apc_label.eq("BAM_like_candidate")][
        ["dataset", "tissue", "n_cells"]
    ].rename(columns={"n_cells": "n_BAM_like"})
    restriction = totals.merge(bam, on=["dataset", "tissue"], how="left").fillna(0)
    restriction["BAM_like_fraction"] = restriction.n_BAM_like / restriction.n_cells
    export_source(restriction, source, "Fig1b_BAM_tissue_restriction")
    for idx, dataset in enumerate(["GSE138266", "GSE194078"]):
        sub = restriction[restriction.dataset.eq(dataset)].set_index("tissue")
        y = 100 * sub.loc[["PBMC", "CSF"], "BAM_like_fraction"].to_numpy()
        color = [COLORS["entry"], COLORS["terminal"]][idx]
        ax_b.plot([0, 1], y, color=color, lw=1.25, marker="o", ms=4.2, zorder=3)
        ax_b.text(1.04, y[1], dataset, fontsize=5.7, color=color, va="center")
    ax_b.set_xticks([0, 1], ["PBMC", "CSF"])
    ax_b.set_ylabel("BAM-like cells among APCs (%)")
    ax_b.set_title("Paired-cohort CSF restriction", loc="left", fontweight="bold")
    ax_b.set_xlim(-0.15, 1.48)
    ax_b.set_ylim(bottom=-1)
    add_panel_label(ax_b, "b")
    clean_axis(ax_b)

    h5_path = root / "results/trajectory/continuum_objects/GSE138266_monocyte_BAM_continuum.h5ad"
    umap_obs, umap = read_h5_obs(
        h5_path,
        ["transition_state", "consensus_pseudotime", "bam_reference", "bam_stable_core"],
        "X_umap",
    )
    umap_frame = umap_obs.copy()
    umap_frame[["UMAP1", "UMAP2"]] = umap
    bam_reference = (
        umap_frame["bam_reference"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    monocyte_root = umap_frame.transition_state.astype(str).eq("PBMC_classical_monocyte")
    csf_entry = umap_frame.transition_state.astype(str).eq("CSF_classical_monocyte")
    branch = monocyte_root | csf_entry | bam_reference
    umap_frame = umap_frame.loc[branch].copy()
    umap_frame["display_anchor"] = "continuum"
    umap_frame.loc[monocyte_root[branch], "display_anchor"] = "PBMC monocyte root"
    umap_frame.loc[csf_entry[branch], "display_anchor"] = "CSF monocyte entry"
    umap_frame.loc[bam_reference[branch], "display_anchor"] = "C1Q/APOE-high BAM-like pole"
    export_source(umap_frame, source, "Fig1c_representative_continuum_UMAP")
    trajectory_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trajectory", [COLORS["entry"], COLORS["transition"], COLORS["terminal"]]
    )
    draw_order = np.argsort(umap_frame.consensus_pseudotime.to_numpy(float))
    scatter = ax_c.scatter(
        umap_frame.UMAP1.to_numpy()[draw_order],
        umap_frame.UMAP2.to_numpy()[draw_order],
        c=umap_frame.consensus_pseudotime.to_numpy(float)[draw_order],
        cmap=trajectory_cmap, vmin=0, vmax=1, s=1.7, alpha=0.58,
        linewidths=0, rasterized=True,
    )
    anchor_specs = {
        "PBMC monocyte root": (-8, 20, "right"),
        "CSF monocyte entry": (0, 23, "center"),
        "C1Q/APOE-high BAM-like pole": (-3, -25, "center"),
    }
    for label, (dx, dy, align) in anchor_specs.items():
        selected = umap_frame.display_anchor.eq(label)
        center = np.nanmedian(
            umap_frame.loc[selected, ["UMAP1", "UMAP2"]].to_numpy(), axis=0
        )
        ax_c.annotate(
            label, xy=center, xytext=(dx, dy), textcoords="offset points",
            fontsize=5.2, fontweight="bold", ha=align, va="center",
            color=COLORS["ink"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.8),
            arrowprops=dict(arrowstyle="-", lw=0.55, color=COLORS["grey"]),
        )
    cbar = fig.colorbar(scatter, ax=ax_c, fraction=0.035, pad=0.015)
    cbar.ax.set_title("Pseudotime", fontsize=5.3, pad=3)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=5.2, length=2)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.set_title("Consensus root-to-BAM manifold", loc="left", fontweight="bold")
    add_panel_label(ax_c, "c", x=-0.07)

    stages = ["CSF_classical_monocyte", "CSF_BAM_intermediate", "CSF_BAM_terminal"]
    stage_labels = ["CSF entry", "BAM reference\nnon-core", "High-confidence\nBAM core"]
    modules = {
        "median_module_monocyte_entry": ("Monocyte entry", COLORS["entry"]),
        "median_module_CSF1_GMCSF_response": ("GM-CSF response", COLORS["transition"]),
        "median_module_BAM_identity": ("BAM identity", COLORS["terminal"]),
    }
    scaled_rows = []
    for dataset in topology.dataset:
        sub = summary[
            summary.dataset.eq(dataset) & summary.transition_state.isin(stages)
        ].set_index("transition_state").reindex(stages)
        for module, (label, _) in modules.items():
            values = sub[module].to_numpy(float)
            span = np.nanmax(values) - np.nanmin(values)
            scaled = (values - np.nanmin(values)) / span if span > 0 else values * 0
            for stage, value, raw in zip(stages, scaled, values):
                scaled_rows.append({
                    "dataset": dataset, "stage": stage, "module": label,
                    "stage_display": dict(zip(stages, stage_labels))[stage].replace("\n", " "),
                    "annotation_role": "confidence definition, not a discrete cell type",
                    "scaled_median": value, "raw_median": raw,
                })
    scaled_frame = pd.DataFrame(scaled_rows)
    export_source(scaled_frame, source, "Fig1d_module_progression")
    x = np.arange(3)
    label_offsets = {"Monocyte entry": 0.0, "GM-CSF response": 0.03, "BAM identity": -0.03}
    for module, (label, color) in modules.items():
        sub = scaled_frame[scaled_frame.module.eq(label)]
        matrix = sub.pivot(index="dataset", columns="stage", values="scaled_median").reindex(columns=stages)
        for values in matrix.to_numpy():
            ax_d.plot(x, values, color=color, lw=0.55, alpha=0.22)
        mean = matrix.mean(axis=0).to_numpy()
        ax_d.plot(x, mean, color=color, lw=1.7, marker="o", ms=3.5, label=label)
        ax_d.text(2.08, mean[-1] + label_offsets[label], label, color=color, fontsize=5.7, va="center")
    ax_d.set_xticks(x, stage_labels)
    ax_d.set_xlim(-0.12, 2.75)
    ax_d.set_ylim(-0.08, 1.08)
    ax_d.set_ylabel("Within-cohort scaled median")
    ax_d.set_title("Programs across BAM annotation confidence", loc="left", fontweight="bold")
    add_panel_label(ax_d, "d", x=-0.08)
    clean_axis(ax_d)

    jk = jackknife[jackknife.representation.eq("primary")]
    robustness_rows = []
    for dataset in topology.dataset:
        values = jk.loc[jk.dataset.eq(dataset), "rebuilt_terminal_vs_root_auc"].to_numpy()
        full = float(topology.loc[topology.dataset.eq(dataset), "primary_terminal_vs_root_auc"].iloc[0])
        robustness_rows.append({
            "dataset": dataset, "full_auc": full, "jackknife_min": values.min(),
            "jackknife_median": np.median(values), "jackknife_max": values.max(),
            "n_leave_one_donor_runs": len(values),
        })
    robust = pd.DataFrame(robustness_rows)
    export_source(robust, source, "Fig1e_trajectory_robustness")
    y = np.arange(len(robust))[::-1]
    for yi, row in zip(y, robust.itertuples()):
        ax_e.plot([row.jackknife_min, row.jackknife_max], [yi, yi], color=COLORS["light_grey"], lw=3)
        ax_e.scatter(row.jackknife_median, yi, s=18, color=COLORS["entry"], zorder=3, label="donor jackknife" if yi == y[0] else None)
        ax_e.scatter(row.full_auc, yi, s=24, marker="D", color=COLORS["ink"], zorder=4, label="full model" if yi == y[0] else None)
    ax_e.axvline(0.5, color=COLORS["grey"], ls="--", lw=0.7)
    ax_e.set_yticks(y, robust.dataset)
    ax_e.set_xlabel("High-confidence BAM endpoint versus root AUC")
    ax_e.set_xlim(0.45, 1.02)
    ax_e.set_title("Endpoint topology is donor-robust", loc="left", fontweight="bold")
    ax_e.legend(loc="lower right", handletextpad=0.4)
    add_panel_label(ax_e, "e", x=-0.16)
    clean_axis(ax_e)

    save_figure(fig, output, "Figure_1_CSF_monocyte_BAM_continuum")


def figure_2(root: Path, output: Path, source: Path) -> None:
    discovery = pd.read_csv(root / "results/two_axis/two_axis_donor_scores.csv")
    external = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/GSE328265_two_axis_donor_scores.csv"
    )
    meta_inputs = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/three_cohort_two_axis_meta_inputs.csv"
    )
    meta = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/three_cohort_two_axis_meta.csv"
    )
    lam_pairs = pd.read_csv(root / "results/cross_species_lam/LAM_decoupled_pair_tests.csv")
    lam_meta = pd.read_csv(root / "results/cross_species_lam/LAM_decoupled_meta.csv")

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    ax_a = fig.add_axes([0.055, 0.625, 0.225, 0.27])
    ax_b = fig.add_axes([0.375, 0.625, 0.265, 0.27])
    ax_c = fig.add_axes([0.790, 0.625, 0.185, 0.27])
    ax_d = fig.add_axes([0.075, 0.135, 0.400, 0.285])
    ax_e = fig.add_axes([0.770, 0.135, 0.205, 0.285])

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.annotate("", xy=(0.94, 0.10), xytext=(0.10, 0.10), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=COLORS["ink"]))
    ax_a.annotate("", xy=(0.10, 0.92), xytext=(0.10, 0.10), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=COLORS["ink"]))
    ax_a.text(0.52, 0.01, "Border adaptation / differentiation", ha="center", fontsize=6.1)
    ax_a.text(0.01, 0.52, "Inflammatory activation", ha="center", va="center", rotation=90, fontsize=6.1)
    regions = [
        (0.22, 0.22, 0.19, 0.14, COLORS["entry_light"], "CSF entry\nstate"),
        (0.52, 0.66, 0.25, 0.22, COLORS["ms_light"], "Decoupled\ninflammatory state"),
        (0.78, 0.24, 0.22, 0.16, COLORS["terminal_light"], "BAM-like\nadaptation pole"),
    ]
    for x, y, w, h, color, text in regions:
        ellipse = patches.Ellipse((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.95)
        ax_a.add_patch(ellipse)
        ax_a.text(x, y, text, ha="center", va="center", fontsize=5.8, fontweight="bold")
    ax_a.annotate("", xy=(0.67, 0.26), xytext=(0.32, 0.24), arrowprops=dict(arrowstyle="-|>", color=COLORS["grey"], lw=1.0))
    ax_a.set_title("Frozen two-axis state model", loc="left", fontweight="bold")
    add_panel_label(ax_a, "a", x=-0.12)

    donor_rows = []
    for dataset in ["GSE138266", "GSE286068"]:
        sub = discovery[discovery.dataset.eq(dataset)].copy()
        sub["group"] = np.where(sub.is_MS.astype(bool), "MS", "Control")
        donor_rows.append(sub[["dataset", "subject_id", "group", "continuous_decoupling_index"]])
    ext = external[external.n_continuum_cells.ge(20)].copy()
    ext["dataset"] = "GSE328265\nnew-only frozen projection"
    ext["validation_role"] = "newly sequenced subjects only; model frozen without refitting"
    ext["group"] = np.where(ext.disease_group.eq("control"), "Control", "MS")
    donor_rows.append(ext[["dataset", "subject_id", "group", "continuous_decoupling_index"]])
    donors = pd.concat(donor_rows, ignore_index=True)
    donors["validation_role"] = np.where(
        donors.dataset.str.contains("GSE328265", regex=False),
        "newly sequenced subjects only; frozen projection without refitting",
        "discovery cohort",
    )
    export_source(donors, source, "Fig2b_donor_continuous_decoupling")
    cohorts = ["GSE138266", "GSE286068", "GSE328265\nnew-only frozen projection"]
    for ci, cohort in enumerate(cohorts):
        for gi, group in enumerate(["Control", "MS"]):
            values = donors.loc[
                donors.dataset.eq(cohort) & donors.group.eq(group),
                "continuous_decoupling_index",
            ].to_numpy(float)
            xpos = ci * 2.5 + gi * 0.72
            color = COLORS["control"] if group == "Control" else COLORS["ms"]
            ax_b.scatter(
                xpos + jitter(len(values), 0.11, 20260821 + ci * 10 + gi), values,
                s=14, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.86,
            )
            q1, med, q3 = median_iqr(values)
            ax_b.plot([xpos - 0.18, xpos + 0.18], [med, med], color=COLORS["ink"], lw=1.2)
            ax_b.plot([xpos, xpos], [q1, q3], color=COLORS["ink"], lw=0.75)
    centers = np.arange(3) * 2.5 + 0.36
    ax_b.set_xticks(centers, ["GSE138266", "GSE286068", "GSE328265"])
    ax_b.axhline(0, color=COLORS["light_grey"], lw=0.7)
    ax_b.set_ylabel("Continuous decoupling index")
    ax_b.set_title("Donor-level disease shift", loc="left", fontweight="bold")
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["control"], markeredgecolor="none", label="Control"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["ms"], markeredgecolor="none", label="MS"),
        ], loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8,
    )
    add_panel_label(ax_b, "b")
    clean_axis(ax_b)

    endpoint_labels = {
        "continuous_decoupling_index": "Continuous index",
        "decoupled_inflammatory_fraction": "Decoupled fraction",
    }
    marker_map = {"continuous_decoupling_index": "o", "decoupled_inflammatory_fraction": "s"}
    color_map = {"continuous_decoupling_index": COLORS["ms"], "decoupled_inflammatory_fraction": COLORS["transition"]}
    y_positions = {"GSE138266": 2, "GSE286068": 1, "GSE328265_new_only": 0}
    for endpoint in endpoint_labels:
        sub = meta_inputs[meta_inputs.endpoint.eq(endpoint)]
        offset = 0.12 if endpoint == "continuous_decoupling_index" else -0.12
        for row in sub.itertuples():
            ax_c.scatter(
                row.standardized_effect, y_positions[row.cohort] + offset,
                s=25, marker=marker_map[endpoint], color=color_map[endpoint],
                edgecolor="white", linewidth=0.4, zorder=3,
            )
        meta_row = meta[meta.endpoint.eq(endpoint)].iloc[0]
        ax_c.scatter(
            meta_row.mean_standardized_effect, -0.75 + offset, s=30,
            marker="D", color=color_map[endpoint], edgecolor=COLORS["ink"], linewidth=0.4,
        )
    forest_source = meta_inputs.merge(
        meta[["endpoint", "combined_z", "stouffer_p", "fdr_bh", "mean_standardized_effect"]],
        on="endpoint", how="left",
    )
    export_source(forest_source, source, "Fig2c_three_cohort_effects")
    ax_c.axvline(0, color=COLORS["grey"], lw=0.7, ls="--")
    ax_c.set_yticks(
        [2, 1, 0, -0.75],
        ["GSE138266", "GSE286068", "GSE328265", "Meta"],
    )
    ax_c.set_xlabel("Standardized MS-control effect")
    ax_c.set_xlim(-0.07, 1.58)
    ax_c.set_ylim(-1.2, 2.45)
    ax_c.set_title("Concordant three-cohort effects", loc="left", fontweight="bold")
    add_panel_label(ax_c, "c", x=-0.55)
    clean_axis(ax_c)

    plane_obs, _ = read_h5_obs(
        root / "results/external_validation/GSE328265_two_axis_projection/GSE328265_monocyte_BAM_two_axis.h5ad",
        ["projected_differentiation", "external_activation_residual",
         "external_decoupled_inflammatory", "external_terminal_score_state",
         "disease_group", "subject_id"],
    )
    plane_obs["external_decoupled_inflammatory"] = plane_obs["external_decoupled_inflammatory"].astype(bool)
    plane_obs["external_terminal_score_state"] = plane_obs["external_terminal_score_state"].astype(bool)
    plane_obs["validation_role"] = "GSE328265 new-only frozen projection"
    export_source(plane_obs, source, "Fig2d_external_two_axis_cells")
    terminal = plane_obs.external_terminal_score_state.to_numpy(bool)
    decoupled = plane_obs.external_decoupled_inflammatory.to_numpy(bool)
    other = ~(terminal | decoupled)
    for mask, color, label, zorder in [
        (other, COLORS["light_grey"], "Other continuum", 1),
        (terminal, COLORS["terminal"], "BAM-like endpoint state", 2),
        (decoupled, COLORS["ms"], "Decoupled inflammatory state", 3),
    ]:
        ax_d.scatter(
            plane_obs.loc[mask, "projected_differentiation"],
            plane_obs.loc[mask, "external_activation_residual"],
            s=3.1 if label == "Other continuum" else 5.2, color=color,
            alpha=0.28 if label == "Other continuum" else 0.62,
            linewidth=0, rasterized=True, label=label, zorder=zorder,
        )
    ax_d.axhline(0, color=COLORS["light_grey"], lw=0.6)
    ax_d.set_xlabel("Frozen projected differentiation")
    ax_d.set_ylabel("Frozen activation residual")
    ax_d.set_title("GSE328265 new-only: frozen projection", loc="left", fontweight="bold")
    ax_d.legend(loc="upper right", markerscale=1.7)
    add_panel_label(ax_d, "d", x=-0.07)
    clean_axis(ax_d)

    labels = {
        "decoupled_excess__mouse_GMCSF_LAM_precursor": "GM-CSF LAM precursor program",
        "decoupled_excess__author_human_LAM": "Human meningeal LAM program",
        "decoupled_excess__mouse_LAM2_complement_APC": "LAM2 complement/APC program",
        "decoupled_excess__mouse_LAM4_fibrotic": "LAM4 fibrotic program",
        "decoupled_excess__LAM4_minus_LAM2": "LAM4-LAM2 program contrast",
    }
    ordered = list(labels)
    lam_source = lam_pairs[lam_pairs.endpoint.isin(ordered)].copy()
    lam_source["endpoint_display"] = lam_source.endpoint.map(labels)
    lam_source["evidence_type"] = "gene-program projection, not cell-identity assignment"
    export_source(lam_source, source, "Fig2e_cross_species_LAM_cohort_effects")
    y = np.arange(len(ordered))[::-1]
    cohort_offsets = {"GSE138266": -0.08, "GSE286068": 0.0, "GSE328265": 0.08}
    cohort_markers = {"GSE138266": "o", "GSE286068": "s", "GSE328265": "^"}
    for yi, endpoint in zip(y, ordered):
        sub = lam_pairs[lam_pairs.endpoint.eq(endpoint)]
        for row in sub.itertuples():
            ax_e.scatter(
                row.mean_decoupled_excess, yi + cohort_offsets[row.dataset],
                s=19, marker=cohort_markers[row.dataset], color=COLORS["grey"],
                edgecolor="white", linewidth=0.35,
            )
        meta_row = lam_meta[lam_meta.endpoint.eq(endpoint)].iloc[0]
        ax_e.scatter(meta_row.mean_cohort_excess, yi, s=35, marker="D", color=COLORS["ms"] if endpoint == ordered[0] else COLORS["ink"], zorder=4)
    ax_e.axvline(0, color=COLORS["grey"], lw=0.7, ls="--")
    ax_e.set_yticks(y, [labels[item].replace(" program", "\nprogram") for item in ordered])
    ax_e.set_xlabel("Decoupled-state excess")
    ax_e.set_xlim(-0.28, 0.62)
    ax_e.set_title("Cross-species LAM-program excess", loc="left", fontweight="bold")
    add_panel_label(ax_e, "e", x=-0.21)
    clean_axis(ax_e)

    save_figure(fig, output, "Figure_2_MS_inflammatory_adaptation_decoupling")


def figure_3(root: Path, output: Path, source: Path) -> None:
    meta = pd.read_csv(root / "results/niche_regulator/receiver_state_delta_meta.csv")
    circuits = pd.read_csv(root / "results/niche_regulator/receiver_side_circuit_ranking.csv")
    sender = pd.read_csv(root / "results/niche_regulator/sender_to_LAM_outcome_associations.csv")

    fig = plt.figure(figsize=(183 * MM, 172 * MM))
    ax_a = fig.add_axes([0.080, 0.780, 0.860, 0.160])
    ax_b = fig.add_axes([0.135, 0.420, 0.295, 0.240])
    ax_c = fig.add_axes([0.585, 0.420, 0.355, 0.240])
    ax_d = fig.add_axes([0.225, 0.105, 0.235, 0.220])
    ax_e = fig.add_axes([0.735, 0.105, 0.225, 0.220])

    ax_a.set_axis_off()
    add_panel_label(ax_a, "a", x=-0.06, y=1.04)
    draw_box(ax_a, (0.02, 0.67), 0.19, 0.17, "IFNGR1/2\nreceiver state", COLORS["ms_light"], weight="bold")
    draw_box(ax_a, (0.02, 0.37), 0.19, 0.17, "TNFRSF1A/B\nreceiver state", COLORS["transition_light"], weight="bold")
    draw_box(ax_a, (0.02, 0.07), 0.19, 0.17, "CSF2\nT-cell sender score", COLORS["pale_grey"], weight="bold")
    draw_box(ax_a, (0.38, 0.59), 0.22, 0.24, "IRF1 / STAT1\ninflammatory lock", COLORS["ms_light"], weight="bold")
    draw_box(ax_a, (0.38, 0.29), 0.22, 0.18, "NFKB1\nrelay", COLORS["transition_light"], weight="bold")
    draw_box(ax_a, (0.72, 0.54), 0.25, 0.27, "GM-CSF-responsive\ndecoupled\nstate", COLORS["ms_light"], fontsize=5.5, weight="bold")
    draw_box(ax_a, (0.72, 0.11), 0.25, 0.24, "PPARG / MAF\nBAM-like\nadaptation", COLORS["terminal_light"], weight="bold")
    arrow_axes(ax_a, (0.21, 0.75), (0.38, 0.70), COLORS["ms"])
    arrow_axes(ax_a, (0.21, 0.45), (0.38, 0.38), COLORS["transition"])
    arrow_axes(ax_a, (0.60, 0.71), (0.72, 0.68), COLORS["ms"])
    arrow_axes(ax_a, (0.60, 0.38), (0.72, 0.61), COLORS["transition"])
    ax_a.plot([0.21, 0.28, 0.28, 0.68], [0.15, 0.15, 0.53, 0.53],
              transform=ax_a.transAxes, color=COLORS["grey"], lw=0.55)
    arrow_axes(ax_a, (0.68, 0.53), (0.72, 0.63), COLORS["grey"], lw=0.55)
    ax_a.set_title("Receiver-side circuit architecture", loc="left", fontweight="bold")

    selected = [
        ("receptor", "IFNG"), ("receptor", "TNF"), ("receptor", "CSF2"),
        ("activity", "IRF1"), ("activity", "STAT1"), ("activity", "NFKB1"),
        ("activity", "PPARG"), ("activity", "MAF"), ("activity", "NR1H3"),
    ]
    dec = meta[meta.contrast.eq("decoupled_minus_other_incomplete")].copy()
    selection = pd.concat([
        dec[dec.feature_type.eq(ft) & dec.feature_name.eq(name)] for ft, name in selected
    ], ignore_index=True)
    receptor_labels = {
        "IFNG": "IFNGR1/2", "TNF": "TNFRSF1A/B", "CSF2": "CSF2RA/B",
    }
    selection["display_label"] = [
        receptor_labels.get(name, name) if feature_type == "receptor" else name
        for feature_type, name in selected
    ]
    selection["molecular_layer"] = [
        "receiver receptor-complex expression" if feature_type == "receptor" else "regulator activity"
        for feature_type, _ in selected
    ]
    export_source(selection, source, "Fig3b_decoupled_receiver_regulator_meta")
    labels = selection.display_label.tolist()
    raw_labels = [item[1] for item in selected]
    values = selection.combined_z.to_numpy()
    y = np.arange(len(labels))[::-1]
    colors = [
        COLORS["ms"] if name in {"IFNG", "IRF1", "STAT1"}
        else COLORS["transition"] if name in {"TNF", "NFKB1"}
        else COLORS["terminal"] if name in {"PPARG", "MAF", "NR1H3"}
        else COLORS["grey"]
        for name in raw_labels
    ]
    ax_b.axvline(0, color=COLORS["grey"], ls="--", lw=0.7)
    for yi, value, color, fdr in zip(y, values, colors, selection.fdr_bh_within_contrast):
        ax_b.plot([0, value], [yi, yi], color=color, lw=1.25, alpha=0.75)
        ax_b.scatter(value, yi, s=28, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    ax_b.set_yticks(y, labels)
    ax_b.set_xlabel("Decoupled versus incomplete meta z")
    ax_b.set_title("Decoupled-state meta-effects", loc="left", fontweight="bold")
    add_panel_label(ax_b, "b", x=-0.35)
    clean_axis(ax_b)

    circuit_labels = {
        "IFNG": "IFNG -> IFNGR", "TNF": "TNF -> TNFR",
        "CSF2": "CSF2 -> CSF2R", "CCL5": "CCL5 -> CCR1/5",
        "LTA_LTB": "LTA/LTB -> LTBR",
    }
    circuit_plot = circuits.sort_values("receiver_chain_score").copy()
    circuit_plot["circuit_display"] = circuit_plot.circuit.map(circuit_labels)
    circuit_plot["evidence_scope"] = "receiver-side ranking; sender association is non-significant"
    export_source(circuit_plot, source, "Fig3d_receiver_circuit_ranking")
    y_c = np.arange(len(circuit_plot))
    ax_d.barh(y_c, circuit_plot.receiver_chain_score, color=[COLORS["ms"] if value == "A" else COLORS["light_grey"] for value in circuit_plot.receiver_evidence_tier], edgecolor="none")
    ax_d.set_yticks(y_c, circuit_plot.circuit_display)
    ax_d.set_xlabel("Receiver-chain score")
    ax_d.set_title("Receiver circuit ranking", loc="left", fontweight="bold")
    for yi, row in zip(y_c, circuit_plot.itertuples()):
        ax_d.text(row.receiver_chain_score + 0.12, yi, row.receiver_evidence_tier, va="center", fontsize=5.4, fontweight="bold")
    ax_d.set_xlim(right=float(circuit_plot.receiver_chain_score.max()) + 0.8)
    add_panel_label(ax_d, "d", x=-0.65)
    clean_axis(ax_d)

    contrasts = [
        "decoupled_minus_other_incomplete",
        "transitional_minus_early",
        "terminal_minus_transitional",
    ]
    regulators = ["IRF1", "STAT1", "NFKB1", "PPARG", "NR1H3", "MAF", "MAFB", "BHLHE41"]
    heat = meta[
        meta.feature_type.eq("activity")
        & meta.contrast.isin(contrasts)
        & meta.feature_name.isin(regulators)
    ].pivot(index="feature_name", columns="contrast", values="combined_z").reindex(index=regulators, columns=contrasts)
    # BHLHE41 is expression-only in the source table.
    if heat.loc["BHLHE41"].isna().all():
        replacement = meta[
            meta.feature_type.eq("expression")
            & meta.feature_name.eq("BHLHE41")
            & meta.contrast.isin(contrasts)
        ].set_index("contrast").combined_z
        heat.loc["BHLHE41"] = replacement.reindex(contrasts).to_numpy()
    contrast_labels = {
        "decoupled_minus_other_incomplete": "Decoupled state - other incomplete",
        "transitional_minus_early": "Mid-trajectory window - early window",
        "terminal_minus_transitional": "Late-trajectory window - mid window",
    }
    contrast_definitions = {
        "decoupled_minus_other_incomplete": "pseudotime <0.8, activation-high versus activation-not-high",
        "transitional_minus_early": "consensus pseudotime 0.4-0.8 versus <0.4",
        "terminal_minus_transitional": "consensus pseudotime >=0.8 versus 0.4-0.8",
    }
    heat_source = heat.reset_index().melt(id_vars="feature_name", var_name="contrast", value_name="combined_z")
    heat_source["contrast_display"] = heat_source.contrast.map(contrast_labels)
    heat_source["contrast_definition"] = heat_source.contrast.map(contrast_definitions)
    export_source(heat_source, source, "Fig3c_regulator_trajectory_dynamics")
    image = ax_c.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto", interpolation="nearest")
    ax_c.set_yticks(np.arange(len(regulators)), regulators)
    ax_c.set_xticks(
        np.arange(3),
        ["Decoupled state -\nother incomplete", "Mid - early\ntrajectory window", "Late - mid\ntrajectory window"],
    )
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            ax_c.text(j, i, f"{value:.1f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=5.2, color="white" if abs(value) > 3.2 else COLORS["ink"])
    cbar = fig.colorbar(image, ax=ax_c, fraction=0.025, pad=0.02)
    cbar.set_label("Meta z", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    ax_c.set_title("Regulator programs across prespecified trajectory windows", loc="left", fontweight="bold")
    add_panel_label(ax_c, "c", x=-0.22)

    pooled = sender[
        sender.scope.eq("pooled_dataset_disease_residualized")
        & sender.y.eq("decoupled_excess__mouse_GMCSF_LAM_precursor")
    ].copy()
    ligand_labels = {
        "all_ligand_ensemble": "All-ligand ensemble",
        "cytokine_ensemble": "Cytokine ensemble",
        "LTA_LTB": "LTA/LTB",
    }
    pooled["ligand"] = pooled.x.str.replace("sender_score__", "", regex=False)
    pooled["ligand_display"] = pooled.ligand.map(ligand_labels).fillna(pooled.ligand)
    pooled["sender_cell_type"] = "CSF T cells"
    pooled = pooled.sort_values("correlation")
    export_source(pooled, source, "Fig3e_sender_precursor_associations")
    y_s = np.arange(len(pooled))
    ax_e.axvline(0, color=COLORS["grey"], lw=0.7, ls="--")
    ax_e.scatter(pooled.correlation, y_s, s=24, color=COLORS["grey"], edgecolor="white", linewidth=0.4)
    ax_e.set_yticks(y_s, pooled.ligand_display)
    ax_e.set_xlim(-0.28, 0.28)
    ax_e.set_xlabel("Partial rank correlation")
    ax_e.set_title("CSF T-cell ligand scores\ndo not explain state excess", loc="left", fontweight="bold")
    add_panel_label(ax_e, "e", x=-0.66)
    clean_axis(ax_e)

    save_figure(fig, output, "Figure_3_receiver_regulatory_analysis")


def figure_4(root: Path, output: Path, source: Path) -> None:
    bridge_meta = pd.read_csv(root / "results/in_vivo_crispr_bridge/bridge_meta_analysis.csv")
    bridge_tests = pd.read_csv(root / "results/in_vivo_crispr_bridge/cohort_bridge_tests.csv")
    bridge_decisions = pd.read_csv(root / "results/in_vivo_crispr_bridge/primary_bridge_decisions.csv")
    primary = pd.read_csv(root / "results/virtual_perturbation_celloracle/primary_endpoint_meta_analysis.csv")
    harm = pd.read_csv(root / "results/virtual_perturbation_celloracle/harm_veto_meta_analysis.csv")

    fig = plt.figure(figsize=(183 * MM, 150 * MM))
    gs = fig.add_gridspec(
        2, 12, height_ratios=[0.92, 1.08], hspace=0.44, wspace=1.45,
        left=0.075, right=0.955, bottom=0.14, top=0.91,
    )
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 4:12])
    ax_c = fig.add_subplot(gs[1, 0:4])
    ax_d = fig.add_subplot(gs[1, 4:8])
    ax_e = fig.add_subplot(gs[1, 8:12])

    ax_a.set_axis_off()
    add_panel_label(ax_a, "a", x=-0.10, y=1.01)
    draw_box(ax_a, (0.05, 0.78), 0.90, 0.16, "In vivo receptor KO\nEAE macrophages", COLORS["probe_light"], fontsize=5.6, weight="bold")
    draw_box(ax_a, (0.05, 0.54), 0.90, 0.16, "Human CSF\nstate coordinates", COLORS["entry_light"], fontsize=5.6, weight="bold")
    draw_box(ax_a, (0.05, 0.30), 0.90, 0.16, "Response / KO\nsignature projection", COLORS["ms_light"], fontsize=5.6, weight="bold")
    draw_box(ax_a, (0.05, 0.06), 0.90, 0.16, "IRF1 / STAT1\nvirtual perturbation", COLORS["terminal_light"], fontsize=5.6, weight="bold")
    arrow_axes(ax_a, (0.50, 0.78), (0.50, 0.70))
    arrow_axes(ax_a, (0.50, 0.54), (0.50, 0.46))
    arrow_axes(ax_a, (0.50, 0.30), (0.50, 0.22))
    ax_a.set_title("Experimental-to-human perturbation bridge", loc="left", fontweight="bold")

    perturbations = ["Ifngr1", "Tnfrsf1a", "Csf2ra", "Tgfbr1"]
    perturbation_labels = {
        "Ifngr1": "IFNGR1", "Tnfrsf1a": "TNFR1", "Csf2ra": "CSF2RA", "Tgfbr1": "TGFBR1",
    }
    endpoints = [
        "decoupled_minus_other_incomplete", "axis2_spearman", "pseudotime_spearman",
        "lam2_residual_spearman", "precursor_residual_spearman",
    ]
    endpoint_labels = [
        "Decoupled\nresponse", "Axis 2", "Differentiation", "LAM2\nresidual", "Precursor\nresidual",
    ]
    published = bridge_meta.loc[bridge_meta.signature_type.eq("published_response")].copy()
    published["perturbation_display"] = published.perturbation.map(perturbation_labels)
    published["endpoint_display"] = published.endpoint.map(dict(zip(endpoints, endpoint_labels)))
    published["signature_scope"] = "author-defined in vivo cytokine-response signature"
    published["endpoint_gene_holdout"] = True
    export_source(published, source, "Fig4b_in_vivo_response_bridge")
    matrix = published.pivot(index="perturbation", columns="endpoint", values="combined_z").reindex(
        index=perturbations, columns=endpoints
    )
    fdr = published.pivot(
        index="perturbation", columns="endpoint", values="fdr_bh_within_signature_endpoint"
    ).reindex(index=perturbations, columns=endpoints)
    image = ax_b.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto", interpolation="nearest")
    ax_b.set_yticks(np.arange(len(perturbations)), [perturbation_labels[x] for x in perturbations])
    ax_b.set_xticks(np.arange(len(endpoints)), endpoint_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            star = "*" if fdr.iloc[i, j] < 0.05 else ""
            ax_b.text(
                j, i, f"{value:.1f}{star}", ha="center", va="center", fontsize=5.5,
                color="white" if abs(value) > 3.2 else COLORS["ink"],
            )
    cbar = fig.colorbar(image, ax=ax_b, fraction=0.018, pad=0.015)
    cbar.set_label("Signed Stouffer z", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    ax_b.set_title("IFNGR1 response maps selectively to the inflammatory lock", loc="left", fontweight="bold")
    add_panel_label(ax_b, "b", x=-0.08)

    cohort = bridge_tests.loc[
        bridge_tests.perturbation.eq("Ifngr1")
        & bridge_tests.endpoint.eq("decoupled_minus_other_incomplete")
    ].copy()
    cohort = cohort.loc[cohort.dataset.isin(["GSE138266", "GSE286068", "GSE328265"])]
    cohort["signature_display"] = cohort.signature_type.map({
        "published_response": "IFNGR1 response", "direct_ko_consensus": "IFNGR1 KO-like",
    })
    cohort["endpoint_display"] = "Decoupled minus other incomplete"
    cohort["statistical_unit"] = "donor"
    export_source(cohort, source, "Fig4c_IFNGR1_cohort_directionality")
    datasets = ["GSE138266", "GSE286068", "GSE328265"]
    y = np.arange(len(datasets))
    for signature_type, label, color, marker, offset in [
        ("published_response", "Response signature", COLORS["ms"], "o", 0.08),
        ("direct_ko_consensus", "Direct KO-like", COLORS["entry"], "s", -0.08),
    ]:
        frame = cohort.loc[cohort.signature_type.eq(signature_type)].set_index("dataset").reindex(datasets)
        ax_c.scatter(
            frame.standardized_effect, y + offset, s=27, color=color, marker=marker,
            edgecolor="white", linewidth=0.4, label=label, zorder=3,
        )
    ax_c.axvline(0, color=COLORS["grey"], lw=0.7, ls="--")
    ax_c.set_yticks(y, [x.replace("GSE", "") for x in datasets])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Donor-standardized state effect")
    ax_c.set_title("IFNGR1 response and KO-like\nsignatures reverse in every cohort", loc="left", fontweight="bold")
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.01, 0.42),
                handletextpad=0.3, borderaxespad=0.2)
    add_panel_label(ax_c, "c", x=-0.18)
    clean_axis(ax_c)

    direct = bridge_decisions.copy()
    direct["perturbation_display"] = direct.perturbation.map(perturbation_labels)
    direct["inflammatory_reversal_z"] = -direct.direct_ko_decoupled_z
    direct["differentiation_association_z"] = direct.direct_ko_pseudotime_z
    direct["interpretation_scope"] = "direct-KO sensitivity projection"
    export_source(direct, source, "Fig4d_direct_KO_triangulation")
    point_colors = {
        "Ifngr1": COLORS["ms"], "Tnfrsf1a": COLORS["transition"],
        "Csf2ra": "#2B7A78", "Tgfbr1": COLORS["entry"],
    }
    offsets = {
        "Ifngr1": (-0.14, 0.14, "right"), "Tnfrsf1a": (0.13, 0.34, "left"),
        "Csf2ra": (0.13, 0.12, "left"), "Tgfbr1": (0.13, -0.20, "left"),
    }
    for row in direct.itertuples():
        ax_d.scatter(
            row.inflammatory_reversal_z, row.differentiation_association_z,
            s=43, color=point_colors[row.perturbation], edgecolor="white", linewidth=0.5, zorder=3,
        )
        dx, dy, align = offsets[row.perturbation]
        ax_d.text(
            row.inflammatory_reversal_z + dx, row.differentiation_association_z + dy,
            row.perturbation_display, fontsize=5.4, ha=align,
        )
    ax_d.axhline(0, color=COLORS["grey"], lw=0.7, ls="--")
    ax_d.axvline(0, color=COLORS["grey"], lw=0.7, ls="--")
    ax_d.set_xlabel("KO-like inflammatory reversal (-z)")
    ax_d.set_ylabel("KO-like differentiation association (z)")
    ax_d.set_title("IFNGR1 uniquely enters the\nreversal-preservation quadrant", loc="left", fontweight="bold")
    add_panel_label(ax_d, "d", x=-0.19)
    clean_axis(ax_d)

    virtual_rows = []
    for candidate in ["IRF1", "STAT1"]:
        for endpoint, display in [("axis2", "Axis 2 suppression"), ("gmcsf_lam_precursor", "Precursor reduction")]:
            row = primary.loc[primary.candidate.eq(candidate) & primary.endpoint.eq(endpoint)].iloc[0]
            virtual_rows.append({
                "candidate": candidate, "virtual_perturbation": "KO", "endpoint": endpoint,
                "endpoint_display": display, "evidence_z": row.combined_expected_z,
                "fdr": row.fdr_bh_primary_endpoints, "evidence_role": "expected-direction benefit",
            })
        row = harm.loc[harm.candidate.eq(candidate) & harm.veto_endpoint.eq("lam2_complement_apc")].iloc[0]
        virtual_rows.append({
            "candidate": candidate, "virtual_perturbation": "KO", "endpoint": "lam2_complement_apc",
            "endpoint_display": "LAM2 harm veto", "evidence_z": row.combined_harmful_z,
            "fdr": row.fdr_bh_veto_endpoints, "evidence_role": "harmful-direction veto",
        })
    virtual = pd.DataFrame(virtual_rows)
    export_source(virtual, source, "Fig4e_virtual_downstream_control")
    categories = ["Axis 2\nsuppression", "Precursor\nreduction", "LAM2 harm\n(veto)"]
    x = np.arange(3)
    width = 0.32
    for candidate, offset, color in [("IRF1", -width / 2, COLORS["entry"]), ("STAT1", width / 2, COLORS["ms"])]:
        frame = virtual.loc[virtual.candidate.eq(candidate)].set_index("endpoint_display").reindex(
            ["Axis 2 suppression", "Precursor reduction", "LAM2 harm veto"]
        )
        bars = ax_e.bar(x + offset, frame.evidence_z, width=width, color=color, label=f"{candidate} virtual KO")
    ax_e.axhline(1.96, color=COLORS["grey"], lw=0.7, ls="--")
    ax_e.axhline(0, color=COLORS["ink"], lw=0.6)
    ax_e.set_xticks(x, categories)
    ax_e.set_ylabel("Directional meta z")
    ax_e.set_ylim(-5.0, 6.7)
    ax_e.set_title("Virtual downstream control\nexposes the STAT1 adaptation veto", loc="left", fontweight="bold")
    ax_e.legend(loc="lower left", handlelength=1.1, handletextpad=0.3)
    add_panel_label(ax_e, "e", x=-0.19)
    clean_axis(ax_e)

    save_figure(fig, output, "Figure_4_in_vivo_CRISPR_directional_control")


def figure_5(root: Path, output: Path, source: Path) -> None:
    signature = pd.read_csv(root / "results/transition_aware_drug_screen/signatures/signature_build_summary.csv").iloc[0]
    profiles = pd.read_csv(root / "results/transition_aware_drug_screen/nibr_full_profiles/NIBR_transition_candidate_profile_scores.csv")
    compounds = pd.read_csv(root / "results/transition_aware_drug_screen/nibr_full_profiles/NIBR_transition_candidate_compound_summary.csv")
    genetic = pd.read_csv(root / "results/transition_aware_drug_screen/genetic_deconvolution/genetic_target_cross_library_decisions.csv")
    neighbors = pd.read_csv(root / "results/transition_aware_drug_screen/mechanism_deconvolution/BX912_compound_level_directional_neighbors.csv")
    pdpk1_neighbors = pd.read_csv(root / "results/transition_aware_drug_screen/mechanism_deconvolution/BX912_PDPK1_directional_neighbors.csv")
    redocking = pd.read_csv(root / "results/docking/1Z5M_PDPK1_BX320_site/1Z5M_BX320_redocking_validation.csv")
    robustness = pd.read_csv(root / "results/docking/1Z5M_PDPK1_BX912/BX912_pose_robustness.csv")
    molecule = json.loads((root / "data/reference/chembl_bx912_molecule.json").read_text())
    activities = json.loads((root / "data/reference/chembl_bx912_activities.json").read_text())["activities"]

    fig = plt.figure(figsize=(183 * MM, 190 * MM))
    ax_a = fig.add_axes([0.070, 0.855, 0.895, 0.100])
    ax_b = fig.add_axes([0.110, 0.570, 0.335, 0.215])
    ax_c = fig.add_axes([0.635, 0.570, 0.335, 0.215])
    ax_d = fig.add_axes([0.110, 0.295, 0.310, 0.180])
    ax_e = fig.add_axes([0.635, 0.295, 0.335, 0.180])
    ax_f = fig.add_axes([0.160, 0.090, 0.385, 0.115])
    ax_g = fig.add_axes([0.650, 0.055, 0.320, 0.150])

    ax_a.set_axis_off()
    add_panel_label(ax_a, "a", x=-0.05, y=1.04)
    cascade = [
        (0.01, "32 donor pairs\n20,751 common genes", COLORS["entry_light"]),
        (0.27, "3,483 concordant\n300 frozen lock genes", COLORS["transition_light"]),
        (0.53, "76 NIBR U2OS profiles\n19 compounds", COLORS["probe_light"]),
        (0.79, "1 phenotypic profile\nBX-912 0.1 uM", COLORS["ms_light"]),
    ]
    for index, (x, text, color) in enumerate(cascade):
        draw_box(ax_a, (x, 0.22), 0.20, 0.45, text, color, fontsize=5.4, weight="bold")
        if index < len(cascade) - 1:
            arrow_axes(ax_a, (x + 0.20, 0.445), (x + 0.26, 0.445))
    ax_a.set_title("Transition-aware phenotype selection", loc="left", fontweight="bold")
    cascade_source = pd.DataFrame([{
        "n_eligible_donor_pairs": int(signature.n_eligible_donor_pairs),
        "n_common_genes": int(signature.n_common_genes),
        "n_direction_concordant": int(signature.n_direction_concordant),
        "n_frozen_lock_genes": int(signature.n_lock_high + signature.n_lock_low),
        "n_NIBR_profiles": len(profiles), "n_compounds": len(compounds),
        "n_advancing_profiles": int(profiles.profile_advance.astype(bool).sum()),
        "perturbation_cell_line": "U2OS", "profile_duration_h": 24,
        "screen_role": "phenotypic execution screen; not disease-cell efficacy",
    }])
    export_source(cascade_source, source, "Fig5a_screen_cascade")

    props = molecule["molecule_properties"]
    activity_rows = pd.DataFrame(activities)
    peff = activity_rows.loc[activity_rows.standard_type.eq("Peff"), "standard_value"].iloc[0]
    hepg2 = activity_rows.loc[activity_rows.target_pref_name.eq("HepG2"), "standard_value"].iloc[0]
    developability = pd.DataFrame([
        {"metric": "Molecular weight", "numeric_value": float(props["full_mwt"]), "unit": "Da", "display_value": f"{float(props['full_mwt']):.2f} Da", "assessment": "high"},
        {"metric": "Polar surface area", "numeric_value": float(props["psa"]), "unit": "A2", "display_value": f"{float(props['psa']):.2f} A2", "assessment": "high"},
        {"metric": "PAMPA Peff", "numeric_value": float(peff), "unit": "1e-7 cm/s", "display_value": f"{float(peff):.2f} x 10^-7 cm/s", "assessment": "poor"},
        {"metric": "HepG2 viability IC50", "numeric_value": float(hepg2) / 1000, "unit": "uM", "display_value": f"{float(hepg2)/1000:.1f} uM", "assessment": "narrow"},
        {"metric": "Phenotypic assignment", "numeric_value": np.nan, "unit": "", "display_value": "phenotypic probe", "assessment": "non-therapeutic assignment"},
        {"metric": "Target attribution", "numeric_value": np.nan, "unit": "", "display_value": "unresolved", "assessment": "not PDPK1-confirmed"},
    ])
    export_source(developability, source, "Fig5g_developability_gate")
    ax_g.set_axis_off()
    add_panel_label(ax_g, "g", x=-0.10, y=1.04)
    ax_g.set_title("Developability and attribution gate", loc="left", fontweight="bold")
    rows = [
        ("MW", f"{float(props['full_mwt']):.2f} Da"),
        ("PSA", f"{float(props['psa']):.2f} A2"),
        ("PAMPA Peff", f"{float(peff):.2f} x 10^-7 cm/s"),
        ("HepG2 IC50", f"{float(hepg2)/1000:.1f} uM"),
    ]
    for index, (key, value) in enumerate(rows):
        y = 0.76 - index * 0.19
        ax_g.text(0.02, y, key, transform=ax_g.transAxes, fontsize=5.8, color=COLORS["grey"])
        ax_g.text(0.42, y, value, transform=ax_g.transAxes, fontsize=5.9, fontweight="bold", color=COLORS["ink"])

    profiles = profiles.copy()
    profiles["perturbation_cell_line"] = "U2OS"
    profiles["evidence_role"] = "phenotypic profile; not CSF myeloid efficacy"
    profiles["adaptation_floor"] = profiles[["lam2_adaptation_score", "trajectory_adaptation_score"]].min(axis=1)
    export_source(profiles, source, "Fig5b_NIBR_profile_screen")
    nonadvance = ~profiles.profile_advance.astype(bool)
    ax_b.scatter(
        profiles.loc[nonadvance, "lock_weighted_reversal_score"],
        profiles.loc[nonadvance, "adaptation_floor"],
        s=9 + 2.5 * np.log10(profiles.loc[nonadvance, "dose_um"] + 1),
        color=COLORS["light_grey"], alpha=0.7, edgecolor="none",
    )
    advance = profiles.profile_advance.astype(bool)
    ax_b.scatter(
        profiles.loc[advance, "lock_weighted_reversal_score"],
        profiles.loc[advance, "adaptation_floor"],
        s=48, color=COLORS["probe"], edgecolor=COLORS["ink"], linewidth=0.5, zorder=4,
    )
    bx = profiles[advance].iloc[0]
    ax_b.annotate(
        "BX-912 0.1 uM", xy=(bx.lock_weighted_reversal_score, bx.adaptation_floor),
        xytext=(-16, 14), textcoords="offset points",
        fontsize=5.7, ha="center", color=COLORS["probe"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["probe"], lw=0.7),
    )
    ax_b.axvline(0, color=COLORS["grey"], lw=0.6, ls="--")
    ax_b.axhline(0, color=COLORS["grey"], lw=0.6, ls="--")
    ax_b.set_xlabel("Inflammatory-lock reversal score")
    ax_b.set_ylabel("Minimum adaptation score")
    ax_b.set_title("Only one U2OS profile survives expanded-family control", loc="left", fontweight="bold")
    add_panel_label(ax_b, "b", x=-0.09)
    clean_axis(ax_b)

    selected_drugs = ["BX-912", "GSK2334470", "AR-12", "BAG-956", "PDK1-AKT inhibitor"]
    dose = profiles[profiles.drug.isin(selected_drugs)].copy()
    export_source(dose, source, "Fig5c_mechanism_matched_dose_profiles")
    for drug in selected_drugs:
        sub = dose[dose.drug.eq(drug)].sort_values("dose_um")
        if sub.empty:
            continue
        color = COLORS["probe"] if drug == "BX-912" else COLORS["grey"]
        lw = 1.8 if drug == "BX-912" else 0.75
        alpha = 1.0 if drug == "BX-912" else 0.55
        ax_c.plot(np.log10(sub.dose_um), sub.lock_weighted_reversal_score, marker="o", ms=3.0, lw=lw, color=color, alpha=alpha)
        label_y = {
            "BX-912": 0.046,
            "AR-12": 0.013,
            "PDK1-AKT inhibitor": -0.026,
            "GSK2334470": -0.050,
            "BAG-956": -0.071,
        }[drug]
        ax_c.text(
            np.log10(sub.dose_um.iloc[-1]) + 0.65,
            label_y,
            drug.replace("PDK1-AKT inhibitor", "PDK1-AKT inh."),
            fontsize=5.1,
            color=color,
            va="center",
            ha="right",
        )
    ax_c.axhline(0, color=COLORS["light_grey"], lw=0.7)
    ax_c.set_xticks([-2, -1, 0, 1], ["0.01", "0.1", "1", "10"])
    ax_c.set_xlabel("Dose (uM)")
    ax_c.set_ylabel("Lock reversal")
    ax_c.set_xlim(-2.12, 1.90)
    ax_c.set_title("PDPK1-pathway comparators do not reproduce BX-912", loc="left", fontweight="bold")
    add_panel_label(ax_c, "c", x=-0.12)
    clean_axis(ax_c)

    target_order = ["STAT1", "IRF1", "GSK3B", "PDPK1", "AKT2", "JAK2", "TYK2", "AURKA", "CHEK1"]
    g = genetic[genetic.target.isin(target_order)].set_index("target").reindex(target_order).dropna(how="all").reset_index()
    export_source(g, source, "Fig5d_genetic_target_deconvolution")
    y = np.arange(len(g))[::-1]
    fractions = g.n_libraries_advance / g.n_libraries_with_paired_signatures
    colors = [COLORS["terminal"] if "advance" in decision and "no_" not in decision else COLORS["light_grey"] for decision in g.decision]
    ax_d.barh(y, fractions, color=colors, edgecolor="none")
    ax_d.set_yticks(y, g.target)
    ax_d.set_xlim(0, 1.02)
    ax_d.set_xlabel("Advancing perturbation libraries / tested")
    ax_d.set_title("Genetic target deconvolution", loc="left", fontweight="bold")
    add_panel_label(ax_d, "d", x=-0.10)
    clean_axis(ax_d)

    top = neighbors.sort_values("compound_rank").head(8).copy()
    top["label"] = top.nibr_drug_name.str.replace(r" [A-Z]{2}-\d{2}-[A-Z]{2}\d{2}$", "", regex=True).str.slice(0, 18)
    pdpk = pdpk1_neighbors.sort_values("rank").head(5).copy()
    export_source(top, source, "Fig5e_top_NIBR_neighbors")
    export_source(pdpk, source, "Fig5e_PDPK1_NIBR_neighbors")
    y_top = np.arange(len(top))[::-1]
    ax_e.barh(y_top, top.directional_cosine, color=COLORS["probe_light"], edgecolor="none")
    ax_e.set_yticks([])
    for yi, label in zip(y_top, top.label):
        ax_e.text(0.008, yi, label, fontsize=4.8, va="center", ha="left", color=COLORS["ink"])
    ax_e.set_xlim(0, 0.76)
    ax_e.set_xlabel("Directional signature cosine")
    ax_e.set_title("Unbiased neighbors lack a coherent target class", loc="left", fontweight="bold")
    add_panel_label(ax_e, "e", x=-0.12)
    clean_axis(ax_e)

    export_source(redocking, source, "Fig5f_BX320_redocking_control")
    export_source(robustness, source, "Fig5f_BX912_pose_robustness")
    control_rmsd = float(redocking.loc[redocking.pose_rank.eq(1), "symmetry_aware_heavy_atom_rmsd_A"].iloc[0])
    ax_f.scatter(control_rmsd, 1, s=25, color=COLORS["terminal"], zorder=3)
    ax_f.plot([2, 2], [0.72, 1.28], color=COLORS["grey"], lw=0.7, ls="--")
    blind = robustness[robustness.analysis.eq("blind_docking_top_pose")]
    distances = blind.minimum_ligand_nitrogen_to_ALA162_backbone_A.to_numpy()
    ax_f.scatter(distances, np.linspace(-0.12, 0.12, len(distances)), s=14,
                 color=COLORS["ms"], alpha=0.80, zorder=3)
    seeded = robustness[robustness.analysis.eq("congeneric_seeded_local_optimization")].iloc[0]
    ax_f.scatter(seeded.minimum_ligand_nitrogen_to_ALA162_backbone_A, -0.24,
                 s=25, marker="s", color=COLORS["probe"], zorder=3)
    ax_f.plot([3.5, 3.5], [-0.38, 0.28], color=COLORS["grey"], lw=0.7, ls="--")
    ax_f.set_yticks([1, 0], ["BX-320\nRMSD", "BX-912\nhinge distance"])
    ax_f.set_xlim(0, 6.2)
    ax_f.set_ylim(-0.48, 1.48)
    ax_f.set_xlabel("Distance (A)")
    ax_f.set_title("Structure QC", loc="left", fontweight="bold")
    add_panel_label(ax_f, "f", x=-0.32)
    clean_axis(ax_f)

    save_figure(fig, output, "Figure_5_transition_aware_chemical_probe")


def write_source_readme(source: Path) -> None:
    readme = """# Main figure source data

Each CSV contains the exact rows or derived values used by the named manuscript panel.
Cell-level UMAP and state-plane tables preserve the frozen coordinates and scores without
geometric transformation. Values are generated by `scripts/75_generate_manuscript_figures.py`.

Statistical units:

- Figures 1-3: cells are displayed for topology, but disease tests and meta-effects use donors.
- Figure 4: donor-level in vivo CRISPR signature effects are combined across three
  inflammatory-state cohorts; trajectory/LAM endpoints additionally include the
  trajectory-only cohort. Virtual downstream effects use donor-specific GRNs.
- Figure 5: each point in the screen panel is one compound-dose perturbation profile.

Annotation scope:

- `BAM-like` denotes a transcriptional pole or confidence-supported annotation, not
  demonstrated lineage origin.
- Early, mid and late labels in Figure 3 are prespecified consensus-pseudotime windows,
  not discrete cell types.
- LAM labels in Figure 2 are cross-species gene-program projections, not cell-identity
  assignments.
- Receptor-KO labels in Figure 4 denote projected experimental mouse signatures;
  IRF1/STAT1 KO labels explicitly denote virtual perturbations in inferred GRNs.
- Figure 5 uses 24-hour U2OS perturbation profiles; BX-912 is assigned as a phenotypic
  probe with unresolved target attribution, not as a CNS-ready therapy.
"""
    (source / "README.md").write_text(readme)


def main() -> None:
    global EXPORT_FORMATS, LAYOUT_QA_DIR
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / "figures_main/exports"
    source = args.source_dir or root / "figures_main/source_data"
    EXPORT_FORMATS = args.formats
    LAYOUT_QA_DIR = args.qa_dir
    source.mkdir(parents=True, exist_ok=True)
    configure_style()
    functions = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4, 5: figure_5}
    for number in args.figures:
        if number not in functions:
            raise ValueError(f"Unknown figure number: {number}")
        functions[number](root, output, source)
        print(f"Generated Figure {number}", flush=True)
    write_source_readme(source)


if __name__ == "__main__":
    main()
