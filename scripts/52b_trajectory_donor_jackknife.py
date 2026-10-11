#!/usr/bin/env python3
"""Rebuild trajectory graphs after removing each donor in turn."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats


DATASETS = ["GSE138266", "GSE194078", "GSE286068"]
REPRESENTATIONS = {
    "primary": "X_pca",
    "marker_excluded": "X_pca_marker_excluded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return ((stats.rankdata(values, method="average") - 0.5) / len(values)).astype(np.float32)


def medoid_position(adata: sc.AnnData, positions: np.ndarray, rep: str) -> int:
    coordinates = adata.obsm[rep][positions, : min(20, adata.obsm[rep].shape[1])]
    centroid = np.median(coordinates, axis=0)
    return int(positions[np.argmin(np.square(coordinates - centroid).sum(axis=1))])


def donor_root_positions(adata: sc.AnnData, rep: str) -> list[int]:
    root = adata.obs["trajectory_role"].astype(str).eq("root_candidate")
    positions = []
    for _, index in adata.obs.loc[root].groupby("subject_id", observed=True).groups.items():
        candidate_positions = adata.obs_names.get_indexer(index)
        positions.append(medoid_position(adata, candidate_positions, rep))
    if len(positions) < 3:
        raise RuntimeError("Fewer than three donor-specific roots remain")
    return positions


def rebuild_multiroot(adata: sc.AnnData, rep: str, n_neighbors: int, seed: int) -> np.ndarray:
    adata.uns.pop("iroot", None)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(n_neighbors, adata.n_obs - 1),
        n_pcs=min(30, adata.obsm[rep].shape[1]),
        use_rep=rep,
        random_state=seed,
    )
    sc.tl.diffmap(adata, n_comps=15, random_state=seed)
    root_positions = donor_root_positions(adata, rep)
    scores = []
    for root_position in root_positions:
        adata.uns["iroot"] = root_position
        sc.tl.dpt(adata, n_dcs=10)
        scores.append(percentile_rank(adata.obs["dpt_pseudotime"].to_numpy(float)))
    return np.median(np.column_stack(scores), axis=1)


def auc_terminal_root(adata: sc.AnnData, values: np.ndarray) -> float:
    root = adata.obs["trajectory_role"].astype(str).eq("root_candidate").to_numpy()
    terminal = adata.obs["trajectory_role"].astype(str).eq("terminal_candidate").to_numpy()
    return float(
        stats.mannwhitneyu(values[terminal], values[root], alternative="two-sided").statistic
        / (terminal.sum() * root.sum())
    )


def run_dataset(root: Path, dataset: str, args: argparse.Namespace) -> pd.DataFrame:
    path = root / "results" / "trajectory" / "continuum_objects" / f"{dataset}_monocyte_BAM_continuum.h5ad"
    full = sc.read_h5ad(path)
    rows = []
    for held_out in sorted(full.obs["subject_id"].astype(str).unique()):
        keep = ~full.obs["subject_id"].astype(str).eq(held_out).to_numpy()
        for name, rep in REPRESENTATIONS.items():
            adata = full[keep].copy()
            rebuilt = rebuild_multiroot(adata, rep, args.n_neighbors, args.seed)
            full_key = "dpt_multiroot_primary" if name == "primary" else "dpt_multiroot_marker_excluded"
            reference = percentile_rank(full.obs.loc[adata.obs_names, full_key].to_numpy(float))
            rows.append({
                "dataset": dataset,
                "held_out_subject": held_out,
                "representation": name,
                "n_cells": adata.n_obs,
                "n_subjects": adata.obs["subject_id"].nunique(),
                "n_root_donors": adata.obs.loc[
                    adata.obs["trajectory_role"].astype(str).eq("root_candidate"), "subject_id"
                ].nunique(),
                "rebuilt_vs_full_spearman": stats.spearmanr(rebuilt, reference).statistic,
                "rebuilt_state_order_spearman": stats.spearmanr(rebuilt, adata.obs["state_order"]).statistic,
                "rebuilt_terminal_vs_root_auc": auc_terminal_root(adata, rebuilt),
            })
            del adata
    return pd.DataFrame(rows)


def write_report(path: Path, results: pd.DataFrame) -> None:
    summary = (
        results.groupby(["dataset", "representation"], observed=True)
        .agg(
            n_jackknives=("held_out_subject", "size"),
            min_rebuilt_vs_full_r=("rebuilt_vs_full_spearman", "min"),
            median_rebuilt_vs_full_r=("rebuilt_vs_full_spearman", "median"),
            min_terminal_root_auc=("rebuilt_terminal_vs_root_auc", "min"),
            median_terminal_root_auc=("rebuilt_terminal_vs_root_auc", "median"),
        )
        .reset_index()
    )
    lines = [
        "# Donor jackknife trajectory report",
        "",
        "Each donor was removed in turn, after which the neighbor graph, diffusion map, "
        "and donor-rooted DPT consensus were recomputed on the retained cells. PCA loadings "
        "and HVG selection were held fixed so this test isolates graph and root dependence.",
        "",
    ]
    for row in summary.itertuples():
        lines.append(
            f"- {row.dataset} {row.representation}: {row.n_jackknives} jackknives; "
            f"rebuilt/full r min={row.min_rebuilt_vs_full_r:.3f}, median={row.median_rebuilt_vs_full_r:.3f}; "
            f"terminal/root AUC min={row.min_terminal_root_auc:.3f}, median={row.median_terminal_root_auc:.3f}."
        )
    lines.extend([
        "",
        "These are topology robustness metrics, not donor-level disease tests. Disease "
        "contrasts must aggregate transition occupancy and activation residuals per donor.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    outputs = []
    for dataset in DATASETS:
        print(f"Jackknife {dataset}", flush=True)
        outputs.append(run_dataset(root, dataset, args))
    results = pd.concat(outputs, ignore_index=True)
    output_dir = root / "results" / "trajectory"
    results.to_csv(output_dir / "trajectory_donor_jackknife.csv", index=False)
    write_report(output_dir / "DONOR_JACKKNIFE_REPORT.md", results)


if __name__ == "__main__":
    main()
