#!/usr/bin/env python3
"""Export endpoint AUCs from saved multiroot DPT values, without refitting graphs."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("GSE138266", "GSE194078", "GSE286068")
read_h5_obs = importlib.import_module("75_generate_manuscript_figures").read_h5_obs


def endpoint_auc(values: np.ndarray, root: np.ndarray, terminal: np.ndarray) -> float:
    if not root.any() or not terminal.any() or not np.isfinite(values[root | terminal]).all():
        raise ValueError("Endpoint AUC requires finite scores and non-empty root and terminal groups")
    return float(stats.mannwhitneyu(values[terminal], values[root], alternative="two-sided").statistic
                 / (terminal.sum() * root.sum()))


def full_multiroot_endpoints(root: Path) -> pd.DataFrame:
    rows = []
    for dataset in COHORTS:
        source = f"results/trajectory/continuum_objects/{dataset}_monocyte_BAM_continuum.h5ad"
        frame, _ = read_h5_obs(root / source, [
            "trajectory_role", "subject_id", "dpt_multiroot_primary", "dpt_multiroot_marker_excluded",
        ])
        roots = frame.trajectory_role.eq("root_candidate").to_numpy()
        terminals = frame.trajectory_role.eq("terminal_candidate").to_numpy()
        for representation in ("primary", "marker_excluded"):
            column = f"dpt_multiroot_{representation}"
            rows.append({
                "dataset": dataset, "representation": representation,
                "method": "DPT: median percentile rank across donor roots",
                "n_cells": len(frame), "n_subjects": frame.subject_id.nunique(),
                "n_root_donors": frame.loc[roots, "subject_id"].nunique(),
                "n_root": int(roots.sum()), "n_terminal": int(terminals.sum()),
                "full_terminal_vs_root_auc": endpoint_auc(frame[column].to_numpy(float), roots, terminals),
                "score_column": column, "source": source,
                "comparison": "trajectory_role terminal_candidate versus root_candidate",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result = full_multiroot_endpoints(root)
    result.to_csv(root / "results/trajectory/full_multiroot_endpoint_auc.csv", index=False)
    print(result[["dataset", "representation", "full_terminal_vs_root_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
