#!/usr/bin/env python3
"""Test whether the two-axis result depends on arbitrary dichotomization."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests


COHORTS = {
    "GSE138266": "IIH_control",
    "GSE286068": "non_inflammatory_control",
}
PSEUDOTIME_CUTOFFS = [0.70, 0.75, 0.80, 0.85, 0.90]
ACTIVATION_QUANTILES = [0.67, 0.75, 0.80, 0.90]


def exact_test(case: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    pooled = np.concatenate([case, control])
    observed = case.mean() - control.mean()
    differences = []
    for combination in itertools.combinations(range(len(pooled)), len(case)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(combination)] = True
        differences.append(pooled[mask].mean() - pooled[~mask].mean())
    p_value = np.mean(np.abs(differences) >= abs(observed) - 1e-12)
    return float(observed), float(p_value)


def cohort_grid(root: Path, dataset: str, control_group: str) -> pd.DataFrame:
    path = root / "results" / "trajectory" / "continuum_objects" / f"{dataset}_monocyte_BAM_continuum.h5ad"
    adata = sc.read_h5ad(path, backed="r")
    frame = adata.obs.loc[adata.obs["tissue"].astype(str).str.upper().eq("CSF")].copy()
    control = frame["disease_group"].astype(str).eq(control_group)
    rows = []
    for activation_quantile in ACTIVATION_QUANTILES:
        activation_cutoff = frame.loc[control, "two_axis_activation_residual"].quantile(activation_quantile)
        for pseudotime_cutoff in PSEUDOTIME_CUTOFFS:
            frame["endpoint"] = (
                (frame["consensus_pseudotime"] < pseudotime_cutoff)
                & (frame["two_axis_activation_residual"] > activation_cutoff)
            )
            donor = (
                frame.groupby(["subject_id", "disease_group"], observed=True)
                .agg(value=("endpoint", "mean"), n_cells=("endpoint", "size"))
                .reset_index()
            )
            donor = donor.loc[donor["n_cells"] >= 20]
            case = donor.loc[donor["disease_group"].astype(str).eq("MS"), "value"].to_numpy(float)
            control_values = donor.loc[
                donor["disease_group"].astype(str).eq(control_group), "value"
            ].to_numpy(float)
            difference, p_value = exact_test(case, control_values)
            rows.append({
                "dataset": dataset,
                "pseudotime_cutoff": pseudotime_cutoff,
                "activation_control_quantile": activation_quantile,
                "activation_cutoff": activation_cutoff,
                "n_MS": len(case),
                "n_control": len(control_values),
                "mean_difference_MS_minus_control": difference,
                "exact_permutation_p": p_value,
            })
    adata.file.close()
    return pd.DataFrame(rows)


def combine_grid(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in tests.groupby(
        ["pseudotime_cutoff", "activation_control_quantile"], observed=True
    ):
        signs = np.sign(frame["mean_difference_MS_minus_control"].to_numpy(float))
        p_values = np.clip(frame["exact_permutation_p"].to_numpy(float), 1e-12, 1)
        z_scores = signs * stats.norm.isf(p_values / 2)
        weights = np.sqrt((frame["n_MS"] + frame["n_control"]).to_numpy(float))
        z_value = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))
        rows.append({
            "pseudotime_cutoff": keys[0],
            "activation_control_quantile": keys[1],
            "direction_concordant": len(set(signs)) == 1,
            "combined_z": z_value,
            "stouffer_p": 2 * stats.norm.sf(abs(z_value)),
            "mean_case_control_difference": frame["mean_difference_MS_minus_control"].mean(),
        })
    output = pd.DataFrame(rows)
    output["fdr_bh_across_grid"] = multipletests(output["stouffer_p"], method="fdr_bh")[1]
    return output.sort_values(["activation_control_quantile", "pseudotime_cutoff"])


def write_report(path: Path, grid: pd.DataFrame) -> None:
    lines = [
        "# Decoupled-inflammatory fraction threshold sensitivity",
        "",
        f"Tested {len(grid)} prespecified combinations of pseudotime cutoffs "
        f"({', '.join(map(str, PSEUDOTIME_CUTOFFS))}) and control activation quantiles "
        f"({', '.join(map(str, ACTIVATION_QUANTILES))}).",
        "",
        f"- Direction was concordantly higher in MS for {int(grid['direction_concordant'].sum())}/{len(grid)} combinations.",
        f"- {int((grid['stouffer_p'] < 0.05).sum())}/{len(grid)} combinations had meta P<0.05.",
        f"- {int((grid['fdr_bh_across_grid'] < 0.05).sum())}/{len(grid)} combinations survived correction across the grid.",
        f"- Meta P range: {grid['stouffer_p'].min():.4g} to {grid['stouffer_p'].max():.4g}; median={grid['stouffer_p'].median():.4g}.",
        "",
        "The continuous decoupling index in the primary two-axis model is the preferred "
        "endpoint because it avoids dichotomization; this grid tests whether the categorical "
        "result is qualitatively stable rather than selecting an optimal cutoff.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    tests = pd.concat(
        [cohort_grid(root, dataset, control) for dataset, control in COHORTS.items()],
        ignore_index=True,
    )
    grid = combine_grid(tests)
    output_dir = root / "results" / "two_axis"
    tests.to_csv(output_dir / "decoupling_threshold_cohort_tests.csv", index=False)
    grid.to_csv(output_dir / "decoupling_threshold_meta_grid.csv", index=False)
    write_report(output_dir / "DECOUPLING_THRESHOLD_SENSITIVITY.md", grid)


if __name__ == "__main__":
    main()
