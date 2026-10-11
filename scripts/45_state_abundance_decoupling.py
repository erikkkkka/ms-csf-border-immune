#!/usr/bin/env python3
"""Audit lineage specificity of MS-associated abundance/state decoupling."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm, pearsonr, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests


DATASETS = {
    "GSE138266": "IIH_control",
    "GSE286068": "non_inflammatory_control",
}
LINEAGES = [
    "BAM_stable_core",
    "classical_monocyte",
    "cDC2",
    "T",
    "B",
    "pDC",
    "cDC1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def exact_permutation_test(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Two-sided exact label permutation test for a difference in means."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=bool)
    observed = float(values[groups].mean() - values[~groups].mean())
    n_case = int(groups.sum())
    total = 0
    exceed = 0
    all_indices = np.arange(len(values))
    for case_indices in itertools.combinations(all_indices, n_case):
        permuted = np.zeros(len(values), dtype=bool)
        permuted[list(case_indices)] = True
        statistic = values[permuted].mean() - values[~permuted].mean()
        exceed += abs(statistic) >= abs(observed) - 1e-12
        total += 1
    return observed, exceed / total


def hedges_g(values: np.ndarray, groups: np.ndarray) -> float:
    groups = np.asarray(groups, dtype=bool)
    case = np.asarray(values, dtype=float)[groups]
    control = np.asarray(values, dtype=float)[~groups]
    degrees = len(case) + len(control) - 2
    if degrees <= 0:
        return np.nan
    pooled_variance = (
        (len(case) - 1) * case.var(ddof=1)
        + (len(control) - 1) * control.var(ddof=1)
    ) / degrees
    if pooled_variance <= 0:
        return np.nan
    correction = 1 - 3 / (4 * degrees - 1)
    return correction * (case.mean() - control.mean()) / math.sqrt(pooled_variance)


def load_total_cells(root: Path, gse: str) -> pd.DataFrame:
    path = root / "results" / "annotation_cleaned" / f"{gse}_clean_apc_sample_composition.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["tissue"].eq("CSF")]
    return frame[["sample_accession", "n_total_cells"]].drop_duplicates("sample_accession")


def load_abundance_table(root: Path, gse: str) -> pd.DataFrame:
    total = load_total_cells(root, gse)
    stable_path = (
        root / "results" / "bam_robustness" / "pseudobulk"
        / f"{gse}_CSF_BAM_stable_core_sample_metadata.csv"
    )
    stable = pd.read_csv(stable_path).rename(columns={"n_cells": "BAM_stable_core_n_cells"})
    columns = [
        "sample_accession", "subject_id", "diagnosis", "disease_group",
        "BAM_stable_core_n_cells",
    ]
    table = stable[columns].merge(total, on="sample_accession", how="left")

    standard_path = root / "results" / "pseudobulk" / f"{gse}_CSF_pseudobulk_sample_metadata.csv"
    standard = pd.read_csv(standard_path)
    for lineage in LINEAGES[1:]:
        source = standard.loc[
            standard["analysis_label"].eq(lineage), ["sample_accession", "n_cells"]
        ].rename(columns={"n_cells": f"{lineage}_n_cells"})
        table = table.merge(source, on="sample_accession", how="left")

    for lineage in LINEAGES:
        count = table[f"{lineage}_n_cells"].fillna(0).astype(float)
        total_count = table["n_total_cells"].astype(float)
        table[f"{lineage}_proportion"] = count / total_count
        # Empirical logit permits exact testing when a lineage has zero cells.
        table[f"{lineage}_logit_proportion"] = logit(
            (count + 0.5) / (total_count + 1.0)
        )
    table.insert(0, "gse", gse)
    return table


def endpoint_test(
    frame: pd.DataFrame,
    value_column: str,
    endpoint: str,
    lineage: str,
    gse: str,
    control_group: str,
) -> dict | None:
    subset = frame.loc[
        frame["disease_group"].isin(["MS", control_group]),
        ["disease_group", value_column],
    ].dropna()
    groups = subset["disease_group"].eq("MS").to_numpy()
    if groups.sum() < 3 or (~groups).sum() < 3:
        return None
    values = subset[value_column].to_numpy(dtype=float)
    difference, p_value = exact_permutation_test(values, groups)
    return {
        "gse": gse,
        "lineage": lineage,
        "endpoint": endpoint,
        "control_group": control_group,
        "n_ms": int(groups.sum()),
        "n_control": int((~groups).sum()),
        "ms_median": float(np.median(values[groups])),
        "control_median": float(np.median(values[~groups])),
        "mean_difference_ms_minus_control": difference,
        "hedges_g": hedges_g(values, groups),
        "permutation_p": p_value,
    }


def signed_stouffer(group: pd.DataFrame) -> tuple[float, float]:
    clipped = group["permutation_p"].clip(1e-12, 1.0)
    signs = np.sign(group["mean_difference_ms_minus_control"].to_numpy())
    z_values = norm.isf(clipped.to_numpy() / 2) * signs
    weights = np.sqrt((group["n_ms"] + group["n_control"]).to_numpy())
    combined_z = float(np.sum(weights * z_values) / np.sqrt(np.sum(weights ** 2)))
    return combined_z, float(2 * norm.sf(abs(combined_z)))


def meta_analyze(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lineage, endpoint), group in tests.groupby(["lineage", "endpoint"], observed=True):
        if len(group) < 2:
            continue
        z_value, p_value = signed_stouffer(group)
        directions = np.sign(group["mean_difference_ms_minus_control"])
        rows.append({
            "lineage": lineage,
            "endpoint": endpoint,
            "n_cohorts": len(group),
            "direction_concordant": bool(len(np.unique(directions)) == 1),
            "direction": "higher_in_MS" if z_value > 0 else "lower_in_MS",
            "stouffer_z": z_value,
            "stouffer_p": p_value,
        })
    output = pd.DataFrame(rows)
    output["fdr_bh"] = multipletests(output["stouffer_p"], method="fdr_bh")[1]
    return output


def abundance_state_correlations(sample_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value = "BAM_stable_core_logit_proportion"
    state = "BAM_stable_core_MHC_I_processing"
    subset = sample_table[["gse", "disease_group", value, state]].dropna().copy()
    for gse, group in subset.groupby("gse", observed=True):
        if len(group) >= 5:
            test = spearmanr(group[value], group[state])
            rows.append({
                "scope": gse,
                "n_samples": len(group),
                "correlation": test.statistic,
                "p_value": test.pvalue,
                "method": "Spearman",
            })
    subset["stratum"] = subset["gse"] + ":" + subset["disease_group"].astype(str)
    design = pd.get_dummies(subset["stratum"], drop_first=False, dtype=float).to_numpy()
    x = rankdata(subset[value])
    y = rankdata(subset[state])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    test = pearsonr(x_residual, y_residual)
    rows.append({
        "scope": "pooled_residualized",
        "n_samples": len(subset),
        "correlation": test.statistic,
        "p_value": test.pvalue,
        "method": "rank residuals adjusted for dataset:disease",
    })
    output = pd.DataFrame(rows)
    output["fdr_bh"] = multipletests(output["p_value"], method="fdr_bh")[1]
    return output


def write_report(path: Path, tests: pd.DataFrame, meta: pd.DataFrame, correlations: pd.DataFrame) -> None:
    def meta_row(lineage: str, endpoint: str) -> pd.Series:
        return meta.loc[(meta["lineage"] == lineage) & (meta["endpoint"] == endpoint)].iloc[0]

    abundance = meta_row("BAM_stable_core", "abundance_logit_proportion")
    state = meta_row("BAM_stable_core", "MHC_I_processing_state")
    decoupled = abundance.stouffer_z < 0 and state.stouffer_z > 0
    candidate_rows = []
    for lineage in LINEAGES:
        available = meta.loc[meta["lineage"].eq(lineage)].set_index("endpoint")
        if {"abundance_logit_proportion", "MHC_I_processing_state"}.issubset(available.index):
            ab = available.loc["abundance_logit_proportion"]
            st = available.loc["MHC_I_processing_state"]
            candidate_rows.append(
                (lineage, ab.stouffer_z, ab.stouffer_p, st.stouffer_z, st.stouffer_p)
            )

    lines = [
        "# CSF lineage state-abundance decoupling audit",
        "",
        "## Primary question",
        "",
        "Is the strict C1Q/APOE BAM-like core reduced in abundance while its MHC-I "
        "processing state increases in MS, and is that directional combination lineage-specific?",
        "",
        "## Strict-core BAM result",
        "",
        f"- Abundance meta-result: z={abundance.stouffer_z:.3f}, P={abundance.stouffer_p:.4g}, "
        f"FDR={abundance.fdr_bh:.4g}, direction={abundance.direction}, "
        f"cohort-concordant={abundance.direction_concordant}.",
        f"- MHC-I module-score meta-result: z={state.stouffer_z:.3f}, P={state.stouffer_p:.4g}, "
        f"FDR={state.fdr_bh:.4g}, direction={state.direction}, "
        f"cohort-concordant={state.direction_concordant}.",
        f"- Directional state-abundance decoupling criterion met: {decoupled}.",
        "",
        "The state test is a donor-level module-score sensitivity analysis. The gene-set "
        "camera tests remain the primary inference for MHC-I pathway enrichment.",
        "",
        "## Lineage comparison",
        "",
    ]
    for lineage, ab_z, ab_p, state_z, state_p in candidate_rows:
        lines.append(
            f"- {lineage}: abundance z={ab_z:.3f} (P={ab_p:.4g}); "
            f"MHC-I state z={state_z:.3f} (P={state_p:.4g})."
        )
    pooled = correlations.loc[correlations["scope"].eq("pooled_residualized")].iloc[0]
    lines.extend([
        "",
        "## Within-donor abundance-state relation",
        "",
        f"The pooled dataset:disease-adjusted abundance-state correlation was "
        f"r={pooled.correlation:.3f}, P={pooled.p_value:.4g}, FDR={pooled.fdr_bh:.4g}. "
        "A non-significant correlation means that group-level scarcity and activation "
        "should not be rewritten as a within-donor causal trade-off.",
        "",
        "## Interpretation rule",
        "",
        "Use 'state-abundance decoupling' only as a group-level phenotype when both "
        "directions replicate. Do not infer depletion-driven activation, emigration, "
        "or cDC2-to-BAM licensing from these cross-sectional data.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "results" / "state_abundance"
    output_dir.mkdir(parents=True, exist_ok=True)

    abundance_tables = [load_abundance_table(root, gse) for gse in DATASETS]
    abundance = pd.concat(abundance_tables, ignore_index=True)
    scores = pd.read_csv(
        root / "results" / "relay_specificity" / "CSF_stable_BAM_celltype_module_scores.csv"
    )
    sample_table = abundance.merge(
        scores.drop(columns=["subject_id", "diagnosis", "disease_group"], errors="ignore"),
        on=["gse", "sample_accession"],
        how="left",
    )
    sample_table.to_csv(output_dir / "CSF_lineage_state_abundance_sample_table.csv", index=False)

    rows = []
    for gse, control_group in DATASETS.items():
        frame = sample_table.loc[sample_table["gse"].eq(gse)]
        for lineage in LINEAGES:
            abundance_result = endpoint_test(
                frame,
                f"{lineage}_logit_proportion",
                "abundance_logit_proportion",
                lineage,
                gse,
                control_group,
            )
            if abundance_result:
                rows.append(abundance_result)
            state_result = endpoint_test(
                frame,
                f"{lineage}_MHC_I_processing",
                "MHC_I_processing_state",
                lineage,
                gse,
                control_group,
            )
            if state_result:
                rows.append(state_result)
    tests = pd.DataFrame(rows)
    tests["fdr_bh_within_endpoint"] = tests.groupby("endpoint")["permutation_p"].transform(
        lambda values: multipletests(values, method="fdr_bh")[1]
    )
    tests.to_csv(output_dir / "CSF_lineage_state_abundance_cohort_tests.csv", index=False)

    meta = meta_analyze(tests)
    meta.to_csv(output_dir / "CSF_lineage_state_abundance_meta_tests.csv", index=False)
    correlations = abundance_state_correlations(sample_table)
    correlations.to_csv(output_dir / "BAM_abundance_MHCI_correlations.csv", index=False)
    write_report(output_dir / "STATE_ABUNDANCE_DECOUPLING_REPORT.md", tests, meta, correlations)

    print(meta.sort_values(["endpoint", "stouffer_p"]).to_string(index=False))
    print("\nBAM abundance-state correlations")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
