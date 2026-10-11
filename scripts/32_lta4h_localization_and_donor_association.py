#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon


MODULES = {
    "MHC_I_processing": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    ],
    "BAM_homeostatic": [
        "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1",
        "LYVE1", "FOLR2", "DAB2", "MSR1", "CD163", "HAMP",
    ],
}


def gene_vector(adata, gene: str) -> np.ndarray:
    if adata.raw is None or gene not in adata.raw.var_names:
        raise RuntimeError(f"{gene} missing from raw expression matrix")
    values = adata.raw[:, gene].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def summarize_cell_expression(adata, label_col: str, level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = adata.obs[["gse", "sample_accession", "tissue", "disease_group", label_col]].copy()
    obs["LTA4H_log1p_expression"] = gene_vector(adata, "LTA4H")
    obs["LTA4H_detected"] = obs["LTA4H_log1p_expression"] > 0
    obs = obs.rename(columns={label_col: "cell_label"})

    aggregate = obs.groupby(
        ["gse", "tissue", "disease_group", "cell_label"], observed=True, dropna=False
    ).agg(
        n_cells=("LTA4H_log1p_expression", "size"),
        mean_log1p_expression=("LTA4H_log1p_expression", "mean"),
        median_log1p_expression=("LTA4H_log1p_expression", "median"),
        pct_detected=("LTA4H_detected", "mean"),
    ).reset_index()
    aggregate.insert(1, "annotation_level", level)
    aggregate["pct_detected"] *= 100

    sample = obs.groupby(
        ["gse", "sample_accession", "tissue", "disease_group", "cell_label"],
        observed=True, dropna=False,
    ).agg(
        n_cells=("LTA4H_log1p_expression", "size"),
        mean_log1p_expression=("LTA4H_log1p_expression", "mean"),
        pct_detected=("LTA4H_detected", "mean"),
    ).reset_index()
    sample.insert(1, "annotation_level", level)
    sample["pct_detected"] *= 100
    return aggregate, sample


def paired_apc_tests(sample_table: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    rows = []
    frame = sample_table.loc[
        (sample_table["annotation_level"] == "fine_APC")
        & (sample_table["tissue"] == "CSF")
        & (sample_table["n_cells"] >= min_cells)
    ]
    for cohort, cohort_table in frame.groupby("gse", observed=True):
        for comparator in ["cDC2", "classical_monocyte", "pDC"]:
            for metric in ["mean_log1p_expression", "pct_detected"]:
                wide = cohort_table.loc[
                    cohort_table["cell_label"].isin(["BAM_like_candidate", comparator])
                ].pivot(index="sample_accession", columns="cell_label", values=metric).dropna()
                if len(wide) < 4:
                    continue
                differences = wide["BAM_like_candidate"] - wide[comparator]
                test = wilcoxon(
                    wide["BAM_like_candidate"], wide[comparator],
                    alternative="two-sided", zero_method="wilcox",
                )
                rows.append({
                    "gse": cohort,
                    "comparison": f"BAM_like_candidate_vs_{comparator}",
                    "metric": metric,
                    "n_paired_samples": len(wide),
                    "median_BAM": wide["BAM_like_candidate"].median(),
                    "median_comparator": wide[comparator].median(),
                    "median_paired_difference": differences.median(),
                    "p_value": test.pvalue,
                    "test": "paired_wilcoxon",
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_bh"] = np.nan
        for cohort, indices in result.groupby("gse").groups.items():
            p = result.loc[indices, "p_value"].to_numpy()
            order = np.argsort(p)
            adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
            adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
            restored = np.empty_like(adjusted)
            restored[order] = np.minimum(adjusted, 1.0)
            result.loc[indices, "fdr_bh"] = restored
    return result


def load_bam_pseudobulk(root: Path, cohort: str, min_cells: int) -> pd.DataFrame:
    pb_dir = root / "results" / "pseudobulk"
    counts = pd.read_csv(pb_dir / f"{cohort}_CSF_BAM_like_candidate_counts.csv.gz")
    metadata = pd.read_csv(pb_dir / f"{cohort}_CSF_pseudobulk_sample_metadata.csv")
    metadata = metadata.loc[
        (metadata["analysis_label"] == "BAM_like_candidate")
        & (metadata["n_cells"] >= min_cells)
    ].copy()
    samples = [sample for sample in metadata["sample_accession"] if sample in counts.columns]
    genes = counts.loc[counts["gene_symbol"].notna(), ["gene_symbol"] + samples]
    genes = genes.groupby("gene_symbol", observed=True)[samples].sum()
    library_sizes = counts[samples].sum(axis=0).astype(float)
    log_cpm = np.log2((genes.T + 0.5).div(library_sizes + 1.0, axis=0) * 1e6)

    output = metadata.drop_duplicates("sample_accession").set_index("sample_accession").loc[samples, [
        "subject_id", "diagnosis", "disease_group", "n_cells",
    ]].reset_index()
    output.insert(0, "gse", cohort)
    output["LTA4H_log2CPM"] = log_cpm["LTA4H"].to_numpy()
    for module, gene_list in MODULES.items():
        present = [
            gene for gene in gene_list
            if gene in log_cpm.columns and (genes.loc[gene] > 0).sum() >= 2
        ]
        matrix = log_cpm[present]
        informative = matrix.std(axis=0, ddof=0)
        informative = informative.index[informative > 0].tolist()
        standardized = (
            matrix[informative] - matrix[informative].mean(axis=0)
        ) / matrix[informative].std(axis=0, ddof=0)
        output[f"{module}_score"] = standardized.mean(axis=1).to_numpy()
        output[f"{module}_n_genes"] = len(informative)
        output[f"{module}_genes"] = ";".join(informative)
    return output


def residualize_ranks(values: np.ndarray, strata: pd.Series) -> np.ndarray:
    ranked = rankdata(values, method="average")
    design = pd.get_dummies(strata, drop_first=False, dtype=float).to_numpy()
    fitted = design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - fitted


def pooled_test(table: pd.DataFrame, outcome: str, permutations: int, seed: int) -> dict:
    frame = table[["gse", "disease_group", "LTA4H_log2CPM", outcome]].dropna().copy()
    frame["stratum"] = frame["gse"].astype(str) + ":" + frame["disease_group"].astype(str)
    x_res = residualize_ranks(frame["LTA4H_log2CPM"].to_numpy(), frame["stratum"])
    y_res = residualize_ranks(frame[outcome].to_numpy(), frame["stratum"])
    observed = pearsonr(x_res, y_res).statistic
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(frame["stratum"].to_numpy() == key) for key in frame["stratum"].unique()]
    exceed = 0
    for _ in range(permutations):
        permuted = y_res.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        exceed += abs(pearsonr(x_res, permuted).statistic) >= abs(observed)
    return {
        "scope": "pooled_group_residualized", "gse": "pooled",
        "outcome": outcome, "n_samples": len(frame), "correlation": observed,
        "p_value": (exceed + 1) / (permutations + 1),
        "method": f"rank_residual_pearson_block_permutation_{permutations}",
    }


def donor_associations(table: pd.DataFrame, permutations: int) -> pd.DataFrame:
    rows = []
    for outcome in ["MHC_I_processing_score", "BAM_homeostatic_score"]:
        for cohort, frame in table.groupby("gse", observed=True):
            pair = frame[["LTA4H_log2CPM", outcome]].dropna()
            if len(pair) < 4:
                continue
            test = spearmanr(pair["LTA4H_log2CPM"], pair[outcome])
            rows.append({
                "scope": "within_dataset", "gse": cohort, "outcome": outcome,
                "n_samples": len(pair), "correlation": test.statistic,
                "p_value": test.pvalue, "method": "spearman",
            })
        rows.append(pooled_test(table, outcome, permutations, seed=20260810))
    result = pd.DataFrame(rows)
    result["fdr_bh"] = np.nan
    mask = result["scope"].eq("pooled_group_residualized")
    p = result.loc[mask, "p_value"].to_numpy()
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[mask, "fdr_bh"] = restored
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results" / "lta4h"
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_frames = []
    sample_frames = []
    donor_frames = []
    for cohort in ["GSE138266", "GSE194078", "GSE286068"]:
        broad = sc.read_h5ad(root / "results" / "cleaned" / f"{cohort}_clean_decontx_singlets.h5ad")
        aggregate, sample = summarize_cell_expression(broad, "broad_cell_type", "broad")
        aggregate_frames.append(aggregate)
        sample_frames.append(sample)
        del broad

        apc = sc.read_h5ad(root / "results" / "cleaned" / f"{cohort}_clean_apc_refined.h5ad")
        aggregate, sample = summarize_cell_expression(apc, "fine_apc_label", "fine_APC")
        aggregate_frames.append(aggregate)
        sample_frames.append(sample)
        del apc
        donor_frames.append(load_bam_pseudobulk(root, cohort, args.min_cells))

    aggregate = pd.concat(aggregate_frames, ignore_index=True)
    samples = pd.concat(sample_frames, ignore_index=True)
    donors = pd.concat(donor_frames, ignore_index=True)
    aggregate.to_csv(out_dir / "LTA4H_celltype_localization.csv", index=False)
    samples.to_csv(out_dir / "LTA4H_sample_celltype_expression.csv.gz", index=False, compression="gzip")
    paired_apc_tests(samples, args.min_cells).to_csv(
        out_dir / "LTA4H_paired_APC_localization_tests.csv", index=False
    )
    donors.to_csv(out_dir / "LTA4H_BAM_pseudobulk_donor_scores.csv", index=False)
    donor_associations(donors, args.permutations).to_csv(
        out_dir / "LTA4H_BAM_donor_module_associations.csv", index=False
    )


if __name__ == "__main__":
    main()
