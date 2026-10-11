#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


MODULES = {
    "IFN_antiviral": [
        "IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2",
        "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1",
    ],
    "MHC_I_processing": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    ],
    "BAM_homeostatic": [
        "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1",
        "LYVE1", "FOLR2", "DAB2", "MSR1", "CD163", "HAMP",
    ],
    "T_cytotoxic": [
        "CD8A", "CD8B", "NKG7", "CCL5", "PRF1", "GZMB", "GZMK", "CTSW", "CST7",
    ],
    "IFNG_effector": [
        "IFNG", "TBX21", "EOMES", "CXCR3", "CCL5", "NKG7", "STAT4", "IL12RB1", "IL12RB2",
    ],
}

RELAY_PAIRS = [
    ("cDC2_IFN_antiviral", "BAM_like_candidate_MHC_I_processing"),
    ("cDC2_MHC_I_processing", "BAM_like_candidate_MHC_I_processing"),
    ("pDC_prop_total", "cDC2_IFN_antiviral"),
    ("T_IFNG_effector", "cDC2_IFN_antiviral"),
    ("T_IFNG_effector", "BAM_like_candidate_MHC_I_processing"),
    ("T_T_cytotoxic", "BAM_like_candidate_MHC_I_processing"),
    ("BAM_like_candidate_MHC_I_processing", "B_prop_total"),
    ("BAM_like_candidate_MHC_I_processing", "Plasma_prop_total"),
    ("BAM_like_candidate_MHC_I_processing", "BAM_like_candidate_prop_total"),
]


def load_module_scores(pb_dir, gse, label, min_cells):
    counts_path = pb_dir / f"{gse}_CSF_{label}_counts.csv.gz"
    metadata_path = pb_dir / f"{gse}_CSF_pseudobulk_sample_metadata.csv"
    counts = pd.read_csv(counts_path)
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.loc[metadata["analysis_label"] == label].copy()
    samples = metadata.loc[metadata["n_cells"] >= min_cells, "sample_accession"].tolist()
    if len(samples) < 3:
        return pd.DataFrame(columns=["sample_accession"])

    gene_counts = counts.loc[counts["gene_symbol"].notna(), ["gene_symbol"] + samples]
    gene_counts = gene_counts.groupby("gene_symbol", observed=True)[samples].sum()
    library_sizes = counts[samples].sum(axis=0).astype(float)
    log_cpm = np.log2((gene_counts.T + 0.5).div(library_sizes + 1.0, axis=0) * 1e6)
    output = pd.DataFrame({"sample_accession": samples})
    output[f"{label}_n_cells"] = metadata.set_index("sample_accession").loc[samples, "n_cells"].to_numpy()

    for module, genes in MODULES.items():
        present = [
            gene for gene in genes
            if gene in log_cpm.columns and (gene_counts.loc[gene] > 0).sum() >= 2
        ]
        matrix = log_cpm[present]
        std = matrix.std(axis=0, ddof=0)
        informative = std.index[std > 0]
        if len(informative) < 5:
            output[f"{label}_{module}"] = np.nan
            output[f"{label}_{module}_n_genes"] = len(informative)
            continue
        standardized = (matrix[informative] - matrix[informative].mean(axis=0)) / std[informative]
        output[f"{label}_{module}"] = standardized.mean(axis=1).to_numpy()
        output[f"{label}_{module}_n_genes"] = len(informative)
    return output


def add_composition(root, gse, table):
    ann_dir = root / "results/annotation_cleaned"
    broad = pd.read_csv(ann_dir / f"{gse}_clean_broad_sample_composition.csv")
    apc = pd.read_csv(ann_dir / f"{gse}_clean_apc_sample_composition.csv")
    broad = broad.loc[(broad["tissue"] == "CSF") & broad["label"].isin(["B", "Plasma"])]
    apc = apc.loc[
        (apc["tissue"] == "CSF")
        & apc["label"].isin(["pDC", "cDC2", "BAM_like_candidate"])
    ]
    broad_wide = broad.pivot(index="sample_accession", columns="label", values="prop_total")
    apc_wide = apc.pivot(index="sample_accession", columns="label", values="prop_total")
    broad_wide = broad_wide.rename(columns=lambda value: f"{value}_prop_total")
    apc_wide = apc_wide.rename(columns=lambda value: f"{value}_prop_total")
    return table.merge(broad_wide, left_on="sample_accession", right_index=True, how="left").merge(
        apc_wide, left_on="sample_accession", right_index=True, how="left"
    )


def residualize_ranks(values, strata):
    ranked = rankdata(values, method="average")
    design = pd.get_dummies(strata, drop_first=False, dtype=float).to_numpy()
    fitted = design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - fitted


def pooled_stratified_test(table, x_col, y_col, permutations=10000, seed=12345):
    frame = table[["gse", "disease_group", x_col, y_col]].dropna().copy()
    frame["stratum"] = frame["gse"].astype(str) + ":" + frame["disease_group"].astype(str)
    if len(frame) < 8:
        return None
    x_res = residualize_ranks(frame[x_col].to_numpy(), frame["stratum"])
    y_res = residualize_ranks(frame[y_col].to_numpy(), frame["stratum"])
    observed = pearsonr(x_res, y_res).statistic
    rng = np.random.default_rng(seed)
    stratum_indices = [np.flatnonzero(frame["stratum"].to_numpy() == key) for key in frame["stratum"].unique()]
    exceed = 0
    for _ in range(permutations):
        permuted = y_res.copy()
        for indices in stratum_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        value = pearsonr(x_res, permuted).statistic
        exceed += abs(value) >= abs(observed)
    return {
        "scope": "pooled_group_residualized",
        "gse": "pooled",
        "x": x_col,
        "y": y_col,
        "n_samples": len(frame),
        "correlation": observed,
        "p_value": (exceed + 1) / (permutations + 1),
        "method": f"rank_residual_pearson_block_permutation_{permutations}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--min-cells", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    pb_dir = root / "results/pseudobulk"
    out_dir = root / "results/relay"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for gse in ["GSE138266", "GSE286068", "GSE194078"]:
        metadata = pd.read_csv(pb_dir / f"{gse}_CSF_pseudobulk_sample_metadata.csv")
        metadata = metadata.drop_duplicates("sample_accession")[[
            "sample_accession", "subject_id", "diagnosis", "disease_group"
        ]]
        table = metadata.copy()
        for label in ["cDC2", "BAM_like_candidate", "T"]:
            table = table.merge(
                load_module_scores(pb_dir, gse, label, args.min_cells),
                on="sample_accession",
                how="left",
            )
        table = add_composition(root, gse, table)
        table.insert(0, "gse", gse)
        datasets.append(table)
    relay = pd.concat(datasets, ignore_index=True)
    relay.to_csv(out_dir / "CSF_sample_level_relay_scores.csv", index=False)

    rows = []
    for x_col, y_col in RELAY_PAIRS:
        for gse, frame in relay.groupby("gse", observed=True):
            pair = frame[[x_col, y_col]].dropna()
            if len(pair) < 4:
                continue
            test = spearmanr(pair[x_col], pair[y_col])
            rows.append({
                "scope": "within_dataset",
                "gse": gse,
                "x": x_col,
                "y": y_col,
                "n_samples": len(pair),
                "correlation": test.statistic,
                "p_value": test.pvalue,
                "method": "spearman",
            })
        pooled = pooled_stratified_test(
            relay, x_col, y_col, permutations=args.permutations, seed=12345
        )
        if pooled is not None:
            rows.append(pooled)
    results = pd.DataFrame(rows)
    results["fdr_bh"] = np.nan
    pooled_mask = results["scope"].eq("pooled_group_residualized")
    if pooled_mask.any():
        p_values = results.loc[pooled_mask, "p_value"].to_numpy()
        order = np.argsort(p_values)
        adjusted = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.minimum(adjusted, 1.0)
        results.loc[pooled_mask, "fdr_bh"] = restored
    results.to_csv(out_dir / "CSF_sample_level_relay_correlations.csv", index=False)


if __name__ == "__main__":
    main()
