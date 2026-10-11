#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MODULES = {
    "IFN_antiviral": [
        "IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2",
        "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1",
    ],
    "MHC_I_processing": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    ],
}


def read_counts(path, samples):
    table = pd.read_csv(path)
    table = table.loc[table["gene_symbol"].notna(), ["gene_symbol"] + samples]
    return table.groupby("gene_symbol", observed=True)[samples].sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--gse", default="GSE194078")
    parser.add_argument("--min-cells", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    pb_dir = root / "results/pseudobulk"
    out_dir = root / "results/relay"
    out_dir.mkdir(parents=True, exist_ok=True)

    tissue_data = {}
    for tissue in ["CSF", "PBMC"]:
        metadata = pd.read_csv(pb_dir / f"{args.gse}_{tissue}_pseudobulk_sample_metadata.csv")
        metadata = metadata.loc[
            (metadata["analysis_label"] == "cDC2") & (metadata["n_cells"] >= args.min_cells)
        ].copy()
        samples = metadata["sample_accession"].tolist()
        counts = read_counts(pb_dir / f"{args.gse}_{tissue}_cDC2_counts.csv.gz", samples)
        tissue_data[tissue] = (metadata, counts)

    common_genes = tissue_data["CSF"][1].index.intersection(tissue_data["PBMC"][1].index)
    matrices = []
    sample_rows = []
    for tissue, (metadata, counts) in tissue_data.items():
        counts = counts.loc[common_genes]
        libraries = counts.sum(axis=0).astype(float)
        log_cpm = np.log2((counts.T + 0.5).div(libraries + 1.0, axis=0) * 1e6)
        matrices.append(log_cpm)
        rows = metadata.set_index("sample_accession").loc[log_cpm.index, [
            "subject_id", "diagnosis", "disease_group", "n_cells"
        ]]
        rows.index.name = "sample_accession"
        rows = rows.reset_index()
        rows["tissue"] = tissue
        sample_rows.append(rows)
    combined = pd.concat(matrices)
    sample_table = pd.concat(sample_rows, ignore_index=True).set_index("sample_accession")
    std = combined.std(axis=0, ddof=0)
    standardized = (combined.loc[:, std > 0] - combined.loc[:, std > 0].mean(axis=0)) / std[std > 0]

    for module, genes in MODULES.items():
        present = [gene for gene in genes if gene in standardized.columns]
        sample_table[module] = standardized[present].mean(axis=1)
        sample_table[f"{module}_n_genes"] = len(present)
    sample_table.reset_index().to_csv(
        out_dir / f"{args.gse}_paired_CSF_PBMC_cDC2_module_scores.csv", index=False
    )

    rows = []
    for module in MODULES:
        wide = sample_table.reset_index().pivot_table(
            index=["subject_id", "disease_group"], columns="tissue", values=module, aggfunc="first"
        ).dropna()
        for group in ["all"] + sorted(wide.index.get_level_values("disease_group").unique()):
            frame = wide if group == "all" else wide.loc[wide.index.get_level_values("disease_group") == group]
            if len(frame) < 3:
                continue
            test = wilcoxon(frame["CSF"], frame["PBMC"], alternative="two-sided", method="auto")
            rows.append({
                "module": module,
                "disease_group": group,
                "n_pairs": len(frame),
                "median_csf": frame["CSF"].median(),
                "median_pbmc": frame["PBMC"].median(),
                "median_paired_difference": (frame["CSF"] - frame["PBMC"]).median(),
                "p_value": test.pvalue,
                "inference": "primary_paired" if group == "all" else "descriptive_subgroup",
            })
    pd.DataFrame(rows).to_csv(
        out_dir / f"{args.gse}_paired_CSF_PBMC_cDC2_module_tests.csv", index=False
    )


if __name__ == "__main__":
    main()
