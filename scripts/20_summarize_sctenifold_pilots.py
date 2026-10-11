#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


GENE_SETS = {
    "MHC_I_processing": [
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    ],
    "BAM_homeostatic": [
        "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1",
        "LYVE1", "FOLR2", "DAB2", "MSR1", "CD163", "HAMP",
    ],
    "IFN_signaling": [
        "STAT1", "STAT2", "IRF1", "IRF7", "IFNGR1", "IFNGR2", "IFNAR1",
        "IFNAR2", "JAK1", "JAK2", "TYK2", "ISG15", "MX1", "OAS1", "GBP1",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--run-tag", default="795genes_5nets")
    parser.add_argument("--gse", default="GSE138266")
    parser.add_argument("--targets", default="TAP1,PSME2,PSMB9")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "results/virtual_knockout"
    rows = []
    gene_rows = []
    for target in args.targets.split(","):
        path = out_dir / f"{args.gse}_MS_CSF_BAM_{target}_scTenifold_diffRegulation_{args.run_tag}.csv"
        result = pd.read_csv(path).sort_values("distance", ascending=False).reset_index(drop=True)
        result = result.loc[result["gene"] != target].copy()
        result["rank"] = np.arange(1, len(result) + 1)
        result["top_percentile"] = 1 - (result["rank"] - 1) / len(result)
        for gene_set, genes in GENE_SETS.items():
            mask = result["gene"].isin(genes)
            inside = result.loc[mask, "distance"].to_numpy()
            outside = result.loc[~mask, "distance"].to_numpy()
            if len(inside) == 0:
                continue
            test = mannwhitneyu(inside, outside, alternative="greater", method="auto")
            auc = test.statistic / (len(inside) * len(outside))
            rows.append({
                "target": target,
                "gene_set": gene_set,
                "n_genes_present": len(inside),
                "rank_enrichment_auc": auc,
                "rank_enrichment_p": test.pvalue,
                "median_top_percentile": result.loc[mask, "top_percentile"].median(),
                "n_genes_fdr_0_05": int((result.loc[mask, "p.adj"] < 0.05).sum()),
            })
            selected = result.loc[mask, ["gene", "rank", "top_percentile", "distance", "p.adj"]].copy()
            selected.insert(0, "gene_set", gene_set)
            selected.insert(0, "target", target)
            gene_rows.append(selected)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"{args.gse}_scTenifold_pilot_module_enrichment_{args.run_tag}.csv", index=False)
    pd.concat(gene_rows, ignore_index=True).to_csv(
        out_dir / f"{args.gse}_scTenifold_pilot_module_gene_ranks_{args.run_tag}.csv", index=False
    )


if __name__ == "__main__":
    main()
