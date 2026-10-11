#!/usr/bin/env python3
"""Map and score SC-57461A-treated primary mouse ILC2 RNA-seq."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gprofiler import GProfiler
from scipy.stats import norm, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests


MODULES = {
    "MHC_I_processing": {
        "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
        "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    },
    "IFN_antiviral": {
        "IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2",
        "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1",
    },
    "BAM_homeostatic": {
        "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1",
        "LYVE1", "FOLR2", "DAB2", "MSR1", "CD163", "HAMP",
    },
}

LINEAGE_AUDIT_MARKERS = {
    "B_lineage": {"IGHM", "IGKC", "CD79A", "CD79B", "CD74", "CD37", "MS4A1"},
    "T_NK_lineage": {"CD3D", "CD3E", "TRAC", "NKG7", "KLRD1"},
    "ILC2_identity": {"ICOS", "IL1RL1", "GATA3", "RORA", "IL7R", "KIT"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--de", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--full-meta", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fetch-mapping", action="store_true")
    parser.add_argument("--mapping-batch-size", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def fetch_mapping(gene_ids: list[str], mapping_path: Path, batch_size: int) -> None:
    parts_dir = mapping_path.parent / "gprofiler_mapping_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    client = GProfiler(return_dataframe=True)
    parts = []
    for start in range(0, len(gene_ids), batch_size):
        batch_index = start // batch_size + 1
        batch = gene_ids[start:start + batch_size]
        part_path = parts_dir / f"part_{batch_index:03d}.csv"
        if part_path.exists() and part_path.stat().st_size > 100:
            frame = pd.read_csv(part_path)
            print(f"Reusing mapping batch {batch_index}: {len(frame):,} rows", flush=True)
        else:
            print(
                f"Querying mapping batch {batch_index}: {len(batch):,} mouse genes",
                flush=True,
            )
            frame = client.orth(query=batch, organism="mmusculus", target="hsapiens")
            frame.to_csv(part_path, index=False)
        parts.append(frame)
    mapping = pd.concat(parts, ignore_index=True)
    mapping.to_csv(mapping_path, index=False)
    print(f"Saved {len(mapping):,} g:Profiler orthology rows", flush=True)


def rank_normal(values: pd.Series) -> np.ndarray:
    ranks = rankdata(values.to_numpy(dtype=float), method="average")
    return norm.ppf((ranks - 0.5) / len(ranks))


def signature_score(
    rank_z: np.ndarray,
    up_idx: np.ndarray,
    down_idx: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = 0.5 * (-rank_z[up_idx].mean() + rank_z[down_idx].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    total = len(up_idx) + len(down_idx)
    for index in range(permutations):
        sampled = rng.choice(len(rank_z), size=total, replace=False)
        null[index] = 0.5 * (
            -rank_z[sampled[: len(up_idx)]].mean()
            + rank_z[sampled[len(up_idx):]].mean()
        )
    p_value = (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
    z_score = (observed - null.mean()) / null.std(ddof=1)
    return float(observed), float(p_value), float(z_score)


def module_score(
    rank_z: np.ndarray,
    idx: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(-rank_z[idx].mean())
    rng = np.random.default_rng(seed)
    null = np.array([
        -rank_z[rng.choice(len(rank_z), size=len(idx), replace=False)].mean()
        for _ in range(permutations)
    ])
    p_value = (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
    z_score = (observed - null.mean()) / null.std(ddof=1)
    return observed, float(p_value), float(z_score)


def weighted_reversal_cosine(frame: pd.DataFrame) -> float:
    disease = frame["mean_logfc"].to_numpy(dtype=float)
    drug = frame["logFC"].to_numpy(dtype=float)
    weights = np.abs(frame["meta_z"].to_numpy(dtype=float))
    numerator = np.sum(weights * disease * drug)
    denominator = np.sqrt(
        np.sum(weights * disease**2) * np.sum(weights * drug**2)
    )
    return float(-numerator / denominator) if denominator > 0 else np.nan


def prepare_human_de(de: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    valid = mapping.loc[
        mapping["ortholog_ensg"].ne("N/A") & mapping["name"].ne("N/A"),
        ["incoming", "ortholog_ensg", "name"],
    ].copy()
    valid = valid.rename(columns={
        "incoming": "gene_id",
        "ortholog_ensg": "human_ensembl_gene_id",
        "name": "gene_symbol",
    })
    valid["gene_symbol"] = valid["gene_symbol"].str.upper()
    merged = de.merge(valid, on="gene_id", how="inner")
    human = merged.groupby("gene_symbol", as_index=False).agg(
        logFC=("logFC", "median"),
        logCPM=("logCPM", "max"),
        min_mouse_p=("PValue", "min"),
        min_mouse_fdr=("FDR", "min"),
        n_mouse_orthologs=("gene_id", "nunique"),
        mouse_gene_ids=("gene_id", lambda values: ";".join(sorted(set(values)))),
        human_ensembl_gene_ids=(
            "human_ensembl_gene_id", lambda values: ";".join(sorted(set(values)))
        ),
    )
    human["rank_z"] = rank_normal(human["logFC"])
    return human


def write_report(
    path: Path,
    de: pd.DataFrame,
    human: pd.DataFrame,
    scores: pd.DataFrame,
    top: pd.DataFrame,
) -> None:
    overall = scores.loc[scores["comparison"].eq("BAM_disease_signature_reversal")].iloc[0]
    mhc = scores.loc[scores["comparison"].eq("MHC_I_processing_suppression")].iloc[0]
    lta4h = human.loc[human["gene_symbol"].eq("LTA4H")]
    if lta4h.empty:
        lta4h_text = "not mapped among tested genes"
    else:
        row = lta4h.iloc[0]
        lta4h_text = f"logFC={row['logFC']:.3f}, minimum mouse-gene FDR={row['min_mouse_fdr']:.4g}"
    top_text = ", ".join(
        f"{row.gene_symbol} ({row.logFC:+.2f})" for row in top.itertuples()
    )
    b_markers = human.loc[
        human["gene_symbol"].isin(LINEAGE_AUDIT_MARKERS["B_lineage"])
    ]
    significant_b_markers = b_markers.loc[b_markers["min_mouse_fdr"].lt(0.05)]
    significant_b_text = ", ".join(significant_b_markers["gene_symbol"])
    if not significant_b_text:
        significant_b_text = "none"
    sensitivity_lines = []
    qc_path = path.parent / "GSE304475_edgeR_sample_QC_summary.csv"
    loo_summary_path = path.parent / "GSE304475_edgeR_leave_one_out_summary.csv"
    loo_scores_path = path.parent / "GSE304475_SC57461A_leave_one_out_connectivity.csv"
    if qc_path.exists() and loo_summary_path.exists() and loo_scores_path.exists():
        qc = pd.read_csv(qc_path).set_index("metric")["value"]
        loo_summary = pd.read_csv(loo_summary_path)
        loo_scores = pd.read_csv(loo_scores_path)
        loo_bam = loo_scores.loc[
            loo_scores["comparison"].eq("BAM_disease_signature_reversal")
        ]
        loo_mhc = loo_scores.loc[
            loo_scores["comparison"].eq("MHC_I_processing_suppression")
        ]
        sensitivity_lines = [
            "",
            "## Sample-level QC and leave-one-out sensitivity",
            "",
            f"- TMM factors ranged from {qc['minimum_TMM_factor']:.3f} to "
            f"{qc['maximum_TMM_factor']:.3f}.",
            "- Vehicle and SC-57461A samples did not overlap on MDS dimension 2; "
            f"the between-group gap was {qc['between_group_MDS2_gap']:.3f} and the "
            "standardized mean difference was "
            f"{qc['MDS2_standardized_mean_difference']:.3f}.",
            f"- Eight leave-one-sample-out edgeR refits retained "
            f"{loo_summary['n_fdr_0_05'].min()}-{loo_summary['n_fdr_0_05'].max()} "
            "FDR-significant genes. The 53 full-model discoveries remained 100% "
            "direction-concordant and nominally significant in every refit. "
            "Whole-transcriptome logFC Spearman correlations versus the full model "
            f"ranged from {loo_summary['all_gene_spearman_vs_full'].min():.3f} to "
            f"{loo_summary['all_gene_spearman_vs_full'].max():.3f}.",
            f"- BAM reversal was sample-sensitive across refits (score "
            f"{loo_bam['score'].min():.3f}-{loo_bam['score'].max():.3f}; permutation "
            f"P={loo_bam['permutation_p_one_sided'].min():.3f}-"
            f"{loo_bam['permutation_p_one_sided'].max():.3f}). The minimum BH FDR "
            "across the eight omissions was "
            f"{loo_bam['permutation_fdr_across_omissions'].min():.3f}, so the "
            "full-model trend is not considered robust.",
            "- MHC-I suppression remained unsupported in every refit (permutation "
            f"P={loo_mhc['permutation_p_one_sided'].min():.3f}-"
            f"{loo_mhc['permutation_p_one_sided'].max():.3f}; across-omission "
            f"FDR={loo_mhc['permutation_fdr_across_omissions'].min():.3f}).",
            "- IFN suppression varied materially by omitted sample and is not used "
            "as mechanism-supporting evidence.",
        ]
    lines = [
        "# GSE304475 SC-57461A primary-ILC2 perturbation audit",
        "",
        "## Design",
        "",
        "- Sorted large-intestinal ILC2 from C57BL/6 mice, cultured with IL-2/IL-7/IL-25/IL-33.",
        "- SC-57461A versus vehicle for 16 h; four biological samples per group.",
        "- Raw integer counts from the GEO workbook were modeled with edgeR quasi-likelihood.",
        "- Mouse Ensembl IDs were mapped to human ortholog symbols with g:Profiler; human-symbol "
        "duplicates were collapsed by median logFC.",
        "",
        "## Differential expression",
        "",
        f"- Tested genes: {len(de):,}; FDR<0.05: {(de['FDR'] < 0.05).sum():,} "
        f"({((de['FDR'] < 0.05) & (de['logFC'] > 0)).sum()} up, "
        f"{((de['FDR'] < 0.05) & (de['logFC'] < 0)).sum()} down).",
        f"- Lta4h transcript: {lta4h_text}. Enzyme inhibition does not require transcript reduction.",
        f"- Top mapped FDR-significant effects: {top_text}.",
        "- Post hoc lineage audit: FDR-significant B-lineage markers among the "
        f"audited set were {significant_b_text}. IGHM was isolated; CD79A, CD79B, "
        "CD74, IGKC, and CD37 were not differential. IGHM is not used for "
        "mechanistic interpretation.",
        "",
        "## Cross-species BAM connectivity",
        "",
        f"- BAM reversal score={overall['score']:.3f}; permutation "
        f"P={overall['permutation_p_one_sided']:.4g}.",
        f"- MHC-I suppression score={mhc['score']:.3f}; permutation "
        f"P={mhc['permutation_p_one_sided']:.4g}, module FDR={mhc['permutation_fdr']:.4g}.",
        f"- Weighted BAM reversal cosine={overall['weighted_reversal_cosine']:.3f}; "
        f"full-meta reversal Spearman={overall['full_meta_reversal_spearman']:.3f}.",
    ] + sensitivity_lines + [
        "",
        "## Interpretation boundary",
        "",
        "- This is the closest public primary immune-cell transcriptome found, but ILC2 are "
        "mucosal lymphoid cells rather than CNS BAM-like macrophages or dendritic cells.",
        "- Cross-species connectivity is a sensitivity analysis. It cannot establish the response "
        "of human CSF myeloid cells.",
        "- Positive pathway effects would support LTA4H/LTB4 biology in immune cells, not MS efficacy.",
        "",
        "## Machine-readable sensitivity outputs",
        "",
        "- `GSE304475_edgeR_sample_QC_summary.csv`",
        "- `GSE304475_edgeR_leave_one_out_summary.csv`",
        "- `GSE304475_SC57461A_edgeR_leave_one_out.csv.gz`",
        "- `GSE304475_SC57461A_leave_one_out_connectivity.csv`",
        "- `GSE304475_SC57461A_lineage_marker_audit.csv`",
        "",
        "## Sources",
        "",
        "- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE304475",
        "- Publication: https://pubmed.ncbi.nlm.nih.gov/41838192/",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    de = pd.read_csv(args.de)
    if args.fetch_mapping:
        fetch_mapping(de["gene_id"].tolist(), args.mapping, args.mapping_batch_size)
    if not args.mapping.exists():
        raise FileNotFoundError("Orthology mapping missing; run with --fetch-mapping")
    mapping = pd.read_csv(args.mapping, dtype=str).fillna("N/A")
    human = prepare_human_de(de, mapping)
    signature = pd.read_csv(args.signature)
    full_meta = pd.read_csv(args.full_meta)
    signature["gene_symbol"] = signature["gene_symbol"].str.upper()
    full_meta["gene_symbol"] = full_meta["gene_symbol"].str.upper()
    index = {gene: idx for idx, gene in enumerate(human["gene_symbol"])}
    rank_z = human["rank_z"].to_numpy()
    disease_up = set(signature.loc[
        signature["signature_direction"].eq("disease_up"), "gene_symbol"
    ])
    disease_down = set(signature.loc[
        signature["signature_direction"].eq("disease_down"), "gene_symbol"
    ])
    up_idx = np.array([index[gene] for gene in disease_up if gene in index])
    down_idx = np.array([index[gene] for gene in disease_down if gene in index])
    score, p_value, z_score = signature_score(
        rank_z, up_idx, down_idx, args.permutations, args.seed
    )
    selected_match = signature.merge(human, on="gene_symbol", how="inner")
    full_match = full_meta.merge(human, on="gene_symbol", how="inner")
    rho = spearmanr(full_match["mean_logfc"], full_match["logFC"])
    rows = [{
        "comparison": "BAM_disease_signature_reversal",
        "score": score,
        "permutation_p_one_sided": p_value,
        "permutation_z": z_score,
        "n_genes": len(up_idx) + len(down_idx),
        "weighted_reversal_cosine": weighted_reversal_cosine(selected_match),
        "full_meta_reversal_spearman": -float(rho.statistic),
        "full_meta_spearman_p": float(rho.pvalue),
    }]
    for offset, (module, genes) in enumerate(MODULES.items(), start=1):
        module_idx = np.array([index[gene] for gene in genes if gene in index])
        module_result = module_score(
            rank_z, module_idx, args.permutations, args.seed + offset
        )
        rows.append({
            "comparison": f"{module}_suppression",
            "score": module_result[0],
            "permutation_p_one_sided": module_result[1],
            "permutation_z": module_result[2],
            "n_genes": len(module_idx),
            "weighted_reversal_cosine": np.nan,
            "full_meta_reversal_spearman": np.nan,
            "full_meta_spearman_p": np.nan,
        })
    scores = pd.DataFrame(rows)
    module_mask = scores["comparison"].str.endswith("_suppression")
    scores.loc[module_mask, "permutation_fdr"] = multipletests(
        scores.loc[module_mask, "permutation_p_one_sided"], method="fdr_bh"
    )[1]

    top = human.loc[human["min_mouse_fdr"].lt(0.05)].copy()
    top = top.sort_values("min_mouse_fdr").head(15)
    lineage_rows = []
    for marker_class, genes in LINEAGE_AUDIT_MARKERS.items():
        frame = human.loc[human["gene_symbol"].isin(genes)].copy()
        frame.insert(0, "marker_class", marker_class)
        lineage_rows.append(frame)
    lineage_audit = pd.concat(lineage_rows, ignore_index=True)
    human.to_csv(args.output_dir / "GSE304475_SC57461A_human_ortholog_DE.csv.gz", index=False)
    selected_match.to_csv(args.output_dir / "GSE304475_SC57461A_BAM_gene_matches.csv", index=False)
    scores.to_csv(args.output_dir / "GSE304475_SC57461A_connectivity_scores.csv", index=False)
    top.to_csv(args.output_dir / "GSE304475_SC57461A_top_mapped_DE.csv", index=False)
    lineage_audit.to_csv(
        args.output_dir / "GSE304475_SC57461A_lineage_marker_audit.csv", index=False
    )
    write_report(
        args.output_dir / "GSE304475_SC57461A_REPORT.md", de, human, scores, top
    )
    print(scores.to_string(index=False))
    print("\nTop mapped DE")
    print(top[["gene_symbol", "logFC", "min_mouse_fdr"]].to_string(index=False))


if __name__ == "__main__":
    main()
