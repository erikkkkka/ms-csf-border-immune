#!/usr/bin/env python3
"""Score public LTA4H genetic perturbation signatures against the BAM program."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, hypergeom
from statsmodels.stats.multitest import multipletests


MHC_I = {
    "B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP",
    "PSMB8", "PSMB9", "PSMB10", "NLRC5",
}
BAM_HOMEOSTATIC = {
    "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1",
    "LYVE1", "FOLR2", "DAB2", "MSR1", "CD163", "HAMP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1000-gmt", type=Path, required=True)
    parser.add_argument("--replogle-gmt", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def load_gmt(path: Path, target: str = "LTA4H") -> tuple[dict[str, set[str]], set[str]]:
    target_terms: dict[str, set[str]] = {}
    universe: set[str] = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            term = fields[0]
            genes = {gene.upper() for gene in fields[2:] if gene}
            universe.update(genes)
            if term.upper().startswith(f"{target.upper()}::"):
                target_terms[term] = genes
    return target_terms, universe


def safe_hypergeom(k: int, universe_n: int, gene_set_n: int, signature_n: int) -> float:
    if k <= 0 or min(universe_n, gene_set_n, signature_n) <= 0:
        return 1.0
    return float(hypergeom.sf(k - 1, universe_n, gene_set_n, signature_n))


def paired_l1000_results(
    terms: dict[str, set[str]],
    universe: set[str],
    disease_up: set[str],
    disease_down: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs: dict[str, dict[str, set[str]]] = {}
    for term, genes in terms.items():
        if term.endswith(" up"):
            pairs.setdefault(term[:-3], {})["up"] = genes
        elif term.endswith(" down"):
            pairs.setdefault(term[:-5], {})["down"] = genes

    rows: list[dict] = []
    votes: dict[str, list[int]] = {}
    n_universe = len(universe)
    for perturbation, directions in sorted(pairs.items()):
        if set(directions) != {"up", "down"}:
            continue
        up = directions["up"]
        down = directions["down"]
        for gene in up:
            votes.setdefault(gene, []).append(1)
        for gene in down:
            votes.setdefault(gene, []).append(-1)

        reversed_up = disease_up & down
        reversed_down = disease_down & up
        concordant_up = disease_up & up
        concordant_down = disease_down & down
        p_up = safe_hypergeom(len(reversed_up), n_universe, len(down), len(disease_up))
        p_down = safe_hypergeom(len(reversed_down), n_universe, len(up), len(disease_down))
        combined_p = float(combine_pvalues([p_up, p_down], method="fisher").pvalue)
        reversal_fraction = 0.5 * (
            len(reversed_up) / max(len(disease_up), 1)
            + len(reversed_down) / max(len(disease_down), 1)
        )
        concordant_fraction = 0.5 * (
            len(concordant_up) / max(len(disease_up), 1)
            + len(concordant_down) / max(len(disease_down), 1)
        )
        rows.append({
            "perturbation_id": perturbation,
            "cell_line": perturbation.split("::")[1],
            "n_ko_up": len(up),
            "n_ko_down": len(down),
            "disease_up_in_ko_down_n": len(reversed_up),
            "disease_down_in_ko_up_n": len(reversed_down),
            "disease_up_in_ko_up_n": len(concordant_up),
            "disease_down_in_ko_down_n": len(concordant_down),
            "reversal_minus_concordance": reversal_fraction - concordant_fraction,
            "reversal_fisher_p": combined_p,
            "mhc_i_in_ko_down_n": len(MHC_I & down),
            "mhc_i_in_ko_up_n": len(MHC_I & up),
            "bam_homeostatic_in_ko_down_n": len(BAM_HOMEOSTATIC & down),
            "bam_homeostatic_in_ko_up_n": len(BAM_HOMEOSTATIC & up),
            "reversed_disease_up_genes": ";".join(sorted(reversed_up)),
            "reversed_disease_down_genes": ";".join(sorted(reversed_down)),
        })

    results = pd.DataFrame(rows)
    if not results.empty:
        results["reversal_fisher_fdr"] = multipletests(
            results["reversal_fisher_p"], method="fdr_bh"
        )[1]

    vote_rows = []
    n_pairs = len(results)
    for gene in sorted(universe):
        values = votes.get(gene, [])
        vote_rows.append({
            "gene_symbol": gene,
            "ko_up_votes": sum(value == 1 for value in values),
            "ko_down_votes": sum(value == -1 for value in values),
            "ko_vote_mean_all_signatures": sum(values) / max(n_pairs, 1),
            "n_signatures_with_gene_in_top_lists": len(values),
        })
    return results, pd.DataFrame(vote_rows)


def permutation_score(
    values: np.ndarray,
    first_idx: np.ndarray,
    second_idx: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = 0.5 * (values[first_idx].mean() - values[second_idx].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    total = len(first_idx) + len(second_idx)
    for index in range(permutations):
        sampled = rng.choice(len(values), size=total, replace=False)
        null[index] = 0.5 * (
            values[sampled[: len(first_idx)]].mean()
            - values[sampled[len(first_idx):]].mean()
        )
    p_value = (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
    z_score = (observed - null.mean()) / null.std(ddof=1)
    return float(observed), float(p_value), float(z_score)


def consensus_results(
    votes: pd.DataFrame,
    disease_up: set[str],
    disease_down: set[str],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    frame = votes.set_index("gene_symbol")
    values = frame["ko_vote_mean_all_signatures"].to_numpy(dtype=float)
    index = {gene: i for i, gene in enumerate(frame.index)}

    def indices(genes: set[str]) -> np.ndarray:
        return np.array([index[gene] for gene in genes if gene in index], dtype=int)

    rows = []
    comparisons = {
        "BAM_disease_signature_reversal": (disease_down, disease_up),
        "MHC_I_suppression": (set(), MHC_I),
        "BAM_homeostatic_suppression": (set(), BAM_HOMEOSTATIC),
    }
    for offset, (name, (positive, negative)) in enumerate(comparisons.items()):
        if positive:
            first = indices(positive)
            second = indices(negative)
            observed, p_value, z_score = permutation_score(
                values, first, second, permutations, seed + offset
            )
            n_genes = len(first) + len(second)
        else:
            module_idx = indices(negative)
            module_values = -values
            observed = float(module_values[module_idx].mean())
            rng = np.random.default_rng(seed + offset)
            null = np.array([
                module_values[rng.choice(len(values), size=len(module_idx), replace=False)].mean()
                for _ in range(permutations)
            ])
            p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
            z_score = float((observed - null.mean()) / null.std(ddof=1))
            n_genes = len(module_idx)
        rows.append({
            "comparison": name,
            "score": observed,
            "permutation_p_one_sided": p_value,
            "permutation_z": z_score,
            "n_genes": n_genes,
        })
    result = pd.DataFrame(rows)
    result["permutation_fdr"] = multipletests(
        result["permutation_p_one_sided"], method="fdr_bh"
    )[1]
    return result


def replogle_audit(
    terms: dict[str, set[str]], disease_up: set[str], disease_down: set[str]
) -> pd.DataFrame:
    rows = []
    for term, genes in terms.items():
        rows.append({
            "term": term,
            "direction_available": "up" if term.endswith(" up") else "down",
            "n_genes": len(genes),
            "overlap_disease_up_n": len(genes & disease_up),
            "overlap_disease_down_n": len(genes & disease_down),
            "overlap_mhc_i_n": len(genes & MHC_I),
            "overlap_bam_homeostatic_n": len(genes & BAM_HOMEOSTATIC),
            "interpretation": "Incomplete directional signature; do not calculate reversal.",
        })
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    paired: pd.DataFrame,
    consensus: pd.DataFrame,
    replogle: pd.DataFrame,
    universe_n: int,
) -> None:
    disease = consensus.loc[
        consensus["comparison"].eq("BAM_disease_signature_reversal")
    ].iloc[0]
    mhc = consensus.loc[consensus["comparison"].eq("MHC_I_suppression")].iloc[0]
    significant = int((paired["reversal_fisher_fdr"] < 0.05).sum())
    lines = [
        "# LTA4H genetic-perturbation connectivity audit",
        "",
        "## Scope",
        "",
        f"- LINCS L1000 XPR universe: {universe_n:,} genes represented across the library.",
        f"- Paired LTA4H CRISPR-Cas9 KO signatures: {len(paired)} (96 h; cancer cell lines).",
        f"- Per-signature BAM reversal tests passing BH FDR < 0.05: {significant}/{len(paired)}.",
        "- These are non-myeloid perturbations and are used only as orthogonal directional evidence.",
        "",
        "## Cross-signature consensus",
        "",
        f"- BAM disease-signature reversal score: {disease['score']:.4f}; permutation "
        f"P={disease['permutation_p_one_sided']:.4g}, FDR={disease['permutation_fdr']:.4g}.",
        f"- MHC-I suppression score: {mhc['score']:.4f}; permutation "
        f"P={mhc['permutation_p_one_sided']:.4g}, FDR={mhc['permutation_fdr']:.4g}.",
        "- Positive scores indicate the hypothesized direction; a null/negative result does not "
        "support LTA4H loss as a broad BAM/MHC-I reverser.",
        "",
        "## CRISPRi audit",
        "",
        f"- Replogle K562 LTA4H CRISPRi target terms: {len(replogle)}.",
        "- Only one direction is available, so no two-sided disease-reversal score is reported.",
        "",
        "## Decision rule",
        "",
        "This layer can strengthen a mechanism only if direction is reasonably consistent across "
        "cell lines. It cannot establish efficacy in CSF BAM-like macrophages.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature = pd.read_csv(args.signature)
    disease_up = set(signature.loc[
        signature["signature_direction"].eq("disease_up"), "gene_symbol"
    ].str.upper())
    disease_down = set(signature.loc[
        signature["signature_direction"].eq("disease_down"), "gene_symbol"
    ].str.upper())

    l1000_terms, l1000_universe = load_gmt(args.l1000_gmt)
    replogle_terms, _ = load_gmt(args.replogle_gmt)
    paired, votes = paired_l1000_results(
        l1000_terms, l1000_universe, disease_up, disease_down
    )
    consensus = consensus_results(
        votes, disease_up, disease_down, args.permutations, args.seed
    )
    replogle = replogle_audit(replogle_terms, disease_up, disease_down)

    paired.to_csv(args.output_dir / "LTA4H_L1000_XPR_signature_scores.csv", index=False)
    votes.to_csv(args.output_dir / "LTA4H_L1000_XPR_consensus_gene_votes.csv.gz", index=False)
    consensus.to_csv(args.output_dir / "LTA4H_L1000_XPR_consensus_scores.csv", index=False)
    replogle.to_csv(args.output_dir / "LTA4H_Replogle_CRISPRi_audit.csv", index=False)
    write_report(
        args.output_dir / "LTA4H_GENETIC_PERTURBATION_REPORT.md",
        paired,
        consensus,
        replogle,
        len(l1000_universe),
    )
    print(paired[[
        "cell_line", "reversal_minus_concordance", "reversal_fisher_p",
        "reversal_fisher_fdr", "mhc_i_in_ko_down_n", "mhc_i_in_ko_up_n",
    ]].to_string(index=False))
    print("\nConsensus")
    print(consensus.to_string(index=False))


if __name__ == "__main__":
    main()
