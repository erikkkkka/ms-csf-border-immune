#!/usr/bin/env python3
"""Score leave-one-sample-out GSE304475 effects against BAM modules."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--de", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--full-meta", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--scoring-script", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def load_scoring_module(path: Path):
    spec = importlib.util.spec_from_file_location("gse304475_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load scoring functions from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    scoring = load_scoring_module(args.scoring_script)
    de = pd.read_csv(args.de)
    mapping = pd.read_csv(args.mapping, dtype=str).fillna("N/A")
    signature = pd.read_csv(args.signature)
    full_meta = pd.read_csv(args.full_meta)
    metadata = pd.read_csv(args.metadata).set_index("sample_id")
    signature["gene_symbol"] = signature["gene_symbol"].str.upper()
    full_meta["gene_symbol"] = full_meta["gene_symbol"].str.upper()
    disease_up = set(signature.loc[
        signature["signature_direction"].eq("disease_up"), "gene_symbol"
    ])
    disease_down = set(signature.loc[
        signature["signature_direction"].eq("disease_down"), "gene_symbol"
    ])

    rows = []
    for profile_index, (omitted, frame) in enumerate(
        de.groupby("omitted_sample", sort=True)
    ):
        human = scoring.prepare_human_de(frame, mapping)
        index = {gene: idx for idx, gene in enumerate(human["gene_symbol"])}
        rank_z = human["rank_z"].to_numpy()
        up_idx = np.array([index[gene] for gene in disease_up if gene in index])
        down_idx = np.array([index[gene] for gene in disease_down if gene in index])
        score, p_value, z_score = scoring.signature_score(
            rank_z,
            up_idx,
            down_idx,
            args.permutations,
            args.seed + profile_index * 100,
        )
        full_match = full_meta.merge(human, on="gene_symbol", how="inner")
        rho = spearmanr(full_match["mean_logfc"], full_match["logFC"])
        rows.append({
            "omitted_sample": omitted,
            "omitted_group": metadata.loc[omitted, "group"],
            "comparison": "BAM_disease_signature_reversal",
            "score": score,
            "permutation_p_one_sided": p_value,
            "permutation_z": z_score,
            "n_genes": len(up_idx) + len(down_idx),
            "full_meta_reversal_spearman": -float(rho.statistic),
            "full_meta_spearman_p": float(rho.pvalue),
        })
        for module_index, (module_name, genes) in enumerate(scoring.MODULES.items(), start=1):
            module_idx = np.array([index[gene] for gene in genes if gene in index])
            module_score, module_p, module_z = scoring.module_score(
                rank_z,
                module_idx,
                args.permutations,
                args.seed + profile_index * 100 + module_index,
            )
            rows.append({
                "omitted_sample": omitted,
                "omitted_group": metadata.loc[omitted, "group"],
                "comparison": f"{module_name}_suppression",
                "score": module_score,
                "permutation_p_one_sided": module_p,
                "permutation_z": module_z,
                "n_genes": len(module_idx),
                "full_meta_reversal_spearman": np.nan,
                "full_meta_spearman_p": np.nan,
            })

    scores = pd.DataFrame(rows)
    scores["permutation_fdr_within_profile"] = np.nan
    module_mask = scores["comparison"].str.endswith("_suppression")
    for omitted, index_values in scores.loc[module_mask].groupby("omitted_sample").groups.items():
        scores.loc[index_values, "permutation_fdr_within_profile"] = multipletests(
            scores.loc[index_values, "permutation_p_one_sided"], method="fdr_bh"
        )[1]
    scores["permutation_fdr_across_omissions"] = np.nan
    for comparison, index_values in scores.groupby("comparison").groups.items():
        scores.loc[index_values, "permutation_fdr_across_omissions"] = multipletests(
            scores.loc[index_values, "permutation_p_one_sided"], method="fdr_bh"
        )[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
