#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUTS = {
    "GSE138266": [
        "GSE138266_scTenifold_pilot_module_enrichment_795genes_5nets.csv",
        "GSE138266_scTenifold_pilot_module_enrichment_796genes_5nets_2cores.csv",
    ],
    "GSE286068": [
        "GSE286068_scTenifold_pilot_module_enrichment_788genes_5nets_1cores.csv",
        "GSE286068_scTenifold_pilot_module_enrichment_789genes_5nets_1cores.csv",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "results" / "virtual_knockout"
    frames = []
    for cohort, names in INPUTS.items():
        for name in names:
            frame = pd.read_csv(out_dir / name)
            frame.insert(0, "cohort", cohort)
            frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    if long.duplicated(["cohort", "target", "gene_set"]).any():
        raise RuntimeError("Duplicate cohort-target-gene_set rows found")

    wide = long.pivot(
        index=["cohort", "target"], columns="gene_set",
        values=["rank_enrichment_auc", "rank_enrichment_p", "n_genes_fdr_0_05"],
    )
    wide.columns = [f"{metric}__{gene_set}" for metric, gene_set in wide.columns]
    wide = wide.reset_index()
    wide["MHC_minus_homeostatic_AUC"] = (
        wide["rank_enrichment_auc__MHC_I_processing"]
        - wide["rank_enrichment_auc__BAM_homeostatic"]
    )
    wide["MHC_enrichment_p_lt_0_05"] = (
        wide["rank_enrichment_p__MHC_I_processing"] < 0.05
    )
    wide.to_csv(out_dir / "BAM_target_prioritization_by_cohort.csv", index=False)

    rows = []
    for target, frame in wide.groupby("target", sort=False):
        n_cohorts = frame["cohort"].nunique()
        delta = frame["MHC_minus_homeostatic_AUC"]
        mhc_auc = frame["rank_enrichment_auc__MHC_I_processing"]
        if n_cohorts == 2 and (delta > 0).all() and frame["MHC_enrichment_p_lt_0_05"].all():
            tier = "Tier_1_cross_cohort_MHC_biased"
        elif n_cohorts == 2 and frame["MHC_enrichment_p_lt_0_05"].all():
            tier = "Tier_2_cross_cohort_network_hub_not_selective"
        else:
            tier = "Tier_3_single_cohort_only"
        rows.append({
            "target": target,
            "n_cohorts": n_cohorts,
            "cohorts": ";".join(sorted(frame["cohort"].unique())),
            "mean_MHC_AUC": mhc_auc.mean(),
            "minimum_MHC_AUC": mhc_auc.min(),
            "mean_MHC_minus_homeostatic_AUC": delta.mean(),
            "minimum_MHC_minus_homeostatic_AUC": delta.min(),
            "n_cohorts_MHC_p_lt_0_05": int(frame["MHC_enrichment_p_lt_0_05"].sum()),
            "n_cohorts_positive_MHC_bias": int((delta > 0).sum()),
            "evidence_tier": tier,
            "direction_caveat": "scTenifoldKnk_distance_is_unsigned",
        })
    summary = pd.DataFrame(rows).sort_values(
        ["n_cohorts", "n_cohorts_positive_MHC_bias", "minimum_MHC_AUC"],
        ascending=False,
    )
    summary.insert(0, "priority_rank", np.arange(1, len(summary) + 1))
    summary.to_csv(out_dir / "BAM_target_prioritization_summary.csv", index=False)


if __name__ == "__main__":
    main()
