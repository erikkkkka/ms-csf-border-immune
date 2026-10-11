#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


COHORTS = {
    "GSE138266": {
        "file": "GSE138266_CSF_BAM_like_candidate_edgeR_MS_vs_control.csv.gz",
        "n_donors": 12,
    },
    "GSE286068": {
        "file": "GSE286068_CSF_BAM_like_candidate_edgeR_MS_vs_control.csv.gz",
        "n_donors": 14,
    },
}


def load_result(path: Path, cohort: str) -> pd.DataFrame:
    table = pd.read_csv(path).dropna(subset=["gene_symbol"]).copy()
    table["gene_symbol"] = table["gene_symbol"].astype(str).str.strip()
    table = table.loc[table["gene_symbol"].ne("")]
    table = table.sort_values(["PValue", "FDR"]).drop_duplicates("gene_symbol")
    keep = ["gene_symbol", "gene_id", "logFC", "logCPM", "PValue", "FDR"]
    return table[keep].rename(columns={col: f"{cohort}_{col}" for col in keep if col != "gene_symbol"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--n-per-direction", type=int, default=150)
    parser.add_argument("--min-abs-logfc", type=float, default=0.2)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pb_dir = root / "results" / "pseudobulk"
    out_dir = root / "results" / "drug_repurposing"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    for cohort, spec in COHORTS.items():
        tables.append(load_result(pb_dir / spec["file"], cohort))
    meta = tables[0].merge(tables[1], on="gene_symbol", how="inner")

    z_columns = []
    weights = []
    for cohort, spec in COHORTS.items():
        p = meta[f"{cohort}_PValue"].clip(lower=np.finfo(float).tiny)
        direction = np.sign(meta[f"{cohort}_logFC"])
        z = direction * norm.isf(p / 2)
        z_columns.append(z.to_numpy())
        weights.append(np.sqrt(spec["n_donors"]))

    z_matrix = np.vstack(z_columns)
    weights = np.asarray(weights)[:, None]
    meta["meta_z"] = (z_matrix * weights).sum(axis=0) / np.sqrt((weights**2).sum())
    meta["meta_p"] = 2 * norm.sf(np.abs(meta["meta_z"]))
    meta["meta_fdr"] = multipletests(meta["meta_p"], method="fdr_bh")[1]
    directions = np.vstack([
        np.sign(meta[f"{cohort}_logFC"].to_numpy()) for cohort in COHORTS
    ])
    meta["direction_concordant"] = np.all(directions == directions[0], axis=0)
    meta["min_abs_logfc"] = np.min(np.vstack([
        np.abs(meta[f"{cohort}_logFC"].to_numpy()) for cohort in COHORTS
    ]), axis=0)
    meta["mean_logfc"] = np.mean(np.vstack([
        meta[f"{cohort}_logFC"].to_numpy() for cohort in COHORTS
    ]), axis=0)

    eligible = meta.loc[
        meta["direction_concordant"]
        & (meta["min_abs_logfc"] >= args.min_abs_logfc)
    ].copy()
    up = eligible.loc[eligible["meta_z"] > 0].nlargest(args.n_per_direction, "meta_z")
    down = eligible.loc[eligible["meta_z"] < 0].nsmallest(args.n_per_direction, "meta_z")
    selected = pd.concat([
        up.assign(signature_direction="disease_up"),
        down.assign(signature_direction="disease_down"),
    ], ignore_index=True)

    meta.sort_values("meta_p").to_csv(
        out_dir / "BAM_MS_consensus_meta_analysis.csv.gz", index=False, compression="gzip"
    )
    selected.to_csv(out_dir / "BAM_MS_consensus_signature.csv", index=False)
    (out_dir / "BAM_MS_consensus_up.txt").write_text(
        "\n".join(up["gene_symbol"]) + "\n", encoding="utf-8"
    )
    (out_dir / "BAM_MS_consensus_down.txt").write_text(
        "\n".join(down["gene_symbol"]) + "\n", encoding="utf-8"
    )

    summary = pd.DataFrame([{
        "n_intersection_genes": len(meta),
        "n_direction_concordant": int(meta["direction_concordant"].sum()),
        "n_eligible_after_logfc_filter": len(eligible),
        "n_meta_p_0_05_concordant": int(((meta["meta_p"] < 0.05) & meta["direction_concordant"]).sum()),
        "n_meta_fdr_0_05_concordant": int(((meta["meta_fdr"] < 0.05) & meta["direction_concordant"]).sum()),
        "n_signature_up": len(up),
        "n_signature_down": len(down),
        "selection_rule": (
            f"direction concordant; min cohort abs(logFC)>={args.min_abs_logfc}; "
            f"top {args.n_per_direction} per direction by weighted Stouffer meta-z"
        ),
    }])
    summary.to_csv(out_dir / "BAM_MS_consensus_signature_summary.csv", index=False)


if __name__ == "__main__":
    main()
