#!/usr/bin/env python3
"""Find unbiased full-transcriptome neighbors of the BX-912 active profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DE_URL = "https://s3.perturbseqr.maayanlab.cloud/data/nibr-drug-seq-de.parquet"
QUERY_PROFILE = "DE-48-HZ49_0.1_24.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--httpfs-extension", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/transition_aware_drug_screen/mechanism_deconvolution"
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "data/reference/nibr-drug-seq-drug_metadata.parquet"
    targeted_path = (
        root / "data/external_validation/NIBR_DRUGseq_transition_candidates/targeted_full_de.csv.gz"
    )

    connection = duckdb.connect()
    connection.execute("PRAGMA disable_progress_bar")
    if args.httpfs_extension:
        connection.execute(f"LOAD '{args.httpfs_extension.resolve()}'")
    else:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
    print("Scanning all NIBR profiles against BX-912 0.1 uM", flush=True)
    neighbors = connection.execute(f"""
        WITH query AS (
            SELECT gene_symbol, avg(logFC) AS query_logfc
            FROM read_csv_auto('{targeted_path.resolve()}')
            WHERE pert_id = '{QUERY_PROFILE}'
            GROUP BY gene_symbol
        )
        SELECT
            profile.pert_id,
            count(*) AS n_shared_genes,
            corr(profile.logFC, query.query_logfc) AS pearson_r,
            sum(profile.logFC * query.query_logfc)
                / nullif(sqrt(sum(profile.logFC * profile.logFC))
                    * sqrt(sum(query.query_logfc * query.query_logfc)), 0) AS cosine_similarity,
            avg(CASE
                WHEN sign(profile.logFC) = sign(query.query_logfc) THEN 1.0
                ELSE 0.0
            END) AS sign_concordance
        FROM read_parquet('{DE_URL}') AS profile
        INNER JOIN query ON profile.genes = query.gene_symbol
        GROUP BY profile.pert_id
        HAVING count(*) >= 10000
        ORDER BY pearson_r DESC
    """).fetchdf()
    metadata = connection.execute(f"""
        SELECT * FROM read_parquet('{metadata_path.resolve()}')
    """).fetchdf()
    connection.close()

    parsed = neighbors["pert_id"].str.rsplit("_", n=2, expand=True)
    neighbors["drug_code"] = parsed[0]
    neighbors["dose_um"] = pd.to_numeric(parsed[1], errors="coerce")
    neighbors["time_h"] = pd.to_numeric(parsed[2], errors="coerce")
    neighbors = neighbors.merge(
        metadata,
        left_on="drug_code",
        right_on="nsn_code",
        how="left",
        validate="many_to_one",
    )
    non_query = neighbors.loc[neighbors["pert_id"].ne(QUERY_PROFILE)].copy()
    non_query["pearson_p"] = [
        2 * stats.t.sf(
            abs(row.pearson_r) * np.sqrt((row.n_shared_genes - 2) / (1 - row.pearson_r ** 2)),
            df=row.n_shared_genes - 2,
        ) if np.isfinite(row.pearson_r) and abs(row.pearson_r) < 1 else 0.0
        for row in non_query.itertuples(index=False)
    ]
    neighbors.to_csv(output / "BX912_all_NIBR_transcriptomic_neighbors.csv.gz", index=False, compression="gzip")
    non_query.head(100).to_csv(output / "BX912_top100_NIBR_transcriptomic_neighbors.csv", index=False)

    pdpk1_pattern = "PDK1|PDPK1|PHOSPHOINOSITIDE DEPENDENT KINASE"
    mechanism_controls = non_query.loc[
        non_query["nibr_drug_name"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
        | non_query["moabox_moa"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
        | non_query["drh_moa"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
        | non_query["drh_target"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
    ].copy()
    mechanism_controls.to_csv(output / "BX912_PDPK1_annotated_neighbors.csv", index=False)

    top = non_query.head(25)
    lines = [
        "# BX-912 full-transcriptome mechanism-neighbor audit",
        "",
        f"Query: `{QUERY_PROFILE}`. All comparisons use at least 10,000 shared genes in the same NIBR U2OS 24-hour system.",
        "",
        "## Top non-self profiles",
        "",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.nibr_drug_name} {row.dose_um:g} uM: r={row.pearson_r:.3f}, "
            f"cosine={row.cosine_similarity:.3f}, sign concordance={row.sign_concordance:.3f}; "
            f"targets={row.drh_target or 'not annotated'}."
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Transcriptomic proximity is used to generate mechanism hypotheses. It does not establish direct binding, target engagement, or disease-relevant efficacy.",
        "",
    ])
    (output / "BX912_TRANSCRIPTOMIC_NEIGHBOR_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(top[[
        "pert_id", "nibr_drug_name", "dose_um", "pearson_r", "cosine_similarity",
        "sign_concordance", "moabox_moa", "drh_moa", "drh_target",
    ]].to_string(index=False), flush=True)
    print("\nPDPK1-annotated controls", flush=True)
    print(mechanism_controls[[
        "pert_id", "nibr_drug_name", "dose_um", "pearson_r", "cosine_similarity",
        "sign_concordance", "drh_target",
    ]].sort_values("pearson_r", ascending=False).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
