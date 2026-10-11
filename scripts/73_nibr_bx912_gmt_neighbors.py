#!/usr/bin/env python3
"""Infer BX-912 mechanism neighbors from local directional NIBR gene sets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
QUERY = "CID11754511 BX-912 DE-48-HZ49 BAID2 0.1UM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def read_gmt(path: Path) -> dict[str, set[str]]:
    output = {}
    for line in path.read_text().splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) >= 3:
            output[fields[0]] = {gene.upper() for gene in fields[2:] if gene}
    return output


def jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else np.nan


def parse_name(name: str) -> tuple[str | None, float | None]:
    code = re.search(r"\b[A-Z]{2}-\d{2}-[A-Z]{2}\d{2}\b", name)
    dose = re.search(r"\b(\d+(?:\.\d+)?)UM\b", name)
    return (
        code.group(0) if code else None,
        float(dose.group(1)) if dose else None,
    )


def token_enrichment(
    compound: pd.DataFrame, column: str, separator: str, top_fraction: float = 0.05
) -> pd.DataFrame:
    table = compound[["drug_code", column]].copy()
    table[column] = table[column].fillna("")
    table["tokens"] = table[column].map(
        lambda value: {token.strip() for token in value.split(separator) if token.strip()}
    )
    n_top = max(50, int(np.ceil(len(table) * top_fraction)))
    top = table.iloc[:n_top]
    rows = []
    all_tokens = sorted(set().union(*table["tokens"]))
    for token in all_tokens:
        top_with = int(top["tokens"].map(lambda values: token in values).sum())
        all_with = int(table["tokens"].map(lambda values: token in values).sum())
        if all_with < 5:
            continue
        contingency = [
            [top_with, n_top - top_with],
            [all_with - top_with, len(table) - n_top - all_with + top_with],
        ]
        odds, p_value = stats.fisher_exact(contingency, alternative="greater")
        rows.append({
            "annotation_source": column,
            "term": token,
            "n_top_compounds": n_top,
            "n_top_with_term": top_with,
            "n_all_compounds": len(table),
            "n_all_with_term": all_with,
            "top_fraction_with_term": top_with / n_top,
            "background_fraction_with_term": all_with / len(table),
            "odds_ratio": odds,
            "fisher_p": p_value,
        })
    output = pd.DataFrame(rows)
    if len(output):
        output["fdr"] = multipletests(output["fisher_p"], method="fdr_bh")[1]
        output = output.sort_values(["fdr", "odds_ratio"], ascending=[True, False])
    return output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    reference = root / "data/reference/enrichr"
    up = read_gmt(reference / "NIBR_DRUGseq_2025_up.gmt")
    down = read_gmt(reference / "NIBR_DRUGseq_2025_down.gmt")
    if QUERY not in up or QUERY not in down:
        raise RuntimeError("BX-912 0.1 uM query signature is incomplete")
    query_up, query_down = up[QUERY], down[QUERY]
    rows = []
    for name in sorted(set(up) & set(down)):
        code, dose = parse_name(name)
        same_up = len(query_up & up[name])
        same_down = len(query_down & down[name])
        cross_up_down = len(query_up & down[name])
        cross_down_up = len(query_down & up[name])
        denominator = np.sqrt(
            (len(query_up) + len(query_down)) * (len(up[name]) + len(down[name]))
        )
        rows.append({
            "signature_name": name,
            "drug_code": code,
            "dose_um": dose,
            "n_query_up": len(query_up),
            "n_query_down": len(query_down),
            "n_profile_up": len(up[name]),
            "n_profile_down": len(down[name]),
            "same_direction_overlap": same_up + same_down,
            "opposite_direction_overlap": cross_up_down + cross_down_up,
            "directional_cosine": (
                (same_up + same_down - cross_up_down - cross_down_up) / denominator
            ),
            "same_direction_mean_jaccard": np.mean([
                jaccard(query_up, up[name]), jaccard(query_down, down[name])
            ]),
            "opposite_direction_mean_jaccard": np.mean([
                jaccard(query_up, down[name]), jaccard(query_down, up[name])
            ]),
        })
    result = pd.DataFrame(rows)
    metadata_path = root / "data/reference/nibr-drug-seq-drug_metadata.parquet"
    metadata = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(metadata_path)]
    ).fetchdf()
    result = result.merge(
        metadata, left_on="drug_code", right_on="nsn_code", how="left",
        validate="many_to_one",
    ).sort_values(
        ["directional_cosine", "same_direction_mean_jaccard"], ascending=False
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    output = root / "results/transition_aware_drug_screen/mechanism_deconvolution"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "BX912_NIBR_directional_signature_neighbors.csv", index=False)
    non_self = result.loc[result["drug_code"].ne("DE-48-HZ49")].copy()
    non_self.head(100).to_csv(output / "BX912_top100_NIBR_directional_neighbors.csv", index=False)
    pdpk1_pattern = "PDK1|PDPK1|PHOSPHOINOSITIDE DEPENDENT KINASE"
    pdpk1 = non_self.loc[
        non_self["nibr_drug_name"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
        | non_self["moabox_moa"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
        | non_self["drh_moa"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
        | non_self["drh_target"].fillna("").str.contains(pdpk1_pattern, case=False, regex=True)
    ].copy()
    pdpk1.to_csv(output / "BX912_PDPK1_directional_neighbors.csv", index=False)

    compound = non_self.loc[
        non_self.groupby("drug_code")["directional_cosine"].idxmax()
    ].sort_values("directional_cosine", ascending=False).reset_index(drop=True)
    compound["compound_rank"] = np.arange(1, len(compound) + 1)
    compound.to_csv(output / "BX912_compound_level_directional_neighbors.csv", index=False)
    target_enrichment = token_enrichment(compound, "drh_target", "|")
    moa_enrichment = token_enrichment(compound, "moabox_moa", ";")
    annotation_enrichment = pd.concat(
        [target_enrichment, moa_enrichment], ignore_index=True
    ).sort_values(["fdr", "odds_ratio"], ascending=[True, False])
    annotation_enrichment.to_csv(
        output / "BX912_top5pct_neighbor_annotation_enrichment.csv", index=False
    )

    columns = [
        "rank", "signature_name", "nibr_drug_name", "dose_um", "directional_cosine",
        "same_direction_overlap", "opposite_direction_overlap", "moabox_moa",
        "drh_moa", "drh_target",
    ]
    print(non_self.head(30)[columns].to_string(index=False), flush=True)
    print("\nPDPK1-annotated controls", flush=True)
    print(pdpk1[columns].sort_values("rank").to_string(index=False), flush=True)
    print("\nTop-5% compound-level annotation enrichment", flush=True)
    print(annotation_enrichment.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
