#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--max-genes", type=int, default=150)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "results" / "drug_repurposing"
    up = (out_dir / "BAM_MS_consensus_up.txt").read_text().splitlines()[: args.max_genes]
    down = (out_dir / "BAM_MS_consensus_down.txt").read_text().splitlines()[: args.max_genes]
    payload = {
        "data": {"upGenes": up, "dnGenes": down},
        "config": {
            "aggravate": False,
            "searchMethod": "geneSet",
            "share": True,
            "combination": False,
            "db-version": "latest",
        },
    }
    (out_dir / "L1000CDS2_query_payload.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    response = requests.post(
        "https://maayanlab.cloud/L1000CDS2/query",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    (out_dir / "L1000CDS2_reversers_raw.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    rows = []
    for rank, hit in enumerate(result.get("topMeta", []), start=1):
        rows.append({
            "rank": rank,
            "score": hit.get("score"),
            "pert_desc": hit.get("pert_desc"),
            "pert_id": hit.get("pert_id"),
            "pubchem_id": hit.get("pubchem_id"),
            "drugbank_id": hit.get("drugbank_id") or hit.get("drugchem_id"),
            "cell_id": hit.get("cell_id"),
            "pert_time": hit.get("pert_time"),
            "pert_time_unit": hit.get("pert_time_unit"),
            "pert_dose": hit.get("pert_dose"),
            "pert_dose_unit": hit.get("pert_dose_unit"),
            "sig_id": hit.get("sig_id"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "L1000CDS2_reversers_top50.csv", index=False)


if __name__ == "__main__":
    main()
