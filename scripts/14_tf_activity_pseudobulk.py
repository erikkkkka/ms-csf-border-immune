#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import decoupler as dc
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


CELL_TYPES = ["BAM_like_candidate", "cDC2"]


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan)
    keep = np.flatnonzero(np.isfinite(values))
    order = keep[np.argsort(values[keep])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def infer_activities(root, gse, cell_type, net, min_cells):
    pb_dir = root / "results/pseudobulk"
    sample_meta = pd.read_csv(pb_dir / f"{gse}_CSF_pseudobulk_sample_metadata.csv")
    sample_meta = sample_meta.loc[
        (sample_meta["analysis_label"] == cell_type) & (sample_meta["n_cells"] >= min_cells)
    ].copy()
    table = pd.read_csv(pb_dir / f"{gse}_CSF_{cell_type}_counts.csv.gz")
    samples = [x for x in sample_meta["sample_accession"] if x in table.columns]
    gene_counts = table[["gene_symbol"] + samples].groupby("gene_symbol", observed=True).sum()
    counts = gene_counts.T.astype(float)
    library_sizes = counts.sum(axis=1)
    log_cpm = np.log2(counts.div(library_sizes, axis=0) * 1e6 + 1)
    estimates, _ = dc.run_ulm(log_cpm, net, min_n=5, verbose=False)
    estimates.index.name = "sample_accession"
    activities = estimates.reset_index().melt(id_vars="sample_accession", var_name="tf", value_name="activity")
    activities = activities.merge(
        sample_meta[["sample_accession", "subject_id", "diagnosis", "disease_group", "n_cells"]],
        on="sample_accession",
        how="left",
    )
    activities.insert(0, "cell_type", cell_type)
    activities.insert(0, "gse", gse)
    return activities


def contrast_tests(activities, gse, cell_type, contrast_name):
    frame = activities.loc[(activities["gse"] == gse) & (activities["cell_type"] == cell_type)].copy()
    if gse == "GSE138266":
        case = frame["disease_group"] == "MS"
        control = frame["disease_group"] == "IIH_control"
    else:
        control = frame["disease_group"] == "non_inflammatory_control"
        if contrast_name == "MS":
            case = frame["disease_group"] == "MS"
        elif contrast_name == "PMS":
            case = frame["diagnosis"].isin(["PPMS", "SPMS"])
        elif contrast_name == "PPMS":
            case = frame["diagnosis"] == "PPMS"
        else:
            raise ValueError(contrast_name)
    rows = []
    for tf, tf_frame in frame.groupby("tf", observed=True):
        case_values = tf_frame.loc[case.loc[tf_frame.index], "activity"].dropna().to_numpy()
        control_values = tf_frame.loc[control.loc[tf_frame.index], "activity"].dropna().to_numpy()
        if len(case_values) < 4 or len(control_values) < 4:
            continue
        test = mannwhitneyu(case_values, control_values, alternative="two-sided", method="auto")
        rows.append({
            "gse": gse,
            "cell_type": cell_type,
            "contrast": contrast_name,
            "tf": tf,
            "n_case": len(case_values),
            "n_control": len(control_values),
            "median_case": np.median(case_values),
            "median_control": np.median(control_values),
            "activity_difference": np.median(case_values) - np.median(control_values),
            "rank_biserial": 2 * test.statistic / (len(case_values) * len(control_values)) - 1,
            "p_value": test.pvalue,
        })
    result = pd.DataFrame(rows)
    if len(result):
        result["fdr_bh"] = bh_adjust(result["p_value"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    parser.add_argument("--min-cells", type=int, default=10)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    net = pd.read_csv(root / "data/reference/dorothea_human_ABC.csv.gz")
    activities = []
    for gse in ["GSE138266", "GSE286068"]:
        for cell_type in CELL_TYPES:
            activities.append(infer_activities(root, gse, cell_type, net, args.min_cells))
    activities = pd.concat(activities, ignore_index=True)
    out_dir = root / "results/regulon"
    out_dir.mkdir(parents=True, exist_ok=True)
    activities.to_csv(out_dir / "CSF_APC_pseudobulk_DoRothEA_ULM_activities.csv.gz", index=False, compression="gzip")

    tests = []
    for cell_type in CELL_TYPES:
        tests.append(contrast_tests(activities, "GSE138266", cell_type, "MS"))
        for contrast in ["MS", "PMS", "PPMS"]:
            tests.append(contrast_tests(activities, "GSE286068", cell_type, contrast))
    tests = pd.concat(tests, ignore_index=True)
    tests.sort_values(["fdr_bh", "p_value", "gse", "cell_type"]).to_csv(
        out_dir / "CSF_APC_pseudobulk_DoRothEA_ULM_group_tests.csv", index=False
    )


if __name__ == "__main__":
    main()
