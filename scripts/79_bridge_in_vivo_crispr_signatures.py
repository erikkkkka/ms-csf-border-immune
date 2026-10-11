#!/usr/bin/env python3
"""Bridge published in vivo receptor-KO signatures to frozen human MS CSF states."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import patsy
import scanpy as sc
from scipy import sparse, stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


SEED = 20260902
MIN_DONOR_CELLS = 20
MIN_STATE_CELLS = 5
INCOMPLETE_CUTOFF = 0.80
SOURCE_DOI = "10.1038/s41593-025-02151-6"

PERTURBATIONS = ["Ifngr1", "Tnfrsf1a", "Csf2ra", "Tgfbr1"]

DATASETS = {
    "GSE138266": {
        "path": "results/trajectory/continuum_objects/GSE138266_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
        "control": "IIH_control",
        "disease_groups": {"MS"},
        "external": False,
    },
    "GSE194078": {
        "path": "results/trajectory/continuum_objects/GSE194078_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": None,
        "threshold": None,
        "control": None,
        "disease_groups": set(),
        "external": False,
    },
    "GSE286068": {
        "path": "results/trajectory/continuum_objects/GSE286068_monocyte_BAM_continuum.h5ad",
        "pt": "consensus_pseudotime",
        "activation": "two_axis_activation_residual",
        "threshold": "two_axis_control_activation_q75",
        "control": "non_inflammatory_control",
        "disease_groups": {"MS"},
        "external": False,
    },
    "GSE328265": {
        "path": "results/external_validation/GSE328265_two_axis_projection/GSE328265_monocyte_BAM_two_axis.h5ad",
        "pt": "projected_differentiation",
        "activation": "external_activation_residual",
        "threshold": "external_control_activation_q75",
        "control": "control",
        "disease_groups": {"RRMS", "PPMS", "SPMS"},
        "external": True,
    },
}

AXIS2_GENES = {
    "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5",
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "OAS1", "STAT1", "IRF7",
}

LAM2_GENES = {
    "APOE", "C1QA", "C1QB", "C1QC", "GPNMB", "SELENOP", "TREM2", "MERTK",
    "HLA-DRA", "CD74", "CIITA",
}

PRECURSOR_GENES = {"CXCL10", "CXCL9", "TGM2", "TGFB3", "PLAC8", "IRF1", "CSF2RB"}

ANNOTATION_MARKERS = {
    "S100A8", "S100A9", "FCN1", "VCAN", "CCR2", "SELL", "CTSD", "FCGR3A",
    "MS4A7", "LILRB1", "IFITM3", "LST1", "LYZ", "CLEC9A", "XCR1", "BATF3",
    "CADM1", "WDFY4", "CD1C", "FCER1A", "CLEC10A", "CD1E", "CST3", "CD14",
    "CD163", "CTSS", "GZMB", "JCHAIN", "LILRA4", "CLEC4C", "TCF4", "IL3RA",
    "C1QA", "C1QB", "C1QC", "APOE", "TREM2", "LYVE1", "MRC1", "STAB1", "CH25H",
}

ENDPOINTS = [
    "decoupled_minus_other_incomplete",
    "axis2_spearman",
    "pseudotime_spearman",
    "lam2_residual_spearman",
    "precursor_residual_spearman",
]

SOURCE_URLS = {
    "Supplementary_Table_2_CRISPR_screen.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM4_ESM.xlsx",
    "Supplementary_Table_3_bulk_RNAseq.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM5_ESM.xlsx",
    "Supplementary_Table_4_scRNA_signatures.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM6_ESM.xlsx",
    "Supplementary_Code_1.Rmd": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM8_ESM.rmd",
    "Source_Data_Fig4.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM11_ESM.xlsx",
    "Source_Data_Fig5.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM12_ESM.xlsx",
    "Source_Data_Fig6.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM13_ESM.xlsx",
    "Source_Data_Fig8.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM14_ESM.xlsx",
    "Source_Data_Ext6.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM19_ESM.xlsx",
    "Source_Data_Ext7.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM20_ESM.xlsx",
    "Source_Data_Ext8.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM21_ESM.xlsx",
    "Source_Data_Ext9.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM22_ESM.xlsx",
    "Source_Data_Ext10.xlsx": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-025-02151-6/MediaObjects/41593_2025_2151_MOESM23_ESM.xlsx",
    "HOM_MouseHumanSequence.rpt": "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def weighted_mean_sd(matrix: np.ndarray, subjects: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = subjects.value_counts()
    weights = subjects.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    weights /= weights.sum()
    mean = np.sum(matrix * weights[:, None], axis=0)
    variance = np.sum(np.square(matrix - mean) * weights[:, None], axis=0)
    sd = np.sqrt(variance)
    sd[sd < 1e-8] = 1.0
    return mean, sd


def weighted_median(values: np.ndarray, subjects: pd.Series) -> float:
    counts = subjects.value_counts()
    weights = subjects.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(0.5, cumulative, values[order]))


def sign_flip_test(values: np.ndarray, seed: int) -> tuple[float, str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = abs(values.mean())
    total = 2 ** len(values)
    if total <= 100000:
        means = [np.mean(values * np.asarray(signs)) for signs in itertools.product([-1.0, 1.0], repeat=len(values))]
        return float(np.mean(np.abs(means) >= observed - 1e-12)), "exact_sign_flip"
    rng = np.random.default_rng(seed)
    random_signs = rng.choice([-1.0, 1.0], size=(100000, len(values)))
    means = np.mean(random_signs * values[None, :], axis=1)
    return float((np.sum(np.abs(means) >= observed - 1e-12) + 1) / 100001), "mc_sign_flip_100000"


def build_source_manifest(source_dir: Path) -> pd.DataFrame:
    rows = []
    for name, url in SOURCE_URLS.items():
        path = source_dir / name
        if not path.exists():
            continue
        rows.append({
            "file": name,
            "source_url": url,
            "source_doi": SOURCE_DOI,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return pd.DataFrame(rows)


def one_to_one_orthologs(path: Path) -> pd.DataFrame:
    homologs = pd.read_csv(path, sep="\t", dtype=str)
    rows = []
    for key, frame in homologs.groupby("DB Class Key", observed=True):
        mouse = frame.loc[frame["NCBI Taxon ID"].eq("10090"), "Symbol"].dropna().unique()
        human = frame.loc[frame["NCBI Taxon ID"].eq("9606"), "Symbol"].dropna().unique()
        if len(mouse) == 1 and len(human) == 1:
            rows.append({"homology_class": key, "mouse_gene": mouse[0], "human_gene": human[0]})
    return pd.DataFrame(rows).drop_duplicates(["mouse_gene", "human_gene"])


def read_external_signatures(
    source_dir: Path, orthologs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ortholog_map = orthologs.drop_duplicates("mouse_gene").set_index("mouse_gene")["human_gene"].to_dict()

    published_wide = pd.read_excel(
        source_dir / "Supplementary_Table_4_scRNA_signatures.xlsx",
        sheet_name="Neuroinflam.CytokineSignatures",
    )
    published_rows = []
    for perturbation in PERTURBATIONS:
        for order, gene in enumerate(published_wide[perturbation].dropna().astype(str)):
            published_rows.append({
                "signature_type": "published_response",
                "perturbation": perturbation,
                "mouse_gene": gene,
                "human_gene": ortholog_map.get(gene, ""),
                "source_order": order + 1,
                "weight": 1.0,
                "interpretation": "high_score_is_high_cytokine_response",
            })
    published = pd.DataFrame(published_rows)

    direct_sheet = pd.read_excel(source_dir / "Source_Data_Ext8.xlsx", sheet_name="f", header=None)
    direct_frames = []
    invalid_gene_rows = {}
    for start, perturbation in zip([0, 3, 6, 9], PERTURBATIONS):
        frame = direct_sheet.iloc[2:, start : start + 3].copy()
        frame.columns = ["mouse_gene", "hoxb8_log2fc", "chimera_log2fc"]
        invalid_gene = frame["mouse_gene"].map(lambda value: isinstance(value, (date, datetime)))
        invalid_gene_rows[perturbation] = int(invalid_gene.sum())
        frame = frame.loc[~invalid_gene].copy()
        frame["perturbation"] = perturbation
        frame["mouse_gene"] = frame["mouse_gene"].astype(str)
        frame["hoxb8_log2fc"] = pd.to_numeric(frame["hoxb8_log2fc"], errors="coerce")
        frame["chimera_log2fc"] = pd.to_numeric(frame["chimera_log2fc"], errors="coerce")
        frame = frame.loc[
            frame["mouse_gene"].ne("nan")
            & frame["hoxb8_log2fc"].notna()
            & frame["chimera_log2fc"].notna()
        ].copy()
        frame = frame.groupby(["perturbation", "mouse_gene"], as_index=False).agg(
            hoxb8_log2fc=("hoxb8_log2fc", "mean"),
            chimera_log2fc=("chimera_log2fc", "mean"),
            n_source_rows=("mouse_gene", "size"),
        )
        frame["same_direction"] = np.sign(frame["hoxb8_log2fc"]) == np.sign(frame["chimera_log2fc"])
        frame["consensus_weight"] = np.sign(frame["hoxb8_log2fc"]) * np.minimum(
            frame["hoxb8_log2fc"].abs(), frame["chimera_log2fc"].abs()
        )
        frame["human_gene"] = frame["mouse_gene"].map(ortholog_map).fillna("")
        direct_frames.append(frame)
    direct_all = pd.concat(direct_frames, ignore_index=True)

    balanced_rows = []
    for perturbation, frame in direct_all.groupby("perturbation", observed=True):
        eligible = frame.loc[frame["same_direction"] & frame["human_gene"].ne("")].copy()
        eligible = eligible.sort_values("consensus_weight", key=lambda x: x.abs(), ascending=False)
        negative = eligible.loc[eligible["consensus_weight"].lt(0)].head(100)
        positive = eligible.loc[eligible["consensus_weight"].gt(0)].head(100)
        selected = pd.concat([negative, positive], ignore_index=True)
        selected["signature_type"] = "direct_ko_consensus"
        selected["weight"] = selected["consensus_weight"]
        selected["interpretation"] = "high_score_is_KO_like"
        direction = selected["consensus_weight"].gt(0).map({True: "positive", False: "negative"})
        selected["source_rank_within_direction"] = selected["consensus_weight"].abs().groupby(
            direction, observed=True
        ).rank(method="first", ascending=False).astype(int)
        balanced_rows.append(selected)
    balanced = pd.concat(balanced_rows, ignore_index=True)

    direct_lookup = direct_all.set_index(["perturbation", "mouse_gene"])
    published["hoxb8_log2fc"] = [
        direct_lookup.loc[(row.perturbation, row.mouse_gene), "hoxb8_log2fc"]
        if (row.perturbation, row.mouse_gene) in direct_lookup.index else np.nan
        for row in published.itertuples()
    ]
    published["chimera_log2fc"] = [
        direct_lookup.loc[(row.perturbation, row.mouse_gene), "chimera_log2fc"]
        if (row.perturbation, row.mouse_gene) in direct_lookup.index else np.nan
        for row in published.itertuples()
    ]
    published["negative_in_both_KO_models"] = (
        published["hoxb8_log2fc"].lt(0) & published["chimera_log2fc"].lt(0)
    )

    qc_rows = []
    for perturbation, frame in direct_all.groupby("perturbation", observed=True):
        mapped = frame.loc[frame["human_gene"].ne("")]
        pub = published.loc[published["perturbation"].eq(perturbation)]
        selected = balanced.loc[balanced["perturbation"].eq(perturbation)]
        qc_rows.append({
            "perturbation": perturbation,
            "n_direct_genes": len(frame),
            "n_invalid_excel_date_gene_rows": invalid_gene_rows[perturbation],
            "n_collapsed_duplicate_gene_rows": int(frame["n_source_rows"].sub(1).sum()),
            "n_direct_one_to_one_mapped": len(mapped),
            "pearson_hoxb8_chimera": stats.pearsonr(frame["hoxb8_log2fc"], frame["chimera_log2fc"]).statistic,
            "spearman_hoxb8_chimera": stats.spearmanr(frame["hoxb8_log2fc"], frame["chimera_log2fc"]).statistic,
            "same_direction_fraction": frame["same_direction"].mean(),
            "n_published_response_genes": len(pub),
            "n_published_one_to_one_mapped": pub["human_gene"].ne("").sum(),
            "published_fraction_negative_in_both_KO_models": pub["negative_in_both_KO_models"].mean(),
            "n_balanced_direct_negative": selected["weight"].lt(0).sum(),
            "n_balanced_direct_positive": selected["weight"].gt(0).sum(),
        })
    return published, direct_all, balanced, pd.DataFrame(qc_rows)


def endpoint_holdouts(root: Path) -> dict[str, set[str]]:
    trajectory = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/frozen_trajectory_program.csv"
    )
    trajectory_genes = set(trajectory["gene"].astype(str)) | ANNOTATION_MARKERS
    return {
        "axis2": AXIS2_GENES,
        "pseudotime": trajectory_genes,
        "lam2": LAM2_GENES,
        "precursor": PRECURSOR_GENES,
    }


def residualize_score(
    score: np.ndarray,
    obs: pd.DataFrame,
    pt_column: str,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    frame = pd.DataFrame({
        "pt": obs[pt_column].to_numpy(float),
        "log_counts": np.log1p(obs["total_counts"].to_numpy(float)),
        "n_genes": obs["n_genes_by_counts"].to_numpy(float),
    }, index=obs.index)
    reference_subjects = obs.loc[reference_mask, "subject_id"].astype(str)
    for column in ["log_counts", "n_genes"]:
        values = frame.loc[reference_mask, column].to_numpy(float)[:, None]
        mean, sd = weighted_mean_sd(values, reference_subjects)
        frame[f"{column}_z"] = (frame[column] - mean[0]) / sd[0]
    formula = (
        "bs(pt, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound=1) "
        "+ log_counts_z + n_genes_z"
    )
    design_ref = patsy.dmatrix(formula, frame.loc[reference_mask], return_type="dataframe")
    counts = reference_subjects.value_counts()
    weights = reference_subjects.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    model = sm.WLS(score[reference_mask], design_ref, weights=weights).fit()
    design_all = patsy.build_design_matrices([design_ref.design_info], frame, return_type="dataframe")[0]
    residual = score - np.asarray(model.predict(design_all), dtype=float)
    center = weighted_median(residual[reference_mask], reference_subjects)
    _, scale = weighted_mean_sd((residual[reference_mask] - center)[:, None], reference_subjects)
    return (residual - center) / scale[0], float(model.rsquared)


def score_dataset(
    root: Path,
    dataset: str,
    config: dict,
    signatures: dict[tuple[str, str], pd.DataFrame],
    holdouts: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adata = sc.read_h5ad(root / config["path"])
    if not config["external"]:
        adata = adata[adata.obs["tissue"].astype(str).str.upper().eq("CSF")].copy()

    lam_path = root / "results/cross_species_lam" / f"{dataset}_LAM_cell_scores.csv.gz"
    lam = pd.read_csv(lam_path, index_col=0).reindex(adata.obs_names)
    required_lam = [
        "LAM_residual_mouse_LAM2_complement_APC",
        "LAM_residual_mouse_GMCSF_LAM_precursor",
    ]
    if lam[required_lam].isna().any().any():
        raise RuntimeError(f"{dataset}: missing frozen LAM residuals after cell-index merge")
    for column in required_lam:
        adata.obs[column] = lam[column].to_numpy(float)

    control_group = config["control"]
    reference = (
        adata.obs["disease_group"].astype(str).eq(control_group).to_numpy()
        if control_group is not None else np.ones(adata.n_obs, dtype=bool)
    )
    if reference.sum() < 50:
        raise RuntimeError(f"{dataset}: fewer than 50 reference cells")

    all_requested = sorted({
        gene for frame in signatures.values() for gene in frame["human_gene"] if gene
    })
    available = [gene for gene in all_requested if gene in adata.var_names]
    expression = dense(adata[:, available].X).astype(np.float64)
    mean, sd = weighted_mean_sd(expression[reference], adata.obs.loc[reference, "subject_id"].astype(str))
    standardized = (expression - mean) / sd
    gene_index = {gene: index for index, gene in enumerate(available)}

    availability_rows = []
    diagnostic_rows = []
    score_columns = {}
    for (signature_type, perturbation), signature in signatures.items():
        for endpoint_key, holdout in holdouts.items():
            usable = signature.loc[
                signature["human_gene"].isin(gene_index)
                & ~signature["human_gene"].isin(holdout)
            ].drop_duplicates("human_gene").copy()
            positive = int(usable["weight"].gt(0).sum())
            negative = int(usable["weight"].lt(0).sum())
            if signature_type == "published_response":
                powered = len(usable) >= 20
            else:
                powered = positive >= 20 and negative >= 20
            availability_rows.append({
                "dataset": dataset,
                "signature_type": signature_type,
                "perturbation": perturbation,
                "endpoint_holdout": endpoint_key,
                "n_source_genes": signature["human_gene"].ne("").sum(),
                "n_available_after_holdout": len(usable),
                "n_positive": positive,
                "n_negative": negative,
                "powered": powered,
                "available_genes": ";".join(usable["human_gene"]),
            })
            if not powered:
                continue
            indices = [gene_index[gene] for gene in usable["human_gene"]]
            weights = usable["weight"].to_numpy(float)
            raw_score = standardized[:, indices] @ weights / np.sum(np.abs(weights))
            base = f"{signature_type}__{perturbation}__{endpoint_key}"
            score_columns[f"score__{base}"] = raw_score
            if endpoint_key != "pseudotime":
                residual, r_squared = residualize_score(
                    raw_score, adata.obs, config["pt"], reference
                )
                score_columns[f"residual__{base}"] = residual
                diagnostic_rows.append({
                    "dataset": dataset,
                    "signature_type": signature_type,
                    "perturbation": perturbation,
                    "endpoint_holdout": endpoint_key,
                    "reference_kind": "control_cells" if control_group is not None else "all_cells",
                    "n_reference_cells": int(reference.sum()),
                    "n_reference_donors": int(adata.obs.loc[reference, "subject_id"].nunique()),
                    "baseline_r_squared": r_squared,
                    "residual_spearman_log_counts": stats.spearmanr(
                        residual, np.log1p(adata.obs["total_counts"].to_numpy(float))
                    ).statistic,
                    "residual_spearman_n_genes": stats.spearmanr(
                        residual, adata.obs["n_genes_by_counts"].to_numpy(float)
                    ).statistic,
                })

    cell_scores = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "dataset": dataset,
        "subject_id": adata.obs["subject_id"].astype(str).to_numpy(),
        "diagnosis": adata.obs["diagnosis"].astype(str).to_numpy(),
        "disease_group": adata.obs["disease_group"].astype(str).to_numpy(),
        "pseudotime": adata.obs[config["pt"]].to_numpy(float),
        "activation_residual": (
            adata.obs[config["activation"]].to_numpy(float)
            if config["activation"] is not None else np.full(adata.n_obs, np.nan)
        ),
        "lam2_residual": adata.obs["LAM_residual_mouse_LAM2_complement_APC"].to_numpy(float),
        "precursor_residual": adata.obs["LAM_residual_mouse_GMCSF_LAM_precursor"].to_numpy(float),
        "total_counts": adata.obs["total_counts"].to_numpy(float),
        "n_genes_by_counts": adata.obs["n_genes_by_counts"].to_numpy(float),
    })
    for column, values in score_columns.items():
        cell_scores[column] = values

    if config["activation"] is not None:
        threshold = float(adata.uns[config["threshold"]])
        cell_scores["incomplete"] = cell_scores["pseudotime"].lt(INCOMPLETE_CUTOFF)
        cell_scores["decoupled"] = (
            cell_scores["incomplete"] & cell_scores["activation_residual"].gt(threshold)
        )
        if config["external"] and "external_decoupled_inflammatory" in adata.obs:
            original = adata.obs["external_decoupled_inflammatory"].astype(bool).to_numpy()
            if not np.array_equal(original, cell_scores["decoupled"].to_numpy()):
                raise RuntimeError(f"{dataset}: frozen decoupled labels do not match reconstructed labels")
    else:
        cell_scores["incomplete"] = cell_scores["pseudotime"].lt(INCOMPLETE_CUTOFF)
        cell_scores["decoupled"] = False
    cell_scores["disease_eligible"] = cell_scores["disease_group"].isin(config["disease_groups"])
    del adata
    return cell_scores, pd.DataFrame(availability_rows), pd.DataFrame(diagnostic_rows)


def donor_effects(cell_scores: pd.DataFrame, signatures: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    dataset = str(cell_scores["dataset"].iloc[0])
    has_activation = cell_scores["activation_residual"].notna().any()
    for subject, frame in cell_scores.groupby("subject_id", observed=True):
        if len(frame) < MIN_DONOR_CELLS:
            continue
        disease = bool(frame["disease_eligible"].iloc[0])
        incomplete = frame["incomplete"].to_numpy(bool)
        decoupled = frame["decoupled"].to_numpy(bool)
        other = incomplete & ~decoupled
        for signature_type, perturbation in signatures:
            common = {
                "dataset": dataset,
                "subject_id": str(subject),
                "diagnosis": str(frame["diagnosis"].iloc[0]),
                "disease_group": str(frame["disease_group"].iloc[0]),
                "disease_eligible": disease,
                "signature_type": signature_type,
                "perturbation": perturbation,
                "n_cells": len(frame),
            }
            axis_col = f"residual__{signature_type}__{perturbation}__axis2"
            if has_activation and axis_col in frame:
                if disease and decoupled.sum() >= MIN_STATE_CELLS and other.sum() >= MIN_STATE_CELLS:
                    rows.append({
                        **common,
                        "endpoint": "decoupled_minus_other_incomplete",
                        "effect": float(frame.loc[decoupled, axis_col].mean() - frame.loc[other, axis_col].mean()),
                        "n_endpoint_cells": int(decoupled.sum() + other.sum()),
                        "n_case_cells": int(decoupled.sum()),
                        "n_reference_cells": int(other.sum()),
                    })
                if incomplete.sum() >= MIN_DONOR_CELLS:
                    rho = stats.spearmanr(
                        frame.loc[incomplete, axis_col], frame.loc[incomplete, "activation_residual"]
                    ).statistic
                    rows.append({
                        **common,
                        "endpoint": "axis2_spearman",
                        "effect": float(rho),
                        "n_endpoint_cells": int(incomplete.sum()),
                        "n_case_cells": np.nan,
                        "n_reference_cells": np.nan,
                    })

            pt_col = f"score__{signature_type}__{perturbation}__pseudotime"
            if pt_col in frame and np.ptp(frame["pseudotime"].to_numpy(float)) >= 0.05:
                rho = stats.spearmanr(frame[pt_col], frame["pseudotime"]).statistic
                rows.append({
                    **common,
                    "endpoint": "pseudotime_spearman",
                    "effect": float(rho),
                    "n_endpoint_cells": len(frame),
                    "n_case_cells": np.nan,
                    "n_reference_cells": np.nan,
                })

            for key, outcome in [
                ("lam2", "lam2_residual"),
                ("precursor", "precursor_residual"),
            ]:
                score_col = f"residual__{signature_type}__{perturbation}__{key}"
                if score_col not in frame:
                    continue
                rho = stats.spearmanr(frame[score_col], frame[outcome]).statistic
                rows.append({
                    **common,
                    "endpoint": f"{key}_residual_spearman",
                    "effect": float(rho),
                    "n_endpoint_cells": len(frame),
                    "n_case_cells": np.nan,
                    "n_reference_cells": np.nan,
                })
    return pd.DataFrame(rows)


def summarize_effects(
    donor: pd.DataFrame,
    availability: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_rows = []
    for key, frame in donor.groupby(
        ["dataset", "signature_type", "perturbation", "endpoint"], observed=True
    ):
        values = frame["effect"].dropna().to_numpy(float)
        if len(values) < 3:
            continue
        p_value, method = sign_flip_test(values, seed + len(test_rows))
        expected_sign = np.nan
        if key[3] in {"decoupled_minus_other_incomplete", "axis2_spearman"}:
            expected_sign = 1.0 if key[1] == "published_response" else -1.0
        test_rows.append({
            "dataset": key[0],
            "signature_type": key[1],
            "perturbation": key[2],
            "endpoint": key[3],
            "n_donors": len(values),
            "mean_effect": values.mean(),
            "median_effect": np.median(values),
            "standardized_effect": values.mean() / (values.std(ddof=1) if values.std(ddof=1) > 0 else 1),
            "positive_fraction": np.mean(values > 0),
            "expected_sign": expected_sign,
            "expected_direction_success": bool(values.mean() * expected_sign > 0) if np.isfinite(expected_sign) else np.nan,
            "sign_flip_p": p_value,
            "test_method": method,
        })
    tests = pd.DataFrame(test_rows)
    tests["fdr_bh_within_dataset_endpoint"] = tests.groupby(
        ["dataset", "signature_type", "endpoint"], observed=True
    )["sign_flip_p"].transform(lambda x: multipletests(x, method="fdr_bh")[1])

    meta_rows = []
    for key, frame in tests.groupby(["signature_type", "perturbation", "endpoint"], observed=True):
        signs = np.sign(frame["mean_effect"].to_numpy(float))
        p_values = np.clip(frame["sign_flip_p"].to_numpy(float), 1e-12, 1)
        z_scores = signs * stats.norm.isf(p_values / 2)
        weights = np.sqrt(frame["n_donors"].to_numpy(float))
        combined_z = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
        expected_sign = frame["expected_sign"].dropna()
        expected = float(expected_sign.iloc[0]) if len(expected_sign) else np.nan
        meta_rows.append({
            "signature_type": key[0],
            "perturbation": key[1],
            "endpoint": key[2],
            "n_cohorts": len(frame),
            "n_donors": int(frame["n_donors"].sum()),
            "n_positive_cohorts": int(np.sum(signs > 0)),
            "direction_concordant": bool(len(set(signs)) == 1),
            "mean_cohort_effect": frame["mean_effect"].mean(),
            "mean_standardized_effect": frame["standardized_effect"].mean(),
            "combined_z": combined_z,
            "stouffer_p": float(2 * stats.norm.sf(abs(combined_z))),
            "expected_sign": expected,
            "n_expected_direction_cohorts": int(np.sum(signs == expected)) if np.isfinite(expected) else np.nan,
        })
    meta = pd.DataFrame(meta_rows)
    meta["fdr_bh_within_signature_endpoint"] = meta.groupby(
        ["signature_type", "endpoint"], observed=True
    )["stouffer_p"].transform(lambda x: multipletests(x, method="fdr_bh")[1])
    meta["fdr_bh_global_primary"] = np.nan
    primary = meta["signature_type"].eq("published_response")
    meta.loc[primary, "fdr_bh_global_primary"] = multipletests(
        meta.loc[primary, "stouffer_p"], method="fdr_bh"
    )[1]

    decision_rows = []
    for perturbation in PERTURBATIONS:
        row = meta.loc[
            meta["signature_type"].eq("published_response")
            & meta["perturbation"].eq(perturbation)
            & meta["endpoint"].eq("decoupled_minus_other_incomplete")
        ]
        if row.empty:
            decision_rows.append({"perturbation": perturbation, "primary_bridge_decision": "not_testable"})
            continue
        result = row.iloc[0]
        available = availability.loc[
            availability["signature_type"].eq("published_response")
            & availability["perturbation"].eq(perturbation)
            & availability["endpoint_holdout"].eq("axis2")
            & availability["dataset"].isin(["GSE138266", "GSE286068", "GSE328265"])
        ]
        min_genes = int(available["n_available_after_holdout"].min())
        passed = bool(
            result["combined_z"] > 0
            and result["n_expected_direction_cohorts"] >= 2
            and result["fdr_bh_within_signature_endpoint"] < 0.05
            and min_genes >= 20
        )
        adaptation = meta.loc[
            meta["signature_type"].eq("published_response")
            & meta["perturbation"].eq(perturbation)
            & meta["endpoint"].isin(["pseudotime_spearman", "lam2_residual_spearman"])
        ]
        cost = adaptation.loc[
            adaptation["combined_z"].gt(0)
            & adaptation["fdr_bh_within_signature_endpoint"].lt(0.05)
        ]
        direct_decoupled = meta.loc[
            meta["signature_type"].eq("direct_ko_consensus")
            & meta["perturbation"].eq(perturbation)
            & meta["endpoint"].eq("decoupled_minus_other_incomplete")
        ].iloc[0]
        direct_axis2 = meta.loc[
            meta["signature_type"].eq("direct_ko_consensus")
            & meta["perturbation"].eq(perturbation)
            & meta["endpoint"].eq("axis2_spearman")
        ].iloc[0]
        direct_pseudotime = meta.loc[
            meta["signature_type"].eq("direct_ko_consensus")
            & meta["perturbation"].eq(perturbation)
            & meta["endpoint"].eq("pseudotime_spearman")
        ].iloc[0]
        direct_ko_bridge = bool(
            direct_decoupled["combined_z"] < 0
            and direct_decoupled["n_expected_direction_cohorts"] >= 2
            and direct_decoupled["fdr_bh_within_signature_endpoint"] < 0.05
            and direct_axis2["combined_z"] < 0
            and direct_axis2["n_expected_direction_cohorts"] >= 2
            and direct_axis2["fdr_bh_within_signature_endpoint"] < 0.05
        )
        adaptation_preserved = bool(
            direct_pseudotime["combined_z"] > 0
            and direct_pseudotime["fdr_bh_within_signature_endpoint"] < 0.05
        )
        therapeutic_decoupling = bool(passed and direct_ko_bridge and adaptation_preserved)
        decision_rows.append({
            "perturbation": perturbation,
            "primary_bridge_decision": "pass" if passed else "fail",
            "primary_combined_z": result["combined_z"],
            "primary_fdr": result["fdr_bh_within_signature_endpoint"],
            "primary_global_fdr": result["fdr_bh_global_primary"],
            "expected_direction_cohorts": int(result["n_expected_direction_cohorts"]),
            "minimum_axis2_holdout_genes": min_genes,
            "response_adaptation_association_supported": not cost.empty,
            "response_adaptation_endpoints": ";".join(cost["endpoint"].str.replace("_spearman", "")),
            "direct_ko_bridge_decision": "pass" if direct_ko_bridge else "fail",
            "direct_ko_decoupled_z": direct_decoupled["combined_z"],
            "direct_ko_decoupled_fdr": direct_decoupled["fdr_bh_within_signature_endpoint"],
            "direct_ko_axis2_z": direct_axis2["combined_z"],
            "direct_ko_axis2_fdr": direct_axis2["fdr_bh_within_signature_endpoint"],
            "direct_ko_pseudotime_z": direct_pseudotime["combined_z"],
            "direct_ko_pseudotime_fdr": direct_pseudotime["fdr_bh_within_signature_endpoint"],
            "adaptation_preservation_supported": adaptation_preserved,
            "therapeutic_decoupling_supported": therapeutic_decoupling,
            "mechanistic_interpretation": (
                "experimentally_anchored_inflammation_adaptation_decoupling" if therapeutic_decoupling
                else "bidirectional_inflammatory_bridge" if passed and direct_ko_bridge
                else "response_signature_inflammatory_bridge" if passed
                else "external_bridge_not_confirmed"
            ),
        })
    return tests, meta, pd.DataFrame(decision_rows)


def leave_one_cohort_out(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in tests.groupby(["signature_type", "perturbation", "endpoint"], observed=True):
        if len(frame) < 3:
            continue
        for excluded in frame["dataset"]:
            kept = frame.loc[frame["dataset"].ne(excluded)]
            signs = np.sign(kept["mean_effect"].to_numpy(float))
            p_values = np.clip(kept["sign_flip_p"].to_numpy(float), 1e-12, 1)
            z_scores = signs * stats.norm.isf(p_values / 2)
            weights = np.sqrt(kept["n_donors"].to_numpy(float))
            combined_z = float(np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2)))
            rows.append({
                "signature_type": key[0],
                "perturbation": key[1],
                "endpoint": key[2],
                "excluded_dataset": excluded,
                "n_cohorts_retained": len(kept),
                "n_donors_retained": int(kept["n_donors"].sum()),
                "combined_z": combined_z,
                "stouffer_p": float(2 * stats.norm.sf(abs(combined_z))),
            })
    return pd.DataFrame(rows)


def plot_bridge(
    output_dir: Path,
    availability: pd.DataFrame,
    tests: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    colors = {"Ifngr1": "#C23B3B", "Tnfrsf1a": "#D98C2B", "Csf2ra": "#2B7A78", "Tgfbr1": "#4568A6"}
    labels = {"Ifngr1": "IFNGR1", "Tnfrsf1a": "TNFR1", "Csf2ra": "CSF2RA", "Tgfbr1": "TGFBR1"}
    endpoint_labels = {
        "decoupled_minus_other_incomplete": "Decoupled\nresponse excess",
        "axis2_spearman": "Axis 2\nassociation",
        "pseudotime_spearman": "Differentiation\nassociation",
        "lam2_residual_spearman": "LAM2 residual\nassociation",
        "precursor_residual_spearman": "Precursor residual\nassociation",
    }

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[0.9, 1.35], height_ratios=[1.0, 1.15])

    ax = fig.add_subplot(grid[0, 0])
    size = availability.loc[
        availability["signature_type"].eq("published_response")
        & availability["endpoint_holdout"].eq("axis2")
    ].groupby("perturbation", observed=True)["n_available_after_holdout"].min().reindex(PERTURBATIONS)
    ax.barh(range(4), size, color=[colors[p] for p in PERTURBATIONS], height=0.65)
    ax.set_yticks(range(4), [labels[p] for p in PERTURBATIONS])
    ax.invert_yaxis()
    ax.set_xlabel("Mapped genes after Axis 2 holdout")
    ax.set_title("a  Published in vivo response signatures", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    primary = meta.loc[meta["signature_type"].eq("published_response")].copy()
    matrix = primary.pivot(index="perturbation", columns="endpoint", values="combined_z").reindex(
        index=PERTURBATIONS, columns=ENDPOINTS
    )
    fdr = primary.pivot(index="perturbation", columns="endpoint", values="fdr_bh_within_signature_endpoint").reindex(
        index=PERTURBATIONS, columns=ENDPOINTS
    )
    limit = max(2.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_yticks(range(4), [labels[p] for p in PERTURBATIONS])
    ax.set_xticks(range(len(ENDPOINTS)), [endpoint_labels[e] for e in ENDPOINTS], rotation=35, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                star = "*" if fdr.iloc[i, j] < 0.05 else ""
                ax.text(j, i, f"{value:.1f}{star}", ha="center", va="center", color="black", fontsize=6.5)
    ax.set_title("b  Cross-cohort experimental bridge", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Signed Stouffer z")

    ax = fig.add_subplot(grid[1, 0])
    forest = tests.loc[
        tests["signature_type"].eq("published_response")
        & tests["endpoint"].eq("decoupled_minus_other_incomplete")
    ].copy()
    y = 0
    yticks, ylabels = [], []
    for perturbation in PERTURBATIONS:
        frame = forest.loc[forest["perturbation"].eq(perturbation)]
        for dataset in ["GSE138266", "GSE286068", "GSE328265"]:
            row = frame.loc[frame["dataset"].eq(dataset)]
            if row.empty:
                continue
            ax.scatter(row["mean_effect"], y, s=24, color=colors[perturbation], edgecolor="white", linewidth=0.4)
            yticks.append(y)
            ylabels.append(f"{labels[perturbation]}  {dataset.replace('GSE', '')}")
            y += 1
        row = meta.loc[
            meta["signature_type"].eq("published_response")
            & meta["perturbation"].eq(perturbation)
            & meta["endpoint"].eq("decoupled_minus_other_incomplete")
        ]
        if not row.empty:
            ax.scatter(row["mean_cohort_effect"], y, marker="D", s=34, color=colors[perturbation], edgecolor="black", linewidth=0.5)
            yticks.append(y)
            ylabels.append(f"{labels[perturbation]}  meta")
            y += 1.6
    ax.axvline(0, color="#777777", lw=0.7, ls="--")
    ax.set_yticks(yticks, ylabels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Decoupled minus other incomplete\nresponse score")
    ax.set_title("c  Donor-level inflammatory bridge", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1])
    direct = meta.loc[meta["signature_type"].eq("direct_ko_consensus")]
    wide = direct.pivot(index="perturbation", columns="endpoint", values="combined_z").reindex(PERTURBATIONS)
    label_offsets = {
        "Ifngr1": (0.12, 0.12),
        "Tnfrsf1a": (0.12, 0.34),
        "Csf2ra": (0.12, 0.12),
        "Tgfbr1": (0.12, -0.16),
    }
    for perturbation, row in wide.iterrows():
        inflammatory_reversal = -row["decoupled_minus_other_incomplete"]
        differentiation = row["pseudotime_spearman"]
        ax.scatter(
            inflammatory_reversal, differentiation,
            s=62, color=colors[perturbation], edgecolor="white", linewidth=0.6, zorder=3,
        )
        dx, dy = label_offsets[perturbation]
        ax.text(
            inflammatory_reversal + dx,
            differentiation + dy,
            labels[perturbation], fontsize=6.5,
        )
    ax.axhline(0, color="#777777", lw=0.7, ls="--")
    ax.axvline(0, color="#777777", lw=0.7, ls="--")
    ax.set_xlabel("KO-like inflammatory reversal (-z)")
    ax.set_ylabel("KO-like differentiation association (z)")
    ax.set_title("d  Directional KO triangulation", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    for suffix in ["pdf", "png", "svg"]:
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(output_dir / f"in_vivo_CRISPR_bridge.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(
    path: Path,
    source_qc: pd.DataFrame,
    meta: pd.DataFrame,
    decisions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> None:
    lines = [
        "# In vivo CRISPR-to-human MS CSF bridge",
        "",
        "## Design",
        "",
        "Author-defined neuroinflammatory cytokine-response signatures and an independent "
        "Hoxb8FL/chimera direct-KO consensus were mapped through one-to-one mouse-human "
        "orthologs. Human trajectories, Axis 2, decoupled-state calls and LAM programs were "
        "read unchanged. Endpoint genes were removed before each projection, and inference "
        "used donor summaries and sign-flip tests.",
        "",
        "## External source quality",
        "",
    ]
    for row in source_qc.itertuples():
        lines.append(
            f"- {row.perturbation}: Hoxb8FL-chimera Pearson r={row.pearson_hoxb8_chimera:.3f}; "
            f"same-direction fraction={row.same_direction_fraction:.3f}; published signature "
            f"mapped {int(row.n_published_one_to_one_mapped)}/{int(row.n_published_response_genes)} genes; "
            f"{row.published_fraction_negative_in_both_KO_models:.1%} were down in both KO systems."
        )
    lines.extend(["", "## Primary bridge decisions", ""])
    for row in decisions.itertuples():
        lines.append(
            f"- {row.perturbation}: **{row.primary_bridge_decision}**; decoupled-response "
            f"z={getattr(row, 'primary_combined_z', np.nan):.3f}, "
            f"endpoint-family FDR={getattr(row, 'primary_fdr', np.nan):.4g}, "
            f"expected-direction cohorts={getattr(row, 'expected_direction_cohorts', np.nan)}; "
            f"direct-KO bridge={getattr(row, 'direct_ko_bridge_decision', '')}; "
            f"adaptation preservation={getattr(row, 'adaptation_preservation_supported', False)}; "
            f"interpretation={getattr(row, 'mechanistic_interpretation', '')}."
        )
    lines.extend(["", "## Cross-cohort endpoint map", ""])
    primary = meta.loc[meta["signature_type"].eq("published_response")].sort_values(
        ["endpoint", "stouffer_p"]
    )
    for row in primary.itertuples():
        lines.append(
            f"- {row.perturbation} {row.endpoint}: z={row.combined_z:.3f}, "
            f"P={row.stouffer_p:.4g}, FDR={row.fdr_bh_within_signature_endpoint:.4g}, "
            f"cohorts={int(row.n_cohorts)}, donors={int(row.n_donors)}, "
            f"direction-concordant={row.direction_concordant}."
        )
    lines.extend(["", "## Direct-KO directional triangulation", ""])
    direct = meta.loc[
        meta["signature_type"].eq("direct_ko_consensus")
        & meta["endpoint"].isin([
            "decoupled_minus_other_incomplete", "axis2_spearman", "pseudotime_spearman"
        ])
    ].sort_values(["perturbation", "endpoint"])
    for row in direct.itertuples():
        lines.append(
            f"- {row.perturbation} {row.endpoint}: z={row.combined_z:.3f}, "
            f"P={row.stouffer_p:.4g}, FDR={row.fdr_bh_within_signature_endpoint:.4g}, "
            f"cohorts={int(row.n_cohorts)}, donors={int(row.n_donors)}."
        )
    lines.extend(["", "## Leave-one-cohort-out audit", ""])
    for signature_type, endpoint, expected_sign in [
        ("published_response", "decoupled_minus_other_incomplete", 1),
        ("published_response", "axis2_spearman", 1),
        ("direct_ko_consensus", "decoupled_minus_other_incomplete", -1),
        ("direct_ko_consensus", "axis2_spearman", -1),
        ("direct_ko_consensus", "pseudotime_spearman", 1),
    ]:
        frame = leave_one_out.loc[
            leave_one_out["signature_type"].eq(signature_type)
            & leave_one_out["perturbation"].eq("Ifngr1")
            & leave_one_out["endpoint"].eq(endpoint)
        ]
        sign_ok = bool((frame["combined_z"] * expected_sign > 0).all())
        lines.append(
            f"- {signature_type} {endpoint}: expected direction retained={sign_ok}; "
            f"z range={frame['combined_z'].min():.3f} to {frame['combined_z'].max():.3f}; "
            f"maximum P={frame['stouffer_p'].max():.4g}."
        )
    max_technical = diagnostics[["residual_spearman_log_counts", "residual_spearman_n_genes"]].abs().max().max()
    ifngr_diagnostics = diagnostics.loc[diagnostics["perturbation"].eq("Ifngr1")]
    max_ifngr_technical = ifngr_diagnostics[
        ["residual_spearman_log_counts", "residual_spearman_n_genes"]
    ].abs().max().max()
    lines.extend([
        "",
        "## Technical audit",
        "",
        f"- Maximum absolute residual-score correlation with library size or detected genes: {max_technical:.3f}.",
        f"- Maximum for the nominated IFNGR1 bridge: {max_ifngr_technical:.3f}.",
        "- The direct-KO consensus is a source-format sensitivity analysis because the released "
        "whole-macrophage fold-change table does not carry per-gene adjusted P values.",
        "- These projections externally constrain direction but do not convert human observational "
        "single-cell data or virtual perturbations into an experimental efficacy claim.",
    ])
    path.write_text("\n".join(lines) + "\n")


def qa_outputs(
    output_dir: Path,
    manifest: pd.DataFrame,
    source_qc: pd.DataFrame,
    availability: pd.DataFrame,
    donor: pd.DataFrame,
    tests: pd.DataFrame,
    meta: pd.DataFrame,
    decisions: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> dict:
    checks = {
        "source_files_hashed": bool(len(manifest) >= 10 and manifest["sha256"].str.len().eq(64).all()),
        "four_perturbations_present": set(decisions["perturbation"]) == set(PERTURBATIONS),
        "all_primary_axis2_scores_powered": bool(
            availability.loc[
                availability["signature_type"].eq("published_response")
                & availability["endpoint_holdout"].eq("axis2")
            , "powered"].all()
        ),
        "all_direct_ko_scores_powered": bool(
            availability.loc[
                availability["signature_type"].eq("direct_ko_consensus"), "powered"
            ].all()
        ),
        "excel_date_gene_artifacts_recorded": bool(
            source_qc["n_invalid_excel_date_gene_rows"].gt(0).all()
        ),
        "no_unresolved_duplicate_gene_rows": bool(
            source_qc["n_collapsed_duplicate_gene_rows"].eq(0).all()
        ),
        "four_datasets_scored": availability["dataset"].nunique() == 4,
        "donor_effects_nonempty": len(donor) > 0,
        "cohort_tests_nonempty": len(tests) > 0,
        "meta_contains_all_endpoints": set(ENDPOINTS).issubset(set(meta["endpoint"])),
        "leave_one_cohort_out_complete": bool(
            leave_one_out.groupby(["signature_type", "perturbation", "endpoint"], observed=True)
            ["excluded_dataset"].nunique().ge(3).all()
        ),
        "figure_pdf_exists": (output_dir / "in_vivo_CRISPR_bridge.pdf").exists(),
        "figure_png_exists": (output_dir / "in_vivo_CRISPR_bridge.png").exists(),
        "figure_svg_exists": (output_dir / "in_vivo_CRISPR_bridge.svg").exists(),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "n_checks": len(checks),
        "n_passed": sum(checks.values()),
        "checks": checks,
    }




def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_dir = root / "data/reference/de_la_rosa_2026"
    output_dir = root / "results/in_vivo_crispr_bridge"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_source_manifest(source_dir)
    orthologs = one_to_one_orthologs(source_dir / "HOM_MouseHumanSequence.rpt")
    published, direct_all, direct_balanced, source_qc = read_external_signatures(source_dir, orthologs)
    signatures = {}
    for perturbation in PERTURBATIONS:
        signatures[("published_response", perturbation)] = published.loc[
            published["perturbation"].eq(perturbation) & published["human_gene"].ne("")
        ].copy()
        signatures[("direct_ko_consensus", perturbation)] = direct_balanced.loc[
            direct_balanced["perturbation"].eq(perturbation) & direct_balanced["human_gene"].ne("")
        ].copy()

    holdouts = endpoint_holdouts(root)
    cell_frames = []
    availability_frames = []
    diagnostic_frames = []
    donor_frames = []
    for dataset, config in DATASETS.items():
        print(f"Scoring external CRISPR signatures: {dataset}", flush=True)
        cells, availability, diagnostics = score_dataset(
            root, dataset, config, signatures, holdouts
        )
        cells.to_csv(
            output_dir / f"{dataset}_in_vivo_CRISPR_bridge_cell_scores.csv.gz",
            index=False,
            compression="gzip",
        )
        donor_frames.append(donor_effects(cells, signatures))
        availability_frames.append(availability)
        diagnostic_frames.append(diagnostics)
        cell_frames.append(cells[["cell_id", "dataset", "subject_id"]])

    availability = pd.concat(availability_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    donor = pd.concat(donor_frames, ignore_index=True)
    tests, meta, decisions = summarize_effects(donor, availability, args.seed)
    leave_one_out = leave_one_cohort_out(tests)

    manifest.to_csv(output_dir / "external_source_manifest.csv", index=False)
    orthologs.to_csv(output_dir / "mouse_human_one_to_one_orthologs.csv", index=False)
    published.to_csv(output_dir / "published_cytokine_signature_ledger.csv", index=False)
    direct_all.to_csv(output_dir / "direct_ko_all_gene_effects.csv.gz", index=False, compression="gzip")
    direct_balanced.to_csv(output_dir / "direct_ko_consensus_signature_ledger.csv", index=False)
    source_qc.to_csv(output_dir / "external_signature_source_qc.csv", index=False)
    availability.to_csv(output_dir / "human_signature_availability.csv", index=False)
    diagnostics.to_csv(output_dir / "human_score_residualization_diagnostics.csv", index=False)
    donor.to_csv(output_dir / "donor_level_bridge_effects.csv", index=False)
    tests.to_csv(output_dir / "cohort_bridge_tests.csv", index=False)
    meta.to_csv(output_dir / "bridge_meta_analysis.csv", index=False)
    leave_one_out.to_csv(output_dir / "leave_one_cohort_out_audit.csv", index=False)
    decisions.to_csv(output_dir / "primary_bridge_decisions.csv", index=False)

    plot_bridge(output_dir, availability, tests, meta)
    write_report(
        output_dir / "IN_VIVO_CRISPR_BRIDGE_REPORT.md",
        source_qc,
        meta,
        decisions,
        diagnostics,
        leave_one_out,
    )
    qa = qa_outputs(
        output_dir, manifest, source_qc, availability, donor, tests, meta, decisions, leave_one_out
    )
    (output_dir / "CRISPR_BRIDGE_VALIDATION.json").write_text(json.dumps(qa, indent=2) + "\n")
    if qa["status"] != "PASS":
        raise RuntimeError(f"CRISPR bridge validation failed: {qa}")
    print(decisions.to_string(index=False), flush=True)
    print(meta.loc[meta["signature_type"].eq("published_response")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
