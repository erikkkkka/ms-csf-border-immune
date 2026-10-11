#!/usr/bin/env python3
"""Fetch and score targeted LTA4H-inhibitor DRUG-seq profiles."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests


DE_URL = "https://s3.perturbseqr.maayanlab.cloud/data/nibr-drug-seq-de.parquet"
METADATA_URL = (
    "https://s3.perturbseqr.maayanlab.cloud/data/"
    "nibr-drug-seq-drug_metadata.parquet"
)
DRUGS = {
    "MC-43-PK30": {"drug": "Acebilustat", "pubchem_cid": 68488178, "selective": True},
    "AE-71-CB15": {"drug": "Ubenimex", "pubchem_cid": 72172, "selective": False},
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--full-meta", type=Path, required=True)
    parser.add_argument("--ko-votes", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--httpfs-extension", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_target_data(data_dir: Path, extension: Path | None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = data_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA disable_progress_bar")
    if extension:
        connection.execute(f"LOAD '{extension.resolve()}'")
    else:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")

    cids = ",".join(str(item["pubchem_cid"]) for item in DRUGS.values())
    metadata = connection.execute(
        f"""
        SELECT *
        FROM read_parquet('{METADATA_URL}')
        WHERE pubchem_cid IN ({cids})
        ORDER BY pubchem_cid
        """
    ).fetchdf()
    metadata.to_csv(data_dir / "drug_metadata.csv", index=False)
    print(f"Saved metadata for {len(metadata)} compounds", flush=True)

    expected_profiles = [
        f"{code}_{dose}_24.0"
        for code in DRUGS
        for dose in ("0.01", "0.1", "1.0", "10.0")
    ]
    profile_frames = []
    for profile_index, pert_id in enumerate(expected_profiles, start=1):
        output_path = profile_dir / f"{pert_id}.csv.gz"
        if output_path.exists() and output_path.stat().st_size > 1000:
            frame = pd.read_csv(output_path)
            print(
                f"[{profile_index}/{len(expected_profiles)}] Reusing {pert_id}: "
                f"{len(frame):,} genes",
                flush=True,
            )
        else:
            print(
                f"[{profile_index}/{len(expected_profiles)}] Querying {pert_id}",
                flush=True,
            )
            frame = connection.execute(
                f"""
                SELECT pert_id, upper(genes) AS gene_symbol, logFC
                FROM read_parquet('{DE_URL}')
                WHERE pert_id = ?
                ORDER BY gene_symbol
                """,
                [pert_id],
            ).fetchdf()
            if frame.empty:
                raise RuntimeError(f"No DRUG-seq rows returned for {pert_id}")
            frame.to_csv(output_path, index=False)
            print(f"Saved {len(frame):,} genes", flush=True)
        profile_frames.append(frame)
    differential = pd.concat(profile_frames, ignore_index=True)
    connection.close()

    differential.to_csv(data_dir / "targeted_full_de.csv.gz", index=False)
    manifest = pd.DataFrame([
        {
            "resource": "NIBR_DRUGseq_U2OS_MoABox_full_DE",
            "url": DE_URL,
            "local_file": "targeted_full_de.csv.gz",
            "sha256": sha256(data_dir / "targeted_full_de.csv.gz"),
            "selection": "pert_id prefixes MC-43-PK30 and AE-71-CB15",
            "cell_line": "U2OS",
            "time_h": 24,
        },
        {
            "resource": "NIBR_DRUGseq_drug_metadata",
            "url": METADATA_URL,
            "local_file": "drug_metadata.csv",
            "sha256": sha256(data_dir / "drug_metadata.csv"),
            "selection": f"PubChem CID in ({cids})",
            "cell_line": "not_applicable",
            "time_h": np.nan,
        },
    ])
    manifest.to_csv(data_dir / "source_manifest.csv", index=False)


def rank_normal(values: pd.Series) -> np.ndarray:
    ranks = rankdata(values.to_numpy(dtype=float), method="average")
    quantiles = (ranks - 0.5) / len(ranks)
    return norm.ppf(quantiles)


def permutation_signature_score(
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


def permutation_module_suppression(
    rank_z: np.ndarray,
    module_idx: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(-rank_z[module_idx].mean())
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for index in range(permutations):
        sampled = rng.choice(len(rank_z), size=len(module_idx), replace=False)
        null[index] = -rank_z[sampled].mean()
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


def parse_perturbation(pert_id: str) -> tuple[str, str, float, float]:
    code, dose, time_h = pert_id.rsplit("_", 2)
    info = DRUGS[code]
    return code, str(info["drug"]), float(dose), float(time_h)


def score_profiles(
    differential: pd.DataFrame,
    signature: pd.DataFrame,
    full_meta: pd.DataFrame,
    ko_votes: pd.DataFrame,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signature = signature.copy()
    full_meta = full_meta.copy()
    signature["gene_symbol"] = signature["gene_symbol"].str.upper()
    full_meta["gene_symbol"] = full_meta["gene_symbol"].str.upper()
    ko_votes = ko_votes.copy()
    ko_votes["gene_symbol"] = ko_votes["gene_symbol"].str.upper()
    ko_votes = ko_votes.loc[ko_votes["ko_vote_mean_all_signatures"].ne(0)]

    rows: list[dict] = []
    matched_rows: list[pd.DataFrame] = []
    for profile_index, (pert_id, raw) in enumerate(differential.groupby("pert_id", sort=True)):
        code, drug, dose_um, time_h = parse_perturbation(pert_id)
        profile = raw.groupby("gene_symbol", as_index=False)["logFC"].mean()
        profile["rank_z"] = rank_normal(profile["logFC"])
        profile_index_map = {
            gene: index for index, gene in enumerate(profile["gene_symbol"])
        }
        rank_z = profile["rank_z"].to_numpy()
        disease_up = set(signature.loc[
            signature["signature_direction"].eq("disease_up"), "gene_symbol"
        ])
        disease_down = set(signature.loc[
            signature["signature_direction"].eq("disease_down"), "gene_symbol"
        ])
        up_idx = np.array([
            profile_index_map[gene] for gene in disease_up if gene in profile_index_map
        ])
        down_idx = np.array([
            profile_index_map[gene] for gene in disease_down if gene in profile_index_map
        ])
        sig_score, sig_p, sig_z = permutation_signature_score(
            rank_z, up_idx, down_idx, permutations, seed + profile_index
        )

        selected_match = signature.merge(profile, on="gene_symbol", how="inner")
        full_match = full_meta.merge(profile, on="gene_symbol", how="inner")
        rho = spearmanr(full_match["mean_logfc"], full_match["logFC"])
        ko_match = ko_votes.merge(profile, on="gene_symbol", how="inner")
        ko_rho = spearmanr(
            ko_match["ko_vote_mean_all_signatures"], ko_match["logFC"]
        )

        result = {
            "pert_id": pert_id,
            "drug_code": code,
            "drug": drug,
            "dose_um": dose_um,
            "time_h": time_h,
            "selective_lta4h_inhibitor": bool(DRUGS[code]["selective"]),
            "n_profile_genes": len(profile),
            "n_disease_up_matched": len(up_idx),
            "n_disease_down_matched": len(down_idx),
            "bam_reversal_rank_score": sig_score,
            "bam_reversal_permutation_p": sig_p,
            "bam_reversal_permutation_z": sig_z,
            "bam_weighted_reversal_cosine": weighted_reversal_cosine(selected_match),
            "full_meta_reversal_spearman": -float(rho.statistic),
            "full_meta_spearman_p": float(rho.pvalue),
            "lta4h_logfc": float(
                profile.loc[profile["gene_symbol"].eq("LTA4H"), "logFC"].iloc[0]
            ) if profile["gene_symbol"].eq("LTA4H").any() else np.nan,
            "ko_consensus_mimicry_spearman": float(ko_rho.statistic),
            "ko_consensus_mimicry_p": float(ko_rho.pvalue),
            "ko_consensus_genes_matched": len(ko_match),
        }
        for module_offset, (module, genes) in enumerate(MODULES.items()):
            module_idx = np.array([
                profile_index_map[gene] for gene in genes if gene in profile_index_map
            ])
            score, p_value, z_score = permutation_module_suppression(
                rank_z,
                module_idx,
                permutations,
                seed + 100 + profile_index * 10 + module_offset,
            )
            result[f"{module}_n_genes"] = len(module_idx)
            result[f"{module}_suppression_score"] = score
            result[f"{module}_suppression_p"] = p_value
            result[f"{module}_suppression_z"] = z_score
        rows.append(result)

        selected_match.insert(0, "pert_id", pert_id)
        selected_match.insert(1, "drug", drug)
        selected_match.insert(2, "dose_um", dose_um)
        matched_rows.append(selected_match)

    results = pd.DataFrame(rows).sort_values(["drug", "dose_um"]).reset_index(drop=True)
    p_columns = [
        "bam_reversal_permutation_p", "full_meta_spearman_p",
        "ko_consensus_mimicry_p",
    ] + [f"{module}_suppression_p" for module in MODULES]
    for column in p_columns:
        results[f"{column}_fdr"] = multipletests(results[column], method="fdr_bh")[1]
    return results, pd.concat(matched_rows, ignore_index=True)


def dose_consistency(differential: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, info in DRUGS.items():
        subset = differential.loc[differential["pert_id"].str.startswith(code)].copy()
        profiles = {}
        for pert_id, frame in subset.groupby("pert_id"):
            dose = float(pert_id.rsplit("_", 2)[1])
            profiles[dose] = frame.groupby("gene_symbol")["logFC"].mean()
        doses = sorted(profiles)
        for first_index, dose_a in enumerate(doses):
            for dose_b in doses[first_index + 1:]:
                match = pd.concat(
                    [profiles[dose_a], profiles[dose_b]], axis=1, join="inner"
                ).dropna()
                match.columns = ["a", "b"]
                rho = spearmanr(match["a"], match["b"])
                rows.append({
                    "drug": info["drug"],
                    "dose_a_um": dose_a,
                    "dose_b_um": dose_b,
                    "n_shared_genes": len(match),
                    "transcriptome_spearman": rho.statistic,
                    "spearman_p": rho.pvalue,
                })
    return pd.DataFrame(rows)


def drug_summary(scores: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, frame in scores.groupby("drug"):
        correlations = consistency.loc[
            consistency["drug"].eq(drug), "transcriptome_spearman"
        ]
        dose_trend = spearmanr(
            np.log10(frame["dose_um"]), frame["bam_reversal_rank_score"]
        )
        rows.append({
            "drug": drug,
            "n_doses": len(frame),
            "positive_bam_reversal_doses": int(
                frame["bam_reversal_rank_score"].gt(0).sum()
            ),
            "bam_reversal_fdr_lt_0_05_doses": int(
                frame["bam_reversal_permutation_p_fdr"].lt(0.05).sum()
            ),
            "median_bam_reversal_rank_score": frame["bam_reversal_rank_score"].median(),
            "median_mhc_i_suppression_score": frame[
                "MHC_I_processing_suppression_score"
            ].median(),
            "median_ko_mimicry_spearman": frame[
                "ko_consensus_mimicry_spearman"
            ].median(),
            "median_pairwise_transcriptome_spearman": correlations.median(),
            "dose_trend_bam_reversal_spearman": dose_trend.statistic,
            "dose_trend_bam_reversal_p": dose_trend.pvalue,
        })
    return pd.DataFrame(rows)


def plot_scores(scores: pd.DataFrame, output_path: Path) -> None:
    colors = {"Acebilustat": "#007C83", "Ubenimex": "#C44E32"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    panels = [
        ("bam_reversal_rank_score", "BAM reversal rank score"),
        ("MHC_I_processing_suppression_score", "MHC-I suppression score"),
    ]
    for axis, (column, label) in zip(axes, panels):
        for drug, frame in scores.groupby("drug"):
            frame = frame.sort_values("dose_um")
            axis.plot(
                frame["dose_um"], frame[column], marker="o", linewidth=1.8,
                markersize=5, color=colors[drug], label=drug,
            )
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xticks([0.01, 0.1, 1, 10], labels=["0.01", "0.1", "1", "10"])
        axis.set_xlabel("Concentration (uM)")
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    lines = [
        "# LTA4H-inhibitor DRUG-seq directional audit",
        "",
        "## Data and estimand",
        "",
        "- Public NIBR DRUG-seq U2OS MoABox data; U2OS cells, 24 h, 0.01-10 uM.",
        "- The full differential-expression summary was queried by exact Novartis compound code.",
        "- A positive BAM reversal score means disease-up genes rank downward and disease-down "
        "genes rank upward after treatment.",
        "- A positive module suppression score means module genes rank downward after treatment.",
        "- Gene permutations test signature enrichment, not biological replicate uncertainty.",
        "",
        "## Compound identity",
        "",
    ]
    for _, row in metadata.iterrows():
        lines.append(
            f"- {row['nibr_drug_name']} ({row['nsn_code']}, PubChem {int(row['pubchem_cid'])}): "
            f"{row['moabox_moa'] or row['drh_moa']}; DRH targets: {row['drh_target']}."
        )
    lines.extend(["", "## Results", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['drug']}: positive BAM reversal at "
            f"{int(row['positive_bam_reversal_doses'])}/{int(row['n_doses'])} doses; "
            f"FDR<0.05 at {int(row['bam_reversal_fdr_lt_0_05_doses'])} doses; "
            f"median reversal={row['median_bam_reversal_rank_score']:.3f}, "
            f"median MHC-I suppression={row['median_mhc_i_suppression_score']:.3f}, "
            f"median LTA4H-KO mimicry rho={row['median_ko_mimicry_spearman']:.3f}, "
            f"median within-drug dose-profile rho="
            f"{row['median_pairwise_transcriptome_spearman']:.3f}."
        )
    best = scores.sort_values("bam_reversal_rank_score", ascending=False).iloc[0]
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        f"- Strongest nominal BAM reversal: {best['drug']} {best['dose_um']:g} uM "
        f"(score={best['bam_reversal_rank_score']:.3f}, permutation "
        f"P={best['bam_reversal_permutation_p']:.4g}, FDR="
        f"{best['bam_reversal_permutation_p_fdr']:.4g}).",
        "- Acebilustat is the selective LTA4H comparator. Ubenimex also targets ANPEP/RNPEP, "
        "so its signature cannot be attributed specifically to LTA4H.",
        "- U2OS is not a monocyte/macrophage/dendritic-cell model. These results are Tier-4 "
        "directional evidence and cannot establish activity in CSF BAM-like macrophages.",
        "- LYS006 was absent from the queried DRUG-seq and public gene-signature libraries.",
        "",
        "## Sources",
        "",
        "- Novartis U2OS MoABox dataset: https://doi.org/10.5281/zenodo.14291446",
        "- DRUG-seq method: https://pubmed.ncbi.nlm.nih.gov/35508359/",
        "- Perturb-Seqr/Harmonizome processed DE: "
        "https://perturbseqr.maayanlab.cloud/download",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch_target_data(args.data_dir, args.httpfs_extension)
    differential_path = args.data_dir / "targeted_full_de.csv.gz"
    metadata_path = args.data_dir / "drug_metadata.csv"
    if not differential_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run with --fetch to create targeted DRUG-seq inputs")

    differential = pd.read_csv(differential_path)
    differential["gene_symbol"] = differential["gene_symbol"].str.upper()
    metadata = pd.read_csv(metadata_path).fillna("")
    signature = pd.read_csv(args.signature)
    full_meta = pd.read_csv(args.full_meta)
    ko_votes = pd.read_csv(args.ko_votes)

    scores, matched = score_profiles(
        differential,
        signature,
        full_meta,
        ko_votes,
        args.permutations,
        args.seed,
    )
    consistency = dose_consistency(differential)
    summary = drug_summary(scores, consistency)

    scores.to_csv(args.output_dir / "LTA4H_inhibitor_DRUGseq_scores.csv", index=False)
    matched.to_csv(
        args.output_dir / "LTA4H_inhibitor_DRUGseq_BAM_gene_matches.csv.gz", index=False
    )
    consistency.to_csv(
        args.output_dir / "LTA4H_inhibitor_DRUGseq_dose_consistency.csv", index=False
    )
    summary.to_csv(args.output_dir / "LTA4H_inhibitor_DRUGseq_summary.csv", index=False)
    plot_scores(scores, args.output_dir / "LTA4H_inhibitor_DRUGseq_dose_scores.png")
    write_report(
        args.output_dir / "LTA4H_INHIBITOR_DRUGSEQ_REPORT.md",
        scores,
        summary,
        metadata,
    )
    print(scores[[
        "drug", "dose_um", "bam_reversal_rank_score",
        "bam_reversal_permutation_p", "bam_reversal_permutation_p_fdr",
        "MHC_I_processing_suppression_score", "ko_consensus_mimicry_spearman",
    ]].to_string(index=False))
    print("\nDrug summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
