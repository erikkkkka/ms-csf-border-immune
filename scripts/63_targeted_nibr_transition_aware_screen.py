#!/usr/bin/env python3
"""Score full NIBR DRUG-seq profiles against transition-aware MS CSF endpoints."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DE_URL = "https://s3.perturbseqr.maayanlab.cloud/data/nibr-drug-seq-de.parquet"
TARGETS = {
    "AA-66-PC55": "9-Azapaullone",
    "AA-07-FZ97": "Calcitriol",
    "AA-70-DH87": "Laduviglusib",
    "AD-42-OW73": "SB-415286",
    "AB-01-NB05": "Rosiglitazone",
    "BA-28-JD87": "Imatinib",
    "BA-15-VE07": "SB-216763",
    "BD-38-GC64": "Fingolimod phosphate",
    "CC-93-RP38": "Teriflunomide",
    "DE-48-HZ49": "BX-912",
    "HE-14-AT80": "Filgotinib",
    "ME-47-CY39": "Crisaborole",
    "NB-89-OV56": "PF-04802367",
    "ED-91-LY45": "Tideglusib",
    "PB-05-XP37": "PDK1 allosteric activator",
    "TA-25-ZO44": "PDK1-AKT inhibitor",
    "BA-95-YU85": "BAG-956",
    "HB-92-XE24": "GSK2334470",
    "PC-18-VX23": "AR-12",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--httpfs-extension", type=Path)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_profiles(root: Path, data_dir: Path, extension: Path | None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "data/reference/nibr-drug-seq-drug_metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    connection = duckdb.connect()
    connection.execute("PRAGMA disable_progress_bar")
    if extension:
        connection.execute(f"LOAD '{extension.resolve()}'")
    else:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")

    codes = ",".join(f"'{code}'" for code in TARGETS)
    metadata = connection.execute(
        f"""
        SELECT *
        FROM read_parquet('{metadata_path.resolve()}')
        WHERE nsn_code IN ({codes})
        ORDER BY nsn_code
        """
    ).fetchdf()
    metadata.to_csv(data_dir / "candidate_metadata.csv", index=False)

    print(f"Fetching full profiles for {len(TARGETS)} compounds", flush=True)
    differential = connection.execute(
        f"""
        SELECT pert_id, upper(genes) AS gene_symbol, logFC
        FROM read_parquet('{DE_URL}')
        WHERE split_part(pert_id, '_', 1) IN ({codes})
        ORDER BY pert_id, gene_symbol
        """
    ).fetchdf()
    connection.close()
    if differential.empty:
        raise RuntimeError("NIBR DRUG-seq query returned no rows")
    differential.to_csv(data_dir / "targeted_full_de.csv.gz", index=False, compression="gzip")

    counts = differential.groupby("pert_id").size().rename("n_genes").reset_index()
    counts.to_csv(data_dir / "profile_gene_counts.csv", index=False)
    manifest = pd.DataFrame([
        {
            "resource": "NIBR_DRUGseq_U2OS_MoABox_full_DE",
            "url": DE_URL,
            "local_file": "targeted_full_de.csv.gz",
            "sha256": sha256(data_dir / "targeted_full_de.csv.gz"),
            "selection": ";".join(TARGETS),
            "cell_line": "U2OS",
            "time_h": 24,
            "n_rows": len(differential),
            "n_profiles": differential["pert_id"].nunique(),
        },
        {
            "resource": "NIBR_DRUGseq_drug_metadata",
            "url": "https://s3.perturbseqr.maayanlab.cloud/data/nibr-drug-seq-drug_metadata.parquet",
            "local_file": "candidate_metadata.csv",
            "sha256": sha256(data_dir / "candidate_metadata.csv"),
            "selection": ";".join(TARGETS),
            "cell_line": "not_applicable",
            "time_h": np.nan,
            "n_rows": len(metadata),
            "n_profiles": np.nan,
        },
    ])
    manifest.to_csv(data_dir / "source_manifest.csv", index=False)
    print(counts.to_string(index=False), flush=True)


def rank_normal(values: pd.Series) -> np.ndarray:
    rank = stats.rankdata(values.to_numpy(float), method="average")
    return stats.norm.ppf((rank - 0.5) / len(rank))


def permutation_score(
    rank_z: np.ndarray,
    weights: np.ndarray,
    observed_indices: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    observed = float(np.sum(rank_z[observed_indices] * weights) / np.sum(np.abs(weights)))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        sampled = rng.choice(len(rank_z), size=len(observed_indices), replace=False)
        null[index] = np.sum(rank_z[sampled] * weights) / np.sum(np.abs(weights))
    support_p = (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
    harm_p = (1 + np.count_nonzero(null <= observed)) / (permutations + 1)
    z_score = (observed - null.mean()) / null.std(ddof=1)
    return observed, float(support_p), float(harm_p), float(z_score)


def load_endpoints(root: Path) -> dict[str, dict[str, float]]:
    signature_dir = root / "results/transition_aware_drug_screen/signatures"
    signature = pd.read_csv(signature_dir / "inflammatory_activation_signature.csv")
    signature["gene"] = signature["gene"].astype(str).str.upper()
    lock = dict(zip(signature["gene"], -signature["meta_z"]))

    def genes(name: str) -> list[str]:
        return [
            line.strip().upper()
            for line in (signature_dir / name).read_text().splitlines()
            if line.strip()
        ]

    axis2 = {gene: -1.0 for gene in genes("axis2.txt")}
    lam2 = {gene: 1.0 for gene in genes("lam2_complement_apc.txt")}
    trajectory = {gene: 1.0 for gene in genes("trajectory_positive.txt")}
    trajectory.update({gene: -1.0 for gene in genes("trajectory_negative.txt")})
    lock_high = {
        gene: -float(signature.loc[signature["gene"].eq(gene), "meta_z"].iloc[0])
        for gene in genes("lock_high.txt")
    }
    return {
        "lock_weighted_reversal": lock,
        "lock_high_suppression": lock_high,
        "axis2_suppression": axis2,
        "lam2_adaptation": lam2,
        "trajectory_adaptation": trajectory,
    }


def parse_pert_id(pert_id: str) -> tuple[str, str, float, float]:
    code, dose, time = pert_id.rsplit("_", 2)
    return code, TARGETS[code], float(dose), float(time)


def score_profiles(
    differential: pd.DataFrame,
    endpoints: dict[str, dict[str, float]],
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    gene_rows = []
    for profile_number, (pert_id, raw) in enumerate(differential.groupby("pert_id", sort=True)):
        code, drug, dose, time = parse_pert_id(pert_id)
        profile = raw.groupby("gene_symbol", as_index=False)["logFC"].mean()
        profile["rank_z"] = rank_normal(profile["logFC"])
        lookup = {gene: index for index, gene in enumerate(profile["gene_symbol"])}
        rank_z = profile["rank_z"].to_numpy(float)
        row = {
            "pert_id": pert_id,
            "drug_code": code,
            "drug": drug,
            "dose_um": dose,
            "time_h": time,
            "n_profile_genes": len(profile),
        }
        for endpoint_number, (endpoint, definition) in enumerate(endpoints.items()):
            available = [gene for gene in definition if gene in lookup]
            indices = np.asarray([lookup[gene] for gene in available], dtype=int)
            weights = np.asarray([definition[gene] for gene in available], dtype=float)
            score, support_p, harm_p, z_score = permutation_score(
                rank_z,
                weights,
                indices,
                permutations,
                seed + profile_number * 100 + endpoint_number,
            )
            row[f"{endpoint}_n_genes"] = len(available)
            row[f"{endpoint}_score"] = score
            row[f"{endpoint}_support_p"] = support_p
            row[f"{endpoint}_harm_p"] = harm_p
            row[f"{endpoint}_permutation_z"] = z_score
            selected = profile.iloc[indices].copy()
            selected.insert(0, "pert_id", pert_id)
            selected.insert(1, "drug", drug)
            selected.insert(2, "dose_um", dose)
            selected.insert(3, "endpoint", endpoint)
            selected["endpoint_weight"] = weights
            selected["weighted_directional_value"] = selected["rank_z"] * weights
            gene_rows.append(selected)
        for gene in ["STAT1", "IRF1", "PPARG", "MAF", "MAFB", "VDR", "JAK1", "JAK2", "TYK2"]:
            match = profile.loc[profile["gene_symbol"].eq(gene)]
            row[f"{gene}_logFC"] = float(match["logFC"].iloc[0]) if len(match) else np.nan
            row[f"{gene}_rank_z"] = float(match["rank_z"].iloc[0]) if len(match) else np.nan
        score_rows.append(row)
    scores = pd.DataFrame(score_rows).sort_values(["drug", "dose_um"]).reset_index(drop=True)
    for endpoint in endpoints:
        for direction in ["support", "harm"]:
            p_column = f"{endpoint}_{direction}_p"
            scores[f"{p_column}_fdr"] = multipletests(scores[p_column], method="fdr_bh")[1]
    scores["adaptation_harm_veto"] = (
        scores["lam2_adaptation_harm_p_fdr"].lt(0.10)
        | scores["trajectory_adaptation_harm_p_fdr"].lt(0.10)
    )
    scores["profile_advance"] = (
        scores["lock_weighted_reversal_support_p_fdr"].lt(0.10)
        & scores["lock_high_suppression_score"].gt(0)
        & ~scores["adaptation_harm_veto"]
    )
    return scores, pd.concat(gene_rows, ignore_index=True)


def dose_consistency(differential: pd.DataFrame) -> pd.DataFrame:
    rows = []
    differential = differential.copy()
    differential[["drug_code", "dose", "time"]] = differential["pert_id"].str.rsplit("_", n=2, expand=True)
    differential["dose"] = differential["dose"].astype(float)
    for code, compound in differential.groupby("drug_code"):
        profiles = {
            dose: frame.groupby("gene_symbol")["logFC"].mean()
            for dose, frame in compound.groupby("dose")
        }
        doses = sorted(profiles)
        for first, dose_a in enumerate(doses):
            for dose_b in doses[first + 1:]:
                match = pd.concat([profiles[dose_a], profiles[dose_b]], axis=1, join="inner").dropna()
                match.columns = ["a", "b"]
                rho = stats.spearmanr(match["a"], match["b"])
                rows.append({
                    "drug_code": code,
                    "drug": TARGETS[code],
                    "dose_a_um": dose_a,
                    "dose_b_um": dose_b,
                    "n_shared_genes": len(match),
                    "transcriptome_spearman": rho.statistic,
                    "spearman_p": rho.pvalue,
                })
    return pd.DataFrame(rows)


def compound_summary(scores: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drug, frame in scores.groupby("drug"):
        frame = frame.sort_values("dose_um")
        dose_rho = stats.spearmanr(
            np.log10(frame["dose_um"]), frame["lock_weighted_reversal_score"]
        )
        correlations = consistency.loc[
            consistency["drug"].eq(drug), "transcriptome_spearman"
        ]
        effective = frame.loc[frame["lock_weighted_reversal_score"].gt(0)]
        rows.append({
            "drug": drug,
            "drug_code": frame["drug_code"].iloc[0],
            "n_doses": len(frame),
            "n_positive_lock_reversal_doses": int(frame["lock_weighted_reversal_score"].gt(0).sum()),
            "n_lock_reversal_fdr_lt_0_10_doses": int(frame["lock_weighted_reversal_support_p_fdr"].lt(0.10).sum()),
            "n_profile_advance_doses": int(frame["profile_advance"].sum()),
            "median_lock_reversal_score": frame["lock_weighted_reversal_score"].median(),
            "best_lock_reversal_score": frame["lock_weighted_reversal_score"].max(),
            "median_axis2_suppression_score": frame["axis2_suppression_score"].median(),
            "median_lam2_adaptation_score": frame["lam2_adaptation_score"].median(),
            "median_trajectory_adaptation_score": frame["trajectory_adaptation_score"].median(),
            "effective_dose_adaptation_veto_fraction": (
                float(effective["adaptation_harm_veto"].mean()) if len(effective) else np.nan
            ),
            "median_pairwise_transcriptome_spearman": correlations.median(),
            "dose_trend_lock_reversal_spearman": dose_rho.statistic,
            "dose_trend_lock_reversal_p": dose_rho.pvalue,
            "compound_advance": bool(frame["profile_advance"].sum() >= 1 and frame["lock_weighted_reversal_score"].gt(0).sum() >= 2),
        })
    return pd.DataFrame(rows).sort_values(
        ["compound_advance", "n_profile_advance_doses", "best_lock_reversal_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Targeted NIBR DRUG-seq transition-aware screen",
        "",
        "## Design",
        "",
        "Full U2OS 24-hour differential-expression profiles were scored at 0.01, 0.1, 1, and 10 uM. The primary score is the frozen three-cohort inflammatory-lock signature with signed meta-z weights. Axis2 suppression, LAM2 adaptation, and the frozen trajectory program are independently permuted; significant negative LAM2 or trajectory effects veto a dose.",
        "",
        "## Compound decisions",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.drug}: **{'advance' if row.compound_advance else 'do not advance'}**; "
            f"positive lock-reversal doses={row.n_positive_lock_reversal_doses}/{row.n_doses}, "
            f"FDR-supported doses={row.n_lock_reversal_fdr_lt_0_10_doses}, "
            f"non-vetoed advancing doses={row.n_profile_advance_doses}; "
            f"median Axis2 suppression={row.median_axis2_suppression_score:.3f}, "
            f"median LAM2 adaptation={row.median_lam2_adaptation_score:.3f}."
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "U2OS profiles test whether a compound can execute the desired transcriptional direction in a shared perturbational system. They do not establish CSF myeloid efficacy, CNS exposure, or target causality. Those claims require target annotation, quantitative bioactivity, and disease-relevant validation.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    data_dir = root / "data/external_validation/NIBR_DRUGseq_transition_candidates"
    output = root / "results/transition_aware_drug_screen/nibr_full_profiles"
    output.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch_profiles(root, data_dir, args.httpfs_extension)
    differential_path = data_dir / "targeted_full_de.csv.gz"
    if not differential_path.exists():
        raise FileNotFoundError("Run with --fetch to download the targeted full profiles")

    differential = pd.read_csv(differential_path)
    endpoints = load_endpoints(root)
    scores, gene_scores = score_profiles(
        differential, endpoints, args.permutations, args.seed
    )
    consistency = dose_consistency(differential)
    summary = compound_summary(scores, consistency)
    scores.to_csv(output / "NIBR_transition_candidate_profile_scores.csv", index=False)
    gene_scores.to_csv(
        output / "NIBR_transition_candidate_endpoint_gene_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    consistency.to_csv(output / "NIBR_transition_candidate_dose_consistency.csv", index=False)
    summary.to_csv(output / "NIBR_transition_candidate_compound_summary.csv", index=False)
    write_report(summary, output / "NIBR_TRANSITION_CANDIDATE_REPORT.md")
    print(summary.to_string(index=False), flush=True)
    print("\nDose-level scores", flush=True)
    print(scores[[
        "drug", "dose_um", "lock_weighted_reversal_score",
        "lock_weighted_reversal_support_p_fdr", "axis2_suppression_score",
        "lam2_adaptation_score", "trajectory_adaptation_score",
        "adaptation_harm_veto", "profile_advance",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
