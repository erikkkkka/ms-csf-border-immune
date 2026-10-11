#!/usr/bin/env python3
"""Deconvolve drug targets with independent public loss-of-function signatures."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "PDPK1": "BX-912 annotated target",
    "AKT2": "BX-912 annotated target",
    "CDK2": "BX-912 annotated target",
    "CHEK1": "BX-912 annotated target",
    "GSK3B": "BX-912 annotated target",
    "KDR": "BX-912 annotated target",
    "AURKA": "BX-912 high-affinity ChEMBL off-target",
    "MARK3": "BX-912 measured ChEMBL off-target",
    "MARK4": "BX-912 measured ChEMBL off-target",
    "TBK1": "BX-912 measured ChEMBL off-target",
    "STAT1": "validated execution-anchor positive control",
    "IRF1": "validated execution-node positive control",
    "JAK1": "IFNGR relay comparator",
    "JAK2": "IFNGR relay comparator",
    "TYK2": "JAK-family comparator",
    "PPARG": "terminal-adaptation comparator",
    "VDR": "nuclear-receptor comparator",
}

LIBRARIES = {
    "LINCS_L1000_XPR": "data/reference/perturbseqr/lincs-l1000-xpr.gmt.gz",
    "Replogle_CRISPRi": "data/reference/perturbseqr/replogle-perturb-seq.gmt.gz",
    "PerturbAtlas_human": "data/reference/perturbseqr/perturb-atlas-human.gmt.gz",
    "CREEDS": "data/reference/perturbseqr/creeds-gene.gmt.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def read_list(path: Path) -> set[str]:
    return {line.strip().upper() for line in path.read_text().splitlines() if line.strip()}


def load_signatures(root: Path) -> dict[str, set[str]]:
    path = root / "results/transition_aware_drug_screen/signatures"
    return {
        "lock_high": read_list(path / "lock_high.txt"),
        "lock_low": read_list(path / "lock_low.txt"),
        "axis2": read_list(path / "axis2.txt"),
        "lam2": read_list(path / "lam2_complement_apc.txt"),
        "trajectory_positive": read_list(path / "trajectory_positive.txt"),
        "trajectory_negative": read_list(path / "trajectory_negative.txt"),
    }


def parse_term(term: str) -> tuple[str, str, str, bool]:
    if term.endswith(" up"):
        base, direction = term[:-3], "up"
    elif term.endswith(" down"):
        base, direction = term[:-5], "down"
    else:
        return term, "", "", False
    fields = base.split("::")
    target = fields[0].upper()
    upper_fields = {field.upper() for field in fields}
    loss_of_function = bool(upper_fields & {"KO", "KD", "CRISPRI"})
    return base, direction, target, loss_of_function


def load_library(path: Path) -> tuple[dict[str, dict[str, set[str]]], set[str]]:
    pairs: dict[str, dict[str, set[str]]] = {}
    universe: set[str] = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            genes = {gene.strip().upper() for gene in fields[2:] if gene.strip()}
            universe.update(genes)
            base, direction, target, loss_of_function = parse_term(fields[0])
            if target not in TARGETS or not loss_of_function or not direction:
                continue
            record = pairs.setdefault(base, {"target": target})
            record[direction] = genes
    complete = {
        key: value for key, value in pairs.items()
        if "up" in value and "down" in value
    }
    return complete, universe


def hypergeom(query: set[str], observed: set[str], universe: set[str]) -> tuple[int, float, str]:
    query = query & universe
    observed = observed & universe
    overlap = sorted(query & observed)
    if not query or not observed:
        return 0, 1.0, ""
    p_value = stats.hypergeom.sf(len(overlap) - 1, len(universe), len(observed), len(query))
    return len(overlap), float(p_value), ";".join(overlap)


def fisher(values: list[float]) -> float:
    values = np.clip(values, np.finfo(float).tiny, 1.0)
    return float(stats.combine_pvalues(values, method="fisher")[1])


def simes(values: pd.Series) -> float:
    p_value = np.sort(values.to_numpy(float))
    return float(min(1.0, np.min(p_value * len(p_value) / np.arange(1, len(p_value) + 1))))


def score_pair(
    library: str,
    perturbation: str,
    target: str,
    up: set[str],
    down: set[str],
    universe: set[str],
    signatures: dict[str, set[str]],
) -> dict:
    tests = {}
    definitions = {
        "lock_high_down": (signatures["lock_high"], down),
        "lock_low_up": (signatures["lock_low"], up),
        "lock_high_up": (signatures["lock_high"], up),
        "lock_low_down": (signatures["lock_low"], down),
        "axis2_down": (signatures["axis2"], down),
        "axis2_up": (signatures["axis2"], up),
        "lam2_down": (signatures["lam2"], down),
        "lam2_up": (signatures["lam2"], up),
        "trajectory_positive_up": (signatures["trajectory_positive"], up),
        "trajectory_negative_down": (signatures["trajectory_negative"], down),
        "trajectory_positive_down": (signatures["trajectory_positive"], down),
        "trajectory_negative_up": (signatures["trajectory_negative"], up),
    }
    for name, (query, observed) in definitions.items():
        count, p_value, genes = hypergeom(query, observed, universe)
        tests[f"n_{name}"] = count
        tests[f"p_{name}"] = p_value
        tests[f"genes_{name}"] = genes

    lock_reversal_p = fisher([tests["p_lock_high_down"], tests["p_lock_low_up"]])
    lock_mimicry_p = fisher([tests["p_lock_high_up"], tests["p_lock_low_down"]])
    trajectory_support_p = fisher([
        tests["p_trajectory_positive_up"], tests["p_trajectory_negative_down"]
    ])
    trajectory_harm_p = fisher([
        tests["p_trajectory_positive_down"], tests["p_trajectory_negative_up"]
    ])
    lock_direction_score = 0.5 * (
        (tests["n_lock_high_down"] - tests["n_lock_high_up"])
        / max(len(signatures["lock_high"] & universe), 1)
        + (tests["n_lock_low_up"] - tests["n_lock_low_down"])
        / max(len(signatures["lock_low"] & universe), 1)
    )
    return {
        "library": library,
        "target": target,
        "target_role": TARGETS[target],
        "perturbation_id": perturbation,
        "n_perturbation_up": len(up),
        "n_perturbation_down": len(down),
        "lock_reversal_p": lock_reversal_p,
        "lock_mimicry_p": lock_mimicry_p,
        "lock_direction_score": lock_direction_score,
        "axis2_net_suppression": tests["n_axis2_down"] - tests["n_axis2_up"],
        "lam2_net_preservation": tests["n_lam2_up"] - tests["n_lam2_down"],
        "trajectory_support_p": trajectory_support_p,
        "trajectory_harm_p": trajectory_harm_p,
        **tests,
    }


def exact_sign_flip(values: np.ndarray, alternative: str = "greater") -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return 1.0
    observed = float(values.mean())
    if n <= 18:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        null = (signs * values[None, :]).mean(axis=1)
        if alternative == "greater":
            return float(np.mean(null >= observed - 1e-15))
        return float(np.mean(null <= observed + 1e-15))
    rng = np.random.default_rng(20260813 + n)
    signs = rng.choice([-1.0, 1.0], size=(100000, n))
    null = (signs * values[None, :]).mean(axis=1)
    if alternative == "greater":
        return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
    return float((1 + np.count_nonzero(null <= observed)) / (len(null) + 1))


def summarize_targets(pair_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (library, target), frame in pair_scores.groupby(["library", "target"], sort=False):
        direction = frame["lock_direction_score"].to_numpy(float)
        reversal_simes = simes(frame["lock_reversal_p"])
        rows.append({
            "library": library,
            "target": target,
            "target_role": TARGETS[target],
            "n_paired_signatures": len(frame),
            "n_positive_direction": int(np.count_nonzero(direction > 0)),
            "positive_direction_fraction": float(np.mean(direction > 0)),
            "mean_lock_direction_score": float(direction.mean()),
            "median_lock_direction_score": float(np.median(direction)),
            "lock_direction_sign_flip_p": exact_sign_flip(direction, "greater"),
            "target_lock_reversal_simes_p": reversal_simes,
            "n_signature_lock_fdr_candidates": int(frame["lock_reversal_p"].lt(0.05).sum()),
            "mean_axis2_net_suppression": frame["axis2_net_suppression"].mean(),
            "mean_lam2_net_preservation": frame["lam2_net_preservation"].mean(),
            "n_lam2_harm_enrichment_p_lt_0_05": int(
                ((frame["p_lam2_down"] < 0.05) & (frame["lam2_net_preservation"] < 0)).sum()
            ),
            "best_perturbation_id": frame.sort_values("lock_reversal_p")["perturbation_id"].iloc[0],
            "best_lock_reversal_p": frame["lock_reversal_p"].min(),
        })
    summary = pd.DataFrame(rows)
    summary["target_lock_reversal_simes_fdr_within_library"] = np.nan
    summary["lock_direction_sign_flip_fdr_within_library"] = np.nan
    for library, positions in summary.groupby("library").groups.items():
        index = np.asarray(list(positions), dtype=int)
        summary.loc[index, "target_lock_reversal_simes_fdr_within_library"] = multipletests(
            summary.loc[index, "target_lock_reversal_simes_p"], method="fdr_bh"
        )[1]
        summary.loc[index, "lock_direction_sign_flip_fdr_within_library"] = multipletests(
            summary.loc[index, "lock_direction_sign_flip_p"], method="fdr_bh"
        )[1]
    summary["library_target_advance"] = (
        summary["target_lock_reversal_simes_fdr_within_library"].lt(0.10)
        & summary["mean_lock_direction_score"].gt(0)
        & summary["positive_direction_fraction"].ge(0.60)
        & summary["n_lam2_harm_enrichment_p_lt_0_05"].eq(0)
    )
    return summary.sort_values(
        ["library_target_advance", "target_lock_reversal_simes_p", "mean_lock_direction_score"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def cross_library_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        frame = summary.loc[summary["target"].eq(target)].copy()
        if frame.empty:
            rows.append({
                "target": target,
                "target_role": TARGETS[target],
                "n_libraries_with_paired_signatures": 0,
                "n_libraries_positive_mean_direction": 0,
                "n_libraries_advance": 0,
                "primary_XPR_advance": False,
                "secondary_library_advance": False,
                "decision": "not_testable",
            })
            continue
        primary = frame.loc[frame["library"].eq("LINCS_L1000_XPR")]
        primary_advance = bool(primary["library_target_advance"].any())
        secondary_advance = bool(
            frame.loc[~frame["library"].eq("LINCS_L1000_XPR"), "library_target_advance"].any()
        )
        if primary_advance and secondary_advance:
            decision = "advance_cross_library_genetic_support"
        elif primary_advance:
            decision = "advance_primary_genetic_support"
        elif secondary_advance:
            decision = "secondary_only_requires_primary_replication"
        else:
            decision = "no_genetic_support"
        rows.append({
            "target": target,
            "target_role": TARGETS[target],
            "n_libraries_with_paired_signatures": len(frame),
            "n_libraries_positive_mean_direction": int(frame["mean_lock_direction_score"].gt(0).sum()),
            "n_libraries_advance": int(frame["library_target_advance"].sum()),
            "primary_XPR_advance": primary_advance,
            "secondary_library_advance": secondary_advance,
            "decision": decision,
        })
    return pd.DataFrame(rows).sort_values(
        ["primary_XPR_advance", "secondary_library_advance", "n_libraries_positive_mean_direction"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_report(library_summary: pd.DataFrame, cross: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Genetic target deconvolution",
        "",
        "## Design",
        "",
        "Public paired loss-of-function signatures were scored against the frozen inflammatory-lock signature. Target-level evidence requires reversal enrichment, positive direction in at least 60% of contexts, and no LAM2-down enrichment. LINCS L1000 CRISPR-Cas9 is the primary library; CRISPRi, PerturbAtlas, and CREEDS are independent sensitivity layers.",
        "",
        "## Cross-library decisions",
        "",
    ]
    for row in cross.itertuples(index=False):
        lines.append(
            f"- {row.target}: **{row.decision}**; paired libraries={row.n_libraries_with_paired_signatures}, "
            f"positive-mean libraries={row.n_libraries_positive_mean_direction}, advancing libraries={row.n_libraries_advance}."
        )
    lines.extend(["", "## Advancing library-target pairs", ""])
    advancing = library_summary.loc[library_summary["library_target_advance"]]
    if advancing.empty:
        lines.append("- None.")
    else:
        for row in advancing.itertuples(index=False):
            lines.append(
                f"- {row.library} / {row.target}: {row.n_positive_direction}/{row.n_paired_signatures} positive contexts; "
                f"Simes FDR={row.target_lock_reversal_simes_fdr_within_library:.3g}; "
                f"mean direction score={row.mean_lock_direction_score:.4f}."
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These signatures are mostly non-myeloid and establish directionality, not CSF target efficacy. A drug-target claim advances only when pharmacologic phenotype, genetic perturbation, and quantitative target activity converge.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/transition_aware_drug_screen/genetic_deconvolution"
    output.mkdir(parents=True, exist_ok=True)
    signatures = load_signatures(root)

    rows = []
    coverage_rows = []
    for library, relative_path in LIBRARIES.items():
        pairs, universe = load_library(root / relative_path)
        target_counts = pd.Series(
            [record["target"] for record in pairs.values()], dtype="object"
        ).value_counts()
        for target in TARGETS:
            coverage_rows.append({
                "library": library,
                "target": target,
                "n_paired_loss_of_function_signatures": int(target_counts.get(target, 0)),
                "universe_genes": len(universe),
            })
        for perturbation, record in pairs.items():
            rows.append(score_pair(
                library, perturbation, record["target"], record["up"], record["down"],
                universe, signatures,
            ))
    pair_scores = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    if pair_scores.empty:
        raise RuntimeError("No complete target loss-of-function signatures found")
    pair_scores["lock_reversal_fdr_within_library"] = np.nan
    for library, positions in pair_scores.groupby("library").groups.items():
        index = np.asarray(list(positions), dtype=int)
        pair_scores.loc[index, "lock_reversal_fdr_within_library"] = multipletests(
            pair_scores.loc[index, "lock_reversal_p"], method="fdr_bh"
        )[1]
    library_summary = summarize_targets(pair_scores)
    cross = cross_library_summary(library_summary)

    coverage.to_csv(output / "genetic_target_coverage.csv", index=False)
    pair_scores.to_csv(output / "genetic_target_signature_scores.csv.gz", index=False, compression="gzip")
    library_summary.to_csv(output / "genetic_target_library_summary.csv", index=False)
    cross.to_csv(output / "genetic_target_cross_library_decisions.csv", index=False)
    write_report(library_summary, cross, output / "GENETIC_TARGET_DECONVOLUTION_REPORT.md")
    print(cross.to_string(index=False), flush=True)
    print("\nLibrary-target results", flush=True)
    print(library_summary[[
        "library", "target", "n_paired_signatures", "n_positive_direction",
        "mean_lock_direction_score", "target_lock_reversal_simes_p",
        "target_lock_reversal_simes_fdr_within_library", "mean_axis2_net_suppression",
        "mean_lam2_net_preservation", "library_target_advance",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
