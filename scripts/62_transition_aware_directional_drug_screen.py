#!/usr/bin/env python3
"""Screen public directional drug signatures for lock reversal with adaptation vetoes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = {
    "NIBR_DRUGseq_2025": ("NIBR_DRUGseq_2025_up.gmt", "NIBR_DRUGseq_2025_down.gmt"),
    "Drug_Perturbations_from_GEO": (
        "Drug_Perturbations_from_GEO_up.gmt", "Drug_Perturbations_from_GEO_down.gmt"
    ),
    "Old_CMAP": ("Old_CMAP_up.gmt", "Old_CMAP_down.gmt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def read_list(path: Path) -> set[str]:
    return {line.strip().upper() for line in path.read_text().splitlines() if line.strip()}


def read_gmt(path: Path) -> dict[str, set[str]]:
    terms: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3 and fields[0].strip():
                terms[fields[0]] = {gene.strip().upper() for gene in fields[2:] if gene.strip()}
    return terms


def parse_context(term: str, library: str) -> dict:
    if library == "NIBR_DRUGseq_2025":
        parts = term.split()
        cid = parts[0].upper()
        dose = parts[-1].upper() if parts else ""
        baid_index = next((i for i, value in enumerate(parts) if value.upper().startswith("BAID")), None)
        code_index = baid_index - 1 if baid_index is not None and baid_index >= 1 else None
        code = parts[code_index] if code_index is not None else ""
        label = " ".join(parts[1:code_index]) if code_index is not None else " ".join(parts[1:])
        return {
            "drug_key": cid,
            "drug_label": label or cid,
            "perturbation_code": code,
            "dose_or_context": dose,
            "pubchem_cid": re.sub(r"^CID", "", cid),
        }
    if library == "Drug_Perturbations_from_GEO":
        match = re.match(r"(.+?)\s+(DB\d+)\s+", term, flags=re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            return {
                "drug_key": f"{label.lower()}|{match.group(2).upper()}",
                "drug_label": label,
                "perturbation_code": match.group(2).upper(),
                "dose_or_context": term,
                "pubchem_cid": "",
            }
        label = term.split()[0]
        return {
            "drug_key": label.lower(), "drug_label": label,
            "perturbation_code": "", "dose_or_context": term, "pubchem_cid": "",
        }
    base = re.sub(r"-\d+$", "", term).strip()
    return {
        "drug_key": base.lower(), "drug_label": base,
        "perturbation_code": "", "dose_or_context": term, "pubchem_cid": "",
    }


def enrich(query: set[str], observed: set[str], universe: set[str]) -> tuple[int, float, str]:
    query = query & universe
    observed = observed & universe
    overlap = sorted(query & observed)
    if not query or not observed:
        return 0, 1.0, ""
    p_value = stats.hypergeom.sf(len(overlap) - 1, len(universe), len(observed), len(query))
    return len(overlap), float(p_value), ";".join(overlap)


def fisher(values: list[float]) -> float:
    return float(stats.combine_pvalues(np.clip(values, np.finfo(float).tiny, 1.0), method="fisher")[1])


def simes(values: pd.Series) -> float:
    p = np.sort(values.to_numpy(float))
    return float(min(1.0, np.min(p * len(p) / np.arange(1, len(p) + 1))))


def score_library(
    library: str,
    up_path: Path,
    down_path: Path,
    signatures: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drug_up = read_gmt(up_path)
    drug_down = read_gmt(down_path)
    terms = sorted(set(drug_up) & set(drug_down))
    universe = set().union(*(drug_up[term] | drug_down[term] for term in terms))
    rows = []
    for term in terms:
        context = parse_context(term, library)
        tests = {}
        definitions = {
            "lock_high_down": (signatures["lock_high"], drug_down[term]),
            "lock_low_up": (signatures["lock_low"], drug_up[term]),
            "lock_high_up": (signatures["lock_high"], drug_up[term]),
            "lock_low_down": (signatures["lock_low"], drug_down[term]),
            "axis2_down": (signatures["axis2"], drug_down[term]),
            "axis2_up": (signatures["axis2"], drug_up[term]),
            "lam2_up": (signatures["lam2"], drug_up[term]),
            "lam2_down": (signatures["lam2"], drug_down[term]),
            "trajectory_positive_up": (signatures["trajectory_positive"], drug_up[term]),
            "trajectory_negative_down": (signatures["trajectory_negative"], drug_down[term]),
            "trajectory_positive_down": (signatures["trajectory_positive"], drug_down[term]),
            "trajectory_negative_up": (signatures["trajectory_negative"], drug_up[term]),
        }
        for name, (query, observed) in definitions.items():
            count, p_value, genes = enrich(query, observed, universe)
            tests[f"n_{name}"] = count
            tests[f"p_{name}"] = p_value
            tests[f"genes_{name}"] = genes
        tests["lock_reversal_p"] = fisher([tests["p_lock_high_down"], tests["p_lock_low_up"]])
        tests["lock_mimicry_p"] = fisher([tests["p_lock_high_up"], tests["p_lock_low_down"]])
        tests["trajectory_support_p"] = fisher([
            tests["p_trajectory_positive_up"], tests["p_trajectory_negative_down"]
        ])
        tests["trajectory_harm_p"] = fisher([
            tests["p_trajectory_positive_down"], tests["p_trajectory_negative_up"]
        ])
        tests["lock_bidirectional_min_overlap"] = min(
            tests["n_lock_high_down"], tests["n_lock_low_up"]
        )
        tests["lam2_net_overlap"] = tests["n_lam2_up"] - tests["n_lam2_down"]
        tests["axis2_net_suppression"] = tests["n_axis2_down"] - tests["n_axis2_up"]
        rows.append({
            "library": library,
            "signature_term": term,
            "universe_genes": len(universe),
            "n_drug_up": len(drug_up[term]),
            "n_drug_down": len(drug_down[term]),
            **context,
            **tests,
        })
    contexts = pd.DataFrame(rows)
    for p_column in [
        "lock_reversal_p", "lock_mimicry_p", "p_axis2_down", "p_axis2_up",
        "p_lam2_up", "p_lam2_down", "trajectory_support_p", "trajectory_harm_p",
    ]:
        contexts[p_column.replace("_p", "_fdr") if p_column.endswith("_p") else f"{p_column}_fdr"] = (
            multipletests(contexts[p_column], method="fdr_bh")[1]
        )
    contexts["adaptation_harm_veto"] = (
        (contexts["p_lam2_down_fdr"] < 0.10) & (contexts["lam2_net_overlap"] < 0)
    ) | (
        (contexts["trajectory_harm_fdr"] < 0.10)
        & (contexts["trajectory_support_p"] > contexts["trajectory_harm_p"])
    )
    contexts["lock_directional_support"] = contexts["lock_bidirectional_min_overlap"] >= 2
    contexts["context_advance"] = (
        contexts["lock_reversal_fdr"].lt(0.10)
        & contexts["lock_directional_support"]
        & ~contexts["adaptation_harm_veto"]
    )
    contexts = contexts.sort_values(
        ["context_advance", "adaptation_harm_veto", "lock_reversal_p", "lock_bidirectional_min_overlap"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)

    compound_rows = []
    for drug_key, group in contexts.groupby("drug_key", sort=False):
        best = group.sort_values(
            ["adaptation_harm_veto", "lock_reversal_p", "lock_bidirectional_min_overlap"],
            ascending=[True, True, False],
        ).iloc[0]
        compound_rows.append({
            "library": library,
            "drug_key": drug_key,
            "drug_label": best["drug_label"],
            "pubchem_cid": best["pubchem_cid"],
            "n_paired_contexts": len(group),
            "n_contexts_lock_fdr_lt_0_10": int(group["lock_reversal_fdr"].lt(0.10).sum()),
            "n_contexts_advance": int(group["context_advance"].sum()),
            "compound_lock_simes_p": simes(group["lock_reversal_p"]),
            "best_context": best["signature_term"],
            "best_perturbation_code": best["perturbation_code"],
            "best_dose_or_context": best["dose_or_context"],
            "best_lock_reversal_p": best["lock_reversal_p"],
            "best_lock_reversal_fdr": best["lock_reversal_fdr"],
            "best_lock_bidirectional_min_overlap": best["lock_bidirectional_min_overlap"],
            "best_axis2_down_p": best["p_axis2_down"],
            "best_axis2_net_suppression": best["axis2_net_suppression"],
            "best_lam2_down_p": best["p_lam2_down"],
            "best_lam2_net_overlap": best["lam2_net_overlap"],
            "best_trajectory_support_p": best["trajectory_support_p"],
            "best_trajectory_harm_p": best["trajectory_harm_p"],
            "best_adaptation_harm_veto": best["adaptation_harm_veto"],
            "best_lock_high_down_genes": best["genes_lock_high_down"],
            "best_lock_low_up_genes": best["genes_lock_low_up"],
            "best_lam2_down_genes": best["genes_lam2_down"],
        })
    compounds = pd.DataFrame(compound_rows)
    compounds["compound_lock_fdr"] = multipletests(
        compounds["compound_lock_simes_p"], method="fdr_bh"
    )[1]
    compounds["compound_advance"] = (
        compounds["n_contexts_advance"].gt(0)
        & compounds["compound_lock_fdr"].lt(0.10)
        & ~compounds["best_adaptation_harm_veto"]
    )
    compounds = compounds.sort_values(
        ["compound_advance", "best_adaptation_harm_veto", "compound_lock_simes_p"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return contexts, compounds


def build_report(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Transition-aware directional drug screen",
        "",
        "## Frozen decision rule",
        "",
        "The primary phenotype is a three-cohort donor-paired inflammatory-lock signature with Axis2, LAM2, and trajectory genes held out. A treatment must reverse both lock directions and then pass independent LAM2/trajectory harm vetoes. Axis2 suppression is an orthogonal mechanism-support endpoint, not part of the primary reversal score.",
        "",
        "## Library summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.library}: {row.n_contexts:,} paired contexts, {row.n_compounds:,} compounds, "
            f"{row.n_contexts_advance:,} advancing contexts, {row.n_compounds_advance:,} advancing compounds."
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Hits are computational perturbational hypotheses. Advancement to the mechanistic tier requires independent target annotation, quantitative ChEMBL activity, and disease/CNS translatability review. Docking is reserved for a structurally tractable target after target-level convergence.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    signature_dir = root / "results/transition_aware_drug_screen/signatures"
    output = root / "results/transition_aware_drug_screen/directional_screen"
    output.mkdir(parents=True, exist_ok=True)
    signatures = {
        "lock_high": read_list(signature_dir / "lock_high.txt"),
        "lock_low": read_list(signature_dir / "lock_low.txt"),
        "axis2": read_list(signature_dir / "axis2.txt"),
        "lam2": read_list(signature_dir / "lam2_complement_apc.txt"),
        "trajectory_positive": read_list(signature_dir / "trajectory_positive.txt"),
        "trajectory_negative": read_list(signature_dir / "trajectory_negative.txt"),
    }

    all_contexts = []
    all_compounds = []
    summary_rows = []
    for library, (up_file, down_file) in LIBRARIES.items():
        contexts, compounds = score_library(
            library,
            root / "data/reference/enrichr" / up_file,
            root / "data/reference/enrichr" / down_file,
            signatures,
        )
        contexts.to_csv(output / f"{library}_context_scores.csv.gz", index=False, compression="gzip")
        compounds.to_csv(output / f"{library}_compound_scores.csv", index=False)
        all_contexts.append(contexts)
        all_compounds.append(compounds)
        summary_rows.append({
            "library": library,
            "n_contexts": len(contexts),
            "n_compounds": len(compounds),
            "n_contexts_lock_fdr_lt_0_10": int(contexts["lock_reversal_fdr"].lt(0.10).sum()),
            "n_contexts_harm_veto": int(contexts["adaptation_harm_veto"].sum()),
            "n_contexts_advance": int(contexts["context_advance"].sum()),
            "n_compounds_advance": int(compounds["compound_advance"].sum()),
        })
    combined_contexts = pd.concat(all_contexts, ignore_index=True)
    combined_compounds = pd.concat(all_compounds, ignore_index=True)
    combined_contexts.to_csv(output / "all_context_scores.csv.gz", index=False, compression="gzip")
    combined_compounds.to_csv(output / "all_compound_scores.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "directional_screen_summary.csv", index=False)
    build_report(summary, output / "TRANSITION_AWARE_DIRECTIONAL_DRUG_SCREEN_REPORT.md")
    print(summary.to_string(index=False), flush=True)
    print("\nTop non-vetoed compounds by library", flush=True)
    for library, frame in combined_compounds.groupby("library", sort=False):
        print(f"\n[{library}]", flush=True)
        columns = [
            "drug_label", "compound_lock_simes_p", "compound_lock_fdr",
            "best_lock_reversal_p", "best_lock_bidirectional_min_overlap",
            "best_axis2_net_suppression", "best_lam2_net_overlap", "compound_advance",
        ]
        print(frame.loc[~frame["best_adaptation_harm_veto"], columns].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
