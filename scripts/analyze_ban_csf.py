#!/usr/bin/env python3
"""Compare donor inflammatory indices and project IFNGR1 signatures in Ban CSF."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/external_validation/Ban"
BASELINE_CODE = ROOT / "scripts/ban_control_baseline.py"
MIN_CELLS = 20
SEED = 20261113
FILES = {
    "trajectory": "results/external_validation/GSE328265_two_axis_projection/frozen_trajectory_program.csv",
    "response": "results/in_vivo_crispr_bridge/published_cytokine_signature_ledger.csv",
    "knockout": "results/in_vivo_crispr_bridge/direct_ko_consensus_signature_ledger.csv",
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data, path):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def select_counts(source, rows):
    candidates = [(f"layers/{k}", source.layers[k], source.var) for k in ("counts", "raw_counts") if k in source.layers]
    if source.raw is not None:
        candidates.append(("raw/X", source.raw.X, source.raw.var))
    candidates.append(("X", source.X, source.var))
    scale_audit = []
    for name, matrix, features in candidates:
        sample = sparse.csr_matrix(matrix[rows[:128], :])
        integers = (sample.data >= 0).all() and np.allclose(sample.data, np.rint(sample.data), atol=1e-5, rtol=0)
        scale_audit.append({"layer": name, "sample_nonnegative_integer_counts": bool(integers), "n_features": len(features)})
        if integers and sample.nnz:
            counts = sparse.vstack([sparse.csr_matrix(matrix[part, :]) for part in np.array_split(rows, 12)], format="csr")
            counts.eliminate_zeros()
            if np.any(counts.data < 0) or not np.allclose(counts.data, np.rint(counts.data), atol=1e-5, rtol=0):
                raise ValueError(f"Count audit failed on full selected set: {name}")
            return counts, features.copy(), name, scale_audit
    raise ValueError(f"No audited unlogged UMI layer was available: {scale_audit}")


def prepare(matrix, published_obs, cache, raw_umi=False):
    source = ad.read_h5ad(matrix, backed="r")
    metadata = pd.read_csv(published_obs, dtype={"patient_id": str}).set_index("cell_id")
    if not source.obs_names.is_unique or not metadata.index.is_unique or set(source.obs_names) != set(metadata.index):
        raise ValueError("Full H5AD barcodes do not match the published annotation table")
    metadata = metadata.reindex(source.obs_names)
    for key in ("patient_id", "celltype", "celltype_finer", "batch", "disease", "treated"):
        if not source.obs[key].astype(str).str.lower().equals(metadata[key].astype(str).str.lower()):
            raise ValueError(f"H5AD annotation disagrees with the published table: {key}")
    selected = metadata.disease.isin(["MS", "NIND"]) & metadata.celltype.isin(["monocyte", "Macrophage"])
    rows = np.flatnonzero(selected.to_numpy())
    counts, var, layer, scale_audit = select_counts(source, rows)
    raw_counts = counts
    if not raw_umi:
        if "counts_soupx" not in source.layers:
            raise ValueError("The primary analysis requires the authors' counts_soupx layer")
        corrected = source.layers["counts_soupx"]
        counts = sparse.vstack([sparse.csr_matrix(corrected[part, :]) for part in np.array_split(rows, 12)], format="csr")
        counts.eliminate_zeros()
        if not np.isfinite(counts.data).all() or np.any(counts.data < 0) or not source.var_names.equals(var.index):
            raise ValueError("Invalid SoupX count layer")
        layer = "layers/counts_soupx"
        scale_audit.append({"layer": layer, "selected_values_finite_nonnegative": True, "n_features": len(var)})
    source.file.close()
    if not var.index.is_unique:
        raise ValueError("The count layer requires unique gene names")

    obs = metadata.iloc[rows].copy()
    obs["subject_id"] = obs.patient_id.astype(str)
    obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    obs["n_genes_by_counts"] = counts.getnnz(axis=1)
    mito = var.index.astype(str).str.upper().str.startswith("MT-")
    if not mito.any():
        raise ValueError("Mitochondrial genes are unavailable")
    raw_total = np.asarray(raw_counts.sum(axis=1)).ravel()
    obs["pct_mito_raw"] = np.asarray(raw_counts[:, mito].sum(axis=1)).ravel() / np.maximum(raw_total, 1) * 100
    keep = (obs.total_counts.gt(0) & obs.n_genes_by_counts.ge(200) & obs.pct_mito_raw.le(20)
            & (raw_counts.getnnz(axis=1) >= 200))

    unique_donor_fields = ["disease", "treated", "gender"]
    relevant = metadata.loc[metadata.disease.isin(["MS", "NIND"])]
    if relevant.groupby("patient_id")[unique_donor_fields].nunique().gt(1).any().any():
        raise ValueError("Conflicting clinical fields within a donor")
    coverage = relevant.groupby("patient_id")[unique_donor_fields].first()
    coverage["batches"] = relevant.groupby("patient_id").batch.agg(lambda s: ";".join(sorted(set(s.astype(str)))))
    coverage["author_cells"] = obs.groupby("subject_id").size().reindex(coverage.index, fill_value=0)
    coverage["retained_cells"] = obs[keep].groupby("subject_id").size().reindex(coverage.index, fill_value=0)
    coverage["eligible_ge20"] = coverage.retained_cells.ge(MIN_CELLS)
    coverage["primary_donor"] = coverage.eligible_ge20 & ~coverage.treated
    coverage.to_csv(OUTPUT / "donor_coverage.csv")

    counts = counts[keep.to_numpy()].astype(np.float32)
    obs = obs.loc[keep].copy()
    normalized = counts.multiply((1e4 / obs.total_counts.to_numpy())[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data).astype(np.float32)
    cells = ad.AnnData(normalized, obs=obs, var=var)
    cells.write_h5ad(cache, compression="gzip")
    audit = {
        "input_h5ad_sha256": sha256(matrix),
        "annotation_sha256": sha256(published_obs),
        "expression_count_layer": layer,
        "layer_audit": scale_audit,
        "raw_rows_MS_NIND_mono_macro": len(rows),
        "retained_rows_MS_NIND_mono_macro": cells.n_obs,
        "mitochondrial_features": int(mito.sum()),
    }
    write_json(audit, OUTPUT / "input_audit.json")
    return cells


def clinical_cohort(cells, project, baseline, projection):
    program = pd.read_csv(project / FILES["trajectory"])
    genes = sorted(set(sum(baseline.MODULES.values(), [])))
    missing = sorted(set(genes) - set(cells.var_names))
    if missing:
        raise ValueError(f"Inflammatory genes missing from Ban count layer: {missing}")
    if not program.gene.is_unique or program.training_direction.value_counts().to_dict() != {"positive": 40, "negative": 40}:
        raise ValueError("The original trajectory program is not 40 + 40 unique genes")
    usable = program[program.gene.isin(cells.var_names)]
    if len(usable) != len(program) and usable.training_direction.value_counts().min() < 36:
        raise ValueError(f"Too few original trajectory genes: {usable.training_direction.value_counts().to_dict()}")
    missing_trajectory = sorted(set(program.gene) - set(usable.gene))
    raw_score, pt = projection.program_score(cells, usable, np.ones(cells.n_obs, dtype=bool))

    cohort = baseline.Cohort.__new__(baseline.Cohort)
    cohort.name, cohort.external = "Ban", True
    cohort.obs = cells.obs
    cohort.x = projection.dense(cells[:, genes].X).astype(float)
    cohort.module_indices = [[genes.index(gene) for gene in module] for module in baseline.MODULES.values()]
    cohort.subjects, cohort.donor = np.unique(cells.obs.subject_id.astype(str), return_inverse=True)
    cohort.n = len(cohort.subjects)
    cohort.cell_counts = np.bincount(cohort.donor, minlength=cohort.n)
    cohort.eligible = cohort.cell_counts >= MIN_CELLS
    cohort.cell_rows = [np.flatnonzero(cohort.donor == i) for i in range(cohort.n)]
    cohort.case = np.array([cells.obs.disease.iloc[rows[0]] == "MS" for rows in cohort.cell_rows])
    cohort.pt = pt
    cohort.tech_raw = np.column_stack([np.log1p(cells.obs.total_counts), cells.obs.n_genes_by_counts])
    cohort.tech = (cohort.tech_raw - cohort.tech_raw.mean(axis=0)) / cohort.tech_raw.std(axis=0)
    is_macro = cells.obs.celltype.eq("Macrophage").to_numpy()
    early = raw_score[~is_macro]
    late = raw_score[is_macro]
    panel = sorted(set(usable.gene) | set(genes))
    detected = np.asarray(cells[:, panel].X.getnnz(axis=0)).ravel()
    topology = {
        "n_monocytes": int(len(early)),
        "n_macrophages": int(len(late)),
        "median_projected_monocytes": float(np.median(pt[~is_macro])),
        "median_projected_macrophages": float(np.median(pt[is_macro])),
        "macrophage_vs_monocyte_auc": float(stats.mannwhitneyu(late, early).statistic / (len(late) * len(early))),
        "missing_trajectory_genes": missing_trajectory,
        "zero_detection_genes": [gene for gene, count in zip(panel, detected) if count == 0],
    }
    return cohort, topology


def clinical_statistics(cohort, bootstrap_n=2000, crossfit=True):
    eligible = np.flatnonzero(cohort.eligible)
    cases = eligible[cohort.case[eligible]]
    controls = eligible[~cohort.case[eligible]]
    if min(len(cases), len(controls)) < 3:
        raise ValueError("Fewer than three eligible donors in a clinical group")
    multiplicity = cohort.eligible.astype(int)
    _, readings, thresholds, fit = cohort.fit(cohort.case, multiplicity, reference_check=True)
    scores = cohort.aggregate(readings, thresholds, eligible)
    diff, d = cohort.effect(scores, cohort.case)
    methods = [False, True] if crossfit else [False]
    observed = {False: (diff, d)}
    if crossfit:
        observed[True] = cohort.effect(cohort.score(cohort.case, multiplicity, crossfit=True)[0], cohort.case)

    possible = math.comb(len(eligible), len(controls))
    exact = possible <= 5000
    n = possible if exact else 9999
    rng = np.random.default_rng(SEED)
    choices = itertools.combinations(eligible, len(controls)) if exact else (rng.choice(eligible, len(controls), replace=False) for _ in range(n))
    extreme = np.zeros((len(methods), len(diff)), dtype=int)
    for choice in choices:
        labels = cohort.case.copy()
        labels[eligible] = True
        labels[list(choice)] = False
        for i, held_out in enumerate(methods):
            null, _ = cohort.effect(cohort.score(labels, multiplicity, crossfit=held_out)[0], labels)
            if not np.isfinite(null).all():
                raise ValueError("A permutation produced non-finite contrasts")
            extreme[i] += np.abs(null) >= np.abs(observed[held_out][0]) - 1e-12
    p = extreme / n if exact else (extreme + 1) / (n + 1)
    print(f"Clinical permutations finished: {n}, cross-fit={crossfit}", flush=True)

    draws = {m: [] for m in methods}
    rng = np.random.default_rng(SEED + 1)
    for _ in range(bootstrap_n):
        sampled = np.r_[rng.choice(cases, size=len(cases)), rng.choice(controls, size=len(controls))]
        weights = np.bincount(sampled, minlength=cohort.n)
        for held_out in methods:
            try:
                boot_scores = cohort.score(cohort.case, weights, crossfit=held_out)[0]
                estimates = np.asarray(cohort.effect(boot_scores, cohort.case, weights))
                if np.isfinite(estimates).all():
                    draws[held_out].append(estimates)
            except (ValueError, np.linalg.LinAlgError):
                continue
    contrasts = {}
    for i, held_out in enumerate(methods):
        if not draws[held_out]:
            raise ValueError("No valid donor bootstrap draw")
        cis = np.quantile(draws[held_out], [.025, .975], axis=0)
        contrasts["held_out_donor" if held_out else "original_refit"] = {
            "bootstrap_valid": len(draws[held_out]),
            "bootstrap_attempts": bootstrap_n,
            "readouts": {
                label: {
                    "mean_MS_minus_NIND": float(observed[held_out][0][j]),
                    "mean_difference_ci95": cis[:, 0, j].tolist(),
                    "cohen_d": float(observed[held_out][1][j]),
                    "cohen_d_ci95": cis[:, 1, j].tolist(),
                    "p_two_sided": float(p[i, j]),
                }
                for j, label in enumerate(("continuous", "fraction", "MHC_I_continuous", "MHC_I_fraction", "IFN_continuous", "IFN_fraction"))
            },
        }

    jackknife = []
    for donor in eligible:
        weights = multiplicity.copy()
        weights[donor] = 0
        scores_loo = cohort.score(cohort.case, weights)[0]
        delta, _ = cohort.effect(scores_loo, cohort.case, weights)
        jackknife.append(float(delta[0]))
    return {
        "n_MS": len(cases), "n_NIND": len(controls),
        "group_mean_index": {"MS": float(scores[cases, 0].mean()), "NIND": float(scores[controls, 0].mean())},
        "n_permutations": n, "exact_permutation": exact,
        "threshold": float(thresholds[0]), "baseline_fit": fit,
        "contrasts": contrasts, "donor_omission_range": [min(jackknife), max(jackknife)],
    }, scores, readings


def run_primary(cells, project, cache):
    baseline = load_module(BASELINE_CODE, "baseline")
    projection = load_module(project / "scripts/54_project_gse328265_two_axis.py", "projection")
    primary_cells = cells[~cells.obs.treated.to_numpy()].copy()
    cohort, topology = clinical_cohort(primary_cells, project, baseline, projection)
    print(f"Primary cohort: {sum(cohort.eligible & cohort.case)} MS, {sum(cohort.eligible & ~cohort.case)} NIND; topology={topology}", flush=True)
    summary, scores, readings = clinical_statistics(cohort)
    donor_scores = pd.DataFrame(scores, index=pd.Index(cohort.subjects, name="patient_id"), columns=baseline.READOUTS)
    donor_scores["disease"] = primary_cells.obs.groupby("subject_id", observed=True).disease.first()
    donor_scores["n_continuum_cells"] = cohort.cell_counts
    donor_scores["eligible"] = cohort.eligible
    donor_scores.to_csv(OUTPUT / "primary_donor_scores.csv")
    summary.update(topology=topology,
                   sources={"trajectory": sha256(project / FILES["trajectory"]),
                            "baseline_code": sha256(BASELINE_CODE),
                            "projection_code": sha256(project / "scripts/54_project_gse328265_two_axis.py")})
    write_json(summary, OUTPUT / "primary_results.json")
    primary_cells.obs["projected_differentiation"] = cohort.pt
    primary_cells.obs["activation_residual"] = readings[:, 0]
    primary_cells.obs["incomplete"] = cohort.pt < 0.8
    primary_cells.obs["decoupled"] = (cohort.pt < 0.8) & (readings[:, 0] > summary["threshold"])
    primary_cells.obs["donor_eligible"] = cohort.eligible[cohort.donor]
    primary_cells.write_h5ad(cache, compression="gzip")

    if cells.obs.treated.any():
        all_cohort, all_topology = clinical_cohort(cells, project, baseline, projection)
        all_summary, _, _ = clinical_statistics(all_cohort, crossfit=False)
        all_summary["topology"] = all_topology
        write_json(all_summary, OUTPUT / "including_treated_sensitivity.json")


def project_signatures(cells, project):
    bridge = load_module(project / "scripts/79_bridge_in_vivo_crispr_signatures.py", "bridge")
    signatures = {}
    for key in ("response", "knockout"):
        table = pd.read_csv(project / FILES[key], keep_default_na=False)
        table = table.loc[table.perturbation.eq("Ifngr1") & table.human_gene.ne("")]
        signatures[(table.signature_type.iloc[0], "Ifngr1")] = table
    holdouts = bridge.endpoint_holdouts(project)
    exclusion = {k: holdouts[k] for k in ("axis2", "pseudotime")}
    reference = cells.obs.disease.eq("NIND").to_numpy() & cells.obs.donor_eligible.to_numpy()
    if reference.sum() < 50:
        raise ValueError("Fewer than 50 eligible reference cells for the signatures")

    obs = cells.obs.copy()
    obs["subject_id"] = obs.subject_id.astype(str)
    obs["dataset"] = "Ban"
    obs["diagnosis"] = obs.disease
    obs["disease_group"] = obs.disease
    obs["disease_eligible"] = obs.disease.eq("MS") & obs.donor_eligible
    obs["pseudotime"] = obs.projected_differentiation
    availability, all_effects = [], []
    for strict in (False, True):
        scored = obs.copy()
        for (kind, gene), table in signatures.items():
            for endpoint, holdout in exclusion.items():
                if strict:
                    holdout = exclusion["axis2"] | exclusion["pseudotime"]
                selected = table[table.human_gene.isin(cells.var_names) & ~table.human_gene.isin(holdout)].drop_duplicates("human_gene")
                requested = selected.human_gene.tolist()
                positive, negative = int(selected.weight.gt(0).sum()), int(selected.weight.lt(0).sum())
                testable = len(selected) >= 20 if kind == "published_response" else min(positive, negative) >= 20
                info = {"signature_type": kind, "endpoint": endpoint, "strict_joint_holdout": strict,
                        "n_archived_mapped": int(table.human_gene.nunique()), "n_available": len(selected),
                        "n_positive": positive, "n_negative": negative, "testable": testable,
                        "genes": ";".join(requested), "baseline_r_squared": None}
                if testable:
                    expression = bridge.dense(cells[:, requested].X).astype(float)
                    mean, sd = bridge.weighted_mean_sd(expression[reference], obs.loc[reference, "subject_id"])
                    weights = selected.weight.to_numpy(float)
                    raw = ((expression - mean) / sd) @ weights / np.abs(weights).sum()
                    prefix = f"{kind}__{gene}__{endpoint}"
                    scored[f"score__{prefix}"] = raw
                    if endpoint == "axis2":
                        res, r2 = bridge.residualize_score(raw, obs, "pseudotime", reference)
                        scored[f"residual__{prefix}"] = res
                        info["baseline_r_squared"] = r2
                availability.append(info)
        effects = bridge.donor_effects(scored.loc[scored.donor_eligible], signatures)
        if not effects.empty:
            all_effects.append(effects.assign(strict_joint_holdout=strict))

    pd.DataFrame(availability).to_csv(OUTPUT / "signature_gene_coverage.csv", index=False)
    if not all_effects:
        write_json({"status": "no_testable_signatures"}, OUTPUT / "signature_results.json")
        return
    effects = pd.concat(all_effects, ignore_index=True)
    effects.to_csv(OUTPUT / "signature_donor_effects.csv", index=False)
    rows = []
    expected = {("direct_ko_consensus", "pseudotime_spearman"): 1,
                ("published_response", "axis2_spearman"): 1,
                ("direct_ko_consensus", "axis2_spearman"): -1,
                ("published_response", "decoupled_minus_other_incomplete"): 1,
                ("direct_ko_consensus", "decoupled_minus_other_incomplete"): -1}
    rng = np.random.default_rng(SEED + 2)
    for (strict, kind, endpoint), data in effects.groupby(["strict_joint_holdout", "signature_type", "endpoint"]):
        data = data.loc[np.isfinite(data.effect)]
        values = data.effect.to_numpy(float)
        if len(values) < 3:
            continue
        p, method = bridge.sign_flip_test(values, SEED)
        estimates = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
        rows.append({
            "signature_type": kind, "endpoint": endpoint, "strict_joint_holdout": bool(strict),
            "n_donors": len(values), "n_MS": int(data.disease_group.eq("MS").sum()),
            "n_NIND": int(data.disease_group.eq("NIND").sum()),
            "mean_effect": float(values.mean()), "median_effect": float(np.median(values)),
            "ci95_donor_bootstrap_fixed_scores": np.quantile(estimates, [.025, .975]).tolist(),
            "n_positive_donors": int((values > 0).sum()), "n_negative_donors": int((values < 0).sum()),
            "expected_sign": expected.get((kind, endpoint)), "p_two_sided": p, "test_method": method,
        })
    for strict in (False, True):
        family = [r for r in rows if r["strict_joint_holdout"] == strict and r["expected_sign"] is not None]
        if family:
            for row, q in zip(family, multipletests([r["p_two_sided"] for r in family], method="fdr_bh")[1]):
                row["q_bh_within_prespecified_family"] = float(q)
    write_json({"status": "completed", "n_reference_cells": int(reference.sum()),
                "n_reference_donors": int(obs.loc[reference, "subject_id"].nunique()),
                "primary_results_sha256": sha256(OUTPUT / "primary_results.json"),
                "sources": {k: sha256(project / FILES[k]) for k in ("response", "knockout")},
                "results": rows}, OUTPUT / "signature_results.json")


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("primary", "signature"))
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--scratch", type=Path, default=ROOT / "data/external_validation/Ban/work")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-umi", action="store_true", help="Run the uncorrected UMI sensitivity")
    args = parser.parse_args()
    OUTPUT = args.output or (OUTPUT / "raw_umi" if args.raw_umi else OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.raw_umi:
        args.scratch = args.scratch / "raw_umi"
    args.scratch.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.scratch / "mpl"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.scratch / "numba"))
    raw_cache = args.scratch / "normalized_Ban_mono_macro.h5ad"
    scored_cache = args.scratch / "Ban_primary_scored.h5ad"
    if args.stage == "primary":
        if args.matrix is None or args.annotations is None:
            parser.error("primary requires --matrix and --annotations")
        cells = prepare(args.matrix, args.annotations, raw_cache, raw_umi=args.raw_umi)
        run_primary(cells, args.project, scored_cache)
    else:
        if not (OUTPUT / "primary_results.json").is_file() or not scored_cache.is_file():
            parser.error("generate the Ban primary results and scored checkpoint first")
        project_signatures(ad.read_h5ad(scored_cache), args.project)


if __name__ == "__main__":
    main()
