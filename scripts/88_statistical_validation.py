#!/usr/bin/env python3
"""Validate donor-level effects, uncertainty, covariates and multiplicity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260903
N_BOOT = 20_000
PRIMARY_COHORTS = ("GSE138266", "GSE286068", "GSE328265")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(values, [0.025, 0.975]).astype(float))


def pooled_cohen_d(case: np.ndarray, control: np.ndarray) -> float:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    df = len(case) + len(control) - 2
    pooled = np.sqrt(
        ((len(case) - 1) * np.var(case, ddof=1)
         + (len(control) - 1) * np.var(control, ddof=1)) / df
    )
    return float((np.mean(case) - np.mean(control)) / pooled) if pooled > 0 else np.nan


def bootstrap_unpaired_meta(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float]:
    observed_raw = []
    observed_d = []
    raw_boot = []
    d_boot = []
    for case, control in groups.values():
        case = np.asarray(case, dtype=float)
        control = np.asarray(control, dtype=float)
        observed_raw.append(float(case.mean() - control.mean()))
        observed_d.append(pooled_cohen_d(case, control))
        case_b = rng.choice(case, size=(n_boot, len(case)), replace=True)
        control_b = rng.choice(control, size=(n_boot, len(control)), replace=True)
        difference = case_b.mean(axis=1) - control_b.mean(axis=1)
        df = len(case) + len(control) - 2
        pooled = np.sqrt(
            ((len(case) - 1) * case_b.var(axis=1, ddof=1)
             + (len(control) - 1) * control_b.var(axis=1, ddof=1)) / df
        )
        raw_boot.append(difference)
        d_boot.append(np.divide(
            difference,
            pooled,
            out=np.full_like(difference, np.nan),
            where=pooled > 1e-12,
        ))
    raw_distribution = np.nanmean(np.vstack(raw_boot), axis=0)
    d_distribution = np.nanmean(np.vstack(d_boot), axis=0)
    raw_low, raw_high = percentile_ci(raw_distribution)
    d_low, d_high = percentile_ci(d_distribution)
    return {
        "raw_estimate": float(np.mean(observed_raw)),
        "raw_ci95_low": raw_low,
        "raw_ci95_high": raw_high,
        "standardized_estimate": float(np.mean(observed_d)),
        "standardized_ci95_low": d_low,
        "standardized_ci95_high": d_high,
    }


def bootstrap_paired_meta(
    frame: pd.DataFrame,
    value_column: str,
    rng: np.random.Generator,
    n_boot: int,
    mean_weights: str = "equal",
) -> dict[str, float]:
    cohort_means = []
    cohort_standardized = []
    boot_means = []
    boot_standardized = []
    cohort_sizes = []
    for _, cohort in frame.groupby("dataset", sort=True):
        values = cohort[value_column].dropna().to_numpy(float)
        sampled = rng.choice(values, size=(n_boot, len(values)), replace=True)
        means = sampled.mean(axis=1)
        standard_deviations = sampled.std(axis=1, ddof=1)
        cohort_means.append(float(values.mean()))
        cohort_standardized.append(float(values.mean() / values.std(ddof=1)))
        boot_means.append(means)
        boot_standardized.append(np.divide(
            means,
            standard_deviations,
            out=np.full_like(means, np.nan),
            where=standard_deviations > 1e-12,
        ))
        cohort_sizes.append(len(values))
    weights = (
        np.sqrt(np.asarray(cohort_sizes, dtype=float))
        if mean_weights == "sqrt_n"
        else np.ones(len(cohort_sizes), dtype=float)
    )
    weights = weights / weights.sum()
    mean_distribution = np.nansum(np.vstack(boot_means) * weights[:, None], axis=0)
    standardized_distribution = np.nanmean(np.vstack(boot_standardized), axis=0)
    mean_low, mean_high = percentile_ci(mean_distribution)
    standardized_low, standardized_high = percentile_ci(standardized_distribution)
    return {
        "n_cohorts": len(cohort_sizes),
        "n_donors": int(sum(cohort_sizes)),
        "mean_estimate": float(np.average(cohort_means, weights=weights)),
        "mean_ci95_low": mean_low,
        "mean_ci95_high": mean_high,
        "standardized_estimate": float(np.mean(cohort_standardized)),
        "standardized_ci95_low": standardized_low,
        "standardized_ci95_high": standardized_high,
    }


def effect_record(
    figure: str,
    analysis: str,
    endpoint: str,
    estimand: str,
    stats: dict[str, float],
    estimate_key: str,
    low_key: str,
    high_key: str,
    source_estimate: float | None,
    p_value: float | None,
    fdr: float | None,
    source_file: str,
    bootstrap_scheme: str,
) -> dict[str, object]:
    estimate = float(stats[estimate_key])
    low = float(stats[low_key])
    high = float(stats[high_key])
    return {
        "figure": figure,
        "analysis": analysis,
        "endpoint": endpoint,
        "estimand": estimand,
        "n_cohorts": stats.get("n_cohorts", len(PRIMARY_COHORTS)),
        "n_donors": stats.get("n_donors", np.nan),
        "estimate": estimate,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "p_value": p_value,
        "fdr": fdr,
        "source_estimate": source_estimate,
        "absolute_recalculation_difference": (
            abs(estimate - float(source_estimate)) if source_estimate is not None else np.nan
        ),
        "bootstrap_scheme": bootstrap_scheme,
        "ci_status": "estimated",
        "source_file": source_file,
    }


def load_disease_scores(root: Path) -> pd.DataFrame:
    discovery = pd.read_csv(root / "results/two_axis/two_axis_donor_scores.csv")
    discovery = discovery.loc[
        discovery["dataset"].isin(["GSE138266", "GSE286068"])
        & discovery["n_CSF_continuum_cells"].ge(20)
    ].copy()
    discovery["is_ms"] = discovery["is_MS"].astype(bool)
    external = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/"
        "GSE328265_two_axis_donor_scores.csv"
    )
    external = external.loc[external["n_continuum_cells"].ge(20)].copy()
    external["dataset"] = "GSE328265"
    external["is_ms"] = external["diagnosis"].ne("Ctrl")
    columns = [
        "dataset", "subject_id", "diagnosis", "disease_group", "is_ms",
        "continuous_decoupling_index", "decoupled_inflammatory_fraction",
    ]
    return pd.concat([discovery[columns], external[columns]], ignore_index=True)


def build_key_effect_cis(root: Path, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    disease = load_disease_scores(root)
    meta_path = (
        "results/external_validation/GSE328265_two_axis_projection/"
        "three_cohort_two_axis_meta.csv"
    )
    disease_meta = pd.read_csv(root / meta_path).set_index("endpoint")
    for endpoint in ("continuous_decoupling_index", "decoupled_inflammatory_fraction"):
        groups = {}
        for dataset, cohort in disease.groupby("dataset", sort=True):
            groups[dataset] = (
                cohort.loc[cohort["is_ms"], endpoint].to_numpy(float),
                cohort.loc[~cohort["is_ms"], endpoint].to_numpy(float),
            )
        stats = bootstrap_unpaired_meta(groups, rng, n_boot)
        stats["n_cohorts"] = 3
        stats["n_donors"] = len(disease)
        source = disease_meta.loc[endpoint]
        rows.append(effect_record(
            "Figure 2", "MS-control disease contrast", endpoint,
            "equal-cohort mean Cohen d", stats,
            "standardized_estimate", "standardized_ci95_low", "standardized_ci95_high",
            source["mean_standardized_effect"], source["stouffer_p"], source["fdr_bh"],
            meta_path, "case/control donor resampling within cohort",
        ))
        rows.append(effect_record(
            "Figure 2", "MS-control disease contrast", endpoint,
            "equal-cohort mean raw difference", stats,
            "raw_estimate", "raw_ci95_low", "raw_ci95_high",
            None, source["stouffer_p"], source["fdr_bh"], meta_path,
            "case/control donor resampling within cohort",
        ))

    lam_path = "results/cross_species_lam/LAM_donor_scores.csv"
    lam_meta_path = "results/cross_species_lam/LAM_decoupled_meta.csv"
    lam_endpoint = "decoupled_excess__mouse_GMCSF_LAM_precursor"
    lam = pd.read_csv(root / lam_path).dropna(subset=[lam_endpoint])
    lam = lam.loc[lam["disease_group"].isin(["MS", "RRMS", "SPMS", "PPMS"])]
    stats = bootstrap_paired_meta(lam, lam_endpoint, rng, n_boot)
    source = pd.read_csv(root / lam_meta_path).set_index("endpoint").loc[lam_endpoint]
    rows.append(effect_record(
        "Figure 2", "cross-species program projection", lam_endpoint,
        "equal-cohort mean paired excess", stats,
        "mean_estimate", "mean_ci95_low", "mean_ci95_high",
        source["mean_cohort_excess"], source["stouffer_p"], source["fdr_bh"],
        lam_meta_path, "paired-donor resampling within cohort",
    ))

    receiver_path = "results/niche_regulator/receiver_donor_state_deltas.csv.gz"
    receiver_meta_path = "results/niche_regulator/receiver_state_delta_meta.csv"
    receiver = pd.read_csv(root / receiver_path)
    receiver_meta = pd.read_csv(root / receiver_meta_path)
    receiver_keys = (("receptor", "IFNG"), ("receptor", "TNF"), ("activity", "IRF1"), ("activity", "STAT1"))
    for feature_type, feature_name in receiver_keys:
        selected = receiver.loc[
            receiver["contrast"].eq("decoupled_minus_other_incomplete")
            & receiver["feature_type"].eq(feature_type)
            & receiver["feature_name"].eq(feature_name)
        ]
        stats = bootstrap_paired_meta(selected, "delta", rng, n_boot)
        source = receiver_meta.loc[
            receiver_meta["contrast"].eq("decoupled_minus_other_incomplete")
            & receiver_meta["feature_type"].eq(feature_type)
            & receiver_meta["feature_name"].eq(feature_name)
        ].iloc[0]
        rows.append(effect_record(
            "Figure 3", "receiver-state contrast", feature_name,
            "equal-cohort mean paired Cohen dz", stats,
            "standardized_estimate", "standardized_ci95_low", "standardized_ci95_high",
            source["mean_standardized_effect"], source["stouffer_p"],
            source["fdr_bh_within_contrast"], receiver_meta_path,
            "paired-donor resampling within cohort",
        ))

    bridge_path = "results/in_vivo_crispr_bridge/donor_level_bridge_effects.csv"
    bridge_meta_path = "results/in_vivo_crispr_bridge/bridge_meta_analysis.csv"
    bridge = pd.read_csv(root / bridge_path)
    bridge_meta = pd.read_csv(root / bridge_meta_path)
    bridge_endpoints = (
        ("published_response", "decoupled_minus_other_incomplete"),
        ("published_response", "axis2_spearman"),
        ("direct_ko_consensus", "decoupled_minus_other_incomplete"),
        ("direct_ko_consensus", "axis2_spearman"),
        ("direct_ko_consensus", "pseudotime_spearman"),
    )
    for signature_type, endpoint in bridge_endpoints:
        selected = bridge.loc[
            bridge["signature_type"].eq(signature_type)
            & bridge["perturbation"].str.casefold().eq("ifngr1")
            & bridge["endpoint"].eq(endpoint)
        ].dropna(subset=["effect"])
        stats = bootstrap_paired_meta(selected, "effect", rng, n_boot)
        source = bridge_meta.loc[
            bridge_meta["signature_type"].eq(signature_type)
            & bridge_meta["perturbation"].str.casefold().eq("ifngr1")
            & bridge_meta["endpoint"].eq(endpoint)
        ].iloc[0]
        rows.append(effect_record(
            "Figure 4", "in vivo CRISPR signature bridge",
            f"{signature_type}:{endpoint}", "equal-cohort mean paired Cohen dz",
            stats, "standardized_estimate", "standardized_ci95_low", "standardized_ci95_high",
            source["mean_standardized_effect"], source["stouffer_p"],
            source["fdr_bh_within_signature_endpoint"], bridge_meta_path,
            "donor resampling within cohort",
        ))

    oracle_path = "results/virtual_perturbation_celloracle/all_donor_state_perturbation_effects.csv.gz"
    oracle_meta_path = "results/virtual_perturbation_celloracle/primary_endpoint_meta_analysis.csv"
    oracle_veto_path = "results/virtual_perturbation_celloracle/harm_veto_meta_analysis.csv"
    oracle = pd.read_csv(root / oracle_path)
    oracle = oracle.loc[
        oracle["network_model"].eq("observed_GRN")
        & oracle["n_propagation"].eq(3)
        & oracle["endpoint_variant"].eq("heldout_direct_targets")
    ]
    oracle_meta = pd.read_csv(root / oracle_meta_path)
    oracle_veto = pd.read_csv(root / oracle_veto_path)
    oracle_keys = (
        ("IRF1", "axis2", "primary"),
        ("IRF1", "gmcsf_lam_precursor", "primary"),
        ("STAT1", "axis2", "primary"),
        ("STAT1", "lam2_complement_apc", "veto"),
    )
    for candidate, endpoint, family in oracle_keys:
        selected = oracle.loc[
            oracle["candidate"].eq(candidate)
            & oracle["endpoint"].eq(endpoint)
            & oracle["state"].eq("decoupled")
        ]
        stats = bootstrap_paired_meta(
            selected, "mean_standardized_delta", rng, n_boot, mean_weights="sqrt_n"
        )
        if family == "primary":
            source = oracle_meta.loc[
                oracle_meta["candidate"].eq(candidate) & oracle_meta["endpoint"].eq(endpoint)
            ].iloc[0]
            p_value = source["stouffer_expected_one_sided_p"]
            fdr = source["fdr_bh_primary_endpoints"]
            source_estimate = source["weighted_mean_effect"]
            source_file = oracle_meta_path
        else:
            source = oracle_veto.loc[
                oracle_veto["candidate"].eq(candidate)
                & oracle_veto["veto_endpoint"].eq(endpoint)
            ].iloc[0]
            p_value = source["harmful_one_sided_p"]
            fdr = source["fdr_bh_veto_endpoints"]
            source_estimate = stats["mean_estimate"]
            source_file = oracle_veto_path
        rows.append(effect_record(
            "Figure 4", "CellOracle directional perturbation", f"{candidate}:{endpoint}",
            "sqrt-n-weighted cohort mean standardized delta", stats,
            "mean_estimate", "mean_ci95_low", "mean_ci95_high",
            source_estimate, p_value, fdr, source_file,
            "donor resampling within cohort; fixed sqrt(n) cohort weights",
        ))

    drug_path = (
        "results/transition_aware_drug_screen/nibr_full_profiles/"
        "NIBR_transition_candidate_profile_scores.csv"
    )
    profiles = pd.read_csv(root / drug_path)
    bx = profiles.loc[profiles["drug"].eq("BX-912") & np.isclose(profiles["dose_um"], 0.1)].iloc[0]
    rows.append({
        "figure": "Figure 5",
        "analysis": "transition-aware drug screen",
        "endpoint": "BX-912 0.1 uM lock reversal",
        "estimand": "single profile weighted reversal score",
        "n_cohorts": np.nan,
        "n_donors": np.nan,
        "estimate": bx["lock_weighted_reversal_score"],
        "bootstrap_ci95_low": np.nan,
        "bootstrap_ci95_high": np.nan,
        "ci_excludes_zero": np.nan,
        "p_value": bx["lock_weighted_reversal_support_p"],
        "fdr": bx["lock_weighted_reversal_support_p_fdr"],
        "source_estimate": bx["lock_weighted_reversal_score"],
        "absolute_recalculation_difference": 0.0,
        "bootstrap_scheme": "not applicable: one profile per dose; inference uses gene-label permutation",
        "ci_status": "not_estimable_profile_level",
        "source_file": drug_path,
    })
    return pd.DataFrame(rows)


def schafflick_subject_metadata() -> pd.DataFrame:
    rows = [
        ("PST95809", "control", 43, "female"),
        ("PST83775", "control", 43, "male"),
        ("PTC41540", "control", 32, "female"),
        ("PTC85037", "control", 25, "female"),
        ("PTC32190", "control", 33, "male"),
        ("PTC45044", "control", 25, "female"),
        ("MS60249", "MS", 28, "female"),
        ("MS58637", "MS", 22, "male"),
        ("MS74594", "MS", 42, "male"),
        ("MS19270", "MS", 35, "female"),
        ("MS71658", "MS", 47, "female"),
        ("MS49131", "MS", 47, "female"),
    ]
    frame = pd.DataFrame(rows, columns=["subject_id", "group", "age", "sex"])
    frame["dmt_at_sampling"] = np.where(frame["group"].eq("MS"), 0.0, np.nan)
    frame["treatment_note"] = np.where(
        frame["group"].eq("MS"), "treatment-naive", "not applicable"
    )
    frame["dataset"] = "GSE138266"
    frame["source"] = (
        "Schafflick et al. 2020 Supplementary Tables 1-2; "
        "https://doi.org/10.1038/s41467-019-14118-w"
    )
    return frame


def fit_hc3(
    frame: pd.DataFrame,
    outcome: str,
    formula: str,
    terms: list[str],
    dataset: str,
    specification: str,
    caveat: str,
) -> list[dict[str, object]]:
    model = smf.ols(formula, data=frame).fit(cov_type="HC3")
    interval = model.conf_int()
    rows = []
    for term in terms:
        rows.append({
            "dataset": dataset,
            "specification": specification,
            "outcome": outcome,
            "term": term,
            "n_donors": int(model.nobs),
            "model_rank": int(model.model.rank),
            "residual_df": int(model.df_resid),
            "coefficient": float(model.params[term]),
            "hc3_se": float(model.bse[term]),
            "hc3_p": float(model.pvalues[term]),
            "ci95_low": float(interval.loc[term, 0]),
            "ci95_high": float(interval.loc[term, 1]),
            "model_r_squared": float(model.rsquared),
            "formula": formula,
            "full_rank": bool(model.model.rank == model.model.exog.shape[1]),
            "caveat": caveat,
        })
    return rows


def covariate_audit(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = ["continuous_decoupling_index", "decoupled_inflammatory_fraction"]
    discovery = pd.read_csv(root / "results/two_axis/two_axis_donor_scores.csv")
    schafflick = schafflick_subject_metadata()
    gse138 = discovery.loc[
        discovery["dataset"].eq("GSE138266")
        & discovery["n_CSF_continuum_cells"].ge(20)
    ].merge(schafflick, on="subject_id", how="left", validate="one_to_one")
    gse138["is_ms"] = gse138["is_MS"].astype(int)
    gse138["female"] = gse138["sex"].eq("female").astype(int)
    gse138["age_z"] = (gse138["age"] - gse138["age"].mean()) / gse138["age"].std(ddof=0)

    external = pd.read_csv(
        root / "results/external_validation/GSE328265_two_axis_projection/"
        "GSE328265_two_axis_donor_scores.csv"
    )
    external = external.loc[external["n_continuum_cells"].ge(20)].copy()
    external["is_ms"] = external["diagnosis"].ne("Ctrl").astype(int)
    external["female"] = external["sex"].eq("female").astype(int)
    external["fresh"] = external["sample_type"].eq("fresh").astype(int)
    external["progressive"] = external["diagnosis"].isin(["PPMS", "SPMS"]).astype(int)

    model_rows: list[dict[str, object]] = []
    for outcome in outcomes:
        model_rows.extend(fit_hc3(
            gse138, outcome, f"{outcome} ~ is_ms", ["is_ms"], "GSE138266",
            "unadjusted_primary_complete_covariates",
            "All primary donors have age and sex; MS participants were treatment-naive.",
        ))
        model_rows.extend(fit_hc3(
            gse138, outcome, f"{outcome} ~ is_ms + age_z + female", ["is_ms"],
            "GSE138266", "age_sex_adjusted_HC3",
            "Small model (n=11); HC3 limits but cannot remove small-sample uncertainty.",
        ))

        model_rows.extend(fit_hc3(
            external, outcome, f"{outcome} ~ is_ms", ["is_ms"], "GSE328265",
            "unadjusted_primary",
            "All 24 donors meeting the prespecified >=20-cell threshold.",
        ))
        complete = external.dropna(subset=["age", "sex"]).copy()
        complete["age_z"] = (complete["age"] - complete["age"].mean()) / complete["age"].std(ddof=0)
        model_rows.extend(fit_hc3(
            complete, outcome, f"{outcome} ~ is_ms + age_z + female", ["is_ms"],
            "GSE328265", "age_sex_adjusted_HC3",
            "Complete-case sensitivity (23/24 primary donors).",
        ))
        model_rows.extend(fit_hc3(
            complete, outcome, f"{outcome} ~ is_ms + age_z + female + fresh", ["is_ms"],
            "GSE328265", "age_sex_sample_type_adjusted_HC3",
            "Fresh sampling occurs only in progressive MS; the sample-type coefficient is partially confounded.",
        ))
        frozen = external.loc[external["sample_type"].eq("frozen")].dropna(subset=["age", "sex"]).copy()
        frozen["age_z"] = (frozen["age"] - frozen["age"].mean()) / frozen["age"].std(ddof=0)
        model_rows.extend(fit_hc3(
            frozen, outcome, f"{outcome} ~ is_ms + age_z + female", ["is_ms"],
            "GSE328265", "frozen_only_age_sex_adjusted_HC3",
            "Removes fresh/frozen preparation contrast; n=18.",
        ))
        ms_only = external.loc[external["is_ms"].eq(1)].dropna(
            subset=["age", "sex", "dmt_at_sampling"]
        ).copy()
        ms_only["age_z"] = (ms_only["age"] - ms_only["age"].mean()) / ms_only["age"].std(ddof=0)
        model_rows.extend(fit_hc3(
            ms_only, outcome,
            f"{outcome} ~ progressive + age_z + female + dmt_at_sampling",
            ["progressive", "dmt_at_sampling"], "GSE328265",
            "MS_only_stage_age_sex_DMT_HC3",
            "Within-MS sensitivity only; it does not estimate the MS-control disease contrast.",
        ))
    models = pd.DataFrame(model_rows)
    models["fdr_bh_two_outcomes_within_specification_term"] = models.groupby(
        ["dataset", "specification", "term"], observed=True
    )["hc3_p"].transform(lambda x: multipletests(x, method="fdr_bh")[1])

    disease = load_disease_scores(root)
    gse286_n = int(disease["dataset"].eq("GSE286068").sum())
    coverage = pd.DataFrame([
        {
            "dataset": "GSE138266", "analysis_role": "discovery disease contrast",
            "n_primary_donors": len(gse138), "n_age": int(gse138["age"].notna().sum()),
            "n_sex": int(gse138["sex"].notna().sum()),
            "n_treatment_status_among_ms": int(gse138.loc[gse138["is_ms"].eq(1), "dmt_at_sampling"].notna().sum()),
            "n_ms": int(gse138["is_ms"].sum()), "age_sex_model_estimable": True,
            "treatment_model_estimable": False,
            "source": "Schafflick 2020 Supplementary Tables 1-2",
            "reason": "Complete age/sex mapping; all MS donors treatment-naive, so DMT has no variation.",
        },
        {
            "dataset": "GSE286068", "analysis_role": "discovery disease contrast",
            "n_primary_donors": gse286_n, "n_age": 0, "n_sex": 0,
            "n_treatment_status_among_ms": 0,
            "n_ms": int(disease.loc[disease["dataset"].eq("GSE286068"), "is_ms"].sum()),
            "age_sex_model_estimable": False, "treatment_model_estimable": False,
            "source": "Parween 2025 Supplementary Data 2 and GEO metadata",
            "reason": "Supplement gives cohort participant metadata but no verifiable mapping to GEO pseudonyms; no order-based imputation used.",
        },
        {
            "dataset": "GSE328265", "analysis_role": "frozen external disease contrast",
            "n_primary_donors": len(external), "n_age": int(external["age"].notna().sum()),
            "n_sex": int(external["sex"].notna().sum()),
            "n_treatment_status_among_ms": int(external.loc[external["is_ms"].eq(1), "dmt_at_sampling"].notna().sum()),
            "n_ms": int(external["is_ms"].sum()), "age_sex_model_estimable": True,
            "treatment_model_estimable": True,
            "source": "Boersch 2026 Supplementary Table 1",
            "reason": "Age/sex complete for 23/24; DMT sensitivity estimable within 17/18 MS donors.",
        },
    ])
    for field in ("age", "sex"):
        coverage[f"{field}_coverage_fraction"] = coverage[f"n_{field}"] / coverage["n_primary_donors"]

    balance_rows = []
    for dataset, frame in (("GSE138266", gse138), ("GSE328265", external.dropna(subset=["age", "sex"]))):
        case = frame.loc[frame["is_ms"].eq(1)]
        control = frame.loc[frame["is_ms"].eq(0)]
        age_d = pooled_cohen_d(case["age"].to_numpy(), control["age"].to_numpy())
        table = pd.crosstab(frame["is_ms"], frame["female"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        from scipy.stats import fisher_exact
        balance_rows.append({
            "dataset": dataset,
            "n_complete": len(frame),
            "n_ms": len(case),
            "n_control": len(control),
            "mean_age_ms": float(case["age"].mean()),
            "mean_age_control": float(control["age"].mean()),
            "age_cohen_d_ms_minus_control": age_d,
            "female_fraction_ms": float(case["female"].mean()),
            "female_fraction_control": float(control["female"].mean()),
            "sex_fisher_exact_p": float(fisher_exact(table.to_numpy()).pvalue),
        })
    balance = pd.DataFrame(balance_rows)
    subject_table = pd.concat([
        schafflick,
        external.assign(
            dataset="GSE328265",
            group=np.where(external["is_ms"].eq(1), "MS", "control"),
            source="Boersch 2026 Supplementary Table 1",
            treatment_note=np.where(
                external["is_ms"].eq(1),
                external["dmt_at_sampling"].map({0.0: "no DMT", 1.0: "DMT"}),
                "not applicable",
            ),
        )[["subject_id", "group", "age", "sex", "dmt_at_sampling", "treatment_note", "dataset", "source"]],
    ], ignore_index=True)
    return coverage, models, balance, subject_table


def overlap_audit(root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(root / "metadata/all_geo_sample_metadata.csv", dtype=str)
    sets: dict[str, dict[str, set[str]]] = {}
    for dataset in ("GSE138266", "GSE194078", "GSE286068"):
        selected = metadata.loc[metadata["gse"].eq(dataset)]
        sets[dataset] = {
            "accessions": set(selected["sample_accession"].dropna()),
            "subjects": set(selected["subject_id"].dropna()),
        }
    external = pd.read_csv(root / "metadata/GSE328265_sample_metadata.csv", dtype=str)
    sets["GSE328265"] = {
        "accessions": set(external["sample_accession"].dropna()),
        "subjects": set(external["subject_id"].dropna()),
    }
    rows = []
    for left, right in combinations(sets, 2):
        accession_overlap = sorted(sets[left]["accessions"] & sets[right]["accessions"])
        subject_overlap = sorted(sets[left]["subjects"] & sets[right]["subjects"])
        rows.append({
            "cohort_a": left,
            "cohort_b": right,
            "n_accessions_a": len(sets[left]["accessions"]),
            "n_accessions_b": len(sets[right]["accessions"]),
            "n_accession_overlap": len(accession_overlap),
            "accession_overlap": ";".join(accession_overlap),
            "n_pseudonym_overlap": len(subject_overlap),
            "pseudonym_overlap": ";".join(subject_overlap),
            "audit_status": "PASS" if not accession_overlap and not subject_overlap else "REVIEW",
            "identity_caveat": (
                "No accession or pseudonym overlap. External matrices are the newly generated "
                "GSE327243/GSE328264 subseries; anonymization prevents absolute latent-identity exclusion."
            ),
        })
    return pd.DataFrame(rows)


def multiplicity_audit(root: Path) -> pd.DataFrame:
    families: list[dict[str, object]] = []

    def add(
        analysis: str,
        path: str,
        family: str,
        p_column: str,
        fdr_column: str,
        frame: pd.DataFrame | None = None,
    ) -> None:
        data = pd.read_csv(root / path) if frame is None else frame
        fdr_complete = bool(data[fdr_column].notna().all())
        valid_range = bool(data[fdr_column].dropna().between(0, 1).all())
        families.append({
            "analysis": analysis,
            "correction_family": family,
            "n_tests": len(data),
            "p_value_column": p_column,
            "fdr_column": fdr_column,
            "n_nonmissing_fdr": int(data[fdr_column].notna().sum()),
            "minimum_fdr": float(data[fdr_column].min()),
            "fdr_complete": fdr_complete,
            "valid_probability_range": valid_range,
            "audit_status": "PASS" if fdr_complete and valid_range else "FAIL",
            "source_file": path,
        })

    add(
        "primary disease meta-analysis",
        "results/external_validation/GSE328265_two_axis_projection/three_cohort_two_axis_meta.csv",
        "two prespecified disease endpoints", "stouffer_p", "fdr_bh",
    )
    add(
        "threshold sensitivity grid", "results/two_axis/decoupling_threshold_meta_grid.csv",
        "20 pseudotime-by-activation cutoffs", "stouffer_p", "fdr_bh_across_grid",
    )
    add(
        "cross-species LAM projection", "results/cross_species_lam/LAM_decoupled_meta.csv",
        "five projected LAM endpoints", "stouffer_p", "fdr_bh",
    )
    receiver_path = "results/niche_regulator/receiver_state_delta_meta.csv"
    receiver = pd.read_csv(root / receiver_path)
    for contrast, frame in receiver.groupby("contrast", sort=True):
        add(
            "receiver feature screen", receiver_path,
            f"43 features within {contrast}", "stouffer_p", "fdr_bh_within_contrast", frame,
        )
    bridge_path = "results/in_vivo_crispr_bridge/bridge_meta_analysis.csv"
    bridge = pd.read_csv(root / bridge_path)
    for (signature, endpoint), frame in bridge.groupby(["signature_type", "endpoint"], sort=True):
        add(
            "in vivo CRISPR bridge", bridge_path,
            f"four receptors: {signature}/{endpoint}", "stouffer_p",
            "fdr_bh_within_signature_endpoint", frame,
        )
    add(
        "in vivo CRISPR primary global family", bridge_path,
        "20 preregistered published-response primary tests", "stouffer_p", "fdr_bh_global_primary",
        bridge.loc[bridge["fdr_bh_global_primary"].notna()],
    )
    add(
        "CellOracle primary endpoint family",
        "results/virtual_perturbation_celloracle/primary_endpoint_meta_analysis.csv",
        "10 directional candidate-endpoint tests", "stouffer_expected_one_sided_p",
        "fdr_bh_primary_endpoints",
    )
    add(
        "CellOracle harm-veto family",
        "results/virtual_perturbation_celloracle/harm_veto_meta_analysis.csv",
        "seven prespecified harm endpoints", "harmful_one_sided_p", "fdr_bh_veto_endpoints",
    )
    add(
        "CellOracle combination family",
        "results/virtual_perturbation_celloracle_combinations/combination_criterion_meta_analysis.csv",
        "eight compensatory criteria", "stouffer_expected_one_sided_p",
        "fdr_bh_combination_criteria",
    )
    drug_path = (
        "results/transition_aware_drug_screen/nibr_full_profiles/"
        "NIBR_transition_candidate_profile_scores.csv"
    )
    drug = pd.read_csv(root / drug_path)
    endpoint_prefixes = (
        "lock_weighted_reversal", "lock_high_suppression", "axis2_suppression",
        "lam2_adaptation", "trajectory_adaptation",
    )
    for endpoint in endpoint_prefixes:
        for direction in ("support", "harm"):
            add(
                "NIBR Drug-seq screen", drug_path,
                f"76 profiles: {endpoint} {direction}", f"{endpoint}_{direction}_p",
                f"{endpoint}_{direction}_p_fdr", drug,
            )
    add(
        "transcriptomic-neighbor mechanism enrichment",
        "results/transition_aware_drug_screen/mechanism_deconvolution/"
        "BX912_top5pct_neighbor_annotation_enrichment.csv",
        "172 target/MoA enrichment tests", "fisher_p", "fdr",
    )
    return pd.DataFrame(families)






def source_manifest(root: Path) -> pd.DataFrame:
    paths = [
        "results/two_axis/two_axis_donor_scores.csv",
        "results/external_validation/GSE328265_two_axis_projection/GSE328265_two_axis_donor_scores.csv",
        "results/external_validation/GSE328265_two_axis_projection/three_cohort_two_axis_meta.csv",
        "results/cross_species_lam/LAM_donor_scores.csv",
        "results/niche_regulator/receiver_donor_state_deltas.csv.gz",
        "results/in_vivo_crispr_bridge/donor_level_bridge_effects.csv",
        "results/virtual_perturbation_celloracle/all_donor_state_perturbation_effects.csv.gz",
        "results/transition_aware_drug_screen/nibr_full_profiles/NIBR_transition_candidate_profile_scores.csv",
        "metadata/all_geo_sample_metadata.csv",
        "metadata/GSE328265_sample_metadata.csv",
        "data/reference/clinical_metadata/Schafflick_2020_MOESM1_ESM.pdf",
        "data/reference/clinical_metadata/Parween_2025_MOESM34_ESM.xlsx",
    ]
    return pd.DataFrame([
        {
            "relative_path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": sha256(root / relative),
        }
        for relative in paths
    ])




def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/statistical_validation"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    effects = build_key_effect_cis(root, rng, args.n_boot)
    coverage, models, balance, subject_table = covariate_audit(root)
    overlap = overlap_audit(root)
    multiplicity = multiplicity_audit(root)
    manifest = source_manifest(root)

    effects.to_csv(output / "key_effect_confidence_intervals.csv", index=False)
    coverage.to_csv(output / "clinical_covariate_coverage.csv", index=False)
    models.to_csv(output / "clinical_covariate_models.csv", index=False)
    balance.to_csv(output / "clinical_covariate_balance.csv", index=False)
    subject_table.to_csv(output / "clinical_covariate_subject_table.csv", index=False)
    overlap.to_csv(output / "cohort_overlap_matrix.csv", index=False)
    multiplicity.to_csv(output / "multiplicity_family_audit.csv", index=False)
    manifest.to_csv(output / "audit_source_manifest.csv", index=False)

    checks = {
        "key_effect_recalculations_match": bool(
            effects.loc[effects["source_estimate"].notna(), "absolute_recalculation_difference"].max() <= 1e-10
        ),
        "all_estimable_bootstrap_intervals_finite": bool(
            effects.loc[effects["ci_status"].eq("estimated"), ["bootstrap_ci95_low", "bootstrap_ci95_high"]]
            .notna().all().all()
        ),
        "all_hc3_models_full_rank": bool(models["full_rank"].all()),
        "all_hc3_models_positive_residual_df": bool(models["residual_df"].gt(0).all()),
        "no_accession_or_pseudonym_overlap": bool(overlap["audit_status"].eq("PASS").all()),
        "all_multiplicity_families_complete": bool(multiplicity["audit_status"].eq("PASS").all()),
        "primary_disease_donor_count_is_49": int(load_disease_scores(root).shape[0]) == 49,
        "gse138_age_sex_complete": bool(
            coverage.loc[coverage["dataset"].eq("GSE138266"), ["n_age", "n_sex"]].eq(11).all().all()
        ),
        "gse328_age_sex_23_of_24": bool(
            coverage.loc[coverage["dataset"].eq("GSE328265"), ["n_age", "n_sex"]].eq(23).all().all()
        ),
    }
    qa = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "verification_status": "VERIFIED" if all(checks.values()) else "NOT_VERIFIED",
        "seed": args.seed,
        "n_bootstrap_iterations": args.n_boot,
        "n_passed": int(sum(checks.values())),
        "n_checks": len(checks),
        "checks": checks,
    }
    (output / "STATISTICAL_VALIDATION.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, indent=2))
    if qa["status"] != "PASS":
        raise SystemExit("Statistical validation failed; inspect STATISTICAL_VALIDATION.json")


if __name__ == "__main__":
    main()
