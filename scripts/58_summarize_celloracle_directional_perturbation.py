#!/usr/bin/env python3
"""Summarize donor-level CellOracle perturbations and shuffled-GRN controls."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


SEED = 12345
DATASETS = ["GSE138266", "GSE286068", "GSE328265"]
EXTERNAL_DATASET = "GSE328265"

PRIMARY_ENDPOINTS = {
    "IRF1": {
        "perturbation": "KO",
        "route": "inflammation_unlock",
        "state": "decoupled",
        "endpoints": {"axis2": -1, "gmcsf_lam_precursor": -1},
        "veto": {"lam2_complement_apc": -1, "trajectory_signed": -1},
    },
    "STAT1": {
        "perturbation": "KO",
        "route": "inflammation_unlock",
        "state": "decoupled",
        "endpoints": {"axis2": -1, "gmcsf_lam_precursor": -1},
        "veto": {"lam2_complement_apc": -1, "trajectory_signed": -1},
    },
    "PPARG": {
        "perturbation": "OE",
        "route": "adaptation_restoration",
        "state": "incomplete_all",
        "endpoints": {"lam2_complement_apc": 1, "trajectory_signed": 1},
        "veto": {"axis2": 1},
    },
    "NR1H3": {
        "perturbation": "OE",
        "route": "adaptation_restoration",
        "state": "incomplete_all",
        "endpoints": {"lam2_complement_apc": 1, "trajectory_signed": 1},
        "veto": {"axis2": 1},
    },
    "MAFB": {
        "perturbation": "OE",
        "route": "adaptation_restoration",
        "state": "incomplete_all",
        "endpoints": {"lam2_complement_apc": 1, "trajectory_signed": 1},
        "veto": {"axis2": 1},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sign_flip_test(values: np.ndarray, seed: int) -> tuple[float, str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = abs(values.mean())
    total = 2 ** len(values)
    if total <= 100000:
        means = np.asarray([
            np.mean(values * np.asarray(signs))
            for signs in itertools.product([-1.0, 1.0], repeat=len(values))
        ])
        return float(np.mean(np.abs(means) >= observed - 1e-12)), "exact_sign_flip"
    rng = np.random.default_rng(seed)
    exceed = 0
    permutations = 100000
    batch = 5000
    for start in range(0, permutations, batch):
        size = min(batch, permutations - start)
        signs = rng.choice([-1.0, 1.0], size=(size, len(values)))
        means = np.mean(signs * values[None, :], axis=1)
        exceed += int(np.sum(np.abs(means) >= observed - 1e-12))
    return float((exceed + 1) / (permutations + 1)), "mc_sign_flip_100000"


def expected_one_sided_p(two_sided_p: float, mean_effect: float, expected_sign: int) -> float:
    if mean_effect * expected_sign >= 0:
        return min(1.0, two_sided_p / 2)
    return max(0.5, 1 - two_sided_p / 2)


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    donor = []
    status = []
    endpoints = []
    edges = []
    for dataset in DATASETS:
        directory = root / "results/virtual_perturbation_celloracle" / dataset
        donor.append(pd.read_csv(directory / "donor_state_perturbation_effects.csv"))
        status.append(pd.read_csv(directory / "perturbation_status.csv"))
        endpoints.append(pd.read_csv(directory / "endpoint_holdout_ledger.csv"))
        edge = pd.read_csv(directory / "inferred_candidate_outgoing_edges.csv")
        edge["dataset"] = dataset
        edges.append(edge)
    return (
        pd.concat(donor, ignore_index=True),
        pd.concat(status, ignore_index=True),
        pd.concat(endpoints, ignore_index=True),
        pd.concat(edges, ignore_index=True),
    )


def observed_tests(donor: pd.DataFrame, seed: int) -> pd.DataFrame:
    observed = donor.loc[donor["network_model"].eq("observed_GRN")].copy()
    keys = [
        "dataset", "candidate", "perturbation", "n_propagation", "endpoint",
        "endpoint_variant", "state",
    ]
    rows = []
    for counter, (key, frame) in enumerate(observed.groupby(keys, observed=True)):
        values = frame["mean_standardized_delta"].dropna().to_numpy(float)
        if len(values) < 3:
            continue
        p_value, method = sign_flip_test(values, seed + counter)
        rows.append({
            **dict(zip(keys, key)),
            "n_donors": len(values),
            "mean_effect": float(values.mean()),
            "median_effect": float(np.median(values)),
            "sd_effect": float(values.std(ddof=1)),
            "positive_fraction": float(np.mean(values > 0)),
            "sign_flip_p_two_sided": p_value,
            "test_method": method,
        })
    output = pd.DataFrame(rows)
    output["fdr_bh_within_dataset_propagation_variant_state"] = output.groupby(
        ["dataset", "n_propagation", "endpoint_variant", "state"], observed=True
    )["sign_flip_p_two_sided"].transform(lambda x: multipletests(x, method="fdr_bh")[1])
    return output


def shuffled_network_tests(donor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = donor.loc[
        donor["n_propagation"].eq(3)
        & donor["endpoint_variant"].eq("heldout_direct_targets")
    ].copy()
    keys = ["dataset", "candidate", "perturbation", "endpoint", "state"]
    network_means = (
        use.groupby(keys + ["network_model"], observed=True)["mean_standardized_delta"]
        .mean()
        .reset_index(name="network_mean_effect")
    )
    rows = []
    for key, frame in network_means.groupby(keys, observed=True):
        observed = frame.loc[frame["network_model"].eq("observed_GRN"), "network_mean_effect"]
        shuffled = frame.loc[
            frame["network_model"].str.startswith("target_shuffled_GRN_seed_"),
            "network_mean_effect",
        ].to_numpy(float)
        if len(observed) != 1 or len(shuffled) == 0:
            continue
        value = float(observed.iloc[0])
        null_mean = float(shuffled.mean())
        null_sd = float(shuffled.std(ddof=1))
        z_score = (value - null_mean) / null_sd if null_sd > 1e-12 else np.nan
        lower_p = float((np.sum(shuffled <= value + 1e-12) + 1) / (len(shuffled) + 1))
        upper_p = float((np.sum(shuffled >= value - 1e-12) + 1) / (len(shuffled) + 1))
        rows.append({
            **dict(zip(keys, key)),
            "observed_mean_effect": value,
            "n_shuffled_GRNs": len(shuffled),
            "shuffled_mean": null_mean,
            "shuffled_sd": null_sd,
            "shuffled_q025": float(np.quantile(shuffled, 0.025)),
            "shuffled_q50": float(np.quantile(shuffled, 0.50)),
            "shuffled_q975": float(np.quantile(shuffled, 0.975)),
            "observed_minus_shuffled_mean": value - null_mean,
            "network_specific_z": z_score,
            "empirical_lower_p": lower_p,
            "empirical_upper_p": upper_p,
        })
    return network_means, pd.DataFrame(rows)


def primary_ledger(observed: pd.DataFrame, shuffled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, design in PRIMARY_ENDPOINTS.items():
        for endpoint, expected_sign in design["endpoints"].items():
            for dataset in DATASETS:
                obs = observed.loc[
                    observed["dataset"].eq(dataset)
                    & observed["candidate"].eq(candidate)
                    & observed["n_propagation"].eq(3)
                    & observed["endpoint"].eq(endpoint)
                    & observed["endpoint_variant"].eq("heldout_direct_targets")
                    & observed["state"].eq(design["state"])
                ]
                null = shuffled.loc[
                    shuffled["dataset"].eq(dataset)
                    & shuffled["candidate"].eq(candidate)
                    & shuffled["endpoint"].eq(endpoint)
                    & shuffled["state"].eq(design["state"])
                ]
                if len(obs) != 1 or len(null) != 1:
                    continue
                obs = obs.iloc[0]
                null = null.iloc[0]
                rows.append({
                    "dataset": dataset,
                    "candidate": candidate,
                    "perturbation": design["perturbation"],
                    "route": design["route"],
                    "state": design["state"],
                    "endpoint": endpoint,
                    "expected_sign": expected_sign,
                    "n_donors": int(obs["n_donors"]),
                    "mean_effect": obs["mean_effect"],
                    "median_effect": obs["median_effect"],
                    "direction_success": bool(obs["mean_effect"] * expected_sign > 0),
                    "sign_flip_p_two_sided": obs["sign_flip_p_two_sided"],
                    "sign_flip_p_expected_one_sided": expected_one_sided_p(
                        obs["sign_flip_p_two_sided"], obs["mean_effect"], expected_sign
                    ),
                    "shuffled_mean": null["shuffled_mean"],
                    "shuffled_q025": null["shuffled_q025"],
                    "shuffled_q975": null["shuffled_q975"],
                    "network_specific_z_expected": null["network_specific_z"] * expected_sign,
                    "empirical_network_p_expected": null[
                        "empirical_lower_p" if expected_sign < 0 else "empirical_upper_p"
                    ],
                })
    return pd.DataFrame(rows)


def meta_analyze_primary(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in primary.groupby(
        ["candidate", "perturbation", "route", "state", "endpoint", "expected_sign"],
        observed=True,
    ):
        expected_sign = int(key[-1])
        two_p = np.clip(frame["sign_flip_p_two_sided"].to_numpy(float), 1e-12, 1)
        signed_z = np.sign(frame["mean_effect"].to_numpy(float)) * stats.norm.isf(two_p / 2)
        expected_z = expected_sign * signed_z
        weights = np.sqrt(frame["n_donors"].to_numpy(float))
        combined_z = float(np.sum(weights * expected_z) / np.sqrt(np.sum(weights**2)))

        null_z = frame["network_specific_z_expected"].replace([np.inf, -np.inf], np.nan)
        valid = null_z.notna().to_numpy()
        combined_network_z = (
            float(np.sum(weights[valid] * null_z.to_numpy(float)[valid]) / np.sqrt(np.sum(weights[valid] ** 2)))
            if valid.any() else np.nan
        )
        external = frame.loc[frame["dataset"].eq(EXTERNAL_DATASET)]
        rows.append({
            "candidate": key[0],
            "perturbation": key[1],
            "route": key[2],
            "state": key[3],
            "endpoint": key[4],
            "expected_sign": expected_sign,
            "n_cohorts": len(frame),
            "n_donors": int(frame["n_donors"].sum()),
            "n_direction_success_cohorts": int(frame["direction_success"].sum()),
            "direction_concordant": bool(frame["direction_success"].all()),
            "weighted_mean_effect": float(np.average(frame["mean_effect"], weights=weights)),
            "combined_expected_z": combined_z,
            "stouffer_expected_one_sided_p": float(stats.norm.sf(combined_z)),
            "combined_network_specific_z": combined_network_z,
            "min_cohort_empirical_network_p": float(frame["empirical_network_p_expected"].min()),
            "max_cohort_empirical_network_p": float(frame["empirical_network_p_expected"].max()),
            "external_direction_success": bool(external["direction_success"].iloc[0]) if len(external) else False,
            "external_mean_effect": float(external["mean_effect"].iloc[0]) if len(external) else np.nan,
            "external_empirical_network_p": float(external["empirical_network_p_expected"].iloc[0]) if len(external) else np.nan,
        })
    output = pd.DataFrame(rows)
    output["fdr_bh_primary_endpoints"] = multipletests(
        output["stouffer_expected_one_sided_p"], method="fdr_bh"
    )[1]
    return output


def veto_ledger(observed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, design in PRIMARY_ENDPOINTS.items():
        for endpoint, harmful_sign in design["veto"].items():
            frame = observed.loc[
                observed["candidate"].eq(candidate)
                & observed["n_propagation"].eq(3)
                & observed["endpoint"].eq(endpoint)
                & observed["endpoint_variant"].eq("heldout_direct_targets")
                & observed["state"].eq(design["state"])
            ].copy()
            if frame.empty:
                continue
            two_p = np.clip(frame["sign_flip_p_two_sided"].to_numpy(float), 1e-12, 1)
            signed_z = np.sign(frame["mean_effect"].to_numpy(float)) * stats.norm.isf(two_p / 2)
            harmful_z = harmful_sign * signed_z
            weights = np.sqrt(frame["n_donors"].to_numpy(float))
            combined = float(np.sum(weights * harmful_z) / np.sqrt(np.sum(weights**2)))
            rows.append({
                "candidate": candidate,
                "route": design["route"],
                "state": design["state"],
                "veto_endpoint": endpoint,
                "harmful_sign": harmful_sign,
                "n_harmful_direction_cohorts": int((frame["mean_effect"] * harmful_sign > 0).sum()),
                "harmful_direction_concordant": bool((frame["mean_effect"] * harmful_sign > 0).all()),
                "combined_harmful_z": combined,
                "harmful_one_sided_p": float(stats.norm.sf(combined)),
            })
    output = pd.DataFrame(rows)
    output["fdr_bh_veto_endpoints"] = multipletests(output["harmful_one_sided_p"], method="fdr_bh")[1]
    return output


def candidate_decisions(meta: pd.DataFrame, veto: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, design in PRIMARY_ENDPOINTS.items():
        frame = meta.loc[meta["candidate"].eq(candidate)].copy()
        veto_frame = veto.loc[veto["candidate"].eq(candidate)].copy()
        replicated = frame["direction_concordant"] & frame["external_direction_success"]
        statistical = frame["fdr_bh_primary_endpoints"] < 0.10
        network = (
            (frame["combined_network_specific_z"] > 1.0)
            & (frame["external_empirical_network_p"] <= 0.10)
        )
        harmful = (
            (veto_frame["fdr_bh_veto_endpoints"] < 0.10)
            & veto_frame["harmful_direction_concordant"]
        ) if len(veto_frame) else pd.Series(dtype=bool)
        primary_support_count = int((replicated & statistical).sum())
        network_support_count = int(network.sum())
        harmful_veto = bool(harmful.any()) if len(harmful) else False
        if primary_support_count >= 1 and network_support_count >= 1 and harmful_veto:
            decision = "advance_primary_with_compensatory_design"
        elif primary_support_count >= 1 and network_support_count >= 1:
            decision = "advance_primary"
        elif primary_support_count >= 1 and not harmful_veto:
            decision = "advance_mechanistic_network_sensitivity"
        elif bool(replicated.any()) and not harmful_veto:
            decision = "retain_secondary"
        else:
            decision = "do_not_prioritize"
        rows.append({
            "candidate": candidate,
            "perturbation": design["perturbation"],
            "route": design["route"],
            "n_primary_endpoints": len(frame),
            "n_directionally_replicated_endpoints": int(replicated.sum()),
            "n_cross_cohort_fdr_lt_0_10": int(statistical.sum()),
            "n_primary_support_endpoints": primary_support_count,
            "n_network_specific_endpoints": network_support_count,
            "harmful_veto": harmful_veto,
            "best_primary_fdr": float(frame["fdr_bh_primary_endpoints"].min()),
            "best_external_empirical_network_p": float(frame["external_empirical_network_p"].min()),
            "decision": decision,
        })
    return pd.DataFrame(rows).sort_values(
        ["decision", "n_primary_support_endpoints", "n_network_specific_endpoints", "best_primary_fdr"],
        ascending=[True, False, False, True],
    )


def write_report(
    output: Path,
    status: pd.DataFrame,
    one_step_max: float,
    meta: pd.DataFrame,
    decisions: pd.DataFrame,
    veto: pd.DataFrame,
) -> None:
    lines = [
        "# Directional virtual perturbation report",
        "",
        "## Design",
        "",
        "CellOracle 0.20.0 Ridge GRNs were fitted separately in each MS CSF cohort. "
        "IRF1/STAT1 were knocked out; PPARG/NR1H3/MAFB were overexpressed to the "
        "cohort-specific 90th percentile among terminal-state positive cells. Primary "
        "endpoints remove every candidate's direct DoRothEA target before scoring. "
        "Fifty target-shuffled GRNs per cohort form an empirical network null.",
        "",
        f"- Completed candidate-cohort simulations: {int(status['simulation_status'].eq('completed').sum())}/{len(status)}.",
        f"- Maximum absolute one-step held-out effect: {one_step_max:.3g}; expected value is zero.",
        "- Statistical unit: donor. Cohort tests use sign flipping; cross-cohort tests use sample-size-weighted Stouffer aggregation.",
        "",
        "## Candidate decisions",
        "",
    ]
    for row in decisions.itertuples():
        lines.append(
            f"- {row.candidate} {row.perturbation}: **{row.decision}**; "
            f"directionally replicated endpoints={row.n_directionally_replicated_endpoints}/{row.n_primary_endpoints}, "
            f"primary-support endpoints={row.n_primary_support_endpoints}, "
            f"network-specific endpoints={row.n_network_specific_endpoints}, harmful veto={row.harmful_veto}."
        )
    lines.extend(["", "## Primary endpoint meta-analysis", ""])
    for row in meta.sort_values(["route", "candidate", "endpoint"]).itertuples():
        lines.append(
            f"- {row.candidate} {row.endpoint}: weighted effect={row.weighted_mean_effect:.4f}; "
            f"direction success={row.n_direction_success_cohorts}/{row.n_cohorts}; "
            f"Stouffer z={row.combined_expected_z:.3f}, one-sided P={row.stouffer_expected_one_sided_p:.3g}, "
            f"FDR={row.fdr_bh_primary_endpoints:.3g}; external effect={row.external_mean_effect:.4f}, "
            f"external shuffled-GRN P={row.external_empirical_network_p:.3g}."
        )
    lines.extend(["", "## Harm veto audit", ""])
    for row in veto.itertuples():
        lines.append(
            f"- {row.candidate} {row.veto_endpoint}: harmful-direction cohorts="
            f"{row.n_harmful_direction_cohorts}/3; harmful one-sided P={row.harmful_one_sided_p:.3g}, "
            f"FDR={row.fdr_bh_veto_endpoints:.3g}."
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "`advance_primary` requires at least one endpoint with concordant direction including the external cohort, "
        "cross-cohort FDR < 0.10, network specificity in the combined and external analyses, and no concordant "
        "harmful veto. `advance_primary_with_compensatory_design` retains a strong, network-specific primary "
        "effect despite a reproducible state tradeoff and therefore advances to explicit rescue combinations. "
        "`advance_mechanistic_network_sensitivity` meets biological replication and donor-level "
        "statistics but needs stronger shuffled-GRN separation. Full endpoints remain mechanistic sensitivity "
        "analyses; held-out endpoints are the anti-circular primary analyses.",
    ])
    (output / "CELLORACLE_DIRECTIONAL_PERTURBATION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/virtual_perturbation_celloracle"
    donor, status, endpoints, edges = load_results(root)

    tests = observed_tests(donor, args.seed)
    network_means, shuffled = shuffled_network_tests(donor)
    primary = primary_ledger(tests, shuffled)
    meta = meta_analyze_primary(primary)
    veto = veto_ledger(tests)
    decisions = candidate_decisions(meta, veto)

    one_step = donor.loc[
        donor["network_model"].eq("observed_GRN")
        & donor["n_propagation"].eq(1)
        & donor["endpoint_variant"].eq("heldout_direct_targets"),
        "mean_standardized_delta",
    ]
    one_step_max = float(one_step.abs().max())

    donor.to_csv(output / "all_donor_state_perturbation_effects.csv.gz", index=False, compression="gzip")
    status.to_csv(output / "all_perturbation_status.csv", index=False)
    endpoints.to_csv(output / "all_endpoint_holdout_ledger.csv", index=False)
    edges.to_csv(output / "all_inferred_candidate_outgoing_edges.csv.gz", index=False, compression="gzip")
    tests.to_csv(output / "observed_GRN_donor_signflip_tests.csv", index=False)
    network_means.to_csv(output / "network_mean_effects_observed_and_shuffled.csv.gz", index=False, compression="gzip")
    shuffled.to_csv(output / "shuffled_GRN_empirical_tests.csv", index=False)
    primary.to_csv(output / "primary_endpoint_cohort_ledger.csv", index=False)
    meta.to_csv(output / "primary_endpoint_meta_analysis.csv", index=False)
    veto.to_csv(output / "harm_veto_meta_analysis.csv", index=False)
    decisions.to_csv(output / "directional_candidate_decisions.csv", index=False)

    audit = {
        "datasets": DATASETS,
        "n_candidates": len(PRIMARY_ENDPOINTS),
        "n_candidate_cohort_simulations": len(status),
        "n_completed": int(status["simulation_status"].eq("completed").sum()),
        "n_target_shuffled_GRNs_per_cohort": int(
            donor.loc[donor["network_model"].str.startswith("target_shuffled_GRN_seed_"), "network_model"].nunique()
        ),
        "maximum_absolute_one_step_heldout_effect": one_step_max,
        "one_step_heldout_negative_control_pass": bool(one_step_max < 1e-12),
    }
    (output / "directional_perturbation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_report(output, status, one_step_max, meta, decisions, veto)
    print(decisions.to_string(index=False), flush=True)
    print("\nPrimary meta-analysis\n", meta.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
