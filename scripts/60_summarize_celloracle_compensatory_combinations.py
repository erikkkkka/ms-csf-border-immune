#!/usr/bin/env python3
"""Evaluate whether co-perturbations rescue STAT1-KO adaptation costs."""

from __future__ import annotations

import argparse
import importlib.util
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


DATASETS = ["GSE138266", "GSE286068", "GSE328265"]
EXTERNAL_DATASET = "GSE328265"
COMMON_ENDPOINTS = ["axis2", "lam2_complement_apc"]


def load_summary_module():
    path = ROOT / "scripts/58_summarize_celloracle_directional_perturbation.py"
    spec = importlib.util.spec_from_file_location("single_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summary = load_summary_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=summary.SEED)
    return parser.parse_args()


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combination = []
    single = []
    status = []
    for dataset in DATASETS:
        combination_dir = root / "results/virtual_perturbation_celloracle_combinations" / dataset
        single_dir = root / "results/virtual_perturbation_celloracle" / dataset
        combination.append(pd.read_csv(combination_dir / "donor_state_combination_effects.csv"))
        single.append(pd.read_csv(single_dir / "donor_state_perturbation_effects.csv"))
        status.append(pd.read_csv(combination_dir / "combination_status.csv"))
    return (
        pd.concat(combination, ignore_index=True),
        pd.concat(single, ignore_index=True),
        pd.concat(status, ignore_index=True),
    )


def paired_effects(combination: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    combo = combination.loc[
        combination["n_propagation"].eq(3)
        & combination["endpoint_variant"].eq("heldout_direct_targets")
        & combination["state"].eq("decoupled")
        & combination["endpoint"].isin(COMMON_ENDPOINTS)
    ].copy()
    anchor = single.loc[
        single["candidate"].eq("STAT1")
        & single["n_propagation"].eq(3)
        & single["endpoint_variant"].eq("heldout_direct_targets")
        & single["state"].eq("decoupled")
        & single["endpoint"].isin(COMMON_ENDPOINTS)
    ].copy()
    anchor = anchor[[
        "dataset", "subject_id", "network_model", "endpoint", "mean_standardized_delta"
    ]].rename(columns={"mean_standardized_delta": "anchor_effect"})
    combo = combo.rename(columns={"mean_standardized_delta": "combination_effect"})
    paired = combo.merge(
        anchor,
        on=["dataset", "subject_id", "network_model", "endpoint"],
        how="inner",
        validate="many_to_one",
    )
    paired["rescue_delta_vs_STAT1_KO"] = paired["combination_effect"] - paired["anchor_effect"]
    return paired


def cohort_tests(paired: pd.DataFrame, seed: int) -> pd.DataFrame:
    observed = paired.loc[paired["network_model"].eq("observed_GRN")].copy()
    keys = ["dataset", "intervention", "endpoint"]
    rows = []
    for counter, (key, frame) in enumerate(observed.groupby(keys, observed=True)):
        rescue = frame["rescue_delta_vs_STAT1_KO"].to_numpy(float)
        combo_effect = frame["combination_effect"].to_numpy(float)
        anchor_effect = frame["anchor_effect"].to_numpy(float)
        rescue_p_two, method = summary.sign_flip_test(rescue, seed + counter)
        combo_p_two, _ = summary.sign_flip_test(combo_effect, seed + 1000 + counter)
        endpoint = key[2]
        expected_combo_sign = -1 if endpoint == "axis2" else 1
        rescue_expected_sign = 0 if endpoint == "axis2" else 1
        rows.append({
            "dataset": key[0],
            "intervention": key[1],
            "endpoint": endpoint,
            "n_donors": len(frame),
            "mean_combination_effect": float(combo_effect.mean()),
            "mean_STAT1_KO_effect": float(anchor_effect.mean()),
            "mean_rescue_delta_vs_STAT1_KO": float(rescue.mean()),
            "median_rescue_delta_vs_STAT1_KO": float(np.median(rescue)),
            "positive_rescue_fraction": float(np.mean(rescue > 0)),
            "combination_direction_success": bool(combo_effect.mean() * expected_combo_sign > 0),
            "combination_sign_flip_p_expected_one_sided": summary.expected_one_sided_p(
                combo_p_two, combo_effect.mean(), expected_combo_sign
            ),
            "rescue_direction_success": bool(rescue.mean() > 0) if rescue_expected_sign else np.nan,
            "rescue_sign_flip_p_expected_one_sided": (
                summary.expected_one_sided_p(rescue_p_two, rescue.mean(), rescue_expected_sign)
                if rescue_expected_sign else np.nan
            ),
            "axis2_retention_ratio": (
                float(abs(combo_effect.mean()) / abs(anchor_effect.mean()))
                if endpoint == "axis2" and abs(anchor_effect.mean()) > 1e-12 else np.nan
            ),
            "test_method": method,
        })
    return pd.DataFrame(rows)


def shuffled_rescue_tests(paired: pd.DataFrame) -> pd.DataFrame:
    network_means = (
        paired.groupby(["dataset", "intervention", "endpoint", "network_model"], observed=True)
        ["rescue_delta_vs_STAT1_KO"]
        .mean()
        .reset_index(name="mean_network_rescue")
    )
    rows = []
    for key, frame in network_means.groupby(["dataset", "intervention", "endpoint"], observed=True):
        observed = frame.loc[frame["network_model"].eq("observed_GRN"), "mean_network_rescue"]
        shuffled = frame.loc[
            frame["network_model"].str.startswith("target_shuffled_GRN_seed_"),
            "mean_network_rescue",
        ].to_numpy(float)
        if len(observed) != 1 or len(shuffled) == 0:
            continue
        value = float(observed.iloc[0])
        null_sd = float(shuffled.std(ddof=1))
        rows.append({
            "dataset": key[0],
            "intervention": key[1],
            "endpoint": key[2],
            "observed_mean_rescue": value,
            "n_shuffled_GRNs": len(shuffled),
            "shuffled_mean_rescue": float(shuffled.mean()),
            "shuffled_sd_rescue": null_sd,
            "shuffled_q025": float(np.quantile(shuffled, 0.025)),
            "shuffled_q975": float(np.quantile(shuffled, 0.975)),
            "network_specific_rescue_z": (
                (value - shuffled.mean()) / null_sd if null_sd > 1e-12 else np.nan
            ),
            "empirical_upper_p": float((np.sum(shuffled >= value - 1e-12) + 1) / (len(shuffled) + 1)),
            "empirical_lower_p": float((np.sum(shuffled <= value + 1e-12) + 1) / (len(shuffled) + 1)),
        })
    return pd.DataFrame(rows)


def meta_analyze(cohort: pd.DataFrame, shuffled: pd.DataFrame) -> pd.DataFrame:
    merged = cohort.merge(
        shuffled,
        on=["dataset", "intervention", "endpoint"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for key, frame in merged.groupby(["intervention", "endpoint"], observed=True):
        endpoint = key[1]
        if endpoint == "axis2":
            p_values = np.clip(frame["combination_sign_flip_p_expected_one_sided"].to_numpy(float), 1e-12, 1)
            direction = frame["combination_direction_success"].to_numpy(bool)
            signed_z = np.where(direction, stats.norm.isf(p_values), -stats.norm.isf(1 - p_values))
            expected_label = "retain_negative_axis2"
            effect_column = "mean_combination_effect"
        else:
            p_values = np.clip(frame["rescue_sign_flip_p_expected_one_sided"].to_numpy(float), 1e-12, 1)
            direction = frame["rescue_direction_success"].astype(bool).to_numpy()
            signed_z = np.where(direction, stats.norm.isf(p_values), -stats.norm.isf(1 - p_values))
            expected_label = "positive_lam2_rescue_vs_STAT1_KO"
            effect_column = "mean_rescue_delta_vs_STAT1_KO"
        weights = np.sqrt(frame["n_donors"].to_numpy(float))
        combined_z = float(np.sum(weights * signed_z) / np.sqrt(np.sum(weights**2)))
        external = frame.loc[frame["dataset"].eq(EXTERNAL_DATASET)].iloc[0]
        rows.append({
            "intervention": key[0],
            "endpoint": endpoint,
            "criterion": expected_label,
            "n_cohorts": len(frame),
            "n_donors": int(frame["n_donors"].sum()),
            "n_direction_success_cohorts": int(direction.sum()),
            "direction_concordant": bool(direction.all()),
            "weighted_mean_effect": float(np.average(frame[effect_column], weights=weights)),
            "combined_expected_z": combined_z,
            "stouffer_expected_one_sided_p": float(stats.norm.sf(combined_z)),
            "external_direction_success": bool(external[
                "combination_direction_success" if endpoint == "axis2" else "rescue_direction_success"
            ]),
            "external_effect": float(external[effect_column]),
            "external_empirical_network_p": float(external[
                "empirical_lower_p" if endpoint == "axis2" else "empirical_upper_p"
            ]),
            "minimum_axis2_retention_ratio": (
                float(frame["axis2_retention_ratio"].min()) if endpoint == "axis2" else np.nan
            ),
        })
    output = pd.DataFrame(rows)
    output["fdr_bh_combination_criteria"] = multipletests(
        output["stouffer_expected_one_sided_p"], method="fdr_bh"
    )[1]
    return output


def decisions(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for intervention, frame in meta.groupby("intervention", observed=True):
        axis = frame.loc[frame["endpoint"].eq("axis2")].iloc[0]
        rescue = frame.loc[frame["endpoint"].eq("lam2_complement_apc")].iloc[0]
        axis_pass = bool(
            axis["direction_concordant"]
            and axis["fdr_bh_combination_criteria"] < 0.10
            and axis["minimum_axis2_retention_ratio"] >= 0.80
        )
        rescue_pass = bool(
            rescue["direction_concordant"]
            and rescue["fdr_bh_combination_criteria"] < 0.10
            and rescue["external_direction_success"]
            and rescue["external_empirical_network_p"] <= 0.10
        )
        rows.append({
            "intervention": intervention,
            "axis2_retention_pass": axis_pass,
            "minimum_axis2_retention_ratio": axis["minimum_axis2_retention_ratio"],
            "lam2_rescue_direction_success_cohorts": int(rescue["n_direction_success_cohorts"]),
            "lam2_rescue_fdr": rescue["fdr_bh_combination_criteria"],
            "external_lam2_rescue_effect": rescue["external_effect"],
            "external_lam2_rescue_empirical_p": rescue["external_empirical_network_p"],
            "lam2_rescue_pass": rescue_pass,
            "decision": (
                "advance_compensatory_combination" if axis_pass and rescue_pass
                else "axis2_retained_but_no_replicated_lam2_rescue" if axis_pass
                else "do_not_advance"
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["lam2_rescue_pass", "axis2_retention_pass", "lam2_rescue_direction_success_cohorts"],
        ascending=False,
    )


def write_report(output: Path, meta: pd.DataFrame, decision: pd.DataFrame) -> None:
    lines = [
        "# Compensatory combination perturbation report",
        "",
        "## Question",
        "",
        "Can a terminal-adaptation TF overexpression rescue the LAM2 cost of STAT1 knockout "
        "without sacrificing its inflammatory Axis2 suppression? Combination and STAT1-KO "
        "effects are compared within the same donor and the same held-out Axis2/LAM2 gene sets. "
        "Fifty target-shuffled GRNs per cohort define the network rescue null.",
        "",
        "## Decisions",
        "",
    ]
    for row in decision.itertuples():
        lines.append(
            f"- {row.intervention}: **{row.decision}**; minimum Axis2 retention="
            f"{row.minimum_axis2_retention_ratio:.3f}; LAM2 rescue direction success="
            f"{row.lam2_rescue_direction_success_cohorts}/3; external rescue="
            f"{row.external_lam2_rescue_effect:.4f}, empirical P={row.external_lam2_rescue_empirical_p:.3g}."
        )
    lines.extend(["", "## Criterion meta-analysis", ""])
    for row in meta.sort_values(["intervention", "endpoint"]).itertuples():
        lines.append(
            f"- {row.intervention} {row.criterion}: direction success="
            f"{row.n_direction_success_cohorts}/{row.n_cohorts}; weighted effect="
            f"{row.weighted_mean_effect:.4f}; z={row.combined_expected_z:.3f}, "
            f"P={row.stouffer_expected_one_sided_p:.3g}, FDR={row.fdr_bh_combination_criteria:.3g}; "
            f"external effect={row.external_effect:.4f}."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Failure to rescue does not negate STAT1 as the inflammatory-lock anchor. It falsifies the "
        "specific linear compensation hypothesis tested here: acute PPARG, MAF, or MAFB elevation "
        "within the same fitted state is not sufficient to restore the STAT1-dependent LAM2 arm. "
        "This redirects the next mechanism step toward state-conditional or upstream differentiation "
        "drivers rather than additive TF dosing.",
    ])
    (output / "CELLORACLE_COMPENSATORY_COMBINATION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "results/virtual_perturbation_celloracle_combinations"
    combination, single, status = load_results(root)
    paired = paired_effects(combination, single)
    cohort = cohort_tests(paired, args.seed)
    shuffled = shuffled_rescue_tests(paired)
    meta = meta_analyze(cohort, shuffled)
    decision = decisions(meta)

    combination.to_csv(output / "all_donor_state_combination_effects.csv.gz", index=False, compression="gzip")
    status.to_csv(output / "all_combination_status.csv", index=False)
    paired.to_csv(output / "paired_combination_vs_STAT1_KO_effects.csv.gz", index=False, compression="gzip")
    cohort.to_csv(output / "combination_cohort_tests.csv", index=False)
    shuffled.to_csv(output / "combination_shuffled_GRN_rescue_tests.csv", index=False)
    meta.to_csv(output / "combination_criterion_meta_analysis.csv", index=False)
    decision.to_csv(output / "combination_decisions.csv", index=False)
    write_report(output, meta, decision)
    print(decision.to_string(index=False), flush=True)
    print("\nCriterion meta-analysis\n", meta.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
