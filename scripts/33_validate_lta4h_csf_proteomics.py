#!/usr/bin/env python3
"""Validate the prespecified LTA4H target in public MS CSF proteomics."""

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


TARGET_ACCESSION = "P09960"
TARGET_GENE = "LTA4H"
DIAGNOSES = ["Neurological Control", "MS", "OtherAutoimmune", "Infectious"]
STRONG_BLOOD = {"++", "+++", "bloody"}


def extract_target_row(path: Path) -> tuple[dict, pd.DataFrame]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        target = next(
            (
                row
                for row in reader
                if row[0] == TARGET_ACCESSION
                or row[3] == TARGET_GENE
                or TARGET_GENE in row[3].split(";")
            ),
            None,
        )
    if target is None:
        raise RuntimeError(f"{TARGET_GENE} ({TARGET_ACCESSION}) is absent from {path}")
    if len(header) != len(target):
        raise RuntimeError(
            f"Header/row length mismatch in {path}: {len(header)} vs {len(target)}"
        )
    metadata = dict(zip(header[:8], target[:8]))
    values = pd.DataFrame(
        {
            "Run_Astral_Measurement": header[8:],
            "LTA4H_log2_intensity": pd.to_numeric(target[8:], errors="coerce"),
        }
    )
    return metadata, values


def prepare_astral_cohort(
    matrix_path: Path, annotations: pd.DataFrame, cohort: str
) -> tuple[pd.DataFrame, dict]:
    target_metadata, values = extract_target_row(matrix_path)
    merged = values.merge(
        annotations,
        on="Run_Astral_Measurement",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), "Run_Astral_Measurement"]
        raise RuntimeError(f"Unmatched sample annotations in {cohort}: {missing.tolist()[:5]}")
    merged = merged.drop(columns="_merge")
    merged.insert(0, "proteomics_cohort", cohort)

    # The public matrices retain pools and samples outside the author's Astral
    # discovery subset. Reproduce the released R code's biological sample set.
    biological = merged[merged["Diagnosis_group"].isin(DIAGNOSES)].copy()
    if cohort == "astral_discovery":
        subset = biological["Stratified_inflammatory_subset"].notna()
        transferred = biological["Stratified_inflammatory_subset"].eq(
            "transferred_to_replication"
        )
        biological = biological.loc[subset & ~transferred].copy()

    for column in [
        "Age",
        "QAlb",
        "Diff_age_adj_QAlb_log2",
        "Leukocyte_count_log2",
    ]:
        biological[column] = pd.to_numeric(biological[column], errors="coerce")
    biological["erythrocyte_present"] = biological["Erythrocytes_in_CSF"].ne("0").astype(int)
    biological["Sex_M"] = biological["Sex"].eq("M").astype(int)
    biological["MS_indicator"] = biological["Diagnosis_group"].eq("MS").astype(int)
    biological["other_autoimmune_indicator"] = biological["Diagnosis_group"].eq(
        "OtherAutoimmune"
    ).astype(int)
    biological["infectious_indicator"] = biological["Diagnosis_group"].eq(
        "Infectious"
    ).astype(int)
    biological["replication_de_novo"] = biological["ID_MAIN_RP"].fillna("").eq("")

    qc = biological[
        biological["LTA4H_log2_intensity"].notna()
        & biological["Age"].notna()
        & biological["Sex"].isin(["F", "M"])
        & biological["QAlb"].gt(0)
        & biological["Diff_age_adj_QAlb_log2"].notna()
        & biological["Erythrocytes_in_CSF"].notna()
        & ~biological["Erythrocytes_in_CSF"].isin(STRONG_BLOOD)
        & biological["Sample_plate"].notna()
    ].copy()

    audit = {
        "proteomics_cohort": cohort,
        "matrix_sample_columns": len(values),
        "LTA4H_quantified_matrix": int(values["LTA4H_log2_intensity"].notna().sum()),
        "annotation_matches": len(merged),
        "biological_samples_before_qc": len(biological),
        "analysis_samples_after_qc": len(qc),
        "excluded_missing_LTA4H": int(biological["LTA4H_log2_intensity"].isna().sum()),
        "excluded_missing_or_invalid_covariates": int(len(biological) - len(qc)),
        "protein_accession": target_metadata["protein"],
        "protein_name": target_metadata["Protein.Names"],
        "gene": target_metadata["Genes"],
        "matrix_observations": int(float(target_metadata["observations_Astral"])),
        "matrix_data_completeness": float(target_metadata["data_completeness_Astral"]),
    }
    return qc, audit


def robust_ols(
    frame: pd.DataFrame,
    formula: str,
    term: str,
    analysis: str,
    cohort: str,
    contrast: tuple[str, str] | None = None,
) -> dict:
    model = smf.ols(formula, data=frame).fit(cov_type="HC3")
    if contrast is None:
        estimate = float(model.params[term])
        se = float(model.bse[term])
        p_value = float(model.pvalues[term])
    else:
        left, right = contrast
        estimate = float(model.params[left] - model.params[right])
        covariance = model.cov_params()
        variance = (
            covariance.loc[left, left]
            + covariance.loc[right, right]
            - 2 * covariance.loc[left, right]
        )
        se = float(np.sqrt(variance))
        p_value = float(2 * stats.norm.sf(abs(estimate / se)))
    return {
        "proteomics_cohort": cohort,
        "analysis": analysis,
        "term_or_contrast": term if contrast is None else f"{contrast[0]}-{contrast[1]}",
        "n_samples": int(model.nobs),
        "estimate_log2": estimate,
        "standard_error": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "fold_change": 2**estimate,
        "p_value": p_value,
        "r_squared": float(model.rsquared),
        "covariance": "HC3",
        "formula": formula,
    }


def disease_models(frame: pd.DataFrame, cohort: str) -> list[dict]:
    rows = []
    primary_formula = (
        "LTA4H_log2_intensity ~ Age + Sex_M + Diff_age_adj_QAlb_log2 + "
        "erythrocyte_present + MS_indicator + other_autoimmune_indicator + "
        "infectious_indicator + C(Sample_plate)"
    )
    rows.append(
        robust_ols(frame, primary_formula, "MS_indicator", "MS_vs_neurological_control", cohort)
    )
    rows.append(
        robust_ols(
            frame,
            primary_formula,
            "MS_vs_other_autoimmune",
            "MS_vs_other_autoimmune",
            cohort,
            contrast=("MS_indicator", "other_autoimmune_indicator"),
        )
    )
    rows.append(
        robust_ols(
            frame,
            primary_formula,
            "MS_vs_infectious",
            "MS_vs_infectious",
            cohort,
            contrast=("MS_indicator", "infectious_indicator"),
        )
    )

    leukocyte_frame = frame.dropna(subset=["Leukocyte_count_log2"])
    if len(leukocyte_frame) >= 20:
        rows.append(
            robust_ols(
                leukocyte_frame,
                primary_formula + " + Leukocyte_count_log2",
                "MS_indicator",
                "MS_vs_neurological_control_leukocyte_adjusted",
                cohort,
            )
        )
        ms = leukocyte_frame[leukocyte_frame["Diagnosis_group"].eq("MS")]
        if len(ms) >= 20:
            within_ms_formula = (
                "LTA4H_log2_intensity ~ Age + Sex_M + Diff_age_adj_QAlb_log2 + "
                "erythrocyte_present + Leukocyte_count_log2 + C(Sample_plate)"
            )
            rows.append(
                robust_ols(
                    ms,
                    within_ms_formula,
                    "Leukocyte_count_log2",
                    "within_MS_leukocyte_association",
                    cohort,
                )
            )

    ms = frame[frame["Diagnosis_group"].eq("MS")].copy()
    subtype = ms[ms["RMS_PMS"].isin(["RRMS", "PMS"])].copy()
    if subtype["RMS_PMS"].nunique() == 2 and subtype["RMS_PMS"].value_counts().min() >= 4:
        subtype["PMS_indicator"] = subtype["RMS_PMS"].eq("PMS").astype(int)
        rows.append(
            robust_ols(
                subtype,
                "LTA4H_log2_intensity ~ Age + Sex_M + Diff_age_adj_QAlb_log2 + "
                "erythrocyte_present + PMS_indicator + C(Sample_plate)",
                "PMS_indicator",
                "PMS_vs_RRMS",
                cohort,
            )
        )

    ocb = ms[ms["Oligoclonal_bands"].isin(["positive", "negative"])].copy()
    if ocb["Oligoclonal_bands"].nunique() == 2 and ocb["Oligoclonal_bands"].value_counts().min() >= 4:
        ocb["OCB_positive"] = ocb["Oligoclonal_bands"].eq("positive").astype(int)
        rows.append(
            robust_ols(
                ocb,
                "LTA4H_log2_intensity ~ Age + Sex_M + Diff_age_adj_QAlb_log2 + "
                "erythrocyte_present + OCB_positive + C(Sample_plate)",
                "OCB_positive",
                "OCB_positive_vs_negative_within_MS",
                cohort,
            )
        )
    return rows


def unadjusted_tests(frame: pd.DataFrame, cohort: str) -> list[dict]:
    rows = []
    for comparison, reference in [
        ("MS", "Neurological Control"),
        ("MS", "OtherAutoimmune"),
        ("MS", "Infectious"),
    ]:
        case = frame.loc[frame["Diagnosis_group"].eq(comparison), "LTA4H_log2_intensity"]
        control = frame.loc[frame["Diagnosis_group"].eq(reference), "LTA4H_log2_intensity"]
        test = stats.mannwhitneyu(case, control, alternative="two-sided")
        rows.append(
            {
                "proteomics_cohort": cohort,
                "comparison": f"{comparison}_vs_{reference}",
                "n_case": len(case),
                "n_reference": len(control),
                "median_case": case.median(),
                "median_reference": control.median(),
                "median_difference_log2": case.median() - control.median(),
                "p_value": test.pvalue,
                "test": "two_sided_mann_whitney",
            }
        )
    return rows


def summarize_groups(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    summary = (
        frame.groupby("Diagnosis_group", observed=True)["LTA4H_log2_intensity"]
        .agg(n="size", mean="mean", sd="std", median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
        .reset_index()
    )
    summary.insert(0, "proteomics_cohort", cohort)
    return summary


def bh_adjust(values: pd.Series) -> np.ndarray:
    p_values = values.to_numpy(dtype=float)
    order = np.argsort(p_values)
    adjusted = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    return restored


def fixed_and_random_meta(model_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = model_results[model_results["proteomics_cohort"].isin(
        ["astral_discovery", "astral_replication"]
    )]
    for analysis, data in main.groupby("analysis", observed=True):
        if len(data) != 2 or data["proteomics_cohort"].nunique() != 2:
            continue
        y = data["estimate_log2"].to_numpy()
        variance = data["standard_error"].to_numpy() ** 2
        weights = 1 / variance
        fixed = np.sum(weights * y) / np.sum(weights)
        fixed_se = np.sqrt(1 / np.sum(weights))
        q = np.sum(weights * (y - fixed) ** 2)
        df = len(y) - 1
        c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
        random_weights = 1 / (variance + tau2)
        random = np.sum(random_weights * y) / np.sum(random_weights)
        random_se = np.sqrt(1 / np.sum(random_weights))
        for method, estimate, se in [
            ("inverse_variance_fixed", fixed, fixed_se),
            ("dersimonian_laird_random", random, random_se),
        ]:
            rows.append(
                {
                    "analysis": analysis,
                    "method": method,
                    "n_cohorts": len(data),
                    "estimate_log2": estimate,
                    "standard_error": se,
                    "ci_low": estimate - 1.96 * se,
                    "ci_high": estimate + 1.96 * se,
                    "fold_change": 2**estimate,
                    "p_value": 2 * stats.norm.sf(abs(estimate / se)),
                    "cochran_q": q,
                    "q_df": df,
                    "q_p_value": stats.chi2.sf(q, df) if df > 0 else np.nan,
                    "i_squared_percent": max(0.0, (q - df) / q) * 100 if q > 0 else 0.0,
                    "tau_squared": tau2,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_bh_within_meta_method"] = np.nan
    for _, indices in result.groupby("method", observed=True).groups.items():
        result.loc[indices, "fdr_bh_within_meta_method"] = bh_adjust(
            result.loc[indices, "p_value"]
        )
    return result


def kosa_lta4h_tables(kosa_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_excel(kosa_dir / "Supplementary_Data_3.xlsx", header=None)
    hit = master.loc[master[2].astype(str).eq(TARGET_GENE)]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one Kosa2025 {TARGET_GENE} row, found {len(hit)}")
    row = hit.iloc[0]
    sections = pd.Series(master.iloc[0]).ffill()
    all_associations = pd.DataFrame(
        {
            "column_index": np.arange(master.shape[1]),
            "section": sections.astype(str),
            "metric": master.iloc[1].astype(str),
            "value": row.to_numpy(),
        }
    )

    snr = pd.read_excel(kosa_dir / "Supplementary_Data_1.xlsx", header=1)
    snr_row = snr.loc[snr["Gene"].eq(TARGET_GENE)].iloc[0]
    demographics = pd.read_excel(kosa_dir / "Supplementary_Data_2.xlsx", header=1)
    demo_row = demographics.loc[demographics["Gene"].eq(TARGET_GENE)].iloc[0]
    selected_indices = {
        "untreated_MS_age_spearman": 13,
        "untreated_MS_age_fdr": 14,
        "CEL_spearman": 34,
        "CEL_spearman_fdr": 35,
        "CEL_model_coefficient": 42,
        "CEL_model_coefficient_p": 43,
        "CEL_model_overall_p": 49,
        "T2LL_spearman": 50,
        "T2LL_spearman_fdr": 51,
        "T2LL_model_coefficient": 64,
        "T2LL_model_coefficient_p": 65,
        "CombiWISE_spearman": 72,
        "CombiWISE_spearman_fdr": 73,
        "brain_damage_spearman": 94,
        "brain_damage_spearman_fdr": 95,
        "brain_damage_model_coefficient": 108,
        "brain_damage_model_coefficient_p": 109,
        "brain_damage_model_overall_p": 115,
        "GMSD_spearman": 138,
        "GMSD_spearman_fdr": 139,
        "untreated_MS_minus_HC_median_difference": 160,
        "untreated_MS_vs_HC_wilcoxon_fdr": 161,
        "longitudinal_patient_slope_median": 164,
        "longitudinal_slope_wilcoxon_raw_p": 165,
        "longitudinal_slope_wilcoxon_fdr": 166,
    }
    concise = [
        {"evidence_domain": "assay", "metric": "SomaScan_SNR", "value": snr_row["Signal-to-Noise Ratio (SNR) "]},
        {"evidence_domain": "assay", "metric": "technical_residual_sd", "value": snr_row["Technical  residual standard deviation "]},
        {"evidence_domain": "assay", "metric": "clinical_residual_sd", "value": snr_row["Clinical residual standard deviation"]},
        {"evidence_domain": "healthy_demographic", "metric": "age_coefficient", "value": demo_row["Age coefficient"]},
        {"evidence_domain": "healthy_demographic", "metric": "age_fdr", "value": demo_row["Age p-value\n(FDR-adjusted)"]},
        {"evidence_domain": "healthy_demographic", "metric": "male_sex_coefficient", "value": demo_row["Male Sex coefficient"]},
        {"evidence_domain": "healthy_demographic", "metric": "male_sex_fdr", "value": demo_row["Male Sex p-value\n(FDR-adjusted)"]},
    ]
    for metric, index in selected_indices.items():
        domain = "disease_state" if index in [160, 161] else "longitudinal_or_activity"
        concise.append({"evidence_domain": domain, "metric": metric, "value": row.iloc[index]})
    concise = pd.DataFrame(concise)
    concise.insert(0, "study", "Kosa2025_NatureCommunications")
    return all_associations, concise


def file_manifest(paths_and_urls: list[tuple[Path, str]]) -> pd.DataFrame:
    rows = []
    for path, url in paths_and_urls:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "source_url": url,
            }
        )
    return pd.DataFrame(rows)


def plot_diagnoses(all_samples: pd.DataFrame, output: Path) -> None:
    order = ["Neurological Control", "MS", "OtherAutoimmune", "Infectious"]
    labels = ["Neuro control", "MS", "Other AIM", "Infectious"]
    cohorts = ["astral_discovery", "astral_replication"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    rng = np.random.default_rng(20260810)
    for ax, cohort in zip(axes, cohorts):
        frame = all_samples[all_samples["proteomics_cohort"].eq(cohort)]
        groups = [
            frame.loc[frame["Diagnosis_group"].eq(group), "LTA4H_log2_intensity"].to_numpy()
            for group in order
        ]
        ax.boxplot(groups, widths=0.55, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#d9e3e8", "edgecolor": "#344955"},
                   medianprops={"color": "#b23a48", "linewidth": 1.8})
        for position, values in enumerate(groups, start=1):
            if len(values) <= 300:
                keep = np.arange(len(values))
            else:
                keep = rng.choice(len(values), 300, replace=False)
            jitter = rng.normal(position, 0.055, len(keep))
            ax.scatter(jitter, values[keep], s=5, alpha=0.22, color="#2f5964", linewidths=0)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
        ax.set_title(cohort.replace("astral_", "Astral ").title())
        ax.set_ylabel("LTA4H log2 DirectLFQ intensity")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ms_csf_border_immune")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pxd_dir = root / "data" / "external_validation" / "PXD064570"
    kosa_dir = root / "data" / "external_validation" / "Kosa2025"
    out_dir = root / "results" / "external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(pxd_dir / "sample_annotations.tsv", sep="\t")
    annotations = annotations[annotations["Run_Astral_Measurement"].notna()].copy()
    if annotations["Run_Astral_Measurement"].duplicated().any():
        raise RuntimeError("Run_Astral_Measurement is not unique in public annotations")

    cohorts = []
    audits = []
    matrix_files = {
        "astral_discovery": pxd_dir / "Astral_discovery_protein_intensities.tsv",
        "astral_replication": pxd_dir / "Astral_replication_protein_intensities.tsv",
    }
    for cohort, matrix_path in matrix_files.items():
        frame, audit = prepare_astral_cohort(matrix_path, annotations, cohort)
        cohorts.append(frame)
        audits.append(audit)
    all_samples = pd.concat(cohorts, ignore_index=True)

    model_rows = []
    test_rows = []
    summaries = []
    for cohort, frame in all_samples.groupby("proteomics_cohort", observed=True):
        model_rows.extend(disease_models(frame, cohort))
        test_rows.extend(unadjusted_tests(frame, cohort))
        summaries.append(summarize_groups(frame, cohort))
        if cohort == "astral_replication":
            de_novo = frame[frame["replication_de_novo"]].copy()
            if len(de_novo) >= 20:
                model_rows.extend(disease_models(de_novo, "astral_replication_de_novo"))
                test_rows.extend(unadjusted_tests(de_novo, "astral_replication_de_novo"))
                summaries.append(summarize_groups(de_novo, "astral_replication_de_novo"))

    model_results = pd.DataFrame(model_rows)
    model_results["analysis_role"] = np.where(
        model_results["analysis"].eq("MS_vs_neurological_control"),
        "prespecified_primary",
        "secondary_or_sensitivity",
    )
    model_results["fdr_bh_within_cohort"] = np.nan
    for _, indices in model_results.groupby("proteomics_cohort", observed=True).groups.items():
        model_results.loc[indices, "fdr_bh_within_cohort"] = bh_adjust(
            model_results.loc[indices, "p_value"]
        )
    unadjusted = pd.DataFrame(test_rows)
    group_summary = pd.concat(summaries, ignore_index=True)
    meta = fixed_and_random_meta(model_results)
    kosa_all, kosa_concise = kosa_lta4h_tables(kosa_dir)

    all_samples.to_csv(
        out_dir / "PXD064570_LTA4H_sample_values.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(audits).to_csv(out_dir / "PXD064570_LTA4H_QC_audit.csv", index=False)
    group_summary.to_csv(out_dir / "PXD064570_LTA4H_group_summary.csv", index=False)
    unadjusted.to_csv(out_dir / "PXD064570_LTA4H_unadjusted_tests.csv", index=False)
    model_results.to_csv(out_dir / "PXD064570_LTA4H_disease_models.csv", index=False)
    meta.to_csv(out_dir / "PXD064570_LTA4H_meta_analysis.csv", index=False)
    kosa_all.to_csv(out_dir / "Kosa2025_LTA4H_all_associations.csv", index=False)
    kosa_concise.to_csv(out_dir / "Kosa2025_LTA4H_outcome_summary.csv", index=False)
    plot_diagnoses(all_samples, out_dir / "PXD064570_LTA4H_diagnosis_boxplot.png")

    manifest = file_manifest(
        [
            (pxd_dir / "sample_annotations.tsv", "https://osf.io/download/zdcp6/"),
            (matrix_files["astral_discovery"], "https://osf.io/download/tgx2n/"),
            (matrix_files["astral_replication"], "https://osf.io/download/skx4e/"),
            (kosa_dir / "Supplementary_Data_1.xlsx", "https://doi.org/10.1038/s41467-025-65154-8"),
            (kosa_dir / "Supplementary_Data_2.xlsx", "https://doi.org/10.1038/s41467-025-65154-8"),
            (kosa_dir / "Supplementary_Data_3.xlsx", "https://doi.org/10.1038/s41467-025-65154-8"),
        ]
    )
    manifest.to_csv(out_dir / "LTA4H_external_proteomics_file_manifest.csv", index=False)

    print("QC audit")
    print(pd.DataFrame(audits).to_string(index=False))
    print("\nPrespecified MS vs neurological control models")
    print(
        model_results.loc[
            model_results["analysis"].eq("MS_vs_neurological_control"),
            ["proteomics_cohort", "n_samples", "estimate_log2", "ci_low", "ci_high", "p_value"],
        ].to_string(index=False)
    )
    print("\nMeta-analysis")
    print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
