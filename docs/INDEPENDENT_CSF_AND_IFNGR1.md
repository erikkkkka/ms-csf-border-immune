# Independent CSF cohort and IFNGR1 perturbation

| Input | Source | SHA-256 |
|---|---|---|
| `expr_csf_annot_clean_rc5.h5ad` | [Ban et al.](https://doi.org/10.1093/brain/awad404), [processed CSF data](https://sourceforge.net/projects/ms-csf-singlecell/files/expr_csf_annot_clean_rc5.h5ad/download) | `1dcf58dd230b68bcbd5ff67de8dfd9f528229e245a4503538554da8e015480fd` |
| `expr_csf_annot_clean_rc5_obs.csv.gz` | [Ban et al., cell annotations](https://sourceforge.net/projects/ms-csf-singlecell/files/expr_csf_annot_clean_rc5_obs.csv.gz/download) | `c0beb2a01edb72d39fb21383edc6de68e94619d686fdf677844f38dc246bded1` |
| `GSE153056_ECCITE_metadata.tsv.gz` | [GSE153056 supplementary files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153056/suppl/) | `7059b0cb857dc344e6bb53dbdc5ccadccb25bb069372ed6f45eff46b7e992a76` |
| `GSM4633614_ECCITE_cDNA_counts.tsv.gz` | Member of `GSE153056_RAW.tar` in [GSE153056 supplementary files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153056/suppl/) | `2acb14a10b95f9b39536d0aec42896f30f6256332ccd336a2b01d037316b6080` |

Use the full Ban H5AD with the `counts_soupx` and `counts` layers. The associated raw sequencing study is EGAS00001007478. The THP-1 data are from [Papalexi et al.](https://doi.org/10.1038/s41588-021-00778-2). The deposited `TAP2.1` feature corresponds to ENSG00000204267 in the hg19 reference. The [feature mapping](../results/external_validation/GSE153056/feature_mapping.json) lists additional aliases.

The analysis uses the [fixed differentiation program](../results/external_validation/GSE328265_two_axis_projection/frozen_trajectory_program.csv) and the [receptor signature ledgers](../results/in_vivo_crispr_bridge) derived from [de la Rosa et al.](https://doi.org/10.1038/s41593-025-02151-6). The [fixed IFNGR1 knockout panels](../results/external_validation/Ban/frozen_panels.csv) specify the gene weights for each endpoint.

## Reproduce

From the repository root, use Python 3.12 and install the requirements. The plotting command uses the summary tables for Figure 3. Panel a displays medians and interquartile ranges. Running the Ban analysis from the source files adds donor points from the locally generated results.

```sh
python -m pip install -r environments/independent_csf_requirements.txt
python scripts/plot_independent_csf_and_ifngr1.py
```

Place the Ban source files in `data/external_validation/Ban/`.

```sh
ban_data=data/external_validation/Ban
python scripts/analyze_ban_csf.py primary \
  --matrix "$ban_data/expr_csf_annot_clean_rc5.h5ad" \
  --annotations "$ban_data/expr_csf_annot_clean_rc5_obs.csv.gz"
python scripts/analyze_ban_csf.py primary --raw-umi \
  --matrix "$ban_data/expr_csf_annot_clean_rc5.h5ad" \
  --annotations "$ban_data/expr_csf_annot_clean_rc5_obs.csv.gz"
python scripts/analyze_ban_csf.py signature
python scripts/analyze_ifngr1_signatures.py ban
python scripts/analyze_ban_sensitivity.py
```

Place `GSE153056_RAW.tar` and the THP-1 metadata in `data/external_validation/GSE153056/`.

```sh
thp_data=data/external_validation/GSE153056
metadata="$thp_data/GSE153056_ECCITE_metadata.tsv.gz"
counts="$thp_data/GSM4633614_ECCITE_cDNA_counts.tsv.gz"
tar -xOf "$thp_data/GSE153056_RAW.tar" GSM4633614_ECCITE_cDNA_counts.tsv.gz > "$counts"
python scripts/analyze_gse153056.py "$metadata" "$counts" \
  results/external_validation/GSE153056/mhc_i_pseudobulk.json
python scripts/analyze_ifngr1_signatures.py extract --metadata "$metadata" --counts "$counts"
python scripts/analyze_ifngr1_signatures.py thp1
```

## Figure 3 source data

| Panels | Files |
|---|---|
| a–b | [Ban inflammatory index summary](../results/external_validation/Ban/inflammatory_index_summary.csv), [primary and held-out comparisons](../results/external_validation/Ban/primary_results.json), [raw UMI](../results/external_validation/Ban/raw_umi/primary_results.json), [including treated donors](../results/external_validation/Ban/including_treated_sensitivity.json) |
| c–d | [MHC-I pseudobulk contrasts](../results/external_validation/GSE153056/mhc_i_pseudobulk.json), [knockout signature contrasts](../results/external_validation/GSE153056/knockout_signature/guide_and_replicate_contrasts.csv) |
| e | [Ban signature associations](../results/external_validation/Ban/signature_results.json), [component associations](../results/external_validation/Ban/signature_components/results.json) |
| f–h | [Rank scoring and cell-type comparisons](../results/external_validation/Ban/sensitivity/robustness_summary.csv), [donor omission ranges](../results/external_validation/Ban/sensitivity/donor_omission_ranges.csv) |

The primary Ban comparison includes 13 untreated MS and 10 non-inflammatory neurological disease controls with at least 20 retained monocytes or macrophages per donor. Signature associations are summarized across donors. Omission ranges describe estimates after refitting. The THP-1 analysis pairs targeting and non-targeting groups within three transductions. Guide comparisons within each transduction share a control.
