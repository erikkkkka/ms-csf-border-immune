# MS CSF Border Immune Analysis

Analysis code and processed data for **Excess inflammatory activity along a cerebrospinal fluid myeloid continuum in multiple sclerosis**.

## Download

Use the [download manifest](https://github.com/erikkkkka/ms-csf-border-immune/releases/latest/download/DOWNLOAD_MANIFEST.csv) to obtain the processed cohort archives, analysis results, figure source data and code archives. Extract the data archives into the repository root, preserving their directory paths and keeping files already present in the checkout. The manifest lists download URLs and SHA-256 checksums.

```sh
python release_tools/verify_release.py --root .
python release_tools/verify_release.py --root . --require-data
```

## Reproduce

Install the environments in `environments/`. For figure regeneration, extract the results archive, `Processed_GSE138266.zip` and `Processed_GSE328265.zip` first.

```sh
python release_tools/reproduce_public.py --mode figures
```

This command generates Figures 1–6 in `reproduced_figures/main/` and Figures S1–S9 in `reproduced_figures/supplementary/`, using the manuscript numbering. Figure 3 uses the deposited summary tables and adds the donor points when local donor scores are available. Figure S10 uses the genetic analysis workflow below.

To generate the four-panel Figure 1 alone:

```sh
python scripts/75_generate_manuscript_figures.py --figures 1 --formats pdf \
  --output-dir reproduced_figures/main --source-dir qa_generated/source_data/main
```

To rerun the downstream analyses from the cleaned-cell checkpoints, obtain the [external inputs](docs/EXTERNAL_INPUTS.md) and run:

```sh
python release_tools/reproduce_public.py --mode analysis \
  --main-python .conda_trajectory/bin/python \
  --celloracle-python .conda_celloracle/bin/python \
  --docking-python .conda_docking/bin/python \
  --rscript Rscript
```

See [Genetic analysis of monocyte gene expression](genetic_analysis/README.md) for Mendelian randomization, colocalization and Supplementary Figure S10.

## Independent CSF cohort and IFNGR1 perturbation

[Inputs, reproduction commands and Figure 3 source data](docs/INDEPENDENT_CSF_AND_IFNGR1.md) cover the [Ban cohort](results/external_validation/Ban) and [GSE153056 THP-1 perturbations](results/external_validation/GSE153056). Ban results include the donor inflammatory index, IFNGR1 signature associations, rank scoring, cell-type analyses and donor omission.

## Additional results

- [GSE133028](results/external_validation/GSE133028) contains donor inflammatory indices and the comparison between relapsing–remitting MS and healthy controls. The comparison at a minimum of 20 cells per donor is descriptive.
- [GSE308973](results/external_validation/GSE308973) contains donor-level trajectory scores and paired summaries for CSF macrophages and monocytes. The paired comparisons include donors with at least five cells per state.
- [Disease comparisons](results/two_axis/three_cohort_disease_meta_analysis.csv), [cohort estimates](results/two_axis/cohort_disease_effects.csv) and [cutoff analysis](results/two_axis/inflammation_cutoff_grid.csv) accompany Figure 2 and Supplementary Figure S4.
- [Network simulation results](results/virtual_perturbation_celloracle/shuffled_network_tests.csv) and the [shared-gene combination effects](results/virtual_perturbation_celloracle_combinations/shared_gene_combination_effects.csv) accompany Figure 5e and Supplementary Figure S8.

## Data and licensing

See the [data dictionary](docs/DATA_DICTIONARY.md) and [source manifest](reproducibility/DATA_SOURCE_MANIFEST.csv) for file definitions and accessions. Licensing terms are listed in [Copyright Notice](COPYRIGHT_NOTICE.txt) and [Third-party Notices](docs/THIRD_PARTY_NOTICES.md).
