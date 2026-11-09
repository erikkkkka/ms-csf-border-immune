# MS CSF Border Immune Analysis

Analysis code and processed data for **Inflammatory activation is uncoupled from BAM-like differentiation in multiple sclerosis cerebrospinal fluid**.

## Download

Use the [download manifest](https://github.com/erikkkkka/ms-csf-border-immune/releases/latest/download/DOWNLOAD_MANIFEST.csv) to obtain the processed cohort archives, analysis results, figure source data and code archives. Extract the archives into the repository root, preserving their directory paths. The manifest lists download URLs and SHA-256 checksums.

```sh
python release_tools/verify_release.py --root .
python release_tools/verify_release.py --root . --require-data
```

## Reproduce

Install the environments in `environments/`. For figure regeneration, extract `Processed_GSE138266.zip` and `Processed_GSE328265.zip` first.

```sh
python release_tools/reproduce_public.py --mode figures
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

## Data and licensing

See the [data dictionary](docs/DATA_DICTIONARY.md) and [source manifest](reproducibility/DATA_SOURCE_MANIFEST.csv) for file definitions and accessions. Licensing terms are listed in [Copyright Notice](COPYRIGHT_NOTICE.txt) and [Third-party Notices](docs/THIRD_PARTY_NOTICES.md).
