# MS CSF Border Immune Analysis

Analysis code and processed data supporting **Inflammatory activation is uncoupled from
BAM-like differentiation in multiple sclerosis cerebrospinal fluid**.

The `v0.1.1` release contains analysis scripts, complete statistical results, and
figure source tables. It updates the method-matched trajectory AUCs, regulatory
network descriptions, and supplementary exports described in `docs/ANALYSIS_SCOPE.md`.

## Download

The small files are tracked in this repository and packaged in the
[v0.1.1 release](https://github.com/erikkkkka/ms-csf-border-immune/releases/tag/v0.1.1).
The four unchanged processed-cohort archives remain attached to
[v0.1.0](https://github.com/erikkkkka/ms-csf-border-immune/releases/tag/v0.1.0).
The current download manifest links to the appropriate version of every archive.

| Asset | Contents |
|---|---|
| `Processed_GSE138266.zip` | DecontX matrices, cleaned all-cell/APC objects, continuum object, CellOracle input |
| `Processed_GSE194078.zip` | DecontX matrices, cleaned all-cell/APC objects, continuum object |
| `Processed_GSE286068.zip` | DecontX matrices, cleaned all-cell/APC objects, continuum object, CellOracle input |
| `Processed_GSE328265.zip` | DecontX matrices, cleaned annotation object, external two-axis object, CellOracle input |
| `Analysis_Results_and_Metadata.zip` | All released result tables, QC, annotations, sample metadata, and supplementary workbooks |
| `Figure_Source_Data_and_PDF.zip` | 61 source-data CSVs and 14 figure PDFs |
| `Analysis_Code_and_Environment.zip` | Analysis scripts, reproduction tools, environment records, and instructions |

Extract data assets into the repository root, preserving their directory paths. The
archives do not contain another enclosing project directory. Validate downloads with
`reproducibility/RELEASE_ASSET_MANIFEST.csv`; individual processed-file hashes are in
`reproducibility/PROCESSED_DATA_MANIFEST.csv`.
The release attachment `DOWNLOAD_MANIFEST.csv` lists all seven ZIP download URLs;
`SHA256SUMS.txt` provides their checksums. The four processed archives retain the
same file sizes and SHA-256 digests as in `v0.1.0`.

```bash
python release_tools/verify_release.py --root .
python release_tools/verify_release.py --root . --require-data
```

## Reproduce

The released results support figure regeneration without rerunning the complete
single-cell or perturbation analyses. Main Figures 1 and 2 also read the continuum
and two-axis AnnData objects; extract `Processed_GSE138266.zip` and
`Processed_GSE328265.zip` into the repository root first. Install the recorded main
environment, then run:

```bash
python release_tools/reproduce_public.py --mode figures --dry-run
python release_tools/reproduce_public.py --mode figures
```

To rebuild downstream analyses from the cleaned-cell checkpoints, first obtain the
external inputs described in `docs/EXTERNAL_INPUTS.md` and configure all four runtimes:

```bash
python release_tools/reproduce_public.py --mode analysis --dry-run
python release_tools/reproduce_public.py --mode analysis \
  --main-python .conda_trajectory/bin/python \
  --celloracle-python .conda_celloracle/bin/python \
  --docking-python .conda_docking/bin/python \
  --rscript Rscript
```

Scripts `01` through `50` cover upstream preprocessing and exploratory analyses.
The downstream analysis order is recorded in `reproducibility/ANALYSIS_WORKFLOW.json`;
`reproducibility/RUN_ORDER.csv` also lists figure, table, and validation steps.

## Cohorts

| Dataset | Role |
|---|---|
| GSE138266 | Paired CSF/PBMC trajectory and disease discovery |
| GSE194078 | Independent paired CSF/PBMC trajectory; excluded from disease contrasts |
| GSE286068 | Independent CSF trajectory and disease discovery |
| GSE328265 | External projection using newly sequenced GSE327243/GSE328264 subjects |

GSE328265 includes 31 technical matrices from 30 new subjects. The two SPMS_4 technical
libraries are pooled within one donor for inference. Previously published subjects
are not included in this external checkpoint. Donors, not cells, are the independent
units for disease comparisons and simulated-perturbation tests.

## Contents and interpretation

- `metadata/`: source accessions, pseudonymous donor IDs, and sample descriptions.
- `results/`: complete result tables and scientific quality-control records.
- `figures_main/` and `figures_extended/`: figure source tables and PDFs.
- `scripts/`: original Python and R analysis and plotting scripts.
- `environments/` and `reproducibility/`: package versions, execution order, and checksums.
- `release_tools/`: integrity checks and a configurable reproduction entrypoint.

Read `docs/DATA_DICTIONARY.md` before interpreting `.h5ad` matrices. In particular,
AnnData `.raw` contains normalized expression, not necessarily unnormalized counts,
and some `layers['counts']` contain only highly variable genes. Full-gene DecontX
counts are supplied separately as Matrix Market files.

Original GEO data remain at their original accessions. Third-party publications and
full reference databases are not mirrored. Source URLs and identifiers are recorded
in `reproducibility/DATA_SOURCE_MANIFEST.csv` and `docs/EXTERNAL_INPUTS.md`.

## License and citation

See `COPYRIGHT_NOTICE.txt` for author-controlled materials and
`docs/THIRD_PARTY_NOTICES.md` for third-party exclusions and separate licenses.
Cite this repository with its release tag and commit identifier.
