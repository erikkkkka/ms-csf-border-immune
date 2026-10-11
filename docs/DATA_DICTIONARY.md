# Data dictionary

## Identifiers and metadata

- `gse` / `dataset`: source study accession. Use this together with `subject_id`.
- `sample_accession`: GEO GSM identifier for one deposited technical matrix.
- `subject_id`: public, pseudonymous donor identifier; not globally unique across studies.
- `technical_run`: technical library identifier where available.
- `tissue`: CSF or PBMC.
- `diagnosis` / `disease_group` / `group`: source diagnosis or analysis grouping;
  these fields are not interchangeable across cohorts without the documented mapping.
- `source_subseries`: original new-data subseries for the external cohort.
- `age`, `sex`, `dmt_at_sampling`, `treatment_note`: metadata extracted from public
  source records. Missing values remain missing; no private clinical records were added.
- Cell IDs are retained in the AnnData observation index and matching CSV index.
  Join scores and annotations by the exact cell ID, not by row order or barcode alone.

Dataset roles are recorded in `metadata/cohort_roles.csv` and the repository README.

## Expression matrices

`data/processed/GSE*_decontx/GSM*/` contains three gzip-compressed files:

| File | Interpretation |
|---|---|
| `matrix.mtx.gz` | Full-gene DecontX-corrected counts, genes by cells |
| `features.tsv.gz` | Feature identifiers and gene names, in matrix row order |
| `barcodes.tsv.gz` | Cell barcodes, in matrix column order |

DecontX output can be non-integer. These matrices precede the final post-decontamination
removal of zero-count cells and cells with fewer than 200 detected genes. Use the
released cleaned-cell indices and QC tables to identify the final retained cells.

## AnnData checkpoints

| Object | Cells | Matrix interpretation |
|---|---:|---|
| GSE138266 clean singlets | 69,867 | `X` and counts layer restricted to 3,000 HVGs; `.raw` has full-gene log-normalized expression |
| GSE194078 clean singlets | 137,108 | Same organization as above |
| GSE286068 clean singlets | 22,756 | Same organization as above |
| GSE138266 refined APCs | 11,318 | `X` restricted to 2,500 HVGs; `.raw` has full-gene log-normalized expression |
| GSE194078 refined APCs | 21,785 | Same organization as above |
| GSE286068 refined APCs | 2,846 | Same organization as above |
| GSE328265 clean annotated | 49,408 | Full-gene normalized expression and full-gene DecontX counts layer |
| GSE138266 continuum | 7,153 | Full-gene log-normalized expression, embeddings, trajectory and score metadata |
| GSE194078 continuum | 10,432 | Same organization as above |
| GSE286068 continuum | 1,855 | Same organization as above |
| GSE328265 external two-axis | 3,333 | External continuum cells with projected differentiation and inflammatory scores |

AnnData `.raw` is an expression snapshot, not a claim of raw sequencing counts.
HVG-restricted `X` is an analysis representation and must not be substituted for
full-gene counts in pseudobulk analysis. Normalization uses 10,000 counts per cell
followed by log1p where specified in the preprocessing scripts.

The three `*_MS_CSF_continuum_minimal.h5ad` files are compatibility inputs for the
separate CellOracle environment. They are subsets, not additional donors or cohorts.

## QC and annotation

- `decontX_contamination`: estimated ambient-RNA contribution.
- `scrublet_score` and `scDblFinder_score`: caller-specific doublet scores.
- Doublet removal uses the union of positive calls; caller flags and cell retention
  are recorded in `results/qc/`.
- `broad_cell_type`: broad lineage annotation.
- `fine_apc_label`: marker-based APC annotation before continuum confidence stratification.
- `bam_stable_core`: robustness/confidence tier from multiresolution annotation;
  it is not an independently demonstrated developmental stage.
- `bam_resolution_count`: annotation agreement across tested resolutions.
- `transition_state`: analysis state used for donor-balanced trajectory reconstruction.

For discovery cohorts, BAM confidence labels reside in
`results/bam_robustness/*_BAM_cell_stability.csv.gz`; they are joined into continuum
objects by cell ID. Do not assume every label is already present in the refined APC file.

## Scores and statistical outputs

- `consensus_pseudotime`: donor-root and method-consensus rank along the discovery trajectory.
- `projected_differentiation`: fixed signed trajectory program projected into the external cohort.
- `two_axis_activation_residual` / `external_activation_residual`: inflammatory activation
  after adjustment for differentiation and technical covariates.
- Donor-level continuous indices, fractions, program scores, paired contrasts,
  cohort effects, P values, FDR values, and confidence intervals are supplied in
  the corresponding complete result tables. Column definitions and formulas are
  implemented in scripts `52` through `79`.
- `P` / `p_value` and FDR columns are distinct. FDR correction families are recorded
  in the statistical audit tables.

Computational columns containing `lock`, `terminal`, `veto`, or `competence`
refer to the inflammatory signature, trajectory endpoint, program-change
criterion, or program score used in the code, not newly established cell identities.

## Figure source data

The 25 main-figure CSVs and 36 supplementary-figure CSVs contain the plotted values.
Complete tested result tables are also released in `results/`; selected figure
tables must not be interpreted as the complete hypothesis-testing universe.
