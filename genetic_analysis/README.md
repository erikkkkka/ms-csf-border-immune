# Genetic analysis of monocyte gene expression

Use Python 3.12 and R 4.6.0. Obtain the following inputs.

- [TenK10K version 2](https://doi.org/10.5281/zenodo.18221260): `common_all_cis_pvalues_100kb.zip` and `part4_common_all_cis_pvalues_1Mb.zip`.
- [International Multiple Sclerosis Genetics Consortium](https://imsgc.net/): request the MS discovery summary statistics (`discovery_metav3.0.meta.gz`).

Run from this directory. Substitute local paths for the three input files.

```sh
python -m pip install -r requirements.txt
Rscript scripts/install_r_packages.R

python scripts/prepare_summary_statistics.py \
  --eqtl-100kb /path/to/common_all_cis_pvalues_100kb.zip \
  --eqtl-1mb /path/to/part4_common_all_cis_pvalues_1Mb.zip \
  --imsgc /path/to/discovery_metav3.0.meta.gz
python scripts/download_ld_reference.py
python scripts/mr_wald.py
Rscript scripts/coloc_single_signal.R
python scripts/build_ld.py --window 100kb
python scripts/build_ld.py --window 1Mb
python scripts/mr_ld_pruned_ivw.py
python scripts/select_expression_instruments.py
python scripts/mr_pca_ivw.py
Rscript scripts/coloc_susie.R 100kb
Rscript scripts/coloc_susie.R 1Mb
Rscript scripts/coloc_credible_sets.R
python scripts/sensitivity_analyses.py
python scripts/figure_s10.py
```

The reference download uses the 1000 Genomes Phase 3 European samples. The `reference/` directory contains Ensembl GRCh38-to-GRCh37 mappings and TenK10K lead-variant identifiers. Results and Figure S10 are written to `results/`. Keep IMSGC files and variant-level outputs in `work/` local under the consortium's data access conditions.
