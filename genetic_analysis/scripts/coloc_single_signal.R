#!/usr/bin/env Rscript
# Single-signal colocalization.
suppressPackageStartupMessages(library(data.table))
args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=', args)])
root <- normalizePath(file.path(dirname(script), '..'))
source_dir <- file.path(root, "work", "coloc_core")
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)
source_url <- "https://raw.githubusercontent.com/chr1swallace/coloc/27776b1bb14e19d1542798ac7477fa378fc2fc0d/R/"
for (name in c("private.R", "check.R", "claudia.R")) {
    path <- file.path(source_dir, name)
    if (!file.exists(path)) {
        temporary <- tempfile(tmpdir = source_dir)
        download.file(paste0(source_url, name), temporary, mode = "wb", quiet = TRUE)
        stopifnot(file.rename(temporary, path))
    }
    source(path)
}
contexts <- fread(file.path(root, 'results', 'lead_variants.csv'))
exposures <- fread(file.path(root, 'work', 'exposure_regions_mapped.tsv'))
harmonized <- fread(file.path(root, 'work', 'harmonized_regions.tsv'))
summaries <- list()
warning_log <- character()
for (i in seq_len(nrow(contexts))) {
    gene_name <- contexts$gene[i]
    cell_name <- contexts$cell_type[i]
    e <- exposures[gene == gene_name & cell_type == cell_name]
    h <- harmonized[gene == gene_name & cell_type == cell_name]
    for (prior in c(1e-5, 1e-6)) {
        row <- data.table(gene=gene_name, cell_type=cell_name,
                          planned_role=contexts$planned_role[i], p1=1e-4, p2=1e-4, p12=prior,
                          analysis=if (prior == 1e-5) 'primary_default_prior' else 'one_lower_shared_prior_check',
                          status='', n_eqtl=nrow(e), nsnps=NA_integer_, estimated_expression_sdY=NA_real_,
                          PP.H0=NA_real_, PP.H1=NA_real_, PP.H2=NA_real_, PP.H3=NA_real_, PP.H4=NA_real_,
                          trait1='MS_susceptibility', trait2='model_scale_expression',
                          sdY_method='estimated_from_eQTL_variance_frequency_and_sample_size',
                          outcome_SE='reconstructed_from_OR_and_P', model='at_most_one_causal_variant_per_trait')
        if (nrow(e) == 0 || nrow(h) == 0) {
            row$status <- 'not_estimated_missing_exposure_region'
        } else {
            stopifnot(length(unique(e$N)) == 1, !anyDuplicated(h$variant_GRCh38))
            sdy <- withCallingHandlers(sdY.est(e$se^2, e$MAF, unique(e$N)), warning=function(w) {
                warning_log <<- c(warning_log, paste(gene_name, cell_name, conditionMessage(w)))
                invokeRestart('muffleWarning')
            })
            d1 <- list(beta=h$gwas_beta_aligned, varbeta=h$gwas_se_approx^2,
                       snp=h$variant_GRCh38, position=h$pos_GRCh38, type='cc')
            d2 <- list(beta=e$beta, varbeta=e$se^2, snp=e$variant_GRCh38,
                       position=e$pos_GRCh38, type='quant', sdY=sdy)
            res <- withCallingHandlers(coloc.abf(d1, d2, p1=1e-4, p2=1e-4, p12=prior), warning=function(w) {
                warning_log <<- c(warning_log, paste(gene_name, cell_name, conditionMessage(w)))
                invokeRestart('muffleWarning')
            })
            pp <- res$summary[paste0('PP.H', 0:4, '.abf')]
            stopifnot(abs(sum(pp)-1) < 1e-10, res$summary[['nsnps']] == nrow(h))
            row$status <- 'estimated_approximate_input_single_signal_model'
            row$nsnps <- as.integer(res$summary[['nsnps']])
            row$estimated_expression_sdY <- sdy
            for (k in 0:4) row[[paste0('PP.H', k)]] <- unname(pp[k+1])
        }
        summaries[[length(summaries)+1]] <- row
    }
}
summary <- rbindlist(summaries)
fwrite(summary, file.path(root, 'results', 'COLOC_ALL_CONTEXTS_AND_PRIORS.csv'))
fwrite(summary[analysis == 'primary_default_prior'], file.path(root, 'results', 'coloc_single_signal.csv'))
