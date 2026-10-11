#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BiocParallel)
  library(DropletUtils)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "ms_csf_border_immune"
gse <- if (length(args) >= 2) args[[2]] else "GSE138266"
root <- normalizePath(root, mustWork = TRUE)
raw_dir <- file.path(root, "data", "raw", gse)
qc_dir <- file.path(root, "results", "qc")
log_dir <- file.path(root, "logs")
dir.create(qc_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

min_genes <- 200L
max_mt <- 20
emptydrops_fdr <- 0.001
set.seed(12345)

find_file <- function(accession, suffix) {
  hits <- Sys.glob(file.path(raw_dir, paste0(accession, "_*_", suffix)))
  if (length(hits) != 1) {
    stop("Expected one ", suffix, " file for ", accession, "; found ", length(hits))
  }
  hits[[1]]
}

read_sample <- function(accession) {
  matrix_file <- find_file(accession, "matrix.mtx.gz")
  barcode_file <- find_file(accession, "barcodes.tsv.gz")
  gene_hits <- c(
    Sys.glob(file.path(raw_dir, paste0(accession, "_*_genes.tsv.gz"))),
    Sys.glob(file.path(raw_dir, paste0(accession, "_*_features.tsv.gz")))
  )
  gene_hits <- unique(gene_hits)
  if (length(gene_hits) != 1) stop("Expected one gene/feature file for ", accession)

  counts <- readMM(gzfile(matrix_file))
  counts <- as(counts, "dgCMatrix")
  barcodes <- read.delim(gzfile(barcode_file), header = FALSE, stringsAsFactors = FALSE)[[1]]
  features <- read.delim(gzfile(gene_hits[[1]]), header = FALSE, stringsAsFactors = FALSE)
  if (ncol(counts) != length(barcodes) || nrow(counts) != nrow(features)) {
    stop("Dimension mismatch for ", accession)
  }
  rownames(counts) <- make.unique(as.character(features[[2]]))
  colnames(counts) <- barcodes
  list(counts = counts, barcodes = barcodes, features = features)
}

dedicated_meta <- file.path(root, "metadata", paste0(gse, "_sample_metadata.csv"))
if (file.exists(dedicated_meta)) {
  meta <- read.csv(dedicated_meta, stringsAsFactors = FALSE, check.names = FALSE)
} else {
  meta <- read.csv(file.path(root, "metadata", "all_geo_sample_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
  meta <- meta[meta$gse == gse, , drop = FALSE]
}

cell_records <- list()
sample_records <- list()

for (i in seq_len(nrow(meta))) {
  accession <- meta$sample_accession[[i]]
  message("Cell calling: ", accession)
  dat <- read_sample(accession)
  counts <- dat$counts
  total <- Matrix::colSums(counts)
  genes <- Matrix::colSums(counts > 0)
  mt_idx <- startsWith(toupper(rownames(counts)), "MT-")
  mt_counts <- if (any(mt_idx)) Matrix::colSums(counts[mt_idx, , drop = FALSE]) else rep(0, ncol(counts))
  pct_mt <- ifelse(total > 0, mt_counts / total * 100, 0)
  raw_droplet_matrix <- ncol(counts) >= 100000 && mean(total > 0) < 0.5

  empty_fdr <- rep(NA_real_, ncol(counts))
  if (raw_droplet_matrix) {
    message("  Running EmptyDrops on raw droplet matrix with ", ncol(counts), " barcodes")
    ed <- emptyDrops(
      counts,
      lower = 100,
      niters = 10000,
      test.ambient = FALSE,
      BPPARAM = SerialParam(progressbar = FALSE)
    )
    empty_fdr <- ed$FDR
    cell_called <- !is.na(empty_fdr) & empty_fdr <= emptydrops_fdr
    call_method <- "EmptyDrops_FDR_0.001"
  } else {
    cell_called <- total > 0
    call_method <- "depositor_filtered_matrix"
  }

  base_qc <- genes >= min_genes & pct_mt <= max_mt
  keep <- cell_called & base_qc
  idx <- which(keep)
  cell_records[[accession]] <- data.frame(
    cell_id = paste0(accession, ":", dat$barcodes[idx]),
    barcode = dat$barcodes[idx],
    sample_accession = accession,
    sample_title = meta$sample_title[[i]],
    subject_id = meta$subject_id[[i]],
    tissue = meta$tissue[[i]],
    disease_group = meta$disease_group[[i]],
    total_counts_precleanup = as.numeric(total[idx]),
    n_genes_precleanup = as.integer(genes[idx]),
    pct_mt_precleanup = as.numeric(pct_mt[idx]),
    emptydrops_fdr = empty_fdr[idx],
    cell_call_method = call_method,
    stringsAsFactors = FALSE
  )
  sample_records[[accession]] <- data.frame(
    sample_accession = accession,
    sample_title = meta$sample_title[[i]],
    tissue = meta$tissue[[i]],
    disease_group = meta$disease_group[[i]],
    raw_droplet_matrix = raw_droplet_matrix,
    n_barcodes_input = ncol(counts),
    n_nonzero_barcodes = sum(total > 0),
    n_cell_called = sum(cell_called),
    n_pre_doublet_qc = sum(keep),
    cell_call_method = call_method,
    stringsAsFactors = FALSE
  )
  rm(counts, dat)
  gc(verbose = FALSE)
}

cell_out <- do.call(rbind, cell_records)
sample_out <- do.call(rbind, sample_records)
write.csv(cell_out, gzfile(file.path(qc_dir, paste0(gse, "_cell_calling_metrics.csv.gz"))), row.names = FALSE)
write.csv(sample_out, file.path(qc_dir, paste0(gse, "_cell_calling_summary.csv")), row.names = FALSE)

writeLines(c(
  paste0("n_samples=", nrow(sample_out)),
  paste0("n_pre_doublet_qc=", nrow(cell_out)),
  paste0("raw_droplet_samples=", paste(sample_out$sample_accession[sample_out$raw_droplet_matrix], collapse = ","))
), file.path(log_dir, paste0(gse, "_cell_calling.log")))
