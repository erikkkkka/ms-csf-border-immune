#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BiocParallel)
  library(DropletUtils)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "ms_csf_border_immune"
root <- normalizePath(root, mustWork = TRUE)
raw_dir <- file.path(root, "data", "raw", "GSE286068")
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
  if (length(hits) != 1) stop("Expected one ", suffix, " file for ", accession, "; found ", length(hits))
  hits[[1]]
}

read_sample <- function(accession) {
  matrix_file <- find_file(accession, "matrix.mtx.gz")
  barcode_file <- find_file(accession, "barcodes.tsv.gz")
  feature_file <- find_file(accession, "features.tsv.gz")
  counts <- as(readMM(gzfile(matrix_file)), "dgCMatrix")
  barcodes <- read.delim(gzfile(barcode_file), header = FALSE, stringsAsFactors = FALSE)[[1]]
  features <- read.delim(gzfile(feature_file), header = FALSE, stringsAsFactors = FALSE)
  if (ncol(counts) != length(barcodes) || nrow(counts) != nrow(features)) stop("Dimension mismatch for ", accession)
  gex <- if (ncol(features) >= 3) features[[3]] == "Gene Expression" else rep(TRUE, nrow(features))
  counts <- counts[gex, , drop = FALSE]
  features <- features[gex, , drop = FALSE]
  rownames(counts) <- make.unique(as.character(features[[2]]))
  colnames(counts) <- barcodes
  list(counts = counts, barcodes = barcodes, features = features)
}

meta <- read.csv(file.path(root, "metadata", "all_geo_sample_metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
meta <- meta[meta$gse == "GSE286068" & meta$disease_group != "unknown" & !startsWith(meta$sample_title, "multiplexed"), , drop = FALSE]
meta <- meta[order(meta$sample_accession), , drop = FALSE]

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
    message("  Running EmptyDrops on ", ncol(counts), " raw barcodes")
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
  keep <- cell_called & genes >= min_genes & pct_mt <= max_mt
  idx <- which(keep)
  cell_records[[accession]] <- data.frame(
    cell_id = paste0(accession, ":", dat$barcodes[idx]),
    barcode = dat$barcodes[idx],
    sample_accession = accession,
    sample_title = meta$sample_title[[i]],
    subject_id = meta$subject_id[[i]],
    tissue = meta$tissue[[i]],
    diagnosis = meta$diagnosis[[i]],
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
    diagnosis = meta$diagnosis[[i]],
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
  if (exists("ed")) rm(ed)
  gc(verbose = FALSE)
}

cell_out <- do.call(rbind, cell_records)
sample_out <- do.call(rbind, sample_records)
write.csv(cell_out, gzfile(file.path(qc_dir, "GSE286068_cell_calling_metrics.csv.gz")), row.names = FALSE)
write.csv(sample_out, file.path(qc_dir, "GSE286068_cell_calling_summary.csv"), row.names = FALSE)
writeLines(c(
  paste0("n_samples=", nrow(sample_out)),
  paste0("n_pre_doublet_qc=", nrow(cell_out)),
  paste0("raw_droplet_samples=", paste(sample_out$sample_accession[sample_out$raw_droplet_matrix], collapse = ","))
), file.path(log_dir, "GSE286068_cell_calling.log"))
