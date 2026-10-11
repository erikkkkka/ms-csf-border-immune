#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BiocParallel)
  library(celda)
  library(Matrix)
  library(R.utils)
  library(scDblFinder)
  library(SingleCellExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "ms_csf_border_immune"
target_accession <- if (length(args) >= 2) args[[2]] else NULL
root <- normalizePath(root, mustWork = TRUE)
raw_dir <- file.path(root, "data", "raw", "GSE138266")
processed_dir <- file.path(root, "data", "processed", "GSE138266_decontx")
qc_dir <- file.path(root, "results", "qc")
log_dir <- file.path(root, "logs", "decontx")
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(qc_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(12345)

find_file <- function(accession, suffix) {
  hits <- Sys.glob(file.path(raw_dir, paste0(accession, "_*_", suffix)))
  if (length(hits) != 1) stop("Expected one ", suffix, " for ", accession)
  hits[[1]]
}

read_sample <- function(accession) {
  matrix_file <- find_file(accession, "matrix.mtx.gz")
  barcode_file <- find_file(accession, "barcodes.tsv.gz")
  gene_hits <- unique(c(
    Sys.glob(file.path(raw_dir, paste0(accession, "_*_genes.tsv.gz"))),
    Sys.glob(file.path(raw_dir, paste0(accession, "_*_features.tsv.gz")))
  ))
  if (length(gene_hits) != 1) stop("Expected one feature file for ", accession)
  counts <- as(readMM(gzfile(matrix_file)), "dgCMatrix")
  barcodes <- read.delim(gzfile(barcode_file), header = FALSE, stringsAsFactors = FALSE)[[1]]
  features <- read.delim(gzfile(gene_hits[[1]]), header = FALSE, stringsAsFactors = FALSE)
  rownames(counts) <- make.unique(as.character(features[[2]]))
  colnames(counts) <- barcodes
  list(counts = counts, barcodes = barcodes, features = features)
}

write_gzip_lines <- function(x, path) {
  con <- gzfile(path, open = "wt")
  on.exit(close(con), add = TRUE)
  writeLines(x, con)
}

write_gzip_table <- function(x, path) {
  con <- gzfile(path, open = "wt")
  on.exit(close(con), add = TRUE)
  write.table(x, con, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}

metrics <- read.csv(gzfile(file.path(qc_dir, "GSE138266_scrublet_metrics.csv.gz")),
                    stringsAsFactors = FALSE, check.names = FALSE)
metrics$scrublet_predicted_doublet <- tolower(as.character(metrics$scrublet_predicted_doublet)) == "true"
if (!is.null(target_accession)) {
  if (!target_accession %in% metrics$sample_accession) stop("Unknown accession: ", target_accession)
  metrics <- metrics[metrics$sample_accession == target_accession, , drop = FALSE]
}
cell_outputs <- list()
sample_outputs <- list()

for (accession in sort(unique(metrics$sample_accession))) {
  frame <- metrics[metrics$sample_accession == accession, , drop = FALSE]
  message("scDblFinder + DecontX: ", accession, " (", nrow(frame), " cells)")
  dat <- read_sample(accession)
  idx <- match(frame$barcode, dat$barcodes)
  if (anyNA(idx)) stop("Missing called barcodes for ", accession)
  counts <- dat$counts[, idx, drop = FALSE]
  colnames(counts) <- frame$cell_id

  sce <- SingleCellExperiment(assays = list(counts = counts))
  sce <- scDblFinder(
    sce,
    dbr.per1k = 0.008,
    multiSampleMode = "split",
    BPPARAM = SerialParam(progressbar = FALSE),
    verbose = TRUE
  )
  frame$scDblFinder_score <- as.numeric(colData(sce)$scDblFinder.score)
  frame$scDblFinder_class <- as.character(colData(sce)$scDblFinder.class)
  frame$doublet_union <- as.logical(frame$scrublet_predicted_doublet) |
    frame$scDblFinder_class == "doublet"

  singlet_idx <- which(!frame$doublet_union)
  sce_singlet <- sce[, singlet_idx, drop = FALSE]
  background <- NULL
  raw_like <- ncol(dat$counts) >= 100000 && mean(Matrix::colSums(dat$counts) > 0) < 0.5
  if (raw_like) {
    totals <- Matrix::colSums(dat$counts)
    bg_idx <- which(totals > 0 & totals <= 100)
    if (length(bg_idx) > 20000) bg_idx <- sample(bg_idx, 20000)
    background_counts <- dat$counts[, bg_idx, drop = FALSE]
    background <- SingleCellExperiment(assays = list(counts = background_counts))
    message("  Using ", ncol(background), " empty droplets as empirical background")
  }

  decontx_status <- "ok"
  tryCatch({
    sce_singlet <- decontX(
      sce_singlet,
      background = background,
      maxIter = 500,
      estimateDelta = TRUE,
      varGenes = min(5000L, nrow(sce_singlet)),
      seed = 12345,
      logfile = file.path(log_dir, paste0(accession, ".log")),
      verbose = TRUE
    )
    decont_counts <- round(decontXcounts(sce_singlet))
    contamination <- as.numeric(colData(sce_singlet)$decontX_contamination)
  }, error = function(e) {
    decontx_status <<- paste0("failed:", conditionMessage(e))
    decont_counts <<- SummarizedExperiment::assay(sce_singlet, "counts")
    contamination <<- rep(NA_real_, ncol(sce_singlet))
  })

  decont_counts <- as(decont_counts, "dgCMatrix")
  out_dir <- file.path(processed_dir, accession)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  tmp_mtx <- file.path(out_dir, "matrix.mtx")
  writeMM(decont_counts, tmp_mtx)
  gzip(tmp_mtx, destname = paste0(tmp_mtx, ".gz"), overwrite = TRUE, remove = TRUE)
  write_gzip_lines(frame$barcode[singlet_idx], file.path(out_dir, "barcodes.tsv.gz"))
  write_gzip_table(dat$features, file.path(out_dir, "features.tsv.gz"))

  frame$decontX_contamination <- NA_real_
  frame$decontX_contamination[singlet_idx] <- contamination
  frame$cleanup_keep <- !frame$doublet_union
  frame$decontX_status <- decontx_status
  cell_outputs[[accession]] <- frame
  sample_outputs[[accession]] <- data.frame(
    sample_accession = accession,
    n_pre_doublet_qc = nrow(frame),
    n_scrublet_doublets = sum(frame$scrublet_predicted_doublet),
    n_scDblFinder_doublets = sum(frame$scDblFinder_class == "doublet"),
    n_doublet_union = sum(frame$doublet_union),
    n_singlets = sum(frame$cleanup_keep),
    pct_doublet_union = mean(frame$doublet_union) * 100,
    median_decontX_contamination = median(contamination, na.rm = TRUE),
    decontX_status = decontx_status,
    stringsAsFactors = FALSE
  )
  rm(dat, counts, sce, sce_singlet, decont_counts, background)
  gc(verbose = FALSE)
}

cell_out <- do.call(rbind, cell_outputs)
sample_out <- do.call(rbind, sample_outputs)
if (!is.null(target_accession)) {
  old_cell_path <- file.path(qc_dir, "GSE138266_cleanup_cell_metrics.csv.gz")
  old_sample_path <- file.path(qc_dir, "GSE138266_cleanup_sample_summary.csv")
  if (file.exists(old_cell_path)) {
    old_cell <- read.csv(gzfile(old_cell_path), stringsAsFactors = FALSE, check.names = FALSE)
    old_cell <- old_cell[old_cell$sample_accession != target_accession, , drop = FALSE]
    cell_out <- rbind(old_cell, cell_out)
  }
  if (file.exists(old_sample_path)) {
    old_sample <- read.csv(old_sample_path, stringsAsFactors = FALSE, check.names = FALSE)
    old_sample <- old_sample[old_sample$sample_accession != target_accession, , drop = FALSE]
    sample_out <- rbind(old_sample, sample_out)
  }
}
cell_out <- cell_out[order(cell_out$sample_accession, cell_out$cell_id), , drop = FALSE]
sample_out <- sample_out[order(sample_out$sample_accession), , drop = FALSE]
write.csv(cell_out, gzfile(file.path(qc_dir, "GSE138266_cleanup_cell_metrics.csv.gz")), row.names = FALSE)
write.csv(sample_out, file.path(qc_dir, "GSE138266_cleanup_sample_summary.csv"), row.names = FALSE)
