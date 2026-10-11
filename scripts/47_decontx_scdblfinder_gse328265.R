#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(BiocParallel)
  library(celda)
  library(Matrix)
  library(R.utils)
  library(scDblFinder)
  library(Seurat)
  library(SingleCellExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "."
root <- normalizePath(root, mustWork = TRUE)
metadata <- read.csv(file.path(root, "metadata", "GSE328265_sample_metadata.csv"),
                     stringsAsFactors = FALSE, check.names = FALSE)
metrics <- read.csv(gzfile(file.path(root, "results", "qc", "GSE328265_scrublet_metrics.csv.gz")),
                    stringsAsFactors = FALSE, check.names = FALSE)
metrics$scrublet_predicted_doublet <-
  tolower(as.character(metrics$scrublet_predicted_doublet)) == "true"

processed_dir <- file.path(root, "data", "processed", "GSE328265_decontx")
qc_dir <- file.path(root, "results", "qc")
log_dir <- file.path(root, "logs", "decontx_GSE328265")
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(12345)

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

cell_outputs <- list()
sample_outputs <- list()
for (accession in metadata$sample_accession) {
  row <- metadata[metadata$sample_accession == accession, , drop = FALSE]
  frame <- metrics[metrics$sample_accession == accession, , drop = FALSE]
  message("scDblFinder + DecontX: ", accession, " (", nrow(frame), " QC cells)")
  counts <- Read10X_h5(row$h5_path[[1]], use.names = TRUE, unique.features = TRUE)
  idx <- match(frame$barcode, colnames(counts))
  if (anyNA(idx)) stop("Missing QC barcodes for ", accession)
  counts <- counts[, idx, drop = FALSE]
  colnames(counts) <- frame$cell_id

  if (ncol(counts) >= 50) {
    sce <- SingleCellExperiment(assays = list(counts = counts))
    sce <- scDblFinder(
      sce,
      dbr.per1k = 0.008,
      multiSampleMode = "split",
      BPPARAM = SerialParam(progressbar = FALSE),
      verbose = FALSE
    )
    frame$scDblFinder_score <- as.numeric(colData(sce)$scDblFinder.score)
    frame$scDblFinder_class <- as.character(colData(sce)$scDblFinder.class)
  } else {
    sce <- SingleCellExperiment(assays = list(counts = counts))
    frame$scDblFinder_score <- NA_real_
    frame$scDblFinder_class <- "not_run_lt_50_cells"
  }
  frame$doublet_union <- frame$scrublet_predicted_doublet |
    frame$scDblFinder_class == "doublet"
  singlet_idx <- which(!frame$doublet_union)
  sce_singlet <- sce[, singlet_idx, drop = FALSE]

  decontx_status <- "ok_filtered_matrix_no_empty_droplet_background"
  if (ncol(sce_singlet) >= 50) {
    tryCatch({
      sce_singlet <- decontX(
        sce_singlet,
        maxIter = 500,
        estimateDelta = TRUE,
        varGenes = min(5000L, nrow(sce_singlet)),
        seed = 12345,
        logfile = file.path(log_dir, paste0(accession, ".log")),
        verbose = FALSE
      )
      decont_counts <- round(decontXcounts(sce_singlet))
      contamination <- as.numeric(colData(sce_singlet)$decontX_contamination)
    }, error = function(e) {
      decontx_status <<- paste0("failed_fallback_raw:", conditionMessage(e))
      decont_counts <<- SummarizedExperiment::assay(sce_singlet, "counts")
      contamination <<- rep(NA_real_, ncol(sce_singlet))
    })
  } else {
    decontx_status <- "not_run_lt_50_singlets"
    decont_counts <- SummarizedExperiment::assay(sce_singlet, "counts")
    contamination <- rep(NA_real_, ncol(sce_singlet))
  }

  decont_counts <- as(decont_counts, "dgCMatrix")
  out_dir <- file.path(processed_dir, accession)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  tmp_mtx <- file.path(out_dir, "matrix.mtx")
  writeMM(decont_counts, tmp_mtx)
  gzip(tmp_mtx, destname = paste0(tmp_mtx, ".gz"), overwrite = TRUE, remove = TRUE)
  write_gzip_lines(frame$barcode[singlet_idx], file.path(out_dir, "barcodes.tsv.gz"))
  feature_table <- data.frame(rownames(counts), rownames(counts), "Gene Expression")
  write_gzip_table(feature_table, file.path(out_dir, "features.tsv.gz"))

  frame$decontX_contamination <- NA_real_
  frame$decontX_contamination[singlet_idx] <- contamination
  frame$cleanup_keep <- !frame$doublet_union
  frame$decontX_status <- decontx_status
  cell_outputs[[accession]] <- frame
  sample_outputs[[accession]] <- data.frame(
    sample_accession = accession,
    subject_id = row$subject_id[[1]],
    diagnosis = row$diagnosis[[1]],
    n_qc_pass = nrow(frame),
    n_scrublet_doublets = sum(frame$scrublet_predicted_doublet),
    n_scDblFinder_doublets = sum(frame$scDblFinder_class == "doublet"),
    n_doublet_union = sum(frame$doublet_union),
    n_singlets = sum(frame$cleanup_keep),
    pct_doublet_union = mean(frame$doublet_union) * 100,
    median_decontX_contamination = median(contamination, na.rm = TRUE),
    decontX_status = decontx_status,
    stringsAsFactors = FALSE
  )
  rm(counts, sce, sce_singlet, decont_counts)
  gc(verbose = FALSE)
}

cell_out <- do.call(rbind, cell_outputs)
sample_out <- do.call(rbind, sample_outputs)
cell_out <- cell_out[order(cell_out$sample_accession, cell_out$cell_id), , drop = FALSE]
sample_out <- sample_out[order(sample_out$sample_accession), , drop = FALSE]
write.csv(cell_out, gzfile(file.path(qc_dir, "GSE328265_cleanup_cell_metrics.csv.gz")), row.names = FALSE)
write.csv(sample_out, file.path(qc_dir, "GSE328265_cleanup_sample_summary.csv"), row.names = FALSE)
print(sample_out)
