#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(scTenifoldKnk)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "ms_csf_border_immune"
target <- if (length(args) >= 2) args[[2]] else "TAP1"
n_networks <- if (length(args) >= 3) as.integer(args[[3]]) else 5L
n_cells_per_network <- if (length(args) >= 4) as.integer(args[[4]]) else 250L
gse <- if (length(args) >= 5) args[[5]] else "GSE138266"
n_cores <- if (length(args) >= 6) as.integer(args[[6]]) else 2L
root <- normalizePath(root, mustWork = TRUE)
out_dir <- file.path(root, "results", "virtual_knockout")
input_path <- file.path(out_dir, paste0(gse, "_MS_CSF_BAM_scTenifold_input.csv.gz"))
counts <- read.csv(gzfile(input_path), row.names = 1, check.names = FALSE)
counts <- as.matrix(counts)
storage.mode(counts) <- "numeric"
if (!target %in% rownames(counts)) stop("Target absent from matrix: ", target)

set.seed(12345)
result <- scTenifoldKnk(
  countMatrix = counts,
  qc = FALSE,
  gKO = target,
  nc_nNet = n_networks,
  nc_nCells = n_cells_per_network,
  nc_nComp = 3,
  nc_q = 0.9,
  td_K = 3,
  ma_nDim = 2,
  nCores = n_cores
)
run_tag <- paste0(nrow(counts), "genes_", n_networks, "nets_", n_cores, "cores")
saveRDS(result, file.path(out_dir, paste0(gse, "_MS_CSF_BAM_", target, "_scTenifoldKnk_", run_tag, ".rds")))
write.csv(
  result$diffRegulation,
  file.path(out_dir, paste0(gse, "_MS_CSF_BAM_", target, "_scTenifold_diffRegulation_", run_tag, ".csv")),
  row.names = FALSE
)
