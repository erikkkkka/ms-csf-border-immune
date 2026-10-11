#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(slingshot)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "ms_csf_border_immune"
root <- normalizePath(root, mustWork = TRUE)
input_dir <- file.path(root, "results", "trajectory", "slingshot_input")
output_dir <- file.path(root, "results", "trajectory", "slingshot_output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

anchors <- read.csv(
  file.path(root, "results", "trajectory", "multiroot_DPT_qc.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

run_basis <- function(frame, basis, start_cluster, end_cluster) {
  pc_columns <- grep(paste0("^", basis, "_PC[0-9]+$"), names(frame), value = TRUE)
  reduced <- as.matrix(frame[, pc_columns, drop = FALSE])
  clusters <- factor(frame$cluster)
  fit <- slingshot(
    reduced,
    clusterLabels = clusters,
    start.clus = as.character(start_cluster),
    end.clus = as.character(end_cluster),
    reweight = TRUE,
    reassign = TRUE,
    shrink = TRUE,
    stretch = 0
  )
  pseudotime <- slingPseudotime(fit, na = TRUE)
  weights <- slingCurveWeights(fit)
  if (is.null(dim(pseudotime))) {
    pseudotime <- matrix(pseudotime, ncol = 1)
    weights <- matrix(weights, ncol = 1)
  }
  output <- data.frame(row.names = seq_len(nrow(frame)))
  paths <- slingLineages(fit)
  path_output <- data.frame(
    basis = basis,
    lineage = paste0(basis, "_lineage", seq_along(paths)),
    cluster_path = vapply(paths, paste, collapse = ">", character(1)),
    start_cluster = vapply(paths, function(path) as.character(path[[1]]), character(1)),
    end_cluster = vapply(paths, function(path) as.character(path[[length(path)]]), character(1)),
    stringsAsFactors = FALSE
  )
  for (index in seq_len(ncol(pseudotime))) {
    output[[paste0(basis, "_lineage", index, "_pseudotime")]] <- pseudotime[, index]
    output[[paste0(basis, "_lineage", index, "_weight")]] <- weights[, index]
  }
  list(scores = output, paths = path_output)
}

files <- list.files(input_dir, pattern = "_slingshot_input\\.csv\\.gz$", full.names = TRUE)
if (length(files) == 0) stop("No Slingshot input files")

for (path in files) {
  dataset <- sub("_slingshot_input\\.csv\\.gz$", "", basename(path))
  message("Slingshot: ", dataset)
  frame <- read.csv(gzfile(path), stringsAsFactors = FALSE, check.names = FALSE)
  anchor <- anchors[anchors$dataset == dataset, , drop = FALSE]
  if (nrow(anchor) != 1) stop("Missing or duplicate anchor row for ", dataset)
  primary <- run_basis(frame, "primary", anchor$start_cluster[[1]], anchor$end_cluster[[1]])
  marker <- run_basis(frame, "marker_excluded", anchor$start_cluster[[1]], anchor$end_cluster[[1]])
  output <- cbind(
    frame[, c("cell_id", "transition_state", "trajectory_role", "subject_id"), drop = FALSE],
    primary$scores,
    marker$scores
  )
  write.csv(
    output,
    gzfile(file.path(output_dir, paste0(dataset, "_slingshot_results.csv.gz"))),
    row.names = FALSE
  )
  write.csv(
    rbind(primary$paths, marker$paths),
    file.path(output_dir, paste0(dataset, "_slingshot_lineages.csv")),
    row.names = FALSE
  )
}
