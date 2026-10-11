#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "ms_csf_border_immune"
min_cells <- if (length(args) >= 2) as.integer(args[[2]]) else 5L
root <- normalizePath(root, mustWork = TRUE)
pb_dir <- file.path(root, "results", "bam_robustness", "pseudobulk")
out_dir <- file.path(root, "results", "bam_robustness")

gene_sets <- list(
  IFN_antiviral = c("IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2", "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1"),
  MHC_I_processing = c("B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5"),
  BAM_homeostatic = c("C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1", "LYVE1", "FOLR2", "DAB2", "MSR1", "CD163", "HAMP"),
  Lipid_scavenger = c("APOE", "LPL", "ABCA1", "ABCG1", "CH25H", "SOAT1", "NR1H3", "PLIN2", "MSR1", "TREM2")
)

summary_rows <- list()
target_rows <- list()
pathway_rows <- list()

for (gse in c("GSE138266", "GSE286068")) {
  control_label <- if (gse == "GSE138266") "IIH_control" else "non_inflammatory_control"
  metadata <- read.csv(
    file.path(pb_dir, paste0(gse, "_CSF_BAM_stable_core_sample_metadata.csv")),
    stringsAsFactors = FALSE
  )
  metadata <- metadata[
    metadata$n_cells >= min_cells & metadata$disease_group %in% c(control_label, "MS"),
    , drop = FALSE
  ]
  group_sizes <- table(factor(metadata$disease_group, levels = c(control_label, "MS")))
  if (any(group_sizes < 4)) {
    stop(gse, " has insufficient stable-core donors after min-cell filtering")
  }
  metadata$disease_group <- relevel(factor(metadata$disease_group), ref = control_label)
  table <- read.csv(
    gzfile(file.path(pb_dir, paste0(gse, "_CSF_BAM_stable_core_counts.csv.gz"))),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  metadata <- metadata[
    match(intersect(names(table), metadata$sample_accession), metadata$sample_accession),
    , drop = FALSE
  ]
  counts <- as.matrix(table[, metadata$sample_accession, drop = FALSE])
  storage.mode(counts) <- "integer"
  rownames(counts) <- make.unique(table$gene_id)
  y <- DGEList(
    counts = counts,
    genes = table[, c("gene_id", "gene_symbol")],
    group = metadata$disease_group
  )
  keep <- filterByExpr(y, group = metadata$disease_group, min.count = 5)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y)
  design <- model.matrix(~ disease_group, data = metadata)
  y <- estimateDisp(y, design, robust = TRUE)
  fit <- glmQLFit(y, design, robust = TRUE)
  test <- glmQLFTest(fit, coef = "disease_groupMS")
  result <- topTags(test, n = Inf, sort.by = "PValue")$table
  result$gene_id <- y$genes[rownames(result), "gene_id"]
  result$gene_symbol <- y$genes[rownames(result), "gene_symbol"]
  result <- result[, c("gene_id", "gene_symbol", "logFC", "logCPM", "F", "PValue", "FDR")]
  write.csv(
    result,
    gzfile(file.path(out_dir, paste0(gse, "_CSF_BAM_stable_core_edgeR_MS_vs_control.csv.gz"))),
    row.names = FALSE
  )

  lta4h <- result[result$gene_symbol == "LTA4H", , drop = FALSE]
  if (nrow(lta4h) > 0) {
    lta4h$gse <- gse
    lta4h$n_ms <- unname(group_sizes["MS"])
    lta4h$n_control <- unname(group_sizes[control_label])
    target_rows[[gse]] <- lta4h[, c("gse", "n_ms", "n_control", names(result))]
  }

  symbols <- as.character(y$genes$gene_symbol)
  indices <- lapply(gene_sets, function(genes) which(symbols %in% genes))
  indices <- indices[lengths(indices) >= 5]
  camera_result <- limma::camera(
    y,
    index = indices,
    design = design,
    contrast = ncol(design),
    sort = FALSE
  )
  camera_result$pathway <- rownames(camera_result)
  camera_result$gse <- gse
  pathway_rows[[gse]] <- camera_result[, c("gse", "pathway", "NGenes", "Direction", "PValue", "FDR")]
  summary_rows[[gse]] <- data.frame(
    gse = gse,
    n_ms = unname(group_sizes["MS"]),
    n_control = unname(group_sizes[control_label]),
    n_genes_tested = nrow(result),
    n_fdr_0_05 = sum(result$FDR < 0.05),
    top_gene = result$gene_symbol[[1]],
    top_logFC = result$logFC[[1]],
    top_fdr = result$FDR[[1]],
    stringsAsFactors = FALSE
  )
}

write.csv(
  do.call(rbind, summary_rows),
  file.path(out_dir, "BAM_stable_core_edgeR_summary.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, target_rows),
  file.path(out_dir, "BAM_stable_core_LTA4H_edgeR.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, pathway_rows),
  file.path(out_dir, "BAM_stable_core_camera_pathways.csv"),
  row.names = FALSE
)
