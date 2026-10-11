#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "ms_csf_border_immune"
gse <- if (length(args) >= 2) args[[2]] else "GSE138266"
min_cells <- if (length(args) >= 3) as.integer(args[[3]]) else 10L
case_subset <- if (length(args) >= 4) args[[4]] else "allMS"
root <- normalizePath(root, mustWork = TRUE)
pb_dir <- file.path(root, "results", "pseudobulk")
metadata <- read.csv(file.path(pb_dir, paste0(gse, "_CSF_pseudobulk_sample_metadata.csv")), stringsAsFactors = FALSE)
files <- Sys.glob(file.path(pb_dir, paste0(gse, "_CSF_*_counts.csv.gz")))
summary_rows <- list()
pathway_rows <- list()
control_label <- if (gse == "GSE138266") "IIH_control" else "non_inflammatory_control"
if (case_subset == "PMS") {
  metadata <- metadata[metadata$disease_group == control_label | metadata$diagnosis %in% c("PPMS", "SPMS"), , drop = FALSE]
  metadata$disease_group[metadata$diagnosis %in% c("PPMS", "SPMS")] <- "MS"
} else if (case_subset %in% c("RRMS", "PPMS", "SPMS")) {
  metadata <- metadata[metadata$disease_group == control_label | metadata$diagnosis == case_subset, , drop = FALSE]
  metadata$disease_group[metadata$diagnosis == case_subset] <- "MS"
} else if (case_subset != "allMS") {
  stop("Unsupported case subset: ", case_subset)
}
contrast_tag <- if (case_subset == "allMS") "MS" else case_subset
threshold_tag <- if (min_cells == 10L) "" else paste0("_min", min_cells)
gene_sets <- list(
  IFN_antiviral = c("IFI6", "IFI27", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2", "IFITM3", "ISG15", "MX1", "OAS1", "OAS2", "GBP1", "IRF1", "STAT1"),
  MHC_I_processing = c("B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSMB10", "NLRC5"),
  MHC_II_presentation = c("HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "CD74", "CIITA", "CTSS"),
  DC_like_identity = c("CD1C", "FCER1A", "CLEC10A", "CD1E", "CST3", "FLT3", "IRF4"),
  BAM_homeostatic = c("C1QA", "C1QB", "C1QC", "APOE", "TREM2", "STAB1", "MRC1", "LYVE1", "FOLR2", "DAB2", "MSR1", "CD163", "HAMP"),
  Lipid_scavenger = c("APOE", "LPL", "ABCA1", "ABCG1", "CH25H", "SOAT1", "NR1H3", "PLIN2", "MSR1", "TREM2"),
  Migratory_DC = c("LAMP3", "CCR7", "FSCN1", "CCL19", "CCL22", "CD274", "MARCKSL1"),
  B_cell_support = c("CXCL13", "TNFSF13B", "TNFSF13", "CCL19", "CCL22", "IL6", "IL15", "CD40")
)

for (path in files) {
  label <- sub("_counts.csv.gz$", "", sub(paste0("^", gse, "_CSF_"), "", basename(path)))
  sample_meta <- metadata[metadata$analysis_label == label & metadata$n_cells >= min_cells, , drop = FALSE]
  group_sizes <- table(factor(sample_meta$disease_group, levels = c(control_label, "MS")))
  if (any(group_sizes < 4)) {
    summary_rows[[label]] <- data.frame(
      analysis_label = label,
      n_ms = unname(group_sizes["MS"]),
      n_control = unname(group_sizes[control_label]),
      n_genes_tested = 0L,
      n_fdr_0_05 = NA_integer_,
      top_gene = NA_character_,
      top_logFC = NA_real_,
      top_fdr = NA_real_,
      status = paste0("skipped:min_cells_", min_cells),
      stringsAsFactors = FALSE
    )
    next
  }
  sample_meta$disease_group <- relevel(factor(sample_meta$disease_group), ref = control_label)

  table <- read.csv(gzfile(path), check.names = FALSE, stringsAsFactors = FALSE)
  sample_meta <- sample_meta[match(intersect(names(table), sample_meta$sample_accession), sample_meta$sample_accession), , drop = FALSE]
  counts <- as.matrix(table[, sample_meta$sample_accession, drop = FALSE])
  storage.mode(counts) <- "integer"
  rownames(counts) <- make.unique(table$gene_id)
  y <- DGEList(counts = counts, genes = table[, c("gene_id", "gene_symbol")], group = sample_meta$disease_group)
  keep <- filterByExpr(y, group = sample_meta$disease_group, min.count = 5)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- normLibSizes(y)
  design <- model.matrix(~ disease_group, data = sample_meta)
  y <- estimateDisp(y, design, robust = TRUE)
  fit <- glmQLFit(y, design, robust = TRUE)
  test <- glmQLFTest(fit, coef = "disease_groupMS")
  result <- topTags(test, n = Inf, sort.by = "PValue")$table
  result$gene_id <- y$genes[rownames(result), "gene_id"]
  result$gene_symbol <- y$genes[rownames(result), "gene_symbol"]
  result <- result[, c("gene_id", "gene_symbol", "logFC", "logCPM", "F", "PValue", "FDR")]
  write.csv(result, gzfile(file.path(pb_dir, paste0(gse, "_CSF_", label, "_edgeR_", contrast_tag, threshold_tag, "_vs_control.csv.gz"))), row.names = FALSE)

  symbols <- as.character(y$genes$gene_symbol)
  indices <- lapply(gene_sets, function(genes) which(symbols %in% genes))
  indices <- indices[lengths(indices) >= 5]
  if (length(indices) > 0) {
    camera_result <- limma::camera(y, index = indices, design = design, contrast = ncol(design), sort = FALSE)
    camera_result$pathway <- rownames(camera_result)
    camera_result$analysis_label <- label
    pathway_rows[[label]] <- camera_result[, c("analysis_label", "pathway", "NGenes", "Direction", "PValue", "FDR")]
  }

  summary_rows[[label]] <- data.frame(
    analysis_label = label,
    n_ms = unname(group_sizes["MS"]),
    n_control = unname(group_sizes[control_label]),
    n_genes_tested = nrow(result),
    n_fdr_0_05 = sum(result$FDR < 0.05),
    top_gene = result$gene_symbol[[1]],
    top_logFC = result$logFC[[1]],
    top_fdr = result$FDR[[1]],
    status = "ok",
    stringsAsFactors = FALSE
  )
}

summary <- do.call(rbind, summary_rows)
summary <- summary[order(summary$status, summary$top_fdr), , drop = FALSE]
write.csv(summary, file.path(pb_dir, paste0(gse, "_CSF_edgeR_", contrast_tag, threshold_tag, "_summary.csv")), row.names = FALSE)
if (length(pathway_rows) > 0) {
  pathways <- do.call(rbind, pathway_rows)
  pathways <- pathways[order(pathways$FDR, pathways$PValue), , drop = FALSE]
  write.csv(pathways, file.path(pb_dir, paste0(gse, "_CSF_camera_", contrast_tag, threshold_tag, "_pathways.csv")), row.names = FALSE)
}
