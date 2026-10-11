#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: 41_edger_gse304475.R counts.csv.gz metadata.csv output_dir")
}

counts_path <- args[[1]]
metadata_path <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts_table <- read.csv(counts_path, check.names = FALSE)
metadata <- read.csv(metadata_path, stringsAsFactors = FALSE)
rownames(counts_table) <- counts_table$gene_id
counts_table$gene_id <- NULL

if (!identical(colnames(counts_table), metadata$sample_id)) {
  stop("Count columns do not match sample metadata order")
}
if (any(counts_table < 0) || any(abs(as.matrix(counts_table) - round(as.matrix(counts_table))) > 0)) {
  stop("Counts must be non-negative integers")
}

group <- factor(metadata$group, levels = c("Vehicle", "SC57461A"))
design <- model.matrix(~ group)
y <- DGEList(counts = counts_table, group = group, genes = data.frame(gene_id = rownames(counts_table)))
keep <- filterByExpr(y, design = design)
y <- y[keep, , keep.lib.sizes = FALSE]
y <- normLibSizes(y, method = "TMM")
y <- estimateDisp(y, design, robust = TRUE)
fit <- glmQLFit(y, design, robust = TRUE)
test <- glmQLFTest(fit, coef = "groupSC57461A")
result <- topTags(test, n = Inf, sort.by = "none")$table
result$gene_id <- rownames(result)
result <- result[, c("gene_id", "logFC", "logCPM", "F", "PValue", "FDR")]
result <- result[order(result$PValue), ]

write.csv(result, gzfile(file.path(output_dir, "GSE304475_SC57461A_edgeR.csv.gz")), row.names = FALSE)
write.csv(
  data.frame(
    sample_id = colnames(y),
    group = group,
    library_size = y$samples$lib.size,
    norm_factor = y$samples$norm.factors,
    effective_library_size = y$samples$lib.size * y$samples$norm.factors
  ),
  file.path(output_dir, "GSE304475_edgeR_normalization.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    n_input_genes = nrow(counts_table),
    n_tested_genes = nrow(y),
    n_fdr_0_05 = sum(result$FDR < 0.05),
    n_fdr_0_05_up = sum(result$FDR < 0.05 & result$logFC > 0),
    n_fdr_0_05_down = sum(result$FDR < 0.05 & result$logFC < 0),
    contrast = "SC57461A_vs_Vehicle"
  ),
  file.path(output_dir, "GSE304475_edgeR_summary.csv"),
  row.names = FALSE
)

mds <- plotMDS(y, plot = FALSE)
mds_frame <- data.frame(
  sample_id = metadata$sample_id,
  group = group,
  MDS1 = mds$x,
  MDS2 = mds$y
)
write.csv(
  mds_frame,
  file.path(output_dir, "GSE304475_edgeR_MDS_coordinates.csv"),
  row.names = FALSE
)
pdf(file.path(output_dir, "GSE304475_edgeR_MDS.pdf"), width = 6, height = 5)
mds_colors <- ifelse(group == "SC57461A", "#C44E32", "#007C83")
x_padding <- diff(range(mds$x)) * 0.12
y_padding <- diff(range(mds$y)) * 0.13
plot(
  mds$x,
  mds$y,
  type = "n",
  xlim = range(mds$x) + c(-x_padding, x_padding),
  ylim = range(mds$y) + c(-y_padding, y_padding),
  xlab = "Leading logFC dimension 1",
  ylab = "Leading logFC dimension 2"
)
points(mds$x, mds$y, pch = 16, col = mds_colors)
text(
  mds$x,
  mds$y,
  labels = metadata$sample_id,
  col = mds_colors,
  pos = ifelse(mds$x > median(mds$x), 2, 4),
  offset = 0.45,
  cex = 0.82
)
legend(
  "top",
  legend = levels(group),
  col = c("#007C83", "#C44E32"),
  pch = 16,
  horiz = TRUE,
  bty = "n"
)
dev.off()

vehicle_mds2 <- mds_frame$MDS2[mds_frame$group == "Vehicle"]
drug_mds2 <- mds_frame$MDS2[mds_frame$group == "SC57461A"]
pooled_mds2_sd <- sqrt(
  ((length(vehicle_mds2) - 1) * var(vehicle_mds2) +
     (length(drug_mds2) - 1) * var(drug_mds2)) /
    (length(vehicle_mds2) + length(drug_mds2) - 2)
)
write.csv(
  data.frame(
    metric = c(
      "vehicle_MDS2_min", "vehicle_MDS2_max", "SC57461A_MDS2_min",
      "SC57461A_MDS2_max", "between_group_MDS2_gap",
      "MDS2_standardized_mean_difference", "minimum_TMM_factor",
      "maximum_TMM_factor"
    ),
    value = c(
      min(vehicle_mds2), max(vehicle_mds2), min(drug_mds2), max(drug_mds2),
      min(vehicle_mds2) - max(drug_mds2),
      (mean(vehicle_mds2) - mean(drug_mds2)) / pooled_mds2_sd,
      min(y$samples$norm.factors), max(y$samples$norm.factors)
    )
  ),
  file.path(output_dir, "GSE304475_edgeR_sample_QC_summary.csv"),
  row.names = FALSE
)

fit_leave_one_out <- function(omitted_sample) {
  retained <- colnames(counts_table) != omitted_sample
  loo_counts <- counts_table[, retained, drop = FALSE]
  loo_group <- droplevels(group[retained])
  loo_design <- model.matrix(~ loo_group)
  loo_y <- DGEList(
    counts = loo_counts,
    group = loo_group,
    genes = data.frame(gene_id = rownames(loo_counts))
  )
  loo_keep <- filterByExpr(loo_y, design = loo_design)
  loo_y <- loo_y[loo_keep, , keep.lib.sizes = FALSE]
  loo_y <- normLibSizes(loo_y, method = "TMM")
  loo_y <- estimateDisp(loo_y, loo_design, robust = TRUE)
  loo_fit <- glmQLFit(loo_y, loo_design, robust = TRUE)
  loo_test <- glmQLFTest(loo_fit, coef = "loo_groupSC57461A")
  loo_result <- topTags(loo_test, n = Inf, sort.by = "none")$table
  loo_result$gene_id <- rownames(loo_result)
  loo_result$omitted_sample <- omitted_sample
  loo_result[, c("omitted_sample", "gene_id", "logFC", "logCPM", "F", "PValue", "FDR")]
}

loo_results <- do.call(rbind, lapply(colnames(counts_table), fit_leave_one_out))
write.csv(
  loo_results,
  gzfile(file.path(output_dir, "GSE304475_SC57461A_edgeR_leave_one_out.csv.gz")),
  row.names = FALSE
)

full_effect <- result[, c("gene_id", "logFC", "FDR")]
names(full_effect)[2:3] <- c("full_logFC", "full_FDR")
full_discoveries <- full_effect$gene_id[full_effect$full_FDR < 0.05]
loo_summary <- do.call(rbind, lapply(split(loo_results, loo_results$omitted_sample), function(frame) {
  merged <- merge(frame, full_effect, by = "gene_id")
  discovery_frame <- merged[merged$gene_id %in% full_discoveries, ]
  data.frame(
    omitted_sample = unique(frame$omitted_sample),
    omitted_group = as.character(metadata$group[metadata$sample_id == unique(frame$omitted_sample)]),
    n_tested_genes = nrow(frame),
    n_fdr_0_05 = sum(frame$FDR < 0.05),
    all_gene_spearman_vs_full = cor(merged$logFC, merged$full_logFC, method = "spearman"),
    full_discovery_direction_concordance = mean(
      sign(discovery_frame$logFC) == sign(discovery_frame$full_logFC)
    ),
    full_discovery_median_abs_logFC = median(abs(discovery_frame$logFC)),
    full_discovery_nominal_p_0_05_fraction = mean(discovery_frame$PValue < 0.05)
  )
}))
write.csv(
  loo_summary,
  file.path(output_dir, "GSE304475_edgeR_leave_one_out_summary.csv"),
  row.names = FALSE
)

print(read.csv(file.path(output_dir, "GSE304475_edgeR_summary.csv")))
print(head(result, 20))
print(read.csv(file.path(output_dir, "GSE304475_edgeR_sample_QC_summary.csv")))
print(loo_summary)
