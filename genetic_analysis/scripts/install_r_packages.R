cran <- "https://cloud.r-project.org"
install.packages(c("data.table", "jsonlite", "matrixStats", "mixsqp", "reshape", "crayon", "ggplot2", "viridis"),
                 repos = cran)
install.packages(paste0(cran, "/src/contrib/Archive/susieR/susieR_0.14.2.tar.gz"), repos = NULL, type = "source")
install.packages(paste0(cran, "/src/contrib/Archive/coloc/coloc_5.2.3.tar.gz"), repos = NULL, type = "source")
stopifnot(packageVersion("susieR") == "0.14.2", packageVersion("coloc") == "5.2.3")
