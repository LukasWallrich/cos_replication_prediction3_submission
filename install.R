# install.R — install the R packages used by this package.
#
#   Rscript install.R            # install everything
#   Rscript install.R core       # only the submission-assembly smoke test
#   Rscript install.R model      # core + modelling/validation
#   Rscript install.R data       # core + data-assembly/feature build
#
# Tested with R 4.5.3. Package versions are listed in DEPENDENCIES.md.

repos <- "https://cloud.r-project.org"

groups <- list(
  # needed everywhere, incl. `Rscript pipeline/10_submission.R`
  core  = c("here", "dplyr", "readr", "tibble", "tidyr", "purrr", "stringr",
            "jsonlite"),
  # base learners, calibration, imputation, validation, plots
  model = c("glmnet", "mgcv", "ranger", "xgboost", "lightgbm", "dbarts",
            "rstanarm", "mice", "miceadds", "pROC",
            "ggplot2", "ggrepel", "scales", "corrplot"),
  # external + text features for the data-assembly stage (01–04)
  data  = c("openalexR", "FReD", "wbstats", "countrycode",
            "quanteda", "quanteda.textstats", "sentimentr",
            "BayesFactor", "scrutiny")
)

arg <- commandArgs(trailingOnly = TRUE)
sel <- if (length(arg) == 0 || arg[1] == "all") names(groups) else {
  switch(arg[1],
    core  = "core",
    model = c("core", "model"),
    data  = c("core", "data"),
    stop("Unknown group '", arg[1], "'. Use: all | core | model | data."))
}

pkgs <- unique(unlist(groups[sel], use.names = FALSE))
need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]

cat(sprintf("Groups: %s\nPackages required: %d | already installed: %d | to install: %d\n",
            paste(sel, collapse = "+"), length(pkgs), length(pkgs) - length(need), length(need)))

if (length(need)) {
  cat("Installing:", paste(need, collapse = ", "), "\n")
  install.packages(need, repos = repos)
} else {
  cat("Nothing to install.\n")
}

# FReD (optional helper) lives on CRAN; if it is unavailable the pipeline degrades
# gracefully (02_external_features.R wraps it in tryCatch).
still <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(still))
  cat("\nNOTE: could not install:", paste(still, collapse = ", "),
      "\n  Install these manually if you need the stage that uses them.\n")
cat("Done.\n")
