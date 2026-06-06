# =============================================================================
# 00_packages_config.R — packages, global CONFIG, and shared utilities
#
# Source this FIRST in every pipeline script. It loads the packages the
# pipeline actually uses, sets the global seed / fold configuration, and defines
# the small shared helpers (brier_score, normalize_doi, %||%).
#
# Packages are loaded if installed; they are NOT auto-installed here. Run
# `Rscript install.R` once to install everything (see DEPENDENCIES section of
# README.md for the per-stage breakdown).
#
# PIPELINE LAYOUT (this folder = the path that reproduces the submission;
#                  supporting analyses live in ../experiments/. See ARCHITECTURE.md.)
#   00_packages_config.R       packages + CONFIG + utilities      (source first)
#   --- data construction (needs the challenge data; see DATA.md) ---
#   01_data_raw_assembly.R     merge rounds + TEI full-text features
#   02_external_features.R     OpenAlex, SJR, TOP Factor, GDP
#   03_llm_features.R          parse structured LLM ratings -> features
#   04_base_dataset.R          master feature matrix (train_base / test_base)
#   --- model infrastructure ---
#   05_imputation.R            leak-free median / MICE recipes
#   06_models.R                base learners + MODEL_GRID + PREDICTOR_SETS
#   07_calibration.R           temperature / Platt / isotonic
#   --- evaluation + submission ---
#   08_ci_brier_cluster_bootstrap.R  study-clustered bootstrap CIs
#   09_final_comparison.R         leak-free candidate comparison (-> ElNet+ m1, CIs)
#   10_submission.R            assemble the m1/m2/m3 deliverable
# =============================================================================

# Toggle: when TRUE, features whose underlying source was missing are set to NA
# (then handled by the imputation recipe); when FALSE they are collapsed to 0.
# The submitted models were produced with FALSE — flipping this to TRUE changes
# the elastic-net fit and its predictions, so keep it FALSE to reproduce the
# deliverable. (Count-style features where 0 is the natural "none observed"
# value are the reason FALSE is the reproducing setting here.)
NA_counts_Zero <- FALSE

# Packages actually used by the pipeline scripts in this package, grouped by the
# stage that needs them. (install.R installs the same set.) Stage-0 is needed
# everywhere, including the submission-assembly smoke test.
packages <- c(
  # stage 0 — core data handling (every script)
  "here", "dplyr", "readr", "tibble", "tidyr", "purrr", "stringr", "jsonlite",
  # stage 1 — modelling, calibration, validation
  "glmnet", "mgcv", "ranger", "xgboost", "lightgbm", "dbarts", "rstanarm",
  "mice", "miceadds", "pROC",
  # stage 2 — data assembly / feature construction (01–04)
  "openalexR", "FReD", "wbstats", "countrycode",
  "quanteda", "quanteda.textstats", "sentimentr", "BayesFactor", "scrutiny",
  # stage 3 — plots / diagnostics
  "ggplot2", "ggrepel", "scales", "corrplot", "pROC"
)

invisible(lapply(packages, function(p) {
  if (requireNamespace(p, quietly = TRUE)) {
    suppressPackageStartupMessages(
      tryCatch(library(p, character.only = TRUE),
               error = function(e)
                 warning(sprintf("Package '%s' could not be loaded: %s", p, e$message)))
    )
  } else {
    warning(sprintf("Package '%s' is not installed — run `Rscript install.R`.", p))
  }
}))

options(scipen = 999)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG <- list(
  seed      = 4267,
  n_folds   = 5,
  group_var = "doi_o"   # stratify CV folds by original-study DOI (no leakage)
)

set.seed(CONFIG$seed)

# ── Shared utilities ──────────────────────────────────────────────────────────
`%||%` <- function(a, b) if (!is.null(a)) a else b

brier_score <- function(pred, actual) {
  pred <- pmax(pmin(as.numeric(pred), 1), 0)
  mean((pred - as.numeric(actual))^2, na.rm = TRUE)
}

normalize_doi <- function(x) {
  x |> stringr::str_remove("^https?://doi\\.org/") |>
    stringr::str_remove("^doi:") |>
    stringr::str_to_lower() |>
    stringr::str_trim()
}
