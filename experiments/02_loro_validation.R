# =============================================================================
# 02_loro_validation.R  —  Leave-One-Round-Out (LORO) Validation
# =============================================================================
#
# PURPOSE
# -------
# Standard k-fold CV shuffles all rows at random and assumes observations are
# i.i.d. (independently and identically distributed). But in the COS challenge,
# training data comes from multiple rounds that differ in time, journal scope,
# and curation: patterns exploited within Round 2 may not exist in Round 3.
#
# LORO mimics the true generalization task:
#   Fold A: train on (FORRT + Round 1), predict Round 2
#   Fold B: train on (FORRT + Round 2), predict Round 1
#
# This gives two held-out Brier scores; their average is the LORO Brier.
#
# INTERPRETATION GUIDE
# --------------------
#   LORO ≈ CV        → model generalises across rounds; CV is trustworthy.
#   LORO ≫ CV        → model exploits within-round distributional artefacts
#                       (e.g., round-specific label rates, journal quirks).
#                       These models will likely underperform on Round 3.
#   LORO ≪ CV (rare) → CV was pessimistic; model benefits from cross-round
#                       signal.
#
# WHY ONLY 2 FOLDS?
# -----------------
# We have exactly two historical challenge rounds with ground truth (R1, R2).
# A third fold would need R3 labels which are the submission target. Two folds
# is correct, not a compromise.
#
# OUTPUTS
# -------
#   output/loro_results.csv   — per-(model × predictor_set × fold) Brier rows
#   output/loro_vs_cv.csv     — LORO avg vs. 5-fold CV Brier (if CV results exist)
#
# =============================================================================

suppressPackageStartupMessages({
  # PREDICTOR_SETS, MODEL_GRID, fit_*/pred_* and brier_score all come from
  # 06_models.R — the single source of truth — so this script never drifts from
  # the deployed model definitions.
  source(here::here("pipeline/06_models.R"))
  source(here::here("pipeline/05_imputation.R"))
})

# ── 1. Load data ──────────────────────────────────────────────────────────────

train_base_path <- here::here("data/train_base.rds")

if (!file.exists(train_base_path)) {
  stop("data/train_base.rds not found. Run: Rscript pipeline/04_base_dataset.R")
}

train_base <- readRDS(train_base_path)

# Check for R1/R2 rows; if missing, try to regenerate train_base
if (!all(c("round1", "round2") %in% unique(train_base$dataset))) {
  message(
    "train_base.rds does not contain round1/round2 rows (found: ",
    paste(unique(train_base$dataset), collapse = ", "), ").\n",
    "Regenerating via pipeline/04_base_dataset.R ..."
  )
  regen_result <- tryCatch(
    system(paste("Rscript", here::here("pipeline/04_base_dataset.R")),
           wait = TRUE),
    error = function(e) {
      stop("Could not run pipeline/04_base_dataset.R: ", e$message,
           "\nPlease run it manually first.")
    }
  )
  if (regen_result != 0) {
    stop(
      "pipeline/04_base_dataset.R exited with non-zero status.\n",
      "Please run it manually: Rscript pipeline/04_base_dataset.R"
    )
  }
  train_base <- readRDS(train_base_path)
}

if (!all(c("round1", "round2") %in% unique(train_base$dataset))) {
  stop(
    "After regeneration, train_base.rds still lacks round1/round2 rows.\n",
    "Unique dataset values found: ", paste(unique(train_base$dataset), collapse = ", "), "\n",
    "Check pipeline/04_base_dataset.R for errors."
  )
}

message("Loaded train_base.rds: ", nrow(train_base), " rows; datasets: ",
        paste(sort(unique(train_base$dataset)), collapse = ", "))

# ── 2. Filter to rows with ground-truth outcomes ───────────────────────────────

data_gt <- train_base |>
  filter(!is.na(statistical_success))

message("Rows with ground-truth statistical_success: ", nrow(data_gt))
message("  training: ", sum(data_gt$dataset == "training"),
        "  round1: ", sum(data_gt$dataset == "round1"),
        "  round2: ", sum(data_gt$dataset == "round2"))

# ── 3. Helper: Brier score ─────────────────────────────────────────────────────
# Clips predictions to [0, 1] before computing mean squared error.
brier_score <- function(pred, actual) {
  pred <- pmax(pmin(as.numeric(pred), 1), 0)
  mean((pred - as.numeric(actual))^2, na.rm = TRUE)
}

# ── 4. Core LORO fold function ────────────────────────────────────────────────

#' Run one LORO fold for one (model, predictor_set) combination.
#'
#' @param data              Full ground-truth data frame.
#' @param train_datasets    Character vector of dataset values used for training.
#' @param holdout_dataset   Single dataset value held out for evaluation.
#' @param model_spec        One entry from MODEL_GRID (list with $name, $fit, $pred).
#' @param predictor_set     Character vector of column names to use as features.
#' @param predictor_set_name Short name (key from PREDICTOR_SETS) stored in output.
#' @return A one-row tibble with columns:
#'   model_name, predictor_set, holdout_round, n_train, n_holdout, brier.
run_loro_fold <- function(data, train_datasets, holdout_dataset,
                          model_spec, predictor_set, predictor_set_name) {
  holdout_dois <- data |>
    filter(dataset == holdout_dataset) |>
    pull(doi_o) |>
    unique()
  
  train_df   <- data |> filter(dataset %in% train_datasets,
                               !doi_o %in% holdout_dois)
  holdout_df <- data |> filter(dataset == holdout_dataset)

  predictors <- intersect(predictor_set, names(data))

  # imputation strategy
  strat <- if (!is.null(model_spec$imputation_strategy)) model_spec$imputation_strategy else "none"

  # model-specific imputation branch
  if (strat == "pmm") {
    # --- PMM path: average predictions over m imputed datasets ---
    imputed_data <- apply_imputation(train_df, holdout_df, predictor_set_name, m = 5)
    if (is.null(imputed_data)) return(NULL)

    preds_matrix <- sapply(seq_len(5), function(i) {
      train_imp <- imputed_data$train[[i]]
      test_imp  <- imputed_data$test[[i]]
      cols      <- setdiff(names(train_imp), "statistical_success")
      
      model <- model_spec$fit(train_imp, "statistical_success", cols, rep(1, nrow(train_imp)))
      model_spec$pred(model, test_imp, cols)
    })
    final_preds <- rowMeans(preds_matrix)
    
  } else {
    # --- "none" path: zero-fill missing predictors, then fit once ---
    zero_fill <- function(df) {
      df[, predictors] <- lapply(df[, predictors, drop = FALSE], function(x) { x[is.na(x)] <- 0; x })
      df
    }
    train_df   <- zero_fill(train_df)
    holdout_df <- zero_fill(holdout_df)
    
    model <- model_spec$fit(train_df, "statistical_success", predictors, rep(1, nrow(train_df)))
    final_preds <- model_spec$pred(model, holdout_df, predictors)
  }
  
  brier <- brier_score(final_preds, holdout_df$statistical_success)
  
  tibble(
    model_name    = model_spec$name,
    predictor_set = predictor_set_name,
    holdout_round = holdout_dataset,
    imputation    = strat, 
    n_train       = nrow(train_df),         
    n_holdout     = nrow(holdout_df),   
    brier         = brier
  )
}

# ── 5. LORO loop ──────────────────────────────────────────────────────────────
# Two folds × |MODEL_GRID| models × |PREDICTOR_SETS| predictor sets.

# LORO fold definitions
FOLDS <- list(
  list(train = c("training", "round1"), holdout = "round2"),
  list(train = c("training", "round2"), holdout = "round1")
)

# Generate all combinations
combos <- expand.grid(
  model_idx = seq_along(MODEL_GRID),
  pset_name = names(PREDICTOR_SETS),
  fold_idx  = seq_along(FOLDS),
  stringsAsFactors = FALSE
)

message("\nRunning LORO: ", nrow(combos), " combinations (",
        length(MODEL_GRID), " models × ",
        length(PREDICTOR_SETS), " predictor sets × ",
        length(FOLDS), " folds) ...")

results_list <- vector("list", nrow(combos))

for (i in seq_len(nrow(combos))) {
  m_spec  <- MODEL_GRID[[combos$model_idx[i]]]
  pset_nm <- combos$pset_name[i]
  pset    <- PREDICTOR_SETS[[pset_nm]]
  fold    <- FOLDS[[combos$fold_idx[i]]]

  label   <- sprintf("%-9s × %-14s → holdout=%s",
                     m_spec$name, pset_nm, fold$holdout)
  cat(sprintf("  [%2d/%2d] %s  ", i, nrow(combos), label))

  results_list[[i]] <- tryCatch(
    {
      res <- run_loro_fold(
        data              = data_gt,
        train_datasets    = fold$train,
        holdout_dataset   = fold$holdout,
        model_spec        = m_spec,
        predictor_set     = pset,
        predictor_set_name = pset_nm
      )
      cat(sprintf("Brier=%.4f\n", res$brier))
      res
    },
    error = function(e) {
      warning(sprintf("FAILED: %s | %s → %s | %s",
                      m_spec$name, pset_nm, fold$holdout, e$message))
      cat("FAILED\n")
      tibble(
        model_name    = m_spec$name,
        predictor_set = pset_nm,
        holdout_round = fold$holdout,
        imputation    = m_spec$imputation_strategy %||% "none",
        n_train       = NA_integer_,
        n_holdout     = NA_integer_,
        brier         = NA_real_
      )
    }
  )
}

loro_results <- bind_rows(results_list)

# ── 6. Aggregate: average Brier across the two folds ──────────────────────────

loro_avg <- loro_results |>
  group_by(model_name, predictor_set, imputation) |>
  summarise(
    n_folds        = sum(!is.na(brier)),
    brier_fold_A   = brier[holdout_round == "round2"][1],
    brier_fold_B   = brier[holdout_round == "round1"][1],
    brier_loro_avg = mean(brier, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(brier_loro_avg)

# ── 7. Write outputs ──────────────────────────────────────────────────────────

output_dir <- here::here("output")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

write_csv(loro_results, file.path(output_dir, "loro_results.csv"))
message("\nWrote: output/loro_results.csv (", nrow(loro_results), " rows)")

# ── 8. Print top 10 by LORO Brier ───────────────────────────────────────────

cat("\n========================================\n")
cat("Top 10 model × predictor-set by LORO Brier\n")
cat("========================================\n")
print(loro_avg |> head(10), n = 10)

# ── 9. Compare LORO vs. 5-fold CV (if results_summary.csv exists) ────────────

cv_path <- file.path(output_dir, "results_summary.csv")

if (file.exists(cv_path)) {
  cv_results <- read_csv(cv_path, show_col_types = FALSE)

  # Join on model algorithm name + predictor set name.
  # results_summary.csv has a "model" string like "RF_extended" (abbreviation)
  # and a separate "pset" column with the canonical name ("extended", "full",
  # "health_focused", "round2"). We join on (model_name × pset) to avoid
  # fragile string parsing of the "model" abbreviations.
  #
  # For the LORO side: model_name ("RF") and predictor_set ("extended") are
  # already separate columns.
  cv_agg <- cv_results |>
    # Take best-performing config per (algorithm × predictor set) if duplicates
    group_by(model_name = sub("_.*", "", model), pset) |>
    summarise(brier_cv_mean = min(brier_mean, na.rm = TRUE),
              brier_cv_sd   = brier_sd[which.min(brier_mean)],
              .groups = "drop") |>
    rename(predictor_set = pset)

  loro_vs_cv <- loro_avg |>
    left_join(cv_agg, by = c("model_name", "predictor_set")) |>
    mutate(cv_loro_gap = brier_cv_mean - brier_loro_avg) |>
    arrange(brier_loro_avg)

  write_csv(loro_vs_cv, file.path(output_dir, "loro_vs_cv.csv"))
  message("Wrote: output/loro_vs_cv.csv")

  cat("\n========================================\n")
  cat("Top 5 by |CV - LORO| gap\n")
  cat("  (Large gap = model exploits within-round artefacts)\n")
  cat("========================================\n")
  their_gaps <- loro_vs_cv |>
    filter(!is.na(cv_loro_gap)) |>
    arrange(desc(cv_loro_gap)) |>
    mutate(model_key = paste0(model_name, "_", predictor_set)) |>
    select(model_key, brier_cv_mean, brier_loro_avg, cv_loro_gap) |>
    head(5)
  print(their_gaps)
} else {
  message("No output/results_summary.csv found; skipping CV vs LORO comparison.")
}

message("\nDone.")
