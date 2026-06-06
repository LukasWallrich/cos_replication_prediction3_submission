# =============================================================================
# 07_health_validation.R
#
# PURPOSE
#   Domain-stratified, group-aware k-fold CV over a (model x predictor_set)
#   grid. Produces leak-free out-of-fold (OOF) predictions plus health-transfer
#   diagnostics: nested-CV temperature calibration, post-CV ensembles, and a
#   deploy-faithful health holdout for the top candidates.
#
# KEY DESIGN POINTS
# * IPW is not used (evaluated and found not to improve performance).
# * screen_EQ / screen_Mono (from 19) are added as custom models in the CV loop.
# * Post-CV ensembles (Section 11):
#     ens_18_equal       equal-weight RF+GAM+ElNet+XGB on enriched (18-style)
#     ens_nnls           per-fold NNLS stack over all non-LLM feature models
#     ens_greedy_equal   greedy |cor|<threshold uncorrelated single models
# * Deploy-faithful health holdout (Section 12): top N candidates only,
#   targeted refit — no cross-product of all models x all psets.
# * Imputation diagnostics: fraction missing, range plausibility,
#   between-imputation variance, optional convergence trace check.
#
# INPUTS
#   data/train_base.rds, data/test_base.rds, judge_llm_ensemble/ensemble_scores.csv,
#   output/prior_features_for_harness.csv (optional, for the Prior_Shrinkage model)
# OUTPUTS
#   output/20_feature_stability.csv, output/20_t_calibration.csv,
#   output/20_oof_summary.csv, output/20_deploy_faithful_results.csv,
#   output/20_predictor_audit.csv, output/20_imputation_diag/*.pdf,
#   data/cache/oof_grid.rds
# =============================================================================

suppressPackageStartupMessages({
  source(here::here("pipeline/00_packages_config.R"))
  source(here::here("pipeline/06_models.R"))
  source(here::here("pipeline/07_calibration.R"))
  source(here::here("pipeline/05_imputation.R"))
  library(dplyr); library(tidyr); library(readr); library(tibble); library(purrr)
})

# =============================================================================
# 1. RUN CONFIG
# =============================================================================

RUN_CFG <- list(
  # ---- predictor sets ----
  pset_spec            = c("round2", "transferable_lean", "transferable_lean_indicators","transferable_lean_plus","stable_top"),

  # ---- models ----
  # NULL = all from MODEL_GRID + MonoXGB + screen_EQ + screen_Mono + LLMs.
  model_subset         = c("RF", "GAM",#"XGB", "LGBM",  "MonoXGB","BART",
                           "Logistic", "ElNet_1se","ElNet_min", "ElNet_relax", "Ridge_min",
                           "LLM_raw", "LLM_bincorr"),
  # BayesLogistic excluded from the default run (runtime: ~2-4h for 3 psets x 5
  # folds). To include: add "BayesLogistic" here, set its imputation_strategy to
  # "pmm", and remove the internal zero-fill in fit_bayes_logistic() in 06_models.R.
  include_mono_xgb     = TRUE,
  include_screen_eq    = TRUE,   # equal-weight screened ensemble (from step 19)
  include_screen_mono  = TRUE,   # monotone screened model (from step 19)
  include_llm_raw      = TRUE,
  include_llm_bincorr  = TRUE,
  # LLM substantive-plausibility prior + power-for-observed shrinkage (health-only
  # candidate; per-fold logistic on prior_3fam + p_base). See PRIOR_SHRINKAGE_REPORT.md.
  include_prior_shrinkage = TRUE,
  prior_features_path     = here::here("output/prior_features_for_harness.csv"),

  # ---- CV setup ----
  n_folds              = 3,
  stratify_vars        = c("dataset", "is_health_strat"),
  group_var            = "doi_o",
  outcome_col          = "statistical_success",

  # ---- imputation ----
  n_imputations        = 1,
  mice_maxit           = 5,

  # ---- post-CV ensembles ----
  greedy_cor_threshold = 0.7,    # |cor| threshold for greedy uncorr. ensemble
  greedy_top_n_start   = 20L,    # rank at most this many candidates before greedy

  # ---- deploy-faithful health holdout ----
  deploy_faithful_top_n          = 10L,
  deploy_faithful_n_health_folds = 5L,
  # For PMM models inside deploy-faithful use m=1 (speed); set higher if needed.
  deploy_faithful_n_imp          = 1,

  # ---- LLM ----
  llm_csv_path          = here::here("judge_llm_ensemble/ensemble_scores.csv"),
  bin_correction_factor = 0.6,

  # ---- diagnostics ----
  run_imputation_diagnostics = TRUE,
  run_convergence_diag       = TRUE,   # separate mice run for convergence traces
  diag_n_features            = 10,

  # ---- output paths ----
  output_dir  = here::here("output"),
  cache_dir   = here::here("data/cache"),

  # ---- checkpoint / resume ----
  # Set enable_checkpoint = FALSE to disable (e.g. for short test runs).
  # Delete the checkpoint file manually to force a full re-run.
  enable_checkpoint = TRUE,
  checkpoint_path   = here::here("data/cache/20_checkpoint.rds"),

  # ---- misc ----
  seed    = if (exists("CONFIG") && !is.null(CONFIG$seed)) CONFIG$seed else 42L,
  verbose = TRUE
)

dir.create(RUN_CFG$output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(RUN_CFG$cache_dir,  showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# 2. CONSTANTS
# =============================================================================

DIR_MONO <- c(
  n_o = 1L, n_o_log = 1L, n_o_sqrt = 1L, pval_z = 1L,
  evidence_strength_score = 1L, power_proxy_h = 1L,
  snr_proxy_h = 1L, t_proxy_from_r_h = 1L,
  r_o_abs = 1L, r_harmonized_low_confidence = -1L,
  llm_sample_adequacy = 1L,
  llm_sample_size_ord = 1L, llm_sample_size_ord_mean = 1L,
  llm_power_guess = 1L, llm_power_guess_mean = 1L,
  llm_open_data = 1L, llm_open_code = 1L, llm_preregistration = 1L,
  n_f_tests_tei = -1L, n_t_tests_tei = -1L, tests_per_1000_words_tei = -1L,
  p_hack_index_tei = -1L, p_heaping_flag_tei = -1L,
  llm_perceived_surprisingness = -1L, llm_perceived_surprisingness_mean = -1L,
  pval_value_o = -1L, p_expected = -1L
)

MODEL_INPUT_POLICY <- list(
  Logistic      = "numeric_or_factor",
  GAM           = "numeric_or_factor",
  RF            = "numeric_or_factor",
  ElNet         = "numeric_only",
  XGB           = "numeric_only",
  BART          = "numeric_only",
  LGBM          = "numeric_only",
  MonoXGB       = "numeric_only",
  BayesLogistic = "numeric_only"
)

MODEL_NA_HANDLING <- list(
  RF      = "native",
  XGB     = "native",
  BART    = "zero_fill",
  LGBM    = "native",
  MonoXGB = "native"
)

NBIN_BREAKS <- c(-Inf, 100, 300, 1000, 3000, 10000, Inf)
NBIN_LABELS <- c("<=100", "100-300", "300-1k", "1k-3k", "3k-10k", ">10k")

# Features excluded from screen_features() candidate pool (leakage/ID stoplist).
SCREEN_STOP_COLS <- c(
  "entry_id", "effect_id", "cos_phase1",
  "statistical_success", "reported_success",
  "n_r", "es_value_r", "pval_value_r", "r_r", "year_r",
  "duration_o_r", "duration_o_r_sqrtabs", "is_score_replication",
  "prereg_r_bin", "author_overlap_bin", "author_overlap_missing",
  "same_author_country", "same_author_country_missing",
  "judge_mean_p", "judge_median_p", "judge_sd_p",
  "judge_model_disagreement", "judge_persona_disagreement",
  "judge_llm_miss_ind"
)

# =============================================================================
# 3. HELPERS
# =============================================================================

# ---- fold construction ------------------------------------------------------
make_stratified_group_folds <- function(df, group_var, strat_vars,
                                        n_folds, seed) {
  groups <- as.character(df[[group_var]])
  na_idx <- is.na(groups) | nchar(groups) == 0L
  if (any(na_idx)) groups[na_idx] <- paste0(".na_", seq_len(sum(na_idx)))

  df_aux <- tibble::tibble(.grp = groups)
  for (sv in strat_vars) {
    df_aux[[sv]] <- as.character(df[[sv]])
    df_aux[[sv]][is.na(df_aux[[sv]])] <- "_NA_"
  }

  group_strata <- df_aux |>
    dplyr::group_by(.grp) |>
    dplyr::summarise(dplyr::across(dplyr::all_of(strat_vars), ~ {
      tab <- sort(table(.x), decreasing = TRUE); names(tab)[1]
    }), .groups = "drop") |>
    dplyr::mutate(.stratum = do.call(paste,
      c(dplyr::across(dplyr::all_of(strat_vars)), sep = "__")))

  set.seed(seed)
  group_strata <- group_strata |>
    dplyr::group_by(.stratum) |>
    dplyr::mutate(.fold = sample(rep_len(seq_len(n_folds), dplyr::n()))) |>
    dplyr::ungroup()

  fold_map <- stats::setNames(group_strata$.fold, group_strata$.grp)
  unname(fold_map[groups])
}

# ---- audit log --------------------------------------------------------------
make_audit_log <- function() {
  records <- list()
  list(
    add = function(model_name, dropped) {
      if (length(dropped) == 0L) return(invisible())
      records[[length(records) + 1L]] <<- tibble::tibble(
        model_name = model_name, dropped_predictor = dropped)
    },
    as_tibble = function() {
      if (length(records) == 0L)
        tibble::tibble(model_name = character(0), dropped_predictor = character(0))
      else dplyr::distinct(dplyr::bind_rows(records))
    }
  )
}

# ---- predictor filtering ----------------------------------------------------
filter_predictors_for_model <- function(train_df, predictors, model_name,
                                        audit_log = NULL) {
  policy <- MODEL_INPUT_POLICY[[model_name]] %||% "numeric_only"
  if (policy == "numeric_or_factor") return(predictors)
  is_num <- vapply(predictors, function(p) {
    if (!p %in% names(train_df)) return(FALSE)
    v <- train_df[[p]]
    is.numeric(v) || is.logical(v) || is.integer(v)
  }, logical(1))
  if (!is.null(audit_log) && any(!is_num))
    audit_log$add(model_name, predictors[!is_num])
  predictors[is_num]
}

# ---- NA handling helpers ----------------------------------------------------
zero_fill <- function(df, cols) {
  cols <- intersect(cols, names(df))
  for (c in cols) { x <- as.numeric(df[[c]]); x[is.na(x)] <- 0; df[[c]] <- x }
  df
}

# Adds binary miss-indicator columns for native-NA models.
# NOTE: 21_submission_assembly.R must apply the same expansion when refitting.
prepare_for_native_na <- function(tr_df, va_df, predictors) {
  has_na <- vapply(predictors, function(p) {
    (p %in% names(tr_df) && any(is.na(tr_df[[p]]))) ||
    (p %in% names(va_df) && any(is.na(va_df[[p]])))
  }, logical(1))
  new_preds <- predictors
  for (p in predictors[has_na]) {
    ind <- paste0(p, "_miss_ind")
    # Skip if the miss indicator is already an explicit predictor (e.g. r_harmonized_miss_ind
    # in the enriched set) — adding it again causes duplicate-column crashes in LGBM/XGB.
    if (ind %in% new_preds) next
    tr_df[[ind]] <- as.integer(is.na(tr_df[[p]]))
    va_df[[ind]] <- as.integer(is.na(va_df[[p]]))
    new_preds <- c(new_preds, ind)
  }
  list(tr = tr_df, va = va_df, predictors = new_preds)
}

# ---- MonoXGB ----------------------------------------------------------------
fit_mono_xgb <- function(train, outcome_col, predictors, weights = NULL,
                         nrounds_max  = 1500L,
                         early_stopping_rounds = 30L,
                         nfold = 5L) {
  x <- as.matrix(train[, predictors, drop = FALSE])
  y <- as.numeric(train[[outcome_col]])
  w <- pmax(tidyr::replace_na(
    if (is.null(weights)) rep(1, nrow(train)) else weights, 1.0), 1e-6)

  # Resolve monotone direction from global DIR_MONO; unknown predictors -> 0 (free)
  mono <- vapply(predictors, function(p) {
    d <- DIR_MONO[p]
    if (length(d) == 0L || is.na(d)) 0L else as.integer(d)
  }, integer(1))

  dtrain <- xgboost::xgb.DMatrix(data = x, label = y, weight = w)

  params <- list(
    objective            = "binary:logistic",
    eval_metric          = "logloss",
    tree_method          = "hist",
    eta                  = 0.03,
    max_depth            = 3,
    min_child_weight     = 15,
    subsample            = 0.85,
    colsample_bytree     = 0.8,
    gamma                = 0.1,
    monotone_constraints = paste0("(", paste(mono, collapse = ","), ")")
  )

  # Pick nrounds via CV-based early stopping
  set.seed(CONFIG$seed)
  cv <- xgboost::xgb.cv(
    params                = params,
    data                  = dtrain,
    nrounds               = nrounds_max,
    nfold                 = nfold,
    early_stopping_rounds = early_stopping_rounds,
    verbose               = 0,
    maximize              = FALSE
  )
  best_iter <- cv$best_iteration
  if (is.null(best_iter) || best_iter < 1L) best_iter <- 100L

  set.seed(CONFIG$seed)
  mod <- xgboost::xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = best_iter,
    verbose = 0
  )

  list(model      = mod,
       predictors = predictors,
       best_iter  = best_iter,
       type       = "MonoXGB")
}

pred_mono_xgb <- function(model_obj, newdata, predictors, shrink_factor = 0.01) {
  x_new <- as.matrix(newdata[, model_obj$predictors, drop = FALSE])
  p     <- as.numeric(predict(model_obj$model, xgboost::xgb.DMatrix(x_new)))
  p     <- p * (1 - 2 * shrink_factor) + shrink_factor
  pmax(pmin(p, 1), 0)
}

# ---- LLM bin correction (matches 17/18) ------------------------------------
n_bin_of <- function(n_o) {
  cut(suppressWarnings(as.numeric(n_o)),
      breaks = NBIN_BREAKS, labels = NBIN_LABELS, right = TRUE)
}
build_shift_table <- function(train_outcomes, train_llm, train_n_o) {
  d <- tibble::tibble(outcome = train_outcomes, llm_p = train_llm,
                      n_bin = n_bin_of(train_n_o)) |>
    dplyr::filter(!is.na(n_bin), !is.na(outcome), !is.na(llm_p))
  d |> dplyr::group_by(n_bin) |>
    dplyr::summarise(shift = mean(outcome - llm_p), .groups = "drop")
}
apply_shift_table <- function(p_llm, n_o, shift_tbl, fac = 0.6) {
  nb    <- n_bin_of(n_o)
  shift <- shift_tbl$shift[match(nb, shift_tbl$n_bin)]
  shift[is.na(shift)] <- 0
  pmin(pmax(as.numeric(p_llm) + fac * shift, 1e-3), 1 - 1e-3)
}

# ---- feature stability ------------------------------------------------------
compute_feature_stability <- function(gt_pool, candidates, outcome_col,
                                      health_col) {
  y    <- as.numeric(gt_pool[[outcome_col]])
  is_h <- gt_pool[[health_col]] == 1
  is_p <- gt_pool[[health_col]] == 0
  rho_safe <- function(x, yy) {
    ok <- !is.na(x) & !is.na(yy)
    if (sum(ok) < 10L) return(NA_real_)
    suppressWarnings(cor(rank(x[ok]), rank(yy[ok])))
  }
  rows <- lapply(candidates, function(f) {
    if (!f %in% names(gt_pool)) return(NULL)
    x <- gt_pool[[f]]
    if (!is.numeric(x) && !is.logical(x) && !is.integer(x)) return(NULL)
    rho_all <- rho_safe(x, y)
    rho_h   <- rho_safe(x[is_h], y[is_h])
    rho_p   <- rho_safe(x[is_p], y[is_p])
    sign_flip <- !is.na(rho_h) && !is.na(rho_p) &&
      abs(rho_h) > 0.1 && abs(rho_p) > 0.1 && sign(rho_h) != sign(rho_p)
    tibble::tibble(feature = f, rho_all = rho_all, rho_psych = rho_p,
                   rho_health = rho_h, sign_flip = sign_flip,
                   n_obs_psych = sum(is_p & !is.na(x)),
                   n_obs_health = sum(is_h & !is.na(x)))
  })
  dplyr::bind_rows(rows)
}

# ---- nested-CV temperature calibration on health OOF -----------------------
nested_cv_temperature <- function(oof, fold_id, is_health, outcome) {
  cal_oof  <- oof
  fold_ids <- sort(unique(fold_id))
  T_per_fold <- stats::setNames(rep(NA_real_, length(fold_ids)),
                                as.character(fold_ids))
  for (k in fold_ids) {
    other <- which(fold_id != k & is_health & !is.na(oof) & !is.na(outcome))
    own   <- which(fold_id == k & !is.na(oof))
    if (length(other) < .CALIB_MIN_N) next
    cal <- fit_calibrator(oof[other], outcome[other], method = "temperature")
    if (cal$method != "temperature") next
    T_per_fold[as.character(k)] <- cal$T
    cal_oof[own] <- apply_calibrator(cal, oof[own])
  }
  list(calibrated = cal_oof, T_per_fold = T_per_fold)
}

# ---- screen_EQ / screen_Mono (from step 19) --------------------------------
# screen_features: per-fold transferability filter (19's screen() function).
# Returns character vector of selected feature names.
screen_features <- function(train_df, outcome_col, is_health_col,
                             cand, test_cov, smd_tt,
                             maxk = 30L, min_rho = 0.05,
                             min_cov = 0.5, max_smd = 0.5,
                             anti_rho = 0.10, min_h = 25L, min_p = 50L) {
  y    <- as.numeric(train_df[[outcome_col]])
  is_h <- train_df[[is_health_col]] == 1
  is_p <- !is_h
  rho_s <- function(x, yy) {
    ok <- !is.na(x) & !is.na(yy)
    if (sum(ok) < 20L) return(NA_real_)
    suppressWarnings(cor(rank(x[ok]), rank(yy[ok])))
  }
  scored <- lapply(cand, function(f) {
    if (!f %in% names(train_df)) return(NULL)
    x     <- train_df[[f]]
    r_all <- rho_s(x, y)
    if (is.na(r_all) || abs(r_all) < min_rho)        return(NULL)
    if (is.na(test_cov[f]) || test_cov[f] < min_cov) return(NULL)
    if (!is.na(smd_tt[f]) && abs(smd_tt[f]) > max_smd) return(NULL)
    if (sum(is_h, na.rm = TRUE) >= min_h && sum(is_p, na.rm = TRUE) >= min_p) {
      rh <- rho_s(x[is_h], y[is_h]); rp <- rho_s(x[is_p], y[is_p])
      if (!is.na(rh) && !is.na(rp) && abs(rh) > anti_rho && abs(rp) > anti_rho &&
          sign(rh) != sign(rp)) return(NULL)
    }
    data.frame(feature = f, ar = abs(r_all), stringsAsFactors = FALSE)
  })
  scored <- do.call(rbind, scored)
  if (is.null(scored) || nrow(scored) == 0L) return(character(0))
  scored <- scored[order(-scored$ar), , drop = FALSE]
  kept <- character(0)
  for (f in scored$feature) {
    if (length(kept) == 0L) { kept <- f; next }
    cc <- suppressWarnings(sapply(kept, function(k)
      abs(cor(train_df[[f]], train_df[[k]], use = "pairwise.complete.obs"))))
    if (all(is.na(cc)) || max(cc, na.rm = TRUE) < 0.95) kept <- c(kept, f)
    if (length(kept) >= maxk) break
  }
  kept
}

# Equal-weight ensemble of RF+GAM+ElNet+XGB on a given feature set.
# Identical to 19's fit_eq/pred_eq. Uses zero-fill internally.
fit_eq_screen <- function(train_df, outcome_col, predictors) {
  trz <- zero_fill(train_df, predictors)
  w   <- rep(1, nrow(trz))
  list(
    predictors = predictors,
    RF    = tryCatch(fit_rf(trz, outcome_col, predictors, w),    error = function(e) NULL),
    GAM   = tryCatch(fit_gam(trz, outcome_col, predictors, w),   error = function(e) NULL),
    ElNet = tryCatch(fit_elnet(trz, outcome_col, predictors, w), error = function(e) NULL),
    XGB   = tryCatch(fit_xgb(trz, outcome_col, predictors, w),   error = function(e) NULL)
  )
}

pred_eq_screen <- function(model_obj, newdata) {
  ndz <- zero_fill(newdata, model_obj$predictors)
  cols <- cbind(
    RF    = if (!is.null(model_obj$RF))    pred_rf(model_obj$RF, ndz, model_obj$predictors)    else NA_real_,
    GAM   = if (!is.null(model_obj$GAM))   pred_gam(model_obj$GAM, ndz, model_obj$predictors)  else NA_real_,
    ElNet = if (!is.null(model_obj$ElNet)) pred_elnet(model_obj$ElNet, ndz, model_obj$predictors) else NA_real_,
    XGB   = if (!is.null(model_obj$XGB))   pred_xgb(model_obj$XGB, ndz, model_obj$predictors)  else NA_real_
  )
  valid <- colSums(is.na(cols)) == 0
  if (sum(valid) == 0L) return(rep(NA_real_, nrow(newdata)))
  rowMeans(cols[, valid, drop = FALSE])
}

# Monotone-constrained gradient booster on screened features (19's fit_mono).
# Uses binary:logistic (fix vs 19's reg:squarederror) since outcome is binary.
fit_mono_screen <- function(train_df, outcome_col, predictors) {
  trz <- zero_fill(train_df, predictors)
  x   <- as.matrix(trz[, predictors, drop = FALSE])
  y   <- as.numeric(trz[[outcome_col]])
  mc  <- vapply(predictors, function(p) {
    d <- DIR_MONO[p]; if (is.na(d)) 0L else as.integer(d)
  }, integer(1))
  dtrain <- xgboost::xgb.DMatrix(data = x, label = y)
  params <- list(
    objective            = "binary:logistic",
    eval_metric          = "logloss",
    eta                  = 0.03, max_depth = 3, min_child_weight = 15,
    subsample = 0.85, colsample_bytree = 0.8, gamma = 0.1,
    monotone_constraints = paste0("(", paste(mc, collapse = ","), ")")
  )
  mod <- tryCatch(
    xgboost::xgb.train(params = params, data = dtrain, nrounds = 250, verbose = 0),
    error = function(e) { warning("fit_mono_screen: ", e$message); NULL }
  )
  list(model = mod, predictors = predictors)
}

pred_mono_screen <- function(model_obj, newdata, shrink_factor = 0.01) {
  if (is.null(model_obj$model)) return(rep(NA_real_, nrow(newdata)))
  ndz <- zero_fill(newdata, model_obj$predictors)
  x   <- as.matrix(ndz[, model_obj$predictors, drop = FALSE])
  p   <- as.numeric(predict(model_obj$model, xgboost::xgb.DMatrix(x)))
  p   <- p * (1 - 2 * shrink_factor) + shrink_factor
  pmax(pmin(p, 1), 0)
}

# ---- softmax-NNLS meta-learner (from 18) ------------------------------------
fitw <- function(Z, y, w = NULL) {
  M <- ncol(Z)
  if (is.null(w)) w <- rep(1, length(y))
  w <- w / mean(w)
  obj <- function(p) {
    a <- exp(p - max(p)); a <- a / sum(a)
    sum(w * (as.numeric(Z %*% a) - y)^2) / sum(w)
  }
  o <- optim(rep(0, M), obj, method = "BFGS", control = list(maxit = 500))
  a <- exp(o$par - max(o$par)); a <- a / sum(a)
  names(a) <- colnames(Z)
  a
}

# ---- post-CV ensemble helpers -----------------------------------------------
# Per-fold NNLS stack: for each fold k, fit NNLS weights on OOF from other
# folds, apply to fold k. Leakage-free.
compute_nnls_oof <- function(oof_storage, base_keys, gt_pool, outcome_col) {
  n        <- nrow(gt_pool)
  y        <- as.numeric(gt_pool[[outcome_col]])
  fold_ids <- sort(unique(gt_pool$fold_id))
  oof_ens  <- rep(NA_real_, n)
  weights_per_fold <- list()

  for (k in fold_ids) {
    tr_idx <- which(gt_pool$fold_id != k)
    va_idx <- which(gt_pool$fold_id == k)
    Z_tr <- do.call(cbind, lapply(base_keys, function(key) {
      v <- oof_storage[[key]]; if (is.null(v)) rep(NA_real_, length(tr_idx)) else v[tr_idx]
    }))
    colnames(Z_tr) <- base_keys
    y_tr <- y[tr_idx]
    ok   <- stats::complete.cases(Z_tr) & is.finite(y_tr)
    if (sum(ok) < max(20L, length(base_keys) + 1L)) {
      w <- stats::setNames(rep(1 / length(base_keys), length(base_keys)), base_keys)
    } else {
      w <- tryCatch(
        fitw(Z_tr[ok, , drop = FALSE], y_tr[ok]),
        error = function(e) stats::setNames(
          rep(1 / length(base_keys), length(base_keys)), base_keys)
      )
    }
    weights_per_fold[[as.character(k)]] <- w
    Z_va <- do.call(cbind, lapply(base_keys, function(key) {
      v <- oof_storage[[key]]; if (is.null(v)) rep(NA_real_, length(va_idx)) else v[va_idx]
    }))
    colnames(Z_va) <- base_keys
    oof_ens[va_idx] <- as.numeric(Z_va %*% w)
  }
  list(oof = oof_ens, weights_per_fold = weights_per_fold)
}

# Greedy uncorrelated ensemble: rank by brier_chal_health_cal, add each model
# only if |cor| < threshold with all already-selected on chal_health OOF rows.
compute_greedy_oof <- function(oof_storage, candidate_keys, summary_df,
                                gt_pool, cor_threshold = 0.7, top_n_start = 20L) {
  is_r12   <- gt_pool$dataset %in% c("round1", "round2")
  is_h     <- gt_pool$is_health_strat == 1
  sel_rows <- is_r12 & is_h

  ranked <- summary_df |>
    dplyr::filter(key %in% candidate_keys, !is.na(brier_chal_health_cal)) |>
    dplyr::slice_min(brier_chal_health_cal, n = top_n_start, with_ties = FALSE) |>
    dplyr::arrange(brier_chal_health_cal) |>
    dplyr::pull(key)

  selected <- character(0)
  for (key in ranked) {
    p <- oof_storage[[key]]
    if (is.null(p)) next
    if (length(selected) == 0L) { selected <- key; next }
    max_cor <- max(vapply(selected, function(s) {
      ps <- oof_storage[[s]]
      ok <- sel_rows & !is.na(p) & !is.na(ps)
      if (sum(ok) < 10L) return(0)
      abs(cor(p[ok], ps[ok]))
    }, numeric(1)), na.rm = TRUE)
    if (max_cor < cor_threshold) selected <- c(selected, key)
  }
  if (length(selected) == 0L && length(ranked) > 0L) selected <- ranked[1L]

  mat      <- do.call(cbind, lapply(selected, function(k) oof_storage[[k]]))
  oof_mean <- rowMeans(mat, na.rm = TRUE)
  message(sprintf("[Greedy ens] %d models selected: %s",
                  length(selected), paste(selected, collapse = ", ")))
  list(selected = selected, oof_equal = oof_mean)
}

# ---- imputation diagnostics -------------------------------------------------
# Improved version: density + mean stability (existing) + fraction missing +
# range plausibility + between-imputation variance.
write_imputation_diagnostics <- function(tr_raw, imputed_list, predictors,
                                         out_pdf, top_n = 10) {
  if (length(imputed_list) == 0L) return(invisible())
  miss_pct <- vapply(predictors, function(p) {
    if (!p %in% names(tr_raw)) return(0)
    mean(is.na(tr_raw[[p]]))
  }, numeric(1))
  miss_pct <- miss_pct[miss_pct > 0]
  if (length(miss_pct) == 0L) {
    message("[20] no variables had missingness in fold 1; skipping diag PDF.")
    return(invisible())
  }
  top_vars <- names(sort(miss_pct, decreasing = TRUE))[
    seq_len(min(top_n, length(miss_pct)))]

  grDevices::pdf(out_pdf, width = 10, height = 7, onefile = TRUE)
  on.exit(grDevices::dev.off(), add = TRUE)

  # Page 1: fraction missing bar chart
  op <- graphics::par(mar = c(8, 4, 3, 1))
  barplot(sort(miss_pct, decreasing = TRUE) * 100,
          las = 2, col = "#e74c3c",
          main = "Fraction missing per variable (training fold 1)",
          ylab = "% missing")
  graphics::abline(h = 30, lty = 2, col = "grey50")
  graphics::par(op)

  m <- length(imputed_list)
  for (v in top_vars) {
    obs_idx  <- !is.na(tr_raw[[v]])
    obs_vals <- as.numeric(tr_raw[[v]][obs_idx])
    obs_min  <- min(obs_vals, na.rm = TRUE)
    obs_max  <- max(obs_vals, na.rm = TRUE)

    # Pool all imputed values for this variable across m datasets
    imp_vals_all <- numeric(0)
    imp_means    <- numeric(m)
    out_of_range <- 0L
    total_imp    <- 0L
    for (i in seq_len(m)) {
      d_i <- imputed_list[[i]]
      if (!v %in% names(d_i)) next
      iv <- as.numeric(d_i[[v]][!obs_idx])
      imp_vals_all <- c(imp_vals_all, iv)
      imp_means[i] <- mean(iv, na.rm = TRUE)
      out_of_range <- out_of_range + sum(iv < obs_min | iv > obs_max, na.rm = TRUE)
      total_imp    <- total_imp + sum(!is.na(iv))
    }
    if (length(obs_vals) < 3L || length(imp_vals_all) < 3L) next

    pct_oob <- if (total_imp > 0L) 100 * out_of_range / total_imp else 0
    cv_imp  <- if (mean(imp_means, na.rm=TRUE) != 0)
      sd(imp_means, na.rm=TRUE) / abs(mean(imp_means, na.rm=TRUE)) else NA_real_

    op <- graphics::par(mfrow = c(1, 3), oma = c(0, 0, 3, 0), mar = c(4, 4, 2, 1))

    # Panel 1: density overlay
    d_obs <- stats::density(obs_vals, na.rm = TRUE)
    d_imp <- stats::density(imp_vals_all, na.rm = TRUE)
    xlim  <- range(c(d_obs$x, d_imp$x)); ylim <- range(c(d_obs$y, d_imp$y))
    plot(d_obs, col = "#2c3e50", lwd = 2, xlim = xlim, ylim = ylim,
         main = sprintf("%s (%.0f%% miss)", v, 100 * miss_pct[v]),
         xlab = v, ylab = "density")
    graphics::lines(d_imp, col = "#e74c3c", lwd = 2, lty = 2)
    graphics::abline(v = c(obs_min, obs_max), col = "#27ae60", lty = 3)
    graphics::legend("topright", c("observed", "imputed", "obs range"),
                     col = c("#2c3e50","#e74c3c","#27ae60"),
                     lwd = 2, lty = c(1,2,3), bty = "n", cex = 0.75)

    # Panel 2: mean per imputation index (stability)
    plot(seq_len(m), imp_means, type = "b", pch = 19, col = "#e74c3c",
         xlab = "imputation index m", ylab = "mean of imputed values",
         main = sprintf("mean stability (CV=%.2f)", cv_imp %||% NA))
    graphics::abline(h = mean(obs_vals), col = "#2c3e50", lty = 2)
    graphics::legend("topright", c("imp mean", "obs mean"),
                     col = c("#e74c3c","#2c3e50"), lwd=2, lty=c(1,2), bty="n", cex=0.75)

    # Panel 3: range plausibility text panel
    plot.new()
    graphics::text(0.5, 0.7,
                   sprintf("Obs range: [%.3g, %.3g]", obs_min, obs_max),
                   cex = 1.0)
    graphics::text(0.5, 0.5,
                   sprintf("Out-of-range imp: %.1f%%", pct_oob),
                   col = if (pct_oob > 5) "#e74c3c" else "#27ae60",
                   cex = 1.0)
    graphics::text(0.5, 0.3,
                   sprintf("Between-imp CV: %s",
                           if (!is.na(cv_imp)) sprintf("%.3f", cv_imp) else "N/A"),
                   col = if (!is.na(cv_imp) && cv_imp > 0.1) "#e74c3c" else "#27ae60",
                   cex = 1.0)
    graphics::mtext(
      sprintf("%s — imputation diagnostics (m=%d)", v, m),
      outer = TRUE, cex = 1.0, line = 1)
    graphics::par(op)
  }
  invisible()
}

# Optional convergence trace check: runs a separate minimal mice() call on
# training data only (m=3) and saves plot.mids() trace plots to PDF.
# NOTE: This is an APPROXIMATION of the actual 05_imputation.R setup
# (uses default pmm for all variables, not 2lonly.pmm for paper-level vars).
# For a definitive convergence check, 05_imputation.R should be modified to
# optionally return the mice object. See README/diagnostics note.
run_mice_convergence_check <- function(tr_raw, predictors, outcome_col,
                                        maxit, out_pdf, seed = 42L) {
  pool_vars  <- intersect(c(predictors, outcome_col), names(tr_raw))
  data_check <- tr_raw[, pool_vars, drop = FALSE]
  # Drop non-numeric columns — PMM requires numeric input. Coercing factor
  # codes to integer would be statistically incorrect (arbitrary level ordering).
  non_num <- names(data_check)[vapply(names(data_check), function(v) {
    !is.numeric(data_check[[v]]) && !is.integer(data_check[[v]]) &&
    !is.logical(data_check[[v]])
  }, logical(1))]
  if (length(non_num) > 0L) data_check <- data_check[, setdiff(names(data_check), non_num), drop = FALSE]
  pool_vars <- names(data_check)

  has_miss <- vapply(pool_vars, function(v) anyNA(data_check[[v]]), logical(1))
  if (sum(has_miss) == 0L) {
    message("[20] Convergence check: no missing values in fold 1 pool, skipping.")
    return(invisible())
  }

  pm <- tryCatch(
    mice::quickpred(data_check, mincor = 0.10, exclude = outcome_col),
    error = function(e) {
      m_f <- matrix(1L, length(pool_vars), length(pool_vars),
                    dimnames = list(pool_vars, pool_vars))
      diag(m_f) <- 0L; m_f[, outcome_col] <- 0L; m_f
    }
  )
  if (outcome_col %in% rownames(pm)) pm[outcome_col, ] <- 0L

  mice_diag <- tryCatch(
    suppressWarnings(mice::mice(
      data_check, m = 3L, method = "pmm", predictorMatrix = pm,
      maxit = maxit, printFlag = FALSE, seed = seed
    )),
    error = function(e) {
      message("[20] Convergence check mice failed: ", e$message); NULL
    }
  )
  if (is.null(mice_diag)) return(invisible())

  vars_to_plot <- pool_vars[has_miss][seq_len(min(6L, sum(has_miss)))]
  grDevices::pdf(out_pdf, width = 10, height = 6)
  on.exit(grDevices::dev.off(), add = TRUE)
  tryCatch(
    print(plot(mice_diag, y = vars_to_plot,
               main = "MICE convergence traces (approx — pmm only, m=3)")),
    error = function(e) {
      message("[20] Convergence plot failed: ", e$message)
      plot.new()
      graphics::text(0.5, 0.5, paste("Plot failed:", e$message), cex = 0.8)
    }
  )
  message("[20] Convergence check PDF written: ", out_pdf)
  invisible()
}

# =============================================================================
# 4. DATA LOADING + SCREEN PRE-COMPUTATION
# =============================================================================

message("[20] loading data ...")
train_base <- readRDS(here::here("data/train_base.rds"))
test_base  <- readRDS(here::here("data/test_base.rds"))
ens_raw    <- readr::read_csv(RUN_CFG$llm_csv_path, show_col_types = FALSE)

gt_pool <- train_base |>
  dplyr::filter(!is.na(.data[[RUN_CFG$outcome_col]]),
                dataset %in% c("training", "round1", "round2"))

if ("is_boyce_soto" %in% names(gt_pool)) {
  n_before <- nrow(gt_pool)
  gt_pool  <- gt_pool |> dplyr::filter(is.na(is_boyce_soto) | !is_boyce_soto)
  if (nrow(gt_pool) < n_before)
    message("[20] excluded ", n_before - nrow(gt_pool), " is_boyce_soto rows.")
}

# Prior+shrinkage features (LLM substantive prior + power-for-observed shrinkage;
# see PRIOR_SHRINKAGE_REPORT.md). Health rows only; NA elsewhere. These are NOT in
# any PREDICTOR_SET, so they are excluded from imputation and the feature models —
# they are consumed only by the custom Prior_Shrinkage model in the fold loop.
if (isTRUE(RUN_CFG$include_prior_shrinkage) &&
    !is.null(RUN_CFG$prior_features_path) && file.exists(RUN_CFG$prior_features_path)) {
  .prior_feats <- readr::read_csv(RUN_CFG$prior_features_path, show_col_types = FALSE)
  gt_pool <- dplyr::left_join(gt_pool, .prior_feats[, c("claim_id", "prior_3fam", "p_base")],
                              by = "claim_id")
  message(sprintf("[20] prior+shrinkage features merged: %d / %d rows have prior_3fam.",
                  sum(!is.na(gt_pool$prior_3fam)), nrow(gt_pool)))
}

if ("health_related_big" %in% names(gt_pool)) {
  gt_pool$is_health_strat <- as.integer(
    !is.na(gt_pool$health_related_big) &
    (gt_pool$health_related_big == 1 | gt_pool$health_related_big == TRUE))
} else if ("is_health" %in% names(gt_pool)) {
  gt_pool$is_health_strat <- as.integer(
    tidyr::replace_na(as.integer(gt_pool$is_health), 0L))
} else {
  stop("[20] gt_pool lacks both 'health_related_big' and 'is_health'.")
}

# IPW dropped: always use unit weights.
gt_pool$w_ipw <- 1.0

message(sprintf(
  "[20] gt_pool n=%d | FORRT=%d R1=%d R2=%d | health=%d non-health=%d",
  nrow(gt_pool),
  sum(gt_pool$dataset == "training"),
  sum(gt_pool$dataset == "round1"),
  sum(gt_pool$dataset == "round2"),
  sum(gt_pool$is_health_strat == 1),
  sum(gt_pool$is_health_strat == 0)
))

# Pre-compute candidate pool and statistics for screen_features() ---------
# Mirrors 19's logic: numeric features in gt_pool ∩ test_base, non-zero-var,
# excluding leakage/outcome/ID columns.
{
  num_both <- intersect(
    names(gt_pool)[vapply(gt_pool, is.numeric, logical(1))],
    names(test_base)[vapply(test_base, is.numeric, logical(1))]
  )
  screen_cand <- setdiff(num_both, SCREEN_STOP_COLS)
  screen_cand <- screen_cand[vapply(screen_cand, function(f) {
    isTRUE(stats::sd(gt_pool[[f]], na.rm = TRUE) > 1e-9)
  }, logical(1))]

  screen_test_cov <- stats::setNames(
    vapply(screen_cand, function(f) mean(!is.na(test_base[[f]])), numeric(1)),
    screen_cand)

  screen_smd_tt <- stats::setNames(vapply(screen_cand, function(f) {
    s <- stats::sd(gt_pool[[f]], na.rm = TRUE)
    if (is.na(s) || s < 1e-9) return(NA_real_)
    (mean(test_base[[f]], na.rm = TRUE) - mean(gt_pool[[f]], na.rm = TRUE)) / s
  }, numeric(1)), screen_cand)

  message(sprintf("[20] screen_features candidate pool: %d features", length(screen_cand)))
}

# =============================================================================
# 5. BUILD FOLDS
# =============================================================================

gt_pool$fold_id <- make_stratified_group_folds(
  df = gt_pool, group_var = RUN_CFG$group_var,
  strat_vars = RUN_CFG$stratify_vars,
  n_folds = RUN_CFG$n_folds, seed = RUN_CFG$seed)

cat("\nFold composition (rows per fold x dataset x health flag):\n")
fold_comp <- gt_pool |>
  dplyr::count(fold_id, dataset, is_health_strat) |>
  dplyr::mutate(label = paste(dataset, ifelse(is_health_strat == 1, "h", "p"), sep = "_")) |>
  dplyr::select(fold_id, label, n) |>
  tidyr::pivot_wider(names_from = label, values_from = n, values_fill = 0L) |>
  dplyr::arrange(fold_id)
print(fold_comp)

# =============================================================================
# 6. RESOLVE PREDICTOR SETS + MODEL REGISTRY
# =============================================================================

resolve_pset_spec <- function(spec, available = names(PREDICTOR_SETS)) {
  if (length(spec) == 1L && identical(spec, "all")) return(available)
  spec <- as.character(spec)
  unknown <- setdiff(spec, available)
  if (length(unknown) > 0L)
    stop("Unknown predictor set(s): ", paste(unknown, collapse = ", "))
  spec
}

active_psets <- resolve_pset_spec(RUN_CFG$pset_spec)
message("[20] active predictor sets: ", paste(active_psets, collapse = ", "))

model_specs <- list()
for (m in MODEL_GRID) {
  model_specs[[m$name]] <- list(
    name                = m$name,
    fit                 = m$fit,
    pred                = m$pred,
    imputation_strategy = if (m$name == "BART") "pmm"
                          else (m$imputation_strategy %||% "none")
  )
}
if (RUN_CFG$include_mono_xgb) {
  model_specs[["MonoXGB"]] <- list(
    name = "MonoXGB", fit = fit_mono_xgb, pred = pred_mono_xgb,
    imputation_strategy = "none")
}

if (!is.null(RUN_CFG$model_subset)) {
  bad <- setdiff(RUN_CFG$model_subset,
                 c(names(model_specs), "LLM_raw", "LLM_bincorr"))
  if (length(bad) > 0L)
    stop("Unknown models in model_subset: ", paste(bad, collapse = ", "))
  model_specs <- model_specs[intersect(names(model_specs), RUN_CFG$model_subset)]
}
active_model_names <- names(model_specs)
message("[20] feature models: ", paste(active_model_names, collapse = ", "))

use_llm_raw     <- RUN_CFG$include_llm_raw &&
  (is.null(RUN_CFG$model_subset) || "LLM_raw" %in% RUN_CFG$model_subset)
use_llm_bincorr <- RUN_CFG$include_llm_bincorr &&
  (is.null(RUN_CFG$model_subset) || "LLM_bincorr" %in% RUN_CFG$model_subset)
use_screen_eq   <- isTRUE(RUN_CFG$include_screen_eq)
use_screen_mono <- isTRUE(RUN_CFG$include_screen_mono)
use_prior_shrinkage <- isTRUE(RUN_CFG$include_prior_shrinkage) &&
  all(c("prior_3fam", "p_base") %in% names(gt_pool))

# =============================================================================
# 7. FEATURE STABILITY
# =============================================================================

all_candidate_features <- unique(unlist(PREDICTOR_SETS[active_psets]))
stab <- compute_feature_stability(
  gt_pool     = gt_pool,
  candidates  = all_candidate_features,
  outcome_col = RUN_CFG$outcome_col,
  health_col  = "is_health_strat")
readr::write_csv(stab, file.path(RUN_CFG$output_dir, "20_feature_stability.csv"))
n_flips <- sum(stab$sign_flip, na.rm = TRUE)
message(sprintf("[20] feature stability: %d / %d features sign-flip", n_flips, nrow(stab)))
if (n_flips > 0L) {
  cat("\nSign-flipping features:\n")
  print(stab |> dplyr::filter(sign_flip) |>
          dplyr::select(feature, rho_psych, rho_health, n_obs_health) |>
          dplyr::arrange(dplyr::desc(abs(rho_psych))))
}

# =============================================================================
# 8. MAIN CV LOOP
# =============================================================================
# Imputation efficiency: apply_imputation always builds its pool from ALL
# PREDICTOR_SETS regardless of pset_name — the pool is identical across psets
# for the same fold. Running it once per fold (not per pset x fold) saves ~2/3
# of MICE computation time. We cache the full imputation result per fold and
# slice to each pset's variables on demand.

# Combined pset covering all active pset variables — used for the shared
# per-fold MICE call. get_valid_predictors() filters to those actually present
# and below the missing threshold, so over-specifying is safe.
PREDICTOR_SETS[["_all_active"]] <- unique(unlist(PREDICTOR_SETS[active_psets]))

# Slice helper: filter imputed datasets to columns relevant for a pset.
# outcome_col must be passed so statistical_success is retained in the sliced
# imputed datasets — 05_imputation.R includes it via .CORE_DONOR_VARS but it
# is not in predictors_pset (absent from test_base), so without this it would
# be stripped and every spec$fit() call would error with "object not found".
slice_imp_to_pset <- function(imp_full, pset_vars, outcome_col = NULL) {
  if (is.null(imp_full) || !isTRUE(imp_full$is_imputed)) return(imp_full)
  extra <- if (!is.null(outcome_col) && outcome_col %in% names(imp_full$train[[1]]))
    outcome_col else character(0)
  cols <- c(intersect(pset_vars, names(imp_full$train[[1]])), extra)
  list(
    train      = lapply(imp_full$train, function(d) d[, cols, drop = FALSE]),
    test       = lapply(imp_full$test,  function(d) d[, cols, drop = FALSE]),
    is_imputed = TRUE
  )
}

# Per-fold imputation cache (populated on first PMM-needing pset for each fold).
# Initialised above; checkpoint may have pre-filled some entries.

oof_storage <- list()
key_meta <- tibble::tibble(
  key = character(0), weighting = character(0),
  pset = character(0), model = character(0))
audit_log <- make_audit_log()

# LLM_raw passthrough
llm_lookup <- ens_raw |>
  dplyr::select(claim_id, p_replication_observed) |>
  dplyr::distinct(claim_id, .keep_all = TRUE)
gt_llm    <- dplyr::left_join(gt_pool |> dplyr::select(claim_id),
                              llm_lookup, by = "claim_id")
llm_p_gt  <- gt_llm$p_replication_observed

if (use_llm_raw) {
  oof_storage[["LLM_raw"]] <- llm_p_gt
  key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
    key = "LLM_raw", weighting = NA_character_,
    pset = NA_character_, model = "LLM_raw"))
  message(sprintf("[20] LLM_raw: %d / %d rows have scores.",
                  sum(!is.na(llm_p_gt)), length(llm_p_gt)))
}

if (use_llm_bincorr && "n_o" %in% names(gt_pool)) {
  bc       <- rep(NA_real_, nrow(gt_pool))
  fold_ids <- sort(unique(gt_pool$fold_id))
  for (k in fold_ids) {
    tr_idx <- which(gt_pool$fold_id != k & !is.na(llm_p_gt) &
                    !is.na(gt_pool[[RUN_CFG$outcome_col]]))
    va_idx <- which(gt_pool$fold_id == k & !is.na(llm_p_gt))
    if (length(tr_idx) < 20L || length(va_idx) == 0L) next
    shift_tbl <- build_shift_table(
      train_outcomes = gt_pool[[RUN_CFG$outcome_col]][tr_idx],
      train_llm      = llm_p_gt[tr_idx],
      train_n_o      = gt_pool$n_o[tr_idx])
    bc[va_idx] <- apply_shift_table(
      p_llm     = llm_p_gt[va_idx],
      n_o       = gt_pool$n_o[va_idx],
      shift_tbl = shift_tbl,
      fac       = RUN_CFG$bin_correction_factor)
  }
  oof_storage[["LLM_bincorr"]] <- bc
  key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
    key = "LLM_bincorr", weighting = NA_character_,
    pset = NA_character_, model = "LLM_bincorr"))
  message(sprintf("[20] LLM_bincorr: %d / %d rows.", sum(!is.na(bc)), length(bc)))
}

# screen_EQ / screen_Mono OOF storage (not pset-keyed, own feature selection)
if (use_screen_eq) {
  oof_storage[["screen_EQ"]] <- rep(NA_real_, nrow(gt_pool))
  key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
    key = "screen_EQ", weighting = NA_character_,
    pset = "screen", model = "screen_EQ"))
}
if (use_screen_mono) {
  oof_storage[["screen_Mono"]] <- rep(NA_real_, nrow(gt_pool))
  key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
    key = "screen_Mono", weighting = NA_character_,
    pset = "screen", model = "screen_Mono"))
}
if (use_prior_shrinkage) {
  oof_storage[["Prior_Shrinkage"]] <- rep(NA_real_, nrow(gt_pool))
  key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
    key = "Prior_Shrinkage", weighting = NA_character_,
    pset = "prior", model = "Prior_Shrinkage"))
}

# Pre-build all (pset x model) OOF slots and key_meta rows in one pass —
# avoids O(n^2) bind_rows growth inside the CV loop. Also required so the
# checkpoint overlay (below) can write into already-existing slots.
{
  fold_ids       <- sort(unique(gt_pool$fold_id))
  new_model_rows <- dplyr::bind_rows(lapply(active_psets, function(ps) {
    preds_ps <- intersect(intersect(PREDICTOR_SETS[[ps]], names(gt_pool)),
                          names(test_base))
    if (length(preds_ps) < 2L) return(NULL)
    dplyr::bind_rows(lapply(active_model_names, function(mn) {
      tibble::tibble(key = paste("none", ps, mn, sep = "."),
                     weighting = "none", pset = ps, model = mn)
    }))
  }))
  key_meta <- dplyr::bind_rows(key_meta, new_model_rows)
  for (k_init in new_model_rows$key)
    oof_storage[[k_init]] <- rep(NA_real_, nrow(gt_pool))
}

# =============================================================================
# 8a. CHECKPOINT LOAD
# =============================================================================
# If a valid checkpoint exists (same pset_spec and model_subset), overlay
# the partial OOF fills and cached imputations so completed units are skipped.

imp_fold_cache  <- vector("list", max(gt_pool$fold_id))
completed_units <- character(0)  # "pset__fold" pairs already finished

.ckpt_compatible <- function(ckpt, cfg) {
  isTRUE(identical(sort(ckpt$config$pset_spec),   sort(cfg$pset_spec))) &&
  isTRUE(identical(sort(ckpt$config$model_subset %||% character(0)),
                   sort(cfg$model_subset   %||% character(0))))
}

if (isTRUE(RUN_CFG$enable_checkpoint) && file.exists(RUN_CFG$checkpoint_path)) {
  ckpt <- tryCatch(readRDS(RUN_CFG$checkpoint_path), error = function(e) NULL)
  if (!is.null(ckpt) && .ckpt_compatible(ckpt, RUN_CFG)) {
    # Overlay OOF fills — only keys that already exist in our storage
    for (ck in names(ckpt$oof_storage)) {
      if (ck %in% names(oof_storage))
        oof_storage[[ck]] <- ckpt$oof_storage[[ck]]
    }
    # Restore cached MICE runs
    for (fi in seq_along(ckpt$imp_fold_cache)) {
      if (!is.null(ckpt$imp_fold_cache[[fi]]))
        imp_fold_cache[[fi]] <- ckpt$imp_fold_cache[[fi]]
    }
    completed_units <- ckpt$completed_units %||% character(0)
    n_total_units   <- length(active_psets) * length(fold_ids)
    message(sprintf(
      "[20] Checkpoint loaded: %d/%d CV units already done (saved %s). Resuming.",
      length(completed_units), n_total_units,
      format(ckpt$timestamp, "%Y-%m-%d %H:%M")))
    rm(ckpt)
  } else {
    if (!is.null(ckpt))
      message("[20] Checkpoint found but config differs — starting fresh.")
  }
}

# Progress counters for the CV loop
n_total_units  <- length(active_psets) * length(fold_ids)
n_done_units   <- length(completed_units)

# Feature-model loop
t0 <- Sys.time()
for (pset_name in active_psets) {
  predictors_pset <- intersect(
    intersect(PREDICTOR_SETS[[pset_name]], names(gt_pool)),
    names(test_base))
  if (length(predictors_pset) < 2L) {
    warning("[20] pset '", pset_name, "' has <2 usable predictors; skipping.")
    next
  }
  message(sprintf("[20] pset=%s | %d usable predictors", pset_name, length(predictors_pset)))

  fold_ids <- sort(unique(gt_pool$fold_id))
  for (k in fold_ids) {
    unit_key <- paste(pset_name, k, sep = "__")
    if (unit_key %in% completed_units) {
      message(sprintf("[20]   pset=%s fold=%d skipped (checkpoint)", pset_name, k))
      next
    }
    tr_idx <- which(gt_pool$fold_id != k)
    va_idx <- which(gt_pool$fold_id == k)
    tr_raw <- gt_pool[tr_idx, , drop = FALSE]
    va_raw <- gt_pool[va_idx, , drop = FALSE]

    pmm_active <- any(vapply(model_specs[active_model_names],
      function(s) s$imputation_strategy == "pmm", logical(1)))

    imp <- NULL
    if (pmm_active) {
      # Populate cache on first PMM-needing pset for this fold (shared pool).
      if (is.null(imp_fold_cache[[k]])) {
        imp_fold_cache[[k]] <- tryCatch(
          apply_imputation(tr_raw, va_raw, "_all_active",
                           m     = RUN_CFG$n_imputations,
                           maxit = RUN_CFG$mice_maxit),
          error = function(e) {
            warning("[20] shared imputation failed (fold ", k, "): ", e$message,
                    " | zero-fill fallback for all PMM models this fold.")
            NULL
          })
      }
      # Slice to current pset's variables.
      imp <- slice_imp_to_pset(imp_fold_cache[[k]], predictors_pset, RUN_CFG$outcome_col)
    }

    # Density/stability diagnostics: use the FULL shared imputation (not the
    # pset-sliced version), run only once per fold on the first active pset.
    if (RUN_CFG$run_imputation_diagnostics &&
        k == fold_ids[1L] &&
        pset_name == active_psets[1L] &&
        !is.null(imp_fold_cache[[k]]) &&
        isTRUE(imp_fold_cache[[k]]$is_imputed)) {
      diag_dir <- file.path(RUN_CFG$output_dir, "20_imputation_diag")
      dir.create(diag_dir, showWarnings = FALSE, recursive = TRUE)
      # Show all variables imputed in the shared run (all active psets combined).
      all_diag_preds <- intersect(PREDICTOR_SETS[["_all_active"]], names(tr_raw))
      tryCatch(write_imputation_diagnostics(
        tr_raw       = tr_raw,
        imputed_list = imp_fold_cache[[k]]$train,
        predictors   = all_diag_preds,
        out_pdf      = file.path(diag_dir, "imp_diag_fold1_shared.pdf"),
        top_n        = RUN_CFG$diag_n_features
      ), error = function(e) message("[20] diag PDF failed: ", e$message))
    }

    # Convergence trace: runs its own separate mice call, independent of imp.
    # Shared imputation -> only needs to run once per fold (first pset).
    if (RUN_CFG$run_convergence_diag &&
        k == fold_ids[1L] &&
        pset_name == active_psets[1L]) {
      diag_dir <- file.path(RUN_CFG$output_dir, "20_imputation_diag")
      dir.create(diag_dir, showWarnings = FALSE, recursive = TRUE)
      tryCatch(run_mice_convergence_check(
        tr_raw      = tr_raw,
        predictors  = PREDICTOR_SETS[["_all_active"]],
        outcome_col = RUN_CFG$outcome_col,
        maxit       = RUN_CFG$mice_maxit,
        out_pdf     = file.path(diag_dir, "mice_convergence_fold1.pdf"),
        seed        = RUN_CFG$seed
      ), error = function(e) message("[20] convergence check failed: ", e$message))
    }

    # screen_EQ and screen_Mono (per fold, independent of pset_name loop)
    # Only run on the first pset iteration to avoid duplicate filling.
    if (pset_name == active_psets[1L]) {
      sel_screen <- tryCatch(
        screen_features(tr_raw, RUN_CFG$outcome_col, "is_health_strat",
                        screen_cand, screen_test_cov, screen_smd_tt),
        error = function(e) { warning("screen_features failed: ", e$message); character(0) })
      if (length(sel_screen) < 3L) {
        sel_screen <- names(sort(abs(screen_test_cov), decreasing = TRUE))[1:3]
        message(sprintf("[20] screen fallback fold %d: using top-3 by test_cov", k))
      }

      if (use_screen_eq) {
        oof_storage[["screen_EQ"]][va_idx] <- tryCatch({
          m_eq <- fit_eq_screen(tr_raw, RUN_CFG$outcome_col, sel_screen)
          pred_eq_screen(m_eq, va_raw)
        }, error = function(e) {
          warning("[20] screen_EQ fold ", k, ": ", e$message); rep(NA_real_, length(va_idx))
        })
      }
      if (use_screen_mono) {
        oof_storage[["screen_Mono"]][va_idx] <- tryCatch({
          m_mo <- fit_mono_screen(tr_raw, RUN_CFG$outcome_col, sel_screen)
          pred_mono_screen(m_mo, va_raw)
        }, error = function(e) {
          warning("[20] screen_Mono fold ", k, ": ", e$message); rep(NA_real_, length(va_idx))
        })
      }
      # Prior_Shrinkage: per-fold logistic on the LLM prior + shrinkage evidence,
      # health rows only (where prior_3fam/p_base exist). Mirrors screen_* —
      # uses raw (un-imputed) tr_raw/va_raw; leak-free (refit each fold).
      if (use_prior_shrinkage) {
        oof_storage[["Prior_Shrinkage"]][va_idx] <- tryCatch({
          trh <- tr_raw[!is.na(tr_raw$prior_3fam) & !is.na(tr_raw$p_base), , drop = FALSE]
          out <- rep(NA_real_, length(va_idx))
          if (nrow(trh) >= 5L) {
            trh[[".ps_y"]] <- as.integer(trh[[RUN_CFG$outcome_col]])
            m_ps <- stats::glm(stats::reformulate(c("prior_3fam", "p_base"), ".ps_y"),
                               data = trh, family = stats::binomial())
            hv <- !is.na(va_raw$prior_3fam) & !is.na(va_raw$p_base)
            if (any(hv))
              out[hv] <- stats::predict(m_ps, va_raw[hv, , drop = FALSE], type = "response")
          }
          out
        }, error = function(e) {
          warning("[20] Prior_Shrinkage fold ", k, ": ", e$message); rep(NA_real_, length(va_idx))
        })
      }
    }

    # Feature model predictions
    for (mname in active_model_names) {
      spec      <- model_specs[[mname]]
      preds_safe <- filter_predictors_for_model(tr_raw, predictors_pset,
                                                mname, audit_log)
      if (length(preds_safe) < 2L) next

      key      <- paste("none", pset_name, mname, sep = ".")
      w_vec    <- rep(1, nrow(tr_raw))

      pred_va <- tryCatch({
        if (spec$imputation_strategy == "pmm" && !is.null(imp)) {
          # Safety: restrict predictors to columns actually present in the
          # imputed datasets. Variables dropped by the 70%-missing threshold
          # in apply_imputation won't be in imp$train — including them in
          # preds_safe would cause a "column not found" error in spec$fit.
          imp_cols   <- names(imp$train[[1]])
          preds_imp  <- intersect(preds_safe, imp_cols)
          preds_miss <- setdiff(preds_safe, imp_cols)
          if (length(preds_miss) > 0L)
            message(sprintf("[20] %s pset=%s fold=%d: %d predictor(s) absent from",
                            mname, pset_name, k, length(preds_miss)),
                    " imputed data (>70%% missing), dropped: ",
                    paste(preds_miss, collapse = ", "))
          if (length(preds_imp) < 2L) {
            warning(sprintf("[20] %s pset=%s fold=%d: <2 predictors after imp filter; skipping.",
                            mname, pset_name, k))
            rep(NA_real_, length(va_idx))
          } else {
            acc <- numeric(length(va_idx))
            for (mi in seq_len(RUN_CFG$n_imputations)) {
              tr_i <- imp$train[[mi]]; va_i <- imp$test[[mi]]
              fit_i <- spec$fit(tr_i, RUN_CFG$outcome_col, preds_imp, w_vec)
              acc   <- acc + spec$pred(fit_i, va_i, preds_imp)
            }
            acc / RUN_CFG$n_imputations
          }
        } else if (spec$imputation_strategy == "pmm" && is.null(imp)) {
          tr_fb <- zero_fill(tr_raw, preds_safe)
          va_fb <- zero_fill(va_raw, preds_safe)
          fit_fb <- spec$fit(tr_fb, RUN_CFG$outcome_col, preds_safe, w_vec)
          spec$pred(fit_fb, va_fb, preds_safe)
        } else {
          na_policy <- MODEL_NA_HANDLING[[mname]] %||% "zero_fill"
          if (na_policy == "native") {
            prepared  <- prepare_for_native_na(tr_raw, va_raw, preds_safe)
            preds_fit <- filter_predictors_for_model(prepared$tr,
                           prepared$predictors, mname, audit_log = NULL)
            fit_n <- spec$fit(prepared$tr, RUN_CFG$outcome_col, preds_fit, w_vec)
            spec$pred(fit_n, prepared$va, preds_fit)
          } else {
            tr_z <- zero_fill(tr_raw, preds_safe)
            va_z <- zero_fill(va_raw, preds_safe)
            fit_z <- spec$fit(tr_z, RUN_CFG$outcome_col, preds_safe, w_vec)
            spec$pred(fit_z, va_z, preds_safe)
          }
        }
      }, error = function(e) {
        warning("[20] ", mname, " pset=", pset_name, " fold=", k, ": ", e$message)
        rep(NA_real_, length(va_idx))
      })
      oof_storage[[key]][va_idx] <- pred_va
    }

    n_done_units <- n_done_units + 1L
    elapsed_min  <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    eta_min      <- if (n_done_units > 0)
      elapsed_min / n_done_units * (n_total_units - n_done_units) else NA_real_
    message(sprintf(
      "[20] %d/%d units done | pset=%s fold=%d/%d | %.1f min elapsed | ETA %.0f min",
      n_done_units, n_total_units,
      pset_name, k, max(fold_ids),
      elapsed_min,
      if (is.finite(eta_min)) eta_min else NA_real_))

    # Checkpoint save: persist progress so a restart can resume here.
    if (isTRUE(RUN_CFG$enable_checkpoint)) {
      completed_units <- c(completed_units, unit_key)
      tryCatch({
        saveRDS(list(
          oof_storage     = oof_storage,
          imp_fold_cache  = imp_fold_cache,
          completed_units = completed_units,
          config          = RUN_CFG,
          timestamp       = Sys.time()
        ), RUN_CFG$checkpoint_path)
      }, error = function(e)
        warning("[20] checkpoint save failed: ", e$message))
    }
  }
}

# =============================================================================
# 9. NESTED-CV TEMPERATURE CALIBRATION
# =============================================================================

calibrated_storage <- list()
T_rows <- list()  # collected per-key; bound once after the loop

for (key in names(oof_storage)) {
  oof_vec <- oof_storage[[key]]
  if (all(is.na(oof_vec))) { calibrated_storage[[key]] <- oof_vec; next }
  cal <- nested_cv_temperature(
    oof       = oof_vec,
    fold_id   = gt_pool$fold_id,
    is_health = gt_pool$is_health_strat == 1,
    outcome   = as.numeric(gt_pool[[RUN_CFG$outcome_col]]))
  calibrated_storage[[key]] <- cal$calibrated
  T_rows[[key]] <- tibble::tibble(
    key  = key,
    fold = names(cal$T_per_fold),
    T    = as.numeric(cal$T_per_fold))
}
T_table <- dplyr::bind_rows(T_rows)
readr::write_csv(T_table, file.path(RUN_CFG$output_dir, "20_t_calibration.csv"))
T_summary <- T_table |>
  dplyr::group_by(key) |>
  dplyr::summarise(T_mean = mean(T, na.rm = TRUE), T_sd = sd(T, na.rm = TRUE),
                   n_folds_calibrated = sum(!is.na(T)), .groups = "drop") |>
  dplyr::mutate(T_cv = T_sd / T_mean)
message("[20] T summary (most unstable):")
print(T_summary |> dplyr::arrange(dplyr::desc(T_cv)) |> head(10))

# =============================================================================
# 10. BRIER SUMMARIES
# =============================================================================

y          <- as.numeric(gt_pool[[RUN_CFG$outcome_col]])
is_h_strat <- gt_pool$is_health_strat == 1
is_r12     <- gt_pool$dataset %in% c("round1", "round2")
is_chal_h  <- is_r12 & is_h_strat

brier_subset <- function(pred, mask) {
  ok <- !is.na(pred) & !is.na(y) & mask
  if (sum(ok) < 3L) return(NA_real_)
  mean((pred[ok] - y[ok])^2)
}

summary_rows <- lapply(names(oof_storage), function(key) {
  p_raw <- oof_storage[[key]]
  p_cal <- calibrated_storage[[key]]
  if (is.null(p_cal)) p_cal <- rep(NA_real_, length(p_raw))
  tibble::tibble(
    key                   = key,
    n_oof_filled          = sum(!is.na(p_raw)),
    brier_all_raw         = brier_subset(p_raw, rep(TRUE, length(p_raw))),
    brier_r12_raw         = brier_subset(p_raw, is_r12),
    brier_health_all_raw  = brier_subset(p_raw, is_h_strat),
    brier_chal_health_raw = brier_subset(p_raw, is_chal_h),
    brier_all_cal         = brier_subset(p_cal, rep(TRUE, length(p_cal))),
    brier_r12_cal         = brier_subset(p_cal, is_r12),
    brier_health_all_cal  = brier_subset(p_cal, is_h_strat),
    brier_chal_health_cal = brier_subset(p_cal, is_chal_h))
})

summary_df <- dplyr::bind_rows(summary_rows) |>
  dplyr::left_join(key_meta, by = "key") |>
  dplyr::arrange(brier_chal_health_cal)

readr::write_csv(summary_df, file.path(RUN_CFG$output_dir, "20_oof_summary.csv"))
cat("\nTop 15 keys by brier_chal_health_cal:\n")
print(summary_df |>
  dplyr::select(key, pset, model, brier_chal_health_raw, brier_chal_health_cal) |>
  head(15))

# =============================================================================
# 11. POST-CV ENSEMBLES
# =============================================================================

message("[20] ── Post-CV Ensemble Computation ──────────────────────────────")

# 11a: 18-style equal-weight ensemble (enriched RF+GAM+ElNet+XGB)
keys_18 <- c("none.enriched.RF", "none.enriched.GAM",
             "none.enriched.ElNet", "none.enriched.XGB")
keys_18_present <- intersect(keys_18, names(oof_storage))
if (length(keys_18_present) >= 2L) {
  ens18_mat <- do.call(cbind, lapply(keys_18_present, function(k) oof_storage[[k]]))
  oof_storage[["ens_18_equal"]] <- rowMeans(ens18_mat, na.rm = TRUE)
  calibrated_storage[["ens_18_equal"]] <- {
    cal18 <- nested_cv_temperature(oof_storage[["ens_18_equal"]], gt_pool$fold_id,
                                   gt_pool$is_health_strat == 1, y)
    T_rows[["ens_18_equal"]] <- tibble::tibble(
      key = "ens_18_equal", fold = names(cal18$T_per_fold),
      T = as.numeric(cal18$T_per_fold))
    T_table <- dplyr::bind_rows(T_rows)
    cal18$calibrated
  }
  key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
    key = "ens_18_equal", weighting = NA_character_, pset = "enriched",
    model = paste0("EQ(", paste(sub("none\\.enriched\\.", "", keys_18_present), collapse="+"), ")")))
  message(sprintf("[20] ens_18_equal: %d base models", length(keys_18_present)))
} else {
  message("[20] ens_18_equal: enriched models not available (need enriched in pset_spec)")
}

# 11b: Per-fold NNLS stack over all non-LLM, non-screen feature models
nnls_base_keys <- key_meta |>
  dplyr::filter(!model %in% c("LLM_raw", "LLM_bincorr", "screen_EQ",
                               "screen_Mono", "ens_18_equal"),
                !is.na(pset)) |>
  dplyr::pull(key)
if (length(nnls_base_keys) >= 2L) {
  nnls_result <- compute_nnls_oof(oof_storage, nnls_base_keys, gt_pool,
                                   RUN_CFG$outcome_col)
  oof_storage[["ens_nnls"]] <- nnls_result$oof
  calibrated_storage[["ens_nnls"]] <- {
    caln <- nested_cv_temperature(oof_storage[["ens_nnls"]], gt_pool$fold_id,
                                   gt_pool$is_health_strat == 1, y)
    T_rows[["ens_nnls"]] <- tibble::tibble(
      key = "ens_nnls", fold = names(caln$T_per_fold),
      T = as.numeric(caln$T_per_fold))
    T_table <- dplyr::bind_rows(T_rows)
    caln$calibrated
  }
  key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
    key = "ens_nnls", weighting = NA_character_, pset = "stacked",
    model = sprintf("NNLS(%d)", length(nnls_base_keys))))
  message(sprintf("[20] ens_nnls: %d base models", length(nnls_base_keys)))
}

# 11c: Greedy uncorrelated ensemble (user's approach)
# Excludes LLM and stacked/ensemble keys — purely single-model candidates.
feature_keys_all <- key_meta |>
  dplyr::filter(!model %in% c("LLM_raw", "LLM_bincorr", "screen_EQ", "screen_Mono"),
                !key %in% c("ens_18_equal", "ens_nnls"),
                !is.na(pset)) |>
  dplyr::pull(key)

# Update summary_df to include ensemble keys before greedy computation
summary_df_tmp <- dplyr::bind_rows(
  summary_df,
  dplyr::bind_rows(lapply(
    intersect(c("ens_18_equal", "ens_nnls"), names(oof_storage)),
    function(key) {
      p_raw <- oof_storage[[key]]; p_cal <- calibrated_storage[[key]]
      if (is.null(p_cal)) p_cal <- p_raw
      tibble::tibble(
        key                   = key,
        brier_chal_health_raw = brier_subset(p_raw, is_chal_h),
        brier_chal_health_cal = brier_subset(p_cal, is_chal_h))
    }))
)

greedy_result <- compute_greedy_oof(
  oof_storage, feature_keys_all, summary_df_tmp, gt_pool,
  cor_threshold = RUN_CFG$greedy_cor_threshold,
  top_n_start   = RUN_CFG$greedy_top_n_start)

oof_storage[["ens_greedy_equal"]] <- greedy_result$oof_equal
calibrated_storage[["ens_greedy_equal"]] <- {
  calg <- nested_cv_temperature(oof_storage[["ens_greedy_equal"]], gt_pool$fold_id,
                                 gt_pool$is_health_strat == 1, y)
  T_rows[["ens_greedy_equal"]] <- tibble::tibble(
    key = "ens_greedy_equal", fold = names(calg$T_per_fold),
    T = as.numeric(calg$T_per_fold))
  T_table <- dplyr::bind_rows(T_rows)
  calg$calibrated
}
key_meta <- dplyr::bind_rows(key_meta, tibble::tibble(
  key = "ens_greedy_equal", weighting = NA_character_, pset = "greedy",
  model = sprintf("greedy_EQ(%d, cor<%.1f)",
                  length(greedy_result$selected), RUN_CFG$greedy_cor_threshold)))

# Rebuild full summary_df with ensembles included
summary_rows_ens <- lapply(c("ens_18_equal", "ens_nnls", "ens_greedy_equal"), function(key) {
  if (!key %in% names(oof_storage)) return(NULL)
  p_raw <- oof_storage[[key]]; p_cal <- calibrated_storage[[key]]
  if (is.null(p_cal)) p_cal <- rep(NA_real_, length(p_raw))
  tibble::tibble(
    key                   = key,
    n_oof_filled          = sum(!is.na(p_raw)),
    brier_all_raw         = brier_subset(p_raw, rep(TRUE, length(p_raw))),
    brier_r12_raw         = brier_subset(p_raw, is_r12),
    brier_health_all_raw  = brier_subset(p_raw, is_h_strat),
    brier_chal_health_raw = brier_subset(p_raw, is_chal_h),
    brier_all_cal         = brier_subset(p_cal, rep(TRUE, length(p_cal))),
    brier_r12_cal         = brier_subset(p_cal, is_r12),
    brier_health_all_cal  = brier_subset(p_cal, is_h_strat),
    brier_chal_health_cal = brier_subset(p_cal, is_chal_h))
})
# Attach meta columns (weighting/pset/model) to ensemble summary rows,
# then append to existing summary_df which already has those columns.
ens_keys_added <- c("ens_18_equal", "ens_nnls", "ens_greedy_equal")
ens_summary_with_meta <- dplyr::bind_rows(summary_rows_ens) |>
  dplyr::left_join(
    key_meta |> dplyr::filter(key %in% ens_keys_added),
    by = "key")
summary_df <- dplyr::bind_rows(summary_df, ens_summary_with_meta) |>
  dplyr::arrange(brier_chal_health_cal)

readr::write_csv(T_table, file.path(RUN_CFG$output_dir, "20_t_calibration.csv"))
readr::write_csv(summary_df, file.path(RUN_CFG$output_dir, "20_oof_summary.csv"))
cat("\nTop 20 keys including ensembles:\n")
print(summary_df |>
  dplyr::select(key, pset, model, brier_chal_health_raw, brier_chal_health_cal) |>
  head(25))
cat("\nLLM benchmarks (for reference regardless of rank):\n")
print(summary_df |>
  dplyr::filter(model %in% c("LLM_raw", "LLM_bincorr")) |>
  dplyr::select(key, model, brier_chal_health_raw, brier_chal_health_cal))

# =============================================================================
# 12. DEPLOY-FAITHFUL HEALTH HOLDOUT (top N candidates only)
# =============================================================================
# Each candidate is refitted on:
#   training = all non-health rows + health rows NOT in the held-out fold
#   holdout  = ~20 health rows
# Prediction is pooled across all health folds -> Brier on all 102 health rows.
# Only the top N candidates (by brier_chal_health_cal) are evaluated.
# No cross-product: each key maps to exactly one (model, pset) combination.

message("[20] ── Deploy-Faithful Health Holdout ──────────────────────────────")

hth_pool  <- gt_pool |> dplyr::filter(is_health_strat == 1)
nonh_pool <- gt_pool |> dplyr::filter(is_health_strat == 0)
y_health  <- as.numeric(hth_pool[[RUN_CFG$outcome_col]])
is_forrH  <- hth_pool$dataset == "training"
is_chalH  <- hth_pool$dataset %in% c("round1", "round2")

top_cand_keys <- summary_df |>
  dplyr::slice_min(brier_chal_health_cal,
                   n = RUN_CFG$deploy_faithful_top_n, with_ties = FALSE) |>
  dplyr::pull(key)
top_cand_keys <- unique(c(top_cand_keys,
                           intersect(c("LLM_raw", "LLM_bincorr"), names(oof_storage))))
message(sprintf("[20] deploy-faithful candidates (%d): %s",
                length(top_cand_keys), paste(top_cand_keys, collapse = ", ")))

hth_pool$hfold_id <- make_stratified_group_folds(
  df         = hth_pool,
  group_var  = RUN_CFG$group_var,
  strat_vars = "dataset",
  n_folds    = RUN_CFG$deploy_faithful_n_health_folds,
  seed       = RUN_CFG$seed)
hfold_ids <- sort(unique(hth_pool$hfold_id))

deploy_oos <- stats::setNames(
  lapply(top_cand_keys, function(k) rep(NA_real_, nrow(hth_pool))),
  top_cand_keys)

for (ckey in top_cand_keys) {
  kmeta <- key_meta |> dplyr::filter(key == ckey)
  mname <- if (nrow(kmeta) > 0L) kmeta$model[1] else ckey
  pname <- if (nrow(kmeta) > 0L) kmeta$pset[1]  else NA_character_

  # LLM: no refitting, reuse main OOF health rows
  if (mname %in% c("LLM_raw", "LLM_bincorr")) {
    h_idx_main <- which(gt_pool$is_health_strat == 1)
    deploy_oos[[ckey]] <- oof_storage[[ckey]][h_idx_main]
    message(sprintf("[20] deploy-faithful %s: using main OOF (no refit needed)", ckey))
    next
  }

  # ens_nnls: use main CV OOF (refitting all components is too expensive)
  if (ckey == "ens_nnls") {
    h_idx_main <- which(gt_pool$is_health_strat == 1)
    deploy_oos[[ckey]] <- oof_storage[[ckey]][h_idx_main]
    message("[20] deploy-faithful ens_nnls: approximated from main CV OOF")
    next
  }

  preds_candidate <- if (!is.na(pname) && !pname %in% c("screen", "stacked", "greedy"))
    intersect(intersect(PREDICTOR_SETS[[pname]], names(gt_pool)), names(test_base))
  else NULL

  spec_candidate <- if (!is.null(mname) && mname %in% names(model_specs))
    model_specs[[mname]] else NULL

  for (hf in hfold_ids) {
    va_h_idx <- which(hth_pool$hfold_id == hf)
    tr_h_idx <- which(hth_pool$hfold_id != hf)
    va_h <- hth_pool[va_h_idx, , drop = FALSE]
    trh  <- dplyr::bind_rows(nonh_pool, hth_pool[tr_h_idx, , drop = FALSE])
    trh$w_ipw <- 1.0

    tryCatch({
      pred_va <- if (ckey == "screen_EQ") {
        sel <- screen_features(trh, RUN_CFG$outcome_col, "is_health_strat",
                               screen_cand, screen_test_cov, screen_smd_tt)
        if (length(sel) < 3L) sel <- screen_cand[1:3]
        pred_eq_screen(fit_eq_screen(trh, RUN_CFG$outcome_col, sel), va_h)

      } else if (ckey == "screen_Mono") {
        sel <- screen_features(trh, RUN_CFG$outcome_col, "is_health_strat",
                               screen_cand, screen_test_cov, screen_smd_tt)
        if (length(sel) < 3L) sel <- screen_cand[1:3]
        pred_mono_screen(fit_mono_screen(trh, RUN_CFG$outcome_col, sel), va_h)

      } else if (ckey == "ens_18_equal") {
        p18 <- lapply(keys_18_present, function(k18) {
          mk18 <- key_meta |> dplyr::filter(key == k18)
          if (nrow(mk18) == 0L) return(rep(NA_real_, nrow(va_h)))
          sp18 <- model_specs[[mk18$model[1]]]
          if (is.null(sp18)) return(rep(NA_real_, nrow(va_h)))
          pr18 <- intersect(PREDICTOR_SETS[["enriched"]], names(trh))
          pr18 <- intersect(pr18, names(test_base))
          if (sp18$imputation_strategy == "pmm") {
            imp18 <- tryCatch(apply_imputation(trh, va_h, "enriched",
                                               m = RUN_CFG$deploy_faithful_n_imp,
                                               maxit = RUN_CFG$mice_maxit),
                              error = function(e) NULL)
            if (!is.null(imp18)) {
              f18 <- sp18$fit(imp18$train[[1L]], RUN_CFG$outcome_col, pr18, NULL)
              sp18$pred(f18, imp18$test[[1L]], pr18)
            } else {
              f18 <- sp18$fit(zero_fill(trh, pr18), RUN_CFG$outcome_col, pr18, NULL)
              sp18$pred(f18, zero_fill(va_h, pr18), pr18)
            }
          } else {
            na_pol <- MODEL_NA_HANDLING[[mk18$model[1]]] %||% "zero_fill"
            if (na_pol == "native") {
              pr18p <- prepare_for_native_na(trh, va_h, pr18)
              pr18f <- filter_predictors_for_model(pr18p$tr, pr18p$predictors, mk18$model[1])
              f18   <- sp18$fit(pr18p$tr, RUN_CFG$outcome_col, pr18f, NULL)
              sp18$pred(f18, pr18p$va, pr18f)
            } else {
              f18 <- sp18$fit(zero_fill(trh, pr18), RUN_CFG$outcome_col, pr18, NULL)
              sp18$pred(f18, zero_fill(va_h, pr18), pr18)
            }
          }
        })
        rowMeans(do.call(cbind, p18), na.rm = TRUE)

      } else if (ckey == "ens_greedy_equal") {
        pg <- lapply(greedy_result$selected, function(sk) {
          smk <- key_meta |> dplyr::filter(key == sk)
          if (nrow(smk) == 0L) return(rep(NA_real_, nrow(va_h)))
          spk <- model_specs[[smk$model[1]]]
          if (is.null(spk)) return(rep(NA_real_, nrow(va_h)))
          prk <- intersect(PREDICTOR_SETS[[smk$pset[1]]], names(trh))
          prk <- intersect(prk, names(test_base))
          if (spk$imputation_strategy == "pmm") {
            impk <- tryCatch(apply_imputation(trh, va_h, smk$pset[1],
                                              m = RUN_CFG$deploy_faithful_n_imp,
                                              maxit = RUN_CFG$mice_maxit),
                             error = function(e) NULL)
            if (!is.null(impk)) {
              fk <- spk$fit(impk$train[[1L]], RUN_CFG$outcome_col, prk, NULL)
              spk$pred(fk, impk$test[[1L]], prk)
            } else {
              fk <- spk$fit(zero_fill(trh, prk), RUN_CFG$outcome_col, prk, NULL)
              spk$pred(fk, zero_fill(va_h, prk), prk)
            }
          } else {
            na_pol <- MODEL_NA_HANDLING[[smk$model[1]]] %||% "zero_fill"
            if (na_pol == "native") {
              prkp <- prepare_for_native_na(trh, va_h, prk)
              prkf <- filter_predictors_for_model(prkp$tr, prkp$predictors, smk$model[1])
              fk   <- spk$fit(prkp$tr, RUN_CFG$outcome_col, prkf, NULL)
              spk$pred(fk, prkp$va, prkf)
            } else {
              fk <- spk$fit(zero_fill(trh, prk), RUN_CFG$outcome_col, prk, NULL)
              spk$pred(fk, zero_fill(va_h, prk), prk)
            }
          }
        })
        rowMeans(do.call(cbind, pg), na.rm = TRUE)

      } else if (!is.null(spec_candidate) && !is.null(preds_candidate)) {
        # Single model candidate
        if (spec_candidate$imputation_strategy == "pmm") {
          imp_dh <- tryCatch(apply_imputation(trh, va_h, pname,
                                              m = RUN_CFG$deploy_faithful_n_imp,
                                              maxit = RUN_CFG$mice_maxit),
                             error = function(e) NULL)
          if (!is.null(imp_dh)) {
            acc <- numeric(nrow(va_h))
            for (mi in seq_len(RUN_CFG$deploy_faithful_n_imp)) {
              fi  <- spec_candidate$fit(imp_dh$train[[mi]], RUN_CFG$outcome_col,
                                        preds_candidate, NULL)
              acc <- acc + spec_candidate$pred(fi, imp_dh$test[[mi]], preds_candidate)
            }
            acc / RUN_CFG$deploy_faithful_n_imp
          } else {
            fi <- spec_candidate$fit(zero_fill(trh, preds_candidate),
                                     RUN_CFG$outcome_col, preds_candidate, NULL)
            spec_candidate$pred(fi, zero_fill(va_h, preds_candidate), preds_candidate)
          }
        } else {
          na_pol <- MODEL_NA_HANDLING[[mname]] %||% "zero_fill"
          if (na_pol == "native") {
            pr <- prepare_for_native_na(trh, va_h, preds_candidate)
            pf <- filter_predictors_for_model(pr$tr, pr$predictors, mname)
            fi <- spec_candidate$fit(pr$tr, RUN_CFG$outcome_col, pf, NULL)
            spec_candidate$pred(fi, pr$va, pf)
          } else {
            fi <- spec_candidate$fit(zero_fill(trh, preds_candidate),
                                     RUN_CFG$outcome_col, preds_candidate, NULL)
            spec_candidate$pred(fi, zero_fill(va_h, preds_candidate), preds_candidate)
          }
        }
      } else {
        warning("[20] deploy-faithful: no handler for key ", ckey, " hfold=", hf)
        rep(NA_real_, nrow(va_h))
      }

      deploy_oos[[ckey]][va_h_idx] <- pred_va
    }, error = function(e) {
      warning("[20] deploy-faithful failed: key=", ckey, " hfold=", hf, ": ", e$message)
    })
  }

  b_all  <- brier_score(deploy_oos[[ckey]], y_health)
  b_chal <- brier_score(deploy_oos[[ckey]][is_chalH], y_health[is_chalH])
  message(sprintf("[20] deploy-faithful %s: Brier_all=%.4f Brier_chal=%.4f",
                  ckey, b_all, b_chal))
}

# ── Eval B: leave-ALL-health-out (psych-only train -> all health) ─────────────
# Mirrors Eval B: single fit per candidate on non-health rows only,
# predict on all health rows. LLMs and ens_nnls skipped (no refit possible).
message("[20] ── Eval B: Leave-All-Health-Out ─────────────────────────────────")

tr_evalb       <- nonh_pool
tr_evalb$w_ipw <- 1.0
va_evalb       <- hth_pool

eval_b_preds <- stats::setNames(
  lapply(top_cand_keys, function(k) rep(NA_real_, nrow(hth_pool))),
  top_cand_keys)

for (ckey in top_cand_keys) {
  kmeta <- key_meta |> dplyr::filter(key == ckey)
  mname <- if (nrow(kmeta) > 0L) kmeta$model[1] else ckey
  pname <- if (nrow(kmeta) > 0L) kmeta$pset[1]  else NA_character_

  if (mname %in% c("LLM_raw", "LLM_bincorr") || ckey == "ens_nnls") {
    message(sprintf("[20] eval-B %s: skipped (no refit)", ckey))
    next
  }

  preds_candidate <- if (!is.na(pname) && !pname %in% c("screen", "stacked", "greedy"))
    intersect(intersect(PREDICTOR_SETS[[pname]], names(tr_evalb)), names(test_base))
  else NULL

  spec_candidate <- if (!is.null(mname) && mname %in% names(model_specs))
    model_specs[[mname]] else NULL

  tryCatch({
    pred_va <- if (ckey == "screen_EQ") {
      sel <- screen_features(tr_evalb, RUN_CFG$outcome_col, "is_health_strat",
                             screen_cand, screen_test_cov, screen_smd_tt)
      if (length(sel) < 3L) sel <- screen_cand[1:3]
      pred_eq_screen(fit_eq_screen(tr_evalb, RUN_CFG$outcome_col, sel), va_evalb)

    } else if (ckey == "screen_Mono") {
      sel <- screen_features(tr_evalb, RUN_CFG$outcome_col, "is_health_strat",
                             screen_cand, screen_test_cov, screen_smd_tt)
      if (length(sel) < 3L) sel <- screen_cand[1:3]
      pred_mono_screen(fit_mono_screen(tr_evalb, RUN_CFG$outcome_col, sel), va_evalb)

    } else if (ckey == "ens_18_equal") {
      p18 <- lapply(keys_18_present, function(k18) {
        mk18 <- key_meta |> dplyr::filter(key == k18)
        if (nrow(mk18) == 0L) return(rep(NA_real_, nrow(va_evalb)))
        sp18 <- model_specs[[mk18$model[1]]]
        if (is.null(sp18)) return(rep(NA_real_, nrow(va_evalb)))
        pr18 <- intersect(PREDICTOR_SETS[["enriched"]], names(tr_evalb))
        pr18 <- intersect(pr18, names(test_base))
        if (sp18$imputation_strategy == "pmm") {
          imp18 <- tryCatch(apply_imputation(tr_evalb, va_evalb, "enriched",
                                             m = RUN_CFG$deploy_faithful_n_imp,
                                             maxit = RUN_CFG$mice_maxit),
                            error = function(e) NULL)
          if (!is.null(imp18)) {
            f18 <- sp18$fit(imp18$train[[1L]], RUN_CFG$outcome_col, pr18, NULL)
            sp18$pred(f18, imp18$test[[1L]], pr18)
          } else {
            f18 <- sp18$fit(zero_fill(tr_evalb, pr18), RUN_CFG$outcome_col, pr18, NULL)
            sp18$pred(f18, zero_fill(va_evalb, pr18), pr18)
          }
        } else {
          na_pol <- MODEL_NA_HANDLING[[mk18$model[1]]] %||% "zero_fill"
          if (na_pol == "native") {
            pr18p <- prepare_for_native_na(tr_evalb, va_evalb, pr18)
            pr18f <- filter_predictors_for_model(pr18p$tr, pr18p$predictors, mk18$model[1])
            f18   <- sp18$fit(pr18p$tr, RUN_CFG$outcome_col, pr18f, NULL)
            sp18$pred(f18, pr18p$va, pr18f)
          } else {
            f18 <- sp18$fit(zero_fill(tr_evalb, pr18), RUN_CFG$outcome_col, pr18, NULL)
            sp18$pred(f18, zero_fill(va_evalb, pr18), pr18)
          }
        }
      })
      rowMeans(do.call(cbind, p18), na.rm = TRUE)

    } else if (ckey == "ens_greedy_equal") {
      pg <- lapply(greedy_result$selected, function(sk) {
        smk <- key_meta |> dplyr::filter(key == sk)
        if (nrow(smk) == 0L) return(rep(NA_real_, nrow(va_evalb)))
        spk <- model_specs[[smk$model[1]]]
        if (is.null(spk)) return(rep(NA_real_, nrow(va_evalb)))
        prk <- intersect(PREDICTOR_SETS[[smk$pset[1]]], names(tr_evalb))
        prk <- intersect(prk, names(test_base))
        if (spk$imputation_strategy == "pmm") {
          impk <- tryCatch(apply_imputation(tr_evalb, va_evalb, smk$pset[1],
                                            m = RUN_CFG$deploy_faithful_n_imp,
                                            maxit = RUN_CFG$mice_maxit),
                           error = function(e) NULL)
          if (!is.null(impk)) {
            fk <- spk$fit(impk$train[[1L]], RUN_CFG$outcome_col, prk, NULL)
            spk$pred(fk, impk$test[[1L]], prk)
          } else {
            fk <- spk$fit(zero_fill(tr_evalb, prk), RUN_CFG$outcome_col, prk, NULL)
            spk$pred(fk, zero_fill(va_evalb, prk), prk)
          }
        } else {
          na_pol <- MODEL_NA_HANDLING[[smk$model[1]]] %||% "zero_fill"
          if (na_pol == "native") {
            prkp <- prepare_for_native_na(tr_evalb, va_evalb, prk)
            prkf <- filter_predictors_for_model(prkp$tr, prkp$predictors, smk$model[1])
            fk   <- spk$fit(prkp$tr, RUN_CFG$outcome_col, prkf, NULL)
            spk$pred(fk, prkp$va, prkf)
          } else {
            fk <- spk$fit(zero_fill(tr_evalb, prk), RUN_CFG$outcome_col, prk, NULL)
            spk$pred(fk, zero_fill(va_evalb, prk), prk)
          }
        }
      })
      rowMeans(do.call(cbind, pg), na.rm = TRUE)

    } else if (!is.null(spec_candidate) && !is.null(preds_candidate)) {
      if (spec_candidate$imputation_strategy == "pmm") {
        imp_dh <- tryCatch(apply_imputation(tr_evalb, va_evalb, pname,
                                            m = RUN_CFG$deploy_faithful_n_imp,
                                            maxit = RUN_CFG$mice_maxit),
                           error = function(e) NULL)
        if (!is.null(imp_dh)) {
          acc <- numeric(nrow(va_evalb))
          for (mi in seq_len(RUN_CFG$deploy_faithful_n_imp)) {
            fi  <- spec_candidate$fit(imp_dh$train[[mi]], RUN_CFG$outcome_col,
                                      preds_candidate, NULL)
            acc <- acc + spec_candidate$pred(fi, imp_dh$test[[mi]], preds_candidate)
          }
          acc / RUN_CFG$deploy_faithful_n_imp
        } else {
          fi <- spec_candidate$fit(zero_fill(tr_evalb, preds_candidate),
                                   RUN_CFG$outcome_col, preds_candidate, NULL)
          spec_candidate$pred(fi, zero_fill(va_evalb, preds_candidate), preds_candidate)
        }
      } else {
        na_pol <- MODEL_NA_HANDLING[[mname]] %||% "zero_fill"
        if (na_pol == "native") {
          pr <- prepare_for_native_na(tr_evalb, va_evalb, preds_candidate)
          pf <- filter_predictors_for_model(pr$tr, pr$predictors, mname)
          fi <- spec_candidate$fit(pr$tr, RUN_CFG$outcome_col, pf, NULL)
          spec_candidate$pred(fi, pr$va, pf)
        } else {
          fi <- spec_candidate$fit(zero_fill(tr_evalb, preds_candidate),
                                   RUN_CFG$outcome_col, preds_candidate, NULL)
          spec_candidate$pred(fi, zero_fill(va_evalb, preds_candidate), preds_candidate)
        }
      }
    } else {
      warning("[20] eval-B: no handler for key ", ckey)
      rep(NA_real_, nrow(va_evalb))
    }

    eval_b_preds[[ckey]] <- pred_va
    b_chal_b <- brier_score(pred_va[is_chalH], y_health[is_chalH])
    message(sprintf("[20] eval-B %s: Brier_chal=%.4f", ckey, b_chal_b))
  }, error = function(e) {
    warning("[20] eval-B failed: key=", ckey, ": ", e$message)
  })
}

# Deploy-faithful results table with bootstrap CIs
base_brier_h <- mean(y_health) * (1 - mean(y_health))
# Base rate computed on challenge-health only (the R3 analog), not all health.
# Mixing in the easy FORRT rows flatters the gate
y_chal_all   <- y_health[is_chalH]
base_brier_chal <- mean(y_chal_all) * (1 - mean(y_chal_all))
deploy_results <- dplyr::bind_rows(lapply(top_cand_keys, function(ckey) {
  ph <- deploy_oos[[ckey]]
  n_h <- sum(!is.na(ph))
  # Actual row count behind brier_chal_health (the R3-relevant column),
  # not the all-health count in n_health.
  n_chal = sum(!is.na(ph[is_chalH]))
  if (n_h < 3L) return(NULL)
  # Bootstrap CI on challenge-health rows to match the brier_chal_health sort column
  ph_chal <- ph[is_chalH]
  y_chal  <- y_health[is_chalH]
  bs <- replicate(4000L, {
    i <- sample(length(y_chal), length(y_chal), TRUE)
    brier_score(ph_chal[i], y_chal[i])
  })
  ci <- quantile(bs, c(.10, .90), na.rm = TRUE)
  eb <- eval_b_preds[[ckey]]
  tibble::tibble(
    key                = ckey,
    n_health_all       = n_h,
    n_chal             = n_chal,
    brier_health_all   = brier_score(ph, y_health),
    brier_forrt_health = brier_score(ph[is_forrH], y_health[is_forrH]),
    brier_chal_health  = brier_score(ph[is_chalH], y_health[is_chalH]),
    ci_lo_80           = unname(ci[1]),
    ci_hi_80           = unname(ci[2]),
    eval_b_brier_chal  = if (!is.null(eb) && sum(!is.na(eb[is_chalH])) >= 3L)
                           brier_score(eb[is_chalH], y_health[is_chalH])
                         else NA_real_,
    base_brier_all     = base_brier_h,
    base_brier_chal    = base_brier_chal,
    beats_base_all     = brier_score(ph, y_health) < base_brier_h,
    beats_base_chal    = brier_score(ph[is_chalH], y_health[is_chalH]) < base_brier_chal)
}))

readr::write_csv(deploy_results,
  file.path(RUN_CFG$output_dir, "20_deploy_faithful_results.csv"))
cat("\n── Deploy-Faithful Health Results (sorted by brier_chal_health) ──────\n")
print(deploy_results |> dplyr::arrange(brier_chal_health))

# =============================================================================
# 13. PREDICTOR AUDIT + SAVE oof_grid.rds
# =============================================================================

audit_df <- audit_log$as_tibble()
readr::write_csv(audit_df, file.path(RUN_CFG$output_dir, "20_predictor_audit.csv"))
if (nrow(audit_df) > 0L)
  message(sprintf("[20] predictor audit: %d (model, predictor) pairs dropped.",
                  nrow(audit_df)))

meta_cols <- intersect(
  c("claim_id", "doi_o", "dataset", "is_health", "is_health_strat",
    "fold_id", "w_ipw", RUN_CFG$outcome_col, "n_o"),
  names(gt_pool))

oof_grid <- list(
  meta              = gt_pool |> dplyr::select(dplyr::all_of(meta_cols)),
  oof_raw           = oof_storage,
  oof_calibrated    = calibrated_storage,
  key_meta          = key_meta,
  T_per_fold        = T_table,
  greedy_selected   = greedy_result$selected,
  deploy_results    = deploy_results,
  config            = RUN_CFG,
  predictor_audit   = audit_df,
  feature_stability = stab,
  fold_composition  = fold_comp
)

out_path <- file.path(RUN_CFG$cache_dir, "oof_grid.rds")
saveRDS(oof_grid, out_path, compress = "xz")
message("[20] wrote ", out_path)

elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat(sprintf("\n[07_health_validation.R] done in %.1f min.\n", elapsed))

# Remove checkpoint file on clean completion — no longer needed.
if (isTRUE(RUN_CFG$enable_checkpoint) && file.exists(RUN_CFG$checkpoint_path)) {
  file.remove(RUN_CFG$checkpoint_path)
  message("[20] checkpoint file removed (run complete).")
}

tryCatch(beepr::beep(8), error = function(e) invisible())
