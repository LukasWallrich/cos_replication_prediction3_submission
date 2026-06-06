# =============================================================================
# 06_models.R  —  Base learners, model grid, predictor sets
# =============================================================================
#
# Single source of truth for the base learners used by:
#   - 02_loro_validation.R        (single-model LORO comparison)
#   - 04_stacking.R              (stacked ensemble built on these learners)
#   - 10_submission.R             (final fit)
#
# CONVENTION
# ----------
#   fit_*(train_df, outcome_col, predictors, weights) -> model object
#   pred_*(model_obj, newdata_df, predictors)         -> numeric vector in [0,1]
#
# Each learner predicts a probability (or [0,1]-clipped soft label) so the
# outputs can be averaged / stacked directly. brier_score() lives in
# 00_packages_config.R.
# =============================================================================

suppressPackageStartupMessages({
  source(here::here("pipeline/00_packages_config.R"))
})

# ── Predictor sets ────────────────────────────────────────────────────────────
# Named feature groups. Models that cannot handle a set are caught upstream by
# tryCatch and logged as NA Brier.

PREDICTOR_SETS <- list(

  # Round-2 baseline (for direct comparison with historical scores)
  round2 = c(
    "n_o_log", "pval_z"
  ),
  round3 = c(
    "n_o_log", "log_bf10_h"
  ),

  # Extended core set (adds effect size + claim features)
  extended = c(
    "n_o_log", "pval_z", "prop_digits",
    "r_o_abs",
    "claim_length_log", "has_pval_text",
    "n_o_miss_ind", "pval_value_o_miss_ind"
  ),

  # Health-domain focused (adds domain indicator + directional ES)
  health_focused = c(
    "n_o_log", "pval_z", "prop_digits",
    "r_o_abs",
    "claim_length_log", "has_pval_text",
    "is_health", "es_directional",
    "n_o_miss_ind", "pval_value_o_miss_ind", "r_o_miss_ind"
  ),

  # GAM set
  GAM_set = c(
    "pval_z", "power_proxy_h",
    "n_o_log",
    "llm_online_study", "llm_rct_study",
    "llm_within_paper_rep", "llm_student_sample",
    "submission_note_xml_miss_ind", "es_directional", "n_authors_oa"
  ),
  transferable_lean = c(
    "n_o_log",
    "pval_z",
    "llm_perceived_surprisingness",
    "llm_sample_adequacy",
    "llm_within_paper_rep",
    "llm_is_intervention",
    "es_directional",
    "n_f_tests_tei"
  ),
  transferable_lean_indicators = c(
    "n_o_log",
    "pval_z",
    "llm_perceived_surprisingness",
    "llm_sample_adequacy",
    "llm_within_paper_rep",
    "llm_is_intervention",
    "es_directional",
    "n_f_tests_tei",
    "llm_perceived_surprisingness_miss_ind",
    "llm_sample_adequacy_miss_ind",
    "llm_within_paper_rep_miss_ind",
    "llm_is_intervention_miss_ind"
  ),
  transferable_lean_plus = c(
    "n_o_log",
    "llm_perceived_surprisingness",
    "llm_sample_adequacy",
    "llm_within_paper_rep",
    "llm_is_intervention",
    "es_directional",
    "n_f_tests_tei",
    "has_multiplicity_correction_tei",
    "log_bf10_h",
    "first_author_productivity_log1p",
    "team_works_top25_median"
  ),

  # Full set (for regularised models like elastic net / random forest)
  full = c(
    "n_o_log", "pval_z", "prop_digits",
    "r_o_abs", "r_o_positive", "r_o_log",
    "author_overlap_missing",
    "claim_length_log", "claim_n_digits",
    "has_pval_text", "has_ci_text", "has_effect_text", "has_interaction",
    "is_health", "es_directional", "pval_exact", "pval_one_tailed",
    "prereg_r_bin"
  ),

  # Enriched set — full + new features from forecast-improvements:
  enriched = c(
    "n_o_log", "pval_z", "prop_digits",
    "r_o_abs", "r_o_positive", "claim_length_log", "has_pval_text",
    "is_health", "es_directional", "pval_exact",
    "p_hack_index_tei", "p_heaping_flag_tei",
    "tests_per_1000_words_tei", "multistudy_paper_tei",
    "n_f_tests_tei", "n_t_tests_tei",
    "hedging_density_discussion_only_tei",
    "llm_open_data", "llm_preregistration", "llm_apriori_power",
    "llm_has_robustness", "llm_study_count",
    # top predictive-AND-stable LLM signals per output/feature_stability.csv
    # (|rho| 0.34-0.39, |SMD| < 0.21) — previously scored well but unused
    "llm_strength_of_evidence_mean", "llm_plausibility_ord_mean", "llm_p_numeric_mean",
    "citation_count_t_plus_2y", "n_referenced_works"
  ),

  # Stable-top set — small list of features that are both predictive
  # (|rho| > 0.10) AND stable across train→test (|SMD| < 0.6), per
  # output/feature_stability.csv. Tests whether the propensity-stability
  # filter improves LORO over the kitchen-sink enriched set.
  stable_top = c(
    "n_o_log", "pval_z",
    "r_harmonized", "evidence_strength_score",
    "llm_strength_of_evidence_mean",
    "llm_open_data", "llm_preregistration",
    "tests_per_1000_words_tei"
  )
)

# Enriched + judge-LLM ensemble predictions (new pool; joined in 04_base_dataset.R).
# The LLM ensemble is the only signal NOT fit to the training sample and is only
# ~0.48 correlated with the model stack — genuine diversity, and it generalises
# out-of-sample (LORO Brier ~0.229 vs naive 0.25). Covered on R1/R2/R3 but not
# FORRT, so `judge_llm_miss_ind` flags the (feature-mean-imputed) FORRT rows and
# lets the learners gate on whether a real LLM prediction is present.
PREDICTOR_SETS$enriched_llm <- c(
  PREDICTOR_SETS$enriched,
  "judge_mean_p", "judge_sd_p", "judge_model_disagreement", "judge_llm_miss_ind"
)

# ── Model fit / predict functions ──────────────────────────────────────────────

## Logistic Regression ─────────────────────────────────────────────────────────
# quasibinomial handles soft labels (0 < y < 1) without warnings.
fit_logistic <- function(train, outcome_col, predictors, weights = NULL) {
  fml <- as.formula(paste(outcome_col, "~", paste(predictors, collapse = " + ")))
  w   <- if (is.null(weights)) rep(1, nrow(train)) else weights
  glm(fml, data = train, family = quasibinomial(link = "logit"), weights = w)
}

pred_logistic <- function(model_obj, newdata, predictors) {
  p <- predict(model_obj, newdata = newdata, type = "response")
  pmax(pmin(as.numeric(p), 1), 0)
}

## GAM  ──────────────────────────────────────
# Smooth terms applied to continuous predictors; linear for binary/indicators.
# Low spline basis dimension (k=4) prevents overfitting on small training folds.
fit_gam <- function(train, outcome_col, predictors, weights = NULL) {
  smooth_candidates <- c("pval_z", "prop_digits", "r_o_abs",
                         "r_o_log", "claim_length_log")
  smooth_preds  <- intersect(predictors, smooth_candidates)
  linear_preds  <- setdiff(predictors, smooth_candidates)

  rhs <- paste(c(
    if (length(smooth_preds) > 0) paste0("s(", smooth_preds, ", k=4)"),
    if (length(linear_preds) > 0) linear_preds
  ), collapse = " + ")

  fml <- as.formula(paste(outcome_col, "~", rhs))
  w   <- if (is.null(weights)) rep(1, nrow(train)) else weights
  y   <- train[[outcome_col]]
  fam <- if (all(na.omit(y) %in% c(0, 1))) binomial() else quasibinomial()

  mgcv::gam(
    fml,
    data = train[, c(outcome_col, predictors)],
    family = fam,
    weights = w,
    method = "REML",
    select = TRUE,
    gamma = 1.4,
    bs = "cr"
  )
}
pred_gam <- function(model_obj, newdata, predictors) {
  p <- predict(model_obj, newdata = newdata[, predictors, drop = FALSE],
               type = "response")
  pmax(pmin(as.numeric(p), 1), 0)
}

## Random Forest (ranger) ──────────────────────────────────────────────────────
# Regression forest (numeric outcome) so that predicted values live in [0, 1].

fit_rf <- function(train, outcome_col, predictors, weights = NULL,
                   min.node.size = 25, mtry = NULL, num.trees = 1000) {
  train_r <- train |> mutate(.rf_y = as.numeric(get(outcome_col)))
  fml <- as.formula(paste(".rf_y ~", paste(predictors, collapse = " + ")))
  mtry_use <- if (!is.null(mtry)) min(as.integer(mtry), length(predictors)) else NULL
  ranger::ranger(fml,
                 data          = train_r[, c(".rf_y", predictors)],
                 num.trees     = num.trees,
                 min.node.size = min.node.size,
                 mtry          = mtry_use,
                 case.weights  = weights,
                 seed          = CONFIG$seed)
}

pred_rf <- function(model_obj, newdata, predictors) {
  p <- predict(model_obj, data = newdata[, predictors, drop = FALSE])$predictions
  pmax(pmin(as.numeric(p), 1), 0)
}

## Elastic Net (glmnet) ────────────────────────────────────────────────────────
fit_elnet <- function(train, outcome_col, predictors, weights = NULL,
                      alpha = 0.5, lambda_choice = "min", relax = FALSE,
                      seed = 3, n_seeds = 25L) {
  x  <- as.matrix(train[, predictors])
  y  <- as.numeric(train[[outcome_col]])
  w  <- pmax(replace_na(if (is.null(weights)) rep(1, nrow(train)) else weights, 1.0), 1e-6)
  fam <- if (all(na.omit(y) %in% c(0, 1))) "binomial" else "gaussian"

  # Average over n_seeds CV partitions to remove cv.glmnet's lambda-selection
  # randomness: reproducible (fixed seed set) and lower-variance than one fit.
  # The global RNG stream is saved/restored so this fit leaves no side effect.
  has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (has_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit(if (has_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)

  seeds <- seed + seq_len(n_seeds) - 1L
  members <- lapply(seeds, function(s) {
    set.seed(s)
    cv_mod <- glmnet::cv.glmnet(x, y, family = fam, alpha = alpha,
                                weights = w, nfolds = 10, relax = relax)
    s_val <- switch(lambda_choice,
                    "min" = cv_mod$lambda.min,
                    "1se" = cv_mod$lambda.1se,
                    stop("unknown lambda_choice: ", lambda_choice))
    list(model = cv_mod, s = s_val,
         gamma = if (relax) cv_mod$gamma.min else NULL)
  })

  list(members       = members,        # n_seeds fitted CV models, averaged at predict time
       family        = fam,
       predictors    = predictors,
       alpha         = alpha,
       lambda_choice = lambda_choice,
       relax         = relax,
       n_seeds       = n_seeds)
}

pred_elnet <- function(model_obj, newdata, predictors) {
  x_new <- as.matrix(newdata[, predictors])
  pmat  <- vapply(model_obj$members, function(m) {
    args <- list(m$model, newx = x_new, s = m$s, type = "response")
    if (isTRUE(model_obj$relax)) args$gamma <- m$gamma
    as.numeric(do.call(predict, args))
  }, numeric(nrow(newdata)))
  pmat <- matrix(pmat, nrow = nrow(newdata))   # robust to nrow(newdata) == 1
  pmax(pmin(rowMeans(pmat), 1), 0)             # average predicted probabilities over seeds
}

## XGBoost ─────────────────────────────────────────────────────────────────────
# binary:logistic objective with CV early stopping; output clipped to [0,1] and
# label-smoothed at predict time to damp overconfidence.
fit_xgb <- function(train, outcome_col, predictors, weights = NULL) {
  x <- as.matrix(train[, predictors])
  y <- as.numeric(train[[outcome_col]])
  w <- pmax(replace_na(if (is.null(weights)) rep(1, nrow(train)) else weights, 1.0), 1e-6)
  dtrain <- xgboost::xgb.DMatrix(data = x, label = y, weight = w)

  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    eta              = 0.015,
    max_depth        = 3,
    min_child_weight = 30,
    subsample        = 0.7,
    colsample_bytree = 0.7,
    colsample_bynode = 0.5,
    gamma            = 0.2,
    alpha            = 1.0,
    lambda           = 2.0,
    tree_method = "hist"
  )
  cv <- xgboost::xgb.cv(params = params, data = dtrain, nrounds = 1500,
                        nfold = 5, early_stopping_rounds = 30, verbose = 0)
  best_iter <- cv$best_iteration
  if (is.null(best_iter) || length(best_iter) == 0L || best_iter < 1L) best_iter <- 200L
  xgboost::xgb.train(params = params, data = dtrain, nrounds = best_iter, verbose = 0)
}

pred_xgb <- function(model_obj, newdata, predictors, shrink_factor = 0.01) {
  x_new <- as.matrix(newdata[, predictors])
  dnew  <- xgboost::xgb.DMatrix(data = x_new)

  p <- predict(model_obj, newdata = dnew)
  p <- as.numeric(p)
  p <- p * (1 - 2 * shrink_factor) + shrink_factor

  pmax(pmin(p, 1), 0)
}

# ── Bayesian Logistic Regression (rstanarm, regularized horseshoe) ────────────
fit_bayes_logistic <- function(train_data, outcome, predictors, weights = NULL) {
  df <- as.data.frame(train_data[, c(predictors, outcome), drop = FALSE])
  df[[outcome]] <- as.numeric(df[[outcome]])
  for (p in predictors) {
    df[[p]] <- as.numeric(df[[p]])
    df[[p]][is.na(df[[p]])] <- 0
  }

  # Save training-set standardization params for consistent test transform
  pred_means <- colMeans(df[, predictors, drop = FALSE])
  pred_sds   <- apply(df[, predictors, drop = FALSE], 2,
                      function(x) { s <- sd(x); if (is.na(s) || s < 1e-9) 1.0 else s })

  dfs <- df
  for (p in predictors) dfs[[p]] <- (df[[p]] - pred_means[p]) / pred_sds[p]

  np <- length(predictors)
  n  <- nrow(dfs)
  p0 <- min(10L, max(3L, np %/% 3L))  # expected number of relevant predictors

  fit <- rstanarm::stan_glm(
    as.formula(paste0(outcome, " ~ ", paste(predictors, collapse = " + "))),
    data    = dfs,
    family  = binomial(),
    prior   = rstanarm::hs(
      global_scale = (p0 / (np - p0)) / sqrt(n),
      slab_df      = 4,
      slab_scale   = 2.5
    ),
    prior_intercept = rstanarm::normal(0, 2.5),
    weights = if (!is.null(weights)) pmax(as.numeric(weights), 1e-6) else rep(1.0, n),
    chains = 2L, iter = 1000L, warmup = 500L,
    cores  = 1L, refresh = 0L,
    seed   = CONFIG$seed
  )

  list(fit = fit, predictors = predictors,
       pred_means = pred_means, pred_sds = pred_sds)
}

pred_bayes_logistic <- function(model_obj, new_data, predictors) {
  nd <- as.data.frame(new_data[, predictors, drop = FALSE])
  for (p in predictors) {
    nd[[p]] <- as.numeric(nd[[p]])
    nd[[p]][is.na(nd[[p]])] <- 0
    nd[[p]] <- (nd[[p]] - model_obj$pred_means[p]) / model_obj$pred_sds[p]
  }
  # Posterior mean of predicted probability = Bayesian natural calibration
  pp <- rstanarm::posterior_epred(model_obj$fit, newdata = nd)
  pmax(pmin(colMeans(pp), 1.0), 0.0)
}

# ── BART ─────────────────────────────────────────────────────────────────

fit_bart <- function(train, outcome_col, predictors, weights = NULL) {
  # Define the structural feature sets allowed for MCMC execution to prevent timeout on high-dimensional blocks
  allowed_sets <- list(
    # Round-2 baseline (for direct comparison with historical scores)
    round2 = c(
      "n_o_log", "pval_z"
    ),
    # GAM set
    GAM_set = c(
      "pval_z", "power_proxy_h",
      "n_o_log",
      "llm_online_study", "llm_rct_study",
      "llm_within_paper_rep", "llm_student_sample",
      "submission_note_xml_miss_ind", "es_directional", "n_authors_oa"
    ),
    transferable_lean = c(
      "n_o_log",
      "pval_z",
      "llm_perceived_surprisingness",
      "llm_sample_adequacy",
      "llm_within_paper_rep",
      "llm_is_intervention",
      "es_directional",
      "n_f_tests_tei"
    ),
    transferable_lean_indicators = c(
      "n_o_log",
      "pval_z",
      "llm_perceived_surprisingness",
      "llm_sample_adequacy",
      "llm_within_paper_rep",
      "llm_is_intervention",
      "es_directional",
      "n_f_tests_tei",
      "llm_perceived_surprisingness_miss_ind",
      "llm_sample_adequacy_miss_ind",
      "llm_within_paper_rep_miss_ind",
      "llm_is_intervention_miss_ind"
    ),
    stable_top = c(
      "n_o_log", "pval_z",
      "r_harmonized", "evidence_strength_score",
      "llm_strength_of_evidence_mean",
      "llm_open_data", "llm_preregistration",
      "tests_per_1000_words_tei"
    )
  )

  # Check if the incoming vector matches any whitelisted configuration (order-invariant)
  is_whitelisted <- any(sapply(allowed_sets, function(allowed) {
    length(predictors) == length(allowed) && all(sort(predictors) == sort(allowed))
  }))

  # Fail-fast exit if the predictor footprint does not match whitelisted configurations
  if (!is_whitelisted) {
    return(NULL)
  }

  x <- as.matrix(train[, predictors])
  y <- as.numeric(train[[outcome_col]])
  w <- pmax(dplyr::coalesce(if (is.null(weights)) rep(1, nrow(train)) else weights, 1.0), 1e-6)

  # Structural priors optimized to limit tree depth and control model complexity.
  # Restricting power/base forces shallow trees, improving calibration under domain shift.
  mod <- tryCatch({
    dbarts::bart(
      x.train = x, y.train = y,
      ntree = 100, k = 3.0, power = 2.0, base = 0.95,
      verbose = FALSE, keeptrainfits = FALSE,
      combinechains = TRUE, ndpost = 1500, nchain = 4L
    )
  }, error = function(e) {
    message("dbarts training failed: ", e$message)
    NULL
  })

  if (is.null(mod)) return(NULL)

  list(model = mod, predictors = predictors, type = "BART")
}

pred_bart <- function(model_obj, newdata, predictors, shrink_factor = 0.01) {
  if (is.null(model_obj)) {
    return(rep(NA_real_, nrow(newdata)))
  }

  x_new <- as.matrix(newdata[, model_obj$predictors, drop = FALSE])

  # Extracts the posterior predictive distribution mean 
  p_mcmc <- predict(model_obj$model, newdata = x_new, type = "ev")
  p <- as.numeric(colMeans(p_mcmc))

  # Post-hoc label smoothing to damp overconfidence and minimize Brier score penalties on shifted domains
  p <- p * (1 - 2 * shrink_factor) + shrink_factor
  pmax(pmin(p, 1), 0)
}

# ── lightGBM ─────────────────────────────────────────────────────────────────

fit_lgbm <- function(train, outcome_col, predictors, weights = NULL,
                     monotone_dir          = NULL,
                     nrounds_max           = 1500L,
                     early_stopping_rounds = 30L,
                     nfold                 = 5L) {

  x <- as.matrix(train[, predictors, drop = FALSE])
  y <- as.numeric(train[[outcome_col]])
  w <- pmax(dplyr::coalesce(
    if (is.null(weights)) rep(1, nrow(train)) else weights, 1.0), 1e-6)

  dtrain <- lightgbm::lgb.Dataset(data = x, label = y, weight = w)

  params <- list(
    objective                    = "binary",
    metric                       = "binary_logloss",
    learning_rate                = 0.015,
    num_leaves                   = 7L,
    max_depth                    = 3L,
    min_data_in_leaf             = 30L,
    feature_fraction             = 0.7,
    feature_fraction_bynode      = 0.5,
    bagging_fraction             = 0.7,
    bagging_freq                 = 1L,
    lambda_l1                    = 1.0,
    lambda_l2                    = 2.0
   )

  # Step 1: pick nrounds via CV early stopping.
  set.seed(CONFIG$seed)
  cv <- lightgbm::lgb.cv(
    params                = params,
    data                  = dtrain,
    nrounds               = nrounds_max,
    nfold                 = nfold,
    early_stopping_rounds = early_stopping_rounds,
    verbose               = -1L,
    eval_freq             = 1L
  )
  best_iter <- which.min(unlist(
    cv$record_evals[["valid"]][["binary_logloss"]][["eval"]]
  ))
  if (is.null(best_iter) || length(best_iter) == 0L || best_iter < 1L) best_iter <- 100L

  # Step 2: refit on full training data with the chosen nrounds
  set.seed(CONFIG$seed)
  mod <- lightgbm::lgb.train(
    params  = params,
    data    = dtrain,
    nrounds = best_iter,
    verbose = -1L
  )

  list(model      = mod,
       predictors = predictors,
       best_iter  = best_iter,
       type       = "LGBM")
}

pred_lgbm <- function(model_obj, newdata, predictors, shrink_factor = 0.01) {
  x_new <- as.matrix(newdata[, model_obj$predictors, drop = FALSE])
  p     <- as.numeric(predict(model_obj$model, x_new))
  p     <- p * (1 - 2 * shrink_factor) + shrink_factor
  pmax(pmin(p, 1), 0)
}

# ── Model grid ─────────────────────────────────────────────────────────────────
# Each entry: list(name, fit = fit_fn, pred = pred_fn, imputation_strategy)

MODEL_GRID <- list(
  list(name = "Logistic",     fit = fit_logistic,      pred = pred_logistic,      imputation_strategy = "pmm"),
  list(name = "GAM",          fit = fit_gam,           pred = pred_gam,           imputation_strategy = "pmm"),
  list(name = "RF",           fit = fit_rf,            pred = pred_rf,            imputation_strategy = "none"),
  list(name = "ElNet_1se",    fit = function(...) fit_elnet(..., alpha = 0.5, lambda_choice = "1se",  relax = FALSE),
       pred = pred_elnet,                                   imputation_strategy = "pmm"),
  list(name = "ElNet_min",    fit = function(...) fit_elnet(..., alpha = 0.5, lambda_choice = "min",  relax = FALSE),
       pred = pred_elnet,                                   imputation_strategy = "pmm"),
  list(name = "ElNet_relax",  fit = function(...) fit_elnet(..., alpha = 0.5, lambda_choice = "min",  relax = TRUE),
       pred = pred_elnet,                                   imputation_strategy = "pmm"),
  list(name = "Ridge_min",    fit = function(...) fit_elnet(..., alpha = 0.0, lambda_choice = "min",  relax = FALSE),
       pred = pred_elnet,                                   imputation_strategy = "pmm"),
  list(name = "XGB",          fit = fit_xgb,           pred = pred_xgb,           imputation_strategy = "none"),
  list(name = "BART",         fit = fit_bart,          pred = pred_bart,          imputation_strategy = "pmm"),
  list(name = "LGBM",         fit = fit_lgbm,          pred = pred_lgbm,          imputation_strategy = "none"),
  list(name = "BayesLogistic",fit = fit_bayes_logistic,pred = pred_bayes_logistic,imputation_strategy = "pmm")
)
cat("[06_models.R] loaded:", length(MODEL_GRID), "base learners,",
    length(PREDICTOR_SETS), "predictor sets\n")
