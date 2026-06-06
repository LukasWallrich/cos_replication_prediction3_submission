# =============================================================================
# bayes_llmprior_model.R  —  Ansatz A: Bayesian logistic regression with
#                            LLM-elicited coefficient priors (health claims).
#
# Pipeline:
#   1. add log1p(n_f_tests_tei)  (new feature; raw column left untouched)
#   2. impute (median, leak-free) — optional MICE via existing apply_imputation()
#   3. fit stan_glm with informative normal priors from the LLM JSON
#      (continuous predictors z-standardized -> per-SD coef; binaries raw 0/1
#       -> 0->1 contrast coef). autoscale = FALSE everywhere.
#   4. KL gate: prior vs a wide-prior "data reference" posterior per coefficient
#   5. deploy-faithful OOF Brier (+ temperature calibration via 07_calibration.R)
#   6. final fit on the chosen scope -> predict Round 3
#
# NOTE: the rstanarm calls (stan_glm, normal(..., autoscale=FALSE),
# posterior_epred) could not be executed in the authoring environment; the
# argument names follow current rstanarm usage but verify once locally, and in
# particular run the names(coef(fit)) == predictors assertion below.
#
# Depends on 00_packages_config.R (rstanarm, glmnet, dplyr, jsonlite, ...),
# 06_models.R (PREDICTOR_SETS), 07_calibration.R (fit_calibrator/...),
# and optionally 05_imputation.R (apply_imputation) for the MICE path.
# =============================================================================

suppressPackageStartupMessages({
  source(here::here("pipeline/00_packages_config.R"))
  source(here::here("pipeline/07_calibration.R"))
})

# ── Predictor specification ──────────────────────────────────────────────────
# Raw set used for imputation (== PREDICTOR_SETS$transferable_lean, registered
# so apply_imputation() accepts it). log1p is derived AFTER imputation.
PREDICTORS_RAW <- c(
  "n_o_log", "pval_z", "llm_perceived_surprisingness", "llm_sample_adequacy",
  "llm_within_paper_rep", "llm_is_intervention", "es_directional", "n_f_tests_tei")

# Model set: raw n_f_tests_tei swapped for its log1p transform.
PREDICTORS_BAYES <- c(
  "n_o_log", "pval_z", "llm_perceived_surprisingness", "llm_sample_adequacy",
  "llm_within_paper_rep", "llm_is_intervention", "es_directional",
  "n_f_tests_tei_log1p")

# 0/1 predictors: NOT standardized; prior is on the 0->1 log-odds contrast.
BINARY_PREDICTORS_BAYES <- c(
  "llm_within_paper_rep", "llm_is_intervention", "es_directional")

# Deterministic feature; NA stays NA so imputation handles it upstream.
add_log1p_feature <- function(df) {
  df$n_f_tests_tei_log1p <- log1p(df$n_f_tests_tei)
  df
}

# ── LLM JSON priors -> (location, scale) vectors ─────────────────────────────
# Expects {"priors":[{"predictor","central_log_odds","ci90_low","ci90_high"},...]}.
# sd = (ci90_high - ci90_low)/3.29 (90% normal interval), floored to curb
# elicited overconfidence. central_log_odds is the signed prior mean.
priors_from_llm_json <- function(json, predictors = PREDICTORS_BAYES,
                                 sd_floor = 0.30) {
  js   <- if (is.character(json)) jsonlite::fromJSON(json, simplifyVector = FALSE) else json
  recs <- js$priors %||% js
  loc  <- setNames(rep(NA_real_, length(predictors)), predictors)
  scl  <- loc
  for (r in recs) {
    p <- r$predictor
    if (is.null(p) || !p %in% predictors) next
    loc[p] <- as.numeric(r$central_log_odds)
    scl[p] <- max((as.numeric(r$ci90_high) - as.numeric(r$ci90_low)) / 3.29, sd_floor)
  }
  if (any(is.na(loc)) || any(is.na(scl)))
    stop("Missing prior for: ",
         paste(predictors[is.na(loc) | is.na(scl)], collapse = ", "))
  list(location = loc[predictors], scale = scl[predictors])
}

# ── Optional MICE pre-imputation (reuses 05_imputation.R) ─────────────────────
# Leak-free (models on train rows only). Returns a single completed pair; on any
# failure returns NULL so the caller falls back to median. m = 1 = single
# imputation (pragmatic; for full MI pool over m > 1 fits, not done here).
mice_impute_pair <- function(train_raw, test_raw, pset_name = "transferable_lean",
                             m = 1L) {
  if (!exists("apply_imputation")) {
    source(here::here("pipeline/05_imputation.R"))
  }
  out <- tryCatch({
    res <- apply_imputation(train_raw, test_raw, pset_name = pset_name, m = m)
    list(train = res$train[[1]], test = res$test[[1]])
  }, error = function(e) {
    message("  [impute] MICE failed (", conditionMessage(e), ") -> median fallback.")
    NULL
  })
  out
}

# ── fit / predict (median imputation lives inside; "none" = pre-imputed) ──────
fit_bayes_logistic_llmprior <- function(train, outcome, predictors,
                                        prior_loc, prior_scale,
                                        binary_predictors = BINARY_PREDICTORS_BAYES,
                                        prior_int_loc = 0, prior_int_scale = 2.5,
                                        impute_method = c("median", "none"),
                                        weights = NULL,
                                        chains = 4L, iter = 2000L, warmup = 1000L,
                                        seed = CONFIG$seed) {
  impute_method <- match.arg(impute_method)
  df <- as.data.frame(train[, c(predictors, outcome), drop = FALSE])
  for (p in predictors) df[[p]] <- as.numeric(df[[p]])
  df[[outcome]] <- as.numeric(df[[outcome]])

  continuous <- setdiff(predictors, binary_predictors)

  # Moments + medians from OBSERVED (pre-imputation) train values, so that one
  # coefficient unit on a continuous predictor equals one OBSERVED SD — the SD
  # the LLM reasons about. Frozen for the test transform.
  cont_mean <- vapply(continuous, function(p) mean(df[[p]], na.rm = TRUE), numeric(1))
  cont_sd   <- vapply(continuous, function(p) {
    s <- sd(df[[p]], na.rm = TRUE); if (is.na(s) || s < 1e-9) 1 else s }, numeric(1))
  medians   <- vapply(predictors, function(p) stats::median(df[[p]], na.rm = TRUE), numeric(1))

  if (impute_method == "median") {
    for (p in predictors) df[[p]][is.na(df[[p]])] <- medians[[p]]
  } else if (anyNA(df[, predictors])) {
    warning("impute_method='none' but NAs present -> filling with train medians.")
    for (p in predictors) df[[p]][is.na(df[[p]])] <- medians[[p]]
  }

  for (p in continuous) df[[p]] <- (df[[p]] - cont_mean[[p]]) / cont_sd[[p]]

  # Fixed term order = predictors; prior vectors reordered to match.
  fml <- as.formula(paste(outcome, "~", paste(predictors, collapse = " + ")))
  loc <- prior_loc[predictors]; scl <- prior_scale[predictors]

  fit <- rstanarm::stan_glm(
    fml, data = df, family = binomial(),
    prior           = rstanarm::normal(location = as.numeric(loc),
                                       scale    = as.numeric(scl),
                                       autoscale = FALSE),
    prior_intercept = rstanarm::normal(prior_int_loc, prior_int_scale, autoscale = FALSE),
    weights = if (!is.null(weights)) pmax(as.numeric(weights), 1e-6) else NULL,
    chains = chains, iter = iter, warmup = warmup,
    cores = 1L, refresh = 0L, seed = seed)

  # Guard: coefficient order MUST match the prior vector order.
  stopifnot(identical(names(coef(fit))[-1], predictors))

  list(fit = fit, predictors = predictors, binary_predictors = binary_predictors,
       continuous = continuous, cont_mean = cont_mean, cont_sd = cont_sd,
       medians = medians, impute_method = impute_method)
}

pred_bayes_logistic_llmprior <- function(model_obj, newdata) {
  nd <- as.data.frame(newdata[, model_obj$predictors, drop = FALSE])
  for (p in model_obj$predictors) {
    nd[[p]] <- as.numeric(nd[[p]])
    nd[[p]][is.na(nd[[p]])] <- model_obj$medians[[p]]   # frozen train medians
  }
  for (p in model_obj$continuous)
    nd[[p]] <- (nd[[p]] - model_obj$cont_mean[[p]]) / model_obj$cont_sd[[p]]
  pp <- rstanarm::posterior_epred(model_obj$fit, newdata = nd)
  pmax(pmin(colMeans(pp), 1), 0)
}

# ── KL gate: prior vs a wide-prior data-reference posterior ───────────────────
# A plain glm() MLE is unusable here (separation, glm_mle_usable = FALSE in the
# diagnostics), so the "what the data say" reference is a stan_glm with wide
# normal(0, ref_scale) priors. Per coefficient: Gaussian KL(prior || reference).
fit_data_reference <- function(train, outcome, predictors,
                               binary_predictors = BINARY_PREDICTORS_BAYES,
                               ref_scale = 5, impute_method = "median",
                               chains = 4L, iter = 2000L, warmup = 1000L,
                               seed = CONFIG$seed) {
  np  <- length(predictors)
  
  # ── detect constant predictors in this training slice ──────────────────────
  df_check <- as.data.frame(train[, predictors, drop = FALSE])
  for (p in predictors) df_check[[p]] <- as.numeric(df_check[[p]])
  for (p in predictors) {
    nas <- is.na(df_check[[p]])
    if (any(nas)) df_check[[p]][nas] <- stats::median(df_check[[p]], na.rm = TRUE)
  }
  is_const <- vapply(predictors, function(p) {
    x <- df_check[[p]]; length(unique(x[is.finite(x)])) <= 1
  }, logical(1))
  
  dropped <- character(0)                                 # ← initialisieren
  if (any(is_const)) {
    dropped <- predictors[is_const]
    message("[fit_data_reference] Dropping constant predictor(s): ",
            paste(dropped, collapse = ", "),
            " — will use fallback wide prior as reference.")
  }
  preds_use <- predictors[!is_const]
  
  # ── fit with non-constant predictors only ──────────────────────────────────
  ref <- fit_bayes_logistic_llmprior(
    train, outcome, preds_use,
    prior_loc   = setNames(rep(0, length(preds_use)), preds_use),
    prior_scale = setNames(rep(ref_scale, length(preds_use)), preds_use),
    binary_predictors = intersect(binary_predictors, preds_use),
    prior_int_scale = ref_scale,
    impute_method = impute_method,
    chains = chains, iter = iter, warmup = warmup, seed = seed)
  
  draws <- as.matrix(ref$fit)[, preds_use, drop = FALSE]
  
  # ── assemble full-length mu/sd vectors (constant -> fallback) ──────────────
  mu_out <- setNames(rep(0, np), predictors)
  sd_out <- setNames(rep(ref_scale, np), predictors)
  mu_out[preds_use] <- colMeans(draws)
  sd_out[preds_use] <- apply(draws, 2, sd)
  
  list(mu = mu_out, sd = sd_out, model = ref, dropped = dropped)
}

kl_gauss <- function(mu0, sd0, mu1, sd1) {           # KL( N(mu0,sd0) || N(mu1,sd1) )
  log(sd1 / sd0) + (sd0^2 + (mu0 - mu1)^2) / (2 * sd1^2) - 0.5
}

prior_kl_gate <- function(prior_loc, prior_scale, ref,
                          predictors = PREDICTORS_BAYES) {
  mu0 <- prior_loc[predictors]; sd0 <- prior_scale[predictors]
  mu1 <- ref$mu[predictors];    sd1 <- ref$sd[predictors]
  kl  <- kl_gauss(mu0, sd0, mu1, sd1)
  overlap <- abs(mu0 - mu1) < sd1           # prior mean within 1 ref-SD of data
  tighter <- sd0 < sd1
  regime <- ifelse(kl > 2,                         "overconfident / conflicting",
            ifelse(sd0 >= 0.9 * sd1 & abs(mu0 - mu1) < 0.5 * sd1, "adds little (~ data)",
            ifelse(tighter & overlap,              "sweet spot",
                                                   "review")))
  tibble::tibble(
    predictor   = predictors,
    prior_mean  = round(mu0, 3), prior_sd = round(sd0, 3),
    data_mean   = round(mu1, 3), data_sd  = round(sd1, 3),
    kl          = round(kl, 3),
    regime      = regime)
}

# ── Deploy-faithful OOF Brier (stratified k-fold; refits the prior model) ─────
# WARNING: refits stan_glm once per fold (~minutes each). At n ~ 47 the OOF Brier
# is very noisy; read it together with the KL gate, not on its own.
oof_brier_bayes <- function(train, outcome, predictors, prior_loc, prior_scale,
                            binary_predictors = BINARY_PREDICTORS_BAYES,
                            group_var = "doi_o",
                            k = 5L, seed = CONFIG$seed, ...) {
  set.seed(seed)
  y <- as.numeric(train[[outcome]])
  n <- length(y)
  
  # ── grouped fold assignment: all rows sharing a group stay together ────────
  if (!is.null(group_var) && group_var %in% names(train)) {
    grp     <- as.character(train[[group_var]])
    u_grps  <- unique(grp)
    n_grps  <- length(u_grps)
    
    if (n_grps < k) {
      warning("Fewer groups (", n_grps, ") than folds (", k,
              "). Reducing k to ", n_grps)
      k <- n_grps
    }
    
    # stratify groups by majority outcome within each group
    grp_y   <- tapply(y, grp, function(x) round(mean(x)))  # 0 or 1 per group
    grp_fold <- integer(n_grps)
    names(grp_fold) <- u_grps
    for (cls in c(0, 1)) {
      g_idx <- which(grp_y[u_grps] == cls)
      grp_fold[u_grps[g_idx]] <- sample(rep_len(seq_len(k), length(g_idx)))
    }
    
    # map group-level folds back to rows
    fold <- grp_fold[grp]
    
    message("[oof_brier_bayes] Grouped CV by '", group_var, "': ",
            n_grps, " groups in ", k, " folds.")
  } else {
    # fallback: ungrouped, outcome-stratified (original behavior)
    fold <- integer(n)
    for (cls in c(0, 1)) {
      idx <- which(y == cls)
      fold[idx] <- sample(rep_len(seq_len(k), length(idx)))
    }
    if (!is.null(group_var))
      warning("group_var '", group_var, "' not found in data. ",
              "Using ungrouped folds.")
  }
  
  # ── OOF predictions ───────────────────────────────────────────────────────
  oof <- rep(NA_real_, n)
  for (f in seq_len(k)) {
    tr <- train[fold != f, , drop = FALSE]
    te <- train[fold == f, , drop = FALSE]
    m  <- fit_bayes_logistic_llmprior(tr, outcome, predictors, prior_loc,
                                      prior_scale, binary_predictors, ...)
    oof[fold == f] <- pred_bayes_logistic_llmprior(m, te)
  }
  brier_raw <- brier_score(oof, y)
  
  # ── Leave-one-fold-out temperature calibration ─────────────────────────────
  oof_cal <- oof
  for (f in seq_len(k)) {
    other <- fold != f & is.finite(oof) & is.finite(y)
    cal   <- fit_calibrator(oof[other], y[other], method = "temperature")
    oof_cal[fold == f] <- apply_calibrator(cal, oof[fold == f])
  }
  
  list(oof = oof, fold = fold,
       brier_raw  = brier_raw,
       brier_temp = brier_score(oof_cal, y),
       k = k, grouped = !is.null(group_var) && group_var %in% names(train))
}

# ── Predictor table for the prompt (exact numbers from the fit data) ──────────
build_predictor_table <- function(df_fit, predictors = PREDICTORS_BAYES,
                                  binary_predictors = BINARY_PREDICTORS_BAYES,
                                  path = here::here("output/prior_predictor_table.csv")) {
  q <- function(x, p) round(unname(stats::quantile(x, p, na.rm = TRUE)), 3)
  rows <- purrr::map_dfr(predictors, function(p) {
    x  <- as.numeric(df_fit[[p]]); xo <- x[is.finite(x)]
    is_bin <- p %in% binary_predictors
    tibble::tibble(
      predictor = p,
      prior_unit = if (is_bin) "contrast 1 vs 0" else "per +1 SD",
      obs_min = round(min(xo), 3), p05 = q(xo, .05), median = q(xo, .5),
      p95 = q(xo, .95), obs_max = round(max(xo), 3),
      spread = if (is_bin) paste0(round(mean(xo) * 100), "% are 1")
               else paste0("SD = ", round(sd(xo), 3)),
      na_rate = round(mean(!is.finite(x)), 3))
  })
  readr::write_csv(rows, path)
  rows
}

# =============================================================================
# WORKFLOW (adapt object/flag names; uncomment to run)
# =============================================================================
# base_df  <- add_log1p_feature(train_base)                       # from 04_base_dataset.R
# r3_df    <- add_log1p_feature(round3_test)                      # unlabeled R3 predictors
# outcome  <- "statistical_success"
#
# fit_scope <- base_df[base_df$is_test_similar == 1, ]            # default: the 47
# ref_scope <- base_df[base_df$is_health      == 1, ]             # stabler KL reference (107)
#
# # 1. priors from the LLM
# pr <- priors_from_llm_json(here::here("output/llm_priors.json"))
#
# # 2. table for the prompt (exact, incl. log1p SD) — regenerate if predictors change
# build_predictor_table(ref_scope)
#
# # 3. KL gate
# ref  <- fit_data_reference(ref_scope, outcome, PREDICTORS_BAYES)
# gate <- prior_kl_gate(pr$location, pr$scale, ref); print(gate)
#
# # 4. OOF Brier on the fit scope
# oof <- oof_brier_bayes(fit_scope, outcome, PREDICTORS_BAYES, pr$location, pr$scale)
# cat("OOF Brier raw:", oof$brier_raw, " temp:", oof$brier_temp, "\n")
#
# # 5. final fit + Round-3 prediction (median path; for MICE pre-impute with
# #    mice_impute_pair() and pass impute_method = "none")
# fit_final <- fit_bayes_logistic_llmprior(fit_scope, outcome, PREDICTORS_BAYES,
#                                          pr$location, pr$scale)
# p_r3 <- pred_bayes_logistic_llmprior(fit_final, r3_df)
# readr::write_csv(tibble::tibble(claim_id = r3_df$claim_id, p = p_r3),
#                  here::here("output/round3_bayes_llmprior_preds.csv"))
