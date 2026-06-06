# =============================================================================
# 04_stacking.R  —  Stacked ensemble (meta-learner over base learners)
# =============================================================================
#
# WHAT STACKING IS
# ----------------
# Instead of picking the single best base learner, we let a small meta-learner
# combine all of them. The catch: the meta-learner must be trained on
# predictions the base learners made on rows they did NOT see, otherwise it
# learns to trust whichever base learner overfits hardest. So we generate
# OUT-OF-FOLD (OOF) base predictions via an inner CV, fit the meta-learner on
# those, then refit each base learner on the full training set for prediction.
#
#   inner CV  ─→  OOF base preds (leak-free)  ─→  fit meta-learner
#   full fit  ─→  base models for predicting new data
#
# The OOF base-pred matrix is also the honest input for fitting a calibrator
# (see 07_calibration.R): it is the only in-pipeline estimate of how the
# stacked model behaves on unseen rows.
#
# GROUPED FOLDS
# -------------
# Inner CV folds are assigned by `group_var` (doi_o) so that all claims from the
# same paper land in the same fold. This matches the project's group-aware CV
# and prevents same-paper leakage from inflating the meta-features.
#
# META-LEARNERS
# -------------
#   "avg"      simple mean of base preds. Zero parameters; the robust baseline.
#   "nnls"     non-negative weights summing to 1 (convex combination), chosen to
#              minimise OOF Brier. Classic Breiman stacking; can't extrapolate
#              past the base learners, so it is hard to overfit at small n.
#   "logistic" quasi-binomial GLM of outcome ~ base preds. Most flexible, but
#              base preds are highly collinear, so watch for instability.
#
# CONTRACT
# --------
#   fit_stack(train_df, base_specs, predictors, outcome_col, meta, ...) -> stack
#   pred_stack(stack, newdata, predictors)                             -> [0,1]
#
# Lower-level pieces (oof_base_predictions, fit_meta, predict_meta,
# base_holdout_matrix) are exported so the health-experiment scripts can reuse one set
# of OOF + full base fits across several meta-learners instead of recomputing them.
# =============================================================================

suppressPackageStartupMessages({
  source(here::here("pipeline/06_models.R"))
  source(here::here("pipeline/07_calibration.R"))  # for calibrate-before-stack
  source(here::here("pipeline/05_imputation.R"))
})

# ── Grouped fold assignment ────────────────────────────────────────────────────
# Returns an integer fold id per row. Rows sharing a `group_var` value get the
# same fold; NA groups are treated as singleton groups (each its own row).
assign_group_folds <- function(groups, n_folds, seed = CONFIG$seed) {
  groups <- as.character(groups)
  na_idx <- is.na(groups)
  groups[na_idx] <- paste0(".na_", seq_len(sum(na_idx)))   # unique per NA row
  ug <- unique(groups)
  set.seed(seed)
  fold_of_group <- setNames(
    sample(rep_len(seq_len(n_folds), length(ug))),
    ug
  )
  unname(fold_of_group[groups])
}

# ── Out-of-fold base predictions ───────────────────────────────────────────────
#' @return list(oof = matrix [n_train × n_base], y = numeric, fold_id = int).
#'   oof[i, m] is base learner m's prediction for row i, made by a model trained
#'   on the other folds. NA where a base learner failed on that fold.
oof_base_predictions <- function(train_df, base_specs, predictors, outcome_col,
                                 n_folds = 5, group_var = "doi_o",
                                 weights = NULL, seed = CONFIG$seed) {

  n <- nrow(train_df)
  M <- length(base_specs)
  model_names <- vapply(base_specs, `[[`, character(1), "name")

  fold_id <- assign_group_folds(train_df[[group_var]], n_folds, seed)
  y       <- as.numeric(train_df[[outcome_col]])
  w_all   <- if (is.null(weights)) rep(1, n) else weights

  oof <- matrix(NA_real_, nrow = n, ncol = M, dimnames = list(NULL, model_names))

  for (f in sort(unique(fold_id))) {
    val_rows <- which(fold_id == f)
    tr_rows  <- which(fold_id != f)
    if (length(val_rows) == 0L || length(tr_rows) == 0L) next

    tr <- train_df[tr_rows, , drop = FALSE]
    va <- train_df[val_rows, , drop = FALSE]
    w  <- w_all[tr_rows]

    for (m in seq_len(M)) {
      spec <- base_specs[[m]]
      oof[val_rows, m] <- tryCatch({
        mod <- spec$fit(tr, outcome_col, predictors, w)
        spec$pred(mod, va, predictors)
      }, error = function(e) {
        warning(sprintf("oof_base_predictions: %s failed on fold %s: %s",
                        spec$name, f, e$message))
        rep(NA_real_, length(val_rows))
      })
    }
  }

  list(oof = oof, y = y, fold_id = fold_id)
}

# ── base_holdout_matrix ─────────────────────────────────────────────────────────
# Predict `newdata` with each already-fitted full base model. Columns aligned to
# base_specs order/names. Used for both the holdout and the final test set.
base_holdout_matrix <- function(full_models, base_specs, newdata, predictors) {
  M <- length(base_specs)
  Z <- matrix(NA_real_, nrow = nrow(newdata), ncol = M,
              dimnames = list(NULL, vapply(base_specs, `[[`, character(1), "name")))
  for (m in seq_len(M)) {
    Z[, m] <- base_specs[[m]]$pred(full_models[[m]], newdata, predictors)
  }
  Z
}

# ── Meta-learners ───────────────────────────────────────────────────────────────
fit_meta <- function(Z, y, meta = c("nnls", "avg", "logistic")) {
  meta <- match.arg(meta)
  ok   <- stats::complete.cases(Z) & is.finite(y)
  Zok  <- Z[ok, , drop = FALSE]
  yok  <- y[ok]
  cols <- colnames(Z)

  if (nrow(Zok) < 5L) {
    warning("fit_meta: <5 complete OOF rows — falling back to simple average.")
    return(list(type = "avg", cols = cols))
  }

  switch(meta,
    "avg" = list(type = "avg", cols = cols),

    "nnls" = {
      # Convex combination via softmax parametrisation (guarantees w >= 0,
      # sum(w) = 1 with no constraint solver), minimising OOF Brier.
      M <- ncol(Zok)
      obj <- function(par) {
        w <- exp(par - max(par)); w <- w / sum(w)
        mean((as.numeric(Zok %*% w) - yok)^2)
      }
      opt <- optim(rep(0, M), obj, method = "BFGS",
                   control = list(maxit = 500))
      w <- exp(opt$par - max(opt$par)); w <- w / sum(w)
      names(w) <- cols
      list(type = "nnls", w = w, cols = cols)
    },

    "logistic" = {
      df <- as.data.frame(Zok); names(df) <- cols; df$.y <- yok
      fit <- suppressWarnings(
        glm(.y ~ ., data = df, family = quasibinomial())
      )
      list(type = "logistic", fit = fit, cols = cols)
    }
  )
}

predict_meta <- function(meta_fit, Z) {
  Zc <- Z[, meta_fit$cols, drop = FALSE]
  p <- switch(meta_fit$type,
    "avg"      = rowMeans(Zc, na.rm = TRUE),
    "nnls"     = as.numeric(Zc %*% meta_fit$w),
    "logistic" = {
      df <- as.data.frame(Zc); names(df) <- meta_fit$cols
      as.numeric(predict(meta_fit$fit, newdata = df, type = "response"))
    },
    stop("unknown meta type: ", meta_fit$type)
  )
  pmax(pmin(p, 1), 0)
}

# ── Leak-free OOF stacked predictions (cross-fitting) ───────────────────────────
# predict_meta(fit_meta(oof), oof) is IN-SAMPLE for the meta-learner: the meta
# coefficients were chosen on the very rows being predicted. Fitting a calibrator
# on those preds inherits that optimism. This cross-fits instead — for each inner
# fold k the meta-learner is fit on folds != k and applied to fold k — so every
# stacked OOF prediction comes from a meta-learner that never saw that row.
# `avg` has no parameters, so cross-fitting is a no-op for it (correct).
oof_stack_predictions <- function(oof, y, fold_id, meta = "nnls") {
  stack_oof <- rep(NA_real_, length(y))
  for (k in sort(unique(fold_id))) {
    tr <- which(fold_id != k); va <- which(fold_id == k)
    if (!length(tr) || !length(va)) next
    mf <- fit_meta(oof[tr, , drop = FALSE], y[tr], meta = meta)
    stack_oof[va] <- predict_meta(mf, oof[va, , drop = FALSE])
  }
  stack_oof
}

# ── Calibrate-before-stack ──────────────────────────────────────────────────────
# Calibrating each base learner BEFORE combining puts the members on comparable
# probability scales, so a diversity-preserving combiner (e.g. the average) can
# actually use them instead of being dominated by whichever learner is sharpest.
#
# calibrate_oof_columns: returns
#   $cal_oof      OOF base preds, each column cross-fit-calibrated (leak-free,
#                 ready to feed fit_meta)
#   $calibrators  per-column calibrators fit on the FULL OOF, for transforming a
#                 holdout/test base-pred matrix via apply_column_calibrators()
calibrate_oof_columns <- function(oof, y, fold_id, method = "temperature") {
  M       <- ncol(oof)
  cal_oof <- oof
  for (m in seq_len(M)) {
    for (k in sort(unique(fold_id))) {
      tr <- which(fold_id != k); va <- which(fold_id == k)
      if (!length(tr) || !length(va)) next
      cal <- fit_calibrator(oof[tr, m], y[tr], method = method)
      cal_oof[va, m] <- apply_calibrator(cal, oof[va, m])
    }
  }
  full_cals <- lapply(seq_len(M), function(m)
    fit_calibrator(oof[, m], y, method = method))
  list(cal_oof = cal_oof, calibrators = full_cals)
}

apply_column_calibrators <- function(Z, calibrators) {
  for (m in seq_len(ncol(Z))) Z[, m] <- apply_calibrator(calibrators[[m]], Z[, m])
  Z
}

# ── High-level convenience API ──────────────────────────────────────────────────
#' Fit a stacked ensemble.
#'
#' @param calibrate_base  if non-NULL (e.g. "temperature"), each base learner is
#'   cross-fit-calibrated before the meta-learner combines them (calibrate-before-
#'   stack). Otherwise the meta-learner combines raw base preds.
#' @return list including `stack_oof` — genuinely leak-free (cross-fit) stacked
#'   OOF predictions for fitting a downstream calibrator.
fit_stack <- function(train_df, base_specs, predictors, outcome_col,
                      meta = "nnls", n_folds = 5, group_var = "doi_o",
                      weights = NULL, calibrate_base = NULL, seed = CONFIG$seed) {

  oo <- oof_base_predictions(train_df, base_specs, predictors, outcome_col,
                             n_folds = n_folds, group_var = group_var,
                             weights = weights, seed = seed)

  # Optional calibrate-before-stack: cross-fit-calibrate each base column.
  col_cals <- NULL
  oof_use  <- oo$oof
  if (!is.null(calibrate_base)) {
    cc       <- calibrate_oof_columns(oo$oof, oo$y, oo$fold_id, method = calibrate_base)
    oof_use  <- cc$cal_oof
    col_cals <- cc$calibrators
  }

  meta_fit <- fit_meta(oof_use, oo$y, meta = meta)

  w_full <- if (is.null(weights)) rep(1, nrow(train_df)) else weights
  full_models <- lapply(base_specs, function(spec)
    spec$fit(train_df, outcome_col, predictors, w_full))

  list(
    base_specs   = base_specs,
    full_models  = full_models,
    meta_fit     = meta_fit,
    meta         = meta,
    predictors   = predictors,
    col_cals     = col_cals,                                       # NULL unless calibrate-before-stack
    # cross-fit (leak-free) stacked OOF preds — correct input for a calibrator:
    stack_oof    = oof_stack_predictions(oof_use, oo$y, oo$fold_id, meta = meta),
    oof_y        = oo$y
  )
}

pred_stack <- function(stack, newdata, predictors = stack$predictors) {
  Z <- base_holdout_matrix(stack$full_models, stack$base_specs, newdata, predictors)
  if (!is.null(stack$col_cals)) Z <- apply_column_calibrators(Z, stack$col_cals)
  predict_meta(stack$meta_fit, Z)
}

# ── RF hyperparameter tuning ────────────────────────────────────────────────────
#' Build a MODEL_GRID-compatible spec for an RF with fixed hyperparameters.
#' Drop-in replacement for the default RF entry, e.g.:
#'   grid[[rf_idx]] <- make_rf_spec(min.node.size = 50, mtry = 5)
make_rf_spec <- function(min.node.size = 5, mtry = NULL, num.trees = 500,
                         name = "RF") {
  force(min.node.size); force(mtry); force(num.trees)
  list(
    name = name,
    fit  = function(train, outcome_col, predictors, weights = NULL)
      fit_rf(train, outcome_col, predictors, weights,
             min.node.size = min.node.size, mtry = mtry, num.trees = num.trees),
    pred = pred_rf
  )
}

#' Tune RF over a min.node.size × mtry grid, scored by INNER grouped-CV OOF Brier.
#'
#' Selection happens entirely within `train_df` using the same grouped-fold logic
#' as oof_base_predictions(), so it is leak-free w.r.t. any outer holdout. mtry
#' values above the predictor count are capped and de-duplicated.
#'
#' @return list(best = 1-row df with min.node.size/mtry/brier_oof,
#'              grid = full grid ordered by brier_oof).
tune_rf <- function(train_df, predictors, outcome_col,
                    min_node_grid = c(5, 10, 20, 50, 100),
                    mtry_grid     = c(3, 5, 8, 12),
                    n_folds = 5, group_var = "doi_o",
                    weights = NULL, num.trees = 500, seed = CONFIG$seed) {

  p         <- length(predictors)
  mtry_grid <- sort(unique(pmin(as.integer(mtry_grid), p)))
  grid      <- expand.grid(min.node.size = min_node_grid, mtry = mtry_grid,
                           KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)

  fold_id <- assign_group_folds(train_df[[group_var]], n_folds, seed)
  y       <- as.numeric(train_df[[outcome_col]])
  w_all   <- if (is.null(weights)) rep(1, nrow(train_df)) else weights
  folds   <- sort(unique(fold_id))

  grid$brier_oof <- vapply(seq_len(nrow(grid)), function(g) {
    mns <- grid$min.node.size[g]; mt <- grid$mtry[g]
    oof_pred <- rep(NA_real_, nrow(train_df))
    for (f in folds) {
      val <- which(fold_id == f); tr <- which(fold_id != f)
      if (!length(val) || !length(tr)) next
      mod <- tryCatch(
        fit_rf(train_df[tr, , drop = FALSE], outcome_col, predictors,
               w_all[tr], min.node.size = mns, mtry = mt, num.trees = num.trees),
        error = function(e) NULL)
      if (!is.null(mod))
        oof_pred[val] <- pred_rf(mod, train_df[val, , drop = FALSE], predictors)
    }
    brier_score(oof_pred, y)
  }, numeric(1))

  grid <- grid[order(grid$brier_oof), , drop = FALSE]
  list(best = grid[1, , drop = FALSE], grid = grid)
}

cat("[04_stacking.R] loaded: oof_base_predictions, fit_meta, predict_meta,",
    "oof_stack_predictions, calibrate_oof_columns, apply_column_calibrators,",
    "base_holdout_matrix, fit_stack, pred_stack, tune_rf, make_rf_spec\n")
