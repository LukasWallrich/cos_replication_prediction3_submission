# =============================================================================
# 09_final_comparison.R
#   Self-contained, leak-free final comparison of the candidate confidence-score
#   models for the 3 submission slots. 
#
# EVAL MODES (all group-aware by doi_o; train side pruned so no doi_o spans the
#             train/eval boundary -> no leakage; eval rows are never dropped)
#   M1 oof_health         : common health-fold partition over health_big (107);
#                           feature train = non-health + other folds,
#                           bayes/prior train = other folds; predict held-out fold
#   M2 leave_chal_out     : train = non-health + FORRT-health; predict chal-health
#   M3 leave_all_health   : train = non-health only;           predict all health
#                           (prior_shrinkage undefined here -> NA)
#   M4 train_on_training  : train = dataset=="training";       predict !=training
#                           (chal-health target); off by default for bayes (cost)
#
# TARGETS reported: chal_health (n=47) and health_big (n=107) where applicable.
#
# CALIBRATION
#   Per candidate, temperature T is estimated on the leak-free M1 health OOF and
#   applied (a) as the deployment T for the R3 scores and the per-mode calibrated
#   Brier, and (b) nested (T for fold k from the other folds) for the honest M1
#   reliability plot. An in-sample/pooled T on the same rows it is scored on would
#   look optimistic; the nested M1 version does not and is the one plotted.
# =============================================================================

# ---- preflight: verify the package root is anchored and complete -------------
# Every path below is resolved by here::here() from the project root (the folder
# holding `.here`). If R was started outside the package, or this copy is
# incomplete (e.g. a partial cloud sync), those lookups resolve to the wrong
# place and fail later with a confusing "file not found". Fail fast instead, and
# say exactly which file and where it looked.
local({
  root     <- tryCatch(here::here(), error = function(e) getwd())
  required <- c("pipeline/06_models.R", "bayesian_priors/bayes_llmprior_model.R")
  missing  <- required[!file.exists(file.path(root, required))]
  if (length(missing))
    stop("[final] package root looks wrong or incomplete.\n",
         "  here::here() -> ", root, "\n",
         "  getwd()      -> ", getwd(), "\n",
         "  not found    -> ", paste(missing, collapse = ", "), "\n",
         "  Fix: start R from the package root (the folder with the `.here` file),\n",
         "       e.g. `cd <package>` then `Rscript pipeline/09_final_comparison.R`;\n",
         "       if the folder is missing files, refresh the copy (git pull / re-sync).",
         call. = FALSE)
})

suppressPackageStartupMessages({
  source(here::here("pipeline/00_packages_config.R"))   # brier_score, CONFIG, %||%, ggplot2, ...
  source(here::here("pipeline/06_models.R"))            # PREDICTOR_SETS, fit_elnet/gam, pred_*
  source(here::here("pipeline/07_calibration.R"))       # temperature_scaling, fit_calibrator, .clip01
  source(here::here("pipeline/05_imputation.R"))        # apply_imputation (MICE)
  source(here::here("pipeline/08_ci_brier_cluster_bootstrap.R"))  # compute_brier_cis (cluster bootstrap)
  library(dplyr); library(tidyr); library(readr); library(tibble); library(purrr); library(ggplot2)
})

# =============================================================================
# 1. CONFIG
# =============================================================================

CMP_CFG <- list(
  outcome_col = "statistical_success",
  group_var   = "doi_o",
  health_col  = "health_related_big",
  id_cols     = c("entry_id", "claim_id", "doi_o", "doi_r"),
  seed        = if (exists("CONFIG") && !is.null(CONFIG$seed)) CONFIG$seed else 42L,

  predictor_sets   = c("transferable_lean", "transferable_lean_indicators", "transferable_lean_plus", "stable_top"),        # fixed, per decision
  # Classical ML models on the predictor_set (must exist in MODEL_GRID of 06_models.R).
  feature_models  = c("ElNet_1se", "GAM", "Logistic", "Ridge_min"),

  # ---- imputation ----
  compare_imputations = FALSE,                  # TRUE: each feature/bayes candidate x {median, mice}
  mice_m     = 10L,
  mice_maxit = 25L,

  # ---- folds ----
  n_health_folds = 5L,

  # ---- greedy ensemble (classical ML only: the feature_models above) ----
  run_greedy           = TRUE,
  greedy_cor_threshold = 0.7,                   # |cor| ceiling for adding a member

  # ---- bayes ----
  run_bayes        = TRUE,
  bayes_in_M4      = FALSE,                      # train_on_training mode for bayes (cost; off-design)
  bayes_src_path   = here::here("bayesian_priors/bayes_llmprior_model.R"),
  bayes_prior_rds  = here::here("bayesian_priors/priors_by_model.rds"),   # from extract_llm_priors.R; $pooled used
  bayes_chains = 4L, bayes_iter = 2000L, bayes_warmup = 1000L,

  # ---- prior_shrinkage ----
  # Features for the labelled OOF eval (claim_id, prior_3fam, p_base) — labelled
  # health claims only. 
  prior_features_path = here::here("output/prior_features_for_harness.csv"),
  prior_shrinkage_r3  = list(path = here::here("output/submission.csv"),
                             id_col = "claim_id", score_col = "m2"),

  # ---- deployed m1 export ----
  # The locked m1 (ElNet+) is the calibrated R3 column of this candidate. It is
  # exported on its own to output/r3_elnet_plus_predictor.csv (claim_id, pred),
  # which is exactly what 10_submission.R reads as the m1 column.
  deploy_m1_key  = "feat_ElNet_1se_transferable_lean_plus_median",
  deploy_m1_path = here::here("output/r3_elnet_plus_predictor.csv"),

  # ---- fixed external score candidates (not fitted; scored on labelled health
  #      AND passed to R3). E.g. the raw LLM ensemble prediction. ----
  fixed_score_sets = list(
    list(name = "LLM_raw", path = here::here("judge_llm_ensemble/ensemble_scores.csv"),
         id_col = "claim_id", score_col = "p_replication_observed")
  ),

  # ---- external R3 score sets (R3 diagnostics only; no labelled eval) ----
  external_r3_sets = list(),

  # ---- doi integrity ----
  enforce_doi_integrity = TRUE,                 # prune train rows whose doi_o is in the eval set

  # ---- diagnostics thresholds (R3 rows-to-inspect) ----
  extreme_lo = 0.10, extreme_hi = 0.90, disagree_range = 0.25, iqr_k = 1.5,
  reliability_bins = 5L,

  output_dir = here::here("output", "final comparison")
)

dir.create(CMP_CFG$output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(CMP_CFG$seed)

TS <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_path <- function(name, ext) file.path(CMP_CFG$output_dir, paste0(name, "_", TS, ".", ext))

# pROC for AUC/ROC 
.have_proc <- requireNamespace("pROC", quietly = TRUE)
if (!.have_proc) {
  tryCatch({ install.packages("pROC"); .have_proc <- requireNamespace("pROC", quietly = TRUE) },
           error = function(e) message("[final] pROC install failed; using rank-based AUC fallback."))
}

# =============================================================================
# 2. HELPERS
# =============================================================================

# Group-aware, outcome/dataset-stratified k-fold: every doi_o stays in one fold.
make_grouped_strat_folds <- function(df, group_var, strat_vars, n_folds, seed) {
  groups <- as.character(df[[group_var]])
  na_idx <- is.na(groups) | nchar(groups) == 0L
  if (any(na_idx)) groups[na_idx] <- paste0(".na_", seq_len(sum(na_idx)))
  aux <- tibble::tibble(.grp = groups)
  for (sv in strat_vars) {
    aux[[sv]] <- as.character(df[[sv]]); aux[[sv]][is.na(aux[[sv]])] <- "_NA_"
  }
  group_strata <- aux |>
    dplyr::group_by(.grp) |>
    dplyr::summarise(dplyr::across(dplyr::all_of(strat_vars), ~ {
      tab <- sort(table(.x), decreasing = TRUE); names(tab)[1] }), .groups = "drop") |>
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

# Remove TRAIN rows whose group also appears in EVAL (guarantees no doi_o spans
# the boundary). Eval rows are never touched. Returns the pruned training frame.
prune_train_doi <- function(train_df, eval_df, group_var) {
  if (!isTRUE(CMP_CFG$enforce_doi_integrity)) return(train_df)
  eval_groups <- unique(as.character(eval_df[[group_var]]))
  keep <- !(as.character(train_df[[group_var]]) %in% eval_groups)
  n_drop <- sum(!keep)
  if (n_drop > 0L)
    message(sprintf("  [doi-integrity] pruned %d train rows sharing a doi_o with the eval set.", n_drop))
  train_df[keep, , drop = FALSE]
}

# Median fill using TRAIN medians only; applied to train and eval.
median_impute_pair <- function(train_df, eval_df, predictors) {
  preds <- intersect(predictors, intersect(names(train_df), names(eval_df)))
  meds  <- vapply(preds, function(p) stats::median(as.numeric(train_df[[p]]), na.rm = TRUE), numeric(1))
  preds <- setdiff(preds, preds[is.na(meds)]); meds <- meds[preds]
  tr <- train_df; ev <- eval_df
  for (p in preds) {
    xt <- as.numeric(tr[[p]]); xe <- as.numeric(ev[[p]])
    xt[is.na(xt)] <- meds[[p]]; xe[is.na(xe)] <- meds[[p]]
    tr[[p]] <- xt; ev[[p]] <- xe
  }
  list(train = tr, eval = ev, predictors = preds)
}

# MICE imputation (averaged over m completed copies at predict time). Falls back
# to median on failure. Returns lists of completed train/eval frames.
mice_impute_pair_local <- function(train_df, eval_df, predictors, pset_name) {
  PREDICTOR_SETS[[pset_name]] <<- predictors            # register for apply_imputation()
  imp <- tryCatch(apply_imputation(train_df, eval_df, pset_name,
                                   m = CMP_CFG$mice_m, maxit = CMP_CFG$mice_maxit),
    error = function(e) { warning("[final] MICE failed (", pset_name, "): ", e$message); NULL })
  if (!is.null(imp) && isTRUE(imp$is_imputed))
    return(list(train = imp$train, eval = imp$test,
                predictors = intersect(predictors, names(imp$train[[1]]))))
  r <- median_impute_pair(train_df, eval_df, predictors)
  list(train = list(r$train), eval = list(r$eval), predictors = r$predictors)
}

# Rank-based AUC with a pROC primary path; NA if a class is absent.
auc_safe <- function(pred, y) {
  ok <- is.finite(pred) & is.finite(y)
  p <- pred[ok]; yy <- as.numeric(y[ok])
  if (length(unique(round(yy))) < 2L || length(p) < 5L) return(NA_real_)
  if (.have_proc) {
    a <- tryCatch(as.numeric(pROC::auc(pROC::roc(yy, p, quiet = TRUE, direction = "<"))),
                  error = function(e) NA_real_)
    if (is.finite(a)) return(a)
  }
  r <- rank(p); n1 <- sum(yy == 1); n0 <- sum(yy == 0)        # Mann-Whitney fallback
  (sum(r[yy == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

brier_mask <- function(pred, y, mask) {
  ok <- is.finite(pred) & is.finite(y) & mask
  if (sum(ok) < 3L) return(NA_real_)
  mean((pred[ok] - y[ok])^2)
}
n_mask <- function(pred, y, mask) sum(is.finite(pred) & is.finite(y) & mask)

# Nested temperature over folds: T for fold k fit on the other folds (leak-free).
nested_temperature_oof <- function(pred, fold, y) {
  out <- pred; folds <- sort(unique(fold[!is.na(fold)]))
  for (k in folds) {
    other <- which(fold != k & is.finite(pred) & is.finite(y))
    own   <- which(fold == k & is.finite(pred))
    if (length(other) < .CALIB_MIN_N) next
    cal <- fit_calibrator(pred[other], y[other], method = "temperature")
    out[own] <- apply_calibrator(cal, pred[own])
  }
  out
}

# Binned reliability table + Expected Calibration Error on a masked subset.
reliability_table <- function(pred, y, mask, bins) {
  ok <- is.finite(pred) & is.finite(y) & mask
  p <- pred[ok]; yy <- y[ok]
  if (length(p) < bins) return(NULL)
  br <- cut(p, breaks = seq(0, 1, length.out = bins + 1L), include.lowest = TRUE)
  tibble::tibble(bin = br, p = p, y = yy) |>
    dplyr::group_by(bin) |>
    dplyr::summarise(n = dplyr::n(), mean_pred = mean(p), obs_freq = mean(y), .groups = "drop") |>
    dplyr::mutate(gap = mean_pred - obs_freq)
}
ece_from_table <- function(tbl) if (is.null(tbl)) NA_real_ else sum(tbl$n / sum(tbl$n) * abs(tbl$gap))

pick_id_col <- function(df) {
  cand <- intersect(c("claim_id", "entry_id", "doi_r", "doi_o", "id"), names(df))
  if (length(cand) == 0L) stop("[final] no id column found in: ", paste(names(df), collapse = ", "))
  cand[1]
}

# =============================================================================
# 3. DATA
# =============================================================================

message("[final] loading data ...")
train_base <- readRDS(here::here("data/train_base.rds"))
test_base  <- readRDS(here::here("data/test_base.rds"))

# Bayesian model functions + pooled prior 
if (isTRUE(CMP_CFG$run_bayes)) {
  if (!file.exists(CMP_CFG$bayes_src_path))
    stop("[final] bayes source not found: ", CMP_CFG$bayes_src_path)
  source(CMP_CFG$bayes_src_path)                 # add_log1p_feature, fit/pred_bayes_logistic_llmprior, PREDICTORS_BAYES
  if (!file.exists(CMP_CFG$bayes_prior_rds))
    stop("[final] priors_by_model.rds not found: ", CMP_CFG$bayes_prior_rds,
         " — run extract_llm_priors.R first (it writes the pooled prior).")
  .priors_by_model <- readRDS(CMP_CFG$bayes_prior_rds)
  if (is.null(.priors_by_model$pooled))
    stop("[final] priors_by_model.rds has no $pooled entry.")
  BAYES_PRIOR <- .priors_by_model$pooled         # list(location, scale) named by PREDICTORS_BAYES
}

# n_f_tests_tei_log1p is a derived feature the bayes model needs
if (exists("add_log1p_feature")) {
  train_base <- add_log1p_feature(train_base)
  test_base  <- add_log1p_feature(test_base)
}

gt_pool <- train_base |>
  dplyr::filter(!is.na(.data[[CMP_CFG$outcome_col]]),
                dataset %in% c("training", "round1", "round2"))
if ("is_boyce_soto" %in% names(gt_pool))
  gt_pool <- gt_pool |> dplyr::filter(is.na(is_boyce_soto) | !is_boyce_soto)
gt_pool$is_health_strat <- as.integer(
  !is.na(gt_pool[[CMP_CFG$health_col]]) &
  (gt_pool[[CMP_CFG$health_col]] == 1 | gt_pool[[CMP_CFG$health_col]] == TRUE))

# Prior-shrinkage features (health rows only). Merged onto the pool; coverage on
# R3 checked separately for the R3-score step.
prior_feats <- NULL
if (file.exists(CMP_CFG$prior_features_path)) {
  prior_feats <- readr::read_csv(CMP_CFG$prior_features_path, show_col_types = FALSE)
  need <- c("claim_id", "prior_3fam", "p_base")
  if (!all(need %in% names(prior_feats)))
    stop("[final] prior_features file must have columns: ", paste(need, collapse = ", "))
  gt_pool <- dplyr::left_join(gt_pool, prior_feats[, need], by = "claim_id")
  message(sprintf("[final] prior features merged: %d / %d pool rows have prior_3fam.",
                  sum(!is.na(gt_pool$prior_3fam)), nrow(gt_pool)))
} else {
  warning("[final] prior_features file missing: ", CMP_CFG$prior_features_path,
          " — prior_shrinkage candidate disabled.")
}
have_prior_shrinkage <- !is.null(prior_feats) && all(c("prior_3fam", "p_base") %in% names(gt_pool))

# Predictor-set validation: each set must have ≥2 usable columns.
get_predictors_for_pset <- function(pset, df1, df2) {
  intersect(PREDICTOR_SETS[[pset]], intersect(names(df1), names(df2)))
}

for (.ps in CMP_CFG$predictor_sets) {
  .preds <- get_predictors_for_pset(.ps, gt_pool, test_base)
  if (length(.preds) < 2L)
    stop("[final] predictor set '", .ps, "' has <2 usable columns (",
         length(.preds), " found).")
  message(sprintf("[final] predictor set '%s': %d usable columns.", .ps, length(.preds)))
}
# Health universe (the common row space for all labelled evals) + its folds.
hth <- gt_pool |> dplyr::filter(is_health_strat == 1)
nonh <- gt_pool |> dplyr::filter(is_health_strat == 0)
y_h  <- as.numeric(hth[[CMP_CFG$outcome_col]])
mask_chal     <- hth$dataset %in% c("round1", "round2")     # 47 challenge-health
mask_forrt    <- hth$dataset == "training"                  # FORRT-health
mask_big      <- rep(TRUE, nrow(hth))                       # 107 health_big
hth$hfold <- make_grouped_strat_folds(hth, CMP_CFG$group_var, "dataset",
                                      CMP_CFG$n_health_folds, CMP_CFG$seed)
message(sprintf("[final] health=%d (chal=%d, forrt=%d) | non-health=%d | predictor_sets=%s",
                nrow(hth), sum(mask_chal), sum(mask_forrt), nrow(nonh),
                paste(CMP_CFG$predictor_sets, collapse = ", ")))

# Fixed external score candidates (e.g. raw LLM ensemble): not fitted, just looked
# up by claim_id. Attached as columns fx_<name> on the health frame (labelled eval)
# and on test_base (R3).
fixed_specs <- list()
for (fs in CMP_CFG$fixed_score_sets) {
  if (!file.exists(fs$path)) { warning("[final] fixed-score file missing: ", fs$path); next }
  df <- readr::read_csv(fs$path, show_col_types = FALSE)
  if (!all(c(fs$id_col, fs$score_col) %in% names(df))) {
    warning("[final] fixed-score '", fs$name, "' lacks columns ", fs$id_col, "/", fs$score_col); next }
  lk <- stats::setNames(as.numeric(df[[fs$score_col]]), as.character(df[[fs$id_col]]))
  lk <- lk[!duplicated(names(lk))]
  col <- paste0("fx_", fs$name)
  hth[[col]]       <- unname(lk[as.character(hth$claim_id)])
  test_base[[col]] <- unname(lk[as.character(test_base$claim_id)])
  fixed_specs[[length(fixed_specs) + 1L]] <- list(
    key = fs$name, type = "fixed", model = fs$name, imputation = "none", fx_col = col)
  message(sprintf("[final] fixed-score '%s': %d / %d health rows, %d / %d R3 rows have scores.",
                  fs$name, sum(!is.na(hth[[col]])), nrow(hth),
                  sum(!is.na(test_base[[col]])), nrow(test_base)))
}

# =============================================================================
# 4. CANDIDATE REGISTRY
# =============================================================================
# Each candidate is a (type, model, imputation) spec. greedy is assembled later.

imputations <- if (isTRUE(CMP_CFG$compare_imputations)) c("median", "mice") else "median"

candidates <- list()

# ---- feature candidates: one per (predictor_set × imputation × model) ----
for (ps in CMP_CFG$predictor_sets) {
  for (im in imputations) {
    for (mdl in CMP_CFG$feature_models) {
      candidates[[length(candidates) + 1L]] <- list(
        key        = paste0("feat_", mdl, "_", ps, "_", im),
        type       = "feature",
        model      = mdl,
        imputation = im,
        pset       = ps
      )
    }
  }
}

# ---- bayes: one per imputation (uses its own PREDICTORS_BAYES, not pset) ----
if (isTRUE(CMP_CFG$run_bayes)) {
  for (im in imputations) {
    candidates[[length(candidates) + 1L]] <- list(
      key        = paste0("bayes_pooled_", im),
      type       = "bayes",
      model      = "BayesPooled",
      imputation = im
    )
  }
}

# ---- prior_shrinkage: single candidate (own features) ----
if (have_prior_shrinkage)
  candidates[[length(candidates) + 1L]] <- list(
    key = "prior_shrinkage", type = "prior_shrinkage",
    model = "PriorShrinkage", imputation = "none")

# ---- fixed external scores (e.g. LLM_raw) ----
for (sp in fixed_specs)
  candidates[[length(candidates) + 1L]] <- sp

cand_keys <- vapply(candidates, function(c0) c0$key, character(1))
message("[final] base candidates (", length(cand_keys), "): ",
        paste(cand_keys, collapse = ", "))

# ---- one fit/predict unit per candidate type --------------------------------
# Returns predictions aligned to eval_df rows (NA where undefined). 

predict_feature <- function(model, imputation, train_df, eval_df, pset) {
  # Resolve predictors from the candidate's own predictor set
  predictors <- get_predictors_for_pset(pset, train_df, eval_df)

  if (imputation == "mice") {
    imp <- mice_impute_pair_local(train_df, eval_df, predictors, pset)
    preds <- imp$predictors; tr_l <- imp$train; ev_l <- imp$eval
  } else {
    r <- median_impute_pair(train_df, eval_df, predictors)
    preds <- r$predictors; tr_l <- list(r$train); ev_l <- list(r$eval)
  }
  if (length(preds) < 2L) return(rep(NA_real_, nrow(eval_df)))

  mg_idx <- which(vapply(MODEL_GRID, function(e) e$name == model, logical(1)))
  if (length(mg_idx) == 0L) {
    warning("[final] model '", model, "' not found in MODEL_GRID.")
    return(rep(NA_real_, nrow(eval_df)))
  }
  mg <- MODEL_GRID[[mg_idx]]
  acc <- numeric(nrow(eval_df)); n_ok <- 0L
  for (i in seq_along(tr_l)) {
    p <- tryCatch(
      mg$pred(mg$fit(tr_l[[i]], CMP_CFG$outcome_col, preds, weights = NULL),
              ev_l[[i]], preds),
      error = function(e) {
        warning("[final] ", model, " (", pset, "): ", e$message)
        rep(NA_real_, nrow(eval_df))
      })
    if (!all(is.na(p))) { acc <- acc + p; n_ok <- n_ok + 1L }
  }
  if (n_ok == 0L) rep(NA_real_, nrow(eval_df)) else acc / n_ok
}

# prior_shrinkage: logistic on prior_3fam + p_base, fit on the train
# rows that carry both features (health rows). NA where eval lacks the features.
predict_prior_shrinkage <- function(train_df, eval_df) {
  out <- rep(NA_real_, nrow(eval_df))
  trh <- train_df[!is.na(train_df$prior_3fam) & !is.na(train_df$p_base), , drop = FALSE]
  if (nrow(trh) < 5L) return(out)
  trh[[".ps_y"]] <- as.integer(trh[[CMP_CFG$outcome_col]])
  fit <- tryCatch(stats::glm(stats::reformulate(c("prior_3fam", "p_base"), ".ps_y"),
                             data = trh, family = stats::binomial()),
                  error = function(e) { warning("[final] prior_shrinkage: ", e$message); NULL })
  if (is.null(fit)) return(out)
  ev_ok <- !is.na(eval_df$prior_3fam) & !is.na(eval_df$p_base)
  if (any(ev_ok))
    out[ev_ok] <- stats::predict(fit, eval_df[ev_ok, , drop = FALSE], type = "response")
  out
}

# Bayesian logistic with pooled LLM priors. MICE pre-imputes the raw set, then
# log1p is (re)derived and impute_method="none" is passed; median path lets the
# model impute internally.
predict_bayes <- function(imputation, train_df, eval_df) {
  tr <- train_df; ev <- eval_df; impute_method <- "median"
  if (imputation == "mice") {
    imp <- tryCatch(apply_imputation(train_df, eval_df, "bayes_pooled",
                                     m = 1L, maxit = CMP_CFG$mice_maxit),
                    error = function(e) NULL)
    if (!is.null(imp) && isTRUE(imp$is_imputed)) {
      tr <- add_log1p_feature(imp$train[[1]])
      ev <- add_log1p_feature(imp$test[[1]])
      impute_method <- "none"
    }
  }
  fit <- tryCatch(
    fit_bayes_logistic_llmprior(tr, CMP_CFG$outcome_col, PREDICTORS_BAYES,
                                BAYES_PRIOR$location, BAYES_PRIOR$scale,
                                binary_predictors = BINARY_PREDICTORS_BAYES,
                                impute_method = impute_method, weights = NULL,
                                chains = CMP_CFG$bayes_chains, iter = CMP_CFG$bayes_iter,
                                warmup = CMP_CFG$bayes_warmup, seed = CMP_CFG$seed),
    error = function(e) { warning("[final] bayes fit: ", e$message); NULL })
  if (is.null(fit)) return(rep(NA_real_, nrow(eval_df)))
  tryCatch(pred_bayes_logistic_llmprior(fit, ev),
           error = function(e) { warning("[final] bayes pred: ", e$message); rep(NA_real_, nrow(eval_df)) })
}

# Dispatch: build the train slice for a candidate type and predict eval rows.
# `train_pool` is the rows allowed to train on for this mode/fold; feature models
# use it as-is, bayes/prior restrict to its health rows (on-design).
predict_candidate <- function(spec, train_pool, eval_df) {
  if (spec$type == "fixed") return(as.numeric(eval_df[[spec$fx_col]]))
  train_pool <- prune_train_doi(train_pool, eval_df, CMP_CFG$group_var)
  if (spec$type %in% c("bayes", "prior_shrinkage")) {
    train_pool <- train_pool[train_pool$is_health_strat == 1, , drop = FALSE]
    if (nrow(train_pool) < 10L) {
      message(sprintf("  [%s] only %d health train rows — undefined, returning NA.",
                      spec$type, nrow(train_pool)))
      return(rep(NA_real_, nrow(eval_df)))
    }
  }
  switch(spec$type,
         "feature"         = predict_feature(spec$model, spec$imputation,
                                             train_pool, eval_df, spec$pset),
         "prior_shrinkage" = predict_prior_shrinkage(train_pool, eval_df),
         "bayes"           = predict_bayes(spec$imputation, train_pool, eval_df),
         stop("unknown candidate type: ", spec$type))
}

# =============================================================================
# 5. EVAL MODES  -> per-candidate predictions over the health universe
# =============================================================================
# Each mode returns, per candidate key, a numeric vector of length nrow(hth) with
# leak-free predictions on the rows that mode evaluates (NA elsewhere), plus the
# mask of evaluated rows and which targets it supports.

empty_pred <- function() stats::setNames(lapply(cand_keys, function(k) rep(NA_real_, nrow(hth))), cand_keys)

# ---- M1: common-fold health OOF --------------------------------------------
run_M1 <- function() {
  preds <- empty_pred()
  for (k in sort(unique(hth$hfold))) {
    ev_idx  <- which(hth$hfold == k)
    oth_idx <- which(hth$hfold != k)
    eval_df <- hth[ev_idx, , drop = FALSE]
    for (spec in candidates) {
      # feature: non-health + other folds; bayes/prior: other folds only
      train_pool <- if (spec$type == "feature")
        dplyr::bind_rows(nonh, hth[oth_idx, , drop = FALSE])
      else hth[oth_idx, , drop = FALSE]
      preds[[spec$key]][ev_idx] <- predict_candidate(spec, train_pool, eval_df)
    }
    message(sprintf("[final] M1 fold %d/%d done.", k, max(hth$hfold)))
  }
  list(preds = preds, eval_mask = mask_big, targets = c("chal_health", "health_big"))
}

# ---- M2: leave only challenge-health out ------------------------------------
run_M2 <- function() {
  preds <- empty_pred()
  eval_df <- hth[mask_chal, , drop = FALSE]
  ev_idx  <- which(mask_chal)
  for (spec in candidates) {
    train_pool <- if (spec$type == "feature")
      dplyr::bind_rows(nonh, hth[mask_forrt, , drop = FALSE])
    else hth[mask_forrt, , drop = FALSE]
    preds[[spec$key]][ev_idx] <- predict_candidate(spec, train_pool, eval_df)
  }
  list(preds = preds, eval_mask = mask_chal, targets = "chal_health")
}

# ---- M3: leave ALL health out (train = non-health) --------------------------
run_M3 <- function() {
  preds <- empty_pred()
  eval_df <- hth                                   # all health predicted
  for (spec in candidates) {
    if (spec$type == "prior_shrinkage") next       # undefined: non-health has no prior feats
    preds[[spec$key]] <- predict_candidate(spec, nonh, eval_df)
  }
  list(preds = preds, eval_mask = mask_big, targets = c("chal_health", "health_big"))
}

# ---- M4: train on dataset=="training", predict !=training (chal target) -----
run_M4 <- function() {
  preds <- empty_pred()
  train_pool <- gt_pool |> dplyr::filter(dataset == "training")
  eval_df <- hth[mask_chal, , drop = FALSE]; ev_idx <- which(mask_chal)
  for (spec in candidates) {
    if (spec$type == "bayes" && !isTRUE(CMP_CFG$bayes_in_M4)) next   # cost / off-design
    preds[[spec$key]][ev_idx] <- predict_candidate(spec, train_pool, eval_df)
  }
  list(preds = preds, eval_mask = mask_chal, targets = "chal_health")
}

message("[final] ── running eval modes ──")
modes <- list(M1_oof_health = run_M1(), M2_leave_chal_out = run_M2(),
              M3_leave_all_health = run_M3(), M4_train_on_training = run_M4())

# =============================================================================
# 6. GREEDY ENSEMBLE (leak-free, re-derived per M1 fold)
# =============================================================================
# Members ranked by Brier on the OTHER folds, added greedily while |cor| with all
# already-selected (on the other folds) stays < threshold. The fold-k prediction
# is the equal-weight mean of the selected members on fold k. Selection never
# sees fold k's labels. For the single-split modes (M2/M3/M4) greedy is reported
# as the equal-weight mean of the candidates defined there (no data-driven
# selection -> still leak-free; annotated as ens_equal in those modes).

greedy_members_by_fold <- list()
if (isTRUE(CMP_CFG$run_greedy)) {
  m1 <- modes$M1_oof_health$preds
  # classical ML only: bayes / prior_shrinkage / fixed (LLM) are excluded
  base_for_greedy <- vapply(candidates, function(s) s$key, character(1))[
    vapply(candidates, function(s) s$type == "feature", logical(1))]
  greedy_oof <- rep(NA_real_, nrow(hth))
  for (k in sort(unique(hth$hfold))) {
    own   <- which(hth$hfold == k)
    other <- which(hth$hfold != k)
    # rank by Brier on the other folds
    br <- vapply(base_for_greedy, function(key)
      brier_mask(m1[[key]], y_h, hth$hfold != k), numeric(1))
    ranked <- names(sort(br[is.finite(br)]))
    selected <- character(0)
    for (key in ranked) {
      if (length(selected) == 0L) { selected <- key; next }
      max_cor <- max(vapply(selected, function(s) {
        ok <- (hth$hfold != k) & is.finite(m1[[key]]) & is.finite(m1[[s]])
        if (sum(ok) < 10L) return(0)
        abs(stats::cor(m1[[key]][ok], m1[[s]][ok]))
      }, numeric(1)), na.rm = TRUE)
      if (max_cor < CMP_CFG$greedy_cor_threshold) selected <- c(selected, key)
    }
    if (length(selected) == 0L && length(ranked) > 0L) selected <- ranked[1L]
    greedy_members_by_fold[[as.character(k)]] <- selected
    mat <- do.call(cbind, lapply(selected, function(key) m1[[key]][own]))
    greedy_oof[own] <- rowMeans(mat, na.rm = TRUE)
  }
  modes$M1_oof_health$preds[["greedy_ens"]] <- greedy_oof

  # ens_equal in M2/M3/M4 (mean over the classical-ML members defined there)
  for (mn in c("M2_leave_chal_out", "M3_leave_all_health", "M4_train_on_training")) {
    pk <- modes[[mn]]$preds
    defined <- base_for_greedy[vapply(base_for_greedy,
                 function(key) any(is.finite(pk[[key]])), logical(1))]
    if (length(defined) >= 2L) {
      mat <- do.call(cbind, lapply(defined, function(key) pk[[key]]))
      modes[[mn]]$preds[["greedy_ens"]] <- rowMeans(mat, na.rm = TRUE)
    }
  }
  cand_keys <- c(cand_keys, "greedy_ens")

  # report selected members
  gm <- tibble::tibble(fold = names(greedy_members_by_fold),
    members = vapply(greedy_members_by_fold, function(x) paste(x, collapse = "+"), character(1)))
  readr::write_csv(gm, out_path("greedy_members", "csv"))
  message("[final] greedy members per fold:"); print(gm)
}

# =============================================================================
# 7. METRICS TABLE  (per candidate x mode x target)
# =============================================================================
# brier_raw, brier_cal (deployment T from the M1 OOF of that candidate), AUC,
# and for M1 additionally nested-cal Brier + ECE.

target_mask <- list(chal_health = mask_chal, health_big = mask_big)

# Deployment T per candidate from its M1 OOF (NA if M1 missing / too few pairs).
m1_pred <- modes$M1_oof_health$preds
T_deploy <- stats::setNames(rep(NA_real_, length(cand_keys)), cand_keys)
for (key in cand_keys) {
  p <- m1_pred[[key]]; ok <- is.finite(p) & is.finite(y_h)
  if (sum(ok) >= .CALIB_MIN_N && length(unique(round(y_h[ok]))) == 2L)
    T_deploy[key] <- temperature_scaling(p[ok], y_h[ok])$T
}
apply_T <- function(p, key) {
  Tk <- T_deploy[[key]]
  if (is.na(Tk)) return(p)
  plogis(qlogis(.clip01(p)) / Tk)
}

metric_rows <- list()
for (mn in names(modes)) {
  mode <- modes[[mn]]
  for (key in cand_keys) {
    p <- mode$preds[[key]]
    if (is.null(p) || all(is.na(p))) next
    p_calOOFT <- apply_T(p, key)
    p_nested  <- if (mn == "M1_oof_health") nested_temperature_oof(p, hth$hfold, y_h) else NULL
    for (tg in mode$targets) {
      msk <- target_mask[[tg]]
      if (n_mask(p, y_h, msk) < 3L) next
      tbl_raw <- if (mn == "M1_oof_health")
        reliability_table(p, y_h, msk, CMP_CFG$reliability_bins) else NULL
      tbl_nst <- if (mn == "M1_oof_health")
        reliability_table(p_nested, y_h, msk, CMP_CFG$reliability_bins) else NULL
      metric_rows[[length(metric_rows) + 1L]] <- tibble::tibble(
        candidate = key, mode = mn, target = tg,
        n = n_mask(p, y_h, msk),
        brier_raw      = brier_mask(p, y_h, msk),
        brier_cal_oofT = brier_mask(p_calOOFT, y_h, msk),
        brier_cal_nested = if (mn == "M1_oof_health") brier_mask(p_nested, y_h, msk) else NA_real_,
        auc            = auc_safe(p[msk], y_h[msk]),
        ece_raw        = ece_from_table(tbl_raw),
        ece_cal_nested = ece_from_table(tbl_nst),
        T_deploy       = T_deploy[[key]])
    }
  }
}
metrics <- dplyr::bind_rows(metric_rows) |>
  dplyr::arrange(target, mode, brier_cal_oofT)

# ---- bootstrap CIs (sampling-only, doi_o-clustered) for the M1 decision metric ----
# Marginal CIs joined into the metrics CSV (raw + nested-cal); paired Brier-difference
# CIs written separately. Predictions are held fixed -> reflects only finite-sample
# noise over the evaluation rows. Set ci_keys <- cand_keys to cover every candidate.
ci_keys <- intersect(c("prior_shrinkage", "LLM_raw", "bayes_pooled_median",
                       "feat_ElNet_1se_transferable_lean_median",
                       "feat_ElNet_1se_transferable_lean_plus_median",
                       "greedy_ens"), cand_keys)
ci_keys <- ci_keys[vapply(ci_keys, function(k)
  !is.null(m1_pred[[k]]) && any(is.finite(m1_pred[[k]])), logical(1))]
ci_ref  <- if ("bayes_pooled_median" %in% ci_keys) "bayes_pooled_median" else ci_keys[1]
ci_doi  <- hth[[CMP_CFG$group_var]]

ci_marg_rows <- list(); ci_pair_rows <- list()
for (tg in names(target_mask)) {
  msk      <- target_mask[[tg]]
  raw_list <- stats::setNames(lapply(ci_keys, function(k) m1_pred[[k]]), ci_keys)
  nst_list <- stats::setNames(lapply(ci_keys, function(k)
    nested_temperature_oof(m1_pred[[k]], hth$hfold, y_h)), ci_keys)
  ci_n <- compute_brier_cis(nst_list, y_h, ci_doi, msk, ref = ci_ref)
  ci_r <- compute_brier_cis(raw_list, y_h, ci_doi, msk, ref = ci_ref)
  ci_marg_rows[[tg]] <- tibble::tibble(
    candidate = ci_n$marginal$candidate, mode = "M1_oof_health", target = tg,
    brier_cal_nested_lo = ci_n$marginal$ci_lo, brier_cal_nested_hi = ci_n$marginal$ci_hi,
    brier_raw_lo = ci_r$marginal$ci_lo,        brier_raw_hi = ci_r$marginal$ci_hi)
  ci_pair_rows[[tg]] <- dplyr::bind_rows(
    dplyr::mutate(ci_n$paired, target = tg, metric = "brier_cal_nested"),
    dplyr::mutate(ci_r$paired, target = tg, metric = "brier_raw"))
}
metrics <- dplyr::left_join(metrics, dplyr::bind_rows(ci_marg_rows),
                            by = c("candidate", "mode", "target"))
readr::write_csv(dplyr::bind_rows(ci_pair_rows), out_path("brier_ci_paired", "csv"))

readr::write_csv(metrics, out_path("comparison_metrics", "csv"))
cat("\n── comparison metrics (sorted within target/mode by brier_cal_oofT) ──\n")
print(metrics |> dplyr::filter(target == "chal_health"))

# =============================================================================
# 8. R3 CONFIDENCE SCORES  (one set per candidate)
# =============================================================================
# Each labelled candidate is fit on its full deployment scope, predicts R3, and
# is calibrated with the deployment T from its M1 OOF. PPV / external sets are
# merged in as-is (diagnostics only).

r3_id <- if ("claim_id" %in% names(test_base)) "claim_id" else pick_id_col(test_base)
id_present <- intersect(CMP_CFG$id_cols, names(test_base))
r3 <- test_base[, id_present, drop = FALSE]
r3[[".id"]] <- as.character(test_base[[r3_id]])

# prior_shrinkage R3 coverage: merge prior feats onto R3 by claim_id.
test_ps <- test_base
if (have_prior_shrinkage && "claim_id" %in% names(test_base)) {
  test_ps <- dplyr::left_join(test_base, prior_feats[, c("claim_id", "prior_3fam", "p_base")],
                              by = "claim_id")
  cov <- sum(!is.na(test_ps$prior_3fam))
  if (cov == 0L)
    warning("[final] prior_shrinkage: 0 of ", nrow(test_ps),
            " R3 rows have prior features — no R3 slot possible for it (NA scores).")
  else message(sprintf("[final] prior_shrinkage R3 coverage: %d / %d rows.", cov, nrow(test_ps)))
} else if (have_prior_shrinkage) {
  warning("[final] test_base has no 'claim_id' — cannot match prior features to R3 (prior_shrinkage R3 = NA).")
}

# prior_shrinkage R3 scores are the already-produced m2 confidence scores
ps_r3_lookup <- NULL
psr <- CMP_CFG$prior_shrinkage_r3
if (have_prior_shrinkage && !is.null(psr$path) && !is.null(psr$score_col) && file.exists(psr$path)) {
  pdf <- readr::read_csv(psr$path, show_col_types = FALSE)
  if (all(c(psr$id_col, psr$score_col) %in% names(pdf))) {
    ps_r3_lookup <- stats::setNames(as.numeric(pdf[[psr$score_col]]), as.character(pdf[[psr$id_col]]))
    ps_r3_lookup <- ps_r3_lookup[!duplicated(names(ps_r3_lookup))]
    message(sprintf("[final] prior_shrinkage R3 scores loaded: %d / %d R3 rows matched.",
                    sum(as.character(test_base$claim_id) %in% names(ps_r3_lookup)), nrow(test_base)))
  } else warning("[final] prior_shrinkage_r3 file lacks ", psr$id_col, "/", psr$score_col, ".")
} else if (have_prior_shrinkage) {
  message("[final] no prior_shrinkage_r3 path set — prior_shrinkage R3 scores will be NA. ",
          "Point CMP_CFG$prior_shrinkage_r3 at the m2 R3 score file to fill them.")
}

r3_raw_cols <- list(); r3_cal_cols <- list()
for (spec in candidates) {
  key <- spec$key; external_final <- FALSE
  if (spec$type == "prior_shrinkage" && !is.null(ps_r3_lookup)) {
    raw <- unname(ps_r3_lookup[as.character(test_base$claim_id)]); external_final <- TRUE
  } else {
    raw <- switch(spec$type,
                  "feature"         = predict_feature(spec$model, spec$imputation,
                                                      gt_pool, test_base, spec$pset),
      "fixed"           = as.numeric(test_base[[spec$fx_col]]),
      "prior_shrinkage" = predict_prior_shrinkage(
                            gt_pool[gt_pool$is_health_strat == 1, , drop = FALSE], test_ps),
      "bayes"           = predict_bayes(spec$imputation,
                            gt_pool[gt_pool$is_health_strat == 1 &
                                    gt_pool$dataset %in% c("round1", "round2"), , drop = FALSE],
                            test_base))    # bayes deploys on its on-design fit_scope = chal-health(47)
  }
  r3_raw_cols[[key]] <- raw
  # External final scores (prior_shrinkage m2) are used as-is; everything
  # else gets the deployment T from its labelled OOF.
  r3_cal_cols[[key]] <- if (external_final) raw else apply_T(raw, key)
}
# greedy_ens R3: mean of RAW member R3, then the greedy deployment T — matching
# how greedy's M1-OOF Brier is computed (raw member mean -> T_deploy[greedy_ens]).
if (isTRUE(CMP_CFG$run_greedy) && length(greedy_members_by_fold) > 0L) {
  members <- intersect(unique(unlist(greedy_members_by_fold)), names(r3_raw_cols))
  if (length(members) >= 1L) {
    mat <- do.call(cbind, lapply(members, function(key) r3_raw_cols[[key]]))
    r3_raw_cols[["greedy_ens"]] <- rowMeans(mat, na.rm = TRUE)
    r3_cal_cols[["greedy_ens"]] <- apply_T(r3_raw_cols[["greedy_ens"]], "greedy_ens")
  }
}

for (key in names(r3_raw_cols)) r3[[paste0("raw_", key)]] <- r3_raw_cols[[key]]
for (key in names(r3_cal_cols)) r3[[paste0("cal_", key)]] <- r3_cal_cols[[key]]

# external R3 score sets (PPV etc.) — diagnostics only
for (es in CMP_CFG$external_r3_sets) {
  if (!file.exists(es$path)) { warning("[final] external set missing: ", es$path); next }
  df  <- readr::read_csv(es$path, show_col_types = FALSE)
  idc <- if (!is.null(es$id_col) && es$id_col %in% names(df)) es$id_col else pick_id_col(df)
  if (!es$score_col %in% names(df)) { warning("[final] score col '", es$score_col, "' not in ", es$path); next }
  jn <- tibble::tibble(.id = as.character(df[[idc]]), .v = as.numeric(df[[es$score_col]])) |>
    dplyr::distinct(.id, .keep_all = TRUE)         # one score per id (avoid row expansion)
  r3 <- dplyr::left_join(r3, jn, by = ".id")
  matched <- sum(!is.na(r3$.v))
  r3[[paste0("ext_", es$name)]] <- r3$.v; r3$.v <- NULL
  message(sprintf("[final] external set '%s': matched %d / %d R3 rows.",
                  es$name, matched, nrow(r3)))
}
r3$.id <- NULL
readr::write_csv(r3, out_path("r3_confidence_scores_all", "csv"))
message("[final] wrote R3 confidence scores for ", length(r3_cal_cols), " model candidates + external sets.")

# ---- deployed m1 export: the calibrated ElNet+ R3 column on its own ----
# Stable (non-timestamped) filename consumed directly by 10_submission.R as m1.
m1_col <- paste0("cal_", CMP_CFG$deploy_m1_key)
if (m1_col %in% names(r3)) {
  m1_out <- tibble::tibble(claim_id = as.character(r3[[r3_id]]),
                           pred     = as.numeric(r3[[m1_col]]))
  readr::write_csv(m1_out, CMP_CFG$deploy_m1_path)
  message(sprintf("[final] wrote deployed m1 (%s) → %s (%d rows).",
                  CMP_CFG$deploy_m1_key, CMP_CFG$deploy_m1_path, nrow(m1_out)))
} else {
  warning("[final] deploy_m1_key column '", m1_col,
          "' not found in R3 table — r3_elnet_plus_predictor.csv NOT updated. ",
          "Check CMP_CFG$deploy_m1_key against the candidate registry.")
}

# =============================================================================
# 9. R3 DIAGNOSTICS  (distribution + agreement + rows to inspect)
# =============================================================================

score_cols <- names(r3)[grepl("^cal_|^ext_", names(r3))]
long_r3 <- r3 |>
  dplyr::select(dplyr::all_of(c(r3_id, score_cols))) |>
  tidyr::pivot_longer(dplyr::all_of(score_cols), names_to = "set", values_to = "score")

# Columns with at least one finite value — used for correlation / disagreement so
# all-NA sets (e.g. a candidate without R3 coverage)
score_cols_cov <- score_cols[vapply(score_cols,
  function(c0) any(is.finite(r3[[c0]])), logical(1))]

q <- function(x, p) unname(stats::quantile(x, p, na.rm = TRUE))
dist_summary <- long_r3 |>
  dplyr::group_by(set) |>
  dplyr::summarise(
    n = sum(is.finite(score)), n_na = sum(!is.finite(score)),
    mean = mean(score, na.rm = TRUE), sd = stats::sd(score, na.rm = TRUE),
    min = suppressWarnings(min(score, na.rm = TRUE)), median = stats::median(score, na.rm = TRUE),
    max = suppressWarnings(max(score, na.rm = TRUE)),
    frac_lt_10 = mean(score < CMP_CFG$extreme_lo, na.rm = TRUE),
    frac_gt_90 = mean(score > CMP_CFG$extreme_hi, na.rm = TRUE),
    sharpness  = mean(abs(score - 0.5), na.rm = TRUE), .groups = "drop")
readr::write_csv(dist_summary, out_path("r3_distribution_summary", "csv"))
cat("\n── R3 score distribution per set ──\n"); print(dist_summary)

wide_r3 <- r3 |> dplyr::select(dplyr::all_of(c(r3_id, score_cols_cov)))
set_names <- score_cols_cov

agreement <- NULL
if (length(set_names) >= 2L) {
  M <- as.matrix(wide_r3[, set_names, drop = FALSE])
  pear  <- suppressWarnings(stats::cor(M, use = "pairwise.complete.obs"))
  spear <- suppressWarnings(stats::cor(M, use = "pairwise.complete.obs", method = "spearman"))
  rows <- list()
  for (i in seq_along(set_names)) for (j in seq_along(set_names)) if (i < j) {
    a <- M[, i]; b <- M[, j]; ok <- is.finite(a) & is.finite(b)
    rows[[length(rows) + 1L]] <- tibble::tibble(
      set_a = set_names[i], set_b = set_names[j],
      pearson = pear[i, j], spearman = spear[i, j],
      mean_abs_diff = if (any(ok)) mean(abs(a[ok] - b[ok])) else NA_real_)
  }
  agreement <- dplyr::bind_rows(rows)
  readr::write_csv(agreement, out_path("r3_agreement", "csv"))
}

# rows to inspect: extreme mean / cross-set disagreement / missing
row_flags <- wide_r3 |>
  dplyr::rowwise() |>
  dplyr::mutate(
    score_min  = suppressWarnings(min(dplyr::c_across(dplyr::all_of(set_names)), na.rm = TRUE)),
    score_max  = suppressWarnings(max(dplyr::c_across(dplyr::all_of(set_names)), na.rm = TRUE)),
    score_mean = mean(dplyr::c_across(dplyr::all_of(set_names)), na.rm = TRUE),
    n_na       = sum(!is.finite(dplyr::c_across(dplyr::all_of(set_names))))) |>
  dplyr::ungroup() |>
  dplyr::mutate(
    spread      = score_max - score_min,
    is_extreme  = score_mean < CMP_CFG$extreme_lo | score_mean > CMP_CFG$extreme_hi,
    is_disagree = spread > CMP_CFG$disagree_range,
    has_na      = n_na > 0L,
    reasons = purrr::pmap_chr(list(is_extreme, is_disagree, has_na), function(ex, dis, na_)
      paste(c(if (isTRUE(ex)) "extreme_mean", if (isTRUE(dis)) "set_disagreement",
              if (isTRUE(na_)) "missing_score"), collapse = "; ")))
inspect <- row_flags |>
  dplyr::filter(nzchar(reasons)) |>
  dplyr::arrange(dplyr::desc(spread), score_mean) |>
  dplyr::select(dplyr::all_of(r3_id), dplyr::all_of(set_names), score_mean, spread, n_na, reasons)
readr::write_csv(inspect, out_path("r3_rows_to_inspect", "csv"))
cat(sprintf("\n── R3 rows to inspect: %d ──\n", nrow(inspect)))

# =============================================================================
# 10. PLOTS
# =============================================================================

# (a) R3 score histograms per set (shared x and y axes across facets)
p_hist <- ggplot(long_r3 |> dplyr::filter(is.finite(score)), aes(score)) +
  geom_histogram(binwidth = 0.05, boundary = 0, fill = "#3b7dd8", colour = "white", linewidth = 0.2) +
  facet_wrap(~ set) +                                 # scales = "fixed" (default) -> same x and y
  scale_x_continuous(limits = c(0, 1)) +
  labs(title = "R3 confidence-score distributions", x = "score", y = "count") +
  theme_minimal(base_size = 11)
ggsave(out_path("r3_score_histograms", "png"), p_hist, width = 11, height = 7, dpi = 150)

# (b) R3 score correlation matrix (Pearson)
if (length(set_names) >= 2L) {
  Mc <- suppressWarnings(stats::cor(as.matrix(wide_r3[, set_names, drop = FALSE]),
                                    use = "pairwise.complete.obs"))
  grDevices::png(out_path("r3_score_correlation", "png"), width = 900, height = 800, res = 130)
  corrplot::corrplot(Mc, method = "color", type = "upper", addCoef.col = "black",
                     tl.col = "black", tl.cex = 0.7, number.cex = 0.6,
                     title = "R3 score correlations", mar = c(0, 0, 2, 0))
  grDevices::dev.off()
}

# (c) Reliability on the M1 health OOF (chal-health): raw vs nested-calibrated.
rel_rows <- list()
for (key in cand_keys) {
  p <- m1_pred[[key]]; if (is.null(p) || all(is.na(p))) next
  p_nst <- nested_temperature_oof(p, hth$hfold, y_h)
  tr_raw <- reliability_table(p, y_h, mask_chal, CMP_CFG$reliability_bins)
  tr_nst <- reliability_table(p_nst, y_h, mask_chal, CMP_CFG$reliability_bins)
  if (!is.null(tr_raw)) rel_rows[[length(rel_rows) + 1L]] <-
    tr_raw |> dplyr::mutate(candidate = key, kind = "raw")
  if (!is.null(tr_nst)) rel_rows[[length(rel_rows) + 1L]] <-
    tr_nst |> dplyr::mutate(candidate = key, kind = "nested_calibrated")
}
if (length(rel_rows) > 0L) {
  rel <- dplyr::bind_rows(rel_rows)
  p_rel <- ggplot(rel, aes(mean_pred, obs_freq, colour = kind)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey60") +
    geom_line() + geom_point(aes(size = n), alpha = 0.7) +
    facet_wrap(~ candidate) + coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Reliability on chal-health OOF (raw vs nested-calibrated)",
         subtitle = "nested_calibrated is leak-free and meaningful; a pooled in-sample T would look optimistic",
         x = "mean predicted", y = "observed frequency", colour = NULL, size = "n") +
    theme_minimal(base_size = 10)
  ggsave(out_path("reliability_oof_chal", "png"), p_rel, width = 11, height = 7, dpi = 150)
}

# (d) ROC on chal-health OOF (overlay candidates) — only if pROC available.
if (.have_proc) {
  roc_rows <- list()
  for (key in cand_keys) {
    p <- m1_pred[[key]]; ok <- is.finite(p) & is.finite(y_h) & mask_chal
    if (sum(ok) < 5L || length(unique(round(y_h[ok]))) < 2L) next
    rc <- tryCatch(pROC::roc(y_h[ok], p[ok], quiet = TRUE, direction = "<"), error = function(e) NULL)
    if (is.null(rc)) next
    roc_rows[[length(roc_rows) + 1L]] <- tibble::tibble(
      candidate = sprintf("%s (AUC=%.3f)", key, as.numeric(pROC::auc(rc))),
      fpr = 1 - rc$specificities, tpr = rc$sensitivities)
  }
  if (length(roc_rows) > 0L) {
    roc_df <- dplyr::bind_rows(roc_rows)
    p_roc <- ggplot(roc_df, aes(fpr, tpr, colour = candidate)) +
      geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey60") +
      geom_line(linewidth = 0.8) + coord_equal() +
      labs(title = "ROC on chal-health OOF", x = "false positive rate", y = "true positive rate",
           colour = NULL) + theme_minimal(base_size = 10)
    ggsave(out_path("roc_oof_chal", "png"), p_roc, width = 8, height = 6, dpi = 150)
  }
}

# (e) Calibrated Brier by candidate x mode (chal-health)
br_plot <- metrics |> dplyr::filter(target == "chal_health", is.finite(brier_cal_oofT))
if (nrow(br_plot) > 0L) {
  p_br <- ggplot(br_plot, aes(reorder(candidate, brier_cal_oofT), brier_cal_oofT, fill = mode)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    coord_flip() +
    labs(title = "Calibrated Brier on chal-health (T from M1 OOF)",
         x = NULL, y = "Brier (lower is better)", fill = NULL) +
    theme_minimal(base_size = 10)
  ggsave(out_path("brier_by_mode_chal", "png"), p_br, width = 9, height = 6, dpi = 150)
}

# =============================================================================
# 11. SAVE
# =============================================================================

saveRDS(list(config = CMP_CFG, timestamp = TS, metrics = metrics,
             modes = lapply(modes, function(m) m$preds),
             y_health = y_h, hfold = hth$hfold,
             mask_chal = mask_chal, mask_forrt = mask_forrt,
             T_deploy = T_deploy, greedy_members_by_fold = greedy_members_by_fold,
             r3_scores = r3, dist_summary = dist_summary, agreement = agreement,
             rows_to_inspect = inspect),
        out_path("09_final_comparison", "rds"), compress = "xz")

cat("\n[final] done. Outputs in: ", CMP_CFG$output_dir, " (suffix _", TS, ")\n", sep = "")
cat("[final] R3 itself has no labels — AUC / reliability are on the chal-health OOF proxy.\n")
