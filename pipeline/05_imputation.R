# =============================================================================
# 05_imputation.R  —  Leak-free MICE Imputation (PMM + 2lonly.pmm)
# =============================================================================
#
# PURPOSE
# -------
# Performs MICE imputation on training/test fold pairs.
# Claim-level variables use standard PMM; paper-level (doi_o-constant)
# variables use 2lonly.pmm to correctly exploit the cluster structure while
# returning actual observed values (0/1) rather than fractional cluster means.
# Imputation models are estimated ONLY on training rows (ignore = TRUE for
# test rows), so no information leaks from the holdout.
#
# KEY DESIGN DECISIONS
# --------------------
# 1. Congeniality:    statistical_success is a column-predictor in every
#                     imputation submodel (van Buuren 2018 §2.4.4) but its
#                     row is zeroed — it is never itself imputed.
# 2. reported_success EXCLUDED: not present in the R3 test set, so including
#                     it would create a systematic train/deploy asymmetry.
# 3. doi_o EXCLUDED from donor vars: clustering is handled via 2lonly.pmm;
#                     doi_o itself (high-cardinality factor) is never a
#                     direct predictor.
# 4. Pool dedup:      One representation per construct; analysis-model
#                     variables are never removed by dedup rules.
# 5. quickpred():     Per-variable predictor selection (mincor = 0.10) for
#                     PMM variables. Zero-row fallback finds the best
#                     available predictor when mincor threshold is not met.
# 6. 2lonly.pmm:      Paper-level LLM variables get method = "2lonly.pmm".
#                     Finds the nearest-neighbour cluster by predicted value
#                     and borrows that cluster's observed value — preserves
#                     binary (0/1) variable semantics and paper-level
#                     within-cluster consistency.
#                     The doi_o-derived .cluster_id column is coded -2 in the
#                     predictor matrix (miceadds convention).
#                     Level-1 anchor predictors (n_o_log, pval_z, etc.) are
#                     included to give the cluster-level regression enough
#                     spread across clusters for stable PMM matching.
# 7. Post-hoc fallback: fully-NA clusters (no observed value in any fold row)
#                     cannot be imputed by 2lonly.pmm. These receive the mode
#                     of observed cluster values (binary variables) or the
#                     cross-cluster mean (continuous variables).
#
# DEPENDENCIES
# ------------
# - PREDICTOR_SETS list (from 06_models.R / loaded upstream)
# - %||% operator (from 00_packages_config.R)
# - Packages: mice, miceadds, dplyr
#
# MAIN INTERFACE
# --------------
# apply_imputation(train_df, test_df, pset_name, m = 5)
#   → list(train = list of m data.frames,
#           test  = list of m data.frames,
#           is_imputed = TRUE)
# =============================================================================

suppressPackageStartupMessages({
  library(mice)
  library(miceadds)
  library(dplyr)
})

# =============================================================================
# A. CONSTANTS
# =============================================================================

# Core donor variable: outcome included for congeniality (never imputed itself)
.CORE_DONOR_VARS      <- c("statistical_success")

# Maximum fraction of missing values allowed in train fold for pool variables.
# Analysis-model variables (donor_vars) are exempt — they pass get_valid_predictors().
.MISSING_MAX_THRESHOLD <- 0.70

# Paper-level LLM variables: constant within doi_o by construction.
# Empirically verified: <0.6% of papers show within-doi_o variation
# (likely annotation noise in multi-study papers, negligible).
# These receive method = "2lonly.pmm".
# Source: PAPER_LLM_COLS in 04_base_dataset.R (extended with verified vars)
.PAPER_LLM_COLS <- c(
  "llm_open_data", "llm_open_data_request", "llm_open_materials_num",
  "llm_open_code", "llm_is_health", "llm_conflict_of_interest",
  "llm_within_paper_rep", "llm_has_funding", "llm_hypothesis_count",
  "llm_writing_quality_ord", "llm_study_count",
  # Verified paper-level (0% within-doi_o variation for online/rct;
  # <0.6% for student_sample and self_report):
  "llm_online_study", "llm_rct_study", "llm_student_sample",
  "llm_self_report"
)

# Auxiliary variables: not in any analysis model but informative for imputing
# incomplete pool variables (journal quality, author track record, paper
# structure, time). Pre-curated to avoid introducing redundant constructs.
.AUXILIARY_VARS <- c(
  # Journal quality → predicts open-science / LLM features
  "sjr_quartile_num", "journal_hindex", "top_factor_score",
  # Author track record → predicts study-quality LLM features
  "first_author_hindex", "last_author_hindex", "first_author_works_count",
  # Temporal → predicts open-science practices
  "year_o", "before_covid",
  # Paper structure → predicts LLM text-derived features
  "n_authors_xml", "n_sections_xml", "article_word_count_xml",
  # ES type category
  "es_type_cat"
)

# Deduplication rules for the imputation pool.
# Format: c(preferred, alternative1, alternative2, ...)
# The preferred variable is kept; alternatives are removed IFF they are
# NOT in any PREDICTOR_SET (analysis-model variables are always protected).
.DEDUP_RULES <- list(
  # n_o: log-transform is preferred; raw/sqrt/bins are auxiliary-only
  n_o      = c("n_o_log",              "n_o", "n_o_sqrt", "n_o_bins"),
  # p-value: pval_z preferred; raw p-value is auxiliary-only
  pval_raw = c("pval_z",               "pval_value_o"),
  # Claim length: log preferred; raw is auxiliary-only
  claim_l  = c("claim_length_log",     "claim_length"),
  # Citations: leakage-safe 2-year count preferred; total count auxiliary-only
  cit      = c("citation_count_t_plus_2y", "number_citations")
)

# =============================================================================
# B. POOL CONSTRUCTION
# =============================================================================

#' Build a deduplicated imputation pool from all predictor sets + auxiliaries.
#'
#' Analysis-model variables (present in any PREDICTOR_SET) are never removed
#' by dedup rules. Only pure auxiliary variables can be deduplicated.
#'
#' @param predictor_sets Named list of character vectors (PREDICTOR_SETS).
#' @param aux_vars       Additional auxiliary variables to include.
#' @return Character vector of pool variable names.
build_imputation_pool <- function(predictor_sets,
                                  aux_vars = .AUXILIARY_VARS) {

  all_pset_vars <- unique(unlist(predictor_sets))
  raw_pool      <- unique(c(all_pset_vars, aux_vars, .CORE_DONOR_VARS))

  to_remove <- character(0)
  for (rule in .DEDUP_RULES) {
    preferred    <- rule[[1]]
    alternatives <- rule[-1]
    if (preferred %in% raw_pool) {
      removable  <- intersect(alternatives, raw_pool)
      # Never remove analysis-model variables
      removable  <- setdiff(removable, all_pset_vars)
      to_remove  <- c(to_remove, removable)
    }
  }

  deduped <- setdiff(raw_pool, to_remove)
  if (length(to_remove) > 0)
    message(sprintf("  [Pool] %d → %d vars after dedup (removed: %s)",
                    length(raw_pool), length(deduped),
                    paste(to_remove, collapse = ", ")))
  deduped
}

# =============================================================================
# C. VALID PREDICTOR SELECTION (per fold)
# =============================================================================

#' Return predictor-set variables present in train_df and below missing cap.
#'
#' @param train_df  Training data frame for the current fold.
#' @param pset_name Name of predictor set in PREDICTOR_SETS.
#' @return Character vector of valid predictor names.
get_valid_predictors <- function(train_df, pset_name) {
  if (!exists("PREDICTOR_SETS"))
    stop("[Imputation] PREDICTOR_SETS not loaded. Source 06_models.R first.")

  predictors <- intersect(PREDICTOR_SETS[[pset_name]], names(train_df))

  ok <- vapply(predictors, function(col) {
    pm <- mean(is.na(train_df[[col]]))
    if (pm > .MISSING_MAX_THRESHOLD) {
      message(sprintf("  [Imputation] DROP '%s' from donor set: %.1f%% missing",
                      col, pm * 100))
      FALSE
    } else TRUE
  }, logical(1))

  predictors[ok]
}

# =============================================================================
# D. CORE IMPUTATION FUNCTION
# =============================================================================

#' Leak-free MICE with PMM (claim-level) and 2lonly.pmm (paper-level).
#'
#' Imputation models are fitted on training rows only (ignore = TRUE for test).
#' Returns m completed copies of donor_vars for both train and test partitions.
#'
#' @param train_df   Training data frame. Must contain doi_o.
#' @param test_df    Test/holdout data frame. Must contain doi_o.
#' @param donor_vars Variables to return in completed datasets (analysis vars).
#'                   Must be a subset of imp_pool.
#' @param imp_pool   Full deduplicated variable pool fed to MICE.
#' @param m          Number of multiple imputations (default 5).
#'
#' @return list(train = list[m], test = list[m], is_imputed = TRUE)
impute_mice <- function(train_df, test_df, donor_vars, imp_pool, m = 5, maxit = 10) {

  train_n    <- nrow(train_df)
  ignore_idx <- c(rep(FALSE, train_n), rep(TRUE, nrow(test_df)))

  # Mask outcome in test rows ─────────────────────────────────────────────────
  test_masked <- test_df
  if ("statistical_success" %in% names(test_masked))
    test_masked[["statistical_success"]] <- NA

  combined <- dplyr::bind_rows(train_df, test_masked)

  # Cluster ID from doi_o (required by 2lonly.pmm) ────────────────────────────
  if (!"doi_o" %in% names(combined))
    stop("[Imputation] 'doi_o' column required for cluster-aware imputation.")
  combined[[".cluster_id"]] <- as.integer(as.factor(as.character(combined$doi_o)))

  # Restrict pool to columns present in combined ──────────────────────────────
  # donor_vars are always included (they were already guardrail-filtered)
  pool_cols <- unique(c(imp_pool, donor_vars, ".cluster_id"))
  pool_cols <- intersect(pool_cols, names(combined))

  # Apply guardrail only to pool-only auxiliary vars (not donor_vars)
  train_rows   <- combined[seq_len(train_n), pool_cols, drop = FALSE]
  miss_train   <- vapply(pool_cols,
                         function(v) mean(is.na(train_rows[[v]])), numeric(1))
  is_donor     <- pool_cols %in% c(donor_vars, ".cluster_id")
  pool_cols    <- pool_cols[is_donor | miss_train <= .MISSING_MAX_THRESHOLD]

  combined_sub <- combined[, pool_cols, drop = FALSE]
  # Remove non-numeric auxiliary vars — PMM/2lonly.pmm require numeric input.
  # Donor vars (analysis model) should always be numeric; dropping only affects
  # auxiliary predictors like es_type_cat.
  non_num_cols <- names(combined_sub)[vapply(names(combined_sub), function(v) {
    !is.numeric(combined_sub[[v]]) &&
      !is.integer(combined_sub[[v]]) &&
      !is.logical(combined_sub[[v]]) &&
      v != ".cluster_id"
  }, logical(1))]
  if (length(non_num_cols) > 0L) {
    message(sprintf("  [Imputation] dropping non-numeric pool vars: %s",
                    paste(non_num_cols, collapse = ", ")))
    combined_sub <- combined_sub[, setdiff(names(combined_sub), non_num_cols),
                                 drop = FALSE]
    col_names <- names(combined_sub)
    n_col     <- length(col_names)
  }
  # Pre-fill paper-level vars: within each cluster, propagate any observed
  # training-row value to NA rows (train + test). Fixes 2lonly.pmm crash on
  # partially-missing level-2 clusters (miceadds stop(), not warning).
  # Uses training rows only as fill source to remain leak-free.
  paper_lvl_present <- intersect(.PAPER_LLM_COLS, names(combined_sub))
  if (length(paper_lvl_present) > 0L && ".cluster_id" %in% names(combined_sub)) {
    cluster_ids <- combined_sub[[".cluster_id"]]
    for (v in paper_lvl_present) {
      vals       <- combined_sub[[v]]
      train_vals <- vals
      train_vals[ignore_idx] <- NA  # restrict fill source to train rows

      cluster_fill <- tapply(train_vals, cluster_ids, FUN = function(x) {
        obs <- x[!is.na(x)]
        if (length(obs) == 0L) return(NA_real_)
        if (all(obs %in% c(0, 1)))
          as.numeric(names(sort(table(obs), decreasing = TRUE))[1L])
        else mean(obs)
      })

      na_rows <- which(is.na(vals))
      if (length(na_rows) == 0L) next
      fill_vals <- cluster_fill[as.character(cluster_ids[na_rows])]
      filled    <- !is.na(fill_vals)
      if (any(filled)) {
        combined_sub[[v]][na_rows[filled]] <- as.numeric(fill_vals[filled])
        message(sprintf(
          "  [Imputation] cluster pre-fill: %d NA rows in '%s' filled from training obs",
          sum(filled), v))
      }
    }
  }
  col_names    <- names(combined_sub)
  n_col        <- length(col_names)

  # ── Method vector ──────────────────────────────────────────────────────────
  has_miss   <- vapply(col_names, function(v) anyNA(combined_sub[[v]]), logical(1))
  method_vec <- setNames(rep("", n_col), col_names)

  # 2lonly.pmm: paper-level LLM variables with missingness.
  # Returns actual observed cluster values (0/1 for binary vars) via
  # nearest-neighbour cluster matching -- no fractional means.
  twol_vars <- intersect(.PAPER_LLM_COLS, col_names)
  twol_vars <- twol_vars[has_miss[twol_vars]]
  method_vec[twol_vars] <- "2lonly.pmm"

  # pmm: all other incomplete variables (excluding outcome and cluster ID)
  excluded_from_pmm <- c(twol_vars, "statistical_success", ".cluster_id")
  pmm_vars <- setdiff(col_names[has_miss], excluded_from_pmm)
  miss_rate_train <- vapply(pmm_vars, function(v)
    mean(is.na(combined_sub[!ignore_idx, v])), numeric(1))

  method_vec[pmm_vars[miss_rate_train <= 0.30]] <- "pmm"
  method_vec[pmm_vars[miss_rate_train >  0.30]] <- "cart"

  n_pmm  <- length(pmm_vars)
  n_twol <- length(twol_vars)

  if (n_pmm == 0 && n_twol == 0) {
    message("  [Imputation] No missing values in pool — returning zero-fill result.")
    train_ret <- combined_sub[seq_len(train_n), intersect(donor_vars, col_names), drop = FALSE]
    test_ret  <- combined_sub[seq(train_n + 1L, nrow(combined_sub)),
                              intersect(donor_vars, col_names), drop = FALSE]
    return(list(train      = rep(list(train_ret), m),
                test       = rep(list(test_ret), m),
                is_imputed = FALSE))
  }

  # ── Predictor matrix ───────────────────────────────────────────────────────
  # quickpred() selects predictors per variable based on correlation with the
  # target and its missingness indicator. Computed on train rows only (leak-free).
  # .cluster_id is excluded from quickpred (handled manually for 2lonly.pmm).
  train_for_qp <- combined_sub[!ignore_idx, , drop = FALSE]

  pred_matrix <- tryCatch({
    mice::quickpred(
      data    = train_for_qp,
      mincor  = 0.10,
      minpuc  = 0.50,
      include = intersect("statistical_success", col_names),
      exclude = ".cluster_id"
    )
  }, error = function(e) {
    message("  [Imputation] quickpred() failed, using full predictor matrix: ", e$message)
    m_full <- matrix(1L, n_col, n_col, dimnames = list(col_names, col_names))
    diag(m_full) <- 0L
    m_full[, ".cluster_id"] <- 0L
    m_full
  })

  # Expand to combined_sub dimensions if quickpred returned smaller matrix
  # (defensive — normally dimensions match)
  if (!identical(rownames(pred_matrix), col_names)) {
    full_pred <- matrix(0L, n_col, n_col, dimnames = list(col_names, col_names))
    shared    <- intersect(rownames(pred_matrix), col_names)
    full_pred[shared, shared] <- pred_matrix[shared, shared]
    pred_matrix <- full_pred
  }
  diag(pred_matrix) <- 0L

  # statistical_success: always a column predictor, never imputed (row = 0)
  if ("statistical_success" %in% col_names)
    pred_matrix["statistical_success", ] <- 0L

  # .cluster_id: never imputed; role set manually per 2lonly.pmm variable below
  if (".cluster_id" %in% col_names)
    pred_matrix[".cluster_id", ] <- 0L

  # Fallback for PMM donor_vars whose quickpred row is all-zero:
  # quickpred(mincor=0.10) may find no predictor meeting the threshold
  # (common for LLM features with high FORRT missingness). Find the top-3
  # correlating available predictors without threshold; add statistical_success
  # as a guaranteed safety net.
  pmm_donor_vars_check <- intersect(
    names(method_vec)[method_vec == "pmm"], donor_vars)
  for (dv in pmm_donor_vars_check) {
    if (sum(pred_matrix[dv, ]) == 0L) {
      cands <- setdiff(col_names,
                       c(dv, ".cluster_id", twol_vars, "statistical_success"))
      if (length(cands) > 0L) {
        cor_vals <- vapply(cands, function(v) {
          x <- train_for_qp[[v]]; y <- train_for_qp[[dv]]
          ok <- !is.na(x) & !is.na(y)
          if (sum(ok) < 10L) return(0)
          abs(cor(x[ok], y[ok]))
        }, numeric(1))
        top3 <- names(sort(cor_vals, decreasing = TRUE)[
          seq_len(min(3L, length(cor_vals)))])
        pred_matrix[dv, top3] <- 1L
      }
      if ("statistical_success" %in% col_names)
        pred_matrix[dv, "statistical_success"] <- 1L
      message(sprintf(
        "  [Imputation] zero-row fallback for PMM var '%s': added predictors",
        dv))
    }
  }

  # 2lonly.pmm rows: override predictor matrix completely.
  # miceadds convention: cluster variable column = -2L.
  # Level-1 anchor predictors are included so the cluster-level regression
  # has enough spread across clusters for stable nearest-neighbour matching.
  if (n_twol > 0 && ".cluster_id" %in% col_names) {
    l1_anchors <- intersect(
      c("n_o_log", "pval_z", "evidence_strength_score",
        "sjr_quartile_num", "year_o", "n_authors_oa"),
      col_names
    )
    for (v in twol_vars) {
      pred_matrix[v, ]              <- 0L
      pred_matrix[v, ".cluster_id"] <- -2L
      other_twol <- setdiff(twol_vars, v)
      if (length(other_twol) > 0)
        pred_matrix[v, other_twol]  <- 1L
      if (length(l1_anchors) > 0)
        pred_matrix[v, l1_anchors]  <- 1L
      if ("statistical_success" %in% col_names)
        pred_matrix[v, "statistical_success"] <- 1L
    }
  }

  message(sprintf("  [Imputation] MICE: %d pmm vars, %d 2lonly.pmm vars, pool=%d, m=%d",
                  n_pmm, n_twol, n_col, m))

  # ── Run MICE ───────────────────────────────────────────────────────────────
  miss_rate_all <- vapply(col_names, function(v)
    mean(is.na(combined_sub[!ignore_idx, v])), numeric(1))
  visit_seq <- col_names[order(miss_rate_all)]
  visit_seq <- visit_seq[method_vec[visit_seq] != ""]
  mice_out <- suppressWarnings(
    mice::mice(
      data            = combined_sub,
      m               = m,
      method          = method_vec,
      predictorMatrix = pred_matrix,
      ignore          = ignore_idx,
      maxit           = maxit,
      seed            = (`%||%`(CONFIG$seed %||% NULL, 42L)),
      printFlag       = FALSE
    )
  )

  # ── Extract completed datasets ─────────────────────────────────────────────
  # Return ONLY donor_vars (not auxiliary pool vars, not .cluster_id).
  # Post-hoc fallback: 2lonly.pmm cannot impute fully-NA clusters (no observed
  # value in any training row for that doi_o). These residual NAs receive:
  #   binary (0/1) variables  -> mode of observed cluster values
  #   continuous variables     -> cross-cluster mean of observed values
  # Both choices give all rows in a cluster the same value, preserving
  # paper-level consistency.
  .is_binary01 <- function(x) {
    obs <- x[!is.na(x)]
    length(obs) > 0L && all(obs %in% c(0, 1))
  }

  return_cols <- intersect(donor_vars, col_names)

  train_list <- vector("list", m)
  test_list  <- vector("list", m)
  for (i in seq_len(m)) {
    cdf <- mice::complete(mice_out, action = i)

    for (v in twol_vars) {
      if (!v %in% names(cdf)) next
      still_na <- is.na(cdf[[v]])
      if (!any(still_na)) next
      obs_vals     <- cdf[[v]][!still_na]
      fallback_val <- if (.is_binary01(obs_vals)) {
        as.numeric(names(sort(table(obs_vals), decreasing = TRUE))[1L])
      } else {
        median(obs_vals, na.rm = TRUE)
      }
      if (is.na(fallback_val)) fallback_val <- 0
      cdf[[v]][still_na] <- fallback_val
      message(sprintf(
        "  [Imputation] 2lonly.pmm fallback (fully-NA cluster): %d rows in '%s' -> %s (%.3f)",
        sum(still_na), v,
        if (.is_binary01(obs_vals)) "mode" else "median",
        fallback_val))
    }
    # Post-hoc fallback for regular pmm/cart variables that mice could not impute
    # (e.g. test row had all-NA predictors, or the donor pool was too sparse).
    # Mirrors the 2lonly.pmm fallback above. Fill source = training rows only
    # (leak-free); same mode/mean logic as in twol_vars to keep semantics consistent.
    pmm_cart_vars <- names(method_vec)[method_vec %in% c("pmm", "cart")]
    train_rows_idx <- seq_len(train_n)
    for (v in pmm_cart_vars) {
      if (!v %in% names(cdf)) next
      still_na <- is.na(cdf[[v]])
      if (!any(still_na)) next
      train_obs <- cdf[[v]][train_rows_idx]
      train_obs <- train_obs[!is.na(train_obs)]
      if (length(train_obs) == 0L) next
      fallback_val <- if (.is_binary01(train_obs)) {
        as.numeric(names(sort(table(train_obs), decreasing = TRUE))[1L])
      } else {
        median(train_obs, na.rm = TRUE)
      }
      if (is.na(fallback_val)) fallback_val <- 0
      cdf[[v]][still_na] <- fallback_val
      message(sprintf(
        "  [Imputation] pmm/cart fallback (unimputable rows): %d rows in '%s' -> %s (%.3f)",
        sum(still_na), v,
        if (.is_binary01(train_obs)) "train mode" else "train median",
        fallback_val))
    }

    train_list[[i]] <- cdf[seq_len(train_n),              return_cols, drop = FALSE]
    test_list[[i]]  <- cdf[seq(train_n + 1L, nrow(cdf)), return_cols, drop = FALSE]
  }

  list(train = train_list, test = test_list, is_imputed = TRUE)
}

# =============================================================================
# E. MAIN INTERFACE
# =============================================================================

#' Build pool, validate predictors, and run impute_mice() for a given fold.
#'
#' @param train_df  Training data frame (must contain doi_o).
#' @param test_df   Test/holdout data frame (must contain doi_o).
#' @param pset_name Name of predictor set in PREDICTOR_SETS.
#' @param m         Number of MICE imputations (default 5).
#'
#' @return list(train, test, is_imputed), or NULL if no valid predictors found.
apply_imputation <- function(train_df, test_df, pset_name, m = 5, maxit = 10) {

  # Validate and filter predictor-set variables for this fold
  donor_vars <- get_valid_predictors(train_df, pset_name)
  if (length(donor_vars) == 0) {
    warning("[Imputation] No valid predictors for '", pset_name, "'. Returning NULL.")
    return(NULL)
  }
  donor_vars <- unique(c(donor_vars, .CORE_DONOR_VARS))

  # Build deduplicated imputation pool from all sets + auxiliaries
  imp_pool <- build_imputation_pool(PREDICTOR_SETS)
  # Restrict to columns actually present in train_df
  imp_pool <- intersect(imp_pool, names(train_df))

  impute_mice(
    train_df   = train_df,
    test_df    = test_df,
    donor_vars = donor_vars,
    imp_pool   = imp_pool,
    m          = m,
    maxit      = maxit
  )
}

cat("[05_imputation.R] loaded:",
    "apply_imputation(), impute_mice(), build_imputation_pool()\n")
