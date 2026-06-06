# =============================================================================
# 06_health_transfer_model.R
#   A feature model that actually predicts HEALTH replication
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# The deployed feature slot (m1 in 10_submission.R) is a fair-meta stack over the
# 32-feature `enriched` set, dominated by random forest (~0.82 weight). It is
# tuned to win on PSYCHOLOGY leave-one-round-out — but Round 3 is 100% HEALTH.
# On the most R3-shaped honest holdout (train on non-health, predict the 102
# labelled health claims) it scores ~0.235 Brier with weak discrimination
# (Spearman p-vs-outcome ~0.27) and, on the n=44 R1/R2 health subset, it does
# not beat the base rate. The LLM ensemble reaches ~0.217 on health.
#
# DIAGNOSIS (see reports/HEALTH_FEATURE_MODEL.html for the full write-up)
#   1. RF cannot extrapolate. Its prediction is the mean outcome of the training
#      rows in the same leaves; past the largest split it ever learned it returns
#      a flat value. Health n_o reaches ~55k vs mostly small-n psychology training,
#      so RF flattens across the entire large-n health regime and its health
#      discrimination is stuck at ~0.25 regardless of how many features it is given.
#   2. It is NOT a "too many features" problem. Of 437 evaluable features, 152 have
#      |rho| > 0.15 with the health outcome; ~100 point the SAME way in psych and
#      health (transferable), but ~52 point the OPPOSITE way (anti-transfer). Those
#      52 poison any parametric learner trained on psychology and applied to health
#      (e.g. a GAM on `enriched` blows up to 0.308 on the health holdout).
#
# THE FIX TESTED HERE
#   * Select features for TRANSFERABILITY, not scarcity: predictive, distribution-
#     stable train->test, well-covered on test, and NOT empirically anti-transfer
#     (sign of the association agrees between psych and health training rows).
#   * Use learners that can use that signal: an equal-weight ensemble, and a
#     MONOTONE-constrained gradient booster that hard-codes the discipline-invariant
#     laws (larger n / larger effect / stronger evidence -> more replicable;
#     more multiple-testing / more surprising -> less replicable), so it
#     extrapolates the RIGHT relationship into health's large-n regime.
#
# VALIDATION INTEGRITY (the important part)
#   The transferability screen needs the health outcomes, so a naive "select on all
#   102 health rows, then score on them" would be CIRCULAR. We avoid this two ways:
#     EVAL A (realistic deploy): repeated grouped 5-fold CV over ALL ground-truth
#       rows, scored on the held-out HEALTH rows. Feature selection + fitting happen
#       INSIDE each fold on the training rows only; the out-of-fold health rows are
#       never seen during selection or fitting.
#     EVAL B (conservative transfer stress test): leave-ALL-health-out (train on
#       non-health, predict the 102 health). With zero health in training the
#       transferability check cannot run, so selection falls back to psych-
#       predictive + stable + covered (no health outcomes used at all).
#   Both are leakage-free with respect to the scored rows. R3 outcomes are never
#   used anywhere (they do not exist).
#
# OUTPUTS
#   output/health_transfer_results.csv  — per-model Brier on each eval + CI
#   output/health_transfer_features.csv — the deploy-selected feature set
#   (prints a shareable summary table to stdout)
#
# REPRODUCE:  Rscript pipeline/06_health_transfer_model.R
#   depends on data/train_base.rds, data/test_base.rds (04_base_dataset.R),
#   data/cache/train_weights.rds (07), judge_llm_ensemble/ensemble_scores.csv
# =============================================================================

suppressPackageStartupMessages({
  source(here::here("pipeline/06_models.R"))      # base learners, PREDICTOR_SETS
  source(here::here("experiments/04_stacking.R"))
  library(dplyr); library(tidyr); library(readr); library(glmnet)
})
set.seed(CONFIG$seed)
OUT   <- "statistical_success"
brier <- function(y, p) mean((p - y)^2, na.rm = TRUE)
clip  <- function(x) pmin(pmax(x, 0.01), 0.99)

# ── 1. Data ──────────────────────────────────────────────────────────────────
train_base <- readRDS(here::here("data/train_base.rds"))
test_base  <- readRDS(here::here("data/test_base.rds"))
r3w        <- readRDS(here::here("data/cache/train_weights.rds")) |> select(claim_id, w_ipw)
ens        <- read_csv(here::here("judge_llm_ensemble/ensemble_scores.csv"), show_col_types = FALSE) |>
  select(claim_id, dataset, p_replication_observed)

gt <- train_base |>
  filter(!is.na(.data[[OUT]]), dataset %in% c("training", "round1", "round2")) |>
  left_join(r3w, by = "claim_id") |>
  mutate(w_ipw = replace_na(w_ipw, 1.0),
         hb    = ifelse(is.na(health_related_big), 0L,
                        as.integer(health_related_big == 1 | health_related_big == TRUE)),
         grp   = ifelse(is.na(doi_o) | doi_o == "", claim_id, as.character(doi_o)))
cat(sprintf("GT rows: %d  (health: %d  psych: %d)  | R3 rows: %d\n",
            nrow(gt), sum(gt$hb == 1), sum(gt$hb == 0), nrow(test_base)))

# ── 2. Candidate pool (mirror 12f leakage/ID stoplist) ───────────────────────
STOP <- c("entry_id", "effect_id", "cos_phase1", OUT, "reported_success",
          "n_r", "es_value_r", "pval_value_r", "r_r", "year_r",
          "duration_o_r", "duration_o_r_sqrtabs", "is_score_replication",
          "prereg_r_bin", "author_overlap_bin", "author_overlap_missing",
          "same_author_country", "same_author_country_missing",
          # the LLM synthetic-crowd FORECASTS are the m2/m3 slots, not a tabular
          # feature — exclude so this stays a genuinely independent feature model.
          "judge_mean_p", "judge_median_p", "judge_sd_p",
          "judge_model_disagreement", "judge_persona_disagreement", "judge_llm_miss_ind")
num_both <- intersect(gt |> select(where(is.numeric)) |> names(),
                      test_base |> select(where(is.numeric)) |> names())
cand <- setdiff(num_both, STOP)
nzv  <- vapply(cand, function(c) { v <- gt[[c]]; isTRUE(sd(v, na.rm = TRUE) > 1e-9) }, logical(1))
cand <- cand[nzv]

# Coverage on the 45 test rows (features only) + train->test SMD (features only).
test_cov <- vapply(cand, function(f) mean(!is.na(test_base[[f]])), numeric(1))
smd_tt   <- vapply(cand, function(f) {
  s <- sd(gt[[f]], na.rm = TRUE); if (is.na(s) || s < 1e-9) return(NA_real_)
  (mean(test_base[[f]], na.rm = TRUE) - mean(gt[[f]], na.rm = TRUE)) / s
}, numeric(1))

# ── 3. Transferability screen (uses TRAIN rows only) ─────────────────────────
# Returns a focused, transferable, de-correlated feature set. When the training
# data contains health rows it DROPS empirically anti-transfer features (sign of
# the psych vs health association disagrees). Caps to MAXK by predictiveness.
MAXK <- 30
rho  <- function(x, y) { ok <- !is.na(x) & !is.na(y); if (sum(ok) < 20) return(NA_real_)
                         suppressWarnings(cor(rank(x[ok]), rank(y[ok]))) }
screen <- function(df) {
  y <- as.numeric(df[[OUT]]); H <- df$hb == 1; P <- df$hb == 0
  scored <- lapply(cand, function(f) {
    x <- df[[f]]
    r_all <- rho(x, y)
    if (is.na(r_all) || abs(r_all) < 0.05)              return(NULL)  # not predictive
    if (is.na(test_cov[f]) || test_cov[f] < 0.5)        return(NULL)  # poor test coverage
    if (!is.na(smd_tt[f]) && abs(smd_tt[f]) > 0.5)      return(NULL)  # distribution-shifted
    # empirical anti-transfer check (only if enough health AND psych rows)
    if (sum(H) >= 25 && sum(P) >= 50) {
      rh <- rho(x[H], y[H]); rp <- rho(x[P], y[P])
      if (!is.na(rh) && !is.na(rp) && abs(rh) > 0.1 && abs(rp) > 0.1 && sign(rh) != sign(rp))
        return(NULL)                                                  # anti-transfer -> drop
    }
    data.frame(feature = f, ar = abs(r_all))
  })
  scored <- do.call(rbind, scored)
  if (is.null(scored)) return(character(0))
  scored <- scored[order(-scored$ar), , drop = FALSE]
  # greedy de-correlation (drop near-duplicates of an already-kept stronger feature)
  kept <- character(0)
  for (f in scored$feature) {
    if (length(kept) == 0) { kept <- f; next }
    cc <- suppressWarnings(sapply(kept, function(k)
      abs(cor(df[[f]], df[[k]], use = "pairwise.complete.obs"))))
    if (all(is.na(cc)) || max(cc, na.rm = TRUE) < 0.95) kept <- c(kept, f)
    if (length(kept) >= MAXK) break
  }
  kept
}

# ── 4. Learners ──────────────────────────────────────────────────────────────
zf <- function(d, p) { d[, p] <- lapply(d[, p, drop = FALSE],
  function(x) { x <- as.numeric(x); x[is.na(x)] <- 0; x }); d }

# Monotone direction prior: +1 = more of this -> more replicable; -1 = less.
DIR <- c(n_o = 1, n_o_log = 1, n_o_sqrt = 1, pval_z = 1,
  evidence_strength_score = 1, power_proxy_h = 1, snr_proxy_h = 1, t_proxy_from_r_h = 1,
  r_o_abs = 1, r_harmonized_low_confidence = -1,
  llm_sample_adequacy = 1, llm_sample_size_ord = 1, llm_sample_size_ord_mean = 1,
  llm_power_guess = 1, llm_power_guess_mean = 1,
  llm_open_data = 1, llm_open_code = 1, llm_preregistration = 1,
  n_f_tests_tei = -1, n_t_tests_tei = -1, tests_per_1000_words_tei = -1,
  p_hack_index_tei = -1, p_heaping_flag_tei = -1,
  llm_perceived_surprisingness = -1, llm_perceived_surprisingness_mean = -1,
  pval_value_o = -1, p_expected = -1)

fit_eq <- function(tr, preds) {           # equal-weight RF+GAM+ElNet+XGB
  trz <- zf(tr, preds)
  list(set = preds,
       RF    = fit_rf(trz, OUT, preds, rep(1, nrow(trz))),
       GAM   = tryCatch(fit_gam(trz, OUT, preds, trz$w_ipw), error = function(e) NULL),
       ElNet = fit_elnet(trz, OUT, preds, trz$w_ipw),
       XGB   = fit_xgb(trz, OUT, preds, trz$w_ipw))
}
pred_eq <- function(m, nd) {
  ndz <- zf(nd, m$set)
  cols <- cbind(RF = pred_rf(m$RF, ndz, m$set),
                GAM = if (is.null(m$GAM)) NA else pred_gam(m$GAM, ndz, m$set),
                ElNet = pred_elnet(m$ElNet, ndz, m$set),
                XGB = pred_xgb(m$XGB, ndz, m$set))
  rowMeans(cols[, colSums(is.na(cols)) == 0, drop = FALSE])
}
fit_mono <- function(tr, preds) {         # monotone-constrained gradient booster
  trz <- zf(tr, preds); x <- as.matrix(trz[, preds]); y <- as.numeric(trz[[OUT]])
  mc <- vapply(preds, function(f) { d <- DIR[f]; if (is.na(d)) 0L else as.integer(d) }, integer(1))
  d  <- xgboost::xgb.DMatrix(x, label = y, weight = pmax(trz$w_ipw, 1e-6))
  m  <- xgboost::xgb.train(list(objective = "reg:squarederror", eta = 0.03, max_depth = 3,
          min_child_weight = 15, subsample = 0.85, colsample_bytree = 0.8, gamma = 0.1,
          monotone_constraints = paste0("(", paste(mc, collapse = ","), ")")),
          d, nrounds = 250, verbose = 0)
  list(set = preds, m = m)
}
pred_mono <- function(m, nd) { ndz <- zf(nd, m$set)
  pmax(pmin(predict(m$m, xgboost::xgb.DMatrix(as.matrix(ndz[, m$set]))), 1), 0) }

# Fixed-set baselines (no per-fold screen) for apples-to-apples comparison.
ENR <- intersect(PREDICTOR_SETS$enriched,   num_both)
STB <- intersect(PREDICTOR_SETS$stable_top, num_both)

# ── 5. EVAL A — repeated grouped 5-fold CV, scored on OOF health rows ─────────
# Models: screened/EQ, screened/Mono, enriched/RF (deployed-ish), stable_top/EQ.
REPS <- 3; K <- 5
models <- c("screen_EQ", "screen_Mono", "enriched_RF", "stabletop_EQ")
oof_sum <- setNames(lapply(models, function(.) rep(0, nrow(gt))), models)
oof_cnt <- rep(0, nrow(gt))
sel_all <- character(0)
for (rep in seq_len(REPS)) {
  set.seed(100 + rep)
  ug <- unique(gt$grp); fold_of <- setNames(sample(rep_len(seq_len(K), length(ug))), ug)
  gt$fold <- fold_of[gt$grp]
  for (k in seq_len(K)) {
    tr <- gt |> filter(fold != k); ho <- gt |> filter(fold == k); idx <- which(gt$fold == k)
    sel <- screen(tr); if (length(sel) < 3) sel <- STB
    sel_all <- c(sel_all, sel)
    m_eq  <- fit_eq(tr, sel);   oof_sum$screen_EQ[idx]    <- oof_sum$screen_EQ[idx]    + pred_eq(m_eq, ho)
    m_mo  <- fit_mono(tr, sel); oof_sum$screen_Mono[idx]  <- oof_sum$screen_Mono[idx]  + pred_mono(m_mo, ho)
    m_er  <- fit_rf(zf(tr, ENR), OUT, ENR, rep(1, nrow(tr)))
    oof_sum$enriched_RF[idx] <- oof_sum$enriched_RF[idx] + pred_rf(m_er, zf(ho, ENR), ENR)
    m_st  <- fit_eq(tr, STB);   oof_sum$stabletop_EQ[idx] <- oof_sum$stabletop_EQ[idx] + pred_eq(m_st, ho)
    oof_cnt[idx] <- oof_cnt[idx] + 1
  }
  cat(sprintf("  EVAL A rep %d/%d done\n", rep, REPS))
}
`%||%` <- function(a, b) if (is.null(a)) b else a
oof <- lapply(oof_sum, function(s) s / pmax(oof_cnt, 1))
yH  <- as.numeric(gt[[OUT]])[gt$hb == 1]
Hsel<- gt$hb == 1
llm_cov <- gt$claim_id %in% (ens |> filter(dataset %in% c("round1","round2")) |> pull(claim_id))
bootci <- function(y, p, B = 4000) { n <- length(y)
  v <- replicate(B, { i <- sample(n, n, TRUE); brier(y[i], p[i]) }); quantile(v, c(.1, .9)) }

# ── 6. EVAL B — leave-ALL-health-out (psych -> health) ───────────────────────
tr0 <- gt |> filter(hb == 0); hoH <- gt |> filter(hb == 1); yB <- as.numeric(hoH[[OUT]])
selB <- screen(tr0); if (length(selB) < 3) selB <- STB        # no health rows -> psych-only screen
evalB <- list(
  screen_EQ    = pred_eq(fit_eq(tr0, selB), hoH),
  screen_Mono  = pred_mono(fit_mono(tr0, selB), hoH),
  enriched_RF  = pred_rf(fit_rf(zf(tr0, ENR), OUT, ENR, rep(1, nrow(tr0))), zf(hoH, ENR), ENR),
  stabletop_EQ = pred_eq(fit_eq(tr0, STB), hoH))

# raw-LLM reference (covers R1/R2 health only)
llmH <- gt |> filter(hb == 1) |> select(claim_id) |>
  inner_join(ens |> select(claim_id, p_replication_observed), by = "claim_id")
yLLM <- as.numeric(gt[[OUT]])[match(llmH$claim_id, gt$claim_id)]

# ── 7. Results table ─────────────────────────────────────────────────────────
# Health rows split by SOURCE. R3 is challenge-round health, so the R1/R2 health
# rows (n=47) are the true R3 analog; the FORRT-database health rows (n=55) come
# from a different sampling process and tend to be in-distribution / easier.
idxH  <- which(gt$hb == 1)
dsH   <- gt$dataset[idxH]
chalH <- dsH %in% c("round1", "round2")        # challenge-round health (R3 analog)
forrH <- dsH == "training"                       # FORRT-database health
res <- list()
for (mdl in models) {
  pH <- oof[[mdl]][idxH]; ciA <- bootci(yH, pH)
  ci_ch <- bootci(yH[chalH], pH[chalH])
  res[[mdl]] <- data.frame(model = mdl,
    A_health_all_n102 = round(brier(yH, pH), 4),
    A_FORRT_n55       = round(brier(yH[forrH], pH[forrH]), 4),
    A_CHALLENGE_n47   = round(brier(yH[chalH], pH[chalH]), 4),
    A_CHAL_CI         = sprintf("%.3f-%.3f", ci_ch[1], ci_ch[2]),
    B_LHO_n102        = round(brier(yB, evalB[[mdl]]), 4))
}
res <- do.call(rbind, res)
# rawLLM covers R1/R2 (challenge) health only — its natural R3-analog comparison
yChalLLM <- yH[chalH]
res <- rbind(res, data.frame(model = "rawLLM (ref)",
  A_health_all_n102 = NA, A_FORRT_n55 = NA,
  A_CHALLENGE_n47 = round(brier(yLLM, llmH$p_replication_observed), 4),
  A_CHAL_CI = NA, B_LHO_n102 = NA))
res <- rbind(res, data.frame(model = "base rate (0.48)",
  A_health_all_n102 = round(brier(yH, rep(mean(yH), length(yH))), 4),
  A_FORRT_n55     = round(brier(yH[forrH], rep(mean(yH[forrH]), sum(forrH))), 4),
  A_CHALLENGE_n47 = round(brier(yH[chalH], rep(mean(yH[chalH]), sum(chalH))), 4),
  A_CHAL_CI = NA, B_LHO_n102 = NA))

cat("\n=========================== RESULTS ============================\n")
cat("EVAL A = repeated grouped 5-fold CV, OOF health rows (health IN training; realistic deploy)\n")
cat("EVAL B = leave-ALL-health-out (psych->health; conservative transfer stress test)\n")
cat("A_CHALLENGE_n47 = the R1/R2 challenge-round health rows = the true R3 analog.\n")
cat("rawLLM is on the SAME 47 challenge-round health rows.\n\n")
print(res, row.names = FALSE)

# ── 8. Deploy: select on all GT, fit, predict R3; report the feature set ──────
selD <- screen(gt)
dep_eq   <- pred_eq(fit_eq(gt, selD), test_base)
dep_mono <- pred_mono(fit_mono(gt, selD), test_base)
r3_llm   <- test_base |> select(claim_id) |>
  inner_join(ens |> filter(dataset == "round3") |> select(claim_id, p_replication_observed), by = "claim_id")
cat(sprintf("\nDEPLOY screened feature set (%d features):\n  %s\n", length(selD),
            paste(selD, collapse = ", ")))
cat(sprintf("\nR3 predictions: screen_EQ mean=%.3f sd=%.3f | screen_Mono mean=%.3f sd=%.3f\n",
            mean(dep_eq), sd(dep_eq), mean(dep_mono), sd(dep_mono)))
cat(sprintf("R3 corr with raw-LLM: screen_EQ=%.2f  screen_Mono=%.2f  (lower = more decorrelated)\n",
            cor(dep_eq, r3_llm$p_replication_observed), cor(dep_mono, r3_llm$p_replication_observed)))

# ── 9. Persist ───────────────────────────────────────────────────────────────
write_csv(res, here::here("output/health_transfer_results.csv"))
sel_counts <- table(sel_all)
sel_tbl <- tibble(feature = names(sel_counts), folds_selected = as.integer(sel_counts),
                  in_deploy_set = names(sel_counts) %in% selD,
                  rho_health = vapply(names(sel_counts), function(f) round(rho(gt[[f]][gt$hb==1], yH), 3), numeric(1)),
                  rho_psych  = vapply(names(sel_counts), function(f) round(rho(gt[[f]][gt$hb==0], as.numeric(gt[[OUT]])[gt$hb==0]), 3), numeric(1)),
                  monotone_dir = vapply(names(sel_counts), function(f) { d <- DIR[f]; if (is.na(d)) 0L else as.integer(d) }, integer(1))) |>
  arrange(desc(folds_selected))
write_csv(sel_tbl, here::here("output/health_transfer_features.csv"))
cat("\nWrote output/health_transfer_results.csv and output/health_transfer_features.csv\n")
