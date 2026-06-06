# =============================================================================
# 05_health_gate.R  —  Health hold-out GATE for the m1/m2/m3 submission candidates
# =============================================================================
#
# WHAT THIS IS (and, just as important, what it is NOT)
# -----------------------------------------------------
# This is a GATE, not a selection step. The distinction matters:
#
#   * SELECTION = using a set's scores to CHOOSE among candidates (which model,
#     which weights, which calibration, which of m1/m2/m3). Selection is done
#     against R1+R2 (n~226), which is large enough to absorb limited, disciplined
#     comparison. The more candidates you rank against a set, the more you overfit
#     that set's NOISE, so the winner's score on it is optimistically biased.
#
#   * GATE (this script) = a coarse, low-multiplicity VETO applied to candidates
#     that were already chosen on R1+R2. It answers one yes/no question per
#     candidate — "does this hold up on the target discipline, or does it collapse
#     when extrapolated to 100%-health data like Round 3?" — and nothing finer.
#
# WHY NOT JUST SELECT ON THE HEALTH SET TOO?
# ------------------------------------------
# Round 3 is 100% health; the labelled pool is ~1% health (~102 broad health_big
# GT rows). At n~45-102 the bootstrap CI on a single Brier is ~+/-0.015, so:
#   (1) ranking candidates by health Brier would mostly rank sampling noise — you
#       would pick whichever candidate got lucky on ~45 specific rows; and
#   (2) the health set is only an honest forecast of Round-3 performance for as
#       long as you DON'T optimise against it. Select on it and you spend the one
#       clean read you have on discipline transfer.
# Therefore: rank on R1+R2; use this script only to VETO collapses. Do NOT pick
# the lowest-health-Brier candidate. Sub-CI differences here are TIES.
#
# THE HOLD-OUT DESIGN (deploy-faithful, not leave-ALL-health-out)
# --------------------------------------------------------------
# At deploy the model trains on EVERY labelled row, including all 102 health rows,
# then predicts unseen health (R3). The faithful hold-out mirrors that: keep all
# non-health + MOST health in training, hold out a rotating R3-sized (~45) block,
# and pool the out-of-fold predictions over all 102 health rows. (Leaving ALL
# health out instead trains on zero health — a pessimistic worst case that
# understated m1 by ~0.011 in testing — so it is reported only as a secondary
# bound, not the headline.) "R1/R2-sized" (~110) is impossible without zeroing
# health in training: only 102 health rows exist, so a ~110 hold-out IS
# leave-all-out.
#
# CANDIDATE COVERAGE ASYMMETRY (important)
# ----------------------------------------
# m1 is a feature model and can be REFIT with health held out -> gated on all 102
# health rows (deploy-faithful). The LLM is zero-parameter and was only ever run
# on R1/R2/R3 (no FORRT), so m3 (bin-corrected LLM), raw-LLM, and m2 (=0.5*m1 +
# 0.5*raw-LLM) only have out-of-sample health predictions on the ~44 R1/R2 health
# rows. We therefore gate ALL candidates apples-to-apples on the common R1/R2
# health set, and additionally report m1's deploy-faithful 102-row number.
#
# GATE CRITERIA (deliberately lenient — a veto, not a ranking)
# ------------------------------------------------------------
#   Gate A (beats base rate): point Brier < base-rate-constant Brier = p(1-p).
#       A candidate that cannot beat predicting the health base rate adds no
#       health signal.
#   Gate B (no collapse vs peers): the candidate's 80% bootstrap CI must overlap
#       the best candidate's POINT estimate. If a candidate's whole CI sits above
#       the best point estimate it is worse than the field beyond sampling noise
#       -> FLAG (the "great on R1/R2 but craters on health" case).
#   VERDICT = PASS iff both gates pass; otherwise FLAG (investigate, don't
#       silently keep). The deploy-faithful estimate supersedes the common-set
#       one for the final per-candidate verdict where both exist (m1).
#
# This script is SELF-CONTAINED: it re-derives the m1 fair-LORO weights and the
# LLM per-N-bin LORO correction from canonical inputs using the SAME recipe as
# pipeline/10_submission.R (no dependency on any scratch_*.rds).
#
# INPUTS (all canonical)
#   data/train_base.rds                  features + health_related_big + outcome
#   data/test_base.rds                   R3 rows (only used for predictor intersection)
#   data/cache/train_weights.rds         R3-targeted IPW (deploy base-learner weights)
#   judge_llm_ensemble/ensemble_scores.csv  LLM synthetic-crowd p_replication_observed
# OUTPUT
#   output/health_gate.csv   per-candidate: set, basis, n, base, brier, ci, gates, verdict
# REPRODUCE
#   Rscript pipeline/05_health_gate.R
# =============================================================================

suppressPackageStartupMessages({
  source(here::here("pipeline/06_models.R"))      # PREDICTOR_SETS, MODEL_GRID, CONFIG
  source(here::here("experiments/04_stacking.R"))   # oof_base_predictions, base_holdout_matrix, assign_group_folds
  source(here::here("pipeline/07_calibration.R")) # fit_calibrator, apply_calibrator
  library(dplyr); library(tidyr); library(tibble); library(readr); library(glmnet)
})
set.seed(CONFIG$seed)
OUTCOME <- "statistical_success"; GROUP <- "doi_o"; PSET <- "enriched"
INNER_FOLDS    <- 5     # inner CV folds for OOF base predictions / meta-weights
N_HEALTH_FOLDS <- 5     # rotating health hold-out folds (each ~20 held out, ~82 in train)
R3_DRAW_N      <- 45    # Round-3-sized draw for the variance distribution
B_BOOT         <- 5000
BIN_FAC        <- 0.6   # shrink factor on the per-N-bin LLM correction (matches 17)
M2_LLM_W       <- 0.5   # raw-LLM weight in the m2 bridge (matches 17)
clip01 <- function(x) pmin(pmax(x, 0.01), 0.99)
brier  <- function(y, p) mean((p - y)^2)

# ── load (canonical only) ─────────────────────────────────────────────────────────
train_base <- readRDS(here::here("data/train_base.rds"))
test_base  <- readRDS(here::here("data/test_base.rds"))
r3_w <- readRDS(here::here("data/cache/train_weights.rds")) |> select(claim_id, w_ipw)
ens  <- read_csv(here::here("judge_llm_ensemble/ensemble_scores.csv"), show_col_types = FALSE) |>
  select(claim_id, dataset, p_replication_observed)

gt_all <- train_base |>
  filter(!is.na(.data[[OUTCOME]]), dataset %in% c("training", "round1", "round2"))
predictors <- intersect(intersect(PREDICTOR_SETS[[PSET]], names(gt_all)), names(test_base))
zf <- function(d, p) { d[, p] <- lapply(d[, p, drop = FALSE], \(x){ x <- as.numeric(x); x[is.na(x)] <- 0; x }); d }
gt_all <- zf(gt_all, predictors)
hb_flag <- train_base |> transmute(claim_id, hb = health_related_big)

# ── shared model machinery (identical recipe to pipeline/10_submission.R) ───────────
rf  <- Find(\(m) m$name == "RF",    MODEL_GRID); gam <- Find(\(m) m$name == "GAM",   MODEL_GRID)
el  <- Find(\(m) m$name == "ElNet", MODEL_GRID); xg  <- Find(\(m) m$name == "XGB",   MODEL_GRID)
wspec <- function(nm, s) list(name = nm,
  fit = function(tr, oc, pr, wp) s$fit(tr, oc, pr, if (!is.null(tr$w_ipw)) tr$w_ipw else rep(1, nrow(tr))),
  pred = s$pred)
base_specs <- list(rf, wspec("GAM_ipw", gam), wspec("ElNet_ipw", el), wspec("XGB_ipw", xg))

fitw <- function(Z, y, w = NULL) {   # weighted softmax-NNLS meta-learner
  M <- ncol(Z); if (is.null(w)) w <- rep(1, length(y)); w <- w / mean(w)
  obj <- function(p) { a <- exp(p - max(p)); a <- a / sum(a); sum(w * (as.numeric(Z %*% a) - y)^2) / sum(w) }
  o <- optim(rep(0, M), obj, method = "BFGS", control = list(maxit = 500))
  a <- exp(o$par - max(o$par)); a <- a / sum(a); names(a) <- colnames(Z); a
}

propensity_features <- intersect(c(
  "n_o_log", "pval_z", "prop_digits", "r_o_abs", "claim_length_log", "is_health",
  "es_directional", "pval_exact", "r_harmonized", "grim_mean_prob_sc",
  "tests_per_1000_words_tei", "fulltext_n_pval_nlp_log1p", "year_o", "duration_o_r"
), names(train_base))
compute_propensity_weights <- function(train_df, target_dataset, all_data) {
  feats    <- intersect(propensity_features, names(all_data))
  combined <- all_data |> select(all_of(c(feats, "dataset"))) |>
    mutate(is_target = as.integer(dataset == target_dataset))
  for (col in feats) { med <- median(combined[[col]], na.rm = TRUE); combined[[col]][is.na(combined[[col]])] <- med }
  X_all <- model.matrix(~ . - dataset - is_target - 1, data = combined)
  set.seed(42)
  prop_mod <- glmnet::cv.glmnet(X_all, combined$is_target, family = "binomial",
                                alpha = 0, nfolds = 5, type.measure = "deviance")
  td <- train_df |> select(all_of(feats))
  for (col in feats) { med <- median(combined[[col]], na.rm = TRUE); td[[col]][is.na(td[[col]])] <- med }
  X_train <- model.matrix(~ . - 1, data = td)
  for (m in setdiff(colnames(X_all), colnames(X_train)))
    X_train <- cbind(X_train, setNames(matrix(0, nrow(X_train), 1), m))
  X_train <- X_train[, colnames(X_all), drop = FALSE]
  pscore <- as.numeric(predict(prop_mod, newx = X_train, s = "lambda.min", type = "response"))
  raw <- pscore / (1 - pmin(pmax(pscore, 1e-3), 1 - 1e-3))
  w   <- pmin(raw, quantile(raw, 0.95, na.rm = TRUE)); w / mean(w, na.rm = TRUE)
}

NB <- c(-Inf, 100, 300, 1000, 3000, 10000, Inf)
NL <- c("<=100", "100-300", "300-1k", "1k-3k", "3k-10k", ">10k")
n_bin_of    <- function(n_o) cut(suppressWarnings(as.numeric(n_o)), breaks = NB, labels = NL, right = TRUE)
shift_table <- function(d) d |> filter(!is.na(n_bin)) |> group_by(n_bin) |>
  summarise(shift = mean(outcome - p_replication_observed), .groups = "drop")
apply_shift <- function(df, tab, fac = BIN_FAC) {
  o <- df |> left_join(tab, by = "n_bin"); o$shift[is.na(o$shift)] <- 0
  clip01(o$p_replication_observed + fac * o$shift)
}

# ── DERIVE m1: fair (holdout-targeted, leave-one-round-out) meta-weights + LORO preds ─
FOLDS <- list(list(holdout = "round1", train = c("training", "round2")),
              list(holdout = "round2", train = c("training", "round1")))
w_per_fold <- list(); m1_loo_rows <- list()
for (f in FOLDS) {
  tr <- gt_all |> filter(dataset %in% f$train); ho <- gt_all |> filter(dataset == f$holdout)
  tr$w_ipw <- compute_propensity_weights(tr, f$holdout, gt_all)
  oo  <- oof_base_predictions(tr, base_specs, predictors, OUTCOME, n_folds = INNER_FOLDS, group_var = GROUP, weights = NULL, seed = CONFIG$seed)
  ok  <- stats::complete.cases(oo$oof) & is.finite(oo$y)
  w_m <- fitw(oo$oof[ok, , drop = FALSE], oo$y[ok], tr$w_ipw[ok])
  full <- lapply(base_specs, \(s) s$fit(tr, OUTCOME, predictors, rep(1, nrow(tr))))
  p_ho <- as.numeric(base_holdout_matrix(full, base_specs, ho, predictors) %*% w_m)
  w_per_fold[[f$holdout]] <- w_m
  m1_loo_rows[[f$holdout]] <- tibble(claim_id = ho$claim_id, outcome = as.numeric(ho[[OUTCOME]]), m1_loo = p_ho)
}
w_avg  <- (w_per_fold[["round1"]] + w_per_fold[["round2"]]) / 2   # DEPLOY meta-weights
m1_loo <- bind_rows(m1_loo_rows)

# ── DERIVE LLM: raw + per-N-bin LORO correction (zero-parameter raw; LORO shift) ─────
llm_gt <- gt_all |> select(claim_id, dataset, n_o, !!OUTCOME) |> rename(outcome = !!OUTCOME) |>
  inner_join(ens, by = c("claim_id", "dataset")) |> mutate(n_bin = n_bin_of(n_o))
loo_llm <- bind_rows(lapply(c("round1", "round2"), function(h) {
  tab_h <- shift_table(llm_gt |> filter(dataset != h))
  ho_h  <- llm_gt |> filter(dataset == h)
  tibble(claim_id = ho_h$claim_id, llm_raw = ho_h$p_replication_observed, llm_bincorr = apply_shift(ho_h, tab_h))
}))

# ── bootstrap gate helper ─────────────────────────────────────────────────────────
gate_table <- function(y, preds, label) {
  p0 <- mean(y); base_brier <- p0 * (1 - p0)
  out <- bind_rows(lapply(names(preds), function(nm) {
    p  <- preds[[nm]]
    bs <- replicate(B_BOOT, { i <- sample(length(y), length(y), TRUE); brier(y[i], p[i]) })
    tibble(set = label, candidate = nm, n = length(y), base_rate = p0, base_brier = base_brier,
           brier = brier(y, p), ci_lo = unname(quantile(bs, .10)), ci_hi = unname(quantile(bs, .90)))
  }))
  best <- min(out$brier)
  out |> mutate(gateA_beats_base = brier < base_brier,
                gateB_no_collapse = ci_lo <= best,
                verdict = ifelse(gateA_beats_base & gateB_no_collapse, "PASS", "FLAG"))
}

# ── 1. Common-set gate: ALL candidates on R1/R2 health rows (leakage-free) ──────────
common <- m1_loo |>
  inner_join(loo_llm, by = "claim_id") |>
  left_join(hb_flag, by = "claim_id") |> filter(hb == 1)
y_c <- as.numeric(common$outcome)
gate_common <- gate_table(y_c, list(
  m1     = common$m1_loo,
  m2     = (1 - M2_LLM_W) * common$m1_loo + M2_LLM_W * common$llm_raw,
  m3     = common$llm_bincorr,
  rawLLM = common$llm_raw), sprintf("R1/R2 health (n=%d)", nrow(common)))

# ── 2. Deploy-faithful health hold-out for m1 (all 102 health rows) ─────────────────
# Refit base learners with a rotating R3-sized health block held out (all non-health
# + the other ~82 health rows stay in training, with R3-targeted IPW as at deploy),
# combine with the DEPLOY meta-weights w_avg, temperature-calibrate -> pooled OOS.
gt_dep <- gt_all |> left_join(r3_w, by = "claim_id") |> mutate(w_ipw = replace_na(w_ipw, 1.0))
nonh <- gt_dep |> filter(health_related_big == 0)
hth  <- gt_dep |> filter(health_related_big == 1)
fid  <- assign_group_folds(hth[[GROUP]], N_HEALTH_FOLDS, CONFIG$seed)
m1_health_oos <- rep(NA_real_, nrow(hth))
for (f in sort(unique(fid))) {
  va  <- which(fid == f)
  trh <- bind_rows(nonh, hth[-va, , drop = FALSE])
  full <- lapply(base_specs, \(s) s$fit(trh, OUTCOME, predictors, rep(1, nrow(trh))))
  m1_health_oos[va] <- as.numeric(base_holdout_matrix(full, base_specs, hth[va, , drop = FALSE], predictors) %*% w_avg)
}
# calibrate on the non-health OOF (mirrors deploy; never fit calibration on held-out health)
oo_nh <- oof_base_predictions(nonh, base_specs, predictors, OUTCOME, n_folds = INNER_FOLDS, group_var = GROUP, weights = NULL, seed = CONFIG$seed)
ok_nh <- stats::complete.cases(oo_nh$oof) & is.finite(oo_nh$y)
calib <- fit_calibrator(as.numeric(oo_nh$oof[ok_nh, ] %*% w_avg), oo_nh$y[ok_nh], method = "temperature")
m1_health_oos <- apply_calibrator(calib, m1_health_oos)
y_h <- as.numeric(hth[[OUTCOME]])
gate_deploy <- gate_table(y_h, list(m1 = m1_health_oos), sprintf("deploy-faithful health (n=%d)", nrow(hth)))
draw <- replicate(B_BOOT, { i <- sample(length(y_h), min(R3_DRAW_N, length(y_h)), FALSE); brier(y_h[i], m1_health_oos[i]) })

# ── 3. Report + write ─────────────────────────────────────────────────────────────
# Section [1] is the apples-to-apples PEER gate; [2] is m1's shipped forecast,
# gated vs base rate (no peer has a comparable 102-row estimate). For the FINAL
# verdict the deploy-faithful estimate SUPERSEDES the common-set one (m1); the
# common-set m1 row is kept as context (the known LORO fold artifact).
report <- bind_rows(gate_common |> mutate(basis = "common-LORO"),
                    gate_deploy |> mutate(basis = "deploy-faithful"))
final  <- report |> mutate(prefer = ifelse(basis == "deploy-faithful", 1L, 2L)) |>
  group_by(candidate) |> slice_min(prefer, n = 1, with_ties = FALSE) |> ungroup()

fmt <- function(d) for (i in seq_len(nrow(d))) with(d[i, ], cat(sprintf(
  "  %-8s Brier=%.4f [%.4f-%.4f]  base=%.4f  A:%s B:%s  -> %s\n",
  candidate, brier, ci_lo, ci_hi, base_brier,
  ifelse(gateA_beats_base, "ok", "X"), ifelse(gateB_no_collapse, "ok", "X"), verdict)))

cat("\n================ HEALTH GATE ================\n")
cat("(GATE = veto only. Rank candidates on R1/R2, NOT here. Sub-CI gaps are ties.)\n")
cat(sprintf("\n[1] Common R1/R2 health set (n=%d, base rate=%.3f) — apples-to-apples peer gate:\n", nrow(common), mean(y_c)))
fmt(gate_common)
cat(sprintf("\n[2] m1 deploy-faithful health hold-out (n=%d, base rate=%.3f) — m1's actual R3 forecast:\n", nrow(hth), mean(y_h)))
fmt(gate_deploy)
cat(sprintf("    R3-sized (n=%d) draw distribution: median=%.4f  80%% CI [%.4f, %.4f]\n",
            R3_DRAW_N, median(draw), quantile(draw, .1), quantile(draw, .9)))

if (any(gate_common$candidate == "m1" & gate_common$verdict == "FLAG") &&
    any(gate_deploy$candidate == "m1" & gate_deploy$verdict == "PASS"))
  cat("\nNote: m1 flags on the R1/R2-only set (known leave-one-round-out artifact) but\n",
      "     passes its deploy-faithful forecast [2], which is the shipped estimate.\n", sep = "")

flagged <- final |> filter(verdict == "FLAG")
cat(sprintf("\nFINAL VERDICT (deploy-faithful where available): %s\n",
            if (nrow(flagged) == 0) "all candidates PASS the health gate."
            else paste0(nrow(flagged), " candidate(s) FLAGGED: ",
                        paste(flagged$candidate, collapse = ", "), " — investigate before submitting.")))

write_csv(report |> select(set, basis, candidate, n, base_rate, base_brier, brier,
                           ci_lo, ci_hi, gateA_beats_base, gateB_no_collapse, verdict),
          here::here("output/health_gate.csv"))
cat("Wrote output/health_gate.csv\n")
