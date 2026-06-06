# ==============================================================================
# 03_consensus_capping.R — defensive Brier hedge for the submission
# ==============================================================================
#
# WHY THIS EXISTS
# ---------------
# Brier loss penalises confident wrongness *asymmetrically* when n_test is
# small. For our 45-claim test set:
#
#   prediction p = 0.99, truth = 0  →  Brier contribution = 0.9801 / 45 ≈ 0.0218
#   prediction p = 0.50, truth = 0  →  Brier contribution = 0.2500 / 45 ≈ 0.0056
#
# So a single confidently-wrong extreme costs ~4x what a uniform prediction
# would cost. The current `output/submission.csv` puts 62% of m1 predictions
# above 0.99 and 33% below 0.01 — if even a handful are wrong, the entire
# Brier budget evaporates. Consensus capping is the round-2 #1 winner's hedge
# against exactly this scenario.
#
# TWO STRATEGIES PROVIDED
# -----------------------
# 1) `apply_shrinkage(preds, lambda, threshold, base_rate)` — independent
#    per-slot shrinkage. When a prediction is extreme (|p - 0.5| > threshold),
#    pull it toward `base_rate` by `(1 - lambda) * p + lambda * base_rate`.
#    Moderate predictions are left alone. This is a single-column hedge.
#
# 2) `apply_consensus_capping(preds_mat, lambda, sd_threshold, p_threshold,
#                              base_rate)` — multi-base-learner hedge.
#    When ALL base learners agree on an extreme (sd across learners is small
#    AND |mean - 0.5| > p_threshold), pull the mean toward `base_rate`.
#    The intuition is bayesian: agreement on extremes is suspicious when
#    n_test is small, because all learners may share the same systematic
#    error (data artefact, leakage, distribution shift). Disagreement is
#    treated as honest uncertainty and left alone.
#
# DEFAULTS
# --------
#   lambda = 0.3            (structural bound, not a tuned parameter)
#   threshold = 0.4         (extreme = |p - 0.5| > 0.4 → p < 0.10 or p > 0.90)
#   sd_threshold = 0.05     (base learners "agree")
#   p_threshold = 0.35      (extreme mean = mean < 0.15 or mean > 0.85)
#   base_rate = 0.5         (uninformative target; can be overridden to
#                            empirical training base rate ≈ 0.41)
#
# WHY 0.5 vs. EMPIRICAL BASE RATE
# -------------------------------
# Training base rate is ~0.41 (slightly below 50/50). R1+R2 ground truth is
# ~54%. The Round 3 test set is health-only and we don't know its base rate.
# Shrinking toward 0.5 is the maximum-entropy choice and minimises worst-case
# Brier under unknown base rate; shrinking toward the empirical training rate
# only helps if the test set genuinely shares it. Default to 0.5 here; expose
# the parameter so the user can override.
#
# OUTPUTS
# -------
# Writes:
#   output/submission_shrunk.csv    — each slot independently shrunk
#   output/submission_consensus.csv — m1/m3 consensus-capped; m2 left alone
#
# Does NOT overwrite `output/submission.csv`. The user chooses which to submit.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

# ── apply_shrinkage ───────────────────────────────────────────────────────────
# Per-slot independent shrinkage toward base_rate for extreme predictions.
#
# @param preds      numeric vector in [0, 1]
# @param lambda     shrinkage weight in [0, 1]; 0 = no shrinkage, 1 = full pull
# @param threshold  extremity gate: shrink only when |p - 0.5| > threshold
# @param base_rate  the value to shrink toward
# @return shrunk vector clipped to [1e-6, 1 - 1e-6]
apply_shrinkage <- function(preds, lambda = 0.3, threshold = 0.4,
                            base_rate = 0.5) {
  stopifnot(lambda >= 0, lambda <= 1)
  stopifnot(threshold >= 0, threshold < 0.5)
  is_extreme <- abs(preds - 0.5) > threshold
  out <- preds
  out[is_extreme] <- (1 - lambda) * preds[is_extreme] + lambda * base_rate
  pmin(pmax(out, 1e-6), 1 - 1e-6)
}

# ── apply_consensus_capping ───────────────────────────────────────────────────
# Round-2 #1 winner's recipe. Takes a matrix where columns are base-learner
# predictions for the same claim. When learners AGREE (low sd across columns)
# AND their mean is EXTREME (|mean - 0.5| > p_threshold), shrink the mean
# toward base_rate by lambda. Otherwise return the mean unmodified.
#
# Agreement-on-extreme is treated as a red flag: independent learners
# shouldn't all hit p > 0.95 on the same claim by accident — but they CAN
# share a single systematic error. With n_test = 45, the asymmetric Brier
# penalty makes the safer bet to soften the extreme.
#
# @param preds_mat    matrix with one row per claim, one column per learner
# @param lambda       shrinkage weight (default 0.3)
# @param sd_threshold "agreement" gate: shrink only when sd(across cols) < this
# @param p_threshold  "extremity" gate: shrink only when |mean - 0.5| > this
# @param base_rate    target of the shrinkage
# @return numeric vector of length nrow(preds_mat)
apply_consensus_capping <- function(preds_mat, lambda = 0.3,
                                    sd_threshold = 0.05,
                                    p_threshold = 0.35,
                                    base_rate = 0.5) {
  stopifnot(is.matrix(preds_mat) || is.data.frame(preds_mat))
  preds_mat <- as.matrix(preds_mat)
  row_mean <- rowMeans(preds_mat, na.rm = TRUE)
  row_sd   <- apply(preds_mat, 1, sd, na.rm = TRUE)
  trigger  <- !is.na(row_sd) & row_sd < sd_threshold &
              abs(row_mean - 0.5) > p_threshold
  out <- row_mean
  out[trigger] <- (1 - lambda) * row_mean[trigger] + lambda * base_rate
  pmin(pmax(out, 1e-6), 1 - 1e-6)
}

# ── Run on the current submission ─────────────────────────────────────────────
# Only execute the read/transform/write workflow when this file is run directly.
# Other scripts (e.g. 10_submission.R) source this file only to reuse
# `apply_shrinkage()` / `apply_consensus_capping()` — otherwise they would
# process a stale `output/submission.csv`, or error on a clean checkout where
# that file does not yet exist. Such callers must set
# `.consensus_capping_helpers_only <- TRUE` before sourcing.
if (!isTRUE(get0(".consensus_capping_helpers_only",
                 envir = globalenv(), ifnotfound = FALSE))) {

cat("Consensus capping / shrinkage helper for the COS R3 submission.\n")
cat("Rationale: with n_test = 45, confidently-wrong predictions cost ~4x\n")
cat("more than 0.5 predictions. We shrink extreme predictions toward 0.5.\n\n")

# Empirical training base rate (informational; we use 0.5 as the target)
train_base_path <- here::here("data/train_base.rds")
if (file.exists(train_base_path)) {
  tb <- readRDS(train_base_path)
  br_train <- mean(tb$statistical_success, na.rm = TRUE)
  cat(sprintf("Empirical training base rate: %.4f (R1+R2 ground truth)\n",
              br_train))
}

sub <- read_csv(here::here("output/submission.csv"), show_col_types = FALSE)
slots <- c("m1", "m2", "m3")
stopifnot(all(slots %in% names(sub)))

# Strategy A: each slot independently shrunk (lambda = 0.3 default)
sub_shrunk <- sub
for (s in slots) {
  sub_shrunk[[s]] <- apply_shrinkage(sub[[s]], lambda = 0.3,
                                     threshold = 0.4, base_rate = 0.5)
}

# Strategy B: m1 and m3 consensus-capped across the three slots (using m1/m2/m3
# as our three "base learners"). m2 is left as the calibration anchor.
preds_mat <- as.matrix(sub[, slots])
m1_capped <- apply_consensus_capping(preds_mat, lambda = 0.3,
                                     sd_threshold = 0.05, p_threshold = 0.35)
m3_capped <- apply_consensus_capping(preds_mat, lambda = 0.5,
                                     sd_threshold = 0.05, p_threshold = 0.35)
sub_consensus <- tibble(
  claim_id = sub$claim_id,
  m1 = m1_capped,
  m2 = sub$m2,
  m3 = m3_capped
)

# ── Diagnostic table ──────────────────────────────────────────────────────────
diag_stats <- function(name, df) {
  do.call(rbind, lapply(slots, function(s) {
    v <- df[[s]]
    tibble(
      submission = name,
      slot       = s,
      mean       = round(mean(v), 3),
      sd         = round(sd(v), 3),
      below_05   = round(mean(v < 0.05), 2),
      above_95   = round(mean(v > 0.95), 2),
      in_30_70   = round(mean(v > 0.3 & v < 0.7), 2)
    )
  }))
}

diag <- bind_rows(
  diag_stats("original",  sub),
  diag_stats("shrunk",    sub_shrunk),
  diag_stats("consensus", sub_consensus)
)

cat("\nDistribution comparison (per slot):\n")
print(diag, n = 30)

n_shifted_shrunk <- sum(rowSums(abs(sub_shrunk[, slots] - sub[, slots]) > 1e-3) > 0)
n_shifted_consensus <- sum(abs(sub_consensus$m1 - sub$m1) > 1e-3 |
                           abs(sub_consensus$m3 - sub$m3) > 1e-3)
cat(sprintf("\nRows shifted by SHRUNK strategy:   %d / %d\n", n_shifted_shrunk, nrow(sub)))
cat(sprintf("Rows shifted by CONSENSUS strategy: %d / %d\n",
            n_shifted_consensus, nrow(sub)))

# ── Save (without overwriting original) ───────────────────────────────────────
out_shrunk    <- here::here("output/submission_shrunk.csv")
out_consensus <- here::here("output/submission_consensus.csv")
write_csv(sub_shrunk,    out_shrunk)
write_csv(sub_consensus, out_consensus)
cat(sprintf("\nWrote: %s\nWrote: %s\n", out_shrunk, out_consensus))
cat("Original output/submission.csv left UNCHANGED.\n")

}  # end of `if (!isTRUE(.consensus_capping_helpers_only))` guard
