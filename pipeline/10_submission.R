# =============================================================================
# 10_submission.R  —  Final m1/m2/m3 submission for COS Round 3
#   "Decorrelated trio": ElNet+ feature model + prior×evidence + raw LLM crowd
#   (updated 2026-06-05; rationale in ROUND3_SUBMISSION_REPORT.{md,html})
# =============================================================================
#
# SCORING CONTEXT
# ---------------
# The challenge accepts THREE prediction columns (m1/m2/m3) and scores the BEST
# of the three (lowest single Brier). The goal is therefore three *decorrelated*
# competitive bets that minimise E[min(B1,B2,B3)], NOT three columns that each
# minimise average error. See ROUND3_SUBMISSION_REPORT.md §1.
#
# THE THREE SLOTS (all 45 R3 claims covered; no missing values)
#   m1 — ElNet+ : elastic-net logistic (glmnet, 1-SE lambda) on the
#        "transferable_lean_plus" feature set, median-aggregated over folds and
#        temperature-calibrated. The only column that learns from outcome labels.
#        Fit and scored from scratch in pipeline/09_final_comparison.R; the deployed
#        per-claim R3 column is exported to output/r3_elnet_plus_predictor.csv.
#   m2 — prior×evidence : LLM substantive-plausibility prior combined with a
#        power-for-observed-effect shrinkage evidence term, logistic-combined on
#        labelled health rows. Built by the prior POC; per-claim R3 predictor in
#        output/r3_prior_shrinkage_predictor.csv (see PRIOR_SHRINKAGE_REPORT.md).
#   m3 — raw LLM crowd : zero-parameter synthetic-crowd probability
#        (3 model families x 4 personas), judge_llm_ensemble/ensemble_scores.csv.
#
# WHY THIS TRIO (see ROUND3_SUBMISSION_REPORT.md §4)
#   The three candidates are statistically tied on central Brier at n=45 (every
#   paired bootstrap CI includes 0). They were selected for being the three
#   least-correlated competitive predictors (mean pairwise r ~ 0.62 on R3), which
#   is the lever that lowers the expected minimum under best-of-three scoring.
#
# VALIDATED REFERENCE NUMBERS (honest health OOF, challenge-health analog)
#   per-slot Brier (calibrated):  ElNet+ ~0.211   prior ~0.213   raw-LLM ~0.222
#   R3 correlations:  m1-m2 ~0.64   m1-m3 ~0.56   m2-m3 ~0.64   (mean ~0.62)
#
# REPRODUCE
#   This script ASSEMBLES the deliverable from the three deployed per-claim
#   predictor artifacts (each produced and validated by its own upstream stage,
#   so the numbers can be trusted independently of the assembly step):
#     Rscript pipeline/09_final_comparison.R   # fits/scores ElNet+ (and all cands)
#     Rscript pipeline/10_submission.R      # assembles + writes the trio
#
# INPUTS
#   output/r3_elnet_plus_predictor.csv          claim_id, pred  (m1, ElNet+)
#   output/r3_prior_shrinkage_predictor.csv     claim_id, pred  (m2, prior+evid)
#   judge_llm_ensemble/ensemble_scores.csv      claim_id, dataset, p_replication_observed (m3)
#
# OUTPUTS
#   output/submission.csv             claim_id, m1, m2, m3 (the deliverable)
#   output/submission_backup_*.csv    timestamped backup of any prior submission
#   output/submission_diagnostics.csv tidy metric/value diagnostics
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tibble); library(tidyr)
})

clip01 <- function(x) pmin(pmax(x, 0.01), 0.99)

ROOT     <- here::here()
ELNET_F  <- file.path(ROOT, "output/r3_elnet_plus_predictor.csv")
PRIOR_F  <- file.path(ROOT, "output/r3_prior_shrinkage_predictor.csv")
LLM_F    <- file.path(ROOT, "judge_llm_ensemble/ensemble_scores.csv")

# ── 1. Load the three deployed per-claim predictors ──────────────────────────
elnet <- read_csv(ELNET_F, show_col_types = FALSE) |>
  transmute(claim_id, m1 = clip01(pred))

prior <- read_csv(PRIOR_F, show_col_types = FALSE) |>
  transmute(claim_id, m2 = clip01(pred))

llm <- read_csv(LLM_F, show_col_types = FALSE) |>
  filter(dataset == "round3") |>
  transmute(claim_id, m3 = clip01(p_replication_observed))

# ── 2. Assemble the trio (inner join on claim_id, ElNet+ order) ──────────────
trio <- elnet |>
  inner_join(prior, by = "claim_id") |>
  inner_join(llm,   by = "claim_id")

# ── 3. Sanity checks ─────────────────────────────────────────────────────────
stopifnot(
  "trio has 45 rows"  = nrow(trio) == 45L,
  "no NA predictions" = !anyNA(trio[, c("m1", "m2", "m3")]),
  "m1 in [0,1]" = all(trio$m1 >= 0 & trio$m1 <= 1),
  "m2 in [0,1]" = all(trio$m2 >= 0 & trio$m2 <= 1),
  "m3 in [0,1]" = all(trio$m3 >= 0 & trio$m3 <= 1),
  "m1 not flat" = sd(trio$m1) > 0.02,
  "m2 not flat" = sd(trio$m2) > 0.02,
  "m3 not flat" = sd(trio$m3) > 0.02
)

# ── 4. Diagnostics (R3 correlations: low = good diversification) ─────────────
cmat <- round(cor(trio[, c("m1", "m2", "m3")]), 3)
cat("\n--- Trio R3 correlations (low = good diversification) ---\n"); print(cmat)
cat(sprintf("  mean pairwise r = %.3f\n",
            mean(c(cmat["m1","m2"], cmat["m1","m3"], cmat["m2","m3"]))))

diag <- tibble(
  metric = c("m1_source", "m2_source", "m3_source", "n_claims",
             "cor_m1_m2", "cor_m1_m3", "cor_m2_m3", "mean_pairwise_r",
             "m1_mean", "m1_sd", "m2_mean", "m2_sd", "m3_mean", "m3_sd"),
  value  = as.character(c(
    "ElNet+ (transferable_lean_plus, calibrated)", "prior x evidence", "raw LLM crowd", nrow(trio),
    cmat["m1","m2"], cmat["m1","m3"], cmat["m2","m3"],
    round(mean(c(cmat["m1","m2"], cmat["m1","m3"], cmat["m2","m3"])), 3),
    round(mean(trio$m1), 3), round(sd(trio$m1), 3),
    round(mean(trio$m2), 3), round(sd(trio$m2), 3),
    round(mean(trio$m3), 3), round(sd(trio$m3), 3)))
)
write_csv(diag, file.path(ROOT, "output/submission_diagnostics.csv"))

# ── 5. Write submission (with timestamped backup) ────────────────────────────
sub <- trio |> transmute(claim_id, m1 = round(m1, 6), m2 = round(m2, 6), m3 = round(m3, 6))
out_path <- file.path(ROOT, "output/submission.csv")
if (file.exists(out_path))
  file.copy(out_path,
            file.path(ROOT, sprintf("output/submission_backup_%s.csv",
                                    format(Sys.time(), "%Y%m%d_%H%M%S"))),
            overwrite = FALSE)
write_csv(sub, out_path)

cat("\n=== Submission distribution ===\n")
print(sub |> pivot_longer(-claim_id, names_to = "slot", values_to = "p") |>
        group_by(slot) |>
        summarise(mean = round(mean(p), 3), sd = round(sd(p), 3),
                  min = round(min(p), 3), max = round(max(p), 3), .groups = "drop"))
cat(sprintf("\nWrote: %s (%d rows) and output/submission_diagnostics.csv\n", out_path, nrow(sub)))
cat("Done.\n")
