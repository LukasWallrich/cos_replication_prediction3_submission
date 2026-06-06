# =============================================================================
# extract_llm_priors.R
# Load one or more LLM prior JSON files, validate them, and turn each into a
# (location, scale) prior object aligned to PREDICTORS_BAYES for use with
# bayes_llmprior_model.R (fit_bayes_logistic_llmprior / fit_data_reference /
# prior_kl_gate / oof_brier_bayes).
#
# Outputs:
#   priors_by_model  : named list, each = list(location, scale) ready to fit
#   comparison       : tidy cross-model table (directions, locations, scales)
#   priors_pooled    : optional linear-opinion-pool prior across models
# =============================================================================

suppressPackageStartupMessages({
  library(jsonlite); library(dplyr); library(tibble)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

# The prior objects below are built entirely from the shipped LLM-elicitation
# JSON files and contain no restricted challenge data, so they ship in
# bayesian_priors/priors_by_model.rds. The labelled feature matrices are only
# needed by the validation / usage demo at the bottom of this script; they are
# loaded lazily there so prior generation runs from the JSONs alone.

PREDICTORS_BAYES <- c(
  "n_o_log", "pval_z", "llm_perceived_surprisingness", "llm_sample_adequacy",
  "llm_within_paper_rep", "llm_is_intervention", "es_directional",
  "n_f_tests_tei_log1p")

# SD from a 90% normal interval, floored to curb elicited overconfidence.
# NOTE: the elicited sds here are ~0.21-0.30, so the model script's default
# floor of 0.30 would flatten almost all of them and erase each model's
# confidence structure. A low floor (0.10) only binds for genuine outliers.
SD_FLOOR <- 0.10

ci90_to_sd <- function(lo, hi, sd_floor = SD_FLOOR)
  max((as.numeric(hi) - as.numeric(lo)) / 3.29, sd_floor)

parse_prior_json <- function(path, predictors = PREDICTORS_BAYES, sd_floor = SD_FLOOR) {
  js   <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  recs <- js$priors %||% js
  out  <- tibble::tibble(predictor = predictors, location = NA_real_,
                         scale = NA_real_, direction = NA_character_)
  for (r in recs) {
    p <- r$predictor
    if (is.null(p) || !p %in% predictors) next
    i <- match(p, out$predictor)
    out$location[i]  <- as.numeric(r$central_log_odds)
    out$scale[i]     <- ci90_to_sd(r$ci90_low, r$ci90_high, sd_floor)
    out$direction[i] <- as.character(r$direction %||% NA_character_)
  }
  miss <- out$predictor[is.na(out$location) | is.na(out$scale)]
  if (length(miss))
    stop(basename(path), " is missing prior(s) for: ", paste(miss, collapse = ", "))
  # Sign sanity: stated direction vs sign of the central estimate.
  flip <- which((out$direction == "positive" & out$location < 0) |
                (out$direction == "negative" & out$location > 0))
  if (length(flip))
    warning(basename(path), ": direction/sign mismatch for ",
            paste(out$predictor[flip], collapse = ", "))
  out
}

# Wrap a parsed tibble into the list(location, scale) the model script expects,
# both vectors named and ordered exactly to PREDICTORS_BAYES.
as_prior_object <- function(tbl, predictors = PREDICTORS_BAYES) {
  tbl <- tbl[match(predictors, tbl$predictor), ]
  list(location = setNames(tbl$location, predictors),
       scale    = setNames(tbl$scale,    predictors))
}

# ── Map model label -> file path (add the GPT file once available) ───────────
prior_files <- c(
  claude48    = "bayesian_priors/priors_claude48.json",
  gemini31pro = "bayesian_priors/priors_gemini31pro.json",
  gpt55   = "bayesian_priors/priors_gpt55.json"
)
present      <- file.exists(prior_files)
if (any(!present))
  message("Missing file(s), skipped: ",
          paste(names(prior_files)[!present], collapse = ", "))
prior_files  <- prior_files[present]

parsed          <- lapply(prior_files, parse_prior_json)
priors_by_model <- lapply(parsed, as_prior_object)   # <- feed these to the model

# ── (a) cross-model comparison (eyeball directions & magnitudes) ─────────────
stopifnot(length(parsed) >= 1)   # guard: fail clearly if no files were loaded

comparison <- dplyr::bind_rows(parsed, .id = "model") |>
  dplyr::mutate(eff_ci90_low  = round(location - 1.6449 * scale, 3),
                eff_ci90_high = round(location + 1.6449 * scale, 3),
                location = round(location, 3),
                scale    = round(scale, 3)) |>
  dplyr::arrange(match(predictor, PREDICTORS_BAYES), model) |>
  dplyr::select(predictor, model, direction, location, scale,
                eff_ci90_low, eff_ci90_high)

print(as.data.frame(comparison), row.names = FALSE)
dir.create(here::here("output"), showWarnings = FALSE)
readr::write_csv(comparison, here::here("output/llm_priors_comparison.csv"))
saveRDS(priors_by_model, here::here("bayesian_priors/priors_by_model.rds"))

# ── (b) OPTIONAL pooled prior (linear opinion pool) ──────────────────────────
# Mean of locations; scale combines mean within-model variance with between-
# model disagreement: sqrt(mean(sd^2) + var(location)). The brief's plan is to
# instead pick the single model with the smallest KL to the data reference
# (prior_kl_gate); pooling is offered only as an alternative.
pool_priors <- function(parsed_list, predictors = PREDICTORS_BAYES) {
  L <- sapply(parsed_list, function(t) t$location[match(predictors, t$predictor)])
  S <- sapply(parsed_list, function(t) t$scale[match(predictors, t$predictor)])
  if (is.null(dim(L))) { L <- matrix(L, nrow = length(predictors)); S <- matrix(S, nrow = length(predictors)) }
  loc <- rowMeans(L)
  btw <- if (ncol(L) > 1) apply(L, 1, var) else rep(0, nrow(L))
  scl <- sqrt(rowMeans(S^2) + btw)
  list(location = setNames(loc, predictors), scale = setNames(scl, predictors))
}
priors_pooled <- pool_priors(parsed)

# ── (b) Pooled prior (linear opinion pool) ───────────────────────────────────
# Mean of locations; scale combines mean within-model variance with between-
# model disagreement: sqrt(mean(sd^2) + var(location)).
pool_priors <- function(parsed_list, predictors = PREDICTORS_BAYES) {
  L <- sapply(parsed_list, function(t) t$location[match(predictors, t$predictor)])
  S <- sapply(parsed_list, function(t) t$scale[match(predictors, t$predictor)])
  if (is.null(dim(L))) {
    L <- matrix(L, nrow = length(predictors))
    S <- matrix(S, nrow = length(predictors))
  }
  loc <- rowMeans(L)
  btw <- if (ncol(L) > 1) apply(L, 1, var) else rep(0, nrow(L))
  scl <- sqrt(rowMeans(S^2) + btw)
  list(location = setNames(loc, predictors),
       scale    = setNames(scl, predictors))
}

priors_pooled <- pool_priors(parsed)
priors_by_model$pooled <- priors_pooled          # include pooled for downstream use

cat("\n── Pooled prior ──\n")
cat("Location:\n"); print(round(priors_pooled$location, 3))
cat("Scale:\n");    print(round(priors_pooled$scale, 3))

# ── (c) KL divergence: individual LLMs vs pooled, and pairwise ──────────────
# Gaussian KL:  KL( N(mu0,sd0) || N(mu1,sd1) )
kl_gauss <- function(mu0, sd0, mu1, sd1)
  log(sd1 / sd0) + (sd0^2 + (mu0 - mu1)^2) / (2 * sd1^2) - 0.5

# (c.1) Each individual LLM vs the pooled prior ──────────────────────────────
individual_models <- setdiff(names(priors_by_model), "pooled")

kl_vs_pool_detail <- dplyr::bind_rows(lapply(individual_models, function(nm) {
  pr <- priors_by_model[[nm]]
  kl <- kl_gauss(pr$location[PREDICTORS_BAYES], pr$scale[PREDICTORS_BAYES],
                 priors_pooled$location[PREDICTORS_BAYES],
                 priors_pooled$scale[PREDICTORS_BAYES])
  tibble::tibble(model = nm, predictor = PREDICTORS_BAYES,
                 kl_to_pool = round(kl, 4))
}))

kl_vs_pool_wide <- kl_vs_pool_detail |>
  tidyr::pivot_wider(names_from = model, values_from = kl_to_pool)

cat("\n── KL( individual LLM || pooled ) per predictor ──\n")
print(as.data.frame(kl_vs_pool_wide), row.names = FALSE)

kl_vs_pool_summary <- kl_vs_pool_detail |>
  dplyr::group_by(model) |>
  dplyr::summarise(mean_kl = round(mean(kl_to_pool), 4),
                   max_kl  = round(max(kl_to_pool), 4),
                   sum_kl  = round(sum(kl_to_pool), 4),
                   .groups = "drop") |>
  dplyr::arrange(mean_kl)

cat("\nSummary — mean / max / sum KL to pooled:\n")
print(as.data.frame(kl_vs_pool_summary), row.names = FALSE)

# (c.2) Pairwise KL between all models (incl. pooled) ────────────────────────
all_models <- names(priors_by_model)
if (length(all_models) >= 2) {
  pairs <- expand.grid(from = all_models, to = all_models,
                       stringsAsFactors = FALSE) |>
    dplyr::filter(from != to)
  
  pairwise_kl <- dplyr::bind_rows(lapply(seq_len(nrow(pairs)), function(r) {
    a <- priors_by_model[[pairs$from[r]]]
    b <- priors_by_model[[pairs$to[r]]]
    kl <- kl_gauss(a$location[PREDICTORS_BAYES], a$scale[PREDICTORS_BAYES],
                   b$location[PREDICTORS_BAYES], b$scale[PREDICTORS_BAYES])
    tibble::tibble(from = pairs$from[r], to = pairs$to[r],
                   mean_kl = round(mean(kl), 4))
  }))
  
  pairwise_wide <- pairwise_kl |>
    tidyr::pivot_wider(names_from = to, values_from = mean_kl)
  
  cat("\n── Pairwise mean KL( row || col ) ──\n")
  print(as.data.frame(pairwise_wide), row.names = FALSE)
  
  # Which pair is most divergent?
  worst <- pairwise_kl |> dplyr::slice_max(mean_kl, n = 1)
  cat("\nMost divergent pair: ", worst$from, " -> ", worst$to,
      " (mean KL = ", worst$mean_kl, ")\n", sep = "")
}

# (c.3) Per-predictor: which LLM deviates most from consensus? ────────────────
outlier_per_pred <- kl_vs_pool_detail |>
  dplyr::group_by(predictor) |>
  dplyr::slice_max(kl_to_pool, n = 1) |>
  dplyr::ungroup() |>
  dplyr::arrange(dplyr::desc(kl_to_pool))

cat("\n── Largest per-predictor deviation from pooled ──\n")
print(as.data.frame(outlier_per_pred), row.names = FALSE)

# ── Save everything ──────────────────────────────────────────────────────────
saveRDS(priors_by_model, here::here("bayesian_priors/priors_by_model.rds"))          # now includes "pooled"
dir.create(here::here("output"), showWarnings = FALSE)
readr::write_csv(comparison, here::here("output/llm_priors_comparison.csv"))
readr::write_csv(kl_vs_pool_detail, here::here("output/llm_priors_kl_vs_pooled.csv"))

cat("\n── Done. priors_by_model (incl. 'pooled') saved to priors_by_model.rds ──\n")

# ── Validation / usage demo (needs the restricted labelled data) ─────────────
# Everything below requires data/train_base.rds and data/test_base.rds, which
# are NOT redistributed (see DATA.md). The shipped priors_by_model.rds written
# above is all the downstream pipeline (09_final_comparison.R) needs, so skip
# this demo when the labelled matrices are absent.
.has_restricted <- file.exists(here::here("data/train_base.rds")) &&
                   file.exists(here::here("data/test_base.rds"))
if (!.has_restricted) {
  message("Restricted data (train_base/test_base) not present — skipping the ",
          "validation/usage demo. priors_by_model.rds has been written.")
} else {
train_base <- readRDS(here::here("data/train_base.rds"))
test_base  <- readRDS(here::here("data/test_base.rds"))

# ── How to use with bayes_llmprior_model.R ───────────────────────────────────
source(here::here("bayesian_priors/bayes_llmprior_model.R"))   # kl_gauss, fit_data_reference, etc.
source(here::here("pipeline/07_calibration.R"))          # fit_calibrator (schon geladen)

base_df   <- add_log1p_feature(train_base)
base_df$is_test_similar <- ifelse(base_df$dataset %in% c("round1", "round2") & base_df$health_related_big == 1, 1, 0)
ref_scope <- base_df[base_df$health_related_big == 1, ]            # 107 health claims
fit_scope <- base_df[base_df$is_test_similar == 1, ]      # 47 test-similar claims (Fit-Scope)
pr <- priors_by_model$pooled               # or $claude48 / $gemini31pro / $gpt55

# KL gate against data reference (per model + pooled):
ref <- fit_data_reference(ref_scope, "statistical_success", PREDICTORS_BAYES)
for (nm in names(priors_by_model)) {
  g <- prior_kl_gate(priors_by_model[[nm]]$location,
                     priors_by_model[[nm]]$scale, ref)
  cat("\n== ", nm, " mean KL =", round(mean(g$kl), 3), "==\n"); print(g)
}

# final fit with the chosen prior:
fit_final <- fit_bayes_logistic_llmprior(fit_scope, "statistical_success",
               PREDICTORS_BAYES, pr$location, pr$scale)
summary(fit_final$fit)

# =============================================================================
# FULL WORKFLOW — after extract_llm_priors.R has run
# =============================================================================
outcome   <- "statistical_success"
# base_df, fit_scope, ref_scope existieren bereits vom Block oben
r3_df     <- add_log1p_feature(test_base)

# ── 1. Choose prior set ──────────────────────────────────────────────────────
pr <- priors_by_model$pooled

cat("\n── Prior scales ──\n")
print(round(data.frame(location = pr$location, scale = pr$scale), 3))

# ── 2. KL gate ───────────────────────────────────────────────────────────────
ref      <- fit_data_reference(ref_scope, outcome, PREDICTORS_BAYES)
gate     <- prior_kl_gate(pr$location, pr$scale, ref)
cat("\n── KL gate (pooled) ──\n")
print(as.data.frame(gate), row.names = FALSE)

# ── 3. Final fit ─────────────────────────────────────────────────────────────
fit_final <- fit_bayes_logistic_llmprior(
  fit_scope, outcome, PREDICTORS_BAYES, pr$location, pr$scale)
cat("\n── Posterior summary ──\n")
print(summary(fit_final$fit))

# ── 4. OOF Brier ────────────────────────────────────────────────────────────
oof <- oof_brier_bayes(fit_scope, outcome, PREDICTORS_BAYES,
                       pr$location, pr$scale)
cat("\nOOF Brier raw:", round(oof$brier_raw, 4),
    "| temp:", round(oof$brier_temp, 4), "\n")

# ── 5. Round-3 predictions ──────────────────────────────────────────────────
p_r3 <- pred_bayes_logistic_llmprior(fit_final, r3_df)
cat("\n── Round 3 prediction summary ──\n")
print(summary(p_r3))

readr::write_csv(
  tibble::tibble(claim_id = r3_df$claim_id, p = round(p_r3, 4)),
  here::here("output/round3_bayes_llmprior_preds.csv"))



# ── 6. Compare: pooled-adjusted vs each individual LLM (OOF) ────────────────
# Optional: if you want to know whether the adjusted pooled beats individual LLMs
if (FALSE) {   # set to TRUE to run (~20 min for 3 models × 5 folds)
  for (nm in individual_models) {
    p <- priors_by_model[[nm]]
    o <- oof_brier_bayes(fit_scope, outcome, PREDICTORS_BAYES,
                         p$location, p$scale)
    cat(nm, "OOF Brier raw:", round(o$brier_raw, 4),
        " temp:", round(o$brier_temp, 4), "\n")
  }
}

cat("\n── Done. Predictions saved to output/round3_bayes_llmprior_preds.csv ──\n")




# ── 7. Analyze prior vs posterior ────────────────

# 1. Wie stark spreizen die Predictions?
cat("SD der Predictions (mit LLM-Priors):", round(sd(p_r3), 3), "\n")
cat("Range:", round(range(p_r3), 3), "\n")

# 2. Vergleich: wie sähe es mit völlig uninformativen Priors aus?
np <- length(PREDICTORS_BAYES)
fit_flat <- fit_bayes_logistic_llmprior(
  fit_scope, outcome, PREDICTORS_BAYES,
  prior_loc   = setNames(rep(0, np), PREDICTORS_BAYES),
  prior_scale = setNames(rep(5, np), PREDICTORS_BAYES))

p_r3_flat <- pred_bayes_logistic_llmprior(fit_flat, r3_df)

cat("\n── Vergleich: LLM-Prior vs Flat Prior ──\n")
cat("Mean   — LLM:", round(mean(p_r3), 3),
    "| Flat:", round(mean(p_r3_flat), 3), "\n")
cat("SD     — LLM:", round(sd(p_r3), 3),
    "| Flat:", round(sd(p_r3_flat), 3), "\n")
cat("Range  — LLM:", round(range(p_r3), 3),
    "| Flat:", round(range(p_r3_flat), 3), "\n")

# ── Shrinkage: wie stark ziehen die Priors die Koeffizienten? ────────────────
posterior_llm  <- coef(fit_final$fit)[-1]          # mit LLM-Priors
posterior_flat <- coef(fit_flat$fit)[-1]            # mit Flat Priors

shrinkage <- tibble::tibble(
  predictor      = PREDICTORS_BAYES,
  prior_mean     = round(pr$location, 3),
  posterior_llm  = round(posterior_llm, 3),
  posterior_flat = round(posterior_flat, 3),
  shift_by_prior = round(posterior_llm - posterior_flat, 3)
)
cat("\n── Shrinkage durch LLM-Priors ──\n")
print(as.data.frame(shrinkage), row.names = FALSE)
}  # end validation/usage demo (restricted data present)
