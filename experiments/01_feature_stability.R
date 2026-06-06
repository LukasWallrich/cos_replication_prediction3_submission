# =============================================================================
# 01_feature_stability.R
# Feature Predictiveness × Train→Test Distributional Stability Diagnostic
# =============================================================================
#
# INPUTS  : data/train_base.rds, data/test_base.rds
# OUTPUTS : output/feature_stability.csv, output/feature_stability.pdf
#
# RATIONALE
# A feature that correlates with the outcome in training is only useful in
# production if its distribution on the test set resembles training.  When
# train and test are drawn from different strata (e.g., different journals,
# years, or corpora), many engineered features shift substantially. A model
# that relied on a shifted feature will produce biased predictions on test.
#
# This script ranks every numeric feature by a *joint* criterion that rewards
# predictive correlation AND penalises distributional shift.
#
# SMD FORMULA
# We use the standardised mean difference  smd = (mean_test − mean_train) / sd_train.
# Dividing by the TRAIN SD (not the pooled SD) is deliberate: we care how far
# the test mean sits in the training distribution, since the model is trained on
# train and applied to test.  A one-sided (train-referenced) Cohen's-d analogy.
#
# STABILITY SCORE (heuristic, re-tune as needed)
#   stability_score = abs_rho − 0.1 × |smd|
# Coefficient 0.1 means one unit of SMD costs 0.1 of Spearman ρ.
# This is intentionally conservative — features rarely shift by more than 1–2
# SDs, so the penalty is at most 0.1–0.2 ρ units.  Increase the coefficient
# (e.g. 0.2) to be more aggressive about penalising shift.
#
# OUTPUT GROUPINGS
#   1. Top 20 by stability_score   → "use these" nominees
#   2. Top 10 by abs_rho with |smd| > 0.5  → "predictive but shifted — beware"
#   3. Top 20 by |smd|             → worst covariate-shift candidates
# =============================================================================

# ── 0. Bootstrap & config ────────────────────────────────────────────────────
source(here::here("pipeline/00_packages_config.R"))

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
}
library(ggplot2)
if (!requireNamespace("ggrepel", quietly = TRUE)) {
  install.packages("ggrepel", repos = "https://cloud.r-project.org")
}
library(ggrepel)

# ── 1. Load data ──────────────────────────────────────────────────────────────
train <- readRDS(here::here("data/train_base.rds"))
test  <- readRDS(here::here("data/test_base.rds"))

cat("Train rows:", nrow(train), "| Test rows:", nrow(test), "\n")

# ── 2. Feature universe ────────────────────────────────────────────────────────
# Exclude IDs, outcome, and any replication-side (leak) columns
STOPLIST <- c(
  "entry_id", "effect_id",
  # outcome and COS phase flag
  "cos_phase1", "statistical_success",
  # replication-side numeric columns (would be data leakage)
  "n_r", "es_value_r", "pval_value_r", "r_r", "year_r",
  "duration_o_r", "duration_o_r_sqrtabs",
  "is_score_replication",
  # derived replication flags that encode the outcome
  "prereg_r_bin",
  # author-overlap cols referencing replication team
  "author_overlap_bin", "author_overlap_missing",
  "same_author_country", "same_author_country_missing"
)

num_cols <- train |>
  dplyr::select(where(is.numeric)) |>
  names()

feature_cols <- setdiff(num_cols, STOPLIST)
cat("Numeric features after stoplist:", length(feature_cols), "\n")

# ── 3. Training rows with observed outcome ─────────────────────────────────────
train_labeled <- train[!is.na(train$statistical_success), ]
cat("Train rows with observed outcome:", nrow(train_labeled), "\n")

# ── 4. Compute per-feature statistics ─────────────────────────────────────────
compute_stats <- function(col) {
  y    <- train_labeled$statistical_success
  x_tr <- train_labeled[[col]]
  x_te <- test[[col]]

  n_obs_train  <- sum(!is.na(x_tr))
  pct_na_train <- mean(is.na(x_tr))
  pct_na_test  <- if (col %in% names(test)) mean(is.na(x_te)) else NA_real_

  # Spearman correlation with outcome
  rho <- tryCatch(
    cor(x_tr, y, use = "pairwise.complete.obs", method = "spearman"),
    error = function(e) NA_real_
  )

  # Train distribution moments
  mean_train <- mean(x_tr, na.rm = TRUE)
  sd_train   <- sd(x_tr, na.rm = TRUE)

  # Test mean (might be missing if column absent in test)
  mean_test  <- if (col %in% names(test)) mean(x_te, na.rm = TRUE) else NA_real_

  # SMD: referenced to train SD
  smd <- if (!is.na(sd_train) && sd_train > 0 && !is.na(mean_test)) {
    (mean_test - mean_train) / sd_train
  } else {
    NA_real_
  }

  data.frame(
    feature      = col,
    n_obs_train  = n_obs_train,
    pct_na_train = pct_na_train,
    pct_na_test  = pct_na_test,
    spearman_rho = rho,
    abs_rho      = abs(rho),
    mean_train   = mean_train,
    mean_test    = mean_test,
    smd          = smd,
    abs_smd      = abs(smd),
    stringsAsFactors = FALSE
  )
}

results_list <- lapply(feature_cols, compute_stats)
results      <- do.call(rbind, results_list)

# ── 5. Stability score ─────────────────────────────────────────────────────────
# stability_score = abs_rho − 0.1 × |smd|
# Coefficient 0.1 is a starting heuristic — adjust upward to penalise shift more.
PENALTY_COEF <- 0.1
results$stability_score <- results$abs_rho - PENALTY_COEF * results$abs_smd

# Sort
results <- results[order(-results$stability_score, na.last = TRUE), ]

# ── 6. Write CSV ───────────────────────────────────────────────────────────────
dir.create("output", showWarnings = FALSE)
out_cols <- c("feature", "n_obs_train", "pct_na_train", "pct_na_test",
              "spearman_rho", "abs_rho", "mean_train", "mean_test",
              "smd", "abs_smd", "stability_score")
write.csv(results[, out_cols], here::here("output/feature_stability.csv"), row.names = FALSE)
cat("Wrote output/feature_stability.csv\n")

# ── 7. Console summaries ───────────────────────────────────────────────────────
fmt <- function(df) {
  df_fmt <- df[, c("feature", "spearman_rho", "abs_rho", "smd", "abs_smd", "stability_score")]
  df_fmt[, 2:6] <- lapply(df_fmt[, 2:6], function(x) round(x, 4))
  df_fmt
}

cat("\n", strrep("=", 70), "\n")
cat("TOP 20 by stability_score (predictive AND stable — use these)\n")
cat("Interpretation: highest joint value of abs_rho penalised by |smd|.\n")
cat(strrep("-", 70), "\n")
print(fmt(head(results, 20)), row.names = FALSE)

beware_set <- results[!is.na(results$abs_smd) & results$abs_smd > 0.5, ]
beware_set <- beware_set[order(-beware_set$abs_rho), ]
cat("\n", strrep("=", 70), "\n")
cat("TOP 10 by abs_rho where |smd| > 0.5 (predictive but SHIFTED — beware)\n")
cat("Interpretation: correlated with outcome but train & test distributions differ;\n")
cat("  using these features raw risks model miscalibration on test.\n")
cat(strrep("-", 70), "\n")
print(fmt(head(beware_set, 10)), row.names = FALSE)

shift_set <- results[order(-results$abs_smd, na.last = TRUE), ]
cat("\n", strrep("=", 70), "\n")
cat("TOP 20 by |smd| irrespective of predictiveness (covariate-shift candidates)\n")
cat("Interpretation: largest distributional gap between train and test;\n")
cat("  even low-rho features here signal domain differences worth understanding.\n")
cat(strrep("-", 70), "\n")
print(fmt(head(shift_set, 20)), row.names = FALSE)

# Summary counts
n_beware <- nrow(beware_set)
cat("\n", strrep("=", 70), "\n")
cat(sprintf("Total features evaluated:            %d\n", nrow(results)))
cat(sprintf("Features with |smd| > 0.5:           %d\n",
            sum(!is.na(results$abs_smd) & results$abs_smd > 0.5)))
cat(sprintf("'Predictive but shifted' (beware) set: %d features\n", n_beware))
cat(strrep("=", 70), "\n\n")

# ── 8. Scatter plot ────────────────────────────────────────────────────────────
plot_df <- results[!is.na(results$abs_smd) & !is.na(results$abs_rho), ]
top20   <- head(results[!is.na(results$stability_score), ], 20)

p <- ggplot(plot_df, aes(x = abs_smd, y = abs_rho)) +
  geom_point(alpha = 0.4, colour = "#3B82F6", size = 1.5) +
  geom_point(data = top20, aes(x = abs_smd, y = abs_rho),
             colour = "#EF4444", size = 2.5) +
  ggrepel::geom_text_repel(
    data = top20,
    aes(label = feature),
    size = 2.2, max.overlaps = 30, segment.alpha = 0.4
  ) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.52, y = max(plot_df$abs_rho, na.rm = TRUE) * 0.95,
           label = "|smd| = 0.5", hjust = 0, size = 3, colour = "grey40") +
  labs(
    title    = "Feature Predictiveness vs. Distributional Stability",
    subtitle = "Red = top-20 by stability_score  |  dashed = |smd| = 0.5 threshold",
    x        = "|SMD| (train to test shift, in train SDs)",
    y        = "|Spearman rho| with statistical_success",
    caption  = sprintf("stability_score = abs_rho - %.1f * |smd|  |  n = %d features",
                       PENALTY_COEF, nrow(plot_df))
  ) +
  theme_bw(base_size = 11)

ggsave(here::here("output/feature_stability.pdf"), p, width = 9, height = 6)
cat("Wrote output/feature_stability.pdf\n")
