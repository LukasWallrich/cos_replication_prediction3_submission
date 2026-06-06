# 04_base_dataset.R — Master feature-matrix builder
#
# PURPOSE: Joins raw replication data with external and LLM-extracted features,
# harmonizes heterogeneous effect sizes into a common Pearson-r scale, and
# engineers the full set of derived predictors (numeric transforms, claim-text,
# study-design, geography, NLP, scrutiny/GRIM, TEI bias indicators, reference
# recency, log1p/sqrtabs expansions, missingness indicators, and judge-LLM
# ensemble scores). No imputation is performed here.
#
# INPUTS:
#   data/train_raw.rds, data/test_raw.rds        — raw effect/claim-level data
#   data/external_features.rds                   — bibliometric/journal/author features
#   data/cache/llm_features.rds                  — LLM-extracted features (optional)
#   data/external/flora.csv                      — author lists for overlap back-fill
#   data/FORRT_Training_Data.csv                 — claim_id → doi_o mapping
#   judge_llm_ensemble/ensemble_scores.csv       — 12-judge ensemble predictions
#
# OUTPUTS:
#   data/train_base.rds, data/test_base.rds      — engineered feature matrices

source(here::here("pipeline/00_packages_config.R"))

# ── Config toggles ────────────────────────────────────────────────────────────
# Fix A — paper-level LLM fallback for TRAINING rows.
#   The LLM extraction runs ONE claim per paper (llm_extraction/_prep_claims.R), but
#   train_base is effect-level (multiple rows per paper). So non-primary effects
#   (and any paper not picked) have NO claim-level LLM match and end up NA on all
#   llm_* descriptors. The R3 test set already sidesteps this via a paper-level
#   ("paper_id") fallback; train did not, creating a train/test feature mismatch.
#   When TRUE, train rows lacking a claim-level LLM match inherit their paper's
#   extracted LLM features. Claim-anchored fields (surprisingness, sample_adequacy,
#   …) are then borrowed from the paper's primary extracted claim — identical to
#   what the test set already does. Set FALSE to revert to claim-id-only joins.
USE_PAPER_LEVEL_LLM_FALLBACK_TRAIN <- TRUE

train_raw         <- readRDS(here::here("data/train_raw.rds"))         |> mutate(doi_o = normalize_doi(doi_o))
test_raw          <- readRDS(here::here("data/test_raw.rds"))          |> mutate(doi_o = normalize_doi(doi_o))
external_features <- readRDS(here::here("data/external_features.rds"))
llm_features      <- if (file.exists(here::here("data/cache/llm_features.rds")))
  readRDS(here::here("data/cache/llm_features.rds")) else NULL

# ── Back-fill author_overlap from flora (forrtproject/fred-data) ───────────────
# FORRT's author_overlap flag is NA for ~31% of train_raw rows. flora ships the
# full author lists for both original and replication; matching authors on
# first-initial + family reproduces the human flag with 100% concordance on the
# 1528 rows where both exist, and fills ~340 of the gaps. Only NA values are
# back-filled — existing human codings are never overwritten.
flora_overlap <- local({
  fp <- here::here("data/external/flora.csv")
  if (!file.exists(fp)) return(NULL)
  fl <- readr::read_csv(fp, show_col_types = FALSE)
  author_ids <- function(j) {
    if (is.na(j) || !nzchar(j)) return(character(0))
    a <- tryCatch(jsonlite::fromJSON(j), error = function(e) NULL)
    if (is.null(a) || is.null(a$family)) return(character(0))
    gi <- substr(tolower(trimws(ifelse(is.na(a$given), "", a$given))), 1, 1)
    paste0(gi, "|", tolower(trimws(a$family)))
  }
  any_shared <- function(ao, ar) {
    a <- author_ids(ao); b <- author_ids(ar)
    if (!length(a) || !length(b)) return(NA)
    length(intersect(a, b)) > 0
  }
  tibble::tibble(
    doi_o     = normalize_doi(fl$doi_o),
    doi_r     = normalize_doi(fl$doi_r),
    .ao_flora = mapply(any_shared, fl$author_o, fl$author_r)
  ) |>
    dplyr::filter(!is.na(doi_o), !is.na(doi_r), !is.na(.ao_flora)) |>
    dplyr::distinct(doi_o, doi_r, .keep_all = TRUE)
})

if (!is.null(flora_overlap)) {
  train_raw <- train_raw |>
    mutate(.dr = normalize_doi(doi_r)) |>
    left_join(flora_overlap, by = c("doi_o", ".dr" = "doi_r")) |>
    mutate(author_overlap = coalesce(author_overlap, .ao_flora)) |>
    select(-.dr, -.ao_flora)
}

# build_base(): join raw data with external and optional LLM features.
# Round 3 test rows are keyed by paper_id (4-letter codes).
llm_features_test <- llm_features |>
  filter(str_detect(claim_id, "^[A-Z]{4}$")) |>
  rename(paper_id = claim_id)

test_raw <- test_raw |>
  mutate(paper_id = str_extract(claim_id, "^[^_]+"))

build_base <- function(raw, external, llm, llm_key = "claim_id", fallback_key = NULL) {
  df <- raw |>
    left_join(external, by = "doi_o") |>
    select(-any_of("publisher_xml"))

  if (is.null(llm)) return(df)

  # Primary join on llm_key
  df <- df |> left_join(llm, by = llm_key)

  # Fallback join on fallback_key
  if (!is.null(fallback_key) && fallback_key %in% names(raw)) {
    llm_cols <- setdiff(names(llm), llm_key)

    llm_fb <- llm |> rename(!!fallback_key := all_of(llm_key))

    df <- df |>
      left_join(llm_fb, by = fallback_key, suffix = c("", "_fb"))

    # Coalesce: keep primary match, otherwise use fallback
    fb_cols <- paste0(llm_cols, "_fb")
    for (col in llm_cols) {
      fb_col <- paste0(col, "_fb")
      if (fb_col %in% names(df))
        df[[col]] <- coalesce(df[[col]], df[[fb_col]])
    }

    df <- select(df, -any_of(fb_cols))
  }

  df
}

train_joined <- build_base(train_raw, external_features, llm_features,
                           llm_key    = "claim_id",
                           fallback_key = "claim_id_forrt")
test_joined  <- build_base(test_raw,  external_features, llm_features_test, "paper_id")

# ── Fix A: paper-level LLM fallback for training rows ─────────────────────────
# Fill train rows that have no claim-level LLM match with their paper's extracted
# LLM features (one extracted claim per paper). Paper key drops the trailing
# _<effect_id> so "507_3"->"507" while "flora_health_04_1"->"flora_health_04".
if (USE_PAPER_LEVEL_LLM_FALLBACK_TRAIN && !is.null(llm_features)) {
  .paper_uid <- function(x) sub("_[^_]+$", "", x)
  .llm_cols  <- setdiff(names(llm_features), "claim_id")
  .llm_paper <- llm_features |>
    mutate(.pk = .paper_uid(claim_id)) |>
    group_by(.pk) |> slice(1L) |> ungroup() |>      # one extracted claim per paper
    select(.pk, all_of(.llm_cols)) |>
    rename_with(~ paste0(.x, "_pfb"), all_of(.llm_cols))
  .n_before <- sum(!is.na(train_joined$llm_model))
  train_joined <- train_joined |>
    mutate(.pk = .paper_uid(claim_id)) |>
    left_join(.llm_paper, by = ".pk")
  for (col in .llm_cols) {
    pc <- paste0(col, "_pfb")
    if (pc %in% names(train_joined))
      train_joined[[col]] <- coalesce(train_joined[[col]], train_joined[[pc]])
  }
  train_joined <- train_joined |> select(-any_of(paste0(.llm_cols, "_pfb")), -.pk)
  cat(sprintf("Fix A (paper-level LLM fallback): train LLM-matched rows %d -> %d\n",
              .n_before, sum(!is.na(train_joined$llm_model))))
}

# ── Sanity check ──────────────────────────────────────────────────────────────
cat("Train LLM-Match:", sum(!is.na(train_joined$llm_model)), "/", nrow(train_joined), "\n")
cat("Test  LLM-Match:", sum(!is.na(test_joined$llm_model)),  "/", nrow(test_joined),  "\n")

# ── Paper/study-level LLM features: propagate across rows sharing the same doi_o ─
PAPER_LLM_COLS <- c(
  # paper_schema fields
  "llm_open_data", "llm_open_data_request", "llm_open_materials_num",
  "llm_open_code", "llm_is_health", "llm_conflict_of_interest",
  "llm_within_paper_rep", "llm_has_funding", "llm_hypothesis_count",
  "llm_writing_quality_ord", "llm_study_count")
STUDY_LLM_COLS <- c( # claim_schema study-level fields
  "llm_rct_study", "llm_experimental", "llm_observational",
  "llm_cross_sectional", "llm_longitudinal",
  "llm_student_sample", "llm_paid_sample", "llm_online_paid_sample",
  "llm_online_study", "llm_weird_sample",
  "llm_self_report", "llm_validated_scale", "llm_multi_item",
  "llm_apriori_power", "llm_posthoc_power", "llm_power_section",
  "llm_has_robustness", "llm_secondary_data", "llm_preregistration",
  "llm_exploratory_study", "llm_is_intervention", "llm_has_prepost"
)

# claim_id → doi_o mapping from FORRT training data
claim_doi_map <- read_csv(here::here("data/FORRT_Training_Data.csv"),
                          show_col_types = FALSE) |>
  mutate(doi_o    = normalize_doi(doi_o),
         claim_id = paste0(entry_id, "_", effect_id)) |>
  select(claim_id, doi_o) |>
  distinct()

# One row per doi_o: first non-NA value per col (PAPER + STUDY)
llm_paper_features <- llm_features |>
  filter(!str_detect(claim_id, "^[A-Z]{4}$")) |>
  left_join(claim_doi_map, by = "claim_id") |>
  filter(!is.na(doi_o)) |>
  select(doi_o, all_of(intersect(c(PAPER_LLM_COLS, STUDY_LLM_COLS), names(llm_features)))) |>
  group_by(doi_o) |>
  summarise(across(everything(), ~first(na.omit(.))), .groups = "drop")

fill_paper_llm <- function(df, paper_llm, cols) {
  cols_present <- intersect(cols, names(df))
  if (length(cols_present) == 0 || is.null(paper_llm)) return(df)

  tmp <- left_join(
    df,
    select(paper_llm, doi_o, all_of(cols_present)),
    by     = "doi_o",
    suffix = c("", ".pfill")
  )

  for (col in cols_present) {
    fill_col <- paste0(col, ".pfill")
    if (fill_col %in% names(tmp)) {
      tmp[[col]]      <- coalesce(tmp[[col]], tmp[[fill_col]])
      tmp[[fill_col]] <- NULL
    }
  }
  tmp
}

# Flag BEFORE fill: llm_model is NA when there is no direct claim match, so these
# cells receive study-level values only via doi_o propagation.
train_joined <- train_joined |> mutate(llm_filled_via_doi = is.na(llm_model))
test_joined  <- test_joined  |> mutate(llm_filled_via_doi = is.na(llm_model))

# Fill PAPER + STUDY cols per doi_o
train_joined <- fill_paper_llm(train_joined, llm_paper_features, c(PAPER_LLM_COLS, STUDY_LLM_COLS))
test_joined  <- fill_paper_llm(test_joined,  llm_paper_features, c(PAPER_LLM_COLS, STUDY_LLM_COLS))

# Sanity check
cat("Train Paper-LLM fill (open_data):",
    sum(!is.na(train_joined$llm_open_data)), "/", nrow(train_joined), "\n")
cat("Test  Paper-LLM fill (open_data):",
    sum(!is.na(test_joined$llm_open_data)),  "/", nrow(test_joined),  "\n")

# ── Final match check ─────────────────────────────────────────────────────────

# 1. Overall: how many of the JSONs are matched?
cat("=== JSON Match Rate ===\n")
cat("LLM features total:              ", nrow(llm_features), "\n")
cat("Unique matched in train:         ", n_distinct(train_joined$claim_id[!is.na(train_joined$llm_model)]), "\n")
cat("Matched in test (paper_id):      ", sum(!is.na(test_joined$llm_model)), "\n")

forrt_llm_ids <- llm_features$claim_id[!str_detect(llm_features$claim_id, "^[A-Z]{4}$")]
cat("Unmatched (claim_id only):       ",
    sum(!forrt_llm_ids %in% train_raw$claim_id), "\n")
cat("Unmatched (incl. claim_id_forrt):",
    sum(!forrt_llm_ids %in% c(train_raw$claim_id, train_raw$claim_id_forrt)), "\n")

# 2. Duplicates in train_raw?
cat("\n=== Duplicates in train_raw ===\n")
dup <- train_raw |> filter(!is.na(claim_id)) |> count(claim_id) |> filter(n > 1)
cat("claim_ids with duplicates:", nrow(dup), "\n")
if (nrow(dup) > 0) print(dup)

# 3. LLM coverage by dataset x claim_status
cat("\n=== LLM Coverage: Claim-Level ===\n")
train_joined |>
  mutate(has_llm = !is.na(llm_model)) |>
  group_by(dataset, claim_status_llm) |>
  summarise(n = n(), matched = sum(has_llm),
            pct = round(mean(has_llm) * 100, 1), .groups = "drop") |>
  arrange(dataset, claim_status_llm) |>
  print(n = 30)

# 4. LLM coverage by dataset at paper level (doi_o)
cat("\n=== LLM Coverage: Paper-Level (via doi_o fill) ===\n")
train_joined |>
  group_by(dataset) |>
  summarise(
    n_rows         = n(),
    n_direct_match = sum(!is.na(llm_model)),
    n_paper_fill   = sum(!is.na(llm_open_data)),
    pct_paper_fill = round(mean(!is.na(llm_open_data)) * 100, 1),
    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# compute_readability(): Flesch-Kincaid + SMOG per text vector (paper-level)
# requires: quanteda, quanteda.textstats
# ------------------------------------------------------------------------------

compute_readability <- function(texts, prefix) {
  texts_clean <- ifelse(is.na(texts), "", as.character(texts))
  corp        <- quanteda::corpus(texts_clean)
  scores      <- quanteda.textstats::textstat_readability(
    corp,
    measure = c("Flesch.Kincaid", "SMOG")
  )
  word_counts <- quanteda::ntoken(quanteda::tokens(corp))
  scores$SMOG[word_counts < 30]    <- NA_real_
  scores$Flesch.Kincaid[word_counts == 0] <- NA_real_

  tibble(
    !!paste0(prefix, "_fk")   := scores$Flesch.Kincaid,
    !!paste0(prefix, "_smog") := scores$SMOG
  )
}

add_readability <- function(df) {
  papers <- df |> distinct(doi_o, .keep_all = TRUE)

  read_full <- bind_cols(select(papers, doi_o),
                         compute_readability(papers$article_text_xml, "readability_full"))
  read_abs  <- bind_cols(select(papers, doi_o),
                         compute_readability(papers$abstract_xml,     "readability_abs"))

  df |>
    left_join(read_full, by = "doi_o") |>
    left_join(read_abs,  by = "doi_o")
}

train_joined <- add_readability(train_joined)
test_joined  <- add_readability(test_joined)

# ------------------------------------------------------------------------------
# harmonize_effect_sizes(): unify heterogeneous effect-size types into Pearson r
# ------------------------------------------------------------------------------
#
# WHY: ~31% of the test set reports the primary effect as something other than a
# Pearson r (odds/hazard ratios, Cohen's d, η², β, etc.). Round 2 #2 winners
# cited effect-size harmonization as their single biggest contribution. This
# function builds `r_harmonized` ∈ [-1, 1] from whatever effect size (or test
# statistic, or p+n) is available so that downstream proxies aren't NA when
# r_o is missing.
#
# CONVERSIONS (all cite Borenstein 2009 / Lipsey-Wilson 2001 / Chinn 2000):
#   r_family           : identity                        — copy r_o (else es_value_o)
#   partial_r          : copy as r (proxy; not zero-order r) — LOW CONFIDENCE
#   partial_r_squared  : r = sqrt(partial_r²)            — LOW CONFIDENCE
#   smd_between        : r = d / sqrt(d^2 + 4)           — Borenstein 2009 eq. 7.5
#   smd_within (dz)    : treat as d (assumes rho≈0.5)    — LOW CONFIDENCE
#   variance_explained : r = sqrt(eta^2)   (η², ω², R²)  — sign unrecoverable, kept +
#                        r = sqrt(f^2 / (1 + f^2))       for Cohen's f²
#   or_family          : Chinn 2000:  d ≈ log(OR) * sqrt(3)/π   then d → r
#                        non-positive OR → NA (data error, not floored)
#                        (HR/IRR/RR treated as OR proxy) — LOW CONFIDENCE
#   nonparametric      : φ ≈ r (2×2), Cramér's V ≈ r upper bound,
#                        Wilcoxon r is already an r, Cohen's h: r ≈ h/2
#                                                              — LOW CONFIDENCE
#   regression_std     : β as single-predictor approximation to r,
#                        clamped to [-0.95, 0.95]              — LOW CONFIDENCE
#   regression_unstd   : b/B — NOT convertible without SD → NA
#   latent_scale_coef  : probit / ordered-probit coefficients — NOT convertible
#                        (latent-z scale ≠ standardised β)    → NA
#   non_convertible    : Cohen's q (difference of Fisher-z'd r's) → NA
#   raw_difference     : mean/median diff — NOT convertible without SD → NA
#   test_stat_only     : from t: r = t / sqrt(t^2 + df),  df = n - 2
#                        requires n > 2; NA otherwise (no df-floor fabrication)
#
# P+N FALLBACK ("synthesized_from_pn"):
#   When no effect size is available but p and n are: invert p → t (df = n - 2)
#   then t → r. This is the post-hoc MINIMUM-DETECTABLE-EFFECT under
#   significance, NOT an estimate of the true effect. We cap |r| ≤ 0.30 to
#   avoid feeding a strong fake signal to the model. Sign defaulted via
#   directional keywords in claim text (increase/higher/positive → +, etc.).
#
# CAPS:
#   * Final |r_harmonized| ≤ 0.95 (numerical stability for r→t conversions),
#     flagged in r_harmonized_capped.
#   * P+n fallback further capped at 0.30 (post-hoc MDE caveat above).
#
# LOW-CONFIDENCE FLAG (`r_harmonized_low_confidence` = 1):
#   methods ∈ {from_dz, from_or, from_beta, from_partial_r,
#              from_nonparam, synthesized_from_pn}
#   so the downstream model can learn to downweight these.
#
# DERIVED PROXIES (added to df):
#   t_proxy_from_r_h      — back-computed t from r_harmonized and n
#   snr_proxy_h           — t² / n  (signal-to-noise proxy)
#   power_proxy_h         — unbounded evidence score: |r| * sqrt(n) / (p + 0.01)
#                           NOTE: this is NOT statistical power ∈ [0,1]
#   evidence_strength_score — pval_z * log10(n) * |r_harmonized|, capped ±50
# ------------------------------------------------------------------------------

harmonize_effect_sizes <- function(df) {

  # ── ES-type map (mirrors apply_features() so we don't depend on call order) ─
  es_type_map <- c(
    "r" = "r_family", "pearson_r" = "r_family",
    # Partial correlations are NOT zero-order r; give them their own category
    # so we can convert (or skip) them correctly instead of copying as r.
    "partial r" = "partial_r", "partial correlation" = "partial_r",
    "Partial correlation coefficient" = "partial_r",
    "partial r-square" = "partial_r_squared",
    "d" = "smd_between", "Cohen's d" = "smd_between", "cohen_d" = "smd_between",
    "hedges' g" = "smd_between", "dv" = "smd_between",
    "Standard deviation units" = "smd_between",
    "Cohen's dz" = "smd_within", "cohen_dz" = "smd_within", "dz" = "smd_within",
    "etasq" = "variance_explained", "etasq (partial)" = "variance_explained",
    "partial_eta_squared" = "variance_explained",
    "r-squared" = "variance_explained", "omega squared" = "variance_explained",
    "cohen_f_squared" = "variance_explained", "Cohen's f^2" = "variance_explained",
    "cohen's f" = "variance_explained",
    "Eta-squared" = "variance_explained", "eta_squared" = "variance_explained",
    "b (unstd)" = "regression_unstd", "be" = "regression_unstd",
    "regression_coefficient" = "regression_unstd",
    "beta" = "regression_std", "beta (std)" = "regression_std",
    "path_coefficient" = "regression_std",
    # Probit / ordered-probit coefficients live on the latent-z scale and are
    # NOT standardized betas; route them to a non-convertible category.
    "ordered probit regression coefficient" = "latent_scale_coef",
    "probit beta (marginal change in odds)" = "latent_scale_coef",
    "OR" = "or_family", "odds_ratio" = "or_family",
    "log_odds_ratio" = "or_family",
    "hazard ratio" = "or_family", "hazard_ratio" = "or_family",
    "incidence_rate_ratio" = "or_family", "relative_risk_ratio" = "or_family",
    "Odds Ratio" = "or_family",
    "Wilcoxon r" = "nonparametric", "kendall's w" = "nonparametric",
    "cramer's V" = "nonparametric", "cramer_v" = "nonparametric",
    "phi" = "nonparametric",
    "cohen's h" = "nonparametric",
    # Cohen's q is a difference of Fisher-z'd correlations, not a correlation;
    # it cannot be copied as r.
    "cohen_q" = "non_convertible", "q" = "non_convertible",
    "mean difference" = "raw_difference", "Mean difference" = "raw_difference",
    "median difference" = "raw_difference",
    "percentage difference" = "raw_difference",
    "difference_in_coefficients" = "raw_difference",
    "ser_method" = "ser_method", "ser_method_t" = "ser_method",
    "t" = "test_stat_only", "test statistic" = "test_stat_only",
    "coefficient of loss aversion" = "other", "proportion" = "other"
  )

  es_cat_local <- ifelse(
    is.na(es_type_map[as.character(df$es_type_o)]),
    "other",
    es_type_map[as.character(df$es_type_o)]
  )

  n_vec     <- suppressWarnings(as.numeric(df$n_o))
  pval_vec  <- suppressWarnings(as.numeric(df$pval_value_o))
  es_vec    <- suppressWarnings(as.numeric(df$es_value_o))
  r_vec     <- suppressWarnings(as.numeric(df$r_o))

  # df for test-stat → r conversions; for OLS/correlation contexts df = n - 2
  df_t  <- pmax(n_vec - 2, 1)

  # ── d → r (Borenstein 2009, eq. 7.5):  r = d / sqrt(d^2 + 4) ─────────────────
  d_to_r <- function(d) d / sqrt(d^2 + 4)

  # ── log(OR) → d (Chinn 2000):  d = log(OR) * sqrt(3) / pi ───────────────────
  logor_to_d <- function(logor) logor * sqrt(3) / pi

  # Safe OR → log(OR): non-positive ORs are data errors → NA, NOT floored.
  safe_log_or <- function(or) ifelse(!is.na(or) & or > 0, log(or), NA_real_)

  # Pre-compute conversions per row using vectorised expressions ---------------
  # We assume es_value_o holds the effect size in its native scale.
  r_native     <- r_vec  # already r in the input
  r_from_d     <- d_to_r(es_vec)                                  # d → r
  r_from_dz    <- d_to_r(es_vec)                                  # dz treated as d
  r_from_var   <- sqrt(pmin(pmax(es_vec, 0), 1))                  # sqrt(η²/ω²/R²)
  # f² → r: clamp es_vec to [0, ∞) first since f² is non-negative by definition
  r_from_fsq   <- sqrt(pmin(pmax(es_vec, 0) / (1 + pmax(es_vec, 0)), 1))
  r_from_or    <- d_to_r(logor_to_d(safe_log_or(es_vec)))         # OR → d → r
  r_from_logor <- d_to_r(logor_to_d(es_vec))                      # log_OR direct
  # β as a proxy for r is only valid in single-predictor models; with multiple
  # predictors β is partial and can exceed 1 / flip sign. We keep it as a
  # last-resort, low-confidence proxy and clamp to a defensible r range.
  r_from_beta  <- pmax(pmin(es_vec, 0.95), -0.95)
  r_from_h     <- es_vec / 2                                      # Cohen's h: r ≈ h/2

  # ── Determine variance-explained sub-case (Cohen's f² vs η²/ω²/R²) ─────────
  es_raw_lower <- tolower(as.character(df$es_type_o))
  is_fsq <- !is.na(es_raw_lower) &
    grepl("f.?squared|f\\^2|cohen.?s? f", es_raw_lower)

  # ── Determine OR sub-case (log_OR vs OR) ───────────────────────────────────
  is_logor <- !is.na(es_raw_lower) & grepl("^log.?odds", es_raw_lower)

  # ── Determine nonparametric sub-case (Cohen's h vs others) ─────────────────
  is_cohen_h <- !is.na(es_raw_lower) & grepl("cohen.?s? h", es_raw_lower)

  # ── Test-stat-only: only "t" / "test statistic" known here; treat as t ─────
  # r from t with df = n - 2:  r = t / sqrt(t^2 + df). Require valid n > 2 so we
  # don't fabricate r from a df_t=1 floor when n is missing/tiny.
  n_ok_for_t <- !is.na(n_vec) & n_vec > 2
  r_from_t <- ifelse(n_ok_for_t, es_vec / sqrt(es_vec^2 + df_t), NA_real_)

  # ── Sign for variance_explained (lost in sqrt/abs) ─────────────────────────
  # η²/ω²/R²/f² are non-negative by construction, so es_value_o carries no sign
  # and r_o (when present) is handled by the r_native branch. We therefore
  # cannot recover sign here and leave these conversions positive by convention.
  # (No fake sign_keep that silently always returns +1.)

  # ── Assemble r_harmonized + method ─────────────────────────────────────────
  r_h     <- rep(NA_real_, nrow(df))
  method  <- rep("missing", nrow(df))

  has_r  <- !is.na(r_vec)
  has_es <- !is.na(es_vec)

  # 1) r already there
  idx <- has_r & is.na(r_h)
  r_h[idx]    <- r_vec[idx]
  method[idx] <- "r_native"

  # 2) r missing → use es-type-specific conversion
  # r_family: copy es_value_o (when es_type says r but r_o is missing)
  idx <- !has_r & has_es & es_cat_local == "r_family" & is.na(r_h)
  r_h[idx]    <- es_vec[idx]
  method[idx] <- "r_native"

  # partial_r: a partial correlation is on the r scale but is NOT zero-order r;
  # we record it as a low-confidence proxy rather than silently treating as r.
  idx <- !has_r & has_es & es_cat_local == "partial_r" & is.na(r_h)
  r_h[idx]    <- pmax(pmin(es_vec[idx], 0.95), -0.95)
  method[idx] <- "from_partial_r"

  # partial_r_squared: on a squared scale → r = sign-less sqrt, clamped to [0,1]
  idx <- !has_r & has_es & es_cat_local == "partial_r_squared" & is.na(r_h)
  r_h[idx]    <- sqrt(pmin(pmax(es_vec[idx], 0), 1))
  method[idx] <- "from_partial_r"

  # smd_between (Cohen's d, Hedges' g, dv, SD units): d → r
  idx <- !has_r & has_es & es_cat_local == "smd_between" & is.na(r_h)
  r_h[idx]    <- r_from_d[idx]
  method[idx] <- "from_d"

  # smd_within (Cohen's dz): treat as d; flag low confidence
  idx <- !has_r & has_es & es_cat_local == "smd_within" & is.na(r_h)
  r_h[idx]    <- r_from_dz[idx]
  method[idx] <- "from_dz"

  # variance_explained: η², ω², R² → r = sqrt(eta²); f² uses f²/(1+f²).
  # Sign cannot be recovered (see note above): kept positive.
  idx <- !has_r & has_es & es_cat_local == "variance_explained" & !is_fsq & is.na(r_h)
  r_h[idx]    <- r_from_var[idx]
  method[idx] <- "from_variance_explained"

  idx <- !has_r & has_es & es_cat_local == "variance_explained" & is_fsq & is.na(r_h)
  r_h[idx]    <- r_from_fsq[idx]
  method[idx] <- "from_variance_explained"

  # or_family: Chinn 2000 (HR/IRR/RR approximated by OR). Non-positive OR → NA.
  idx <- !has_r & has_es & es_cat_local == "or_family" & is_logor & is.na(r_h)
  r_h[idx]    <- r_from_logor[idx]
  method[idx] <- "from_or"

  idx <- !has_r & has_es & es_cat_local == "or_family" & !is_logor & is.na(r_h)
  r_h[idx]    <- r_from_or[idx]
  method[idx] <- "from_or"

  # nonparametric (Wilcoxon r ≈ r; phi = r; Cramér's V ≈ r upper bound; h ≈ 2 arcsin)
  idx <- !has_r & has_es & es_cat_local == "nonparametric" & is_cohen_h & is.na(r_h)
  r_h[idx]    <- r_from_h[idx]
  method[idx] <- "from_nonparam"

  idx <- !has_r & has_es & es_cat_local == "nonparametric" & !is_cohen_h & is.na(r_h)
  r_h[idx]    <- es_vec[idx]   # phi / Cramér's V / Wilcoxon r / Kendall's W ≈ r
  method[idx] <- "from_nonparam"

  # regression_std (β as proxy for r) — single-predictor assumption, low conf
  idx <- !has_r & has_es & es_cat_local == "regression_std" & is.na(r_h)
  r_h[idx]    <- r_from_beta[idx]
  method[idx] <- "from_beta"

  # test_stat_only: es_value_o is the t-statistic (NA when n invalid)
  idx <- !has_r & has_es & es_cat_local == "test_stat_only" & !is.na(r_from_t) & is.na(r_h)
  r_h[idx]    <- r_from_t[idx]
  method[idx] <- "from_test_stat"

  # ser_method: treat as t-statistic equivalent (same conversion path)
  idx <- !has_r & has_es & es_cat_local == "ser_method" & !is.na(r_from_t) & is.na(r_h)
  r_h[idx]    <- r_from_t[idx]
  method[idx] <- "from_test_stat"

  # Non-convertible by construction: raw differences, unstandardized & latent
  # coefficients, and Cohen's q.
  idx <- !has_r & es_cat_local == "raw_difference" & is.na(r_h)
  method[idx] <- "unstandardized_skipped"

  idx <- !has_r & es_cat_local == "regression_unstd" & is.na(r_h)
  method[idx] <- "unstandardized_skipped"

  idx <- !has_r & es_cat_local == "latent_scale_coef" & is.na(r_h)
  method[idx] <- "latent_scale_skipped"

  idx <- !has_r & es_cat_local == "non_convertible" & is.na(r_h)
  method[idx] <- "non_convertible_skipped"

  # ── Fallback: synthesize r from p + n (post-hoc MDE; NOT a real estimate) ──
  # Sign from claim-text keywords; cap |r| ≤ 0.30 to avoid feeding a strong
  # fake signal to the model.
  pn_eligible <- is.na(r_h) & !has_es & !is.na(pval_vec) & !is.na(n_vec) & n_vec > 2

  pos_kw <- regex("\\b(increase|higher|greater|positive|more|elevat|enhanc|improv)",
                  ignore_case = TRUE)
  neg_kw <- regex("\\b(decrease|lower|smaller|negative|less|reduc|diminish|decline)",
                  ignore_case = TRUE)
  claim_txt <- as.character(df$claim_text_o)
  sign_pn <- ifelse(!is.na(claim_txt) & str_detect(claim_txt, neg_kw) &
                      !str_detect(claim_txt, pos_kw), -1, 1)

  # invert p → t (two-tailed): t = qt(1 - p/2, df)
  p_safe <- pmax(pmin(pval_vec, 1 - 1e-10), 1e-15)
  t_from_p <- suppressWarnings(qt(1 - p_safe / 2, df = df_t))
  r_pn <- t_from_p / sqrt(t_from_p^2 + df_t)
  r_pn <- pmax(pmin(r_pn, 0.30), -0.30) * sign_pn  # cap then sign

  idx <- pn_eligible
  r_h[idx]    <- r_pn[idx]
  method[idx] <- "synthesized_from_pn"

  # ── Final NA-rounding + cap at ±0.95 ────────────────────────────────────────
  r_h[is.nan(r_h) | is.infinite(r_h)] <- NA_real_
  r_h_capped <- as.integer(!is.na(r_h) & abs(r_h) > 0.95)
  r_h <- pmax(pmin(r_h, 0.95), -0.95)

  # ── Low-confidence flag ────────────────────────────────────────────────────
  low_conf_methods <- c("from_dz", "from_or", "from_beta", "from_partial_r",
                        "from_nonparam", "synthesized_from_pn")
  low_conf <- as.integer(method %in% low_conf_methods)

  df$r_harmonized                <- r_h
  df$r_harmonized_method         <- factor(method)
  df$r_harmonized_low_confidence <- low_conf
  df$r_harmonized_capped         <- r_h_capped

  # ── Re-derive proxies from r_harmonized (parallel to r_o-based ones) ───────
  # NA-propagating where r_harmonized or n_o is missing. Use a single clamped
  # p-value consistently across all derived features.
  pval_safe  <- pmax(pmin(pval_vec, 1 - 1e-10), 1e-15)
  pval_z_vec <- -qnorm(pval_safe / 2)

  t_proxy_h <- pmax(pmin(
    r_h * sqrt(pmax(n_vec - 2, 1)) / sqrt(pmax(1 - r_h^2, 1e-10)),
    500), -500)
  snr_h <- t_proxy_h^2 / pmax(n_vec, 1)
  # NOTE: this is an unbounded evidence score, NOT statistical power in [0,1].
  evidence_proxy_h <- abs(r_h) * sqrt(n_vec) / (pval_safe + 0.01)

  df$t_proxy_from_r_h <- t_proxy_h
  df$snr_proxy_h      <- snr_h
  df$power_proxy_h    <- evidence_proxy_h  # kept column name for compatibility

  # ── Combined evidence-strength score (Round 2 #2 winner's proxy) ───────────
  # evidence_strength_score = pval_z * log10(max(n_o, 1)) * |r_harmonized|
  # Cap at ±50 to prevent any one outlier driving downstream features.
  ess <- pval_z_vec * log10(pmax(n_vec, 1)) * abs(r_h)
  ess <- pmax(pmin(ess, 50), -50)
  df$evidence_strength_score <- ess

  # ── Bayes-factor evidence feature (BIC approximation; Wagenmakers 2007) ─────
  # log BF10 ≈ (t^2 - ln n) / 2 for a single-parameter test. Penalised by n via
  # the -ln n term, so unlike pval_z it does not keep rising with n at fixed t.
  # Sign-free (uses t^2) -> does NOT inherit r_harmonized's sign-flip. Capped to
  # +/-100. This is an approximation to the JZS
  # BF (Rouder 2009), not the exact correlationBF.
  log_bf10 <- vapply(seq_along(t_proxy_h), function(i) {
    if (is.na(t_proxy_h[i]) || is.na(n_vec[i]) || n_vec[i] < 3) return(NA_real_)
    tryCatch(BayesFactor::ttest.tstat(t = t_proxy_h[i], n1 = n_vec[i],
                                      rscale = "medium")$bf,
             error = function(e) NA_real_)
  }, numeric(1))

  log_bf10 <- pmax(pmin(log_bf10, 100), -100)
  df$log_bf10_h  <- log_bf10
  df$bf10_gt3_h  <- as.integer(log_bf10 > log(3))
  df$bf10_gt10_h <- as.integer(log_bf10 > log(10))

  df
}

# Call harmonization on the joined frames BEFORE apply_features so the new
# numeric columns (r_harmonized, r_harmonized_low_confidence, t_proxy_from_r_h,
# snr_proxy_h, power_proxy_h, evidence_strength_score) automatically flow
# through the missingness-indicator / log1p / sqrtabs loops.
train_joined <- harmonize_effect_sizes(train_joined)
test_joined  <- harmonize_effect_sizes(test_joined)

cat("\n=== r_harmonized coverage ===\n")
cat("Train: r_o NA rate         =", round(mean(is.na(train_joined$r_o)), 3),
    "| r_harmonized NA rate =", round(mean(is.na(train_joined$r_harmonized)), 3), "\n")
cat("Test : r_o NA rate         =", round(mean(is.na(test_joined$r_o)),  3),
    "| r_harmonized NA rate =", round(mean(is.na(test_joined$r_harmonized)),  3), "\n")

# ------------------------------------------------------------------------------
# apply_features(): all transformations from engineer_features() except imputation.
# Raw columns (n_o, pval_value_o, r_o) are used directly; NAs propagate into
# derived features and are resolved downstream in 05_impute.R.
# No *_imp columns are created here.
# ------------------------------------------------------------------------------

apply_features <- function(data) {

  data <- data |> mutate(n_o = as.numeric(n_o))

  # ── Missingness indicators ──────────────────────────────────────────────────
  data <- data |>
    mutate(
      n_o_miss_ind          = as.integer(is.na(n_o)),
      pval_value_o_miss_ind = as.integer(is.na(pval_value_o)),
      r_o_miss_ind          = as.integer(is.na(r_o))
    )

  # ── Numeric transformations (raw columns; NAs where inputs missing) ─────────
  data <- data |>
    mutate(
      n_o_log   = log10(pmax(n_o, 1)),
      n_o_sqrt  = sqrt(n_o),

      pval_z = {
        p_c <- pmax(pmin(pval_value_o, 1 - 1e-10), 1e-15)
        -qnorm(p_c / 2)
      },
      evidence_mlogp = -log10(pmax(pval_value_o, 1e-300)),

      r_o_abs      = abs(r_o),
      r_o_positive = as.integer(r_o > 0),
      r_o_log      = log1p(abs(r_o)),

      n_o_bins = cut(
        n_o,
        breaks = c(0, 20, 50, 100, 200, 500, Inf),
        labels = c("<20", "20-50", "51-100", "101-200", "201-500", ">500"),
        include.lowest = TRUE, right = TRUE
      ),
      p_val_bins = cut(
        pval_value_o,
        breaks = c(0, 0.001, 0.01, 0.05, 0.10, 1),
        labels = c("p<.001", "p<.01", "p<.05", "p<.10", "p>=.10"),
        include.lowest = TRUE, right = FALSE
      ),

      t_proxy_from_r = pmax(
        pmin(r_o * sqrt(n_o - 2) / sqrt(1 - r_o^2), 500),
        -500
      ),
      snr_proxy  = t_proxy_from_r^2 / pmax(n_o, 1),

      power_proxy1 = abs(r_o) * sqrt(n_o) / (pval_value_o + 0.01),
      power_proxy2 = snr_proxy * evidence_mlogp,

      p_expected        = 2 * pt(-abs(t_proxy_from_r), df = pmax(n_o - 2, 1)),
      p_consistency_gap = abs(
        log10(pmax(pval_value_o, 1e-15)) -
          log10(pmax(p_expected,   1e-15))
      ),

      just_significant = as.integer(
        !is.na(pval_value_o) & pval_value_o >= 0.04 & pval_value_o < 0.05
      )
    )

  # ── Claim-text features ─────────────────────────────────────────────────────
  data <- data |>
    mutate(
      claim_length     = nchar(claim_text_o),
      claim_n_digits   = str_count(claim_text_o, "\\d"),
      prop_digits      = claim_n_digits / pmax(claim_length, 1),
      claim_length_log = log1p(claim_length),

      has_pval_text   = as.integer(str_detect(claim_text_o,
        regex("p\\s*[<=]|p\\s*=|p-val",          ignore_case = TRUE))),
      has_ci_text     = as.integer(str_detect(claim_text_o,
        regex("95%|confidence interval|\\bCI\\b", ignore_case = TRUE))),
      has_effect_text = as.integer(str_detect(claim_text_o,
        regex("Cohen|effect size|\\br\\s*=|OR\\s*=|odds ratio", ignore_case = TRUE))),
      has_interaction = as.integer(str_detect(claim_text_o,
                                              regex("interact|moderat", ignore_case = TRUE)))
    )

  # Legacy: missing claim_text_o collapses every claim-text feature to 0.
  # NA_counts_Zero = TRUE keeps them NA (cannot be discounted by GAM/elnet/logreg).
  if (!NA_counts_Zero) {
    data <- data |> mutate(across(
      c(claim_length, claim_n_digits, prop_digits, claim_length_log,
        has_pval_text, has_ci_text, has_effect_text, has_interaction),
      ~ replace_na(.x, 0)
    ))
  }

  # ── Effect size type categorisation ────────────────────────────────────────
  es_type_map <- c(
    "r"                                     = "r_family",
    "pearson_r"                             = "r_family",
    "partial r"                             = "r_family",
    "partial correlation"                   = "r_family",
    "partial r-square"                      = "r_family",
    "d"                                     = "smd_between",
    "Cohen's d"                             = "smd_between",
    "cohen_d"                               = "smd_between",
    "hedges' g"                             = "smd_between",
    "dv"                                    = "smd_between",
    "Standard deviation units"              = "smd_between",
    "Cohen's dz"                            = "smd_within",
    "cohen_dz"                              = "smd_within",
    "dz"                                    = "smd_within",
    "etasq"                                 = "variance_explained",
    "etasq (partial)"                       = "variance_explained",
    "partial_eta_squared"                   = "variance_explained",
    "r-squared"                             = "variance_explained",
    "omega squared"                         = "variance_explained",
    "cohen_f_squared"                       = "variance_explained",
    "Cohen's f^2"                           = "variance_explained",
    "cohen's f"                             = "variance_explained",
    "b (unstd)"                             = "regression_unstd",
    "be"                                    = "regression_unstd",
    "b"                                     = "regression_unstd",
    "eta_squared"                           = "variance_explained",
    "regression_coefficient"                = "regression_unstd",
    "beta"                                  = "regression_std",
    "beta (std)"                            = "regression_std",
    "path_coefficient"                      = "regression_std",
    "ordered probit regression coefficient" = "regression_std",
    "probit beta (marginal change in odds)" = "regression_std",
    "OR"                                    = "or_family",
    "odds_ratio"                            = "or_family",
    "log_odds_ratio"                        = "or_family",
    "hazard ratio"                          = "or_family",
    "hazard_ratio"                          = "or_family",
    "incidence_rate_ratio"                  = "or_family",
    "relative_risk_ratio"                   = "or_family",
    "Wilcoxon r"                            = "nonparametric",
    "kendall's w"                           = "nonparametric",
    "cramer's V"                            = "nonparametric",
    "cramer_v"                              = "nonparametric",
    "phi"                                   = "nonparametric",
    "cohen's h"                             = "nonparametric",
    "cohen_q"                               = "nonparametric",
    "q"                                     = "nonparametric",
    "mean difference"                       = "raw_difference",
    "Mean difference"                       = "raw_difference",
    "median difference"                     = "raw_difference",
    "percentage difference"                 = "raw_difference",
    "difference_in_coefficients"            = "raw_difference",
    "ser_method"                            = "ser_method",
    "ser_method_t"                          = "ser_method",
    "t"                                     = "test_stat_only",
    "test statistic"                        = "test_stat_only",
    "coefficient of loss aversion"          = "other",
    "proportion"                            = "other",
    "Odds Ratio"                          = "or_family",
    "Partial correlation coefficient"     = "r_family",
    "Eta-squared"                         = "variance_explained"
  )

  data <- data |>
    mutate(
      es_type_cat = factor(
        if_else(
          is.na(es_type_map[as.character(es_type_o)]),
          "other",
          es_type_map[as.character(es_type_o)]
        )
      ),
      es_is_standardized = as.integer(es_type_cat %in% c(
        "r_family", "smd_between", "smd_within",
        "variance_explained", "regression_std", "nonparametric"
      )),
      es_directional = as.integer(
        es_type_cat %in% c(
          "r_family", "smd_between", "smd_within",
          "regression_std", "regression_unstd",
          "or_family", "raw_difference", "test_stat_only"
        ) |
          as.character(es_type_o) %in% c("phi", "Wilcoxon r", "cohen's h")
      )
    )

  # ── p-value reporting categorisation ───────────────────────────────────────
  if (!"pval_tails_o" %in% names(data))
    data <- data |> mutate(pval_tails_o = NA_character_)

  data <- data |>
    mutate(
      pval_type_cat = factor(case_when(
        pval_type_o %in% c("exact", "=", "e (=)")        ~ "exact",
        pval_type_o %in% c("<", "l (<)", "le (<=)",
                           "less-than")                   ~ "less_than",
        pval_type_o %in% c("g (>)", "ge (>)", "ge (>=)") ~ "greater_than",
        TRUE                                              ~ "other"
      )),
      pval_tails_cat = factor(case_when(
        pval_tails_o %in% c("1", "1 (one)", "one-tailed")               ~ "one_tailed",
        pval_tails_o %in% c("2", "2 (two)", "two-tailed", "two_tailed") ~ "two_tailed",
        TRUE                                                             ~ "unknown"
      )),
      pval_exact      = as.integer(pval_type_cat  == "exact"),
      pval_one_tailed = as.integer(pval_tails_cat == "one_tailed")
    )

  # ── Study / design features ─────────────────────────────────────────────────
  for (col in c("author_overlap", "prereg_r")) {
    if (!col %in% names(data))
      data <- data |> mutate(!!col := NA)
  }

  data <- data |>
    mutate(
      author_overlap_bin = case_when(
        author_overlap == TRUE  ~ 1L,
        TRUE                    ~ 0L
      ),
      author_overlap_missing = as.integer(is.na(author_overlap)),
      prereg_r_bin           = as.integer(
        !is.na(prereg_r) & nchar(as.character(prereg_r)) > 0
      )
    )

  # ── Discipline / health tag ─────────────────────────────────────────────────
  for (col in c("journal_name_oa", "topics_via_oa", "keywords_via_oa")) {
    if (!col %in% names(data))
      data <- data |> mutate(!!col := NA_character_)
  }

  health_pattern <- regex(
    "health|medicine|clinical|medical|epidemiol|pharmacol|nursing|oncol|cardiol",
    ignore_case = TRUE
  )

  data <- data |>
    mutate(
      is_health = as.integer(
        str_detect(as.character(discipline), regex("health", ignore_case = TRUE))
      )
    ) |>
    group_by(doi_o) |>
    mutate(
      is_health_tagged = as.integer(any(
        str_detect(as.character(tags), health_pattern), na.rm = TRUE
      )),
      health_related_big = as.integer(any(
        str_detect(as.character(tags),            health_pattern),
        str_detect(as.character(journal_name_oa), health_pattern),
        str_detect(as.character(discipline),      health_pattern),
        str_detect(as.character(topics_via_oa),   health_pattern),
        str_detect(as.character(keywords_via_oa), health_pattern),
        na.rm = TRUE
      ))
    ) |>
    ungroup() |>
    mutate(across(
      c(is_health, is_health_tagged, health_related_big),  # confirmation flags: 0 stays correct
      ~ replace_na(.x, 0L)
    ))

  # Property flags (standardisation / directionality / p-value reporting): legacy
  # collapses an unknown source field to 0. With NA_counts_Zero = TRUE keep NA
  # when the source was missing, and expose the p-value-tails missingness (77 %)
  # that previously had no indicator.
  if (!NA_counts_Zero) {
    data <- data |> mutate(across(
      c(es_directional, pval_exact, pval_one_tailed),
      ~ replace_na(.x, 0L)
    ))
  } else {
    data <- data |> mutate(
      es_is_standardized = if_else(is.na(es_type_o),    NA_integer_, es_is_standardized),
      es_directional     = if_else(is.na(es_type_o),    NA_integer_, es_directional),
      pval_exact         = if_else(is.na(pval_type_o),  NA_integer_, pval_exact),
      pval_one_tailed    = if_else(is.na(pval_tails_o), NA_integer_, pval_one_tailed),
      pval_tails_missing = as.integer(is.na(pval_tails_o))
    )
  }

  # ── Temporal features ───────────────────────────────────────────────────────
  if (!"dataset" %in% names(data))
    data <- data |> mutate(dataset = NA_character_)
  if (!"ref_r" %in% names(data))
    data <- data |> mutate(ref_r = NA_character_)

  data <- data |>
    mutate(
      year_o = as.integer(year_o),
      year_r = case_when(
        is.na(ref_r) ~ 2026L,
        str_detect(ref_r, regex("^SCORE", ignore_case = TRUE)) ~ 2024L,
        TRUE ~ as.integer(str_extract(ref_r, "\\b(19|20)\\d{2}\\b"))
      ),
      duration_o_r         = year_r - year_o,
      before_covid         = as.integer(year_o < 2020),
      is_score_replication = as.integer(
        !is.na(dataset) & dataset %in% c("round1", "round2")
      )
    )

  # ── Journal + author geography ──────────────────────────────────────────────
  for (col in c("journal_sjr_best_quartile", "first_author_country",
                "last_author_country")) {
    if (!col %in% names(data))
      data <- data |> mutate(!!col := NA_character_)
  }

  data <- data |>
    mutate(
      sjr_quartile_num = case_when(
        str_detect(as.character(journal_sjr_best_quartile), "Q1") ~ 1L,
        str_detect(as.character(journal_sjr_best_quartile), "Q2") ~ 2L,
        str_detect(as.character(journal_sjr_best_quartile), "Q3") ~ 3L,
        str_detect(as.character(journal_sjr_best_quartile), "Q4") ~ 4L,
        TRUE ~ NA_integer_
      ),
      sjr_quartile_fct     = factor(sjr_quartile_num, levels = 1:4, ordered = TRUE),
      sjr_quartile_missing = as.integer(is.na(sjr_quartile_num)),

      same_author_country = as.integer(
        !is.na(first_author_country) &
          !is.na(last_author_country)  &
          as.character(first_author_country) == as.character(last_author_country)
      ),
      same_author_country_missing = as.integer(
        is.na(first_author_country) | is.na(last_author_country)
      )
    )

  # ── Authorship diversity (counts from full team; complements same_author_country) ──
  for (col in c("n_author_countries", "n_author_institutions")) {
    if (!col %in% names(data)) data <- data |> mutate(!!col := NA_integer_)
  }
  data <- data |>
    mutate(
      is_multinational_team         = as.integer(n_author_countries > 1),
      is_multisite_team             = as.integer(n_author_institutions > 1),
      n_author_countries_missing    = as.integer(is.na(n_author_countries)),
      n_author_institutions_missing = as.integer(is.na(n_author_institutions))
    )

  # ── Author productivity rate (works per active year) ───────────────────────
  for (col in c("first_author_works_count", "first_author_experience_since",
                "last_author_works_count",  "last_author_experience_since")) {
    if (!col %in% names(data)) data <- data |> mutate(!!col := NA_real_)
  }
  .pipeline_year <- 2026L  # reference "current" year (matches year_r default)
  data <- data |>
    mutate(
      # NOTE: experience_since = first year in OpenAlex counts_by_year, which is
      # window-capped (~10y), so this overestimates output rate for senior authors.
      first_author_active_years = pmax(.pipeline_year - first_author_experience_since + 1, 1),
      last_author_active_years  = pmax(.pipeline_year - last_author_experience_since  + 1, 1),
      first_author_productivity = first_author_works_count / first_author_active_years,
      last_author_productivity  = last_author_works_count  / last_author_active_years
    )
  # ── Readability guard (columns added by add_readability()) ─────────────────
  for (v in c("readability_full_fk", "readability_full_smog",
              "readability_abs_fk",  "readability_abs_smog")) {
    if (!v %in% names(data))
      data <- data |> mutate(!!v := NA_real_)
  }

  # ── NLP / text features ─────────────────────────────────────────────────────
  for (col in c("title_xml", "abstract_xml", "article_text_xml", "language_xml")) {
    if (!col %in% names(data))
      data <- data |> mutate(!!col := NA_character_)
  }

  .to_txt <- function(x) {
    x <- as.character(x)
    x[trimws(x) %in% c("", "NA", "NULL")] <- NA_character_
    x
  }

  .sent_batch <- function(txt) {
    out   <- rep(NA_real_, length(txt))
    valid <- !is.na(txt) & nchar(trimws(txt)) >= 10L
    if (!any(valid)) return(out)
    tryCatch({
      res        <- sentimentr::sentiment_by(txt[valid])
      out[valid] <- res$ave_sentiment
    }, error = function(e) {
      for (i in which(valid)) {
        tryCatch(
          { out[i] <<- sentimentr::sentiment_by(txt[i])$ave_sentiment },
          error = function(e2) NULL
        )
      }
    })
    out
  }

  .count_stat_types <- function(txt) {
    patterns <- c(
      "\\bt[- ]test\\b",
      "\\bANO[CV]A\\b|\\bMANO[CV]A\\b",
      "chi[- ]?(?:square|sq)",
      "logistic regression",
      "(?:multiple|hierarchical|linear) regression",
      "\\bfactor analysis\\b|principal component",
      "structural equation|\\bpath analysis\\b",
      "\\bWilcoxon\\b|Mann[- ]Whitney|Kruskal[- ]Wallis",
      "(?:pearson|spearman) (?:r\\b|correlation)",
      "mixed[- ]model|multilevel|hierarchical linear model"
    )
    mat <- vapply(
      patterns,
      function(p) grepl(p, txt, ignore.case = TRUE, perl = TRUE),
      logical(length(txt))
    )
    out             <- as.integer(rowSums(mat))
    out[is.na(txt)] <- NA_integer_
    out
  }

  .hedging_rx <- regex(paste0(
    "(?:may|might|could) (?:suggest|indicate|reflect|imply|explain)|",
    "appear(?:s)? to (?:be|reflect|suggest)|",
    "seem(?:s)? to (?:be|reflect|suggest)|",
    "tentative(?:ly)?|provisionally?|",
    "preliminary (?:evidence|finding|result|data)|",
    "should be (?:interpreted|viewed|treated) with caution|",
    "it (?:is|seems|appears) (?:possible|plausible|likely) that|",
    "we speculate|\\binconclusive\\b|",
    "further (?:research|study|investigation|replication) ",
    "(?:is |are )?(?:needed|required|warranted)|",
    "warrant(?:s)? (?:further|future|replication|caution)"
  ), ignore_case = TRUE)

  .pval_rx <- regex(
    "\\bp\\s*(?:[<>]=?|[=≤≥≈~])\\s*(?:\\.?[0-9]|n\\.?\\s?s\\.?)",
    ignore_case = TRUE
  )

  .hypothesis_rx <- regex(paste0(
    "\\bhypothes[ie]s\\b|",
    "\\bH[0-9]+[a-z]?\\s*:|",
    "we (?:hypothesize|hypothesized|predicted|expected) that|",
    "(?:it was |was )(?:hypothesized|predicted)"
  ), ignore_case = TRUE)

  .significant_rx  <- regex("\\bsignificant(?:ly)?\\b",         ignore_case = TRUE)
  .limitations_rx  <- regex("\\blimitation(?:s)?\\b|\\bcaveat(?:s)?\\b", ignore_case = TRUE)

  .papers_unique <- data |> distinct(doi_o, .keep_all = TRUE) |>
    mutate(
      tmp_title    = .to_txt(title_xml),
      tmp_abstract = .to_txt(abstract_xml),
      tmp_fulltext = .to_txt(article_text_xml)
    )

  .papers_unique <- .papers_unique |>
    mutate(
      is_english_text_nlp = as.integer(
        is.na(language_xml) |
          str_detect(as.character(language_xml), regex("^en", ignore_case = TRUE))
      ),
      title_txt_missing_nlp    = as.integer(is.na(tmp_title)),
      abstract_txt_missing_nlp = as.integer(is.na(tmp_abstract)),
      fulltext_missing_nlp     = as.integer(is.na(tmp_fulltext)),

      title_length_nlp        = nchar(tmp_title),
      title_length_log_nlp    = log1p(nchar(tmp_title)),
      title_n_words_nlp       = str_count(tmp_title, "\\S+"),
      title_punct_density_nlp = str_count(tmp_title, "[[:punct:]]") /
        pmax(nchar(tmp_title), 1L),
      title_is_question_nlp   = as.integer(
        str_detect(tmp_title, "\\?") |
          str_detect(tmp_title, regex(
            "^(?:does|do|is|are|can|will|should|how|why|what|which)\\s",
            ignore_case = TRUE
          ))
      ),
      title_has_colon_nlp  = as.integer(str_detect(tmp_title, ":")),
      title_n_digits_nlp   = str_count(tmp_title, "\\d"),
      title_sentiment_nlp  = .sent_batch(tmp_title),

      abstract_length_nlp          = nchar(tmp_abstract),
      abstract_length_log_nlp      = log1p(nchar(tmp_abstract)),
      abstract_n_hedging_nlp       = str_count(tmp_abstract, .hedging_rx),
      abstract_hedging_density_nlp = str_count(tmp_abstract, .hedging_rx) /
        pmax(nchar(tmp_abstract), 1L) * 1000,
      abstract_n_pval_nlp          = str_count(tmp_abstract, .pval_rx),
      abstract_has_hypothesis_nlp  = as.integer(
        str_detect(tmp_abstract, .hypothesis_rx)
      ),
      abstract_n_significant_nlp   = str_count(tmp_abstract, .significant_rx),
      abstract_sentiment_nlp       = .sent_batch(tmp_abstract),

      fulltext_length_nlp            = nchar(tmp_fulltext),
      fulltext_n_pval_nlp            = str_count(tmp_fulltext, .pval_rx),
      fulltext_n_stat_test_types_nlp = .count_stat_types(tmp_fulltext),
      fulltext_n_hedging_nlp         = str_count(tmp_fulltext, .hedging_rx),
      fulltext_hedging_density_nlp   = str_count(tmp_fulltext, .hedging_rx) /
        pmax(nchar(tmp_fulltext), 1L) * 1000,
      fulltext_n_hypothesis_nlp      = str_count(tmp_fulltext, .hypothesis_rx),
      fulltext_n_significant_nlp     = str_count(tmp_fulltext, .significant_rx),
      fulltext_n_limitations_nlp     = str_count(tmp_fulltext, .limitations_rx),
      fulltext_n_figures_nlp         = str_count(
        tmp_fulltext, regex("\\bFig(?:ure)?[.\\s]+[0-9]", ignore_case = TRUE)
      ),
      fulltext_n_tables_nlp          = str_count(
        tmp_fulltext, regex("\\bTable[\\s]+[0-9]", ignore_case = TRUE)
      ),
      sig_per_pval_nlp = fulltext_n_significant_nlp / pmax(fulltext_n_pval_nlp, 1L)
    ) |>
    select(-any_of(c("tmp_title", "tmp_abstract", "tmp_fulltext")))

  # ── GRIM / scrutiny features ────────────────────────────────────────────────
  .extract_m_sd_pairs <- function(txt) {
    pats <- c(
      "(?i)\\bM\\s*\\(\\s*SD\\s*\\)\\s*=\\s*(-?\\d+\\.\\d+)\\s*\\(\\s*(\\d+\\.\\d+)\\s*\\)",
      "(?i)\\bM\\s*=\\s*(-?\\d+\\.\\d+)\\s*\\(\\s*SD\\s*=\\s*(\\d+\\.\\d+)\\s*\\)",
      "(?i)\\bM\\s*=\\s*(-?\\d+\\.\\d+)\\s*[,;]\\s*SD\\s*=\\s*(\\d+\\.\\d+)",
      "(?i)\\bmean\\s*=\\s*(-?\\d+\\.\\d+)\\s*\\(\\s*SD\\s*=\\s*(\\d+\\.\\d+)\\s*\\)",
      "(?i)\\bmean\\s*=\\s*(-?\\d+\\.\\d+)\\s*[,;]\\s*SD\\s*=\\s*(\\d+\\.\\d+)",
      "(?i)\\bmean\\s*\\(\\s*SD\\s*\\)\\s*[=:]\\s*(-?\\d+\\.\\d+)\\s*\\(\\s*(\\d+\\.\\d+)\\s*\\)"
    )
    lapply(txt, function(x) {
      empty <- data.frame(mean_str = character(0), sd_str = character(0),
                          stringsAsFactors = FALSE)
      if (is.na(x) || nchar(trimws(x)) == 0L) return(empty)
      results <- lapply(pats, function(p) {
        m <- str_match_all(x, p)[[1]]
        if (nrow(m) == 0L) return(empty)
        data.frame(mean_str = m[, 2L], sd_str = m[, 3L], stringsAsFactors = FALSE)
      })
      out <- do.call(rbind, results)
      if (nrow(out) == 0L) return(empty)
      distinct(out, mean_str, sd_str)
    })
  }

  .grim_check <- function(pairs_df, n_val) {
    if (nrow(pairs_df) == 0L || is.na(n_val) || n_val < 2) {
      return(list(n_tested = 0L, n_fail = 0L, mean_grim_prob = NA_real_))
    }
    n_val      <- as.integer(round(n_val))
    grim_input <- tibble(x = pairs_df$mean_str, n = n_val)
    result <- tryCatch(
      scrutiny::grim_map(grim_input, items = 1L),
      error = function(e) NULL
    )
    if (is.null(result) || nrow(result) == 0L)
      return(list(n_tested = 0L, n_fail = 0L, mean_grim_prob = NA_real_))
    list(
      n_tested       = nrow(result),
      n_fail         = sum(!result$consistency),
      mean_grim_prob = mean(result$probability, na.rm = TRUE)
    )
  }

  .papers_unique <- .papers_unique |>
    mutate(tmp_fulltext_sc = .to_txt(article_text_xml))

  .grim_features <- do.call(rbind, mapply(
    function(txt, n_val) {
      pairs  <- .extract_m_sd_pairs(txt)[[1]]
      result <- .grim_check(pairs, n_val)
      data.frame(
        n_m_sd_pairs_sc   = nrow(pairs),
        grim_n_tested_sc  = result$n_tested,
        grim_n_fail_sc    = result$n_fail,
        grim_any_fail_sc  = as.integer(result$n_fail > 0L),
        grim_fail_rate_sc = if (result$n_tested > 0L)
          result$n_fail / result$n_tested else NA_real_,
        grim_mean_prob_sc = result$mean_grim_prob
      )
    },
    txt   = .papers_unique$tmp_fulltext_sc,
    n_val = .papers_unique$n_o,   # raw n_o (NA-safe via .grim_check guard)
    SIMPLIFY = FALSE, USE.NAMES = FALSE
  )) |> as_tibble()

  .papers_unique <- bind_cols(
    .papers_unique |> select(-tmp_fulltext_sc),
    .grim_features
  )
  rm(.grim_features)

  # join paper-level NLP + scrutiny features back onto claim-level data
  data <- data |>
    left_join(.papers_unique |> select(doi_o, ends_with("_sc")),  by = "doi_o") |>
    left_join(.papers_unique |> select(doi_o, ends_with("_nlp")), by = "doi_o")
  rm(.papers_unique)

  # GRIM is dropped from the model. A TEI audit found ~all flagged GRIM failures
  # are artifacts of applying study-level n_o to subgroup means (0/94 audited
  # failures were genuine), so grim_*fail/prob/rate carry no signal. We compute
  # `grim_applicable` (= at least one mean was GRIM-checkable) only as an
  # available column for an optional LORO ablation — it is a noisy, indirect
  # "reports scale means" proxy that is redundant with study_design / es-type /
  # is_health, so it is NOT in the default predictor sets.
  data$grim_applicable <- as.integer(!is.na(data$grim_n_tested_sc) &
                                       data$grim_n_tested_sc > 0L)

  # ── TEI-derived bias indicators (Tasks A + B) ────────────────────────────────
  # Computed on one-row-per-paper, then left_join back to data.
  # NA propagates safely when article_text_xml is missing.
  {
    .tei_papers <- data |> distinct(doi_o, .keep_all = TRUE) |>
      mutate(.txt = {
        x <- as.character(article_text_xml)
        x[trimws(x) %in% c("", "NA", "NULL")] <- NA_character_
        x
      })

    # ── Helper: extract numeric p-values from text ────────────────────────────
    .extract_pvals <- function(txt) {
      lapply(txt, function(x) {
        if (is.na(x)) return(numeric(0))
        # Case-insensitive (P/p), all comparators (< = > ≤ ≥ <= >=), and any
        # decimal (not just .0x). The old "p[<=].0\\d+" form missed `P<.05`,
        # `p>.1`, `p=.12`, leaving ~1/3 of TEI papers with zero matched p-values
        # (p_hack_index_tei NA). Lifts test-set coverage from 69% to ~98%.
        m <- stringr::str_match_all(x, "(?i)\\bp\\s*(?:<=|>=|[<>=≤≥])\\s*(0?\\.\\d+)")[[1]]
        if (nrow(m) == 0L) return(numeric(0))
        suppressWarnings(as.numeric(m[, 2L]))
      })
    }

    .pvals_list <- .extract_pvals(.tei_papers$.txt)

    .tei_papers <- .tei_papers |>
      mutate(
        # 1. p_hack_index_tei
        p_hack_index_tei = {
          vapply(seq_along(.pvals_list), function(i) {
            pv <- .pvals_list[[i]]
            pv <- pv[!is.na(pv)]
            if (length(pv) == 0L) return(NA_real_)
            mean(pv >= 0.045 & pv < 0.05)
          }, numeric(1))
        },

        # 2. p_heaping_flag_tei
        p_heaping_flag_tei = {
          vapply(seq_along(.pvals_list), function(i) {
            pv <- .pvals_list[[i]]
            pv <- pv[!is.na(pv)]
            n_total <- length(pv)
            if (n_total < 5L) return(NA_real_)
            n_thresh <- sum(pv == 0.05) + sum(pv == 0.01) + sum(pv == 0.001)
            as.numeric(n_thresh / n_total > 0.15)
          }, numeric(1))
        },

        # 3. Test-type counts
        n_t_tests_tei    = if_else(is.na(.txt), NA_integer_,
                                   as.integer(str_count(.txt, regex("\\bt\\s*\\(\\s*\\d")))),
        n_f_tests_tei    = if_else(is.na(.txt), NA_integer_,
                                   as.integer(str_count(.txt, regex("\\bF\\s*\\(\\s*\\d")))),
        n_chi2_tests_tei = if_else(is.na(.txt), NA_integer_,
                                   as.integer(str_count(.txt,
                                                        regex("(?:χ²|chi[- ]?(?:square|sq)|χ2)\\s*\\(\\s*\\d", ignore_case = TRUE)))),
        n_or_reported_tei = if_else(is.na(.txt), NA_integer_,
          as.integer(str_count(.txt, regex("\\bOR\\s*[=<>]")))),
        n_hr_reported_tei = if_else(is.na(.txt), NA_integer_,
          as.integer(str_count(.txt, regex("\\bHR\\s*[=<>]")))),

        # 4. tests_per_1000_words_tei
        tests_per_1000_words_tei = if_else(
          is.na(.txt), NA_real_,
          pmin(
            (n_t_tests_tei + n_f_tests_tei + n_chi2_tests_tei +
               n_or_reported_tei + n_hr_reported_tei) /
              pmax(fulltext_length_nlp / 1000, 1),
            200
          )
        ),

        # 5. hedging_density_discussion_only_tei
        hedging_density_discussion_only_tei = {
          vapply(seq_along(.tei_papers$.txt), function(i) {
            x <- .tei_papers$.txt[i]
            if (is.na(x)) return(NA_real_)
            m <- regexpr("(?i)\\b(General Discussion|Discussion|Conclusions?|Limitations)\\b",
                         x, perl = TRUE)
            if (m == -1L) {
              # No explicit heading found: discussions sit at the end, so fall
              # back to the final 30% of the document rather than returning NA.
              # LORO ablation kept this: it helps RF/XGB ~0.001-0.002 Brier; the
              # linear learners regress slightly but don't drive the stack.
              tail_txt <- substr(x, floor(nchar(x) * 0.7) + 1L, nchar(x))
            } else {
              tail_txt <- substr(x, m + attr(m, "match.length"), nchar(x))
            }
            nch <- nchar(tail_txt)
            if (nch < 10L) return(NA_real_)
            str_count(tail_txt, .hedging_rx) / nch * 1000
          }, numeric(1))
        },

        # 6. has_apriori_power_methods_tei
        has_apriori_power_methods_tei = {
          vapply(seq_along(.tei_papers$.txt), function(i) {
            x <- .tei_papers$.txt[i]
            if (is.na(x)) return(NA_integer_)
            m_start <- regexpr("(?i)\\bMethods?\\b", x, perl = TRUE)
            m_end   <- regexpr("(?i)\\bResults?\\b",  x, perl = TRUE)
            if (m_start == -1L && m_end == -1L) {
              # fallback: full text
              return(as.integer(str_detect(x,
                regex("a[- ]priori power|power analysis a priori", ignore_case = TRUE))))
            }
            start_pos <- if (m_start != -1L) m_start else 1L
            end_pos   <- if (m_end   != -1L) m_end   else nchar(x)
            section   <- substr(x, start_pos, end_pos)
            as.integer(str_detect(section,
              regex("a[- ]priori power|power analysis a priori", ignore_case = TRUE)))
          }, integer(1))
        },

        # 7. has_preregistration_rx_tei
        has_preregistration_rx_tei = if_else(is.na(.txt), NA_integer_,
          as.integer(str_detect(.txt,
            regex("preregister|pre-register|aspredicted|OSF Registries|prereg",
                  ignore_case = TRUE)))),

        # 8. has_open_data_rx_tei
        has_open_data_rx_tei = if_else(is.na(.txt), NA_integer_,
          as.integer(str_detect(.txt,
            regex(paste0("data (?:are|will be|is) available|",
                         "deposited (?:at|in)|github\\.com|osf\\.io|",
                         "figshare|zenodo|dataverse"),
                  ignore_case = TRUE)))),

        # 9. has_ethics_approval_tei
        has_ethics_approval_tei = if_else(is.na(.txt), NA_integer_,
          as.integer(str_detect(.txt,
            regex(paste0("ethics (?:approval|committee|board)|IRB|",
                         "institutional review board|ethic(?:al|s) approval"),
                  ignore_case = TRUE)))),

        # 10. has_robustness_section_tei
        has_robustness_section_tei = {
          vapply(seq_along(.tei_papers$.txt), function(i) {
            x <- .tei_papers$.txt[i]
            hdgs_col <- if ("section_headings_xml" %in% names(.tei_papers))
              .tei_papers$section_headings_xml[i] else NA_character_
            search_in <- if (!is.na(hdgs_col) && nchar(trimws(hdgs_col)) > 0)
              hdgs_col else x
            if (is.na(search_in)) return(NA_integer_)
            as.integer(str_detect(search_in,
              regex(paste0("robustness|sensitivity analysis|",
                           "supplementary analysis|additional analyses?"),
                    ignore_case = TRUE)))
          }, integer(1))
        },

        # 11. multistudy_paper_tei
        multistudy_paper_tei = {
          vapply(seq_along(.tei_papers$.txt), function(i) {
            x <- .tei_papers$.txt[i]
            hdgs_col <- if ("section_headings_xml" %in% names(.tei_papers))
              .tei_papers$section_headings_xml[i] else NA_character_
            search_in <- if (!is.na(hdgs_col) && nchar(trimws(hdgs_col)) > 0)
              hdgs_col else x
            if (is.na(search_in)) return(NA_integer_)
            as.integer(str_count(search_in,
              regex("(?:^Study\\s+\\d|^Experiment\\s+\\d)",
                    ignore_case = TRUE, multiline = TRUE)))
          }, integer(1))
        },
        # 12. multiplicity-correction reporting
        has_multiplicity_correction_tei = if_else(is.na(.txt), NA_integer_,
              as.integer(str_detect(.txt,
                        regex(paste0("bonferroni|\\bholm\\b|\\bsid[aá]k\\b|tukey|scheff[eé]|",
                        "false discovery rate|\\bFDR\\b|benjamini|",
                        "(?:correct\\w*|adjust\\w*) for multiple (?:compar|test)|",
                        "multiple[- ]comparison(?:s)? correction|family[- ]wise"),
              ignore_case = TRUE)))),

        has_multiplicity_correction1_tei = if_else(is.na(.txt), NA_integer_,
                                                   as.integer(str_detect(.txt, regex(paste0(
                                                     # Named methods (with word boundaries)
                                                     "\\b(?:bonferroni|holm|sid[aá]k|š[ií]d[aá]k|benjamini|hochberg|hommel|",
                                                     "dunnett|scheff[eé]|westfall)\\b|",
                                                     # Tukey only in a genuine test context
                                                     "\\btukey(?:'s)?(?:[- ](?:hsd|test|post[- ]hoc))?\\b|",
                                                     # Combined names
                                                     "\\bnewman[- ]keuls\\b|\\bbh[- ](?:procedure|method)\\b|\\bby[- ]procedure\\b|",
                                                     # Concepts
                                                     "false discovery rate|\\b(?:FDR|FWER)\\b|\\bq[- ]values?\\b|",
                                                     "family[- ]wise(?:[- ]error[- ]rate)?|",
                                                     # Phrases with p-values
                                                     "(?:adjust|correct)\\w* p[- ]?values?|",
                                                     # Phrases: correct/adjust for multiple ...
                                                     "(?:correct|adjust)\\w* for multiple (?:compar|test|hypoth)|",
                                                     # multiple testing/comparison correction/adjustment
                                                     "multiple[- ](?:compar\\w*|test\\w*|hypothes\\w*)[- ]?(?:correction|adjust\\w*)|",
                                                     # alpha correction
                                                     "\\balpha[- ](?:adjust\\w*|correct\\w*)|α[- ](?:adjust\\w*|correct\\w*)|",
                                                     # step-up/step-down
                                                     "step[- ](?:up|down)[- ]procedure|",
                                                     # permutation-based
                                                     "permutation[- ]based (?:correction|adjust\\w*)"
                                                   ), ignore_case = TRUE)))),

        # 13. interaction: test count left uncorrected (0 if correction reported)
        n_tests_uncorrected_tei = if_else(is.na(.txt), NA_real_,
                                          (n_t_tests_tei + n_f_tests_tei + n_chi2_tests_tei) *
                                            (1 - dplyr::coalesce(has_multiplicity_correction_tei, 0L))),
        n_tests_uncorrected1_tei = if_else(is.na(.txt), NA_real_,
                                          (n_t_tests_tei + n_f_tests_tei + n_chi2_tests_tei) *
                                            (1 - dplyr::coalesce(has_multiplicity_correction1_tei, 0L)))
      )

    rm(.pvals_list)

    # join TEI features back onto claim-level data
    .tei_cols <- grep("_tei$", names(.tei_papers), value = TRUE)
    data <- data |>
      left_join(.tei_papers |> select(doi_o, all_of(.tei_cols)), by = "doi_o")
    rm(.tei_papers, .tei_cols)
  }

  # ── Reference recency + methods-detail features ─────────────────────────────
  # mean_reference_year / pct_recent_references parse the per-reference year list
  # extracted from TEI <biblStruct> dates (reference_years_xml). pct_recent is
  # anchored on year_o (authoritative; pub_date_xml is only ~31% complete) and
  # measures the share of references published within 5 years of the paper.
  # methods_share = methods words / total body words: chosen over methods/results
  # because it normalises for paper length and isn't distorted by a terse or
  # analysis-heavy results section.
  if (!"reference_years_xml" %in% names(data))
    data <- data |> mutate(reference_years_xml = NA_character_)
  if (!"article_word_count_xml" %in% names(data))
    data <- data |> mutate(article_word_count_xml = NA_real_)

  .ref_year_list <- str_split(as.character(data$reference_years_xml), ";")
  .yr0           <- suppressWarnings(as.integer(data$year_o))

  data$n_references_dated <- vapply(.ref_year_list, function(v) {
    yrs <- suppressWarnings(as.integer(v)); sum(!is.na(yrs))
  }, integer(1))
  data$mean_reference_year <- vapply(.ref_year_list, function(v) {
    yrs <- suppressWarnings(as.integer(v)); yrs <- yrs[!is.na(yrs)]
    if (length(yrs) == 0L) NA_real_ else mean(yrs)
  }, numeric(1))
  data$pct_recent_references <- vapply(seq_along(.ref_year_list), function(i) {
    yrs <- suppressWarnings(as.integer(.ref_year_list[[i]])); yrs <- yrs[!is.na(yrs)]
    y0  <- .yr0[i]
    if (length(yrs) == 0L || is.na(y0)) return(NA_real_)
    mean(yrs >= (y0 - 5L))
  }, numeric(1))

  # methods_share: sum methods word counts across (multi-study) sections, then
  # normalise by total body words. methods ⊂ body, so cap at 1 for safety.
  # methods_words == 0 means GROBID detected no methods section (no TEI, or a
  # section-parse failure) — not a genuine zero, since every empirical paper has
  # methods. Both features are set to NA there so the magnitude isn't a false
  # floor; the "no methods section found" signal survives in *_miss_ind and stays
  # separable from "no TEI at all" via fulltext_missing_nlp.
  .methods_cols  <- grep("^methods_[0-9]+_word_count$", names(data), value = TRUE)
  .methods_words <- if (length(.methods_cols) == 0L) rep(NA_real_, nrow(data)) else
    rowSums(as.matrix(data[.methods_cols]), na.rm = TRUE)
  data$methods_words_total <- if_else(.methods_words > 0, .methods_words, NA_real_)
  data$methods_share <- if_else(
    is.na(data$article_word_count_xml) | data$article_word_count_xml == 0 |
      .methods_words == 0,
    NA_real_,
    pmin(.methods_words / data$article_word_count_xml, 1)
  )

  data <- select(data, -any_of("reference_years_xml"))
  rm(.ref_year_list, .yr0, .methods_cols, .methods_words)

  # ── Additional missingness indicators (5m-i) ────────────────────────────────
  .miss_ind_vars <- c(
    "es_value_o", "top_factor_score", "apc_usd",
    "grim_fail_rate_sc", "grim_mean_prob_sc",
    "limitations_1_word_count", "last_author_country_gdp",
    "es_type_o", "pval_type_o", "pub_date_xml",
    "submission_note_xml", "first_author_institution_country",
    "last_author_institution_country",
    # harmonized effect size proxies
    "r_harmonized", "evidence_strength_score",
    "t_proxy_from_r_h", "snr_proxy_h", "power_proxy_h",
    # bibliometric extras
    "citation_count_t_plus_2y", "n_referenced_works",
    # TEI bias indicators
    "p_hack_index_tei", "p_heaping_flag_tei",
    "n_t_tests_tei", "n_f_tests_tei", "n_chi2_tests_tei",
    "n_or_reported_tei", "n_hr_reported_tei",
    "tests_per_1000_words_tei", "hedging_density_discussion_only_tei",
    "has_apriori_power_methods_tei", "has_preregistration_rx_tei",
    "has_open_data_rx_tei", "has_ethics_approval_tei",
    "has_robustness_section_tei", "multistudy_paper_tei",
    # reference recency + methods detail
    "mean_reference_year", "pct_recent_references", "methods_share",
    "n_references_dated", "methods_words_total",
    # LLM features with substantial fallback rates in MICE
    # (flag genuine "LLM had no info" cases; fold-independent)
    "llm_perceived_surprisingness", "llm_sample_adequacy",
    "llm_within_paper_rep",         "llm_is_intervention"
  )
  for (.v in intersect(.miss_ind_vars, names(data))) {
    data[[paste0(.v, "_miss_ind")]] <- as.integer(is.na(data[[.v]]))
  }

  # ── log1p transformations (5m-ii) ──────────────────────────────────────────
  .log1p_vars <- c(
    "number_citations", "number_references", "number_self_references",
    "n_authors_oa", "fwci_via_oa",
    "journal_hindex", "journal_i10index", "journal_2yr_citedness",
    "journal_total_works", "journal_sjr", "journal_Hindex_via_sjr",
    "apc_usd", "top_factor_score",
    "first_author_works_count", "first_author_citations",
    "first_author_hindex",      "first_author_i10_index",
    "first_author_country_gdp",
    "last_author_works_count",  "last_author_citations",
    "last_author_hindex",       "last_author_i10_index",
    "last_author_country_gdp",
    "claim_length", "claim_n_digits",
    "n_authors_xml", "n_affiliations_xml", "n_sections_xml",
    "n_figures_xml", "n_tables_xml", "n_formulas_xml", "n_references_xml",
    "article_word_count_xml",
    "introduction_1_word_count", "methods_1_word_count",
    "results_1_word_count",      "discussion_1_word_count",
    "limitations_1_word_count",
    "title_length_nlp", "title_n_words_nlp", "title_n_digits_nlp",
    "abstract_length_nlp", "abstract_n_hedging_nlp", "abstract_n_pval_nlp",
    "abstract_n_significant_nlp",
    "fulltext_length_nlp", "fulltext_n_pval_nlp",
    "fulltext_n_stat_test_types_nlp", "fulltext_n_hedging_nlp",
    "fulltext_n_hypothesis_nlp", "fulltext_n_significant_nlp",
    "fulltext_n_limitations_nlp", "fulltext_n_figures_nlp",
    "fulltext_n_tables_nlp",
    "n_m_sd_pairs_sc", "grim_n_tested_sc",
    "readability_full_fk", "readability_full_smog",
    "readability_abs_fk",  "readability_abs_smog","first_author_productivity", "last_author_productivity",
    # harmonized effect size extras
    "evidence_strength_score", "citation_count_t_plus_2y", "n_referenced_works",
    # TEI test counts
    "tests_per_1000_words_tei",
    "n_t_tests_tei", "n_f_tests_tei", "n_chi2_tests_tei",
    "n_or_reported_tei", "n_hr_reported_tei",
    # reference / methods counts
    "n_references_dated", "methods_words_total"
  )
  for (.v in intersect(.log1p_vars, names(data))) {
    data[[paste0(.v, "_log1p")]] <- log1p(data[[.v]])
  }

  # ── sqrt(abs()) transformations (5m-iii) ────────────────────────────────────
  .sqrt_vars <- c(
    .log1p_vars,
    "es_value_o", "duration_o_r",
    "title_sentiment_nlp", "abstract_sentiment_nlp"
  )
  for (.v in intersect(.sqrt_vars, names(data))) {
    data[[paste0(.v, "_sqrtabs")]] <- sqrt(abs(data[[.v]]))
  }

  rm(.miss_ind_vars, .log1p_vars, .sqrt_vars, .v)
  data
}

# ------------------------------------------------------------------------------

train_base <- apply_features(train_joined)
test_base  <- apply_features(test_joined)

stopifnot(!any(str_detect(names(train_base), "_imp$")))
stopifnot(!any(str_detect(names(test_base),  "_imp$")))

# ── Optional filters ─────────────────────────────────────────────────────────

# Filter retracted (n = 6 in train, 0 in test)
train_base <- train_base |> filter(is_retracted == FALSE | is.na(is_retracted))

# Hypothesis-specific LLM vars → NA when no corresponding hypothesis found
# Version A: all semantically hypothesis-dependent vars (broader)
HYP_COLS_ALL <- c(
  "llm_structured_hypothesis", "llm_directional_hypothesis",
  "llm_theory_based_hypothesis", "llm_exploratory_finding",
  "llm_hypothesis_main_effect", "llm_hypothesis_mediation",
  "llm_hypothesis_moderation", "llm_complex_hypothesis",
  "llm_hyp_inconsistent", "llm_intuitive_prediction",
  "llm_perceived_surprisingness", "llm_construct_complex_ord",
  "llm_power_guess", "llm_contradicting_abstract",
  "llm_contradicting_theory", "llm_contradicting_discussion",
  "llm_contradicting_any", "llm_strength_power_ratio",
  "llm_surprise_x_strength", "llm_intuition_evidence_gap",
  "llm_spec_x_theory", "llm_sample_adequacy"
)

na_if_no_hyp <- function(df, cols) {
  # Guard: if the gating column is absent (e.g. the LLM extraction didn't yet
  # emit `llm_corresponding_hypothesis_found`), skip the NA-out step instead
  # of erroring. Pre-existing column dependency, not part of harmonization.
  if (!"llm_corresponding_hypothesis_found" %in% names(df)) return(df)
  df |> mutate(across(
    all_of(intersect(cols, names(df))),
    ~ if_else(llm_corresponding_hypothesis_found == 0L, NA, .x,
              missing = .x)
  ))
}

train_base <- na_if_no_hyp(train_base, HYP_COLS_ALL)
test_base  <- na_if_no_hyp(test_base,  HYP_COLS_ALL)

# Filter to central / mentioned / unsure claims in the training dataset.
# Keep rows with NA claim_status_llm (R1/R2 rows have no LLM screening) — otherwise
# `!= "not_implied"` evaluates to NA and silently drops them.
train_base <- train_base %>%
  filter(is.na(claim_status_llm) | claim_status_llm != "not_implied")

# filter on reported_success — keep rows with valid FORRT-style outcome
# (failed/mixed/successful) OR rows with ground-truth statistical_success
# (R1/R2 rows have no reported_success but DO have ground truth — they must
# survive this filter, otherwise leave-one-round-out validation is impossible).
train_base <- train_base %>%
  filter(reported_success %in% c("failed", "mixed", "successful") |
           !is.na(statistical_success))

# filter on author_overlap in train studies — keep rows where it is FALSE OR NA.
# author_overlap is NA for R1/R2 (and many FORRT rows); the previous filter
# silently dropped all of them because NA == FALSE evaluates to NA, which
# dplyr::filter treats as drop. That breaks leave-one-round-out validation.
train_base <- train_base %>%
  filter(is.na(author_overlap) | author_overlap == FALSE)

# ── Judge-LLM ensemble predictions (new predictor pool) ───────────────────────
# judge_llm_ensemble/ensemble_scores.csv holds the 12-judge ensemble's direct
# replication-probability predictions. They cover R1/R2/R3 but NOT FORRT rows,
# so we flag missingness and impute gaps with the column mean over covered rows
# (feature-only imputation — never touches the outcome). Feeds `enriched_llm`.
judge_ens <- readr::read_csv(here::here("judge_llm_ensemble/ensemble_scores.csv"),
                             show_col_types = FALSE) |>
  dplyr::select(claim_id, judge_mean_p, judge_median_p, judge_sd_p,
                judge_model_disagreement, judge_persona_disagreement) |>
  dplyr::distinct(claim_id, .keep_all = TRUE)

.JUDGE_COLS <- setdiff(names(judge_ens), "claim_id")

attach_judge <- function(df) {
  df <- dplyr::left_join(df, judge_ens, by = "claim_id")
  df$judge_llm_miss_ind <- as.integer(is.na(df$judge_mean_p))
  for (col in .JUDGE_COLS) {
    covered <- df[[col]][!is.na(df[[col]])]
    if (length(covered) > 0) df[[col]][is.na(df[[col]])] <- mean(covered)
  }
  df
}
train_base <- attach_judge(train_base)
test_base  <- attach_judge(test_base)
cat("Judge-LLM coverage — train:",
    sum(train_base$judge_llm_miss_ind == 0), "/", nrow(train_base),
    " test:", sum(test_base$judge_llm_miss_ind == 0), "/", nrow(test_base), "\n")

saveRDS(train_base, here::here("data/train_base.rds"), compress = "xz")
saveRDS(test_base,  here::here("data/test_base.rds"),  compress = "xz")

cat("train_base:", nrow(train_base), "x", ncol(train_base), "\n")
cat("test_base: ", nrow(test_base),  "x", ncol(test_base),  "\n")
