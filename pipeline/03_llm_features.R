# ──────────────────────────────────────────────────────────────────────────────
# pipeline/03_llm_features.R
#
# PURPOSE
#   Parse LLM-extracted JSON files from the extraction directories and produce a
#   single row per claim_id.
#
# INPUTS  : JSON files under the directories listed in JSON_DIRS (below)
# OUTPUTS : data/cache/llm_features.rds (one row per claim_id)
#
# PROVIDER DISAGREEMENT
#   Some claim_ids may be extracted by multiple providers (e.g. both "codex"
#   and "openai"). When two providers extract the same claim, their ratings
#   may differ — this difference (disagreement) is itself a signal: claims
#   where providers disagree are likely more ambiguous and therefore harder
#   to replicate. We capture this signal in `_disagree` columns.
#
# AGGREGATION RULES (per column type)
#   Numeric/ordinal cols  : <name>_mean = mean across providers (canonical);
#                           <name>_disagree = abs diff between providers
#                           (NA when only one provider present).
#                           p_numeric and n_numeric are handled in log-space.
#   Binary yes/no cols    : <name>_mean = mean (0, 0.5, or 1 when two
#                           providers present — acts as a consensus probability);
#                           <name>_disagree = 1 if providers conflict, 0 otherwise
#                           (NA when only one provider present).
#   Categorical cols      : keep first provider's value; add <name>_disagree
#                           binary indicator = 1 if providers differ.
#   Text/proof cols       : concatenate both providers' text with " ||| "
#                           separator so all evidence is preserved.
# ──────────────────────────────────────────────────────────────────────────────

library(jsonlite)
library(dplyr)
library(purrr)
library(stringr)
library(tidyr)

# Honor the pipeline-wide toggle if 00_packages_config was sourced; otherwise
# default to legacy behaviour so this script still runs standalone.
if (!exists("NA_counts_Zero")) NA_counts_Zero <- FALSE

# The structured-extraction caches that fed the submission. Each directory is one
# production run of llm_extraction/run_combined.R, named raw_responses_combined_<tag>:
#   codex54hi_v3        →  --provider codex  --tag codex54hi_v3
#   openai54flex_v1     →  --provider openai --model gpt-5.4 --flex --tag openai54flex_v1
#   codex54hi_addl_v1 / openai54flex_addl_v1  →  the same two providers re-run over the
#                          ADDITIONAL claim subset (a different manifest), with those tags
#   openai54flora_v1    →  the 17 new-data health replications from the flora add-on
# NOTE: run_combined.R's default tag is "gem35flash" (gemini) — that directory is NOT
# read here. To regenerate a cache for this parser you must pass the matching --tag
# (and the matching claim manifest for the addl/flora subsets). Most users skip this
# entirely: the parsed values ship as data/cache/llm_features.rds. To point the parser
# at your own regenerated directories, set LLM_EXTRACTION_DIRS (comma-separated paths).
JSON_DIRS <- local({
  override <- Sys.getenv("LLM_EXTRACTION_DIRS", "")
  if (nzchar(override)) trimws(strsplit(override, ",", fixed = TRUE)[[1]]) else c(
    "llm_extraction/raw_responses_combined_codex54hi_v3",
    "llm_extraction/raw_responses_combined_openai54flex_v1",
    "llm_extraction/raw_responses_combined_codex54hi_addl_v1",
    "llm_extraction/raw_responses_combined_openai54flex_addl_v1",
    "llm_extraction/raw_responses_combined_openai54flora_v1"
  )
})
OUT_PATH <- "data/cache/llm_features.rds"

# ── Helper functions ───────────────────────────────────────────────────────────

yesno_to_num <- function(x) {
  # yes=1, no=0, unclear=NA
  case_when(x == "yes" ~ 1L, x == "no" ~ 0L, TRUE ~ NA_integer_)
}

partial_to_num <- function(x) {
  # yes=1, partly=0.5, no=0, unclear=NA
  case_when(
    x == "yes"    ~ 1,
    x == "partly" ~ 0.5,
    x == "no"     ~ 0,
    TRUE          ~ NA_real_
  )
}

parse_p <- function(s) {
  if (is.null(s) || is.na(s)) return(NA_real_)
  s <- str_trim(s)
  if (s %in% c("not reported", "not stated", "none", "")) return(NA_real_)

  # OCR artifact replacements (corrupt unicode chars from PDF extraction)
  s <- str_replace_all(s, "¼",             "=")   # ¼ → =
  s <- str_replace_all(s, "Ͻ|Ͷ|Ͻ",   "<")   # Ͻ → <
  s <- str_replace_all(s, "ϭ|ϭ",           "=")   # ϭ → =
  s <- str_replace_all(s, "(?i)less\\s+than",   "<")
  s <- str_replace_all(s, "(?i)p[- ]?value[:]?","p")
  s <- str_replace_all(s, "po\\.",              "p<0.")  # "po.05" → "p<0.05"
  s <- str_replace_all(s, "pnorm[:]?",          "p=")
  # "p 5 .001" / "p 5 0.001": digit 5 as OCR for < between p and a number
  s <- str_replace_all(s, "(?<=\\bp)\\s+5\\s+(?=[0.])", " < ")
  # "p = < 0.001" → "p < 0.001"
  s <- str_replace_all(s, "=\\s*<",             "<")
  s <- str_replace_all(s, "(\\d)\\s+(\\d{3})\\b", "\\1.\\2")
  # scientific notation: "1.7 × 10 -9" or "1.7×10-9"
  sci <- str_match(s, "([0-9]+\\.?[0-9]*)\\s*[×xX\\*]\\s*10\\s*[-−]\\s*([0-9]+)")
  if (!is.na(sci[1, 1]))
    return(as.numeric(sci[1, 2]) * 10^(-as.numeric(sci[1, 3])))

  # "p value = 0.000" → treat as < 0.001
  if (str_detect(s, "(?i)[=<]\\s*0?\\.0{2,}(?:[^0-9]|$)")) return(0.0005)

  # Extract all numeric p-values from string (handles "P=0.04; P=0.01", "ps < .001" etc.)
  nums <- str_match_all(
    s,
    "(?i)p[s]?\\s*[=<>≤≥\\u2264\\u2265]?\\s*[=<>≤]?\\s*(0?\\.[0-9]+(?:e[-]?[0-9]+)?)"
  )[[1]]

  if (nrow(nums) > 0) {
    vals <- suppressWarnings(as.numeric(nums[, 2]))
    vals <- vals[!is.na(vals) & vals > 0 & vals <= 1]
    if (length(vals) > 0) return(min(vals))
  }

  # Fallback: bare decimal anywhere in string
  bare <- str_extract(s, "0?\\.[0-9]+")
  if (!is.na(bare)) {
    v <- as.numeric(bare)
    if (!is.na(v) && v > 0 && v <= 1) return(v)
  }

  NA_real_
}

parse_n <- function(s) {
  if (is.null(s) || is.na(s) || s %in% c("not reported", "none identified", "")) return(NA_real_)
  # "more than 200,000" → 200000
  s_clean <- str_remove_all(s, ",")
  m <- str_extract(s_clean, "[0-9]+")
  if (!is.na(m)) return(as.numeric(m))
  NA_real_
}

parse_alpha <- function(s) {
  if (is.null(s) || is.na(s) || s %in% c("not stated", "")) return(NA_real_)
  m <- str_extract(s, "0\\.0[0-9]+")
  if (!is.na(m)) return(as.numeric(m))
  NA_real_
}

count_hypotheses <- function(s) {
  if (is.null(s) || is.na(s) || s %in% c("none stated", "")) return(0L)
  str_count(s, fixed("|||")) + 1L
}

safe_get <- function(lst, ...) {
  keys <- list(...)
  tryCatch(
    reduce(keys, function(acc, k) acc[[k]], .init = lst),
    error = function(e) NULL
  )
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

# ── Parse one JSON file into a named list (one row) ───────────────────────────

parse_json_row <- function(path) {

  raw <- tryCatch(fromJSON(path, simplifyVector = FALSE),
                  error = function(e) NULL)
  if (is.null(raw)) {
    warning("Failed to parse: ", path)
    return(NULL)
  }

  so <- raw$structured_output

  yn  <- function(field) yesno_to_num(safe_get(so, field, "status"))
  num <- function(field, sub = "value") {
    v <- safe_get(so, field, sub)
    if (is.null(v)) NA_real_ else as.numeric(v)
  }
  chr <- function(field, sub = "value") {
    v <- safe_get(so, field, sub)
    if (is.null(v)) NA_character_ else as.character(v)
  }

  # -- p-value parse (string normalisation/midpoint handling is done in parse_p)
  p_raw   <- chr("p_o_llm")
  p_num   <- parse_p(p_raw)
  n_num   <- parse_n(chr("n_o_llm"))
  a_num   <- parse_alpha(chr("alpha_thresh_llm"))

  # -- ordinal mappings
  plaus_map  <- c("Low" = 1L, "Medium" = 2L, "High" = 3L)
  ss_map     <- c("very small" = 1L, "small" = 2L, "medium" = 3L,
                  "large" = 4L, "very large" = 5L)
  es_map     <- c("negligible" = 1L, "small" = 2L, "medium" = 3L,
                  "large" = 4L, "very large" = 5L, "not reported" = NA_integer_)
  wq_map     <- c("poor" = 1L, "fair" = 2L, "good" = 3L, "excellent" = 4L)
  cc_map     <- c("low" = 1L, "moderate" = 2L, "high" = 3L)
  rob_map    <- c("results maintained"          = 3L,
                  "results partially maintained" = 2L,
                  "results not maintained"       = 1L,
                  "not applicable"               = NA_integer_)

  plaus_raw <- chr("plausibility_of_successful_replication", "rating")
  ss_raw    <- chr("sample_size_classified_by_llm",  "rating")
  es_raw    <- chr("effect_magnitude_classified_by_llm", "rating")
  wq_raw    <- chr("writing_quality", "rating")
  cc_raw    <- chr("construct_complexity_of_hypothesis", "rating")
  rob_raw   <- chr("outcome_robustness_check", "rating")
  int_raw   <- chr("intervention_study_post_measure", "rating")
  # "not applicable" = genuine 0 (LLM judged non-intervention); NA = not extracted.
  is_interv_  <- as.integer(!is.null(int_raw) & !is.na(int_raw) & int_raw != "not applicable")
  has_prepost_ <- as.integer(!is.null(int_raw) & !is.na(int_raw) &
                               int_raw %in% c("Pre-Post-Design", "Pre-Post-Follow-up"))
  if (NA_counts_Zero && (is.null(int_raw) || is.na(int_raw))) {
    is_interv_   <- NA_integer_
    has_prepost_ <- NA_integer_
  }

  # -- open materials (practice_partial)
  om_status <- safe_get(so, "open_materials", "status")
  om_num    <- partial_to_num(if (is.null(om_status)) NA_character_ else om_status)

  # -- hypothesis count
  hyp_text  <- safe_get(so, "hypothesis_llm", "value")
  hyp_count <- count_hypotheses(if (is.null(hyp_text)) NA_character_ else hyp_text)

  # -- funding
  fund_val  <- chr("funding_disclosure")
  has_fund  <- as.integer(!is.na(fund_val) & !(fund_val %in% c("not reported", "")))
  # "not reported"/"" = genuine 0 (LLM saw the field); NA = field not extracted.
  if (NA_counts_Zero && is.na(fund_val)) has_fund <- NA_integer_

  # -- Auto-extract evidence/reasoning for all fields
  proof_list <- list()
  for (f in names(so)) {
    field <- so[[f]]
    if (!is.list(field)) next
    if ("evidence" %in% names(field))
      proof_list[[paste0("llm_", f, "_proof")]]       <- as.character(field$evidence  %||% NA_character_)
    if ("reasoning" %in% names(field))
      proof_list[[paste0("llm_", f, "_explanation")]] <- as.character(field$reasoning %||% NA_character_)
  }
  proof_list[["llm_notes"]] <- as.character(so$notes %||% NA_character_)
  proof_df <- as_tibble(proof_list)

  tibble(
    claim_id = raw$claim_id %||% NA_character_,

    # -- Meta (preserved per-row; collapsed after aggregation)
    llm_provider    = raw$provider  %||% NA_character_,
    llm_model       = raw$model     %||% NA_character_,
    llm_flex        = as.integer(isTRUE(raw$flex)),
    llm_reasoning   = raw$reasoning %||% NA_character_,
    llm_duration_s  = raw$duration_s %||% NA_real_,

    # -- Claim-level yesno (0/1/NA)
    llm_is_focal_claim               = yn("is_focal_claim"),
    llm_structured_hypothesis        = yn("structured_hypothesis"),
    llm_directional_hypothesis       = yn("directional_hypothesis"),
    llm_hypothesis_main_effect       = yn("hypothesis_on_main_effect"),
    llm_hypothesis_mediation         = yn("hypothesis_on_mediation"),
    llm_hypothesis_moderation        = yn("hypothesis_on_moderation"),
    llm_theory_based_hypothesis      = yn("theory_based_hypothesis"),
    llm_exploratory_finding          = yn("exploratory_finding"),
    llm_within_paper_replication_claim = yn("within_paper_replication_claim"),
    llm_contradicting_abstract       = yn("contradicting_evidence_abstract"),
    llm_contradicting_theory         = yn("contradicting_evidence_theory"),
    llm_contradicting_discussion     = yn("contradicting_evidence_discussion"),

    # -- Study design yesno
    llm_rct_study          = yn("RCT_study"),
    llm_experimental       = yn("experimental_study"),
    llm_observational      = yn("observational_study"),
    llm_cross_sectional    = yn("cross_sectional_study"),
    llm_longitudinal       = yn("longitudinal_study"),
    llm_student_sample     = yn("student_sample"),
    llm_paid_sample        = yn("paid_sample"),
    llm_online_paid_sample = yn("online_paid_sample"),
    llm_online_study       = yn("online_study"),
    llm_weird_sample       = yn("WEIRD_sample"),
    llm_secondary_data     = yn("secondary_data"),

    # -- Measurement / reporting yesno
    llm_self_report       = yn("self_report"),
    llm_validated_scale   = yn("validated_scale"),
    llm_multi_item        = yn("multi_item"),
    llm_apriori_power     = yn("apriori_power"),
    llm_posthoc_power     = yn("posthoc_power"),
    llm_power_section     = yn("power_section"),
    llm_has_robustness    = yn("has_robustness_check"),
    llm_preregistration   = yn("preregistration"),
    llm_exploratory_study = yn("exploratory_study"),

    # -- Open science yesno
    llm_open_data           = yn("open_data"),
    llm_open_data_request   = yn("open_data_uponrequest"),
    llm_open_code           = yn("open_code"),
    llm_is_health           = yn("is_health_llm"),
    llm_conflict_of_interest= yn("conflict_of_interest"),
    llm_within_paper_rep    = yn("within_paper_replication"),

    # -- Open materials (partial; 0/0.5/1/NA)
    llm_open_materials_num = om_num,

    # -- Numeric ratings
    llm_claim_specificity      = num("claim_specificity"),
    llm_theoretical_complexity = num("theoretical_complexity"),
    llm_power_guess            = num("llm_power_guess", "rating"),
    llm_strength_of_evidence   = num("strength_of_evidence", "rating"),
    llm_intuitive_prediction   = num("intuitive_prediction", "rating"),
    llm_perceived_surprisingness = num("perceived_surprisingness", "rating"),
    llm_study_count            = num("study_count"),

    # -- Ordinal → integer
    llm_plausibility_ord   = plaus_map[plaus_raw]  %||% NA_integer_,
    llm_sample_size_ord    = ss_map[ss_raw]         %||% NA_integer_,
    llm_effect_size_ord    = es_map[es_raw]         %||% NA_integer_,
    llm_writing_quality_ord= wq_map[wq_raw]         %||% NA_integer_,
    llm_construct_complex_ord = cc_map[cc_raw]      %||% NA_integer_,
    llm_robustness_ord     = rob_map[rob_raw]        %||% NA_integer_,

    # -- Intervention design
    llm_is_intervention    = is_interv_,
    llm_has_prepost        = has_prepost_,

    # -- Extracted numerics
    llm_n_numeric   = n_num,
    llm_p_numeric   = p_num,
    llm_alpha_num   = a_num,

    # -- Hypothesis & funding
    llm_hypothesis_count = hyp_count,
    llm_has_funding      = has_fund
  ) |>
  bind_cols(proof_df)
}

# ── 1. Load and bind all JSONs ─────────────────────────────────────────────────

json_files <- list.files(JSON_DIRS, pattern = "\\.json$", full.names = TRUE)
cat(sprintf("Found %d JSON files across %d directories\n", length(json_files), length(JSON_DIRS)))

llm_raw <- map(json_files, parse_json_row) |>
  compact() |>
  bind_rows()

cat(sprintf("Parsed %d rows (before aggregation)\n", nrow(llm_raw)))

# ── 2. Feature engineering (per-provider row) ─────────────────────────────────
# These derived columns are computed per row before cross-provider aggregation.
# Disagreement-eligible source fields (ordinals, numerics, binaries) are
# computed later from the raw parsed values.

llm_engineered <- llm_raw |>
  mutate(
    # Log-transforms (used in aggregation below; kept here for transparency)
    llm_log_n        = log1p(llm_n_numeric),
    llm_neg_log_p    = ifelse(llm_p_numeric > 0, -log10(llm_p_numeric), NA_real_),

    # Not-reported indicators (prevent misleading imputation to 0)
    llm_p_not_reported  = as.integer(is.na(llm_p_numeric)),
    llm_n_not_reported  = as.integer(is.na(llm_n_numeric)),
    llm_es_not_reported = as.integer(is.na(llm_effect_size_ord)),

    # Hypothesis inconsistency: LLM says no hypotheses but still found a match
    llm_hyp_inconsistent = as.integer(
      llm_hypothesis_count == 0L & !is.na(llm_structured_hypothesis) & llm_structured_hypothesis == 1L
    ),

    # Distance of p from alpha threshold (negative = significant)
    llm_p_vs_alpha   = log(llm_p_numeric / llm_alpha_num),

    # Open science composite (0-4; open_materials counts as 1 if fully open)
    llm_open_science_score = rowSums(
      cbind(llm_open_data,
            as.integer(llm_open_materials_num == 1),
            llm_open_code,
            llm_preregistration),
      na.rm = !NA_counts_Zero
    ),

    # Any power analysis
    llm_power_any = as.integer(
      pmax(llm_apriori_power, llm_posthoc_power, llm_power_section, na.rm = TRUE) == 1
    ),

    # Any contradicting evidence
    llm_contradicting_any = as.integer(
      pmax(llm_contradicting_abstract,
           llm_contradicting_theory,
           llm_contradicting_discussion, na.rm = TRUE) == 1
    ),

    # Study design type (priority: RCT > experimental > observational > unknown)
    llm_design_type = case_when(
      llm_rct_study    == 1L ~ "rct",
      llm_experimental == 1L ~ "experimental",
      llm_observational== 1L ~ "observational",
      TRUE                   ~ "unknown"
    ),

    # Moderating/mediating complexity flag
    llm_complex_hypothesis = as.integer(
      llm_hypothesis_mediation == 1L | llm_hypothesis_moderation == 1L
    ),

    # Ratio: evidence strength vs. power guess (if LLM over-rates strength rel. to power → concern)
    llm_strength_power_ratio = llm_strength_of_evidence / (llm_power_guess * 5 + 0.01),

    # Surprisingness × evidence strength interaction
    llm_surprise_x_strength  = llm_perceived_surprisingness * llm_strength_of_evidence,

    # Intuition gap: high intuition but low evidence → possibly publication bias
    llm_intuition_evidence_gap = llm_intuitive_prediction - (llm_strength_of_evidence * 2),

    # Hypothesis specificity × theoretical complexity
    llm_spec_x_theory = llm_claim_specificity * llm_theoretical_complexity,

    # LLM plausibility as probability (Low≈0.2, Medium≈0.55, High≈0.85)
    llm_plausibility_prob = case_when(
      llm_plausibility_ord == 1L ~ 0.20,
      llm_plausibility_ord == 2L ~ 0.55,
      llm_plausibility_ord == 3L ~ 0.85,
      TRUE ~ NA_real_
    ),

    # Sample adequacy: large sample + power analysis
    llm_sample_adequacy = (llm_sample_size_ord + coalesce(llm_power_any, 0L)) / 6,

    # Replication risk flag: small n + no prereg + no robustness
    llm_replication_risk = as.integer(
      coalesce(llm_sample_size_ord, 3L) <= 2L &
      coalesce(llm_preregistration,  0L) == 0L &
      coalesce(llm_has_robustness,   0L) == 0L
    )
  )

# ── 3. Identify column types for aggregation strategy ─────────────────────────

# These are the columns we want disagreement scores for.
# Numeric/ordinal: mean + abs-diff disagreement (p & n handled in log-space)
numeric_disagree_cols <- c(
  "llm_plausibility_ord", "llm_sample_size_ord", "llm_effect_size_ord",
  "llm_construct_complex_ord", "llm_writing_quality_ord", "llm_robustness_ord",
  "llm_claim_specificity", "llm_theoretical_complexity",
  "llm_strength_of_evidence", "llm_intuitive_prediction",
  "llm_perceived_surprisingness", "llm_power_guess"
)
# p & n in log-space (log1p to avoid -Inf for p=0 edge cases)
log_disagree_cols <- c("llm_p_numeric", "llm_n_numeric")

# Binary yes/no cols: mean = consensus probability; disagree = 1 if conflict
binary_disagree_cols <- c(
  "llm_open_data", "llm_preregistration", "llm_apriori_power",
  "llm_rct_study", "llm_experimental", "llm_observational",
  "llm_self_report", "llm_validated_scale", "llm_secondary_data",
  "llm_is_health",
  # additional binary cols worth tracking
  "llm_is_focal_claim", "llm_structured_hypothesis", "llm_directional_hypothesis",
  "llm_theory_based_hypothesis", "llm_exploratory_finding",
  "llm_within_paper_replication_claim", "llm_contradicting_abstract",
  "llm_contradicting_theory", "llm_contradicting_discussion",
  "llm_cross_sectional", "llm_longitudinal", "llm_student_sample",
  "llm_paid_sample", "llm_online_paid_sample", "llm_online_study",
  "llm_weird_sample", "llm_multi_item", "llm_posthoc_power",
  "llm_power_section", "llm_has_robustness", "llm_exploratory_study",
  "llm_open_data_request", "llm_open_code", "llm_conflict_of_interest",
  "llm_within_paper_rep"
)

# Categorical: keep first value + add _disagree binary indicator
categorical_cols <- c("llm_design_type")

# Text/proof: concatenate with " ||| " separator
text_cols <- grep("_proof$|_explanation$|llm_notes", names(llm_engineered), value = TRUE)

# Derived (single-row engineering outputs) — take mean across providers
derived_numeric_cols <- c(
  "llm_log_n", "llm_neg_log_p", "llm_p_vs_alpha",
  "llm_open_science_score", "llm_strength_power_ratio",
  "llm_surprise_x_strength", "llm_intuition_evidence_gap",
  "llm_spec_x_theory", "llm_plausibility_prob",
  "llm_sample_adequacy", "llm_plausibility_prob"
)
derived_binary_cols <- c(
  "llm_p_not_reported", "llm_n_not_reported", "llm_es_not_reported",
  "llm_hyp_inconsistent", "llm_power_any", "llm_contradicting_any",
  "llm_complex_hypothesis", "llm_replication_risk",
  "llm_is_intervention", "llm_has_prepost"
)
# Ordinal/count — take mean (round to integer where appropriate)
ordinal_count_cols <- c("llm_hypothesis_count", "llm_study_count", "llm_has_funding")

# Meta (take first/most common)
meta_cols <- c("llm_flex", "llm_duration_s", "llm_open_materials_num", "llm_alpha_num")

# ── 4. Aggregate across providers ─────────────────────────────────────────────

# Concatenate non-NA text values from multiple providers with " ||| " separator
concat_text <- function(x) {
  vals <- x[!is.na(x) & nchar(x) > 0]
  if (length(vals) == 0) return(NA_character_)
  paste(unique(vals), collapse = " ||| ")
}

# NA-safe mean: returns NA_real_ when every value is NA, instead of mean(...
# na.rm = TRUE)'s NaN. NaN is not caught by `is.na()` or `tidyr::replace_na()`
# in some downstream code paths, so we normalise to NA at aggregation time.
safe_mean <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

# Compute disagreement for numeric column: abs diff when both present, else NA
numeric_disagree <- function(x) {
  vals <- x[!is.na(x)]
  if (length(vals) < 2) return(NA_real_)
  abs(vals[1] - vals[2])
}

# Compute disagreement for binary column: 1 if conflict, 0 if agree, NA if only one
binary_disagree <- function(x) {
  vals <- x[!is.na(x)]
  if (length(vals) < 2) return(NA_integer_)
  as.integer(vals[1] != vals[2])
}

# Compute disagreement for categorical column: 1 if different, 0 if same, NA if only one
categorical_disagree <- function(x) {
  vals <- x[!is.na(x)]
  if (length(vals) < 2) return(NA_integer_)
  as.integer(vals[1] != vals[2])
}

# Group by claim_id and aggregate
llm_features <- llm_engineered |>
  group_by(claim_id) |>
  summarise(
    # -- Provider metadata: collapse to comma-separated string
    llm_provider   = paste(unique(na.omit(llm_provider)), collapse = ","),
    llm_model      = paste(unique(na.omit(llm_model)),    collapse = ","),
    llm_reasoning  = paste(unique(na.omit(llm_reasoning)), collapse = ","),
    llm_n_providers = n(),

    # -- Numeric/ordinal cols: mean + abs-diff disagreement
    across(
      all_of(numeric_disagree_cols),
      list(
        mean    = ~safe_mean(.),
        disagree = ~numeric_disagree(as.numeric(.))
      ),
      .names = "{.col}_{.fn}"
    ),

    # -- Log-space disagreement for p and n
    # numeric: mean in log-space
    llm_p_numeric_mean    = safe_mean(llm_p_numeric),
    llm_p_numeric_disagree = numeric_disagree(log1p(llm_p_numeric)),
    llm_n_numeric_mean    = safe_mean(llm_n_numeric),
    llm_n_numeric_disagree = numeric_disagree(log1p(llm_n_numeric)),

    # -- Binary cols: mean = consensus probability; disagreement = 1 if providers conflict
    across(
      all_of(binary_disagree_cols),
      list(
        mean     = ~safe_mean(.),
        disagree = ~binary_disagree(.)
      ),
      .names = "{.col}_{.fn}"
    ),

    # -- Categorical col: keep first non-NA value; add disagreement indicator
    llm_design_type_disagree = categorical_disagree(llm_design_type),
    llm_design_type          = first(na.omit(llm_design_type)),

    # -- Text/proof cols: concatenate all providers' text
    across(all_of(text_cols), concat_text),

    # -- Derived numeric: mean across providers
    across(all_of(intersect(derived_numeric_cols, names(llm_engineered))),
           ~safe_mean(.)),

    # -- Derived binary: mean across providers (0/0.5/1)
    across(all_of(intersect(derived_binary_cols, names(llm_engineered))),
           ~safe_mean(.)),

    # -- Ordinal/count: mean (rounded)
    across(all_of(intersect(ordinal_count_cols, names(llm_engineered))),
           ~safe_mean(.)),

    # -- Meta: mean or first
    across(all_of(intersect(meta_cols, names(llm_engineered))),
           ~safe_mean(.)),

    .groups = "drop"
  ) |>
  mutate(
    # Restore factor for llm_design_type (levels match original)
    llm_design_type = factor(llm_design_type,
                             levels = c("unknown", "observational", "experimental", "rct")),

    # Convenience: _mean columns are the canonical values — also expose under original name
    # so that downstream 04_base_dataset.R column references still work unchanged.
    # Ordinals: rename _mean back to original name for the primary signal
    llm_plausibility_ord      = llm_plausibility_ord_mean,
    llm_sample_size_ord       = llm_sample_size_ord_mean,
    llm_effect_size_ord       = llm_effect_size_ord_mean,
    llm_construct_complex_ord = llm_construct_complex_ord_mean,
    llm_writing_quality_ord   = llm_writing_quality_ord_mean,
    llm_robustness_ord        = llm_robustness_ord_mean,
    llm_claim_specificity     = llm_claim_specificity_mean,
    llm_theoretical_complexity= llm_theoretical_complexity_mean,
    llm_strength_of_evidence  = llm_strength_of_evidence_mean,
    llm_intuitive_prediction  = llm_intuitive_prediction_mean,
    llm_perceived_surprisingness = llm_perceived_surprisingness_mean,
    llm_power_guess           = llm_power_guess_mean,
    llm_p_numeric             = llm_p_numeric_mean,
    llm_n_numeric             = llm_n_numeric_mean,

    # Binary cols: restore original names as mean (consensus probability)
    llm_open_data             = llm_open_data_mean,
    llm_preregistration       = llm_preregistration_mean,
    llm_apriori_power         = llm_apriori_power_mean,
    llm_rct_study             = llm_rct_study_mean,
    llm_experimental          = llm_experimental_mean,
    llm_observational         = llm_observational_mean,
    llm_self_report           = llm_self_report_mean,
    llm_validated_scale       = llm_validated_scale_mean,
    llm_secondary_data        = llm_secondary_data_mean,
    llm_is_health             = llm_is_health_mean,
    llm_is_focal_claim        = llm_is_focal_claim_mean,
    llm_structured_hypothesis = llm_structured_hypothesis_mean,
    llm_directional_hypothesis= llm_directional_hypothesis_mean,
    llm_theory_based_hypothesis= llm_theory_based_hypothesis_mean,
    llm_exploratory_finding   = llm_exploratory_finding_mean,
    llm_within_paper_replication_claim = llm_within_paper_replication_claim_mean,
    llm_contradicting_abstract = llm_contradicting_abstract_mean,
    llm_contradicting_theory  = llm_contradicting_theory_mean,
    llm_contradicting_discussion = llm_contradicting_discussion_mean,
    llm_cross_sectional       = llm_cross_sectional_mean,
    llm_longitudinal          = llm_longitudinal_mean,
    llm_student_sample        = llm_student_sample_mean,
    llm_paid_sample           = llm_paid_sample_mean,
    llm_online_paid_sample    = llm_online_paid_sample_mean,
    llm_online_study          = llm_online_study_mean,
    llm_weird_sample          = llm_weird_sample_mean,
    llm_multi_item            = llm_multi_item_mean,
    llm_posthoc_power         = llm_posthoc_power_mean,
    llm_power_section         = llm_power_section_mean,
    llm_has_robustness        = llm_has_robustness_mean,
    llm_exploratory_study     = llm_exploratory_study_mean,
    llm_open_data_request     = llm_open_data_request_mean,
    llm_open_code             = llm_open_code_mean,
    llm_conflict_of_interest  = llm_conflict_of_interest_mean,
    llm_within_paper_rep      = llm_within_paper_rep_mean,

    # Recompute composite features on aggregated values
    # (these were computed per-provider above; recomputing ensures consistency)
    llm_log_n             = log1p(llm_n_numeric),
    llm_neg_log_p         = ifelse(llm_p_numeric > 0, -log10(llm_p_numeric), NA_real_),
    llm_p_vs_alpha        = log(llm_p_numeric / llm_alpha_num),
    llm_open_science_score = rowSums(
      cbind(llm_open_data,
            as.integer(llm_open_materials_num == 1),
            llm_open_code,
            llm_preregistration),
      na.rm = !NA_counts_Zero
    ),
    llm_plausibility_prob = case_when(
      llm_plausibility_ord >= 2.5 ~ 0.85,
      llm_plausibility_ord >= 1.5 ~ 0.55,
      !is.na(llm_plausibility_ord) ~ 0.20,
      TRUE ~ NA_real_
    ),
    llm_sample_adequacy = (llm_sample_size_ord + coalesce(llm_power_any, 0)) / 6,
    llm_strength_power_ratio = llm_strength_of_evidence / (llm_power_guess * 5 + 0.01),
    llm_surprise_x_strength  = llm_perceived_surprisingness * llm_strength_of_evidence,
    llm_intuition_evidence_gap = llm_intuitive_prediction - (llm_strength_of_evidence * 2),
    llm_spec_x_theory = llm_claim_specificity * llm_theoretical_complexity,

    # Factor columns
    llm_provider  = factor(llm_provider),
    llm_model     = factor(llm_model),
    llm_reasoning = factor(llm_reasoning)
  )

# ── 5. Sanity checks ──────────────────────────────────────────────────────────

stopifnot("Duplicate claim_ids after aggregation" = !anyDuplicated(llm_features$claim_id))

# Provider coverage
provider_counts <- table(llm_features$llm_n_providers)
cat("\n── Provider coverage ────────────────────────────────────────────\n")
cat(sprintf("  Claims with 1 provider : %d\n", provider_counts["1"] %||% 0L))
cat(sprintf("  Claims with 2 providers: %d\n", provider_counts["2"] %||% 0L))
cat(sprintf("  Total claim_ids        : %d\n", nrow(llm_features)))

# Disagreement summary on key fields
disagree_cols_key <- c(
  "llm_plausibility_ord_disagree", "llm_sample_size_ord_disagree",
  "llm_effect_size_ord_disagree",  "llm_claim_specificity_disagree",
  "llm_strength_of_evidence_disagree"
)
cat("\n── Disagreement summary (median abs diff, NA = only 1 provider) ─\n")
for (col in disagree_cols_key) {
  if (col %in% names(llm_features)) {
    med <- median(llm_features[[col]], na.rm = TRUE)
    n_non_na <- sum(!is.na(llm_features[[col]]))
    cat(sprintf("  %-45s median=%s  n=%d\n",
                col,
                ifelse(is.na(med), "NA", sprintf("%.2f", med)),
                n_non_na))
  }
}

# Column count
cat(sprintf("\n── Output ───────────────────────────────────────────────────────\n"))
cat(sprintf("  Total columns in output: %d\n", ncol(llm_features)))
cat(sprintf("  New _disagree columns  : %d\n",
            sum(grepl("_disagree$", names(llm_features)))))

# Parse coverage
cat(sprintf("\n── Parse coverage ───────────────────────────────────────────────\n"))
cat(sprintf("  claim_ids with parsed n: %d / %d\n",
            sum(!is.na(llm_features$llm_n_numeric)), nrow(llm_features)))
cat(sprintf("  claim_ids with parsed p: %d / %d\n",
            sum(!is.na(llm_features$llm_p_numeric)), nrow(llm_features)))

# Ordinal NA check
ord_cols <- c("llm_plausibility_ord", "llm_sample_size_ord", "llm_effect_size_ord",
              "llm_writing_quality_ord", "llm_construct_complex_ord", "llm_robustness_ord")
for (col in ord_cols) {
  nas <- sum(is.na(llm_features[[col]]))
  if (nas > 0) cat(sprintf("  %s: %d NAs\n", col, nas))
}

# NA rate summary
llm_features |>
  summarise(across(everything(), ~mean(is.na(.)))) |>
  tidyr::pivot_longer(everything(), names_to = "col", values_to = "na_rate") |>
  filter(na_rate > 0) |>
  arrange(desc(na_rate)) |>
  print(n = 50)

# ── 6. Save ────────────────────────────────────────────────────────────────────

dir.create(dirname(OUT_PATH), showWarnings = FALSE, recursive = TRUE)
saveRDS(llm_features, OUT_PATH)
cat(sprintf("\nSaved: %s  (%d rows × %d cols)\n", OUT_PATH, nrow(llm_features), ncol(llm_features)))
