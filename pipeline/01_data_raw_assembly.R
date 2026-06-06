# ==============================================================================
# 01_data_raw_assembly.R
#
# PURPOSE
#   Merge all data rounds (FORRT training, Rounds 1/2, flora health add-on),
#   join GROBID TEI/XML features, apply the Round 3 missing-data patch, and
#   flag/exclude Boyce/Soto studies. Run once (or when raw data changes).
#
# INPUTS  : data/FORRT_Training_Data.csv, data/Round1/*, data/Round2/*,
#           data/Round3/*, data/flora_health_addon.csv, data/extracted_tei_all.rds
# OUTPUTS : data/train_raw.rds, data/test_raw.rds
# ==============================================================================

source(here::here("pipeline/00_packages_config.R"))
library(dplyr); library(readr); library(stringr); library(tidyr); library(here)

NA_STRINGS <- c("", "NA", "N/A", "null")

# ── 1. Load raw data ──────────────────────────────────────────────────────────
dat  <- read_csv(here::here("data/FORRT_Training_Data.csv"),
                 show_col_types = FALSE, na = NA_STRINGS)

dat_cos_phase1 <- dat %>%
  filter(cos_phase1==1) %>%
  mutate(n_o = as.numeric(str_extract(n_o, "^[0-9]+")),
         claim_id_forrt = paste0(entry_id, "_", effect_id))

dat <- dat %>%
        filter(cos_phase1==0) %>%
        mutate(n_o = as.numeric(str_extract(n_o, "^[0-9]+")))

dat1 <- read_csv(here::here("data/Round1/test-set_for-participants_1.csv"),
                 show_col_types = FALSE, na = NA_STRINGS)
dat1_projectlink <- read_csv(here::here("data/Round1/test-set_project-links_1.csv"),
                             show_col_types = FALSE, na = NA_STRINGS)
dat1_truth <- read_csv(here::here("data/Round1/ground_truth_1.csv"),
                       show_col_types = FALSE, na = NA_STRINGS)
dat2 <- read_csv(here::here("data/Round2/test-set_for-participants_2.csv"),
                 show_col_types = FALSE, na = NA_STRINGS)
dat2_truth <- read_csv(here::here("data/Round2/ground_truth_2.csv"),
                       show_col_types = FALSE, na = NA_STRINGS)
dat2_projectlink <- read_csv(here::here("data/Round2/test-set_project-links_2.csv"),
                             show_col_types = FALSE, na = NA_STRINGS)

# ── 3. Prepare training data and Round 1 ─────────────────────────────────────
claim_id_bridge <- dat_cos_phase1 |>
  left_join(dat1 |> select(claim_text_o, claim_id_hash = claim_id),
            by = "claim_text_o") |>
  select(claim_id_hash, claim_id_forrt) |>
  filter(!is.na(claim_id_hash)) |>
  distinct(claim_id_hash, .keep_all = TRUE)

dat <- dat |>
  mutate(entry_id = as.character(entry_id),
         claim_id = paste0(entry_id, "_", effect_id),
         dataset  = "training") |>
  mutate(es_type_o = case_when(
    is.na(es_type_o) & str_detect(as.character(es_value_o), "^t\\(") ~ "test statistic",
    .default = es_type_o
  )) |>
  mutate(es_value_o = as.character(es_value_o) |>
           sub(".*[=<>]", "", x = _) |>
           trimws() |>
           parse_number())

# ── flora health-domain add-on ───────────────────────────────────────────────
# 17 large-N health replications hand-coded from flora + the replication PDFs
# (statistical_success derived from each replication's focal p-value + effect
# direction). FORRT-format CSV; treated as `training` data. Extra context columns
# (journal_o/year_o/title_o/repl_is_new_data) are dropped so they don't collide
# with the GROBID/XML join downstream.
dat_flora <- read_csv(here::here("data/flora_health_addon.csv"),
                      show_col_types = FALSE, na = NA_STRINGS) |>
  select(-any_of(c("journal_o", "year_o", "title_o", "repl_is_new_data"))) |>
  filter(cos_phase1 == 0) |>
  mutate(n_o = as.numeric(str_extract(as.character(n_o), "^[0-9]+")),
         claim_id = paste0(entry_id, "_", effect_id),
         dataset  = "training") |>
  mutate(es_value_o = as.character(es_value_o) |>
           sub(".*[=<>]", "", x = _) |>
           trimws() |>
           parse_number())
cat(sprintf("flora health add-on: %d rows (success=%d, fail=%d)\n",
            nrow(dat_flora), sum(dat_flora$statistical_success, na.rm = TRUE),
            sum(dat_flora$statistical_success == FALSE, na.rm = TRUE)))
# Align column types to the FORRT `dat` frame so bind_rows() doesn't choke
# (FORRT stores some numeric-looking fields, e.g. es_value_r, as character).
for (cc in intersect(names(dat), names(dat_flora))) {
  if (is.character(dat[[cc]]) && !is.character(dat_flora[[cc]])) {
    dat_flora[[cc]] <- as.character(dat_flora[[cc]])
  } else if (is.numeric(dat[[cc]]) && !is.numeric(dat_flora[[cc]])) {
    dat_flora[[cc]] <- suppressWarnings(as.numeric(dat_flora[[cc]]))
  } else if (is.logical(dat[[cc]]) && !is.logical(dat_flora[[cc]])) {
    dat_flora[[cc]] <- as.logical(dat_flora[[cc]])
  }
}

# dat1: attach claim_id_forrt via the hash-based bridge
dat1 <- dat1 |>
  left_join(dat1_projectlink |> select(claim_id, osf_project_link),             by = "claim_id") |>
  left_join(dat1_truth       |> select(claim_id, statistical_success = outcome), by = "claim_id") |>
  mutate(dataset = "round1") |>
  mutate(es_value_o = as.character(es_value_o) |>
           sub(".*[=<>]", "", x = _) |>
           trimws() |>
           parse_number()) |>
  left_join(claim_id_bridge, by = c("claim_id" = "claim_id_hash"))

# ── 4. Round 2 ───────────────────────────────────────────────────────────────
dat2 <- dat2 |>
  left_join(dat2_truth       |> select(claim_id, statistical_success = outcome), by = "claim_id") |>
  left_join(dat2_projectlink |> select(claim_id, osf_project_link),  by = "claim_id") |>
  mutate(dataset = "round2")|> mutate(es_value_o = as.character(es_value_o) |>
                                        sub(".*[=<>]", "", x = _) |>
                                        trimws() |>
                                        parse_number())

# ── 5. Combine all sources ────────────────────────────────────────────────────
dat_gesamt <- bind_rows(dat, dat_flora, dat1, dat2)

cat("Rows per dataset:\n")
print(table(dat_gesamt$dataset, useNA = "ifany"))

# ── 6. Join XML features (GROBID TEI extraction) ─────────────────────────────

# Normalized join key: lowercase + alphanumerics only. The old exact-string
# match on filename silently dropped ~400 rows whose file_o differed only in
# case (BF vs bf), whitespace ("10.1 037-- .pdf"), or sanitized special chars
# (GROBID maps "é"/":"/"<" to "_"), even though the TEI existed on disk. Both
# sides collapse to the same key, so those rows now join.
norm_join_key <- function(x) tolower(gsub("[^[:alnum:]]+", "", x))

all_results_clean <- readRDS(here::here("data/extracted_tei_all.rds")) |>
  mutate(join_key = norm_join_key(str_remove(filename_xml, "\\.tei\\.xml$"))) |>
  select(-any_of(c("data_type", "grobid_date_xml"))) |>
  # one TEI per normalized key — keep the longest-text parse when variants collide
  group_by(join_key) |>
  slice_max(order_by = nchar(dplyr::coalesce(article_text_xml, "")),
            n = 1, with_ties = FALSE) |>
  ungroup()

# Count fields where 0 is impossible for a real paper (authors / affiliations /
# sections / references) are GROBID parse failures, not genuine zeros. Legacy
# keeps them 0; NA_counts_Zero = TRUE recodes 0 -> NA. Figures/tables/formulas
# are NOT touched (0 is a valid value there).
if (NA_counts_Zero) {
  all_results_clean <- all_results_clean |>
    mutate(across(
      any_of(c("n_authors_xml", "n_affiliations_xml",
               "n_sections_xml", "n_references_xml")),
      ~ dplyr::na_if(.x, 0L)
    ))
}

dat_gesamt <- dat_gesamt |>
  mutate(join_key = norm_join_key(str_remove(file_o, "\\.pdf$"))) |>
  left_join(all_results_clean, by = "join_key") |>
  mutate(doi_o = normalize_doi(doi_o))

cat(sprintf("After XML join: %d rows x %d cols\n", nrow(dat_gesamt), ncol(dat_gesamt)))

# ── 7. Boyce/Soto flag ────────────────────────────────────────────────────────
# Flagged rows are excluded

boyce_soto_rx <- regex("Boyce|Soto", ignore_case = TRUE)

is_boyce_soto <- function(df) {
  (!is.na(df$fred_id)     & str_detect(df$fred_id,     boyce_soto_rx)) |
    (!is.na(df$fred_id_old) & str_detect(df$fred_id_old, boyce_soto_rx))
}

n_excluded <- sum(is_boyce_soto(dat_gesamt), na.rm = TRUE)
cat(sprintf("Boyce/Soto excluded: %d rows removed from dat_gesamt\n", n_excluded))

dat_gesamt <- dat_gesamt[!is_boyce_soto(dat_gesamt), ]

# ── 8. Quality checks ────────────────────────────────────────────────────────

# Most training rows have no claim_id (FORRT data has none natively;
# only rows matched to Round1 via claim_text_o get one). This is expected.

n_dup <- sum(duplicated(dat_gesamt$claim_id, incomparables = NA))
if (n_dup > 0)
  warning(sprintf("%d duplicate claim_ids – check bind_rows.", n_dup))

cat(sprintf("Final training set: %d rows | %d with known statistical_success\n",
            nrow(dat_gesamt), sum(!is.na(dat_gesamt$statistical_success))))

# ── 9. Round3 test set ────────────────────────────────────────────────────────

test_data_incl_xml <- read_csv(
  here::here("data/Round3/test-set_for-participants_3.csv"),
  show_col_types = FALSE, na = NA_STRINGS
) |>
  mutate(join_key = norm_join_key(str_remove(file_o, "\\.pdf$"))) |>
  left_join(all_results_clean, by = "join_key") |>
  mutate(doi_o = normalize_doi(doi_o))

cat(sprintf("Test set (Round 3): %d rows x %d cols\n",
            nrow(test_data_incl_xml), ncol(test_data_incl_xml)))

# ── 9b. Fill manually-extracted missing test fields ──────────────────────────
# The Round3 CSV ships with n_o / pval_value_o / es_value_o / es_type_o missing
# for a number of health claims. These were hand-verified from the GROBID TEI /
# claim text / PDFs (test_set_missing_data_extraction.xlsx). The resolved patch
# lives in data/Round3/test_set_missing_data_patch.csv with one row per
# (claim_id, field):
#   value        — final numeric (the manually corrected value has priority over
#                  the LLM extracted_value; "<0.05" → 0.05, conservative).
#   es_type_new  — es_type_o to assign (es_value_o rows only).
# Rows marked NOT REPORTED are genuine gaps and are NOT in the patch, so
# they stay NA. Effect-size types here ("b (unstd)", "beta (std)",
# "mean difference", "test statistic", "Cohen's d") are all recognised by
# harmonize_effect_sizes() in 04_base_dataset.R; where r_o is already present
# r_harmonized keeps using r_o (r_native), so these fills mainly complete n_o,
# the p-value, and the effect-size/missingness features.
test_patch <- read_csv(
  here::here("data/Round3/test_set_missing_data_patch.csv"),
  show_col_types = FALSE, na = NA_STRINGS
)

apply_test_patch <- function(df, patch) {
  for (i in seq_len(nrow(patch))) {
    cid <- patch$claim_id[i]; fld <- patch$field[i]
    row <- which(df$claim_id == cid)
    if (length(row) != 1L) {
      warning(sprintf("patch: claim_id %s matched %d rows for %s", cid, length(row), fld))
      next
    }
    df[[fld]][row] <- patch$value[i]
    if (fld == "es_value_o" && !is.na(patch$es_type_new[i])) {
      df[["es_type_o"]][row] <- patch$es_type_new[i]
    }
  }
  df
}

n_miss_before <- sapply(c("n_o","pval_value_o","es_value_o"),
                        function(c) sum(is.na(test_data_incl_xml[[c]])))
test_data_incl_xml <- apply_test_patch(test_data_incl_xml, test_patch)
n_miss_after  <- sapply(c("n_o","pval_value_o","es_value_o"),
                        function(c) sum(is.na(test_data_incl_xml[[c]])))
cat("Test missing-data patch applied (", nrow(test_patch), "fills):\n", sep = "")
print(data.frame(field = names(n_miss_before),
                 NA_before = n_miss_before, NA_after = n_miss_after,
                 row.names = NULL))

# ── 10. Save ──────────────────────────────────────────────────────────────────

saveRDS(dat_gesamt,         here::here("data/train_raw.rds"), compress = "xz")
saveRDS(test_data_incl_xml, here::here("data/test_raw.rds"),  compress = "xz")

cat("Saved: data/train_raw.rds | data/test_raw.rds\n")
