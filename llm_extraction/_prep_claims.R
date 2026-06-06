#!/usr/bin/env Rscript
# Aggregate the four canonical claim sources into one CSV ready for
# _build_manifest.py:
#
#   data/Round1/test-set_for-participants_1.csv          -> dataset=round1
#   data/Round2/test-set_for-participants_2.csv          -> dataset=round2
#   data/Round3/test-set_for-participants_3.csv          -> dataset=round3
#   data/FORRT_Training_Data.csv (filtered+random pick)  -> dataset=forrt
#
# Output: data/all_claims.csv with columns
#   claim_id, doi_o, claim_text_o, file_o, dataset, primary
#
# FORRT filters (decided 2026-05-24):
#   1. file_o non-empty
#   2. claim_status_llm in {'central', 'mentioned'}
#   3. ref_r first author not in {Soto, Boyce}
#   4. one row per file_o, random pick (set.seed(42))
# FORRT claim_id is synthesised as paste0(entry_id, '_', effect_id).
#
# `primary` selection (per paper, decided 2026-05-24):
#   For each file_o, mark exactly one row as primary=TRUE — that's the
#   first claim we extract per paper. Subsequent claims for the same
#   paper stay in the manifest with primary=FALSE so we can decide later
#   whether to extract them too.
#     - if any FORRT row exists for this paper, use it (FORRT was already
#       random-picked from claim_status_llm in {central, mentioned}).
#     - else pick a random row from the surviving rounds rows.

# Each random operation re-seeds with the same seed so adding/removing
# upstream sample() calls doesn't shift downstream picks. With a single
# top-level set.seed, the RNG state would carry between blocks and break
# reproducibility across script revisions.

script_arg <- sub("^--file=", "",
                  grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
HERE <- if (length(script_arg) == 1L) normalizePath(dirname(script_arg)) else getwd()
setwd(HERE)

read_round <- function(path, label) {
  d <- read.csv(path, stringsAsFactors = FALSE, na.strings = "")
  data.frame(
    claim_id     = d$claim_id,
    doi_o        = d$doi_o,
    claim_text_o = d$claim_text_o,
    file_o       = d$file_o,
    dataset      = label,
    stringsAsFactors = FALSE
  )
}

r1 <- read_round("../data/Round1/test-set_for-participants_1.csv", "round1")
r2 <- read_round("../data/Round2/test-set_for-participants_2.csv", "round2")
r3 <- read_round("../data/Round3/test-set_for-participants_3.csv", "round3")

# FORRT — full filter pipeline
forrt <- read.csv("../data/FORRT_Training_Data.csv",
                  stringsAsFactors = FALSE, na.strings = "")
n0 <- nrow(forrt)
forrt <- forrt[!is.na(forrt$file_o) & nzchar(forrt$file_o), ]
n1 <- nrow(forrt)
forrt <- forrt[!is.na(forrt$claim_status_llm) &
               forrt$claim_status_llm %in% c("central", "mentioned"), ]
n2 <- nrow(forrt)
first_author_lc <- function(ref) {
  ifelse(is.na(ref) | !nzchar(ref), "", tolower(trimws(sub(",.*", "", ref))))
}
forrt <- forrt[!(first_author_lc(forrt$ref_r) %in% c("soto", "boyce")), ]
n3 <- nrow(forrt)
groups <- split(seq_len(nrow(forrt)), tolower(forrt$file_o))
set.seed(42)  # FORRT main-pass random pick
picked <- vapply(groups, function(idx) {
  if (length(idx) == 1L) idx else sample(idx, 1L)
}, integer(1L))
forrt <- forrt[sort(picked), ]
n4 <- nrow(forrt)
f <- data.frame(
  claim_id     = paste0(forrt$entry_id, "_", forrt$effect_id),
  doi_o        = forrt$doi_o,
  claim_text_o = forrt$claim_text_o,
  file_o       = forrt$file_o,
  dataset      = "forrt",
  stringsAsFactors = FALSE
)

# FORRT supplementary pass (decided 2026-05-27):
#   Papers whose only viable claims are 'unsure' were dropped by the main
#   central/mentioned filter, so we have no extraction for them at all.
#   Pick one 'unsure' claim per such paper (random, seed 42 carried over)
#   so each paper has at least one coded claim.
forrt_all <- read.csv("../data/FORRT_Training_Data.csv",
                      stringsAsFactors = FALSE, na.strings = "")
forrt_supp <- forrt_all[!is.na(forrt_all$file_o) & nzchar(forrt_all$file_o), ]
forrt_supp <- forrt_supp[!(first_author_lc(forrt_supp$ref_r) %in% c("soto", "boyce")), ]
covered_files <- tolower(unique(f$file_o))
forrt_supp <- forrt_supp[!(tolower(forrt_supp$file_o) %in% covered_files), ]
supp_groups <- split(seq_len(nrow(forrt_supp)), tolower(forrt_supp$file_o))
set.seed(42)  # FORRT supplement random pick
supp_pick <- integer(0)
for (key in names(supp_groups)) {
  idx <- supp_groups[[key]]
  unsure_idx <- idx[!is.na(forrt_supp$claim_status_llm[idx]) &
                     forrt_supp$claim_status_llm[idx] == "unsure"]
  if (length(unsure_idx) == 0L) next
  supp_pick <- c(supp_pick,
                 if (length(unsure_idx) == 1L) unsure_idx else sample(unsure_idx, 1L))
}
forrt_supp <- forrt_supp[sort(supp_pick), ]
n_supp <- nrow(forrt_supp)
f_supp <- data.frame(
  claim_id     = paste0(forrt_supp$entry_id, "_", forrt_supp$effect_id),
  doi_o        = forrt_supp$doi_o,
  claim_text_o = forrt_supp$claim_text_o,
  file_o       = forrt_supp$file_o,
  dataset      = "forrt",
  stringsAsFactors = FALSE
)

all <- rbind(r1, r2, r3, f, f_supp)

# Per-paper primary selection (group by case-insensitive file_o to match
# the case-insensitive fuzzy resolution in _build_manifest.py).
all$primary <- FALSE
file_key <- tolower(all$file_o)
set.seed(42)  # per-paper primary random pick (only used when no FORRT row)
for (fid in unique(file_key)) {
  idx <- which(file_key == fid)
  if (length(idx) == 1L) {
    all$primary[idx] <- TRUE
  } else {
    forrt_idx <- idx[all$dataset[idx] == "forrt"]
    if (length(forrt_idx) > 0L) {
      all$primary[forrt_idx[1L]] <- TRUE
    } else {
      all$primary[sample(idx, 1L)] <- TRUE
    }
  }
}

write.csv(all, "../data/all_claims.csv", row.names = FALSE, quote = TRUE)

cat(sprintf("rounds R1 / R2 / R3 : %4d / %4d / %4d rows\n",
            nrow(r1), nrow(r2), nrow(r3)))
cat(sprintf("FORRT pipeline      : %d -> %d (file_o) -> %d (status) -> %d (auth) -> %d (random pick)\n",
            n0, n1, n2, n3, n4))
cat(sprintf("FORRT supplement    : %d unsure-only papers added (one row each)\n", n_supp))
cat(sprintf("aggregated rows     : %d\n", nrow(all)))
cat(sprintf("distinct papers     : %d\n", length(unique(all$file_o))))
cat(sprintf("primary rows        : %d (one per paper)\n", sum(all$primary)))
cat(sprintf("primary by dataset  : R1=%d  R2=%d  R3=%d  FORRT=%d\n",
            sum(all$primary & all$dataset == "round1"),
            sum(all$primary & all$dataset == "round2"),
            sum(all$primary & all$dataset == "round3"),
            sum(all$primary & all$dataset == "forrt")))
cat(sprintf("wrote               -> %s\n", normalizePath("../data/all_claims.csv")))
