# ==============================================================================
# 02_external_features.R
# Fetch and assemble external features for all papers (train + test).
# Run once (or when new papers are added). All API calls are cached.
#
# Manual downloads required before first run:
#   TOP Factor : https://cos.io/top                        → data/external/topfactor.csv
#   SJR/Scopus : https://www.scimagojr.com/journalrank.php → data/external/sjr.csv
#
# Produces: data/external_features.rds (one row per doi_o)
# ==============================================================================

source(here::here("pipeline/00_packages_config.R"))
library(openalexR); suppressWarnings(tryCatch(library(FReD), error = function(e) message("FReD not available – skipping."))); library(dplyr); library(purrr)
library(readr); library(stringr); library(tibble); library(wbstats)
library(countrycode)

# OpenAlex asks for a contact email to use its faster "polite pool". Set the
# OPENALEX_MAILTO environment variable to your own address (e.g. in ~/.Renviron);
# without it, requests fall back to the slower anonymous pool.
options(openalexR.mailto = Sys.getenv("OPENALEX_MAILTO", unset = ""))

SKIP_SELF_REFS <- FALSE

dir.create(here::here("data/cache"),    showWarnings = FALSE, recursive = TRUE)
dir.create(here::here("data/external"), showWarnings = FALSE, recursive = TRUE)

# Fallback in case script is run standalone without 00_packages_config
if (!exists("normalize_doi")) {
  normalize_doi <- function(x) {
    x |> str_remove("^https?://doi\\.org/") |> str_remove("^doi:") |>
      str_to_lower() |> str_trim()
  }
}

# Helper: resolve issn_l.x / issn_l.y collisions after left_join
coalesce_issn <- function(df) {
  if ("issn_l.x" %in% names(df) && "issn_l.y" %in% names(df))
    df <- df |> mutate(issn_l = coalesce(issn_l.x, issn_l.y)) |>
      select(-issn_l.x, -issn_l.y)
  df
}

# Load base datasets to collect all DOIs
train_raw <- readRDS(here::here("data/train_raw.rds"))
test_raw  <- readRDS(here::here("data/test_raw.rds"))

all_dois <- unique(c(train_raw$doi_o, test_raw$doi_o))
all_dois <- all_dois[!is.na(all_dois) & nchar(all_dois) > 5]
cat(sprintf("Unique DOIs: %d\n", length(all_dois)))

# ── 1. OpenAlex: works-level ──────────────────────────────────────────────────

cache_works <- here::here("data/cache/oa_works.rds")

if (file.exists(cache_works)) {
  works_raw <- readRDS(cache_works)
  # Incremental: fetch only DOIs not already cached (preserves existing values).
  have_dois    <- tolower(normalize_doi(works_raw$doi))
  missing_dois <- all_dois[!tolower(all_dois) %in% have_dois]
  if (length(missing_dois) > 0) {
    cat(sprintf("OpenAlex works: fetching %d new DOIs...\n", length(missing_dois)))
    new_works <- tryCatch(
      oa_fetch(entity = "works", doi = missing_dois, verbose = TRUE),
      error = function(e) { message("Works fetch failed: ", e$message); NULL })
    if (!is.null(new_works) && nrow(new_works) > 0) {
      works_raw <- bind_rows(works_raw, new_works)
      saveRDS(works_raw, cache_works)
    }
  }
} else {
  works_raw <- tryCatch(
    oa_fetch(entity = "works", doi = all_dois, verbose = TRUE),
    error = function(e) { message("Works fetch failed: ", e$message); NULL }
  )
  if (!is.null(works_raw)) saveRDS(works_raw, cache_works)
}

works_features <- NULL

if (!is.null(works_raw) && nrow(works_raw) > 0) {
  works_features <- works_raw |>
    mutate(doi_o = normalize_doi(doi)) |>
    mutate(
      year_o                  = as.integer(publication_year),
      number_citations        = as.integer(cited_by_count),
      n_authors_oa            = map_int(authorships, nrow),
      number_references       = as.integer(referenced_works_count),
      oa_status               = as.character(oa_status),
      journal_open_access_bin = as.integer(oa_status %in% c("gold", "diamond", "green")),
      source_id               = as.character(source_id),
      issn_l                  = as.character(issn_l),
      fwci_via_oa             = as.numeric(fwci),
      is_retracted            = is_retracted,
      topics_via_oa = map_chr(topics, ~ {
        if (is.null(.x) || length(.x) == 0) return(NA_character_)
        df <- as.data.frame(.x); df$display_name[df$type == "topic"][1]
      }),
      keywords_via_oa = map_chr(keywords, ~ {
        if (is.null(.x) || length(.x) == 0) return(NA_character_)
        paste(as.data.frame(.x)$display_name, collapse = ", ")
      }),
      funders_via_oa = map_chr(funders, ~ {
        if (is.null(.x) || length(.x) == 0) return(NA_character_)
        paste(as.data.frame(.x)$display_name, collapse = ", ")
      }),
      apc_via_oa = map_chr(apc, ~ {
        if (is.null(.x) || length(.x) == 0) return(NA_character_)
        df <- as.data.frame(.x)
        if (!"value" %in% names(df) || is.na(df$value[1])) return(NA_character_)
        paste(df$value[1], df$currency[1])
      }),
      # ── New fields from counts_by_year, concepts, and referenced_works ────────
      # citation_count_t_plus_2y: leakage-safe citation proxy summing citations
      #   only from years at or before publication_year+2, so the value is fixed
      #   regardless of when the pipeline runs. Better quality signal than total.
      # concepts_top3: top-3 OpenAlex concept names by score, semi-colon joined.
      #   Useful as a text feature or for topic-cluster dummies.
      # concepts_top3_levels: concept hierarchy depth (0=field, 5=specific) for
      #   the same top-3 concepts. Level 0-1 = broad, 4-5 = narrow/specific.
      # n_referenced_works: direct count of items in referenced_works list-col.
      #   May differ from OpenAlex's referenced_works_count (API field) and is
      #   computed locally from the cached payload.
      citation_count_t_plus_2y = purrr::map2_dbl(
        counts_by_year, publication_year,
        ~ if (is.null(.x) || !is.data.frame(.x) || is.na(.y)) NA_real_
          else sum(.x$cited_by_count[.x$year <= .y + 2], na.rm = TRUE)
      ),
      concepts_top3 = map_chr(concepts, ~ {
        if (is.null(.x) || nrow(as.data.frame(.x)) == 0) return(NA_character_)
        df <- as.data.frame(.x)
        df <- df[order(-df$score), ]
        paste(head(df$display_name, 3), collapse = "; ")
      }),
      concepts_top3_levels = map_chr(concepts, ~ {
        if (is.null(.x) || nrow(as.data.frame(.x)) == 0) return(NA_character_)
        df <- as.data.frame(.x)
        df <- df[order(-df$score), ]
        paste(head(df$level, 3), collapse = ", ")
      }),
      n_referenced_works = purrr::map_int(referenced_works, ~ length(.x %||% character(0)))
    ) |>
    select(doi_o, year_o, number_citations, n_authors_oa, number_references,
           oa_status, journal_open_access_bin, source_id, issn_l,
           topics_via_oa, keywords_via_oa, funders_via_oa, apc_via_oa,
           is_retracted, fwci_via_oa,
           citation_count_t_plus_2y, concepts_top3, concepts_top3_levels,
           n_referenced_works)

  cat(sprintf("Works features: %d rows\n", nrow(works_features)))
}

# ── 2. OpenAlex: journal/source-level ────────────────────────────────────────

cache_sources <- here::here("data/cache/oa_sources.rds")
source_ids    <- if (!is.null(works_features)) unique(na.omit(works_features$source_id)) else character(0)
sources_raw   <- NULL

if (length(source_ids) == 0) {
  cat("No source IDs – skipping journal metrics.\n")
} else if (file.exists(cache_sources)) {
  sources_raw <- readRDS(cache_sources)
  missing_src <- setdiff(source_ids, sources_raw$id)
  if (length(missing_src) > 0) {
    cat(sprintf("OpenAlex sources: fetching %d new sources...\n", length(missing_src)))
    new_src <- tryCatch(
      oa_fetch(entity = "sources", openalex_id = missing_src, verbose = TRUE),
      error = function(e) { message("Sources fetch failed: ", e$message); NULL })
    if (!is.null(new_src) && nrow(new_src) > 0) {
      sources_raw <- bind_rows(sources_raw, new_src)
      saveRDS(sources_raw, cache_sources)
    }
  }
} else {
  sources_raw <- tryCatch(
    oa_fetch(entity = "sources", openalex_id = source_ids, verbose = TRUE),
    error = function(e) { message("Sources fetch failed: ", e$message); NULL }
  )
  if (!is.null(sources_raw)) saveRDS(sources_raw, cache_sources)
}

sources_features <- NULL

if (!is.null(sources_raw) && nrow(sources_raw) > 0) {
  sources_features <- sources_raw |>
    rename_with(~ str_replace(.x, "^2yr", "x2yr")) |>
    hoist(summary_stats,
          h_index      = "h_index",
          i10_index    = "i10_index",
          two_yr_cited = "2yr_mean_citedness") |>
    transmute(
      source_id             = id,
      journal_hindex        = as.integer(h_index),
      journal_i10index      = as.integer(i10_index),
      journal_2yr_citedness = as.numeric(two_yr_cited),
      journal_total_works   = as.integer(works_count),
      journal_issn_l        = issn_l,
      apc_usd               = as.numeric(apc_usd),
      journal_name_oa       = as.character(display_name)
    ) |>
    distinct(source_id, .keep_all = TRUE)

  cat(sprintf("Sources features: %d journals\n", nrow(sources_features)))
}

# ── 3. OpenAlex: author-level (first + last author) ──────────────────────────

extract_from_tibble <- function(authorships_list, target_position, field) {
  map_chr(authorships_list, ~ {
    if (is.null(.x) || nrow(.x) == 0) return(NA_character_)
    row <- .x |> filter(author_position == target_position)
    if (nrow(row) == 0) return(NA_character_)
    if (field == "id") return(row$id[1] %||% NA_character_)
    if (field == "country") {
      aff <- row$affiliations[[1]]
      if (!is.null(aff) && "country_code" %in% names(aff) && nrow(aff) > 0)
        return(aff$country_code[1] %||% NA_character_)
    }
    NA_character_
  })
}

# Count distinct author countries / institutions across the full author team.
# Reuses the proven affiliations access pattern (aff$country_code).
extract_team_diversity <- function(authorships_list, what = c("country", "institution")) {
  what <- match.arg(what)
  map_int(authorships_list, ~ {
    if (is.null(.x) || nrow(.x) == 0 || !"affiliations" %in% names(.x))
      return(NA_integer_)
    vals <- unlist(lapply(.x$affiliations, function(aff) {
      if (is.null(aff) || !is.data.frame(aff) || nrow(aff) == 0) return(character(0))
      if (what == "country") {
        if ("country_code" %in% names(aff)) as.character(aff$country_code) else character(0)
      } else {
        # institution OpenAlex id
        if ("id" %in% names(aff))                as.character(aff$id)
        else if ("institution_id" %in% names(aff)) as.character(aff$institution_id)
        else character(0)
      }
    }))
    vals <- vals[!is.na(vals) & nzchar(vals)]
    if (length(vals) == 0) return(NA_integer_)
    length(unique(vals))
  })
}

author_id_lookup <- NULL

if (!is.null(works_raw) && nrow(works_raw) > 0) {
  author_id_lookup <- works_raw |>
    mutate(
      doi_o                = normalize_doi(doi),
      first_author_id      = extract_from_tibble(authorships, "first",  "id"),
      last_author_id       = extract_from_tibble(authorships, "last",   "id"),
      first_author_country = extract_from_tibble(authorships, "first",  "country"),
      last_author_country  = extract_from_tibble(authorships, "last",   "country"),
      n_author_countries    = extract_team_diversity(authorships, "country"),
      n_author_institutions = extract_team_diversity(authorships, "institution")
    ) |>
    select(doi_o, first_author_id, last_author_id,
           first_author_country, last_author_country,
           n_author_countries, n_author_institutions) |>
    distinct(doi_o, .keep_all = TRUE)
}

# Per-work full author-ID list (intermediate; drives team aggregates below).
# Kept separate from author_id_lookup so the list-column never propagates into
# external_features / the model matrix.
team_author_map <- NULL
if (!is.null(works_raw) && nrow(works_raw) > 0) {
  team_author_map <- works_raw |>
    mutate(
      doi_o          = normalize_doi(doi),
      all_author_ids = map(authorships, ~ {
        if (is.null(.x) || nrow(.x) == 0 || !"id" %in% names(.x)) return(character(0))
        ids <- as.character(.x$id)
        ids[!is.na(ids) & nzchar(ids)]
      })
    ) |>
    select(doi_o, all_author_ids) |>
    distinct(doi_o, .keep_all = TRUE)
}

unique_author_ids <- if (!is.null(author_id_lookup))
  unique(na.omit(c(author_id_lookup$first_author_id,
                   author_id_lookup$last_author_id,
                   if (!is.null(team_author_map))
                     unlist(team_author_map$all_author_ids) else character(0)))) else character(0)

cat(sprintf("Unique author IDs: %d\n", length(unique_author_ids)))

cache_authors <- here::here("data/cache/oa_authors.rds")
authors_raw   <- NULL

if (length(unique_author_ids) == 0) {
  cat("No author IDs – skipping author metrics.\n")
} else if (file.exists(cache_authors)) {
  authors_raw <- readRDS(cache_authors)
  missing_auth <- setdiff(unique_author_ids, authors_raw$id)
  if (length(missing_auth) > 0) {
    cat(sprintf("OpenAlex authors: fetching %d new authors...\n", length(missing_auth)))
    new_auth <- tryCatch(
      oa_fetch(entity = "authors", openalex_id = missing_auth, verbose = TRUE),
      error = function(e) { message("Authors fetch failed: ", e$message); NULL })
    if (!is.null(new_auth) && nrow(new_auth) > 0) {
      authors_raw <- bind_rows(authors_raw, new_auth)
      saveRDS(authors_raw, cache_authors)
    }
  }
} else {
  authors_raw <- tryCatch(
    oa_fetch(entity = "authors", openalex_id = unique_author_ids, verbose = TRUE),
    error = function(e) { message("Authors fetch failed: ", e$message); NULL }
  )
  if (!is.null(authors_raw)) saveRDS(authors_raw, cache_authors)
}

authors_features <- NULL

if (!is.null(authors_raw) && nrow(authors_raw) > 0) {
  authors_features <- authors_raw |>
    transmute(
      author_id                  = id,
      author_display_name        = display_name,
      author_name_alternatives   = display_name_alternatives,
      author_works_count         = as.integer(works_count),
      author_citations           = as.integer(cited_by_count),
      author_hindex              = as.numeric(h_index),
      author_i10_index           = as.numeric(i10_index),
      author_experience_since    = map_dbl(counts_by_year, ~ {
        if (is.null(.x) || !is.data.frame(.x) || nrow(.x) == 0) return(NA_real_)
        as.numeric(.x$year[1])
      }),
      author_institution_country = map_chr(last_known_institutions, ~ {
        if (is.null(.x) || !is.data.frame(.x) || nrow(.x) == 0) return(NA_character_)
        valid <- na.omit(.x$country_code)
        if (length(valid) > 0) valid[1] else NA_character_
      })
    ) |>
    distinct(author_id, .keep_all = TRUE)

  cat(sprintf("Author features: %d authors\n", nrow(authors_features)))
}

# ──  Team-level productivity (median of the top-25% most productive authors) ──
# top25_median(v): median over the ceiling(n/4) authors with the highest values
# (so for teams of 1-3 it equals the single most productive author).
team_features <- NULL

if (!is.null(authors_features) && !is.null(team_author_map)) {
  TEAM_REF_YEAR <- 2026L  # reference "current" year for the productivity rate

  author_metrics <- authors_features |>
    transmute(
      author_id,
      a_works = author_works_count,                       # cumulative; reliable
      a_prod  = author_works_count /                      # rate
        pmax(TEAM_REF_YEAR - author_experience_since + 1, 1)
    )

  top25_median <- function(v) {
    v <- v[!is.na(v)]
    if (length(v) == 0) return(NA_real_)
    k <- max(1L, ceiling(length(v) * 0.25))
    median(sort(v, decreasing = TRUE)[seq_len(k)])
  }

  team_features <- team_author_map |>
    tidyr::unnest_longer(all_author_ids, values_to = "author_id") |>
    left_join(author_metrics, by = "author_id") |>
    group_by(doi_o) |>
    summarise(
      team_size_resolved             = sum(!is.na(a_works)),
      team_works_top25_median        = top25_median(a_works),
      team_productivity_top25_median = top25_median(a_prod),
      .groups = "drop"
    )

  cat(sprintf("Team features: %d papers\n", nrow(team_features)))
}

# ── 3b. World Bank GDP per capita ─────────────────────────────────────────────

cache_gdp <- here::here("data/cache/wb_gdp.rds")

if (file.exists(cache_gdp)) {
  gdp_data <- readRDS(cache_gdp)
} else {
  gdp_raw  <- wbstats::wb_data(indicator = "NY.GDP.PCAP.CD",
                                start_date = 2019, end_date = 2023)
  gdp_data <- gdp_raw |>
    group_by(iso3c) |>
    summarise(gdp_per_capita = median(NY.GDP.PCAP.CD, na.rm = TRUE), .groups = "drop") |>
    filter(!is.na(gdp_per_capita)) |>
    mutate(country_iso2 = countrycode::countrycode(
      iso3c, "iso3c", "iso2c",
      custom_match = c(CHI = NA_character_, XKX = NA_character_)
    )) |>
    filter(!is.na(country_iso2)) |>
    select(country_iso2, gdp_per_capita)
  saveRDS(gdp_data, cache_gdp)
}

cat(sprintf("GDP data: %d countries\n", nrow(gdp_data)))

# ── 4. Self-references (slow – skippable) ────────────────────────────────────

self_ref_features <- NULL

if (file.exists(here::here("data/cache/self_ref_features.rds"))) {
  self_ref_features <- readRDS(here::here("data/cache/self_ref_features.rds"))
} else if (!SKIP_SELF_REFS && !is.null(works_raw) && nrow(works_raw) > 0) {

  all_ref_ids <- unique(unlist(works_raw$referenced_works))
  all_ref_ids <- all_ref_ids[!is.na(all_ref_ids) & nchar(all_ref_ids) > 0]
  cat(sprintf("Referenced work IDs: %d\n", length(all_ref_ids)))

  if (length(all_ref_ids) > 50000) {
    cat("  >50k referenced works – skipping (set SKIP_SELF_REFS <- FALSE to retry).\n")
  } else {
    cache_refs <- here::here("data/cache/oa_referenced_works.rds")
    if (file.exists(cache_refs)) {
      refs_raw <- readRDS(cache_refs)
    } else {
      refs_raw <- tryCatch(
        oa_fetch(entity = "works", openalex_id = all_ref_ids, verbose = TRUE),
        error = function(e) { message("Referenced works fetch failed: ", e$message); NULL }
      )
      if (!is.null(refs_raw)) saveRDS(refs_raw, cache_refs)
    }

    if (!is.null(refs_raw) && nrow(refs_raw) > 0) {
      extract_author_ids <- function(auth) {
        if (is.null(auth) || !is.data.frame(auth) || nrow(auth) == 0) return(character(0))
        na.omit(auth$id)
      }

      ref_author_lookup <- refs_raw |>
        transmute(work_id    = id,
                  author_ids = map(authorships, extract_author_ids))

      self_ref_features <- works_raw |>
        mutate(
          doi_o            = normalize_doi(doi),
          focal_author_ids = map(authorships, extract_author_ids),
          n_refs           = map_int(referenced_works, ~ length(.x %||% character(0)))
        ) |>
        select(doi_o, focal_author_ids, referenced_works, n_refs) |>
        mutate(
          number_self_references = map2_int(
            referenced_works, focal_author_ids,
            function(ref_ids, focal_ids) {
              if (length(ref_ids) == 0 || length(focal_ids) == 0) return(0L)
              matched <- ref_author_lookup |> filter(work_id %in% ref_ids)
              if (nrow(matched) == 0) return(0L)
              sum(map_lgl(matched$author_ids, ~ any(.x %in% focal_ids)))
            }
          ),
          prop_self_references = number_self_references / pmax(n_refs, 1)
        ) |>
        select(doi_o, number_self_references, prop_self_references)

      saveRDS(self_ref_features, here::here("data/cache/self_ref_features.rds"))
      cat(sprintf("Self-references computed for %d papers.\n", nrow(self_ref_features)))
    }
  }
}

# ── 5. Tier 2: TOP Factor ────────────────────────────────────────────────────

top_factor_data <- tryCatch({
  path <- here::here("data/external/topfactor.csv")
  if (!file.exists(path)) { cat("TOP Factor CSV not found – skipping.\n"); return(NULL) }
  raw <- read_csv(path, show_col_types = FALSE)
  raw |>
    transmute(
      issn_l          = tolower(gsub("\\s+", "", coalesce(!!!select(raw, "Issn")))),
      eissn_l         = coalesce(!!!select(raw, "Eissn")),
      top_factor_score = as.numeric(Total)
    ) |>
    filter(!is.na(issn_l)) |>
    distinct(issn_l, .keep_all = TRUE)
}, error = function(e) { message("TOP Factor load error: ", e$message); NULL })

# ── 6. Tier 2: SJR / Scimago ─────────────────────────────────────────────────

sjr_data <- tryCatch({
  path <- here::here("data/external/sjr.csv")
  if (!file.exists(path)) { cat("SJR CSV not found – skipping.\n"); return(NULL) }
  raw <- read_csv2(path, show_col_types = FALSE)

  format_issn <- function(x) {
    x <- tolower(gsub("\\s+", "", x))
    x <- if_else(nchar(x) == 8L, sub("^(\\w{4})(\\w{4})$", "\\1-\\2", x), x)
    if_else(x == "" | is.na(x), NA_character_, x)
  }

  sjr_col  <- intersect(c("SJR", "sjr"), names(raw))[1]
  sjrq_col <- intersect(c("SJR Best Quartile", "SJR.Best.Quartile"), names(raw))[1]

  raw_core <- raw |>
    mutate(
      issn_1                    = format_issn(str_split_fixed(Issn, ",\\s*", 2)[, 1]),
      issn_2                    = format_issn(str_split_fixed(Issn, ",\\s*", 2)[, 2]),
      journal_sjr               = as.numeric(.data[[sjr_col]]),
      journal_sjr_best_quartile = .data[[sjrq_col]],
      journal_Hindex_via_sjr    = as.numeric(`H index`),
      title_sjr                 = tolower(trimws(Title))
    ) |>
    select(issn_1, issn_2, title_sjr,
           journal_sjr, journal_sjr_best_quartile, journal_Hindex_via_sjr)

  bind_rows(
    raw_core |> filter(!is.na(issn_1)) |> rename(issn_l = issn_1) |> select(-issn_2),
    raw_core |> filter(!is.na(issn_2)) |> rename(issn_l = issn_2) |> select(-issn_1)
  ) |>
    distinct(issn_l, .keep_all = TRUE)

}, error = function(e) { message("SJR load error: ", e$message); NULL })

# ── 7. Assemble (one row per doi_o) ──────────────────────────────────────────

external_features <- tibble(doi_o = all_dois)

if (!is.null(works_features))
  external_features <- left_join(external_features, works_features, by = "doi_o")

# Keep one row per doi_o (most cited wins if duplicated)
external_features <- external_features |>
  group_by(doi_o) |>
  slice_max(number_citations, n = 1, with_ties = FALSE) |>
  ungroup()

if (!is.null(sources_features) && "source_id" %in% names(external_features))
  external_features <- left_join(external_features, sources_features, by = "source_id") |>
    coalesce_issn()

if (!is.null(author_id_lookup))
  external_features <- left_join(external_features, author_id_lookup, by = "doi_o") |>
    coalesce_issn()

if (!is.null(authors_features) && "first_author_id" %in% names(external_features)) {
  first_auth <- authors_features |>
    rename_with(~ paste0("first_", .x), -author_id) |>
    rename(first_author_id = author_id)
  external_features <- left_join(external_features, first_auth, by = "first_author_id") |>
    coalesce_issn()
}

if (!is.null(team_features))
  external_features <- left_join(external_features, team_features, by = "doi_o") |>
  coalesce_issn()

if (!is.null(gdp_data) && "first_author_country" %in% names(external_features)) {
  external_features <- left_join(
    external_features,
    rename(gdp_data, first_author_country = country_iso2,
           first_author_country_gdp = gdp_per_capita),
    by = "first_author_country"
  ) |> coalesce_issn()
}

if (!is.null(authors_features) && "last_author_id" %in% names(external_features)) {
  last_auth <- authors_features |>
    rename_with(~ paste0("last_", .x), -author_id) |>
    rename(last_author_id = author_id)
  external_features <- left_join(external_features, last_auth, by = "last_author_id") |>
    coalesce_issn()
}

if (!is.null(gdp_data) && "last_author_country" %in% names(external_features)) {
  external_features <- left_join(
    external_features,
    rename(gdp_data, last_author_country = country_iso2,
           last_author_country_gdp = gdp_per_capita),
    by = "last_author_country"
  ) |> coalesce_issn()
}

if (!is.null(self_ref_features)) {
  self_ref_features <- self_ref_features |>
    group_by(doi_o) |>
    slice_max(number_self_references, n = 1, with_ties = FALSE) |>
    ungroup()
  external_features <- left_join(external_features, self_ref_features, by = "doi_o") |>
    coalesce_issn()
}

# Ensure issn_l is populated before joining Tier-2 CSVs
if (!"issn_l" %in% names(external_features) && "journal_issn_l" %in% names(external_features))
  external_features <- rename(external_features, issn_l = journal_issn_l)

external_features <- coalesce_issn(external_features)

if (!is.null(top_factor_data) && "issn_l" %in% names(external_features))
  external_features <- left_join(external_features, top_factor_data, by = "issn_l") |>
    coalesce_issn()

if (!is.null(sjr_data) && "issn_l" %in% names(external_features)) {
  sjr_by_title <- sjr_data |>
    distinct(title_sjr, .keep_all = TRUE) |>
    select(-issn_l) |>
    rename(journal_sjr_fb               = journal_sjr,
           journal_sjr_best_quartile_fb = journal_sjr_best_quartile,
           journal_Hindex_via_sjr_fb    = journal_Hindex_via_sjr)

  external_features <- external_features |>
    left_join(sjr_data |> select(-title_sjr), by = "issn_l") |>
    coalesce_issn() |>
    mutate(title_lower = tolower(trimws(journal_name_oa))) |>
    left_join(sjr_by_title, by = c("title_lower" = "title_sjr")) |>
    coalesce_issn() |>
    mutate(
      journal_sjr               = coalesce(journal_sjr,               journal_sjr_fb),
      journal_sjr_best_quartile = coalesce(journal_sjr_best_quartile, journal_sjr_best_quartile_fb),
      journal_Hindex_via_sjr    = coalesce(journal_Hindex_via_sjr,    journal_Hindex_via_sjr_fb)
    ) |>
    select(-ends_with("_fb"), -title_lower)
}

external_features <- coalesce_issn(external_features)

# ── 8. Save ───────────────────────────────────────────────────────────────────

cat(sprintf("external_features: %d rows x %d cols\n",
            nrow(external_features), ncol(external_features)))

external_features |>
  summarise(across(everything(), ~ mean(is.na(.x)))) |>
  tidyr::pivot_longer(everything(), names_to = "col", values_to = "pct_missing") |>
  filter(pct_missing > 0) |>
  arrange(desc(pct_missing)) |>
  mutate(pct_missing = scales::percent(pct_missing, accuracy = 0.1)) |>
  print(n = 50)

saveRDS(external_features, here::here("data/external_features.rds"))
cat("Saved: data/external_features.rds\n")
