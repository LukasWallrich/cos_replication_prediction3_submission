#!/usr/bin/env Rscript
# Combined-pass extractor: extracts BOTH claim-anchored and paper-level
# fields in a single API call. The union schema is assembled at runtime
# from schemas/{claim,paper}_schema.json; the prompt template is inline
# below.
#
# Usage:
#   Rscript run_combined.R [N=10] [--provider {openai|gemini|codex}]
#                          [--model <id>] [--tag <suffix>]
#                          [--reasoning {low|medium|high}] [--flex]
#                          [--shard N/M]
#
# --shard N/M partitions the manifest by index modulo M; only entries
# where (index-1) %% M + 1 == N are processed. Use this to run multiple
# workers in parallel against a shared --tag (each worker sees its own
# slice, so they never race on the same claim_id).
#
# For --provider codex: model defaults to gpt-5.4 (gpt-5.4-codex /
# gpt-5.4-flash / gpt-5.5-mini are rejected by ChatGPT-account Codex; only
# gpt-5.4 and gpt-5.4-mini are accepted). Tag defaults to "codex54hi".
# --flex is ignored (no service-tier toggle on the ChatGPT subscription).
#
# Cache: raw_responses_combined_<tag>/<claim_id>.json

suppressPackageStartupMessages({
  library(ellmer)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
positional <- args[!startsWith(args, "--")]
MAX_NEW <- if (length(positional) >= 1L) as.integer(positional[[1L]]) else 10L

get_flag <- function(name, default = NULL) {
  i <- which(args == paste0("--", name))
  if (length(i) == 0L) return(default)
  args[[i[[1L]] + 1L]]
}
has_flag <- function(name) any(args == paste0("--", name))

PROVIDER  <- get_flag("provider", "gemini")
if (!(PROVIDER %in% c("openai", "gemini", "codex"))) {
  stop("unknown --provider: ", PROVIDER,
       " (expected openai, gemini, or codex)")
}
MODEL     <- get_flag("model",
                      switch(PROVIDER,
                             openai = "gpt-5.4",
                             gemini = "gemini-3.5-flash",
                             codex  = "gpt-5.4"))
TAG       <- get_flag("tag",
                      switch(PROVIDER,
                             codex = "codex54hi",
                             "gem35flash"))   # historical default for openai/gemini
REASONING <- get_flag("reasoning", NA_character_)
FLEX      <- has_flag("flex")
if (FLEX && PROVIDER == "codex") {
  message("note: --flex is ignored for --provider codex (no service-tier toggle).")
  FLEX <- FALSE
}

SHARD_RAW <- get_flag("shard", NA_character_)
if (!is.na(SHARD_RAW)) {
  parts <- suppressWarnings(as.integer(strsplit(SHARD_RAW, "/", fixed = TRUE)[[1]]))
  if (length(parts) != 2L || any(is.na(parts)) ||
      parts[[1L]] < 1L || parts[[1L]] > parts[[2L]])
    stop("--shard must be 'N/M' with 1 <= N <= M, got: ", SHARD_RAW)
  SHARD_N <- parts[[1L]]; SHARD_M <- parts[[2L]]
} else { SHARD_N <- 1L; SHARD_M <- 1L }

REVERSE          <- has_flag("reverse")
ALLOW_NONPRIMARY <- has_flag("allow-nonprimary")
# --exclude-from accepts a comma-separated list of cache directories.
# A claim is skipped if its <cid>.json exists in ANY of them.
EXCLUDE_RAW  <- get_flag("exclude-from", NA_character_)
EXCLUDE_DIRS <- if (is.na(EXCLUDE_RAW)) character(0) else
                trimws(strsplit(EXCLUDE_RAW, ",", fixed = TRUE)[[1L]])
for (d in EXCLUDE_DIRS) if (!dir.exists(d))
  stop("--exclude-from directory does not exist: ", d)

script_arg <- sub("^--file=", "",
                  grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
HERE <- if (length(script_arg) == 1L) normalizePath(dirname(script_arg)) else getwd()
setwd(HERE)

SCHEMAS_DIR <- "schemas"
RAW_DIR     <- sprintf("raw_responses_combined_%s", TAG)
MANIFEST    <- "manifest.jsonl"
CLEAN_TEI   <- "_clean_tei.py"

dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)
`%||%` <- function(a, b) if (is.null(a)) b else a

# Transient-error detector (rate limit + network failure). If matched we
# DO NOT write the cache file (so the claim stays retryable) AND we break
# the worker loop (no point hammering an exhausted quota or a dead
# resolver — wait for the reset / network).
TRANSIENT_RX <- paste(c(
  # Rate / quota
  "rate.?limit", "rate[_ ]limited", "\\b429\\b", "Too Many Requests",
  "quota", "usage limit", "Plan limit", "weekly limit",
  "5h limit", "hourly limit", "daily limit", "exceeded",
  # Network / DNS / transport
  "Could not resolve", "Failed to perform HTTP request",
  "Connection refused", "Connection reset", "Connection timed out",
  "ECONNREFUSED", "ECONNRESET", "ETIMEDOUT", "ENETUNREACH", "EAI_AGAIN",
  "Network is unreachable", "Temporary failure in name resolution",
  "SSL.*handshake", "TLS.*handshake", "no route to host",
  "curl_fetch_memory",
  # Server-side transient (OpenAI 5xx tends to clear within seconds)
  "HTTP 5[0-9][0-9]", "Internal Server Error",
  "Bad Gateway", "Service Unavailable", "Gateway Timeout"
), collapse = "|")
is_rate_limit_error <- function(msg) {
  isTRUE(grepl(TRANSIENT_RX, msg, ignore.case = TRUE))
}

# ---- 1. Load source schemas -----------------------------------------------
schema_claim <- fromJSON(file.path(SCHEMAS_DIR, "claim_schema.json"),
                         simplifyVector = FALSE)
schema_paper <- fromJSON(file.path(SCHEMAS_DIR, "paper_schema.json"),
                         simplifyVector = FALSE)

# ---- 2. Build the union schema (drop optional `notes` to avoid collision) -
schema <- list(
  type = "object",
  additionalProperties = FALSE,
  properties = c(
    Filter(function(p) TRUE,
           schema_claim$properties[setdiff(names(schema_claim$properties), "notes")]),
    schema_paper$properties[setdiff(names(schema_paper$properties), "notes")]),
  required = c(setdiff(unlist(schema_claim$required), "notes"),
               setdiff(unlist(schema_paper$required), "notes")),
  `$defs` = schema_claim[["$defs"]])

# ---- 3. Provider-specific schema patches (mirror run.R) -------------------
resolve_ref <- function(node, defs) {
  ref <- node[["$ref"]]
  if (is.null(ref) || !startsWith(ref, "#/$defs/")) return(node)
  key <- sub("^#/\\$defs/", "", ref)
  target <- defs[[key]]
  if (is.null(target)) return(node)
  node[["$ref"]] <- NULL
  for (nm in names(target)) if (is.null(node[[nm]])) node[[nm]] <- target[[nm]]
  node
}
patch_schema <- function(node, defs, provider) {
  if (!is.list(node)) return(node)
  node <- resolve_ref(node, defs)
  if (identical(node$type %||% "", "object") && !is.null(node$properties)) {
    node$required <- as.list(names(node$properties))
    if (provider == "openai") {
      if (is.null(node$additionalProperties)) node$additionalProperties <- FALSE
    } else node$additionalProperties <- NULL
  }
  for (nm in names(node)) node[[nm]] <- patch_schema(node[[nm]], defs, provider)
  node
}
# Codex `--output-schema` is routed through OpenAI strict mode, so the
# OpenAI patch rules (required exhaustive, additionalProperties: false) apply.
schema_patch_provider <- if (PROVIDER == "codex") "openai" else PROVIDER
schema_patched   <- patch_schema(schema, schema_claim[["$defs"]],
                                 schema_patch_provider)
schema_patched[["$defs"]] <- NULL
schema_json_text <- toJSON(schema_patched, auto_unbox = TRUE, null = "null")
# schema_type is ellmer-only; codex passes the schema to the CLI directly.
schema_type <- if (PROVIDER != "codex")
  type_from_schema(text = schema_json_text) else NULL

# ---- 4. Build the FIELDS_LIST / PRACTICES_LIST blocks ---------------------
value_space <- function(p) {
  if (!is.null(p$enum))
    return(paste0("one of ", paste(sprintf("`%s`", p$enum), collapse = ", ")))
  ty <- p$type %||% ""
  if (ty %in% c("integer", "number"))
    return(sprintf("%s in [%s, %s]", ty, p$minimum %||% "-inf", p$maximum %||% "+inf"))
  if (is.list(ty)) return(paste(ty, collapse = " or "))
  if (identical(ty, "")) "value" else ty
}
field_shape <- function(prop) {
  ref <- prop[["$ref"]]
  if (!is.null(ref) && ref == "#/$defs/yesno")
    return("`status` ∈ {yes, no, unclear}; `evidence`: short verbatim quote.")
  if (!is.null(ref) && ref == "#/$defs/practice_partial")
    return("`status` ∈ {yes, no, partly, unclear}; `evidence`: short verbatim quote.")
  if (!is.null(ref) && ref == "#/$defs/extraction")
    return("`value`: free text; `evidence`: section / passage reference.")
  inner <- prop$properties %||% list()
  if (!is.null(inner$rating))
    return(sprintf("`rating`: %s; `reasoning`: justification.", value_space(inner$rating)))
  if (!is.null(inner$value) && !is.null(inner$evidence))
    return(sprintf("`value`: %s; `evidence`: passage reference.", value_space(inner$value)))
  if (!is.null(inner$status) && !is.null(inner$evidence))
    return(sprintf("`status` ∈ {%s}; `evidence`: short verbatim quote.",
                   paste(sprintf("'%s'", inner$status$enum), collapse = ", ")))
  NA_character_
}
build_block <- function(props) {
  ls <- vapply(names(props), function(nm) {
    p <- props[[nm]]; s <- field_shape(p)
    if (is.na(s)) return(NA_character_)
    sprintf("  - **%s** — %s  Returns: %s", nm, p$description %||% "", s)
  }, character(1))
  paste(ls[!is.na(ls)], collapse = "\n")
}
fields_block    <- build_block(schema_claim$properties[
  setdiff(names(schema_claim$properties), "notes")])
practices_block <- build_block(schema_paper$properties[
  setdiff(names(schema_paper$properties), "notes")])

# ---- 5. Assemble combined prompt ------------------------------------------
combined_prompt_tpl <- paste0(
"You are extracting structured metadata from an academic paper to support a
replication-prediction model. The paper is supplied below as cleaned GROBID
TEI XML (references and back-matter that no schema field references have
been stripped):

<paper_tei>
{TEI_XML}
</paper_tei>

A specific claim (claim_id={CLAIM_ID}) from this paper is being scored:

  {CLAIM_TEXT}

You must produce TWO blocks of fields in a single JSON response, conforming
to the requested schema:

---------------------------------------------------------------
PART A — claim-anchored fields
---------------------------------------------------------------
Every field here is anchored on the specific claim above, not on the paper
as a whole. For Likert / categorical judgements, evaluate the specific
claim's evidence — not the paper's contribution overall.

For multistudy papers, the methodology / design / sample / measurement /
power / preregistration / robustness / exploratory_study fields below
MUST anchor on the specific sub-study that produced the claim's test
result — i.e., the same sub-study used for `n_o_llm`, `es_o_llm`,
`p_o_llm`. Do NOT mix evidence from different sub-studies in these
fields. For single-study papers, this is trivially the only study.

", fields_block, "

Rules for Part A:
- `evidence` is a short verbatim quote (≤ 30 words) or a page/section reference. Do not paraphrase.
- `reasoning` (on rated fields) is the only place free-form explanation belongs; keep it tight.
- Use `unclear` (yesno) or `not reported` / `none identified` (extraction) only when the paper is genuinely ambiguous.
- For `is_focal_claim`: judge ONLY by prominence in the paper's framing — does the abstract, the conclusions, or the headline statistic of the paper foreground this claim? A claim can be focal even if the result is null, marginal, surprising, or contradicts the authors' hypothesis. Do NOT downgrade to 'no' merely because the finding is weak, qualified, or framed unfavorably — focal status is about WHAT THE PAPER FOREGROUNDS, not how the result turned out. If the abstract or conclusions lead with this claim, it is focal.
- Hypothesis-related fields (`corresponding_hypothesis`, `directional_hypothesis`, `structured_hypothesis`, `theory_based_hypothesis`, `hypothesis_on_main_effect`, `hypothesis_on_mediation`, `hypothesis_on_moderation`) must be anchored on a hypothesis stated in the Introduction / Hypotheses / Theory section. Do NOT treat sentences in Results, Abstract, or Discussion that describe observed outcomes as if they were hypotheses.
- If `corresponding_hypothesis` is 'none identified' (no hypothesis stated for the claim), set `structured_hypothesis`, `hypothesis_on_main_effect`, `hypothesis_on_mediation`, and `hypothesis_on_moderation` to 'no'. (`directional_hypothesis` and `theory_based_hypothesis` already default to 'no' per their descriptions.) Reserve 'unclear' for the case where a hypothesis IS stated but its structure is genuinely ambiguous.
- Study-design partition. Every empirical study has both a timing structure and a manipulation status. Across the five design fields:
  * Timing (mutually exclusive): exactly one of `cross_sectional_study` or `longitudinal_study` should be 'yes'. Cross-sectional = each participant measured once; longitudinal = same participants measured at ≥ 2 time points.
  * Manipulation (mutually exclusive): exactly one of `observational_study` or `experimental_study` should be 'yes'. Observational = no manipulation of the independent variable; experimental = at least one independent variable is manipulated by the researcher (random assignment is not required).
  * `RCT_study` is a strict subset of `experimental_study`: if `RCT_study`='yes' then `experimental_study` must be 'yes' and `observational_study` must be 'no'.
  * Code 'unclear' only if the paper genuinely does not provide enough information to decide one side of a partition.
  * Never mark both members of a mutually-exclusive pair 'no' for the same study.

---------------------------------------------------------------
PART B — paper-level fields
---------------------------------------------------------------
These fields describe the PAPER as a whole — its open-science practices,
funding / conflict-of-interest disclosures, hypothesis structure, topic,
writing quality, and structural meta-data (study_count,
within_paper_replication). Do NOT anchor on the claim above. Methodology
fields (design, sample, measurement, power, registration, robustness,
exploratory_study) live in PART A above, since they vary by sub-study in
multistudy papers.

", practices_block, "

Rules for Part B:
- For each yes/no question, decide whether the paper reports it as yes, no, or unclear (with `open_materials` additionally supporting 'partly'). Provide a short verbatim quote (≤ 30 words) or a section reference in `evidence`.

Return JSON only, conforming to the requested schema. Do not include any commentary outside the JSON.
")

# ---- 6. Manifest, chat factory, TEI cleaner -------------------------------
manifest <- lapply(readLines(MANIFEST, warn = FALSE), fromJSON, simplifyVector = FALSE)

chat_params <- if (!is.na(REASONING)) params(reasoning_effort = REASONING) else NULL
gemini_api_args <- list()
if (PROVIDER == "gemini" && !is.na(REASONING)) {
  budget <- switch(REASONING, low = 512L, medium = 4096L, high = -1L)
  gemini_api_args$generationConfig <- list(thinkingConfig = list(thinkingBudget = budget))
}
if (PROVIDER == "gemini" && FLEX) gemini_api_args$service_tier <- "flex"

new_chat <- if (PROVIDER == "openai") {
  function() chat_openai(model = MODEL, echo = "none", params = chat_params,
                         service_tier = if (FLEX) "flex" else "auto")
} else if (PROVIDER == "gemini") {
  function() chat_google_gemini(model = MODEL, echo = "none",
                                api_args = gemini_api_args)
} else NULL                                  # codex bypasses ellmer

# Codex setup — see run.R for the rationale; this block is identical.
if (PROVIDER == "codex") {
  find_codex <- function() {
    bin <- Sys.which("codex")
    if (nzchar(bin)) return(unname(bin))
    fallback <- "/Applications/Codex.app/Contents/Resources/codex"
    if (file.exists(fallback)) return(fallback)
    stop("codex CLI not found (tried PATH and ",
         "/Applications/Codex.app/Contents/Resources/codex). ",
         "Install with `brew install --cask codex` or fix the symlink.")
  }
  CODEX_BIN <- find_codex()
  codex_schema_file <- tempfile("codex_schema_", fileext = ".json")
  write(schema_json_text, file = codex_schema_file)
  codex_last_msg <- tempfile("codex_last_", fileext = ".txt")
  codex_args <- c(
    "exec", "-",
    "--ephemeral",
    "--skip-git-repo-check",
    "--sandbox", "read-only",
    "--color", "never",
    "--ignore-rules",
    "-C", tempdir(),
    "--output-schema",       codex_schema_file,
    "--output-last-message", codex_last_msg,
    "-m", MODEL
  )
  if (!is.na(REASONING)) {
    codex_args <- c(codex_args, "-c",
                    paste0("model_reasoning_effort=", REASONING))
  }
}

# Unified LLM call: returns list(ok=, structured=, tokens=, cost=) on success,
# or list(ok=FALSE, error=msg) on failure. Used by the main loop below.
call_llm <- if (PROVIDER == "codex") {
  function(prompt) {
    if (file.exists(codex_last_msg)) unlink(codex_last_msg)
    tryCatch({
      out <- system2(CODEX_BIN, codex_args, input = prompt,
                     stdout = TRUE, stderr = TRUE)
      status <- attr(out, "status") %||% 0L
      if (!identical(as.integer(status), 0L) || !file.exists(codex_last_msg)) {
        stop(sprintf("codex exec exit=%s: %s", status,
                     paste(tail(out, 8), collapse = " | ")))
      }
      last_text <- paste(readLines(codex_last_msg, warn = FALSE), collapse = "\n")
      structured <- fromJSON(last_text, simplifyVector = FALSE)
      tok_idx <- which(out == "tokens used")
      tokens <- if (length(tok_idx) == 1L && tok_idx < length(out)) {
        n <- suppressWarnings(as.integer(gsub("[^0-9]", "", out[[tok_idx + 1L]])))
        if (!is.na(n)) list(total = n) else NULL
      } else NULL
      list(ok = TRUE, structured = structured,
           tokens = tokens, cost = NA_real_)
    }, error = function(e) {
      msg <- conditionMessage(e)
      list(ok = FALSE, error = msg, rate_limited = is_rate_limit_error(msg))
    })
  }
} else {
  function(prompt) {
    tryCatch({
      chat <- new_chat()
      out  <- chat$chat_structured(prompt, type = schema_type)
      list(ok = TRUE, structured = out,
           tokens = chat$last_turn()@tokens,
           cost = tryCatch(chat$get_cost(), error = function(e) NA_real_))
    }, error = function(e) {
      msg <- conditionMessage(e)
      list(ok = FALSE, error = msg, rate_limited = is_rate_limit_error(msg))
    })
  }
}

clean_tei <- function(tei_rel) {
  # tei_path in the manifest is relative to llm_extraction/ (which is a sibling
  # of `data/`).
  tei_abs <- normalizePath(tei_rel, mustWork = TRUE)
  out <- system2("python3", c(CLEAN_TEI, shQuote(tei_abs)),
                 stdout = TRUE, stderr = TRUE)
  status <- attr(out, "status") %||% 0L
  if (!identical(as.integer(status), 0L))
    stop(sprintf("_clean_tei.py failed: %s", paste(out, collapse = " | ")))
  paste(out, collapse = "\n")
}

# ---- 7. Iterate ------------------------------------------------------------
cat(sprintf("[combined %s/%s%s] MAX_NEW=%d shard=%d/%d%s%s%s cache=%s\n",
            PROVIDER, MODEL, if (FLEX) " flex" else "",
            MAX_NEW, SHARD_N, SHARD_M,
            if (REVERSE) " reverse" else "",
            if (ALLOW_NONPRIMARY) " +nonprimary" else "",
            if (length(EXCLUDE_DIRS) > 0L)
              sprintf(" exclude=%s", paste(EXCLUDE_DIRS, collapse = ",")) else "",
            RAW_DIR))
new_attempts <- 0L; processed <- 0L; skipped <- 0L; failed <- 0L

iter <- seq_along(manifest)
if (REVERSE) iter <- rev(iter)
for (i in iter) {
  meta   <- manifest[[i]]
  cid    <- meta$claim_id
  tei    <- meta$tei_path %||% ""
  raw_file <- file.path(RAW_DIR, paste0(cid, ".json"))

  if (((i - 1L) %% SHARD_M) + 1L != SHARD_N) { skipped <- skipped + 1L; next }
  if (!nzchar(tei)) { skipped <- skipped + 1L; next }
  if (!ALLOW_NONPRIMARY && !isTRUE(meta$primary)) { skipped <- skipped + 1L; next }
  if (file.exists(raw_file))               { skipped <- skipped + 1L; next }
  if (length(EXCLUDE_DIRS) > 0L &&
      any(vapply(EXCLUDE_DIRS,
                 function(d) file.exists(file.path(d, paste0(cid, ".json"))),
                 logical(1L)))) {
    skipped <- skipped + 1L; next
  }
  if (new_attempts >= MAX_NEW) break

  new_attempts <- new_attempts + 1L
  pdf_name <- basename(meta$pdf_path %||% "")
  cat(sprintf("\n→ [%d/%d] %s  (%s)\n", new_attempts, MAX_NEW, cid, pdf_name))
  t0 <- Sys.time()

  prompt <- tryCatch({
    tei_xml <- clean_tei(tei)
    p <- combined_prompt_tpl
    p <- sub("{TEI_XML}",    tei_xml,         p, fixed = TRUE)
    p <- sub("{CLAIM_ID}",   cid,             p, fixed = TRUE)
    p <- sub("{CLAIM_TEXT}", meta$claim_text, p, fixed = TRUE)
    p
  }, error = function(e) {
    cat(sprintf("  ✗ prompt build failed: %s\n", conditionMessage(e))); NULL
  })
  if (is.null(prompt)) { failed <- failed + 1L; next }

  result <- call_llm(prompt)

  if (!isTRUE(result$ok) && isTRUE(result$rate_limited)) {
    cat(sprintf("  ✗ RATE LIMIT — not caching, halting worker.\n    %s\n",
                substr(result$error, 1, 300)))
    failed <- failed + 1L
    break
  }

  payload <- list(
    claim_id = cid, provider = PROVIDER, model = MODEL, tag = TAG,
    mode = "combined", flex = FLEX, reasoning = REASONING,
    extracted_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    duration_s = as.numeric(difftime(Sys.time(), t0, units = "secs")),
    is_error = !isTRUE(result$ok))
  if (isTRUE(result$ok)) {
    payload$structured_output <- result$structured
    payload$tokens <- result$tokens
    payload$cost   <- result$cost
  } else payload$result <- result$error

  write(toJSON(payload, auto_unbox = TRUE, pretty = TRUE, null = "null",
               na = "null"), file = raw_file)

  if (isTRUE(result$ok)) {
    processed <- processed + 1L
    cat(sprintf("  ✓ done in %ds\n", round(payload$duration_s)))
  } else {
    failed <- failed + 1L
    cat(sprintf("  ✗ %s\n", substr(result$error, 1, 200)))
  }
}

cat(sprintf("\nProcessed: %d   Skipped: %d   Failed: %d\n",
            processed, skipped, failed))
cat(sprintf("Raw JSON: %s/\n", RAW_DIR))
