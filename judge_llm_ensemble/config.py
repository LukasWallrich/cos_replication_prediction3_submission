"""Config for the LLM ensemble of expert judges.

Paths are interpreted relative to the judge_llm_ensemble/ folder.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ---- Source data -----------------------------------------------------------

# Source CSVs to read claim_id, claim_text_o, doi_o, dataset, stats from.
SOURCES = [
    ROOT / "data" / "Round3" / "test-set_for-participants_3.csv",
    ROOT / "data" / "dataset_incl_round2.csv",
]

# TEI extraction CSV — joined by DOI to supply title + abstract + pub_date.
TEI_CSV = ROOT / "extracted_tei_all.csv"

# Which datasets in dataset_incl_round2.csv to include in the ensemble run.
# (We don't run on FORRT base — see the plan: only R1+R2+R3 needed.)
INCLUDE_DATASETS = {"round1", "round2"}

# ---- Output ----------------------------------------------------------------

MANIFEST = HERE / "manifest.jsonl"
RAW_DIR = HERE / "raw_responses"
OUTPUT_CSV = HERE / "ensemble_scores.csv"

PERSONAS_DIR = HERE / "personas"
PROMPT_TEMPLATE = HERE / "prompt_template.txt"
SCHEMA_FILE = HERE / "schema.json"

# ---- Models ----------------------------------------------------------------
# Each entry is (vendor, model_id, display_name). display_name is what goes
# into cache filenames and CSV column names; keep it short and alphanumeric.

MODELS = [
    ("anthropic", "opus",                   "opus"),     # via `claude -p --model opus` (uses Claude Code subscription, no per-call charge)
    ("openai",    "gpt-5.5",                "gpt55"),    # via openai SDK with service_tier="flex" (~50% off, slower)
    ("gemini",    "gemini-3.1-pro-preview", "gem31"),    # via google.genai (sync API; flex/batch not used)
]

# ---- Personas --------------------------------------------------------------
# One entry per persona; the loader reads personas/<key>.txt for the
# system-prompt body. display_name is what goes into CSV columns.

PERSONAS = [
    ("skeptic",      "Skeptical methodologist"),
    ("field",        "Health-behavioural field expert"),
    ("bayesian",     "Bayesian forecaster"),
    ("generalist",   "Generalist critical reader"),
]

# ---- Power assumption ------------------------------------------------------
# COS protocol assumes replication studies target 80% power on average.
ASSUMED_POWER = 0.80
ALPHA = 0.05

# ---- Bucket scale ----------------------------------------------------------
# Keep in sync with schema.json and prompt_template.txt.

BUCKETS = [
    ("almost_certainly_not", "almost certainly will not replicate", "<= 5%",  0.03),
    ("very_unlikely",        "very unlikely",                       "5-20%",   0.13),
    ("unlikely",             "unlikely",                            "20-40%",  0.30),
    ("about_even",           "about even",                          "40-60%",  0.50),
    ("likely",               "likely",                              "60-80%",  0.70),
    ("very_likely",          "very likely",                         "80-95%",  0.87),
    ("almost_certainly_yes", "almost certainly will replicate",     ">= 95%",  0.97),
]
BUCKET_MIDPOINT = {k: mid for (k, _, _, mid) in BUCKETS}

# ---- Concurrency / runtime -------------------------------------------------

# How many parallel API calls per vendor (separate semaphore per vendor so
# rate limits don't collide). Tune down if hitting rate-limit errors.
PARALLEL_PER_VENDOR = 4

# Per-call timeout in seconds for the default/standard tier.
CALL_TIMEOUT_S = 120

# Hard timeout on flex-tier attempts. Flex tier may queue; we wait up to
# this long, then fast-fail to standard tier instead of blocking a worker.
# 30s is short enough to keep throughput high when flex is congested,
# and long enough to catch flex when it's idle (typical flex call <20s).
FLEX_TIMEOUT_S = 30

# Cap on output tokens. The JSON response is small (~200 tokens) — capping
# keeps OpenRouter credit usage predictable and avoids 402 errors from
# vendors that reserve the full max_tokens budget upfront.
MAX_OUTPUT_TOKENS = 800

# Gemini 3.x is a thinking model: thinking tokens count against the
# max_output_tokens budget. Give it a moderate thinking budget on top of
# the JSON output cap. (Set to 0 to disable thinking; -1 = dynamic.)
GEMINI_THINKING_BUDGET = 2048

# OpenAI GPT-5.x is a reasoning model — reasoning tokens are billed and
# count against max_completion_tokens. Same idea as Gemini's thinking budget.
OPENAI_REASONING_BUDGET = 2048
