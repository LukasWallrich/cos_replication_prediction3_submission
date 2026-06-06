# llm_extraction

LLM-based structured extraction of claim- and paper-level metadata from the
academic PDFs (via GROBID TEI XML). This is the **provenance of the structured
LLM ratings** that `../pipeline/03_llm_features.R` parses into model features
(e.g. perceived surprisingness, sample adequacy, within-paper replication,
intervention flag).

It is a Stage-1 build step: it needs the challenge papers/claims (see
`../DATA.md`) and makes paid API calls, so treat it as documentation of how the
LLM features were produced rather than a turnkey step. Its raw outputs and the
`manifest.jsonl` index (both challenge-derived) are not shipped; run it only when
rebuilding from the raw data. **Run from inside this folder** (`cd llm_extraction`)
— the scripts use `../data/...` paths. Keys come from environment variables; no
key is stored in this repository.

## Running

```bash
# Default: Gemini 3.5 Flash (paid via ellmer)
Rscript run_combined.R 10

# OpenAI gpt-5.4 (paid API)
Rscript run_combined.R 10 --provider openai --tag gpt54flexhi --flex --reasoning high

# OpenAI gpt-5.4 via Codex CLI (free on a ChatGPT subscription)
Rscript run_combined.R 10 --provider codex
```

The numeric arg is `MAX_NEW` — the cap on **new** extractions this run.
Already-cached `<claim_id>.json` files are skipped without counting, so
re-running pulls additional rows up to the cap each time. Use `0` for "no cap".

### Providers

| `--provider` | Auth | Default model | Cache tag |
| --- | --- | --- | --- |
| `openai` | `OPENAI_API_KEY` (ellmer) | `gpt-5.4` | `gem35flash` (legacy default — pass `--tag`) |
| `gemini` | `GEMINI_API_KEY` (ellmer) | `gemini-3.5-flash` | `gem35flash` |
| `codex`  | `~/.codex/auth.json` (ChatGPT subscription) | `gpt-5.4` | `codex54hi` |

Codex bypasses ellmer and shells out to `codex exec` headless mode with
`--output-schema` + `--output-last-message`. ChatGPT-account Codex only
accepts the models it ships with (`gpt-5.4`, `gpt-5.4-mini`) — `gpt-5.4-codex`,
`gpt-5.4-flash`, `gpt-5.5-mini` are rejected.

`--reasoning low|medium|high` is forwarded to the model (ellmer `params`
for openai, `generationConfig.thinkingConfig.thinkingBudget` for gemini,
`-c model_reasoning_effort=<value>` for codex). `--flex` enables the flex
service tier on openai/gemini; it's ignored for codex.

## Layout

```
llm_extraction/
├── run_combined.R                # single-call extractor (claim + paper fields)
├── _prep_claims.R                # assembles the claim list from the round CSVs
├── _clean_tei.py                 # strips GROBID TEI to extraction-relevant sections
├── _build_manifest.py            # builds manifest.jsonl from source CSVs (+ TEI)
├── _enrich_manifest.py           # adds tei_path / claim_count to existing manifest
└── schemas/                      # field definitions / prompts read by run_combined.R
    ├── claim_schema.json
    └── paper_schema.json
```

Not shipped (challenge-derived or regenerable): `manifest.jsonl` (one row per
claim×paper, contains claim text), the `raw_responses_combined_<tag>/` extraction
caches, and the original `assessment/` / `archive/` material. The schemas and
`_clean_tei.py` are read by `run_combined.R` at runtime; the cache directory is
keyed by `<tag>` so multiple model runs coexist without collision.

## Manifest

`manifest.jsonl` covers ~290 (claim, paper) rows: 45 Round 3 test claims plus
245 Round 1/Round 2 training claims. To rebuild after data changes:

```bash
python3 _build_manifest.py \
  --papers-dir ../data/Round3/Round3\ Papers \
  --papers-dir ../data/Round1/Round1\ Papers \
  --papers-dir ../data/Round2/Round2\ Papers \
  --papers-dir ../data/Papers\ Trainingdata \
  --tei-dir ../data/tei_xml/Test\ set\ papers/Round\ 3 \
  --tei-dir ../data/tei_xml/Test\ set\ papers/Round\ 1 \
  --tei-dir ../data/tei_xml/Test\ set\ papers/Round\ 2 \
  --tei-dir ../data/tei_xml/Training\ set\ papers \
  ../data/Round3/test-set_for-participants_3.csv \
  ../data/dataset_incl_round2.csv \
  > manifest.jsonl
```
