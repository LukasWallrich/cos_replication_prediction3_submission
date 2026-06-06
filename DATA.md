# Data

What data this package ships, what it deliberately omits, and how to obtain the
rest so the data-dependent stages can run.

## What reproduces from the shipped data

| Regenerates from this package alone | Needs the restricted challenge data |
|---|---|
| `pipeline/10_submission.R` — rebuilds `output/submission.csv` from the three deployed per-claim columns | `01`–`04` (raw data → feature matrix) and `09` (fit/score every candidate, bootstrap CIs) read the manuscripts, claim text and labels |
| `judge_llm_ensemble/aggregate.py` — rebuilds the m3 crowd scores from the shipped raw responses | `03` re-parses the LLM-extraction caches (held back; `data/cache/llm_features.rds` ships instead) |
| `bayesian_priors/extract_llm_priors.R` — rebuilds the LLM priors from the shipped JSONs | the validation demo at the foot of that script needs `train_base`/`test_base` |

In short: the **final deliverable and the per-column predictors regenerate
offline**; the **model-fitting stages need the challenge data**, obtainable as
described at the end of this file.

## What ships in this package

| Path | Contents | Source / terms |
|---|---|---|
| `output/submission.csv` | the deliverable: `claim_id, m1, m2, m3` for the 45 R3 claims | ours |
| `output/r3_elnet_plus_predictor.csv` | deployed m1 (feature model) per claim | ours |
| `output/r3_prior_shrinkage_predictor.csv` | deployed m2 (prior × evidence): `prior_3fam, p_base, pred` | ours |
| `output/submission_diagnostics.csv` | trio correlations / distribution diagnostics | ours |
| `judge_llm_ensemble/ensemble_scores.csv` | aggregated LLM crowd scores (m3) per claim | ours |
| `judge_llm_ensemble/raw_responses/` | per-call m3 judge responses (bucket + paraphrased reasoning + `claim_id`) — re-run `aggregate.py` offline | ours |
| `data/cache/llm_features.rds` | the **parsed LLM feature values**, one row per `claim_id` (181 model-derived numeric/categorical columns) — lets `04` rebuild the feature matrix with **no LLM API calls**. Scrubbed of the `*_proof`/`*_explanation` audit text that quotes manuscripts | ours |
| `bayesian_priors/priors_*.json`, `priors_by_model.rds`, `prior_predictor_table.csv` | LLM-elicited coefficient priors (the `.rds` is rebuilt from the JSONs by `extract_llm_priors.R`) | ours |
| `data/flora_health_addon.csv` | 17 large-N health replications we coded from the public **FORRT Library of Replication Attempts (FLoRA)** (consumed by `01`) | ours / FLoRA — public |
| `data/external/sjr.csv` | SCImago Journal Rank metrics | SCImago — public, retains its own terms |
| `data/external/topfactor.csv` | TOP Factor journal transparency scores | Center for Open Science — public |
| `data/external/flora.csv` | FLoRA replication records (author-overlap source) | FLoRA — public |

With one exception, these artifacts are keyed by `claim_id` and hold only our
model outputs and public reference tables — **no challenge claim text and no
ground-truth labels**. The exception, `data/flora_health_addon.csv`, does carry
original-claim text (`claim_text_o`) and replication outcomes
(`statistical_success`), but these come entirely from the **public,
openly-licensed FLoRA** database, not the challenge materials — so nothing
challenge-supplied is redistributed.

## What is deliberately NOT shipped (and why)

Under the challenge **Terms & Conditions**, the source manuscripts, claim text,
and ground-truth replication labels may be used only for the challenge and **not
redistributed**. So the following are excluded:

| Excluded | What it is |
|---|---|
| `data/Round1/`, `data/Round2/`, `data/Round3/` | the challenge round datasets (claim text, metadata, labels) |
| `data/Papers Trainingdata/`, any PDFs | the source manuscripts |
| `data/tei_xml/`, `extracted_tei_all.*` | GROBID full-text XML derived from the manuscripts |
| `data/FORRT_Training_Data.csv`, `data/dataset_incl_round2.*`, `data/all_claims.csv` | assembled claim/label tables |
| `data/train_base.rds`, `data/test_base.rds`, the OpenAlex/FReD caches in `data/cache/` (except the scrubbed `llm_features.rds`, which **does** ship) | feature matrices **derived from** the manuscripts, plus large network caches |
| `llm_extraction/raw_responses_combined_*/`, all `manifest*.jsonl` | structured-extraction caches and manifests — these **quote claim/manuscript text verbatim** (the `claim_input`/`evidence`/`claim_text` fields), so they fall under the no-redistribution terms |

The derived `.rds` feature matrices are omitted both because they embed
manuscript-derived content and because they are large; rebuild them locally with
Stage 1 (see below).

The m3 judge `raw_responses/` **do** ship (they hold only model verdicts and
paraphrased reasoning keyed by `claim_id`, no source text), so `aggregate.py` can
be re-run offline. The structured-extraction responses are held back only because
they embed verbatim manuscript quotes; rebuild them with Stage 1 if you have the
manuscripts.

## How to obtain the challenge data

1. **Challenge datasets and papers** — provided by the Center for Open Science to
   participating teams. See the challenge Terms & Conditions and the Paper Access
   Agreement. Place them as the pipeline expects:
   - `data/Round1/`, `data/Round2/`, `data/Round3/` — the round CSVs
   - `data/Round3/test_set_missing_data_patch.csv` — hand-filled effect sizes for
     the R3 test claims that ship with missing stats (read by `01`)
   - `data/Papers Trainingdata/` — training manuscripts (PDF)
2. **Full-text features (GROBID TEI XML)** — produced by running GROBID over the
   manuscript PDFs in our companion repository
   `cos_replication_prediction3_ft` (GROBID 0.8.2). That repo also aggregates the
   per-paper TEI into the single table `01_data_raw_assembly.R` actually reads,
   `data/extracted_tei_all.rds` (keep `data/tei_xml/` too if you re-run the
   aggregation).
3. **Public reference data** — `sjr.csv`, `topfactor.csv`, `flora.csv` already
   ship here. `02_external_features.R` additionally pulls OpenAlex and World Bank
   data live over the network (set `OPENALEX_MAILTO` for the faster polite pool).

## Credentials

No API keys are stored in this repository. For the stages that call external
services, set environment variables in your shell (or `~/.Renviron` for R):

| Variable | Used by | For |
|---|---|---|
| `OPENALEX_MAILTO` | `02_external_features.R` | OpenAlex polite pool (optional but recommended) |
| `OPENAI_API_KEY` | `judge_llm_ensemble/run.py` | OpenAI GPT-5.5 calls |
| `GEMINI_API_KEY` | `judge_llm_ensemble/run.py` | Google Gemini calls |

The Anthropic / Claude vendor uses the local `claude` CLI's own authentication.

> ⚠️ **Cost:** re-running `judge_llm_ensemble/run.py` makes thousands of paid LLM
> API calls. The cached `ensemble_scores.csv` is shipped so you do not need to.
