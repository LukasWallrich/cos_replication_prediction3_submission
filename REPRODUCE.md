# Reproduction guide

Three levels, from "confirm the wiring in 10 seconds" to "rebuild everything from
the challenge papers". Read DATA.md first if you go past Level 1 — Levels 2–3 need
data this package does not (and cannot) ship. Run all commands from the package
root unless noted.

---

## Level 1 — reassemble the submission from shipped artifacts  ✅

No challenge data required. Reproduces `output/submission.csv` exactly.

```bash
Rscript install.R core
Rscript pipeline/10_submission.R
```

Expected tail:

```
--- Trio R3 correlations (low = good diversification) ---
      m1    m2    m3
m1 1.000 0.639 0.565
m2 0.639 1.000 0.644
m3 0.565 0.644 1.000
  mean pairwise r = 0.616
Wrote: .../output/submission.csv (45 rows) ...
```

It joins the three deployed per-claim predictors
(`output/r3_elnet_plus_predictor.csv`, `output/r3_prior_shrinkage_predictor.csv`,
`judge_llm_ensemble/ensemble_scores.csv`) into the final `m1/m2/m3` table.

---

## Level 2 — re-run the headline analyses  📦

Needs the built feature matrices `data/train_base.rds` and `data/test_base.rds`
(from Level 3, Stage 2). Install the modelling packages: `Rscript install.R model`.

`09_final_comparison.R` runs the Bayesian candidate by default (`run_bayes=TRUE`),
which needs the pooled LLM prior in `priors_by_model.rds`. That file is **not
shipped** — generate it first from the shipped `bayesian_priors/priors_*.json`
(run from the package root so `here::here()` and the relative write both land at
the root):

```bash
Rscript bayesian_priors/extract_llm_priors.R   # → priors_by_model.rds (incl. $pooled)
Rscript pipeline/09_final_comparison.R         # leak-free candidate comparison →
                                            #   fits/calibrates m1, writes CIs,
                                            #   exports output/r3_elnet_plus_predictor.csv
```

(To skip the Bayesian candidate entirely, set `run_bayes = FALSE` in `CMP_CFG` at
the top of `09_final_comparison.R`; m1 still exports.)

Supporting validation/ablation analyses (not on the submission path) live in
`experiments/` and also consume `data/*_base.rds`:

```bash
Rscript experiments/02_loro_validation.R     # leave-one-round-out generalization
Rscript experiments/06_health_transfer_model.R  # the health anti-transfer finding
Rscript experiments/07_health_validation.R   # domain-stratified CV grid
# (see experiments/README.md for the rest)
```

After `09_final_comparison.R`, re-run Level 1 to fold the refreshed m1 into the submission.

---

## Level 3 — rebuild everything from the challenge data  📦🌍

### Stage 0 — data
Obtain and place the challenge data + full-text features as in DATA.md
(`data/Round{1,2,3}/`, `data/Papers Trainingdata/`, `data/tei_xml/`). Set
`OPENALEX_MAILTO` for OpenAlex's polite pool.

### Stage 1 — the language-model columns (optional; 💰 paid API calls)
Only needed to rebuild the LLM artifacts rather than use the shipped copies.

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY=...  GEMINI_API_KEY=...      # Claude uses the `claude` CLI's own auth

# m3 crowd → judge_llm_ensemble/ensemble_scores.csv
cd judge_llm_ensemble && python _build_manifest.py > manifest.jsonl && python run.py && python aggregate.py && cd ..

# structured LLM features that pipeline/03 parses.
# NOTE: the parsed values already ship as data/cache/llm_features.rds, so most
# users SKIP this. To rebuild from scratch you must reproduce the exact tagged
# runs 03 reads (run_combined.R's default tag is gem35flash, which 03 ignores):
cd llm_extraction
Rscript run_combined.R 0 --provider codex  --tag codex54hi_v3
Rscript run_combined.R 0 --provider openai --model gpt-5.4 --flex --tag openai54flex_v1
# (plus the *_addl_v1 and *_flora_v1 runs over their respective claim subsets —
#  see llm_extraction/README.md). Or point 03 at your own dirs via LLM_EXTRACTION_DIRS.
cd ..

# m2 priors → data/r3_priors_mm/ (needs the de-leaked claims; see prior_shrinkage/README.md)
python prior_shrinkage/elicit_priors.py
```

### Stage 2 — build the feature matrix

Default (offline) path — uses the shipped `data/cache/llm_features.rds`, no LLM API:
```bash
Rscript install.R                            # all groups
Rscript pipeline/01_data_raw_assembly.R      # → data/*_raw.rds
Rscript pipeline/02_external_features.R      # → data/external_features.rds
Rscript pipeline/04_base_dataset.R           # → data/train_base.rds, data/test_base.rds
```

Only if you re-derived the LLM caches in Stage 1 and want to re-parse them, insert
this before `04` (it overwrites the shipped cache):
```bash
Rscript pipeline/03_llm_features.R           # → data/cache/llm_features.rds
```

The shipped `data/cache/llm_features.rds` carries the 181 model-derived feature
columns per claim, so the default offline path is `01 → 02 → 04` (with the challenge
data in place for `01`/`02`).

### Stage 3 — build the three columns + assemble
```bash
# m1
Rscript bayesian_priors/extract_llm_priors.R    # → priors_by_model.rds (pooled prior for 09)
Rscript pipeline/09_final_comparison.R          # → output/r3_elnet_plus_predictor.csv
# m2  (see prior_shrinkage/README.md for inputs)
Rscript prior_shrinkage/1_evidence_shrinkage.R
Rscript prior_shrinkage/2_prior_aggregation.R
Rscript prior_shrinkage/3_build_predictor.R  # → output/r3_prior_shrinkage_predictor.csv
# m3 ships as judge_llm_ensemble/ensemble_scores.csv (Stage 1 to rebuild)
# assemble
Rscript pipeline/10_submission.R             # → output/submission.csv
```

---

## Notes & caveats

- **Determinism.** The R seed (`4267`) makes the feature/model stages
  reproducible. The language-model columns (m2, m3) are **not** bit-reproducible
  across re-runs; the shipped CSVs hold the exact submitted values.
- **`pROC`** is required by `09_final_comparison.R` (AUC). If `install.R model`
  reports it could not install, install it manually.
- **`FReD`** (self-reference features in `02`) is optional — wrapped in `tryCatch`.
- **Paths.** R scripts use `here::here()`, anchored by the `.here` file at the
  package root; the `prior_shrinkage/` and `llm_extraction/` scripts use relative
  paths and should be run from the package root and module folder respectively
  (see each module's README).
