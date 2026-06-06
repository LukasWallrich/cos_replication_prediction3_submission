# prior_shrinkage/ — the m2 "prior × evidence" predictor

Build code for the **m2** submission column. m2 decomposes replicability as
`posterior ∝ prior × evidence`:

- **Prior** — a language-model panel (4 personas × 3 model families: Claude Opus,
  GPT-5.5, Gemini 3.1 Pro) rates how plausible each hypothesis is *a priori*, from
  the substantive claim alone with all design/sample/statistics cues stripped.
- **Evidence** — a "power-for-observed-effect" shrinkage term in which the
  original effect size cancels analytically, leaving replication probability as a
  shrunken function of the original sample size (the original p-value is
  uninformative among already-significant published findings).

Full method and validation: `../reports/PRIOR_SHRINKAGE_REPORT.md`.

## Build order

| Step | Script | Produces |
|---|---|---|
| 0 | `elicit_priors.py` | LLM prior judgements → `data/r3_priors_mm/*.json` (and the training-set equivalents) |
| 1 | `1_evidence_shrinkage.R` | fits & freezes the shrinkage τ² (≈ 0.0195) on non-health rows → `data/shrinkage_feats.rds` |
| 2 | `2_prior_aggregation.R` | aggregates priors to `prior_3fam` on the 107 labelled health rows → `data/mm_combined.rds` |
| 3 | `3_build_predictor.R` | fits `logistic(y ~ prior_3fam + p_base)` and applies it to R3 → `../output/r3_prior_shrinkage_predictor.csv` |

```bash
# from the package root, with the challenge-derived inputs in place:
python prior_shrinkage/elicit_priors.py     # optional; needs scrubbed claims + API keys
Rscript prior_shrinkage/1_evidence_shrinkage.R
Rscript prior_shrinkage/2_prior_aggregation.R
Rscript prior_shrinkage/3_build_predictor.R
```

## Reproducibility & data

Like the m1 feature model (which needs `data/train_base.rds`) and the m3 crowd
(which needs the source claims), **m2's full rebuild needs challenge-derived
inputs that are not shipped** (see `../DATA.md`): the de-leaked claim text
(`data/r3_scrubbed.txt`), the original sample sizes / p-values (`data/r3_map.csv`),
the claim→id maps, and the replication **labels** used to fit the logistic
combination.

The **deployed m2 column is shipped** as `../output/r3_prior_shrinkage_predictor.csv`
(`claim_id, prior_3fam, p_base, pred`) and is what `pipeline/10_submission.R`
consumes — so the submission reproduces without re-running this build.

Keys (`OPENAI_API_KEY`, `GEMINI_API_KEY`) are read from the environment; the
Claude Opus priors are elicited the same way via the local `claude` CLI. No key
is stored in this repository.
