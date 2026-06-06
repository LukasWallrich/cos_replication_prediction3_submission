# bayesian_priors/

The **`bayes_pooled` comparison candidate** — a Bayesian logistic regression
whose coefficient priors are elicited from LLMs (per-feature location/scale,
pooled across model families). It is one of the candidates evaluated head-to-head
in `../pipeline/09_final_comparison.R` (which sources `bayes_llmprior_model.R`).

> ⚠️ This is **not** the submitted m2 column. m2 is the "prior × evidence"
> predictor built in `../prior_shrinkage/`. `bayes_pooled` was evaluated but
> **not selected** for the trio (it is the most shift-fragile candidate and is
> highly correlated with the feature model — see the submission report). It is
> kept here for transparency of the candidate comparison.

| File | What |
|---|---|
| `bayes_llmprior_model.R` | the model: `add_log1p_feature`, `fit/pred_bayes_logistic_llmprior`, `PREDICTORS_BAYES` |
| `extract_llm_priors.R` | parses the elicited coefficient priors into (location, scale) |
| `priors_claude48.json`, `priors_gpt55.json`, `priors_gemini31pro.json` | per-family elicited coefficient priors |
| `prior_predictor_table.csv` | the predictor list the priors map onto |
