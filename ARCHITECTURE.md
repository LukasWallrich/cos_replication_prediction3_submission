# Architecture

How the pipeline fits together: the data-flow DAG, what each script does, and
which stages need the (non-redistributable) challenge data.

Every R script sources `pipeline/00_packages_config.R` first, which loads
packages, sets the global seed (`4267`) and fold configuration, and defines the
shared helpers (`brier_score`, `normalize_doi`, `%||%`). Paths are resolved with
`here::here()`; the `.here` file at the package root anchors them to this folder.

Legend:  📦 needs challenge data (not shipped) · 🌍 needs internet/API ·
✅ runs from shipped artifacts.

---

## The three submission columns

```
  m1 Feature model   pipeline/  (data build → 09_final_comparison.R)        ─┐
  m2 Prior × evidence  prior_shrinkage/                                   ─┼─► pipeline/10_submission.R
  m3 LLM crowd         judge_llm_ensemble/                                ─┘     joins on claim_id
                                                                                  → output/submission.csv
```

Each column ships its deployed per-claim predictor (`output/r3_elnet_plus_predictor.csv`,
`output/r3_prior_shrinkage_predictor.csv`, `judge_llm_ensemble/ensemble_scores.csv`),
so `10_submission.R` ✅ runs without the challenge data.

---

## pipeline/ — the main path

### Stage 0 — configuration
| Script | Purpose |
|---|---|
| `00_packages_config.R` | packages, `CONFIG` (seed/folds/group var), shared utilities. Source first. |

### Stage 1 — data construction  📦
Builds the feature matrix from the challenge data. Run once; outputs cached as `.rds`.
| Script | Reads | Writes |
|---|---|---|
| `01_data_raw_assembly.R` 📦 | challenge round CSVs + GROBID TEI full-text | `data/train_raw.rds`, `data/test_raw.rds` |
| `02_external_features.R` 🌍 | OpenAlex API; `data/external/{sjr,topfactor}.csv`; World Bank GDP | `data/external_features.rds`, `data/cache/*.rds` |
| `03_llm_features.R` | structured LLM rating JSON (from `llm_extraction/`) | `data/cache/llm_features.rds` |
| `04_base_dataset.R` 📦 | the three above + `judge_llm_ensemble/ensemble_scores.csv` | `data/train_base.rds`, `data/test_base.rds` |

`train_base.rds` / `test_base.rds` are the hub (~570 features/claim); everything downstream consumes them.

### Stage 2 — model infrastructure (sourced, not run alone)
| Script | Provides |
|---|---|
| `05_imputation.R` | leak-free median / MICE recipes (per fold; no outcome imputation) |
| `06_models.R` | base learners, `MODEL_GRID`, named `PREDICTOR_SETS` (incl. `transferable_lean_plus`, the m1 set) |
| `07_calibration.R` | temperature / Platt / isotonic calibration |

### Stage 3 — evaluation & submission
| Script | Role | Needs |
|---|---|---|
| `09_final_comparison.R` | leak-free head-to-head of all slot candidates; fits & calibrates **m1 (ElNet+)**; writes bootstrap CIs; exports `output/r3_elnet_plus_predictor.csv`. Sources `bayesian_priors/bayes_llmprior_model.R` for the `bayes_pooled` comparison candidate. | 📦 `data/*_base.rds` |
| `08_ci_brier_cluster_bootstrap.R` | study-clustered bootstrap CIs (sourced by `09_final_comparison.R`) | — |
| `10_submission.R` | **assembles the deliverable**: joins the three deployed predictors → `output/submission.csv` (+ diagnostics) | ✅ shipped artifacts |

## prior_shrinkage/ — m2  (its own README)
`elicit_priors.py` 🌍📦 → `1_evidence_shrinkage.R` 📦 → `2_prior_aggregation.R` 📦 →
`3_build_predictor.R` 📦 → `output/r3_prior_shrinkage_predictor.csv`. Prior × a
power-for-observed-effect shrinkage term; full rebuild needs labels/stats/claim
text (📦), the deployed column is shipped.

## judge_llm_ensemble/ — m3  (its own README)  🌍
`_build_manifest.py` → `run.py` (Claude Opus, GPT-5.5, Gemini 3.1 Pro × 4 personas;
cached/resumable) → `aggregate.py` → `ensemble_scores.csv` (shipped). Re-running
the API calls is optional.

## llm_extraction/ — provenance of the structured LLM features  🌍📦  (its own README)
Extracts claim/paper fields from the PDFs (via GROBID TEI) that
`pipeline/03_llm_features.R` parses. A Stage-1 build step; raw outputs and the
manifest are challenge-derived and not shipped.

## experiments/ — supporting analyses  📦  (its own README)
Not on the submission path. `01_feature_stability`, `02_loro_validation`,
`03_consensus_capping`, `05_health_gate`, `06_health_transfer_model`,
`07_health_validation`, and the `04_stacking` infra they use. They document the
validation/ablation work (feature transfer, LORO generalization, the health
anti-transfer finding, why consensus-capping was rejected).

## bayesian_priors/ — the bayes_pooled comparison candidate  (its own README)
Evaluated by `09_final_comparison.R`, **not** submitted. Distinct from m2.

## Reproducibility notes

- **Seed** `4267` (set in `00_packages_config.R`); folds stratified by original-study
  DOI (`doi_o`) so same-paper claims never split across train/test.
- **No leakage**: imputation and calibration fit per fold on training rows only;
  `09_final_comparison.R` recomputes from scratch rather than reading cached folds.
- **LLM non-determinism**: the m2/m3 language-model columns are not bit-reproducible
  across re-runs; the shipped CSVs hold the exact submitted values.
