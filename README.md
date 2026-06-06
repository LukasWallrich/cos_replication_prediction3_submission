# COS Predicting Replicability Challenge — Round 3

Reproducible code package for our Round 3 submission.

**Team:** Jessica Röseler, Lukas Wallrich, Lukas Röseler

---

## What this is

The challenge asks us to predict, for each of 45 health-science claims, the
probability that it would replicate (scored by Brier loss). Each team submits up
to **three** prediction columns and is scored on the **best** of the three. That
rule makes the objective three *decorrelated* bets, not three accurate-on-average
columns — see `reports/ROUND3_SUBMISSION_REPORT.pdf` for the full rationale.

Our submission (`output/submission.csv`) has three columns, each built by its own
module:

| Column | Predictor | Built by |
|---|---|---|
| `m1` | **Feature model (ElNet+)** | `pipeline/` (elastic-net on labelled Round 1+2 outcomes; produced by `09_final_comparison.R`) |
| `m2` | **Prior × evidence** | `prior_shrinkage/` (LLM plausibility prior × a sample-size shrinkage term) |
| `m3` | **LLM crowd** | `judge_llm_ensemble/` (zero-parameter synthetic crowd of 3 models × 4 personas) |

They are built from independent machinery and are only weakly correlated
(mean pairwise r ≈ 0.62), which is what lowers the expected best-of-three score.
`pipeline/10_submission.R` joins the three deployed per-claim predictors into the
deliverable.

## Repository layout

```
reproducible_package/
├── README.md            ← you are here
├── REPRODUCE.md         step-by-step reproduction guide (3 levels)
├── ARCHITECTURE.md      data-flow DAG + what every script does
├── DATA.md              data sources: what's included, excluded, how to obtain
├── DEPENDENCIES.md      R/Python package version table
├── install.R            R package installer (staged: core / model / data)
├── requirements.txt     Python deps · LICENSE · .gitignore
│
│   ── the three submission columns ──
├── pipeline/            main path: data build → m1 feature model → assembly
├── prior_shrinkage/     m2: the prior × evidence predictor
├── judge_llm_ensemble/  m3: the LLM synthetic-crowd ensemble
│
│   ── supporting ──
├── experiments/         validation & ablations behind the modelling choices
├── llm_extraction/      how the structured LLM features (used by pipeline/03) were extracted
├── bayesian_priors/     the bayes_pooled *comparison* candidate (evaluated, not submitted)
│
├── data/external/       public reference data (SJR, TOP Factor, FLoRA)
├── output/              our shareable result artifacts (the three predictors, submission)
└── reports/             method reports (submission report, prior method, design plan)
```

Each module folder has its own README. **`pipeline/` + `prior_shrinkage/` +
`judge_llm_ensemble/` are the reproduction path; everything else is supporting.**

## Quick start

### 1. Reproduce the final submission from our artifacts (≈ 10 seconds)

Needs only the *core* R packages and the predictor artifacts in `output/` and
`judge_llm_ensemble/` — no challenge data required.

```bash
Rscript install.R core
Rscript pipeline/10_submission.R
```

This re-assembles `output/submission.csv` from the three deployed per-claim
predictors and prints the trio's cross-correlations (m1–m2 .64, m1–m3 .56,
m2–m3 .64). Fastest way to confirm the package is wired up correctly.

### 2. Re-run the model comparison that produces m1 and the confidence intervals

Needs the *model* packages and the built feature matrices (`data/*_base.rds`),
which are challenge-derived and not shipped (see DATA.md):

```bash
Rscript install.R model
Rscript pipeline/09_final_comparison.R
```

### 3. Rebuild everything from the challenge data

See **REPRODUCE.md** for the full ordered pipeline across all three columns,
including how to obtain the challenge data and full-text features.

## Important: data availability

Under the challenge Terms & Conditions we **cannot redistribute** the source
papers, claim text, or labels. This package contains:

- ✅ all code,
- ✅ our own output artifacts (the three per-claim predictors, the submission, the LLM ensemble scores),
- ✅ public third-party reference data (SJR, TOP Factor, FLoRA),
- ❌ **no** challenge papers / PDFs / TEI XML / claim-text datasets / labels / feature matrices derived from them.

Each of m1, m2, and m3 ships its **deployed per-claim predictor** so the
submission reproduces without the raw data; full *rebuilds* of any column need
the challenge data, exactly as marked in REPRODUCE.md and ARCHITECTURE.md. See
**DATA.md** for specifics.

## Method reports

- `reports/ROUND3_SUBMISSION_REPORT.pdf` — the submitted method description (scores, CIs, biases, limitations).
- `reports/PRIOR_SHRINKAGE_REPORT.md` — derivation and validation of the `m2` prior × evidence column.
- `reports/HEALTH_FEATURE_MODEL.html` — the health-domain transfer investigation behind the m1 feature choices.

## Licence

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — reuse freely with attribution; see `LICENSE`. This covers the material we authored (code, reports, the data tables we created). Redistributed third-party reference data (`data/external/`) keeps its own upstream terms, and the challenge manuscripts, claim text and labels are not included (see `DATA.md`).
