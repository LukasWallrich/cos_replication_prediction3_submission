# Dependencies

Install with `Rscript install.R` (R) and `pip install -r requirements.txt` (Python).
See README.md for the per-stage breakdown — you do not need everything to run the
submission-assembly smoke test.

## Environment

| | Tested with |
|---|---|
| R | 4.5.3 (2026-03-11) |
| Python | 3.10+ (tested on 3.13) |

## R packages

Versions below are those used during development. Anything close should work;
the pin is provided for exact reproduction.

| Stage | Package | Version |
|---|---|---|
| core | here | 1.0.2 |
| core | dplyr | 1.2.0 |
| core | readr | 2.2.0 |
| core | tibble | 3.3.1 |
| core | tidyr | 1.3.2 |
| core | purrr | 1.2.1 |
| core | stringr | 1.6.0 |
| core | jsonlite | 2.0.0 |
| model | glmnet | 4.1.10 |
| model | mgcv | 1.9.4 |
| model | ranger | 0.18.0 |
| model | xgboost | 3.2.1.1 |
| model | lightgbm | 4.6.0 |
| model | dbarts | 0.9.33 |
| model | rstanarm | 2.32.2 |
| model | mice | 3.19.0 |
| model | miceadds | 3.19.16 |
| model | pROC | ≥ 1.18 (AUC in 09_final_comparison.R) |
| model | ggplot2 | 4.0.2 |
| model | ggrepel | 0.9.7 |
| model | scales | 1.4.0 |
| model | corrplot | 0.95 |
| data | openalexR | 3.0.1 |
| data | wbstats | 1.1 |
| data | countrycode | 1.8.0 |
| data | quanteda | 4.4 |
| data | quanteda.textstats | 0.97.2 |
| data | sentimentr | 2.9.0 |
| data | BayesFactor | 0.9.12.4.8 |
| data | scrutiny | 0.6.1 |
| data | FReD | optional (wrapped in tryCatch; self-reference features) |

## Python packages (LLM ensemble)

| Package | Tested version | Notes |
|---|---|---|
| openai | 2.36.0 | OpenAI GPT-5.5 calls |
| google-genai | 1.70.0 | Google Gemini 3.1 Pro calls |

The Anthropic / Claude model is invoked through the local `claude` CLI (Claude
Code), which handles its own authentication — no Python package or API key is
needed for that vendor.
