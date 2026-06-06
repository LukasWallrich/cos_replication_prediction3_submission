# experiments/

Supporting analyses behind the modelling decisions. These are **not** on the
path that produces the submission (that is `../pipeline/` + `../prior_shrinkage/`
+ `../judge_llm_ensemble/`) — they are the validation and ablation work that
justified the final choices. Each consumes the built feature matrices
(`data/train_base.rds`, `data/test_base.rds`), which are challenge-derived and
not shipped (see `../DATA.md`), so they document the evidence rather than run
out of the box.

| Script | What it shows |
|---|---|
| `01_feature_stability.R` | Ranks features by predictiveness × train→test distributional stability — the basis for the curated, transfer-robust feature set used by m1. |
| `02_loro_validation.R` | Leave-one-round-out validation (train on FORRT+R1 → predict R2, and vice versa): the core cross-round generalization check. |
| `03_consensus_capping.R` | Tests shrinking predictions toward the base rate when all learners agree. **Rejected** — it lowers average Brier but compresses the column spread that best-of-three scoring rewards. (Backs the "we did not consensus-cap" point in the submission report.) |
| `05_health_gate.R` | A strict health-holdout veto over the slot candidates — guards against a column that collapses on the health domain. |
| `06_health_transfer_model.R` | The health-transfer study: identifies features that **anti-transfer** (flip sign) from psychology to large-N health — the reason m1 drops raw effect-magnitude terms. |
| `07_health_validation.R` | Domain-stratified CV grid (model × predictor set) with calibration/imputation diagnostics on the health holdout. |
| `04_stacking.R` | Stacked-ensemble framework (out-of-fold meta-learner). **Infrastructure** sourced by `05_health_gate` and `06_health_transfer_model`; not a standalone experiment. |

## What was dropped

To keep this lean, ablations that were superseded by `../pipeline/09_final_comparison.R`
(the single leak-free candidate comparison) were removed rather than shipped:
the RF-tuning and stacking-LORO series (`12c–12f`), the weighting experiments
(`14`, `15`), the legacy results-plotting script (`16`), and the helper used only
by them (`00b_pipeline_helpers.R`). Their conclusions — unweighted beats IPW, a
greedy uncorrelated ensemble is the right comparison baseline — are reflected in
`09_final_comparison.R` and the submission report.
