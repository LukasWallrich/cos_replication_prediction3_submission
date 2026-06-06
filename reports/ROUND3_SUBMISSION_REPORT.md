<h1>COS Replication Prediction Challenge — Round&nbsp;3</h1>
<p class="sub"><strong>Submission method report</strong></p>
<p class="sub">Team: Jessica Röseler, Lukas Wallrich, Lukas Röseler</p>
<p class="small">ORCID — Jessica Röseler <a href="https://orcid.org/0009-0007-7544-393X">0009-0007-7544-393X</a> · Lukas Wallrich <a href="https://orcid.org/0000-0003-2121-5177">0000-0003-2121-5177</a> · Lukas Röseler <a href="https://orcid.org/0000-0002-6446-1901">0000-0002-6446-1901</a></p>
<p class="sub">5 June 2026</p>

::: {.box .key}
#### Summary

The test set is 45 health claims, a domain in which we have almost no labelled training data. Scoring counts only the **best of three** submitted prediction columns, which makes the right objective three *decorrelated* bets rather than three columns that each minimise average error. We submit three individually competitive predictors built from independent machinery, so they are only weakly correlated with one another. All three cover all 45 claims (no missing values).

| Column | What it is | Brier | 95% CI |
|---|---|---:|---:|
| **m1 · Feature model** | regularised model trained on labelled Training and Round 1+2 outcomes | 0.211 | [0.143, 0.273] |
| **m2 · Prior × evidence** | substantive plausibility prior combined with a statistical evidence term | 0.213 | [0.161, 0.259] |
| **m3 · LLM crowd** | zero-parameter language-model synthetic-crowd probability | 0.222 | [0.169, 0.267] |

<p class="small">Brier on the test-size health analog (out-of-fold), temperature-calibrated m1 and m2, raw m3; 95% CIs from a cluster bootstrap that resamples studies. All three beat the base-rate Brier (~0.249); the estimated best-of-three score is <strong>≈ 0.20</strong>, below any single column's average.</p>
:::

## 1 · The task and the scoring rule

<p class="lead">Predict, for each of 45 claims, the probability that it replicates; loss is the Brier score.</p>

Two features of the setup shape the whole approach:

- **Only the best column counts.** We submit three columns and are scored on the lowest of the three Brier scores. The target is therefore the expected *minimum* of the three, not their average. For a fixed average accuracy, the expected minimum falls as the three columns become less correlated, so decorrelation directly improves the score. Two columns built from the same family of signals waste a slot.
- **At 45 claims, small accuracy differences are noise.** Every pairwise difference in Brier between our candidates sits well inside its bootstrap confidence interval (the marginal CIs above span ≈ 0.11–0.13, against gaps of ≈ 0.01 between candidates). No predictor is reliably more accurate than another. The levers that are real are therefore (a) the correlation structure between the three columns and (b) robustness to the domain shift from our mostly-psychology training data to a health test set (and not the ranking of the central scores).

The practical consequence: a third column with a slightly higher average error but *independent* mistakes is almost free, because its higher average is only "paid" in cases where it was never going to be our best column anyway. We optimise for spread, not for the single best number.

## 2 · The three predictors

### m1 · Feature model — learns from past replication outcomes

An elastic-net logistic regression (`glmnet`, regularisation strength at the one-standard-error rule), median-aggregated over cross-validation folds and temperature-calibrated, trained on the labelled Training and Round 1+2 replication outcomes. The feature set is chosen to *transfer across domains* rather than to fit psychology — in particular it drops raw effect-magnitude terms, whose relationship to replication reverses between small-sample psychology and large-sample health. It is the only column that learns from outcome labels; because the training data is almost entirely non-health it is best understood as a *general* replication-feature model applied to health, and in validation removing all health rows from training costs it almost nothing.

**Key features and their expected direction** (the model learns the weights):

- **Original sample size (log)** — Larger samples are expected to replicate more often, which is consistent with the replication literature, and log sample size (n_o_log) is one of the features the model relies on most.
- **Directional effect-size and Bayes-factor evidence** — es_directional is a 0/1 flag for whether the reported effect size is of a type that carries a direction (correlation, standardized and unstandardized regression, between- and within-subject standardized mean differences, odds-ratio family, raw differences, test-statistic-only, plus phi, Wilcoxon r, and Cohen's h). It marks the kind of effect size, not its sign or its magnitude. log_bf10_h is a Bayes-factor approximation (BIC/JZS, computed from a t-proxy derived from the harmonized effect size and N). It is sign-free because it is built on t squared, so it captures the strength of the evidence rather than its direction, and stronger evidence is expected to replicate more often. The bare undirected effect-size magnitude is left out, since on its own it probably transfers poorly across domains.
- **Multiplicity** — Parsed from the full text via GROBID. n_f_tests_tei is the number of reported F-tests detected in the text, used as a rough proxy for how much testing the paper does. has_multiplicity_correction_tei is a 0/1 flag for whether the text mentions a multiple-comparison correction (Bonferroni, Holm, Šidák, Tukey, false discovery rate or Benjamini, family-wise, and similar). More F-tests is treated as a weak negative signal, and an explicit correction is expected to be associated with higher replicability. Note that only F-tests are counted here, not all test statistics.
- **LLM-derived features** — llm_perceived_surprisingness is an LLM rating of how surprising the finding is, where more surprising is expected to replicate less. llm_sample_adequacy is a composite built from the LLM's sample-size rating and a power flag rather than a single raw rating, where higher adequacy is expected to replicate more. llm_within_paper_rep is a yes or no flag for whether the paper reports a within-paper (internal) replication. llm_is_intervention flags an intervention study as opposed to a non-intervention or observational design.
- **First-author and team track record** — first_author_productivity_log1p is the first author's works per active year (a productivity rate), on a log1p scale. team_works_top25_median is the median cumulative publication count of the most productive quartile of the author team, which is an output count rather than a rate.

<p class="files">Source: <code>pipeline/06_models.R</code> (feature sets &amp; model definitions) · <code>pipeline/03_llm_features.R</code> (structured LLM ratings) · <code>pipeline/09_final_comparison.R</code> (fits, calibrates and scores the model; exports the deployed per-claim column to <code>output/r3_elnet_plus_predictor.csv</code>) · full-text features (tests / multiplicity) extracted via GROBID in the companion repository.</p>

### m2 · Prior × evidence — plausibility combined with statistical strength

A Bayesian-flavoured decomposition, `posterior ∝ prior × evidence`. The idea is to use a language model for the part it is good at (judging whether a hypothesis is *a priori* plausible) and to handle the statistical evidence with an explicit, transparent formula.

- **Prior.** A language-model panel estimates the probability that the hypothesis is true in the population from the *substance of the claim alone*. The claim is first surgically scrubbed of every cue to method, design and sample size (e.g. "randomized", "nationally representative") so the prior cannot peek at the statistics. The same 12 panel cells, 4 reviewer personas (skeptical methodologist, health-behavioural field expert, Bayesian forecaster, generalist critical reader) × 3 model families (**Claude Opus**, **GPT-5.5**, **Gemini 3.1 Pro**), each rate the scrubbed claim; the prior is their mean. On its own this substantive prior already predicts replication on health claims (AUC ≈ 0.78).
- **Evidence.** A statistical term derived from the original study. In principle this could draw on both the original *p*-value and the sample size, but in the large-sample health test set *p* barely varies: almost everything there is decisively significant, and its distribution looks nothing like the psychology data that dominates our training set. Sample size, by contrast, varies meaningfully across both domains, so we build the evidence term from **sample size** alone, with larger original studies giving a higher predicted replication probability.
- **Combination.** Prior and evidence are combined by a simple logistic fit on the labelled health rows.

It shares no machinery with the feature model and only the broad "ask a language model" idea with the LLM crowd. It has a different input (scrubbed substance, no statistics) and a different question (is the hypothesis *true*, not will it *replicate*), which is why it ends up genuinely decorrelated from both.

<p class="files">Method &amp; validation: <code>PRIOR_SHRINKAGE_REPORT.md</code> · deployed per-claim predictor: <code>output/r3_prior_shrinkage_predictor.csv</code>.</p>

### m3 · LLM crowd — a zero-parameter synthetic crowd

A synthetic crowd reads each claim together with the original study's metadata (title, year, sample size, reported effect size and *p*-value) and rates how likely the underlying effect is real — the same 12 panel cells as the prior (4 personas × 3 model families). Each judgement is a calibrated seven-point bucket (from "almost certainly will not replicate", ≤5%, to "almost certainly will", ≥95%), mapped to its midpoint and averaged.

The crowd's "effect is real" probability is then converted to a replication probability by a fixed, pre-specified rule encoding the challenge's power assumption (80% power, α = 0.05): a real effect re-emerges with probability 0.80, a null gives a false positive with probability 0.05. Because no parameter is fitted to outcomes, this column is a genuine out-of-sample baseline with no overfitting risk; it is the best-calibrated of the three and, being independent of the training folds, essentially unaffected by the domain shift.

<p class="files">Models &amp; personas: <code>judge_llm_ensemble/config.py</code>, <code>judge_llm_ensemble/personas/</code> · prompt &amp; rating scale: <code>judge_llm_ensemble/prompt_template.txt</code> · run / aggregation: <code>judge_llm_ensemble/run.py</code>, <code>judge_llm_ensemble/aggregate.py</code> → <code>judge_llm_ensemble/ensemble_scores.csv</code>.</p>

## 3 · Scores

Brier scores by predictor, temperature-calibrated (for m1 and m2), across the target set and two domain-shift simulations. **In-domain** is an optimistic out-of-fold estimate; the **shift** columns hold out related data to mimic an unseen health domain. "Shift B" is the ~45-claim, test-size set, and "health set" is a broader ~107-claim set. Both shift B and the health set are drawn from the original training and R1/R2 data, and both contain only health-related papers. A paper counts as health-related if any of its descriptors (tags, journal name, discipline, topic labels, or keywords) match common health terms such as health, medicine, clinical, epidemiology, pharmacology, nursing, oncology, or cardiology. Shift B is the health set restricted to R1/R2 claims only, which brings it down from about 107 to about 45 claims. The 95% CI (on the in-domain Brier) is from a cluster bootstrap that resamples whole studies, so non-independent claims from the same paper do not inflate precision.

| Predictor | in-domain | 95% CI | shift A | shift B | health set | AUC | calib. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| m1 · Feature model | .211 | [.143,.273] | .220 | .213 | .211 | .745 | .122 |
| m2 · Prior × evidence | .213 | [.161,.259] | .223 | — | .227 | .742 | .142 |
| m3 · LLM crowd | .222 | [.169,.267] | .217 | .217 | .222 | .744 | .087 |

All three sit just below the base-rate Brier (~0.249) and within each other's confidence interval. The LLM crowd is the most shift-invariant and best-calibrated; the feature model is the strongest in-domain and the most decisive; prior × evidence trades a little calibration for full independence.

### Correlation between the three columns (on the 45 test claims)

This is the lever that drives the best-of-three score:

|  | Feature model | Prior × evidence | LLM crowd |
|---|---:|---:|---:|
| Feature model | — | .64 | .56 |
| Prior × evidence | .64 | — | .64 |
| LLM crowd | .56 | .64 | — |

Mean pairwise correlation ≈ **0.62**. For comparison, any two variants drawn from the same feature-model family correlate at 0.78 or higher — effectively a single bet. The lower the mean correlation, the more dispersed the three column scores and the lower the expected minimum. The estimated best-of-three score lands at **≈ 0.20**, below any single column's average.

## 4 · Why these three

Because the candidates are statistically tied on accuracy and the test set is domain-shifted, we selected on **independence** and **robustness to the shift** rather than on the ranking of central scores:

- We keep the two most independent predictors (prior × evidence and the LLM crowd) and add one competitive column anchored in the labelled training data (the feature model). This is the most dispersed competitive triple available to us.
- m1 and m2 predictions are temperature-calibrated, each to the sharpness it honestly supports. For the feature model this means sharpening, since it was underconfident (T below 1); for the prior column it means softening toward the base rate, since it was overconfident (T above 1). What we did not do is blend the columns toward each other or add any shrinkage on top of this calibration, because that would compress the spread between them. Under best-of-three the score is the lowest of the three Brier values, so a column that ends up wrong is simply not the one that counts and does not push the score up, and the spread is what gives us the chance that one column lands low. That spread comes from the columns disagreeing about which claims replicate, not from any one of them being more extreme, so calibrating m2 toward the base rate does not work against it. The price of keeping the columns distinct is that the two independent columns sit up to about 0.01 above the feature model in average Brier (see §1).

## 5 · Data and preprocessing

The labelled training data combines the challenge **Training** set with the **Round&nbsp;1 and Round&nbsp;2** replication outcomes, augmented by a small set of large-sample health replications we coded from the FORRT Library of Replication Attempts (FLoRA) to add coverage in the test domain. Before modelling we apply a few principled filters, all aimed at keeping the training signal clean and comparable to the test task:

- **Retracted originals** are dropped (six rows in training, none in the test set).
- **Author-overlap replications** — where the replicating team overlaps with the original authors — are excluded, because same-team replications succeed at a much higher rate and are not representative of the independent-replication question the challenge asks; rows where overlap is unknown are kept.
- **Boyce/Soto multi-study projects** are flagged and removed so that a single large coordinated effort cannot dominate the outcome distribution.
- **Non-implied claims** (those the LLM screen judged not to be genuinely tested by the paper) and rows without a usable replication outcome are dropped.

Missing feature values are filled by **median imputation using training-fold medians only**, so no information leaks from held-out data. We implemented and tested multiple imputation (MICE), but it did not improve out-of-fold accuracy over the simpler median fill, so the deployed pipeline uses the median.

For the 45 test claims we additionally **completed missing original-study metadata by hand** (e.g. sample sizes, effect sizes and *p*-values that the automated extraction did not capture), so that every claim carries the complete set of inputs each of the three predictors needs in order to score it.

<p class="files">Filters &amp; feature matrix: <code>pipeline/04_base_dataset.R</code> · imputation recipes: <code>pipeline/05_imputation.R</code> · health add-on: <code>data/flora_health_addon.csv</code>. What is and is not reproducible from the shared data is documented in <code>DATA.md</code>.</p>

## 6 · Limitations and potential biases

::: {.box .warn}
**The headline caveat:** at 45 claims, none of these predictors is distinguishable from the others (or, by much, from the base rate) within sampling noise. These numbers should be read as well-motivated estimates, not precise claims.
:::

- **Domain shift.** Our labelled training data is mostly non-health and mostly small-sample psychology, whereas the test set is entirely large-sample health. The feature model in particular is extrapolating outside the regime it was trained on. We mitigated this by curating features for cross-domain transfer and by explicitly excluding effect-magnitude terms that reverse sign across domains, but residual shift risk remains.
- **Calibration is validated on a proxy, not the test set.** Because Round 3 has no labels, all accuracy, AUC and calibration figures are computed on a labelled health proxy set, not on the 45 test claims. Calibration that holds on the proxy may drift on the test set.
- **Language-model biases.** All three columns use language models, though to different degrees: m2 and m3 are built around language-model judgements as their core signal, while m1 only adds a subset of language-model-extracted features (perceived surprisingness, sample adequacy, within-paper replication, intervention flag) alongside many non-language-model predictors. Language models can carry training-data recency and popularity biases (well-known findings may be rated as more plausible) and may have seen some of these published papers. The concern is therefore largest for m2 and m3, where the language-model judgement is the signal itself; in m1 these features are a minority of inputs and carry limited weight. For the prior (m2) we reduce, but cannot eliminate, leakage by stripping statistical and design cues and by recording a recognition flag; the substantive-prior result holds when recognised claims are excluded. (The automatically parsed full-text features inherit some OCR/parsing noise and are deliberately given modest weight.)
- **A modelling choice tuned to the scoring rule.** We optimised for the expected best of three columns, not for the accuracy of any single column. Under a conventional single-column metric this trio would be a slightly weaker choice than a shrunk consensus; it is deliberately specialised to this challenge's best-of-three rule.

## Appendix · reproduction

The full reproducible package (code, this report, and the artifacts needed to regenerate the deliverable) is at <a href="https://github.com/LukasWallrich/cos_replication_prediction3_submission">github.com/LukasWallrich/cos_replication_prediction3_submission</a>. What regenerates from the shared data alone, and what needs the restricted challenge data, is set out in <code>DATA.md</code>.

<p class="files"><code>pipeline/09_final_comparison.R</code> — fits and scores every candidate from scratch; produces the comparison metrics, the bootstrap CIs and the deployed ElNet+ column.<br>
<code>pipeline/10_submission.R</code> — assembles the three deployed per-claim predictors into the deliverable <code>output/submission.csv</code> (claim_id, m1, m2, m3) and writes <code>output/submission_diagnostics.csv</code>.<br>
<code>PRIOR_SHRINKAGE_REPORT.md</code> — full method and validation for the prior × evidence column.<br>
Full-text feature extraction (PDF → GROBID structured XML) lives in the companion repository.</p>
