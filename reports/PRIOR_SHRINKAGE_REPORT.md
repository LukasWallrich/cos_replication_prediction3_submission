# Prior-elicitation + power-for-observed shrinkage — POC report

**Date:** 2026-06-03 · **Branch of work:** experiment/fulltext-widening-report → main

## TL;DR

We tested a Bayesian-flavoured idea: have an LLM estimate the **prior plausibility** that a
claim is true (from the *substance* of the claim, with all statistics stripped), then combine it
with an **evidence** term derived from the original study's statistics. Findings:

1. **An LLM substantive-plausibility prior predicts replication.** On the R3-analog challenge-health
   set (R1/R2 health, n=47) it reaches **AUC ≈ 0.78**, robust across three model families
   (Opus / GPT-5.5 / Gemini-3.1, which agree at r ≈ 0.87). On the broader health set (n=107) it is
   weaker (AUC ≈ 0.65).
2. **Statistical significance does *not* predict replication; sample size does.** Among published,
   already-significant claims the original *z* is uninformative (range-restricted by selection). The
   live evidence axis is **sample size**, via the winner's-curse selection funnel
   (effect size and n are r ≈ −0.79 here).
3. **The right "Bayes factor" is a predictive-replication probability, not a nonzero-effect BF.**
   Assuming replicators power to 80% for the *observed* effect fixes the replication n, and the
   predictive replication probability becomes `Φ(2.80·g(n) − 1.96)` with shrinkage
   `g(n) = n·τ²/(n·τ²+1)` — the effect size *cancels*, analytically explaining why n predicts and
   significance does not. This calibrated term beats a plain `log n` logistic on Brier.
4. **Deployment:** `prior + shrinkage` ties the best existing slot (raw LLM) on challenge-health
   (Brier ≈ 0.213) but is **decorrelated** from it (r ≈ 0.64), whereas the old m2 bridge was
   r ≈ 0.95 with m3 (a near-duplicate). Under the challenge's **best-column** scoring, replacing the
   redundant m2 with `prior + shrinkage` lowers the pipeline's E[min] from **0.2129 → ≈0.205**
   (within noise, low-downside). **We swapped m2 → prior+shrinkage in `output/submission.csv`.**

## The idea

Replicability ≈ posterior probability the effect is real (Ioannidis/PPV): `posterior ∝ prior × evidence`.
The novel part is using an LLM to estimate the **prior** from claim substance alone — the part LLMs are
good at — and handling the **evidence** with explicit statistics.

## Method

1. **Derive the substantive claim (full text).** `claim_text` in the data is a raw statistical excerpt
   (e.g. *"the association remained nonsignificant [HR=0.93]"*) — it often does **not** state the
   hypothesis. Reconstructing it from abstract + the bracketed table tag alone is unreliable (it
   produced a confidently *wrong* claim on a spot-check). So each claim is derived by **Sonnet
   sub-agents reading the full paper**, matching the specific coefficient to the authors' stated
   hypothesis, as one clear directional prediction. ~60% of claims required the body, not just the abstract.
2. **De-leak (surgical).** Remove only tokens that telegraph **method/design or sample size**
   (e.g. "randomized", "voucher", "nationally representative") while keeping all substance
   (hypertension, fifth-grade children, the constructs). Design that is *intrinsic* to the construct
   is not strippable and is left in (it is legitimate signal).
3. **Prior elicitation.** 4 personas (bayesian / field / generalist / skeptic) × 3 model families rate
   P(hypothesis true in population) seeing **only** the scrubbed claim — no stats, no abstract — plus a
   recognition flag (contamination check). `prior_3fam` = mean of the 12 judgements.
4. **Evidence (shrinkage predictive-replication probability).** `z` from p, `d = z/√n`. Under a
   power-for-observed-effect replication design, `n_r = (2.80/d)²` and
   `p_base = Φ(2.80·g(n) − 1.96)`, `g(n) = n·τ²/(n·τ²+1)`. τ² fit on **non-health** rows, frozen.
5. **Combination.** `glm(replicated ~ prior_3fam + p_base)` fit on the 107 labelled health rows;
   applied to the 45 R3 claims.

## Validation (what we learned, and what didn't work)

| Test | Result |
|---|---|
| Prior alone, R3-analog (n=47) | AUC 0.78; LOO Brier 0.211 vs base 0.249; holds excl. recognized |
| Prior alone, broad health (n=107) | AUC 0.65 — the 47 was optimistic |
| Unit-information / BIC Bayes factor (`z²/2 − ½ln n`) | **Failed** (AUC 0.53) — Lindley penalty *inverts* the n signal; wrong question |
| de-censored z (parsed CIs/SEs) | AUC 0.62 — worse than `log n` (0.71); z⊥n at r=0.09 (selection funnel) |
| effect × sample (`d×n`) product | marginally best single stat (AUC 0.66) but no gain over `log n` once prior is in |
| Model families (GPT-5.5, Gemini) | agree r≈0.87; no Brier gain on R3-analog; *raise* redundancy with the ensemble (0.69→0.80) |
| Input-swap (derived claim → production ensemble) | **null** (corr A,B = 0.98; no Brier change) |
| Shrinkage `p_base`, frozen non-health→health (n=89) | Brier 0.221 vs logistic-logn 0.227 (better calibration); +het term lifts AUC 0.68→0.76 |
| `prior + p_base`, R3-analog (n=46) | LOO Brier 0.2122 — best config found |

## Submission impact (best-column scoring)

The challenge scores the **best of your three columns** (whole-column Brier), estimated by the
pipeline's `emin_of` (bootstrap min over slots). On challenge-health (n=43):

| slot | whole-column Brier |
|---|---|
| m1 (feature ensemble: RF/GAM/ElNet) | 0.2390 |
| m2 (old bridge 0.5·m1+0.5·m3) | 0.2225 |
| m3 (raw LLM) | 0.2129 |
| **prior + shrinkage (honest LOO)** | **0.2136** (ties m3) |

| trio | best-column E[min] |
|---|---|
| {m1, m2(bridge), m3} (old) | 0.2129 |
| **{m1, prior+shrinkage, m3} (new)** | **0.2048** |

The old m2 is r≈0.95 with m3 (near-duplicate, rarely the best column). The prior+shrinkage column
ties m3 in accuracy but is decorrelated (r≈0.64), so it is a genuinely distinct strong column.
Swapping it in is **low-downside** under best-column scoring (m1 and m3 are retained).

**Change applied:** `output/submission.csv` column **m2 is now the prior+shrinkage predictor**
(m1, m3 unchanged). Reproduce with `Rscript apply_prior_shrinkage_swap.R`.

## Caveats

- The R3-analog gain (~0.008 Brier on n=43) is **within noise**.
- **Calibration risk:** a cross-set fit (FORRT→challenge) gave Brier 0.27; the in-distribution and
  pooled-107 fits give ≈0.214. The deployed combination is fit on the pooled 107 (includes challenge
  rows), so R3 calibration should sit near 0.214 — but this is unverifiable without R3 truth.
- The prior is ~0.7–0.8 correlated with the existing LLM ensemble: it is a *cleaner route to the same
  substantive signal*, not an orthogonal new signal.

## Files (committed)

- `output/submission.csv` — final m1/m2/m3 (m2 = prior+shrinkage).
- `output/r3_prior_shrinkage_predictor.csv` — per-claim `prior_3fam`, `p_base`, `pred` (the Bayesian approach).
- `apply_prior_shrinkage_swap.R` — reproduces the m2 swap from the artifact.
- POC scratch artifacts (derivation, priors, validation): `data/scratch_*`, `scratch_*.R`, `scratch_*.py`.
