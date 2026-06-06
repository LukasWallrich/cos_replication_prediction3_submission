#!/usr/bin/env python3
"""Aggregate cached LLM-judge responses into one CSV row per claim_id.

For each claim, we read every `raw_responses/<claim_id>__<persona>__<model>.json`
file, extract the bucket → midpoint, and compute summary statistics across
the (persona × model) cells. We also apply the fixed-power transform to
produce `p_replication_observed`.

Output: `ensemble_scores.csv`

Usage:
  ./aggregate.py            # write ensemble_scores.csv
  ./aggregate.py --strict   # exit non-zero if any claim has fewer than the
                            # expected number of (persona, model) responses
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import config

EXPECTED_PER_CLAIM = len(config.PERSONAS) * len(config.MODELS)


def load_responses() -> dict[str, list[dict]]:
    """Return claim_id → list of parsed response records (only ok ones)."""
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(config.RAW_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if not data.get("_ok"):
            continue
        task = data.get("_task", {})
        judgement = data.get("judgement", {})
        bucket = judgement.get("bucket")
        if bucket not in config.BUCKET_MIDPOINT:
            continue
        rec = {
            "claim_id": task.get("claim_id"),
            "persona": task.get("persona"),
            "model_key": task.get("model_key"),
            "vendor": task.get("vendor"),
            "dataset": task.get("dataset"),
            "bucket": bucket,
            "p": config.BUCKET_MIDPOINT[bucket],
            "confidence_in_bucket": judgement.get("confidence_in_bucket", ""),
        }
        if rec["claim_id"]:
            by_claim[rec["claim_id"]].append(rec)
    return by_claim


def safe_sd(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def safe_iqr(xs):
    if len(xs) < 4:
        return 0.0
    xs = sorted(xs)
    q1 = statistics.median(xs[: len(xs) // 2])
    q3 = statistics.median(xs[-(len(xs) // 2):])
    return q3 - q1


def aggregate_claim(claim_id: str, recs: list[dict]) -> dict:
    out: dict = {"claim_id": claim_id, "dataset": recs[0].get("dataset", "")}

    # Per-cell columns: p and confidence per (persona, model)
    cell_p: dict[tuple[str, str], float] = {}
    for r in recs:
        cell_p[(r["persona"], r["model_key"])] = r["p"]
        out[f'judge_{r["persona"]}_{r["model_key"]}_p'] = r["p"]
        out[f'judge_{r["persona"]}_{r["model_key"]}_conf'] = r["confidence_in_bucket"]

    all_p = [r["p"] for r in recs]
    out["judge_n_responses"] = len(recs)
    out["judge_mean_p"] = round(statistics.mean(all_p), 4)
    out["judge_median_p"] = round(statistics.median(all_p), 4)
    out["judge_sd_p"] = round(safe_sd(all_p), 4)
    out["judge_min_p"] = round(min(all_p), 4)
    out["judge_max_p"] = round(max(all_p), 4)
    out["judge_iqr_p"] = round(safe_iqr(all_p), 4)

    # Disagreement decompositions
    # persona disagreement = sd of (mean across models within each persona)
    per_persona_means = []
    for persona_key, _ in config.PERSONAS:
        ps = [p for (pk, _mk), p in cell_p.items() if pk == persona_key]
        if ps:
            per_persona_means.append(statistics.mean(ps))
    out["judge_persona_disagreement"] = round(safe_sd(per_persona_means), 4)

    # model disagreement = sd of (mean across personas within each model)
    per_model_means = []
    for _, _, model_key in config.MODELS:
        ps = [p for (_pk, mk), p in cell_p.items() if mk == model_key]
        if ps:
            per_model_means.append(statistics.mean(ps))
    out["judge_model_disagreement"] = round(safe_sd(per_model_means), 4)

    # Power-adjusted observed-replication probability
    p_sub = out["judge_mean_p"]
    out["expected_power"] = config.ASSUMED_POWER
    out["p_replication_observed"] = round(
        p_sub * config.ASSUMED_POWER + (1 - p_sub) * config.ALPHA, 4
    )

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any claim has < EXPECTED_PER_CLAIM responses.")
    args = ap.parse_args()

    by_claim = load_responses()
    if not by_claim:
        print("No responses found in raw_responses/.", file=sys.stderr)
        return 1

    rows = [aggregate_claim(cid, recs) for cid, recs in sorted(by_claim.items())]

    # Build column order: claim_id, dataset, per-cell cols (in fixed order),
    # then summary cols.
    cell_cols: list[str] = []
    for persona_key, _ in config.PERSONAS:
        for _, _, model_key in config.MODELS:
            cell_cols.append(f"judge_{persona_key}_{model_key}_p")
            cell_cols.append(f"judge_{persona_key}_{model_key}_conf")

    summary_cols = [
        "judge_n_responses",
        "judge_mean_p", "judge_median_p", "judge_sd_p",
        "judge_min_p", "judge_max_p", "judge_iqr_p",
        "judge_persona_disagreement", "judge_model_disagreement",
        "expected_power", "p_replication_observed",
    ]
    cols = ["claim_id", "dataset"] + cell_cols + summary_cols

    with config.OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    incomplete = [r for r in rows if r["judge_n_responses"] < EXPECTED_PER_CLAIM]
    print(f"[aggregate] {len(rows)} claims; "
          f"{len(rows) - len(incomplete)} complete ({EXPECTED_PER_CLAIM}/cell), "
          f"{len(incomplete)} incomplete.", file=sys.stderr)
    if incomplete and args.strict:
        for r in incomplete[:5]:
            print(f"  incomplete: {r['claim_id']} ({r['judge_n_responses']}/{EXPECTED_PER_CLAIM})",
                  file=sys.stderr)
        return 2

    print(f"Wrote {config.OUTPUT_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
