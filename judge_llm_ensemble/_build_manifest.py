#!/usr/bin/env python3
"""Build a JSONL manifest of (claim_id, persona, model) tasks for the ensemble.

Reads source CSVs (R3 test set, R1/R2 from the combined training file), joins
each row to the TEI extraction CSV by DOI to pull title/abstract/year, and
emits one JSON object per (claim_id, persona, model) cell. Output is the
manifest read by run.py.

Usage:
  _build_manifest.py > manifest.jsonl
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import config


def norm_doi(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    return s.lower().strip("/")


def year_from_pub_date(s: str | None) -> str:
    """Pull a 4-digit year from a date string like '2020-10-14' or '2020'."""
    if not s:
        return ""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    return m.group(1) if m else ""


def year_from_doi(doi: str) -> str:
    """Fallback: some DOIs encode the publication year, e.g.
    `10.1080/10810730.2024.2354370` (Taylor & Francis Journal of Health
    Communication uses `.YYYY.` in the suffix)."""
    m = re.search(r"\.(19\d{2}|20\d{2})\.", doi)
    return m.group(1) if m else ""


def load_tei_index(tei_csv: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    with tei_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doi = norm_doi(row.get("doi"))
            if doi and doi != "na":
                idx.setdefault(doi, row)
    return idx


def iter_claims():
    """Yield (claim_id, dataset_label, source_row) for every claim we plan
    to ensemble on: all R3 test + all R1+R2 rows from dataset_incl_round2.csv.

    Rows lacking a claim_id, claim_text, or DOI are skipped (counted on stderr).
    """
    seen_ids: set[str] = set()
    skipped = {"no_claim_id": 0, "no_claim_text": 0, "no_doi": 0, "duplicate": 0,
               "wrong_dataset": 0}

    for src in config.SOURCES:
        is_r3 = "Round3" in str(src)
        with src.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ds = (row.get("dataset") or "round3").strip()
                if not is_r3 and ds not in config.INCLUDE_DATASETS:
                    skipped["wrong_dataset"] += 1
                    continue
                cid = (row.get("claim_id") or "").strip()
                if not cid or cid == "NA":
                    skipped["no_claim_id"] += 1
                    continue
                if cid in seen_ids:
                    skipped["duplicate"] += 1
                    continue
                txt = (row.get("claim_text_o") or "").strip()
                if not txt or txt == "NA":
                    skipped["no_claim_text"] += 1
                    continue
                doi = norm_doi(row.get("doi_o"))
                if not doi:
                    skipped["no_doi"] += 1
                    continue
                seen_ids.add(cid)
                yield cid, ("round3" if is_r3 else ds), row

    for k, v in skipped.items():
        if v:
            print(f"[manifest] skipped {v} rows: {k}", file=sys.stderr)


def main() -> int:
    tei_idx = load_tei_index(config.TEI_CSV)
    print(f"[manifest] loaded {len(tei_idx)} TEI rows with DOI", file=sys.stderr)

    emitted = 0
    no_tei = 0
    for cid, ds, row in iter_claims():
        doi = norm_doi(row.get("doi_o"))
        tei = tei_idx.get(doi)
        if not tei:
            no_tei += 1
            continue

        title = (tei.get("title") or "").strip()
        abstract = (tei.get("abstract") or "").strip()
        year = year_from_pub_date(tei.get("pub_date")) or year_from_doi(doi)

        base = {
            "claim_id": cid,
            "dataset": ds,
            "doi": doi,
            "claim_text": (row.get("claim_text_o") or "").strip(),
            "title": title,
            "year": year,
            "n_o": (row.get("n_o") or "").strip(),
            "es_value": (row.get("es_value_o") or "").strip(),
            "es_type": (row.get("es_type_o") or "").strip(),
            "pval_value": (row.get("pval_value_o") or "").strip(),
            "abstract": abstract,
        }

        for persona_key, _ in config.PERSONAS:
            for vendor, model_id, model_key in config.MODELS:
                task = dict(base)
                task["persona"] = persona_key
                task["vendor"] = vendor
                task["model_id"] = model_id
                task["model_key"] = model_key
                task["task_id"] = f"{cid}__{persona_key}__{model_key}"
                print(json.dumps(task, ensure_ascii=False))
                emitted += 1

    print(f"[manifest] emitted {emitted} tasks; {no_tei} claims dropped for missing TEI",
          file=sys.stderr)
    return 0 if emitted else 1


if __name__ == "__main__":
    sys.exit(main())
