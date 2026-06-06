#!/usr/bin/env python3
"""Add `tei_path` and `claim_count_for_paper` to an existing manifest.

This is a one-off complement to `_build_manifest.py`. Use it when you want
to keep the rows that are already in `manifest.jsonl` but enrich them with
the TEI path lookup and per-paper claim count introduced by the TEI-inlined
pipeline. The motivation: when the source CSVs that produced the original
manifest are no longer on disk (e.g. only the `.rds` form remains), a full
rebuild would shrink the manifest. Enriching preserves all rows.

Usage:
  _enrich_manifest.py [--tei-dir DIR ...] [--in PATH] [--out PATH]

Defaults: --in manifest.jsonl --out manifest.jsonl (in place).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path


def fuzzy_key(name: str) -> str:
    decomposed = unicodedata.normalize("NFD", name)
    out = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            out.append("_")
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).lower()


def index_teis(dirs: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    fuzzy: dict[str, str] = {}
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            print(f"[enrich] warning: tei dir not found: {d}", file=sys.stderr)
            continue
        for tei in root.rglob("*.tei.xml"):
            stem = tei.name[:-len(".tei.xml")]
            pdf_like = stem + ".pdf"
            path = str(tei.resolve())
            exact.setdefault(pdf_like, path)
            fuzzy.setdefault(fuzzy_key(pdf_like), path)
    return exact, fuzzy


def lookup(name: str, exact: dict[str, str], fuzzy: dict[str, str]) -> str | None:
    if not name:
        return None
    hit = exact.get(name)
    if hit:
        return hit
    return fuzzy.get(fuzzy_key(name))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tei-dir", action="append", default=[], metavar="DIR")
    p.add_argument("--in", dest="in_path", default="manifest.jsonl")
    p.add_argument("--out", dest="out_path", default="manifest.jsonl")
    args = p.parse_args()

    if not os.path.exists(args.in_path):
        print(f"[enrich] error: {args.in_path} not found", file=sys.stderr)
        return 2

    tei_exact, tei_fuzzy = index_teis(args.tei_dir)

    rows: list[dict] = []
    with open(args.in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    matched = 0
    missed = 0
    for r in rows:
        pdf_basename = os.path.basename(r.get("pdf_path", "") or "")
        tei_path = lookup(pdf_basename, tei_exact, tei_fuzzy)
        if tei_path is None:
            r["tei_path"] = None
            missed += 1
        else:
            try:
                r["tei_path"] = os.path.relpath(tei_path)
            except ValueError:
                r["tei_path"] = tei_path
            matched += 1

    per_paper: dict[str, int] = {}
    for r in rows:
        per_paper[r.get("pdf_path", "")] = per_paper.get(r.get("pdf_path", ""), 0) + 1
    for r in rows:
        r["claim_count_for_paper"] = per_paper[r.get("pdf_path", "")]

    with open(args.out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[enrich] rows={len(rows)}  matched_tei={matched}  missing_tei={missed}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
