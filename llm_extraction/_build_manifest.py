#!/usr/bin/env python3
"""Build a JSONL manifest of (claim_id, doi, claim_text, pdf_path, tei_path) rows.

Reads one or more source CSVs (training/test data) and locates each row's
`file_o` PDF in any of the supplied papers directories, plus a matching
GROBID TEI XML in any of the supplied --tei-dir directories. Emits one
JSON object per locatable row to stdout. Rows with no claim_id, no
claim_text, or no findable PDF are skipped (counts reported on stderr).

Each emitted record also carries `claim_count_for_paper`, the number of
kept rows that share the same `file_o`. Downstream tools use it to filter
to single-claim papers (`claim_count_for_paper == 1`) without a second pass.

Usage:
  _build_manifest.py [--papers-dir DIR ...] [--tei-dir DIR ...] CSV [CSV ...]

The script intentionally has no third-party dependencies (uses only the
stdlib `csv` and `json` modules) so it runs against system Python.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from pathlib import Path


REQUIRED_COLS = ("claim_id", "doi_o", "claim_text_o", "file_o")


def fuzzy_key(name: str) -> str:
    """Normalise a basename so disk and CSV variants collide.

    Variants observed in the wild:
      - Accented chars in the CSV represented with `_` (Pastötter → Pasto_tter,
        Trémolière → Tre_molie_re) vs the same accent on disk.
      - Case differences (Li_WorldDev vs li_worlddev).
      - Punctuation differences: GROBID rewrites parens and spaces to `_`
        when naming TEI files, so `10.1016--s2215-0366(21)00241-8.pdf`
        becomes `10.1016--s2215-0366_21_00241-8.tei.xml`.
      - Optional .pdf extension (some FORRT file_o values are stem-only).

    Strategy: NFD-decompose accents → drop combining marks → lowercase →
    strip a trailing .pdf → reduce to alphanumeric-only. Collides every
    variant above onto the same key while remaining stable.
    """
    decomposed = unicodedata.normalize("NFD", name)
    out = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            continue
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).lower()
    if s.endswith(".pdf"):
        s = s[:-4]
    return re.sub(r"[^a-z0-9]", "", s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--papers-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory to search for PDFs. Repeatable.",
    )
    p.add_argument(
        "--tei-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Directory to search for *.tei.xml. Repeatable. Searched recursively.",
    )
    p.add_argument("csv", nargs="+", help="Input CSV file(s).")
    return p.parse_args()


def index_papers(dirs: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Build exact and fuzzy basename → absolute path indexes.

    Order in `dirs` controls precedence: first directory wins on collision.
    """
    exact: dict[str, str] = {}
    fuzzy: dict[str, str] = {}
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            print(f"[manifest] warning: papers dir not found: {d}", file=sys.stderr)
            continue
        for pdf in root.glob("*.pdf"):
            path = str(pdf.resolve())
            exact.setdefault(pdf.name, path)
            fuzzy.setdefault(fuzzy_key(pdf.name), path)
    return exact, fuzzy


def index_teis(dirs: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Build exact and fuzzy `<basename>.pdf` → TEI path indexes.

    TEI files are named like `<basename>.tei.xml`. We strip the `.tei.xml`
    suffix and append `.pdf` so the same lookup keys we use for PDFs hit a
    TEI when one exists. Search is recursive (handles the
    `Test set papers/Round 3/` nesting under `data/tei_xml/`).
    """
    exact: dict[str, str] = {}
    fuzzy: dict[str, str] = {}
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            print(f"[manifest] warning: tei dir not found: {d}", file=sys.stderr)
            continue
        for tei in root.rglob("*.tei.xml"):
            stem = tei.name[:-len(".tei.xml")]  # strip the compound suffix
            pdf_like = stem + ".pdf"
            path = str(tei.resolve())
            exact.setdefault(pdf_like, path)
            fuzzy.setdefault(fuzzy_key(pdf_like), path)
    return exact, fuzzy


def lookup_pdf(file_o: str, exact: dict[str, str], fuzzy: dict[str, str]) -> str | None:
    if not file_o:
        return None
    candidates = [file_o]
    if not file_o.lower().endswith(".pdf"):
        candidates.append(file_o + ".pdf")
    for cand in candidates:
        hit = exact.get(cand)
        if hit:
            return hit
    for cand in candidates:
        hit = fuzzy.get(fuzzy_key(cand))
        if hit:
            return hit
    return None


def iter_rows(csv_path: str):
    if not os.path.exists(csv_path):
        print(
            f"[manifest] {csv_path}: file not found on disk; skipping",
            file=sys.stderr,
        )
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing:
            print(
                f"[manifest] {csv_path}: missing required columns {missing}; skipping file",
                file=sys.stderr,
            )
            return
        for row in reader:
            yield row


def normalise(value: str | None) -> str:
    return (value or "").strip()


def main() -> int:
    args = parse_args()
    pdf_exact, pdf_fuzzy = index_papers(args.papers_dir)
    if not pdf_exact:
        print("[manifest] error: no PDFs indexed; check --papers-dir", file=sys.stderr)
        return 2
    tei_exact, tei_fuzzy = index_teis(args.tei_dir)

    seen_ids: set[str] = set()
    counts = {"emitted": 0, "no_claim_id": 0, "no_claim_text": 0,
              "no_pdf": 0, "no_tei": 0, "duplicate_id": 0}

    # Collect rows first so we can stamp claim_count_for_paper before emitting.
    records: list[dict] = []
    for csv_path in args.csv:
        source_label = os.path.basename(csv_path)
        for row in iter_rows(csv_path):
            claim_id = normalise(row.get("claim_id"))
            if not claim_id or claim_id == "NA":
                counts["no_claim_id"] += 1
                continue
            if claim_id in seen_ids:
                counts["duplicate_id"] += 1
                continue

            claim_text = normalise(row.get("claim_text_o"))
            if not claim_text or claim_text == "NA":
                counts["no_claim_text"] += 1
                continue

            file_o = normalise(row.get("file_o"))
            pdf_path = lookup_pdf(file_o, pdf_exact, pdf_fuzzy)
            if not pdf_path:
                counts["no_pdf"] += 1
                continue

            tei_path = lookup_pdf(file_o, tei_exact, tei_fuzzy)
            if tei_path is None:
                counts["no_tei"] += 1

            # Emit paths relative to the current working directory so the
            # manifest is portable across machines. run.sh invokes us with
            # cwd = extract_claude_claim/, so paths look like `../data/...`
            # and resolve cleanly when run.sh later re-reads them.
            try:
                rel_pdf = os.path.relpath(pdf_path)
            except ValueError:
                rel_pdf = pdf_path  # different drive on Windows
            rel_tei: str | None
            if tei_path is None:
                rel_tei = None
            else:
                try:
                    rel_tei = os.path.relpath(tei_path)
                except ValueError:
                    rel_tei = tei_path

            doi = normalise(row.get("doi_o"))
            dataset = normalise(row.get("dataset")) or "round3"
            primary_raw = normalise(row.get("primary")).upper()
            primary = primary_raw in ("TRUE", "1", "T", "YES")

            records.append({
                "claim_id": claim_id,
                "doi": doi if doi and doi != "NA" else None,
                "claim_text": claim_text,
                "pdf_path": rel_pdf,
                "tei_path": rel_tei,
                "dataset": dataset,
                "source_csv": source_label,
                "file_o": file_o,
                "primary": primary,
            })
            seen_ids.add(claim_id)

    # Stamp claim_count_for_paper using `file_o` as the grouping key (the
    # same key the manifest already uses to locate the PDF/TEI on disk).
    per_file_counts: dict[str, int] = {}
    for rec in records:
        per_file_counts[rec["file_o"]] = per_file_counts.get(rec["file_o"], 0) + 1
    for rec in records:
        rec["claim_count_for_paper"] = per_file_counts[rec["file_o"]]
        # Drop `file_o` from the emitted record — it was only carried for
        # internal grouping; downstream consumers use `pdf_path`/`tei_path`.
        rec.pop("file_o", None)
        print(json.dumps(rec, ensure_ascii=False))
        counts["emitted"] += 1

    summary = " ".join(f"{k}={v}" for k, v in counts.items())
    print(f"[manifest] {summary}", file=sys.stderr)
    return 0 if counts["emitted"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
