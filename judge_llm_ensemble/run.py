#!/usr/bin/env python3
"""Run the LLM ensemble of judges.

Reads `manifest.jsonl` (built by `_build_manifest.py`), dispatches each
(claim, persona, model) task to the right vendor, and caches the parsed
JSON response to `raw_responses/<task_id>.json`. Resumable: tasks with an
existing cache file are skipped unless --fresh is set.

Usage:
  ./run.py                     # process every task in the manifest
  ./run.py --limit 10          # process at most 10 tasks
  ./run.py --persona skeptic   # filter by persona
  ./run.py --vendor anthropic  # filter by vendor
  ./run.py --fresh             # delete all cached responses first
  ./run.py --concurrency 2     # override PARALLEL_PER_VENDOR

API keys for OpenRouter and Gemini are loaded from ~/.Rprofile.
Anthropic calls go through `claude -p` (uses Claude Code's auth).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import config

# ---- API key loading -------------------------------------------------------

RPROFILE = Path.home() / ".Rprofile"


def load_rprofile_keys() -> dict[str, str]:
    """Parse ~/.Rprofile for `KEY="..."` assignments."""
    keys: dict[str, str] = {}
    if not RPROFILE.exists():
        return keys
    pat = re.compile(r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*"([^"]+)"\s*$')
    for line in RPROFILE.read_text().splitlines():
        m = pat.match(line)
        if m:
            keys[m.group(1)] = m.group(2)
    return keys


_RKEYS = load_rprofile_keys()
# Keys are read from environment variables first (set OPENAI_API_KEY and
# GEMINI_API_KEY in your shell); as a convenience they also fall back to
# ~/.Rprofile if present. No key is ever stored in this repository. The
# Anthropic/Claude path uses the local `claude` CLI's own auth (no key here).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or _RKEYS.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or _RKEYS.get("GEMINI_API_KEY", "")


# ---- Schema (loaded once) --------------------------------------------------

SCHEMA = json.loads(config.SCHEMA_FILE.read_text())


def validate_response(obj) -> tuple[bool, str]:
    """Light schema check. Returns (ok, error_message)."""
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    for k in SCHEMA["required"]:
        if k not in obj:
            return False, f"missing required field {k!r}"
    bucket = obj["bucket"]
    if bucket not in SCHEMA["properties"]["bucket"]["enum"]:
        return False, f"invalid bucket {bucket!r}"
    conf = obj["confidence_in_bucket"]
    if conf not in SCHEMA["properties"]["confidence_in_bucket"]["enum"]:
        return False, f"invalid confidence {conf!r}"
    reasons = obj["reasoning_top2"]
    if not isinstance(reasons, list) or len(reasons) != 2:
        return False, "reasoning_top2 must be a list of exactly 2"
    return True, ""


# ---- Prompt assembly -------------------------------------------------------

PROMPT_TEMPLATE = config.PROMPT_TEMPLATE.read_text()


def load_persona(key: str) -> str:
    return (config.PERSONAS_DIR / f"{key}.txt").read_text().strip()


_PERSONA_CACHE: dict[str, str] = {}


def render_prompt(task: dict) -> str:
    persona_key = task["persona"]
    if persona_key not in _PERSONA_CACHE:
        _PERSONA_CACHE[persona_key] = load_persona(persona_key)
    persona_block = _PERSONA_CACHE[persona_key]
    return (
        PROMPT_TEMPLATE
        .replace("{PERSONA_BLOCK}", persona_block)
        .replace("{CLAIM_ID}", task["claim_id"])
        .replace("{CLAIM_TEXT}", task["claim_text"])
        .replace("{TITLE}", task.get("title") or "(not available)")
        .replace("{YEAR}", task.get("year") or "unknown")
        .replace("{N_O}", task.get("n_o") or "(not reported)")
        .replace("{ES_VALUE}", task.get("es_value") or "(not reported)")
        .replace("{ES_TYPE}", task.get("es_type") or "")
        .replace("{PVAL_VALUE}", task.get("pval_value") or "(not reported)")
        .replace("{ABSTRACT}", task.get("abstract") or "(not available)")
    )


# ---- Vendor dispatchers ----------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a text blob. Lenient: handles
    ```json ... ``` fences and leading/trailing prose."""
    if not text:
        return None
    # strip code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # find first {...} that parses
    start = text.find("{")
    while start != -1:
        for end in range(len(text), start, -1):
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
        start = text.find("{", start + 1)
    return None


def call_anthropic(task: dict, prompt: str) -> dict:
    """Shell out to `claude -p --model <id> --output-format json --json-schema ...`.

    Trim the session prelude to keep concurrent Opus calls from tripping the
    Claude Code subscription's per-minute token budget:
      --tools ""                                 (no tool descriptions)
      --disable-slash-commands                   (no skill descriptions)
      --exclude-dynamic-system-prompt-sections   (move cwd/env/git to user
                                                  message → enables cross-call
                                                  cache reuse on the static
                                                  system prompt)
    Combined effect measured: cache_creation drops from ~10.9k → ~3.5k tokens
    and subsequent calls get a ~7k-token cache hit.

    NOT --bare: that disables keychain auth, switching us to API-key billing.
    """
    model = task["model_id"]
    cmd = [
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--disable-slash-commands",
        "--exclude-dynamic-system-prompt-sections",
        "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA),
        "--permission-mode", "auto",
        "--no-session-persistence",
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=config.CALL_TIMEOUT_S,
    )
    if proc.returncode != 0:
        return {"_error": f"claude exit={proc.returncode}: {proc.stderr[:300]}"}
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"_error": f"claude returned non-JSON: {proc.stdout[:300]}"}
    if envelope.get("is_error"):
        return {"_error": f"claude error: {envelope.get('result', '?')[:300]}"}
    # The structured output lives in `structured_output` or as a JSON string
    # in `result`.
    inner = envelope.get("structured_output")
    if inner is None:
        inner = extract_json(envelope.get("result", ""))
    if inner is None:
        return {"_error": "no parseable structured output", "_raw_envelope": envelope}
    return {"_ok": True, "judgement": inner, "_envelope_keys": list(envelope.keys())}


def call_openai(task: dict, prompt: str) -> dict:
    """Direct OpenAI call with service_tier='flex' (50% off, longer latency).

    Enforces a hard per-attempt timeout via `timeout=` (httpx) and disables
    SDK-internal retries (`max_retries=0`) so flex queueing can't tie up the
    worker indefinitely. On flex timeout/429 we fall back to default tier."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        return {"_error": "OPENAI_API_KEY not found in env or ~/.Rprofile"}

    # max_retries=0 here means *we* control retries via the loop below.
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)

    common = dict(
        model=task["model_id"],
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "judgement", "schema": SCHEMA, "strict": True},
        },
        max_completion_tokens=config.MAX_OUTPUT_TOKENS + config.OPENAI_REASONING_BUDGET,
    )

    # Two flex attempts (cheap, may queue) then one default-tier attempt.
    resp = None
    tried = []
    attempts = [
        ("flex",    config.FLEX_TIMEOUT_S),
        ("flex",    config.FLEX_TIMEOUT_S),
        ("default", config.CALL_TIMEOUT_S),
    ]
    for i, (tier, t_s) in enumerate(attempts):
        try:
            resp = client.chat.completions.create(
                **common,
                service_tier=tier,
                timeout=t_s,
            )
            break
        except Exception as e:
            tried.append(f"{tier}({t_s}s): {type(e).__name__}: {str(e)[:120]}")
            if i < len(attempts) - 1:
                time.sleep(1)
    if resp is None:
        return {"_error": "openai error: " + " | ".join(tried)}

    text = resp.choices[0].message.content or ""
    inner = extract_json(text)
    if inner is None:
        return {"_error": "no parseable JSON", "_raw_text": text[:500]}
    return {
        "_ok": True,
        "judgement": inner,
        "_finish_reason": resp.choices[0].finish_reason,
        "_service_tier": getattr(resp, "service_tier", None),
    }


def call_gemini(task: dict, prompt: str) -> dict:
    """Gemini call via google.genai (NOT the deprecated google.generativeai).

    Gemini's `response_schema` only accepts a restricted subset of JSON Schema
    and rejects unknown keys. To keep cross-vendor consistency we send no
    formal schema and rely on the prompt + local validation."""
    from google import genai

    if not GEMINI_API_KEY:
        return {"_error": "GEMINI_API_KEY not found in ~/.Rprofile"}

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Gemini 3.x is a thinking model — thinking tokens count against
    # max_output_tokens. Use flex service tier (50% off, may queue).
    # google.genai's per-call timeout sits on http_options; we set it via
    # types.HttpOptions on the config so flex queueing can't tie up a worker.
    from google.genai import types as gtypes
    base_cfg = {
        "response_mime_type": "application/json",
        "max_output_tokens": config.MAX_OUTPUT_TOKENS + config.GEMINI_THINKING_BUDGET,
        "thinking_config": {"thinking_budget": config.GEMINI_THINKING_BUDGET},
        "service_tier": "flex",
        # google.genai expects timeout in milliseconds via HttpOptions.
        "http_options": gtypes.HttpOptions(timeout=config.FLEX_TIMEOUT_S * 1000),
    }
    attempts = [("flex", config.FLEX_TIMEOUT_S),
                ("standard", config.CALL_TIMEOUT_S)]
    resp = None
    tried = []
    for tier, t_s in attempts:
        cfg = dict(base_cfg)
        cfg["service_tier"] = tier
        cfg["http_options"] = gtypes.HttpOptions(timeout=t_s * 1000)
        try:
            resp = client.models.generate_content(
                model=task["model_id"],
                contents=prompt + "\n\nReturn JSON only, no prose around it.",
                config=cfg,
            )
            break
        except Exception as e:
            tried.append(f"{tier}({t_s}s): {type(e).__name__}: {str(e)[:120]}")
    if resp is None:
        return {"_error": "gemini error: " + " | ".join(tried)}

    text = getattr(resp, "text", None) or ""
    inner = extract_json(text)
    if inner is None:
        return {"_error": "no parseable JSON", "_raw_text": text[:500]}
    return {"_ok": True, "judgement": inner}


VENDOR_DISPATCH = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "gemini": call_gemini,
}


# ---- Per-vendor semaphores -------------------------------------------------

_VENDOR_LOCKS: dict[str, threading.Semaphore] = {}

# Circuit breaker: when N consecutive Anthropic calls fail with the
# fast-fail "claude exit=1" pattern, the subscription's per-window quota
# is exhausted. Trip the breaker, stop submitting new Anthropic tasks,
# and let the main loop exit cleanly so the user can resume after reset.
_BREAKER_STATE = {
    "consecutive_fast_fails": 0,
    "tripped": False,
}
_BREAKER_LOCK = threading.Lock()
_BREAKER_THRESHOLD = 5  # consecutive matching failures → trip
_BREAKER_FAST_FAIL_S = 5.0  # below this elapsed → likely quota exit-1


def is_quota_signature(result: dict) -> bool:
    """Heuristic: a fast (<5s) failure with 'claude exit=1' message is the
    quota-exhausted pattern, not a normal transient error."""
    if result.get("_ok"):
        return False
    err = result.get("_error", "")
    elapsed = result.get("_elapsed_s", 999)
    return elapsed < _BREAKER_FAST_FAIL_S and "claude exit=1" in err


def get_semaphore(vendor: str, default_concurrency: int,
                   anthropic_concurrency: int | None = None) -> threading.Semaphore:
    if vendor not in _VENDOR_LOCKS:
        c = default_concurrency
        if vendor == "anthropic" and anthropic_concurrency is not None:
            c = anthropic_concurrency
        _VENDOR_LOCKS[vendor] = threading.Semaphore(c)
    return _VENDOR_LOCKS[vendor]


# ---- Worker ----------------------------------------------------------------

def process_task(task: dict, concurrency: int,
                  anthropic_concurrency: int | None = None) -> tuple[str, str]:
    """Return (task_id, status) where status is 'ok' / 'cached' / 'failed' / 'invalid' / 'skipped'."""
    task_id = task["task_id"]
    cache_path = config.RAW_DIR / f"{task_id}.json"
    if cache_path.exists():
        return task_id, "cached"

    vendor = task["vendor"]
    # Skip Anthropic tasks once the circuit breaker is tripped.
    if vendor == "anthropic":
        with _BREAKER_LOCK:
            if _BREAKER_STATE["tripped"]:
                return task_id, "skipped"
    sem = get_semaphore(vendor, concurrency, anthropic_concurrency)
    prompt = render_prompt(task)

    with sem:
        t0 = time.time()
        result = VENDOR_DISPATCH[vendor](task, prompt)
        result["_elapsed_s"] = round(time.time() - t0, 2)
        result["_task"] = {
            "claim_id": task["claim_id"],
            "persona": task["persona"],
            "vendor": vendor,
            "model_id": task["model_id"],
            "model_key": task["model_key"],
            "dataset": task["dataset"],
        }

    # Validate the judgement payload before declaring success
    status = "failed"
    if result.get("_ok"):
        ok, err = validate_response(result.get("judgement", {}))
        if ok:
            status = "ok"
        else:
            result["_validation_error"] = err
            status = "invalid"

    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # Update the Anthropic circuit breaker based on this outcome.
    if vendor == "anthropic":
        with _BREAKER_LOCK:
            if is_quota_signature(result):
                _BREAKER_STATE["consecutive_fast_fails"] += 1
                if _BREAKER_STATE["consecutive_fast_fails"] >= _BREAKER_THRESHOLD:
                    _BREAKER_STATE["tripped"] = True
            else:
                _BREAKER_STATE["consecutive_fast_fails"] = 0

    return task_id, status


# ---- Main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="Max tasks (0 = all)")
    ap.add_argument("--persona", help="Filter by persona key")
    ap.add_argument("--vendor", help="Filter by vendor (anthropic/openrouter/gemini)")
    ap.add_argument("--model-key", help="Filter by model_key")
    ap.add_argument("--claim-id", action="append", help="Filter by claim_id (repeatable)")
    ap.add_argument("--dataset", action="append", help="Filter by dataset (repeatable)")
    ap.add_argument("--fresh", action="store_true", help="Delete cache and start over")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Delete only the cached responses that were errors/invalid (so they're retried).")
    ap.add_argument("--concurrency", type=int, default=config.PARALLEL_PER_VENDOR,
                    help="Parallel calls per vendor (default = config.PARALLEL_PER_VENDOR)")
    ap.add_argument("--anthropic-concurrency", type=int, default=None,
                    help="Override concurrency for Anthropic specifically (e.g. 2)")
    args = ap.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        n = 0
        for f in config.RAW_DIR.glob("*.json"):
            f.unlink()
            n += 1
        print(f"[fresh] deleted {n} cached response(s)", file=sys.stderr)
    elif args.retry_failed:
        n = 0
        for f in config.RAW_DIR.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                f.unlink(); n += 1; continue
            if not d.get("_ok"):
                f.unlink(); n += 1
        print(f"[retry-failed] deleted {n} non-ok cached response(s) for retry", file=sys.stderr)

    if not config.MANIFEST.exists():
        print(f"Missing {config.MANIFEST}. Run _build_manifest.py first.", file=sys.stderr)
        return 1

    # Load and filter tasks
    tasks: list[dict] = []
    with config.MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if args.persona and t["persona"] != args.persona:
                continue
            if args.vendor and t["vendor"] != args.vendor:
                continue
            if args.model_key and t["model_key"] != args.model_key:
                continue
            if args.claim_id and t["claim_id"] not in args.claim_id:
                continue
            if args.dataset and t["dataset"] not in args.dataset:
                continue
            tasks.append(t)

    if args.limit > 0:
        # Apply the limit to *uncached* tasks only — otherwise --limit N gets
        # eaten by already-cached entries before any real work happens.
        kept = []
        n_uncached = 0
        for t in tasks:
            cache_path = config.RAW_DIR / f"{t['task_id']}.json"
            if cache_path.exists():
                kept.append(t)
                continue
            if n_uncached < args.limit:
                kept.append(t)
                n_uncached += 1
        tasks = kept

    if not tasks:
        print("No tasks match the filters.", file=sys.stderr)
        return 1

    # Partition by vendor for visible progress
    by_vendor: dict[str, list[dict]] = {}
    for t in tasks:
        by_vendor.setdefault(t["vendor"], []).append(t)
    print(
        f"[run] {len(tasks)} task(s) to process: "
        + ", ".join(f"{v}={len(ts)}" for v, ts in by_vendor.items()),
        file=sys.stderr,
    )

    counts = {"ok": 0, "cached": 0, "failed": 0, "invalid": 0, "skipped": 0}
    # Use a single pool sized for total concurrency = sum across vendors.
    max_workers = args.concurrency * len(by_vendor)
    breaker_announced = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_task, t, args.concurrency, args.anthropic_concurrency): t
            for t in tasks
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[fut]
            try:
                task_id, status = fut.result()
            except Exception as e:
                task_id, status = task["task_id"], "failed"
                cache_path = config.RAW_DIR / f"{task_id}.json"
                cache_path.write_text(json.dumps({"_error": f"exception: {e}",
                                                   "_task": task}, indent=2))
            counts[status] = counts.get(status, 0) + 1
            if i % 25 == 0 or status in ("failed", "invalid"):
                print(f"  [{i}/{len(tasks)}] {task_id} → {status}", file=sys.stderr)
            # Announce the breaker the first time it trips so it's visible
            # in the log without scanning every "skipped" line.
            with _BREAKER_LOCK:
                tripped = _BREAKER_STATE["tripped"]
            if tripped and not breaker_announced:
                breaker_announced = True
                print(
                    f"\n[circuit-breaker] tripped after {_BREAKER_THRESHOLD} "
                    f"consecutive Anthropic quota failures. "
                    f"Skipping remaining Anthropic tasks; OpenAI/Gemini continue.\n",
                    file=sys.stderr,
                )

    print(f"\n[done] {counts}", file=sys.stderr)
    if _BREAKER_STATE["tripped"]:
        print("[circuit-breaker] Anthropic quota was hit. Re-run later "
              "(after the next quota-window reset) to retry the skipped cells.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
