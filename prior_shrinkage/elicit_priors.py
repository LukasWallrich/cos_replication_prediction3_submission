#!/usr/bin/env python3
"""Prior elicitation for the m2 "prior x evidence" predictor.

Asks GPT-5.5 and Gemini 3.1 Pro, each under 4 reviewer personas, to rate the
PRIOR probability that each hypothesis is true in the population — given ONLY
the substantive (de-leaked) claim, with all sample-size / p-value / effect-size
/ design cues removed. One API call per (model, persona); the Claude Opus
priors are elicited the same way through the local `claude` CLI (not shown here).

INPUT : data/r3_scrubbed.txt   one scrubbed claim per line, numbered "R01. ..."
                               (challenge-derived; NOT shipped — see DATA.md)
OUTPUT: data/r3_priors_mm/<model>_<persona>.json   array of {idx, prob, recognized, reason}

These per-(model,persona) JSONs are aggregated into prior_3fam by
2_prior_aggregation.R / 3_build_predictor.R.

Keys come from environment variables (OPENAI_API_KEY, GEMINI_API_KEY), falling
back to ~/.Rprofile; no key is stored in this repository. Re-running makes paid
API calls — the deployed m2 predictor is already shipped in
output/r3_prior_shrinkage_predictor.csv.
"""
import json, re, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent          # package root
RPROFILE = Path.home() / ".Rprofile"

def rkeys():
    d = {}
    if RPROFILE.exists():
        for line in RPROFILE.read_text().splitlines():
            m = re.match(r'^\s*([A-Z_][A-Z0-9_]*)\s*=\s*"([^"]+)"\s*$', line)
            if m: d[m.group(1)] = m.group(2)
    return d
RK = rkeys()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "") or RK.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "") or RK.get("GEMINI_API_KEY", "")

CLAIMS = (ROOT / "data/r3_scrubbed.txt").read_text().strip()
N = len(CLAIMS.splitlines())
OUT = ROOT / "data/r3_priors_mm"; OUT.mkdir(parents=True, exist_ok=True)

PERSONAS = {
 "bayesian": "You are a calibrated Bayesian forecaster. Base rate that a published hypothesis like this reflects a real effect is ~40-60%; start there and update only on substance. Well-established structural mechanisms are more prior-plausible; specific/novel interactions, moderations, mediations and counter-intuitive reversals less. Don't go extreme without strong reason.",
 "field": "You are an experienced health/behavioural/social science researcher. Direct demographic/behavioural/structural relationships tend to be real; theory-driven multi-construct mediation, specific moderations/interactions, novel mechanisms and reversals are frequently not real. Base rate ~40-60%.",
 "generalist": "You are a thoughtful generalist reader with no strong field priors. Judge whether the asserted relationship is the kind of thing generally true in the world given common sense and broad scientific knowledge. Base rate ~40-60%; simple/intuitive/structural claims more plausible than intricate/counter-intuitive ones.",
 "skeptic": "You are a skeptical methodologist. Many published effects are not real or smaller than claimed; specific mediation/interaction effects and counter-intuitive reversals are least likely real; subtle framing effects often aren't real while large structural relationships usually are. Calibrated not contrarian; base rate ~40-60%.",
}
TASK = (f"\n\nYou will rate {N} scientific hypotheses for their PRIOR PROBABILITY of being TRUE in the population "
        "(a real non-zero effect in the stated direction that would hold in a well-powered test). You are given ONLY "
        "the hypothesis — NO sample size, p-value, effect size, or study/design info; do not assume any. Judge purely "
        "on substantive prior plausibility and base rates.\n\n"
        "For EACH numbered claim return: prob (0.01-0.99), recognized ('yes'/'maybe'/'no' = do you recognize the exact "
        "study/its replication outcome — honesty only, do NOT use such knowledge), and a <=12-word reason.\n"
        f"Return ONLY a JSON array of exactly {N} objects: "
        '[{"idx":1,"prob":0.0,"recognized":"no","reason":"..."}, ...] in order.\n\nCLAIMS:\n' + CLAIMS)

def extract_json(text):
    m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if m: text = m.group(1)
    s, e = text.find("["), text.rfind("]")
    if s>=0 and e>s:
        try: return json.loads(text[s:e+1])
        except Exception: return None
    return None

def call_openai(persona):
    from openai import OpenAI
    c = OpenAI(api_key=OPENAI_KEY, max_retries=1)
    r = c.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role":"user","content":PERSONAS[persona]+TASK}],
        reasoning_effort="low",
        max_completion_tokens=16000,
        timeout=300,
    )
    usage = r.usage
    return r.choices[0].message.content or "", (usage.prompt_tokens, usage.completion_tokens)

def call_gemini(persona):
    from google import genai
    from google.genai import types as gt
    c = genai.Client(api_key=GEMINI_KEY)
    r = c.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=PERSONAS[persona]+TASK+"\n\nReturn JSON only.",
        config={"response_mime_type":"application/json","max_output_tokens":16000,
                "thinking_config":{"thinking_budget":3000},"service_tier":"flex",
                "http_options": gt.HttpOptions(timeout=300000)},
    )
    um = r.usage_metadata
    return (getattr(r,"text",None) or ""), (um.prompt_token_count, um.candidates_token_count)

def run(model, persona):
    try:
        text, usage = (call_openai if model=="gpt55" else call_gemini)(persona)
        arr = extract_json(text)
        if arr is None: return (model,persona,"PARSE_FAIL",0,usage,text[:200])
        (OUT/f"{model}_{persona}.json").write_text(json.dumps(arr))
        return (model,persona,f"ok n={len(arr)}",len(arr),usage,"")
    except Exception as e:
        return (model,persona,f"ERR {type(e).__name__}: {str(e)[:150]}",0,(0,0),"")

if __name__ == "__main__":
    cells = [(m,p) for m in ("gpt55","gem31") for p in PERSONAS]
    tot_in=tot_out=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run,m,p):(m,p) for m,p in cells}
        for f in as_completed(futs):
            m,p,status,n,usage,err = f.result()
            tot_in += usage[0] or 0; tot_out += usage[1] or 0
            print(f"{m:6} {p:11} {status}  tok(in/out)={usage}  {err}")
    print(f"\nTOTAL tokens in={tot_in} out={tot_out}")
