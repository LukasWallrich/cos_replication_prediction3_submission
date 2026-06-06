# judge_llm_ensemble/

LLM synthetic-crowd ensemble that produces the **m3** column (and the substantive
prior used by **m2**). Python; three entry points run in order:

```
python _build_manifest.py   # → manifest.jsonl  (one task per claim × persona × model)
python run.py               # → raw_responses/*.json  (resumable; caches every call)
python aggregate.py         # → ensemble_scores.csv   (12 judgements/claim → probability)
```

- `config.py` — models (Claude Opus, GPT-5.5, Gemini 3.1 Pro), the 4 personas,
  the 7-bucket scale, and the source-data paths.
- `personas/*.txt`, `prompt_template.txt` — the prompts.
- `schema.json` — the structured-output contract each model must return.
- `ensemble_scores.csv` — **shipped**: the aggregated scores behind the submission.
  Re-running `run.py` is optional and incurs paid API calls (see ../DATA.md).

API keys come from environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`);
Claude uses the local `claude` CLI's own auth. No key is stored in this repo.
