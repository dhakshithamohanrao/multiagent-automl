# Lab notebook

## Decisions
- 2026-11-03: Python 3.11.9 in .venv (system default is 3.15, so the venv was created with `py -3.11`).

## What broke and why
(nothing yet)

## Frozen versions
(prompts and thresholds are recorded here in Phase 7)

## Phase 3: single-LLM baseline
- Ran on credit-g, diabetes, blood-transfusion, seeds 0-2: 9 of 9 runs finished, none crashed.
- Gemini calls took 8-230 s and sometimes hung with no timeout. Fixed with the new client: 120 s timeout, 3 attempts, cache in results/raw/llm_cache.jsonl.
- Placeholder names (NAME2, NAME3, DEV2, DEV3) typed into a loop caused FileNotFoundError. Not a system bug.
- Reruns appended duplicate rows to results.jsonl. Deduplicated by run_id.
- Random forest scores changed between identical runs because tuning was cut off by a time limit. Fixed by tuning with a fixed trial count (tuning_timeout: false). Three identical runs confirmed it.