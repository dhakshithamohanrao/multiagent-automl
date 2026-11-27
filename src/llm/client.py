"""The only door to the LLM: temperature 0, cache, token log, timeout and retries."""
import hashlib
import json
import os
import time

import requests
from dotenv import load_dotenv

from src.utils.helpers import PROJECT_ROOT, load_config

load_dotenv(PROJECT_ROOT / ".env")


def _llm_config() -> dict:
    """Read the optional `llm:` block of configs/experiment.yaml."""
    try:
        return load_config().get("llm", {}) or {}
    except Exception:
        return {}


_cfg = _llm_config()
MODEL = _cfg.get("model") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = _cfg.get("temperature", 0)
DEFAULT_SYSTEM = "You are a helpful assistant."   # same default as the old client, keeps old cache keys valid

TIMEOUT_S = 120          # max wait for one API response
MAX_ATTEMPTS = 3         # 1 try + 2 retries
RETRY_STATUS = {429, 500, 502, 503, 504}

CACHE_FILE = PROJECT_ROOT / "results" / "raw" / "llm_cache.jsonl"

# Counters for the cost tracker. Cache hits are counted separately and are not calls.
_stats = {"calls": 0, "tokens": 0, "cache_hits": 0}
_cache = None


class _Retryable(Exception):
    """Temporary API problem (rate limit, server error). Worth another attempt."""


def get_stats() -> dict:
    return dict(_stats)


def get_usage() -> dict:
    """Old name kept for existing code. Free tier, so cost is 0."""
    return {**_stats, "cost": 0.0}


def reset_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def _api_key() -> str:
    key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No API key found. Put LLM_API_KEY=... in your .env file.")
    return key


def _cache_key(system: str, prompt: str) -> str:
    raw = json.dumps([MODEL, system, prompt])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_FILE.exists():
            with open(CACHE_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        _cache[entry["key"]] = entry["text"]
    return _cache


def _save_to_cache(key: str, text: str) -> None:
    _load_cache()[key] = text
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")


def _call_gemini(system: str, prompt: str):
    """One API request. Returns (text, total_tokens)."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": TEMPERATURE},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    r = requests.post(
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": _api_key()},
        json=body,
        timeout=TIMEOUT_S,
    )
    if r.status_code in RETRY_STATUS:
        raise _Retryable(f"HTTP {r.status_code}: {r.text[:200]}")
    if not r.ok:
        # Permanent problem (bad key, bad request): fail fast, do not retry.
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")

    data = r.json()
    try:
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response: {str(data)[:300]}")
    if not text.strip():
        raise _Retryable("Empty answer from Gemini")
    tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
    return text, tokens


def ask(prompt: str, system: str = DEFAULT_SYSTEM) -> str:
    """Return the LLM answer. Served from the cache when the same request was made before."""
    key = _cache_key(system, prompt)
    cache = _load_cache()
    if key in cache:
        _stats["cache_hits"] += 1
        return cache[key]

    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            text, tokens = _call_gemini(system, prompt)
            _stats["calls"] += 1
            _stats["tokens"] += tokens
            _save_to_cache(key, text)
            return text
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                _Retryable) as e:
            last_error = e
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(15 * 2 ** attempt)   # wait 5 s, then 10 s
    raise RuntimeError(f"LLM call failed after {MAX_ATTEMPTS} attempts: {last_error}")
