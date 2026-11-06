# src/llm/client.py
import hashlib, json, os, time
from pathlib import Path
import requests
from dotenv import load_dotenv
from src.utils.helpers import load_config, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env", override=True)
_cfg = load_config()["llm"]          # model, temperature
CACHE_DIR = PROJECT_ROOT / "results" / "raw" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USAGE = {"calls": 0, "tokens": 0, "cache_hits": 0}


def _key(system, prompt):
    raw = json.dumps([_cfg["model"], system, prompt])
    return hashlib.sha256(raw.encode()).hexdigest()


def _call_gemini(system, prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{_cfg['model']}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        timeout=120,
        json={"systemInstruction": {"parts": [{"text": system}]},
              "contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0}})
    if not r.ok:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    d = r.json()
    return (d["candidates"][0]["content"]["parts"][0]["text"],
            d.get("usageMetadata", {}).get("totalTokenCount", 0))

import requests  # already imported at the top

def ask(prompt, system="You are a helpful assistant.", retries=5):
    f = CACHE_DIR / f"{_key(system, prompt)}.json"
    if f.exists():
        USAGE["cache_hits"] += 1
        return json.loads(f.read_text())["text"]
    for attempt in range(retries):
        try:
            text, tokens = _call_gemini(system, prompt)
            USAGE["calls"] += 1
            USAGE["tokens"] += tokens
            f.write_text(json.dumps({"text": text, "tokens": tokens}))
            return text
        except Exception as e:
            temporary = isinstance(e, (requests.exceptions.Timeout,
                                       requests.exceptions.ConnectionError)) \
                or any(code in str(e) for code in ("error 429", "error 500", "error 503"))
            if attempt == retries - 1 or not temporary:
                raise
            time.sleep(5 * 2 ** attempt)   # 5, 10, 20, 40 seconds


def get_usage():
    return {**USAGE, "cost": 0.0}   # free tier, so cost is 0