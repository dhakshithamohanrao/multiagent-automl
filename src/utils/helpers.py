from pathlib import Path
import yaml
import json

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _PROJECT_ROOT
RESULTS_FILE = _PROJECT_ROOT / "results" / "raw" / "results.jsonl"

RESULT_DEFAULTS = {
    "run_id": None, "system": None, "dataset": None, "seed": None,
    "flaw": "none", "val_score": None, "test_score": None,
    "score_drop": None, "flaw_caught": None, "time_s": None,
    "llm_calls": 0, "tokens": 0, "cost": 0.0,
    "crashed": False, "error": "", "flags_raised": [],
}


def write_result(row: dict) -> None:
    """Append one run to results/raw/results.jsonl (one JSON line per run)."""
    full = {**RESULT_DEFAULTS, **row}
    if full["run_id"] is None:
        full["run_id"] = (f'{full["system"]}|{full["dataset"]}|'
                          f'{full["seed"]}|{full["flaw"]}')
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(full) + "\n")


def finished_run_ids() -> set:
    """Ids of runs already saved, so the runner can skip them."""
    if not RESULTS_FILE.exists():
        return set()
    ids = set()
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["run_id"])
    return ids

def load_config(name: str = "experiment.yaml") -> dict:
    """Load a YAML file from the configs/ folder."""
    path = _PROJECT_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
