"""Lead agent: reads a compact state summary and picks the next action from a fixed list.

Reads the summary of profile, results, features, dropped, balance and earlier decisions.
Writes only `next_step`, `loop_count` and `decisions`.
The LLM answer is validated in code. After one failed retry the lead falls back to `finish`,
so an invalid answer or an API outage never breaks a run.
"""
import json
import logging
import re

from src.llm.client import ask
from src.utils.helpers import PROJECT_ROOT, load_config

log = logging.getLogger(__name__)

ACTIONS = ["drop_columns", "fix_imbalance", "add_features", "tune_more", "finish"]
PROMPT_FILE = PROJECT_ROOT / "src" / "llm" / "prompts" / "lead.txt"
LOG_FILE = PROJECT_ROOT / "results" / "raw" / "lead_decisions.jsonl"
SCORE_KEYS = ("val_roc_auc", "f1", "balanced_accuracy", "rare_class_recall")


def build_summary(state: dict, max_decisions: int) -> dict:
    """A short JSON-friendly view of the state for the LLM."""
    profile = state.get("profile", {})
    top_unique = sorted(profile.get("unique_ratios", {}).items(), key=lambda kv: -kv[1])[:5]
    results = {name: {k: r.get(k) for k in SCORE_KEYS}
               for name, r in state.get("results", {}).items()}
    return {
        "n_rows": profile.get("n_rows"),
        "n_features_now": len(state.get("features", [])),
        "class_balance": profile.get("class_balance"),
        "rare_class_ratio": profile.get("rare_class_ratio"),
        "duplicate_rows": profile.get("duplicate_rows"),
        "missing_rates": profile.get("missing_rates"),
        "highest_unique_ratio_columns": dict(top_unique),
        "already_dropped": state.get("dropped", {}),
        "class_weights_applied": bool(state.get("balance", False)),
        "best_model": state.get("best_model"),
        "validation_results": results,
        "previous_decisions": [{k: d[k] for k in ("action", "columns", "reason")}
                               for d in state.get("decisions", [])],
        "decisions_left": max_decisions - state.get("loop_count", 0),
    }


def parse_decision(text: str) -> dict:
    """Pull the JSON object out of the LLM text (tolerates markdown fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def validate_decision(d, state: dict) -> dict:
    if not isinstance(d, dict):
        raise ValueError("the answer is not a JSON object")
    action = d.get("action")
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    reason = str(d.get("reason", "")).strip() or "no reason given"

    columns = d.get("columns") or []
    if action == "drop_columns":
        features = state.get("features", [])
        if not isinstance(columns, list) or not columns:
            raise ValueError("drop_columns needs a non-empty list in 'columns'")
        bad = [c for c in columns if c not in features]
        if bad:
            raise ValueError(f"columns not in the current features: {bad}")
        if len(columns) >= len(features):
            raise ValueError("cannot drop every feature")
    else:
        columns = []
    if action == "fix_imbalance" and state.get("balance"):
        raise ValueError("class weights are already applied")
    return {"action": action, "columns": columns, "reason": reason}

def _decide(summary: dict, state: dict):
    system = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = "Current state summary (JSON):\n" + json.dumps(summary, indent=2)
    error = None
    for _ in range(2):                                   # first try + one retry
        p = prompt if error is None else (
            prompt + f"\n\nYour previous answer was invalid ({error}). "
                     "Answer again with valid JSON only.")
        try:
            text = ask(p, system)
        except Exception as e:                           # API outage: client already retried
            error = str(e)[:200]
            log.warning("Lead API failure: %s", error)
            break
        try:
            return validate_decision(parse_decision(text), state), "llm"
        except Exception as e:                           # bad JSON or bad action
            error = str(e)[:200]
            log.warning("Lead answer rejected: %s", error)
    return {"action": "finish", "columns": [],
            "reason": f"fallback to finish: {error}"}, "fallback"

def _log(state: dict, summary: dict, decision: dict, source: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"dataset": state.get("dataset_name"), "seed": state.get("seed"),
              "decision_number": state.get("loop_count", 0) + 1,
              "summary": summary, "decision": decision, "source": source}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_lead(state: dict) -> dict:
    limit = load_config()["limits"]["lead_max_decisions"]
    count = state.get("loop_count", 0)
    summary = build_summary(state, limit)

    if count >= limit:                                   # hard limit: no more LLM calls
        decision = {"action": "finish", "columns": [], "reason": "decision limit reached"}
        source = "limit"
    else:
        decision, source = _decide(summary, state)
    _log(state, summary, decision, source)

    new_count = count if source == "limit" else count + 1
    decisions = state.get("decisions", []) + [{**decision, "source": source}]
    return {**state, "next_step": decision["action"], "loop_count": new_count,
            "decisions": decisions}


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    demo = {
        "dataset_name": "demo_imbalanced", "target": "class", "seed": 0,
        "profile": {"n_rows": 5000, "class_balance": {"0": 0.96, "1": 0.04},
                    "rare_class_ratio": 0.04, "duplicate_rows": 0,
                    "unique_ratios": {"a": 0.01, "b": 0.5}, "missing_rates": {}},
        "features": ["a", "b", "c"],
        "best_model": "xgboost",
        "results": {
            "xgboost": {"val_roc_auc": 0.91, "f1": 0.12, "balanced_accuracy": 0.55, "rare_class_recall": 0.09},
            "lightgbm": {"val_roc_auc": 0.90, "f1": 0.10, "balanced_accuracy": 0.54, "rare_class_recall": 0.08},
        },
    }
    out = run_lead(demo)
    print("next_step:", out["next_step"])
    print("decision:", out["decisions"][-1])