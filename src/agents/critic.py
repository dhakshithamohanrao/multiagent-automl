"""Critic agent: runs real checks on the current work, asks the LLM for a verdict,
and sets next_step to 'approve' or 'send_back'.

Reads train_path, val_path, features, target, models, results, best_model, critic_loops.
Writes only `critic_flags`, `critic_verdict`, `critic_loops` and `next_step`.
The checks are Python code. The LLM only phrases the verdict, and it can never approve
while a serious flag exists. If the API fails, the code decides alone.
"""
import json
import logging

import pandas as pd

from src.agents.feature_engineer import select_xy
from src.llm.client import ask
from src.tools import leakage_checks as lc
from src.tools.fairness import group_recall_gap
from src.tools.ml_tools import auc
from src.utils.helpers import PROJECT_ROOT, load_config

log = logging.getLogger(__name__)

PROMPT_FILE = PROJECT_ROOT / "src" / "llm" / "prompts" / "critic.txt"
LOG_FILE = PROJECT_ROOT / "results" / "raw" / "critic_decisions.jsonl"

SEVERITY = {
    "single_feature_leakage": "serious",
    "id_like_columns": "serious",
    "suspicious_names": "warning",     # a name alone is weak evidence
    "duplicate_overlap": "serious",
    "time_split_check": "serious",
    "imbalance_check": "warning",
    "overfit_gap_check": "warning",
    "group_recall_gap": "warning",
}

# Defaults. Step 6.9 moves the tuned values into configs/experiment.yaml under `critic:`.
DEFAULTS = {"leakage_auc": 0.97, "id_ratio": 0.95, "dup_min_share": 0.05, "dup_ratio": 2.0,
            "rare_threshold": 0.10, "recall_threshold": 0.30, "overfit_gap": 0.10,
            "fairness_gap": 0.20, "max_sendbacks": 3}


def _cfg() -> dict:
    try:
        return {**DEFAULTS, **(load_config().get("critic", {}) or {})}
    except Exception:
        return dict(DEFAULTS)


def run_checks(state: dict, cfg: dict) -> list:
    """Run every check on the CURRENT features and return the list of raised flags."""
    target = state["target"]
    train = pd.read_csv(state["train_path"])
    val = pd.read_csv(state["val_path"])
    extra = state.get("extra_features")
    Xtr, ytr = select_xy(train, state["features"], target, extra)
    Xva, yva = select_xy(val, state["features"], target, extra)
    best = state["best_model"]
    model = state["models"][best]

    raw = {}
    raw["single_feature_leakage"] = lc.single_feature_leakage(Xtr, ytr, cfg["leakage_auc"])
    raw["id_like_columns"] = lc.id_like_columns(Xtr, cfg["id_ratio"])
    raw["suspicious_names"] = lc.suspicious_names(list(Xtr.columns), target)
    cols = list(state["features"]) + [target]
    raw["duplicate_overlap"] = lc.duplicate_overlap(
        train[cols], val[cols], cfg["dup_min_share"], cfg["dup_ratio"])
    raw["time_split_check"] = lc.time_split_check(train, val, state.get("time_column"))
    raw["imbalance_check"] = lc.imbalance_check(
        ytr, state["results"][best].get("rare_class_recall"),
        cfg["rare_threshold"], cfg["recall_threshold"])
    raw["overfit_gap_check"] = lc.overfit_gap_check(
        auc(model, Xtr, ytr), state["results"][best]["val_roc_auc"], cfg["overfit_gap"])
    sens = state.get("sensitive_column")
    if sens and sens in val.columns:
        raw["group_recall_gap"] = group_recall_gap(
            val, yva, model.predict(Xva), sens, cfg["fairness_gap"])

    return [{"check": name, "details": details, "severity": SEVERITY[name]}
            for name, (flag, details) in raw.items() if flag]


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    d = json.loads(text[start:end + 1])
    if d.get("verdict") not in ("approve", "send_back"):
        raise ValueError(f"unknown verdict {d.get('verdict')!r}")
    return {"verdict": d["verdict"], "reason": str(d.get("reason", "")).strip() or "no reason given"}


def _verdict(flags: list, sends_left: int, use_llm: bool):
    serious = [f for f in flags if f["severity"] == "serious"]
    code_verdict = "send_back" if serious and sends_left > 0 else "approve"
    code_reason = (f"serious flags: {[f['check'] for f in serious]}" if serious
                   else "no serious flags")
    if not use_llm:
        return {"verdict": code_verdict, "reason": code_reason}, "code"

    system = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = "Check results (JSON):\n" + json.dumps(
        {"flags": flags, "sends_left": sends_left}, indent=2)
    try:
        d = _parse(ask(prompt, system))
    except Exception as e:                                # API outage or bad answer
        log.warning("Critic LLM failed, using code verdict: %s", str(e)[:200])
        return {"verdict": code_verdict, "reason": code_reason + " (LLM unavailable)"}, "fallback"
    if d["verdict"] != code_verdict:                      # code is the authority on serious flags
        log.warning("Critic LLM said %s but checks require %s", d["verdict"], code_verdict)
        return {"verdict": code_verdict, "reason": code_reason}, "override"
    return d, "llm"


def run_critic(state: dict, use_llm: bool = True) -> dict:
    cfg = _cfg()
    loops = state.get("critic_loops", 0)
    sends_left = cfg["max_sendbacks"] - loops
    flags = run_checks(state, cfg)
    verdict, source = _verdict(flags, sends_left, use_llm)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"dataset": state.get("dataset_name"), "seed": state.get("seed"),
                            "flags": flags, "verdict": verdict, "source": source,
                            "sends_left": sends_left}) + "\n")

    new_loops = loops + 1 if verdict["verdict"] == "send_back" else loops
    return {**state, "critic_flags": flags, "critic_verdict": {**verdict, "source": source},
            "critic_loops": new_loops, "next_step": verdict["verdict"]}


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse

    from src.agents.feature_engineer import run_feature_engineer
    from src.agents.evaluator import run_evaluator
    from src.agents.profiler import run_profiler
    from src.agents.trainer import run_trainer
    from src.evaluation.metrics import TARGET_COL

    ap = argparse.ArgumentParser(description="Step 6.6 checkpoint: run the critic on one dataset.")
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--llm", action="store_true", help="ask the LLM for the verdict")
    args = ap.parse_args()

    base = PROJECT_ROOT / "data" / "processed" / f"{args.name}_seed{args.seed}"
    s = {"dataset_name": args.name, "goal": "predict the target column", "target": TARGET_COL,
         "seed": args.seed, "train_path": f"{base}_train.csv", "val_path": f"{base}_val.csv"}
    s = run_profiler(s, use_llm=False)
    s = run_feature_engineer(s)
    s = run_trainer(s, n_trials=args.trials)
    s = run_evaluator(s)
    out = run_critic(s, use_llm=args.llm)
    print("flags:", json.dumps(out["critic_flags"], indent=2))
    print("verdict:", out["critic_verdict"], "| next_step:", out["next_step"])