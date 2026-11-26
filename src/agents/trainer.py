"""Trainer agent: tunes and trains the four model types on the TRAINING split only.

Reads train_path, features, target, seed (and balance). Writes only `models` and `tuned_params`.
Tuning uses cross-validation inside the training data, so the validation split stays clean
for the evaluator and the locked test set is never touched.
"""
import logging

import pandas as pd

from src.agents.feature_engineer import select_xy
from src.graph.state import State
from src.tools.ml_tools import train_model, tune_model
from src.utils.helpers import load_config

log = logging.getLogger(__name__)


def run_trainer(state: State, models: list | None = None, n_trials: int | None = None) -> State:
    cfg = load_config()
    names = models or cfg["models"]
    n_trials = n_trials or cfg["optuna_trials"]
    timeout_s = cfg["time_budget_s"] / len(names)     # equal share of the time budget per model
    seed = state.get("seed", 0)
    balance = state.get("balance", False)

    df = pd.read_csv(state["train_path"])
    X, y = select_xy(df, state["features"], state["target"])

    trained, tuned = {}, {}
    for name in names:
        try:
            params, cv_auc = tune_model(name, X, y, n_trials=n_trials, seed=seed,
                                        balance=balance, timeout_s=timeout_s)
            trained[name] = train_model(name, X, y, params=params, seed=seed, balance=balance)
            tuned[name] = {"params": params, "cv_auc": round(cv_auc, 4)}
        except Exception as e:
            log.warning("Tuning %s failed (%s). Trying default settings.", name, e)
            try:
                trained[name] = train_model(name, X, y, seed=seed, balance=balance)
                tuned[name] = {"params": {}, "cv_auc": None}
            except Exception as e2:                      # one failing model must not stop the others
                log.warning("Model %s skipped: %s", name, e2)

    if not trained:
        raise RuntimeError("The trainer could not train any model.")
    return {**state, "models": trained, "tuned_params": tuned}


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse
    import json
    import time

    from src.agents.feature_engineer import run_feature_engineer
    from src.agents.profiler import run_profiler
    from src.evaluation.metrics import TARGET_COL
    from src.utils.helpers import PROJECT_ROOT

    ap = argparse.ArgumentParser(description="Step 4.5 checkpoint: tune and train all models.")
    ap.add_argument("name", help="dataset name, for example credit-g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    train_path = PROJECT_ROOT / "data" / "processed" / f"{args.name}_seed{args.seed}_train.csv"
    state: State = {
        "dataset_name": args.name,
        "goal": "predict the target column",
        "target": TARGET_COL,
        "seed": args.seed,
        "train_path": str(train_path),
    }
    state = run_profiler(state, use_llm=False)
    state = run_feature_engineer(state)

    t0 = time.time()
    state = run_trainer(state, n_trials=args.trials)
    print(f"trained {list(state['models'])} in {time.time() - t0:.1f}s")
    print(json.dumps({k: v["cv_auc"] for k, v in state["tuned_params"].items()}, indent=2))