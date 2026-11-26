"""Evaluator agent: scores the trained models on the VALIDATION split and picks the best.

Reads val_path, features, target, models. Writes only `results` and `best_model`.
The best model is chosen by validation ROC-AUC. The locked test set is never used here.
"""
import numpy as np
import pandas as pd

from src.agents.feature_engineer import select_xy
from src.graph.state import State
from src.tools.ml_tools import auc, evaluate


def _clean(v):
    return round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v


def run_evaluator(state: State) -> State:
    df = pd.read_csv(state["val_path"])
    X, y = select_xy(df, state["features"], state["target"])

    results = {}
    for name, model in state["models"].items():
        metrics = evaluate(model, X, y)
        results[name] = {"val_roc_auc": round(auc(model, X, y), 4),
                         **{k: _clean(v) for k, v in metrics.items()}}

    best = max(results, key=lambda n: results[n]["val_roc_auc"])
    return {**state, "results": results, "best_model": best}


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse
    import json
    import time

    from src.agents.feature_engineer import run_feature_engineer
    from src.agents.profiler import run_profiler
    from src.agents.trainer import run_trainer
    from src.evaluation.metrics import TARGET_COL
    from src.utils.helpers import PROJECT_ROOT

    ap = argparse.ArgumentParser(description="Step 4.6 checkpoint: profile, select, train, evaluate.")
    ap.add_argument("name", help="dataset name, for example credit-g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    base = PROJECT_ROOT / "data" / "processed" / f"{args.name}_seed{args.seed}"
    state: State = {
        "dataset_name": args.name,
        "goal": "predict the target column",
        "target": TARGET_COL,
        "seed": args.seed,
        "train_path": f"{base}_train.csv",
        "val_path": f"{base}_val.csv",
    }
    t0 = time.time()
    state = run_profiler(state, use_llm=False)
    state = run_feature_engineer(state)
    state = run_trainer(state, n_trials=args.trials)
    state = run_evaluator(state)

    print(json.dumps(state["results"], indent=2))
    print("best model:", state["best_model"], f"({time.time() - t0:.1f}s total)")