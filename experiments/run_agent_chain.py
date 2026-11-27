"""Step 4.8: run the fixed agent chain on the dev datasets and save results/tables/agent_chain.csv.

Each run uses the best model by validation ROC-AUC and scores it ONCE on the locked test set.
Rows go to the table only (not results/raw), because system 4 and 5 are registered
in the benchmark runner in later phases.
"""
import time

import pandas as pd
import yaml

from src.evaluation.metrics import TARGET_COL, score_on_test
from src.graph.workflow import build_graph
from src.tools.ml_tools import predict_proba1
from src.utils.helpers import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "results" / "tables" / "agent_chain.csv"


def run_one(graph, name: str, seed: int) -> dict:
    base = PROJECT_ROOT / "data" / "processed" / f"{name}_seed{seed}"
    state = {
        "dataset_name": name,
        "goal": "predict the target column",
        "target": TARGET_COL,
        "seed": seed,
        "train_path": f"{base}_train.csv",
        "val_path": f"{base}_val.csv",
    }
    row = {"system": "agent_chain", "dataset": name, "seed": seed}
    t0 = time.time()
    try:
        out = graph.invoke(state)
        best = out["best_model"]
        model, features = out["models"][best], out["features"]
        test = score_on_test(name, seed, lambda X: predict_proba1(model, X[features]))
        row.update(best_model=best, val_roc_auc=out["results"][best]["val_roc_auc"],
                   **{f"test_{k}": v for k, v in test.items()}, crashed=False, error="")
    except Exception as e:                      # crashes are recorded, not hidden
        row.update(crashed=True, error=str(e)[:200])
    row["time_s"] = round(time.time() - t0, 1)
    return row


if __name__ == "__main__":
    with open(PROJECT_ROOT / "configs" / "datasets.yaml", encoding="utf-8") as f:
        names = [d["name"] for d in yaml.safe_load(f)["dev"]]
    seeds = load_config()["seeds"]

    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["dataset"], r["seed"]) for r in rows}
    graph = build_graph(profiler_llm=False, use_lead=False)

    for name in names:
        for seed in seeds:
            if (name, seed) in done:
                continue                                  # finished runs are skipped
            row = run_one(graph, name, seed)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)   # save after every run
            print(row)
    print(pd.DataFrame(rows).to_string())
