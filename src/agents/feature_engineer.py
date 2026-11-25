"""Feature engineer agent: decides which columns the models may use.

Reads `profile` (and any earlier `dropped` entries), writes only `features` and `dropped`.

Imputing and encoding are NOT done here on the data. They live inside the model Pipeline
(src/tools/ml_tools.py), which is fitted on the training split only and then applied
identically to validation and test data. That keeps the project free of leakage.
Extra features (the lead's add_features action) come in Phase 5.
"""
import pandas as pd

from src.graph.state import State


def run_feature_engineer(state: State) -> State:
    profile = state["profile"]
    all_cols = profile["numeric_columns"] + profile["categorical_columns"]

    dropped = dict(state.get("dropped", {}))          # drops requested earlier stay
    flags = profile.get("flags", {})

    for col in flags.get("constant_columns", []):
        dropped.setdefault(col, "constant column (one unique value)")
    for col in flags.get("high_missing_columns", []):
        rate = profile["missing_rates"].get(col)
        dropped.setdefault(col, f"high missing rate {rate}")   # same number as in the profile

    features = [c for c in all_cols if c not in dropped]
    if not features:
        raise ValueError("No features left after dropping columns.")
    return {**state, "features": features, "dropped": dropped}


def select_xy(df: pd.DataFrame, features: list, target: str):
    """Return (X, y) using only the allowed feature columns."""
    return df[features], df[target]


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse
    import json

    from src.agents.profiler import run_profiler
    from src.evaluation.metrics import TARGET_COL
    from src.utils.helpers import PROJECT_ROOT

    ap = argparse.ArgumentParser(description="Step 4.4 checkpoint: choose features for one split.")
    ap.add_argument("name", help="dataset name, for example credit-g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default=None, help="test: pretend an earlier step dropped this column")
    args = ap.parse_args()

    train_path = PROJECT_ROOT / "data" / "processed" / f"{args.name}_seed{args.seed}_train.csv"
    state: State = {
        "dataset_name": args.name,
        "goal": "predict the target column",
        "target": TARGET_COL,
        "train_path": str(train_path),
    }
    if args.drop:
        state["dropped"] = {args.drop: "manual test"}

    state = run_profiler(state, use_llm=False)
    state = run_feature_engineer(state)
    print("features:", len(state["features"]))
    print("dropped:", json.dumps(state["dropped"], indent=2))