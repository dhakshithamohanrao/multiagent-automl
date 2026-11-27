"""Feature engineer agent: decides which columns the models may use.

Reads profile, earlier `dropped` entries and `extra_features`; writes only `features` and `dropped`.

Imputing and encoding are NOT done here on the data. They live inside the model Pipeline
(src/tools/ml_tools.py), which is fitted on the training split only and then applied
identically to validation and test data. That keeps the project free of leakage.
"""
import numpy as np
import pandas as pd

from src.graph.state import State


def add_simple_features(df: pd.DataFrame, specs) -> pd.DataFrame:
    """Add extra columns from recipes {"name", "op", "a", "b"}; op is 'ratio' or 'diff'."""
    if not specs:
        return df
    out = df.copy()
    for s in specs:
        a, b = out[s["a"]], out[s["b"]]
        if s["op"] == "ratio":
            out[s["name"]] = (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        elif s["op"] == "diff":
            out[s["name"]] = a - b
        else:
            raise ValueError(f"Unknown feature op: {s['op']}")
    return out


def choose_extra_features(df: pd.DataFrame, numeric_cols: list, target: str, top_k: int = 3) -> list:
    """Recipes for ratio and difference features of the top_k numeric columns most
    correlated with the target (computed on the TRAINING split only)."""
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    top = list(corr.index[:top_k])
    specs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            specs.append({"name": f"{a}_per_{b}", "op": "ratio", "a": a, "b": b})
            specs.append({"name": f"{a}_minus_{b}", "op": "diff", "a": a, "b": b})
    return specs


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

    base = [c for c in all_cols if c not in dropped]
    extra = [s["name"] for s in state.get("extra_features", []) if s["name"] not in dropped]
    features = base + extra
    if not features:
        raise ValueError("No features left after dropping columns.")
    return {**state, "features": features, "dropped": dropped}


def select_xy(df: pd.DataFrame, features: list, target: str, extra=None):
    """Return (X, y) using only the allowed feature columns (extra features are built first)."""
    df = add_simple_features(df, extra)
    return df[features], df[target]


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse
    import json

    from src.agents.profiler import run_profiler
    from src.evaluation.metrics import TARGET_COL
    from src.utils.helpers import PROJECT_ROOT

    ap = argparse.ArgumentParser(description="Checkpoint: choose features for one split.")
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