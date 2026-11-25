"""Profiler agent: computes facts about the training data with pandas.

All numbers come from pandas. The LLM only writes a short summary of those facts.
The profiler reads only the training split and writes only the `profile` field.

Design choice: the profiler reports facts and two obvious flags (constant columns and
columns with many missing values). It does NOT decide that a column is an ID or a leak.
Those are the critic's checks (Phase 6), so the comparison "agents without critic" vs
"agents with critic" stays fair.
"""
import json
import logging

import pandas as pd

from src.graph.state import State
from src.llm.client import ask
from src.utils.helpers import load_config

log = logging.getLogger(__name__)

SUMMARY_SYSTEM = (
    "You summarise dataset facts for a data scientist. "
    "Use ONLY the numbers and column names given in the facts. "
    "Do not invent, estimate or round any number. "
    "Write 3 to 4 plain sentences, no bullet points."
)


def compute_profile(df: pd.DataFrame, target: str, high_missing: float = 0.5) -> dict:
    """All facts about a training dataframe. No LLM involved."""
    n_rows = len(df)
    features = [c for c in df.columns if c != target]
    X = df[features]

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]

    missing = X.isna().mean()
    missing_rates = {c: round(float(r), 4) for c, r in missing.items() if r > 0}
    unique_ratios = {c: round(float(X[c].nunique(dropna=True)) / n_rows, 4) for c in features}

    constant = [c for c in features if X[c].nunique(dropna=True) <= 1]
    many_missing = [c for c, r in missing_rates.items() if r >= high_missing]

    counts = df[target].value_counts()
    class_balance = {str(k): round(float(v) / n_rows, 4) for k, v in counts.items()}

    return {
        "n_rows": int(n_rows),
        "n_features": len(features),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_rates": missing_rates,
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_ratios": unique_ratios,
        "class_balance": class_balance,
        "rare_class_ratio": round(float(counts.min()) / n_rows, 4),
        "flags": {"constant_columns": constant, "high_missing_columns": many_missing},
    }


def _compact_facts(profile: dict) -> dict:
    """A short version of the profile for the LLM prompt."""
    top_missing = sorted(profile["missing_rates"].items(), key=lambda kv: -kv[1])[:5]
    return {
        "n_rows": profile["n_rows"],
        "n_features": profile["n_features"],
        "n_numeric": len(profile["numeric_columns"]),
        "n_categorical": len(profile["categorical_columns"]),
        "columns_with_missing_values": len(profile["missing_rates"]),
        "top_missing_rates": dict(top_missing),
        "duplicate_rows": profile["duplicate_rows"],
        "class_balance": profile["class_balance"],
        "flags": profile["flags"],
    }


def _plain_summary(profile: dict) -> str:
    """Fallback summary built from a template, used when the LLM is off or fails."""
    return (
        f"The training data has {profile['n_rows']} rows and {profile['n_features']} features "
        f"({len(profile['numeric_columns'])} numeric, {len(profile['categorical_columns'])} categorical). "
        f"{len(profile['missing_rates'])} columns have missing values and "
        f"{profile['duplicate_rows']} rows are duplicates. "
        f"Class balance: {profile['class_balance']}."
    )


def _summarise(profile: dict, use_llm: bool) -> str:
    if not use_llm:
        return _plain_summary(profile)
    prompt = "Facts about the dataset (JSON):\n" + json.dumps(_compact_facts(profile), indent=2)
    try:
        return ask(prompt, SUMMARY_SYSTEM).strip()
    except Exception as e:                       # an LLM problem must never break the run
        log.warning("Profiler summary fell back to template: %s", e)
        return _plain_summary(profile)


def run_profiler(state: State, use_llm: bool = True) -> State:
    """Agent function: reads train_path and target, writes only `profile`."""
    df = pd.read_csv(state["train_path"])
    high_missing = load_config().get("profiler", {}).get("high_missing", 0.5)
    profile = compute_profile(df, state["target"], high_missing)
    profile["summary"] = _summarise(profile, use_llm)
    return {**state, "profile": profile}


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse

    from src.evaluation.metrics import TARGET_COL
    from src.utils.helpers import PROJECT_ROOT

    ap = argparse.ArgumentParser(description="Step 4.3 checkpoint: profile one training split.")
    ap.add_argument("name", help="dataset name, for example credit-g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-llm", action="store_true", help="use the template summary, no LLM call")
    args = ap.parse_args()

    train_path = PROJECT_ROOT / "data" / "processed" / f"{args.name}_seed{args.seed}_train.csv"
    state: State = {
        "dataset_name": args.name,
        "goal": "predict the target column",
        "target": TARGET_COL,
        "train_path": str(train_path),
    }
    out = run_profiler(state, use_llm=not args.no_llm)
    print(json.dumps(out["profile"], indent=2))