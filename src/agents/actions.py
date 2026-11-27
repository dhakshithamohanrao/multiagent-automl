"""Applies the lead's chosen action to the state.

drop_columns  -> adds the columns to `dropped`
fix_imbalance -> sets `balance` so the trainer uses class weights
add_features  -> builds feature recipes from the TRAINING split (never the LLM)
tune_more     -> raises `extra_trials`
Writes only dropped, balance, extra_features, extra_trials.
"""
import pandas as pd

from src.agents.feature_engineer import choose_extra_features
from src.utils.helpers import load_config


def apply_action(state: dict) -> dict:
    action = state["next_step"]
    decision = state["decisions"][-1]
    new = dict(state)

    if action == "drop_columns":
        dropped = dict(state.get("dropped", {}))
        for col in decision["columns"]:
            dropped[col] = f"dropped by lead: {decision['reason']}"
        new["dropped"] = dropped

    elif action == "fix_imbalance":
        new["balance"] = True

    elif action == "add_features":
        df = pd.read_csv(state["train_path"])
        dropped = state.get("dropped", {})
        numeric = [c for c in state["profile"]["numeric_columns"] if c not in dropped]
        existing = state.get("extra_features", [])
        have = {s["name"] for s in existing}
        new_specs = [s for s in choose_extra_features(df, numeric, state["target"])
                     if s["name"] not in have]
        new["extra_features"] = existing + new_specs

    elif action == "tune_more":
        new["extra_trials"] = state.get("extra_trials", 0) + load_config()["optuna_trials"]

    return new