"""Explainer agent: SHAP top-5 features of the best model, then 3-5 plain sentences.

Reads val_path, features, target, extra_features, models, best_model.
Writes only `explanation`.
All numbers come from SHAP in code. The LLM text is rejected if it contains any number that is
not a share_percent value, and a template text is used instead (also when the API fails).
"""
import json
import logging
import re

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.agents.feature_engineer import select_xy
from src.llm.client import ask
from src.utils.helpers import PROJECT_ROOT

log = logging.getLogger(__name__)
PROMPT_FILE = PROJECT_ROOT / "src" / "llm" / "prompts" / "explainer.txt"
MAX_ROWS = 200


def _original_column(name: str, columns) -> str:
    """'cat__purpose_radio/tv' -> 'purpose' (longest original column that matches)."""
    stripped = name.split("__", 1)[-1]
    best = None
    for c in columns:
        if stripped == c or stripped.startswith(str(c) + "_"):
            if best is None or len(str(c)) > len(str(best)):
                best = c
    return best if best is not None else stripped


def _shap_matrix(pipe, X: pd.DataFrame):
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Z = np.asarray(prep.transform(X), dtype=float)
    names = list(prep.get_feature_names_out())
    if isinstance(model, LogisticRegression):
        values = shap.LinearExplainer(model, Z).shap_values(Z)
    else:
        values = shap.TreeExplainer(model).shap_values(Z)
    if isinstance(values, list):                 # old SHAP: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                         # (rows, features, classes)
        values = values[:, :, 1]
    return values, names


def top_features(pipe, X: pd.DataFrame, k: int = 5) -> list:
    """Top-k original columns by mean |SHAP|, with share of total and a direction note."""
    X = X.head(MAX_ROWS)
    values, names = _shap_matrix(pipe, X)
    groups = {}
    for j, n in enumerate(names):
        groups.setdefault(_original_column(n, X.columns), []).append(j)
    agg = {c: values[:, idx].sum(axis=1) for c, idx in groups.items()}
    importance = {c: float(np.abs(v).mean()) for c, v in agg.items()}
    total = sum(importance.values()) or 1.0

    out = []
    for c in sorted(importance, key=importance.get, reverse=True)[:k]:
        direction = "the effect depends on the category or value"
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
            x = X[c].to_numpy(dtype=float)
            x = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0, x)
            if np.std(x) > 0 and np.std(agg[c]) > 0:
                r = float(np.corrcoef(x, agg[c])[0, 1])
                if r > 0.1:
                    direction = "higher values push the prediction up"
                elif r < -0.1:
                    direction = "higher values push the prediction down"
        out.append({"feature": str(c), "share_percent": int(round(100 * importance[c] / total)),
                    "direction": direction})
    return out


def template_text(top: list) -> str:
    parts = [f"{t['feature']} ({t['share_percent']}% of the model's influence; {t['direction']})"
             for t in top]
    return ("The model relies most on these features: " + "; ".join(parts) +
            ". These are associations found in the data, not proof of cause and effect.")


def valid_text(text: str, top: list) -> bool:
    """Only share_percent values may appear as numbers, and at least one sentence must exist."""
    allowed = {str(t["share_percent"]) for t in top}
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return bool(text.strip()) and all(n in allowed for n in numbers)


def run_explainer(state: dict, use_llm: bool = True) -> dict:
    df = pd.read_csv(state["val_path"])
    X, _ = select_xy(df, state["features"], state["target"], state.get("extra_features"))
    best = state["best_model"]
    top = top_features(state["models"][best], X)

    text, source = template_text(top), "template"
    if use_llm:
        try:
            system = PROMPT_FILE.read_text(encoding="utf-8")
            answer = ask("Top features (JSON):\n" + json.dumps(top, indent=2), system).strip()
            if valid_text(answer, top):
                text, source = answer, "llm"
            else:
                log.warning("Explainer LLM text rejected (invalid number); using template.")
        except Exception as e:
            log.warning("Explainer LLM failed, using template: %s", str(e)[:200])
    return {**state, "explanation": {"model": best, "top_features": top,
                                     "text": text, "source": source}}


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse

    from src.agents.evaluator import run_evaluator
    from src.agents.feature_engineer import run_feature_engineer
    from src.agents.profiler import run_profiler
    from src.agents.trainer import run_trainer
    from src.evaluation.metrics import TARGET_COL

    ap = argparse.ArgumentParser(description="Step 7.1 checkpoint: explain the best model.")
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    base = PROJECT_ROOT / "data" / "processed" / f"{args.name}_seed{args.seed}"
    s = {"dataset_name": args.name, "goal": "predict the target column", "target": TARGET_COL,
         "seed": args.seed, "train_path": f"{base}_train.csv", "val_path": f"{base}_val.csv"}
    s = run_profiler(s, use_llm=False)
    s = run_feature_engineer(s)
    s = run_trainer(s, n_trials=args.trials)
    s = run_evaluator(s)
    out = run_explainer(s, use_llm=args.llm)
    print("best model:", out["best_model"])
    print(json.dumps(out["explanation"], indent=2))