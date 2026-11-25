"""ML building blocks used by the agents: preprocessing, training, tuning, evaluation.

Rules:
- Preprocessing lives inside a scikit-learn Pipeline, so imputing and encoding
  are fitted on the training data only.
- Tuning uses cross-validation on the TRAIN split only. The validation split is
  never used for tuning, and the locked test file is never touched here.
"""
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_metrics
from src.utils.helpers import PROJECT_ROOT, load_config

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def load_models_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_xy(df: pd.DataFrame, target: str):
    """Return (X, y) from a dataframe that contains the target column."""
    return df.drop(columns=[target]), df[target]


# ---------- preprocessing ----------

def build_preprocessor(X: pd.DataFrame, scale: bool = False) -> ColumnTransformer:
    """Impute and encode. Numeric: median. Text columns: most frequent + one-hot.

    scale=True adds standard scaling for numeric columns (needed by logistic regression).
    High-cardinality text columns are capped at 30 categories, so an ID-like column
    cannot create thousands of one-hot columns.
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 max_categories=30, sparse_output=False)),
    ]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))
    return ColumnTransformer(transformers)


# ---------- models ----------

def make_model(name: str, params: dict | None = None, seed: int = 0,
               balance: bool = False, y=None):
    """Create one classifier from models.yaml (fixed params) plus the given tuned params.

    balance=True applies class weighting for imbalanced data (used later by the lead's
    fix_imbalance action). XGBoost needs y to compute scale_pos_weight.
    """
    cfg = load_models_config()[name]
    p = {**cfg.get("fixed", {}), **(params or {})}

    if name == "logistic_regression":
        if balance:
            p["class_weight"] = "balanced"
        return LogisticRegression(random_state=seed, **p)
    if name == "random_forest":
        if balance:
            p["class_weight"] = "balanced"
        return RandomForestClassifier(random_state=seed, **p)
    if name == "xgboost":
        if balance and y is not None:
            pos = max(int((np.asarray(y) == 1).sum()), 1)
            neg = int((np.asarray(y) == 0).sum())
            p["scale_pos_weight"] = neg / pos
        return XGBClassifier(random_state=seed, **p)
    if name == "lightgbm":
        if balance:
            p["class_weight"] = "balanced"
        return LGBMClassifier(random_state=seed, **p)
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name: str, X: pd.DataFrame, y, params: dict | None = None,
                  seed: int = 0, balance: bool = False) -> Pipeline:
    prep = build_preprocessor(X, scale=(name == "logistic_regression"))
    model = make_model(name, params, seed=seed, balance=balance, y=y)
    return Pipeline([("prep", prep), ("model", model)])


def train_model(name: str, X: pd.DataFrame, y, params: dict | None = None,
                seed: int = 0, balance: bool = False) -> Pipeline:
    """Fit preprocessing + model on the training data. Returns the fitted pipeline."""
    pipe = make_pipeline(name, X, y, params, seed=seed, balance=balance)
    pipe.fit(X, y)
    return pipe


# ---------- tuning ----------

def _suggest(trial: optuna.Trial, search: dict) -> dict:
    params = {}
    for pname, spec in search.items():
        if spec["type"] == "int":
            params[pname] = trial.suggest_int(pname, spec["low"], spec["high"])
        else:
            params[pname] = trial.suggest_float(pname, spec["low"], spec["high"],
                                                log=spec.get("log", False))
    return params


def tune_model(name: str, X: pd.DataFrame, y, n_trials: int | None = None,
               seed: int = 0, balance: bool = False, timeout_s: float | None = None):
    """Optuna search on the training split with 3-fold CV (ROC-AUC).

    Returns (best_params, best_cv_auc). timeout_s caps the search time for this model.
    """
    if n_trials is None:
        n_trials = load_config()["optuna_trials"]
    search = load_models_config()[name].get("search", {})
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def objective(trial):
        params = _suggest(trial, search)
        pipe = make_pipeline(name, X, y, params, seed=seed, balance=balance)
        return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    return study.best_params, float(study.best_value)


# ---------- evaluation ----------

def predict_proba1(model, X: pd.DataFrame) -> np.ndarray:
    """Probability of class 1 for each row."""
    return model.predict_proba(X)[:, 1]


def evaluate(model, X: pd.DataFrame, y) -> dict:
    """Score a fitted model with the project's common metrics."""
    return compute_metrics(y, predict_proba1(model, X))


def auc(model, X: pd.DataFrame, y) -> float:
    return float(roc_auc_score(y, predict_proba1(model, X)))


# ---------- checkpoint runner ----------

if __name__ == "__main__":
    import argparse
    import time

    from src.evaluation.metrics import TARGET_COL
    from src.utils.data import load_split

    ap = argparse.ArgumentParser(description="Step 4.2 checkpoint: train (and optionally tune) models.")
    ap.add_argument("name", help="dataset name, for example credit-g")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", choices=MODEL_NAMES, default=None,
                    help="also tune this model with Optuna")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    X_tr, y_tr = split_xy(load_split(args.name, args.seed, "train"), TARGET_COL)
    X_va, y_va = split_xy(load_split(args.name, args.seed, "val"), TARGET_COL)

    print(f"{args.name} seed {args.seed}: train {X_tr.shape}, val {X_va.shape}")
    for m in MODEL_NAMES:
        t0 = time.time()
        fitted = train_model(m, X_tr, y_tr, seed=args.seed)
        print(f"{m:20s} default  val ROC-AUC {auc(fitted, X_va, y_va):.4f}  ({time.time() - t0:.1f}s)")

    if args.tune:
        t0 = time.time()
        best, cv_auc = tune_model(args.tune, X_tr, y_tr, n_trials=args.trials, seed=args.seed)
        fitted = train_model(args.tune, X_tr, y_tr, params=best, seed=args.seed)
        print(f"{args.tune:20s} tuned    val ROC-AUC {auc(fitted, X_va, y_va):.4f}  "
              f"(cv {cv_auc:.4f}, {time.time() - t0:.1f}s)")
        print("best params:", best)