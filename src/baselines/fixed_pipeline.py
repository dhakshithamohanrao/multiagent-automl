import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_metrics, score_on_test
from src.utils.data import (ROOT, TARGET_COL, load_dataset, load_split,
                            make_splits)
from src.utils.seeds import set_seed

import yaml

SEEDS = [0, 1, 2]  # later this will come from configs/experiment.yaml


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Fill missing values, scale numbers, one-hot encode text columns."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def get_models(seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000,
                                                  random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300,
                                                random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05,
                                 max_depth=4, random_state=seed,
                                 eval_metric="logloss"),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=15, random_state=seed,
                                   verbose=-1),
    }


def run_plain_pipeline(name: str, seed: int) -> dict:
    set_seed(seed)
    train = load_split(name, seed, "train")
    val = load_split(name, seed, "val")
    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_val, y_val = val.drop(columns=[TARGET_COL]), val[TARGET_COL]

    # Steps 1-3: train all 4 models, score all 4 on validation
    fitted, val_scores = {}, {}
    for model_name, model in get_models(seed).items():
        pipe = Pipeline([("prep", make_preprocessor(X_train)),
                         ("model", model)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        val_scores[model_name] = compute_metrics(y_val, proba)
        fitted[model_name] = pipe

    # Step 4: pick the winner using validation ROC-AUC only
    best = max(val_scores, key=lambda m: val_scores[m]["roc_auc"])

    # Step 5: score ONLY the winner on the locked test set, once
    best_pipe = fitted[best]
    test_scores = score_on_test(
        name, seed, lambda X: best_pipe.predict_proba(X)[:, 1])

    val_auc = val_scores[best]["roc_auc"]
    return {
        "system": "plain_pipeline",
        "dataset": name,
        "seed": seed,
        "best_model": best,
        "val_score": round(val_auc, 4),
        "test_score": round(test_scores["roc_auc"], 4),
        "score_drop": round(val_auc - test_scores["roc_auc"], 4),
        "test_f1": round(test_scores["f1"], 4),
        "test_balanced_accuracy": round(test_scores["balanced_accuracy"], 4),
        "test_rare_class_recall": round(test_scores["rare_class_recall"], 4),
    }


if __name__ == "__main__":
    with open(ROOT / "configs" / "datasets.yaml") as f:
        cfg = yaml.safe_load(f)

    rows = []
    for entry in cfg["dev"]:
        data = load_dataset(entry["name"])
        for seed in SEEDS:
            make_splits(data, entry["name"], seed)
            row = run_plain_pipeline(entry["name"], seed)
            print(row)
            rows.append(row)

    table = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "plain_pipeline.csv", index=False)

    print("\nAverage over seeds:")
    print(table.groupby("dataset")[["val_score", "test_score", "score_drop"]]
          .mean().round(3))