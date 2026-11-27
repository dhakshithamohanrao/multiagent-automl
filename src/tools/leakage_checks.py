"""Critic checks (real Python tests, no LLM). Each check returns (flag, details)."""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Words that suggest a column is the answer or was created after the outcome.
SUSPICIOUS_TOKENS = {"target", "label", "leak", "leaked", "outcome", "churn",
                     "cancel", "cancelled", "canceled", "answer", "ytrue"}


def encode_one_column(s: pd.Series) -> pd.DataFrame:
    """Turn one column into numbers: median fill for numeric, category codes for text."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        filled = s.fillna(s.median() if s.notna().any() else 0)
    else:
        filled = s.astype("category").cat.codes      # missing becomes -1
    return filled.to_frame(name=s.name or "x")


def single_feature_leakage(X: pd.DataFrame, y, threshold: float = 0.97, cv: int = 5):
    """Shallow tree on ONE column. A very high cross-validated AUC means the column leaks."""
    y = np.asarray(y)
    smallest_class = int(np.bincount(y.astype(int)).min())
    folds = min(cv, smallest_class)
    if folds < 2:
        return False, {}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    flagged = {}
    for col in X.columns:
        x = encode_one_column(X[col])
        try:
            score = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0),
                                    x, y, cv=splitter, scoring="roc_auc").mean()
        except ValueError:
            continue
        if score >= threshold:
            flagged[col] = round(float(score), 3)
    return bool(flagged), flagged


def id_like_columns(X: pd.DataFrame, ratio: float = 0.95, min_rows: int = 50):
    """Columns where almost every value is unique (text or integer). Float columns are skipped,
    because a real measurement such as an amount is naturally almost unique."""
    if len(X) < min_rows:
        return False, {}
    flagged = {}
    for col in X.columns:
        s = X[col].dropna()
        if len(s) == 0 or pd.api.types.is_float_dtype(s):
            continue
        r = s.nunique() / len(s)
        has_id_name = "id" in set(re.split(r"[^a-z0-9]+", str(col).lower()))
        if r >= ratio or (has_id_name and r >= 0.9):
            flagged[col] = round(float(r), 3)
    return bool(flagged), flagged


def suspicious_names(columns, target: str | None = None):
    """Column names that look like the answer, such as 'cancel_date' or 'target_copy'."""
    flagged = {}
    for col in columns:
        if col == target:
            continue
        tokens = set(re.split(r"[^a-z0-9]+", str(col).lower()))
        hit = sorted(tokens & SUSPICIOUS_TOKENS)
        if hit:
            flagged[col] = hit
    return bool(flagged), flagged