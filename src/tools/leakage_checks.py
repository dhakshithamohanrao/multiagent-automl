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

def duplicate_overlap(train: pd.DataFrame, val: pd.DataFrame, max_share: float = 0.0):
    """Share of validation rows that also appear in train (hash of every row)."""
    cols = [c for c in train.columns if c in val.columns]
    if not cols or len(val) == 0:
        return False, {}
    h_train = set(pd.util.hash_pandas_object(train[cols], index=False))
    h_val = pd.util.hash_pandas_object(val[cols], index=False)
    n_overlap = int(h_val.isin(h_train).sum())
    share = n_overlap / len(val)
    details = {"overlap_rows": n_overlap, "share_of_val": round(float(share), 4)}
    return share > max_share, details


def time_split_check(train: pd.DataFrame, val: pd.DataFrame, time_column: str | None):
    """When a date column exists, validation must be later than training."""
    if not time_column or time_column not in train.columns or time_column not in val.columns:
        return False, {}
    t_train = pd.to_datetime(train[time_column], errors="coerce")
    t_val = pd.to_datetime(val[time_column], errors="coerce")
    if t_train.notna().sum() == 0 or t_val.notna().sum() == 0:
        return False, {}
    train_max, val_min = t_train.max(), t_val.min()
    flag = bool(val_min < train_max)
    return flag, {"train_max": str(train_max), "val_min": str(val_min)}


def imbalance_check(y, rare_recall: float | None = None,
                    rare_threshold: float = 0.10, recall_threshold: float = 0.30):
    """Rare class under the threshold, and optionally a low recall on that class."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    ratio = float(counts.min() / counts.sum())
    details = {"rare_class_ratio": round(ratio, 4)}
    flag = ratio < rare_threshold
    if rare_recall is not None:
        details["rare_class_recall"] = round(float(rare_recall), 4)
        flag = flag and rare_recall < recall_threshold
    return bool(flag), details


def overfit_gap_check(train_score: float, val_score: float, max_gap: float = 0.10):
    """Training score minus validation score. A big gap means the model memorised."""
    gap = float(train_score) - float(val_score)
    return gap > max_gap, {"train": round(float(train_score), 4),
                           "val": round(float(val_score), 4),
                           "gap": round(gap, 4)}