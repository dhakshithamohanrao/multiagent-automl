import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)

from src.utils.data import PROCESSED_DIR, TARGET_COL


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Score predicted probabilities of class 1 against the true 0/1 answers."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "rare_class_recall": recall_score(y_true, y_pred, pos_label=1,
                                          zero_division=0),
    }


def score_on_test(name: str, seed: int, predict_fn) -> dict:
    """The ONLY function that reads the locked test file.

    predict_fn takes the test features (a DataFrame without the target)
    and returns the probability of class 1 for each row.
    """
    test = pd.read_csv(PROCESSED_DIR / f"{name}_seed{seed}_test.csv")
    X_test = test.drop(columns=[TARGET_COL])
    proba = predict_fn(X_test)
    return compute_metrics(test[TARGET_COL], proba)