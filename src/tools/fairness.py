"""Group fairness check (optional, used only when a sensitive column is configured)."""
import numpy as np
import pandas as pd


def group_recall_gap(X: pd.DataFrame, y, y_pred, sensitive_column: str | None,
                     max_gap: float = 0.20, min_group_size: int = 20):
    """Recall on the positive class per group. Flag if the best and worst group differ too much."""
    if not sensitive_column or sensitive_column not in X.columns:
        return False, {}
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    recalls = {}
    for g, idx in X.groupby(sensitive_column).indices.items():
        pos = y[idx] == 1
        if len(idx) < min_group_size or pos.sum() == 0:
            continue
        recalls[str(g)] = round(float((y_pred[idx][pos] == 1).mean()), 4)
    if len(recalls) < 2:
        return False, {"recall_by_group": recalls}
    gap = max(recalls.values()) - min(recalls.values())
    return gap > max_gap, {"recall_by_group": recalls, "gap": round(gap, 4)}