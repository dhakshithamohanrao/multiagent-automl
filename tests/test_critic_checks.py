import numpy as np
import pandas as pd

from src.tools import leakage_checks as lc
from src.tools.fairness import group_recall_gap

rng = np.random.default_rng(0)
N = 300
y = rng.integers(0, 2, N)


def _clean():
    return pd.DataFrame({"a": rng.normal(size=N), "b": rng.integers(0, 5, N),
                         "c": rng.choice(["x", "y", "z"], N)})


def test_leakage_flags_leaked_column():
    X = _clean()
    X["leak"] = y + rng.normal(0, 0.05, N)
    flag, details = lc.single_feature_leakage(X, y)
    assert flag and "leak" in details


def test_leakage_quiet_on_clean():
    assert lc.single_feature_leakage(_clean(), y)[0] is False


def test_id_column_flagged():
    X = _clean()
    X["customer_id"] = np.arange(N)
    flag, details = lc.id_like_columns(X)
    assert flag and "customer_id" in details


def test_id_quiet_on_clean():
    assert lc.id_like_columns(_clean())[0] is False


def test_suspicious_name_flagged_and_target_ignored():
    assert lc.suspicious_names(["a", "cancel_date"], "class")[0] is True
    assert lc.suspicious_names(["a", "class"], "class")[0] is False


def test_duplicate_overlap_flags_copied_rows():
    train = _clean()
    val = pd.concat([train.sample(60, random_state=0), _clean().head(40)], ignore_index=True)
    assert lc.duplicate_overlap(train, val)[0] is True


def test_duplicate_overlap_quiet_on_disjoint():
    assert lc.duplicate_overlap(_clean(), _clean())[0] is False


def test_duplicate_overlap_tolerates_natural_duplicates():
    base = pd.DataFrame({"a": rng.integers(0, 3, 400), "b": rng.integers(0, 3, 400)})
    assert lc.duplicate_overlap(base.iloc[:300], base.iloc[300:])[0] is False


def test_time_split_flags_val_earlier_than_train():
    train = pd.DataFrame({"t": pd.date_range("2020-06-01", periods=50)})
    val = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=50)})
    assert lc.time_split_check(train, val, "t")[0] is True


def test_time_split_quiet_when_val_later_or_no_column():
    train = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=50)})
    val = pd.DataFrame({"t": pd.date_range("2020-06-01", periods=50)})
    assert lc.time_split_check(train, val, "t")[0] is False
    assert lc.time_split_check(train, val, None)[0] is False


def test_imbalance_flags_rare_class_with_low_recall():
    yy = np.array([1] * 5 + [0] * 95)
    assert lc.imbalance_check(yy, rare_recall=0.1)[0] is True


def test_imbalance_quiet_when_balanced_or_recall_ok():
    assert lc.imbalance_check(np.array([0, 1] * 50))[0] is False
    assert lc.imbalance_check(np.array([1] * 5 + [0] * 95), rare_recall=0.8)[0] is False


def test_overfit_gap():
    assert lc.overfit_gap_check(1.0, 0.7)[0] is True
    assert lc.overfit_gap_check(0.85, 0.80)[0] is False


def test_fairness_flags_gap_and_is_silent_without_column():
    X = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50})
    yt = np.ones(100, dtype=int)
    yp = np.array([1] * 50 + [0] * 50)
    assert group_recall_gap(X, yt, yp, "g")[0] is True
    assert group_recall_gap(X, yt, np.ones(100, dtype=int), "g")[0] is False
    assert group_recall_gap(X, yt, yp, None)[0] is False