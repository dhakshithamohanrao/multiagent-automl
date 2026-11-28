import re

from src.agents.reporter import build_report


def _numbers(text: str) -> set:
    out = set()
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        out.add(tok)
        out.add(str(float(tok)))
        out.add(str(int(float(tok))) if float(tok) == int(float(tok)) else tok)
    return out


STATE = {
    "dataset_name": "demo", "goal": "predict the target column", "target": "class",
    "profile": {"n_rows": 600, "duplicate_rows": 0, "class_balance": {"0": 0.7, "1": 0.3}},
    "dropped": {"risk_index": "dropped by critic: single_feature_leakage {'risk_index': 1.0}"},
    "decisions": [{"action": "drop_columns", "columns": ["risk_index"], "source": "critic",
                   "reason": "critic: single_feature_leakage {'risk_index': 1.0}"}],
    "results": {"xgboost": {"val_roc_auc": 0.7863, "f1": 0.5102,
                            "balanced_accuracy": 0.6619, "rare_class_recall": 0.4167}},
    "best_model": "xgboost", "test_score": 0.7751,
    "critic_flags": [], "critic_verdict": {"verdict": "approve", "source": "code",
                                           "reason": "no serious flags"},
    "explanation": {"text": "The model relies most on duration (12%).",
                    "top_features": [{"feature": "duration", "share_percent": 12}]},
}


def test_every_report_number_comes_from_the_state():
    report = build_report(STATE)
    allowed = _numbers(str(STATE))
    # list numbering "1." and the markdown header levels are formatting, not data
    body = re.sub(r"^\s*\d+\.\s", "", report, flags=re.M)
    body = re.sub(r"^#+ ", "", body, flags=re.M)
    unknown = [n for n in re.findall(r"\d+(?:\.\d+)?", body) if n not in allowed]
    assert not unknown, f"numbers in the report that are not in the state: {unknown}"


def test_report_without_critic_or_explanation():
    s = {k: v for k, v in STATE.items() if k not in ("critic_verdict", "explanation", "test_score")}
    r = build_report(s)
    assert "did not run" in r and "Explanation" not in r