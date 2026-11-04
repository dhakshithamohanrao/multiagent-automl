import yaml
from flaml import AutoML

from src.evaluation.metrics import compute_metrics, score_on_test
from src.utils.cost_tracker import CostTracker
from src.utils.data import ROOT, TARGET_COL, load_split
from src.utils.seeds import set_seed

with open(ROOT / "configs" / "experiment.yaml") as f:
    _CFG = yaml.safe_load(f) or {}
TIME_BUDGET_S = _CFG.get("time_budget_s", 30)


def run_flaml(name: str, seed: int) -> dict:
    """Run FLAML on one dataset and seed. A crash is recorded, not hidden."""
    tracker = CostTracker()
    row = {"system": "flaml", "dataset": name, "seed": seed}
    try:
        set_seed(seed)
        train = load_split(name, seed, "train")
        val = load_split(name, seed, "val")
        X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
        X_val, y_val = val.drop(columns=[TARGET_COL]), val[TARGET_COL]

        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task="classification",
                   metric="roc_auc", time_budget=TIME_BUDGET_S,
                   seed=seed, verbose=0)

        val_auc = compute_metrics(
            y_val, automl.predict_proba(X_val)[:, 1])["roc_auc"]
        test = score_on_test(
            name, seed, lambda X: automl.predict_proba(X)[:, 1])

        row.update(
            best_model=automl.best_estimator,
            val_score=round(val_auc, 4),
            test_score=round(test["roc_auc"], 4),
            score_drop=round(val_auc - test["roc_auc"], 4),
            test_f1=round(test["f1"], 4),
            test_balanced_accuracy=round(test["balanced_accuracy"], 4),
            test_rare_class_recall=round(test["rare_class_recall"], 4),
        )
    except Exception as e:
        row.update(crashed=True, error=f"{type(e).__name__}: {e}")

    row.update(tracker.summary())
    return row


if __name__ == "__main__":
    # Quick check on one dataset and one seed (does not save anything)
    print(run_flaml("credit-g", 0))