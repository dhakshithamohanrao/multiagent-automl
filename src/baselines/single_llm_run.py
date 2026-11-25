# src/baselines/single_llm_run.py
import argparse, re, time
from pathlib import Path
import pandas as pd

from src.utils.helpers import PROJECT_ROOT, load_config, write_result
from src.utils.data import TARGET_COL, load_split
from src.llm.client import ask, get_usage
from src.tools.sandbox import run_code
from src.evaluation.metrics import score_on_test

MAX_FIXES = 2   # code retries, from the plan


def _clean(code: str) -> str:
    """Remove markdown fences if the LLM added them anyway."""
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)
    return re.sub(r"\s*```$", "", code)


def _rel(p: Path) -> str:
    return p.relative_to(PROJECT_ROOT).as_posix()


def _r(x):
    return None if x is None else round(float(x), 4)


def run_single_llm(name: str, seed: int) -> dict:
    cfg = load_config()
    t0 = time.time()
    usage0 = get_usage()
    row = {"run_id": f"single_llm|{name}|{seed}|none", "system": "single_llm",
           "dataset": name, "seed": seed, "flaw": "none",
           "crashed": False, "error": "", "flags_raised": []}

    # Only train and validation are loaded here. The test set stays locked.
    train = load_split(name, seed, "train")
    val = load_split(name, seed, "val")

    run_dir = PROJECT_ROOT / "results" / "raw" / "single_llm" / f"{name}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_p, val_p = run_dir / "train.csv", run_dir / "val.csv"
    test_p, out_p = run_dir / "test_features.csv", run_dir / "predictions.csv"
    train.to_csv(train_p, index=False)
    val.to_csv(val_p, index=False)

    template = (PROJECT_ROOT / "src/llm/prompts/single_llm.txt").read_text(encoding="utf-8")
    prompt = template.format(
        goal=f"Predict the column '{TARGET_COL}'", target=TARGET_COL,
        train_path=_rel(train_p), val_path=_rel(val_p),
        test_features_path=_rel(test_p), output_path=_rel(out_p),
        columns=train.dtypes.astype(str).to_string(),
        sample_rows=train.head(5).to_string())
    timeout = cfg.get("time_budget_s", 300)

    def predict_fn(X_test):
        """Called by score_on_test with the test features (no target)."""
        X_test.to_csv(test_p, index=False)
        script = _clean(ask(prompt))
        error = ""
        for attempt in range(MAX_FIXES + 1):
            if out_p.exists():
                out_p.unlink()
            result = run_code(script, timeout=timeout)
            error = result["error"]
            if result["success"] and out_p.exists():
                preds = pd.read_csv(out_p)
                if list(preds.columns) == ["prediction"] and len(preds) == len(X_test):
                    return preds["prediction"].values
                error = ("predictions.csv must have exactly one column "
                         "'prediction' and one row per test row")
            elif result["success"]:
                error = f"Script finished but {_rel(out_p)} was not written"
            if attempt < MAX_FIXES:
                fix = (f"{prompt}\n\nYour previous script failed with this error:\n"
                       f"{error}\n\nPrevious script:\n{script}\n\n"
                       f"Return the full corrected script only.")
                script = _clean(ask(fix))
        raise RuntimeError(error or "unknown failure")

    try:
        m = score_on_test(name, seed, predict_fn)
        row.update(test_score=_r(m.get("roc_auc")),
                   test_balanced_accuracy=_r(m.get("balanced_accuracy")),
                   test_rare_class_recall=_r(m.get("rare_class_recall")))
    except Exception as e:          # crashes are results, not hidden
        row.update(crashed=True, error=str(e)[-500:])

    usage1 = get_usage()
    row.update(time_s=round(time.time() - t0, 1),
               llm_calls=usage1["calls"] - usage0["calls"],
               tokens=usage1["tokens"] - usage0["tokens"], cost=0.0)
    write_result(row)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(run_single_llm(args.name, args.seed))