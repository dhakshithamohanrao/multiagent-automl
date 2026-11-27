"""Make a heavily imbalanced copy of credit-g seed 0 for the step 5.6 lead test (train and val only)."""
import pandas as pd

from src.evaluation.metrics import TARGET_COL
from src.utils.helpers import PROJECT_ROOT

P = PROJECT_ROOT / "data" / "processed"
for part, n_pos in (("train", 20), ("val", 7)):
    df = pd.read_csv(P / f"credit-g_seed0_{part}.csv")
    neg, pos = df[df[TARGET_COL] == 0], df[df[TARGET_COL] == 1]
    out = pd.concat([neg, pos.sample(n=n_pos, random_state=0)]).sample(frac=1, random_state=0)
    out.to_csv(P / f"credit-g-imb_seed0_{part}.csv", index=False)
    print(part, len(out), "rows, rare share", round(n_pos / len(out), 3))