"""Dev-only demo: credit-g with a leaked column added to train and val."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import TARGET_COL as T
from src.utils.helpers import PROJECT_ROOT

P = PROJECT_ROOT / "data" / "processed"
rng = np.random.default_rng(0)
for part in ("train", "val"):
    d = pd.read_csv(P / f"credit-g_seed0_{part}.csv")
    d["risk_index"] = d[T] + rng.normal(0, 0.05, len(d))
    d.to_csv(P / f"credit-g-leak_seed0_{part}.csv", index=False)
print("written")