from pathlib import Path

import pandas as pd
import yaml
import openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET_COL = "target"  # every dataset's target column is renamed to this


def _find_entry(name: str) -> dict:
    with open(ROOT / "configs" / "datasets.yaml") as f:
        cfg = yaml.safe_load(f)
    for group in ("dev", "final"):
        for entry in cfg.get(group) or []:
            if entry["name"] == name:
                return entry
    raise ValueError(f"Dataset '{name}' not found in configs/datasets.yaml")


def load_dataset(name: str, force_download: bool = False) -> pd.DataFrame:
    """Download a dataset from OpenML, encode the target as 0/1, save raw CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / f"{name}.csv"
    if csv_path.exists() and not force_download:
        return pd.read_csv(csv_path)

    entry = _find_entry(name)
    dataset = openml.datasets.get_dataset(entry["openml_id"])
    X, y, _, _ = dataset.get_data(target=entry["target"])

    # The rarer class becomes 1 (the "positive" class we care about finding)
    positive = y.value_counts().index[-1]
    print(f"{name}: class '{positive}' -> 1, all others -> 0")
    df = X.copy()
    df[TARGET_COL] = (y == positive).astype(int).values

    df.to_csv(csv_path, index=False)
    return df


def make_splits(df: pd.DataFrame, name: str, seed: int,
                test_size: float = 0.2, val_size: float = 0.25) -> dict:
    """Locked test set first (20%), then train/validation (60% / 20%)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rest, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=seed)
    train, val = train_test_split(
        rest, test_size=val_size, stratify=rest[TARGET_COL], random_state=seed)

    paths = {}
    for part, frame in [("train", train), ("val", val), ("test", test)]:
        path = PROCESSED_DIR / f"{name}_seed{seed}_{part}.csv"
        frame.to_csv(path, index=False)
        paths[part] = str(path)
    return paths

def load_split(name: str, seed: int, part: str) -> pd.DataFrame:
    """Load the train or validation file. The test file is never loaded here."""
    if part not in ("train", "val"):
        raise ValueError("load_split only gives train or val. "
                         "The test set is read only by score_on_test().")
    return pd.read_csv(PROCESSED_DIR / f"{name}_seed{seed}_{part}.csv")

if __name__ == "__main__":
    with open(ROOT / "configs" / "datasets.yaml") as f:
        cfg = yaml.safe_load(f)
    for entry in cfg["dev"]:
        data = load_dataset(entry["name"])
        paths = make_splits(data, entry["name"], seed=0)
        print(entry["name"], data.shape,
              "positive rate:", round(data[TARGET_COL].mean(), 3))  