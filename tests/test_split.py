from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.utils.data as data
from src.utils.data import TARGET_COL, load_split, make_splits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_df():
    """500 fake customers with a unique row_id and 20% positives."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"row_id": range(500), "x": rng.normal(size=500)})
    df[TARGET_COL] = (np.arange(500) % 5 == 0).astype(int)
    return df


def read_parts(folder, name, seed):
    return {p: pd.read_csv(folder / f"{name}_seed{seed}_{p}.csv")
            for p in ("train", "val", "test")}


def test_splits_do_not_overlap(demo_df, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path)
    make_splits(demo_df, "demo", seed=0)
    parts = read_parts(tmp_path, "demo", 0)

    ids = {p: set(frame["row_id"]) for p, frame in parts.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert len(ids["train"] | ids["val"] | ids["test"]) == 500
    assert [len(parts[p]) for p in ("train", "val", "test")] == [300, 100, 100]
    for frame in parts.values():
        assert frame[TARGET_COL].mean() == pytest.approx(0.2, abs=0.02)


def test_same_seed_gives_same_split(demo_df, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path)
    make_splits(demo_df, "demo", seed=1)
    first = read_parts(tmp_path, "demo", 1)
    make_splits(demo_df, "demo", seed=1)
    second = read_parts(tmp_path, "demo", 1)
    for part in first:
        assert first[part].equals(second[part])


def test_load_split_refuses_test():
    with pytest.raises(ValueError):
        load_split("credit-g", 0, "test")


def test_only_score_on_test_reads_test_file():
    offenders = []
    for folder in ("src", "experiments", "app"):
        for path in (ROOT / folder).rglob("*.py"):
            if "_test.csv" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == ["src/evaluation/metrics.py"]