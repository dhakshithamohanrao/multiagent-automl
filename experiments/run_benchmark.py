import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets "src" be imported when run as a script

import pandas as pd
import yaml
from src.graph.runner import run_agents_no_critic, run_agents_with_critic

from src.baselines.fixed_pipeline import run_plain_pipeline
from src.baselines.flaml_run import run_flaml
from src.utils.cost_tracker import CostTracker
from src.utils.data import load_dataset, make_splits
from src.utils.helpers import RESULTS_FILE, finished_run_ids, write_result

SYSTEMS = {
    "plain_pipeline": run_plain_pipeline,  # reference only, not one of the 5 systems
    "flaml": run_flaml,
    "agents_no_critic": run_agents_no_critic,  # system 4
    "agents_with_critic": run_agents_with_critic,  # system 5
}

def export_table() -> None:
    """Copy all saved rows into a CSV table for the write-up."""
    if not RESULTS_FILE.exists():
        return
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.read_json(RESULTS_FILE, lines=True).to_csv(
        out / "baselines.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="dev", choices=["dev", "final"])
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--limit", type=int, default=None, help="stop after this many NEW runs")
    args = parser.parse_args()

    with open(ROOT / "configs" / "experiment.yaml") as f:
        seeds = yaml.safe_load(f)["seeds"]
    with open(ROOT / "configs" / "datasets.yaml") as f:
        datasets = yaml.safe_load(f)[args.set] or []

    done = finished_run_ids()
    jobs = [(system, d["name"], seed)
            for d in datasets for seed in seeds for system in args.systems]
    ran = 0
    for i, (system, name, seed) in enumerate(jobs, 1):
        run_id = f"{system}|{name}|{seed}|none"
        if run_id in done:
            print(f"[{i}/{len(jobs)}] skip (already done): {run_id}")
            continue
        if args.limit is not None and ran >= args.limit:
                break
        ran += 1

        print(f"[{i}/{len(jobs)}] running: {run_id}")
        make_splits(load_dataset(name), name, seed)

        tracker = CostTracker()
        try:
            row = SYSTEMS[system](name, seed)
        except Exception as e:  # a crash is saved, not hidden
            row = {"system": system, "dataset": name, "seed": seed,
                   "crashed": True, "error": f"{type(e).__name__}: {e}"}
        for key, value in tracker.summary().items():
            row.setdefault(key, value)

        write_result(row)
        print("   ", row)

    export_table()


if __name__ == "__main__":
    main()