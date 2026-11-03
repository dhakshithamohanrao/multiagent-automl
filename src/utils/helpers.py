from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "experiment.yaml") -> dict:
    """Load a YAML file from the configs/ folder."""
    path = PROJECT_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)