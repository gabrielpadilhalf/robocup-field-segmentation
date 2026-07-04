"""Configuration loading helpers."""

from pathlib import Path
import yaml


def load_config(filename: str):
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config" / filename

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
