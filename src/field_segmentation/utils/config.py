"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(filename: str | Path) -> dict[str, Any]:
    """Load one YAML config file."""

    return _read_yaml(_resolve_config_path(filename))


def _resolve_config_path(filename: str | Path) -> Path:
    path = Path(filename)
    if path.exists():
        return path

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Config file not found: {filename}")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML config must be a mapping: {path}")
    return data
