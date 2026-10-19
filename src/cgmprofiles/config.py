"""Load config/paper1.yaml once and resolve data, restricted and results paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "paper1.yaml"
DATA_ENV = "AIREADI_DATA_DIR"
RESTRICTED_ENV = "AIREADI_RESTRICTED_DIR"


class Config(SimpleNamespace):
    """Nested read-only view of the YAML file: attribute access, plus ``raw`` for the dict."""

    raw: dict

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and all(isinstance(k, str) and k.isidentifier() for k in value):
        cfg = Config(**{k: _wrap(v) for k, v in value.items()})
        cfg.raw = value
        return cfg
    return value


@lru_cache(maxsize=1)
def get_config() -> Config:
    """The configuration, loaded once per process."""
    return _wrap(yaml.safe_load(CONFIG_PATH.read_text()))


def data_dir() -> Path:
    """Root of the AI-READI data, from AIREADI_DATA_DIR."""
    value = os.environ.get(DATA_ENV)
    if not value:
        raise RuntimeError(f"Set {DATA_ENV} to the folder that holds enriched_multimodal/ and the split folder")
    return Path(value).expanduser()


def data_path(key: str) -> Path:
    """Absolute path of a data file named in config.data (relative to AIREADI_DATA_DIR)."""
    return data_dir() / getattr(get_config().data, key)


def restricted_dir(*parts: str) -> Path:
    """Folder for participant-level intermediates (gitignored), created on demand."""
    base = os.environ.get(RESTRICTED_ENV)
    root = Path(base).expanduser() if base else REPO_ROOT / get_config().data.restricted_dir
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(*parts: str) -> Path:
    """Folder for tracked aggregate outputs, created on demand."""
    path = REPO_ROOT.joinpath("results", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
