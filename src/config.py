"""Load config.yaml and .env once, so every script shares the same settings."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return the parsed config.yaml as a dictionary."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Return an absolute path for a key under `paths:` in config.yaml, creating its folder."""
    p = ROOT / load_config()["paths"][key]
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def get_key(name: str) -> str | None:
    """Read an API key from the environment (.env). Returns None if it is missing or empty."""
    value = os.getenv(name, "").strip()
    return value or None
