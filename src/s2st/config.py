"""Config loading.

The single source of truth is configs/default.yaml. The factory reads it to
decide which stage implementation (dummy / real) to build, so swapping a dummy
for a real model is a one-line config edit (the README's extension path).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    """Load a YAML config, defaulting to configs/default.yaml."""
    p = Path(path) if path else DEFAULT_CONFIG
    return yaml.safe_load(p.read_text())


def resolve_path(path: str) -> Path:
    """Resolve a config-relative path against the repo root if not absolute."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)
