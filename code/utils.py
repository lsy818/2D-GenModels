"""Miscellaneous utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from code.config import SystemConfig

# Re-export for convenience
set_seed_all = SystemConfig.set_seed_all
get_device = SystemConfig.get_device


def save_pickle(obj: Any, path: Path) -> None:
    """Save an arbitrary Python object with pickle."""
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load a pickled Python object."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
