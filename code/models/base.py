"""Abstract base class for all generative models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class GenerativeModel(ABC):
    """Interface that every generative model must implement."""

    name: str = "base"

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> "GenerativeModel":
        """Train the model on data of shape ``(n_samples, 2)``."""
        ...

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Generate *n* samples, returning ``(n, 2)`` float32 array."""
        ...

    def save(self, path: Path) -> None:
        """Persist the model to disk (default: pickle)."""
        import pickle
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "GenerativeModel":
        """Load a model from disk (default: pickle)."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
