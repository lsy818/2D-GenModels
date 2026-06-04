"""Data loading and pre-processing utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from code.config import DATA_DIR


def load_data(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Load a data split.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test"``, ``"hidden_test"``.

    Returns
    -------
    points : np.ndarray  shape (N, 2), float32
    labels : np.ndarray  shape (N,),   int64
    """
    data = np.load(DATA_DIR / f"{split}.npy").astype(np.float32)
    labels = np.load(DATA_DIR / f"{split}_label.npy").astype(np.int64)
    return data, labels


def load_all_splits() -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Return ``(train, test, hidden_test)``, each as ``(points, labels)``."""
    return load_data("train"), load_data("test"), load_data("hidden_test")


def get_class_data(data: np.ndarray, labels: np.ndarray, class_idx: int
                   ) -> np.ndarray:
    """Extract samples belonging to *class_idx*."""
    return data[labels == class_idx]
