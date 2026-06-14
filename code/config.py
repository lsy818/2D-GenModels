"""Project configuration: paths, seeds, hyper-parameters, class metadata."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

CLASS_NAMES = ["Gaussian Mixture", "Ring", "Two Moons", "Spiral"]
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
N_CLASSES = 4
SEED = 42

# ── Model hyper-parameters ────────────────────────────────────────────────────

@dataclass
class VAEConfig:
    latent_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    lr: float = 3e-4
    epochs: int = 800
    batch_size: int = 256
    beta: float = 1.0

@dataclass
class DiffusionConfig:
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 256
    num_layers: int = 4
    lr: float = 2e-4
    epochs: int = 1500
    batch_size: int = 256

# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
