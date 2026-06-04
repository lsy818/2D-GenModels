"""Configuration constants and dataclasses for the project.

All paths are relative to the project root.  Run scripts from the project
root directory using ``python -m code.main_analysis`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# ── Path constants (relative to project root) ─────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"

CLASS_NAMES = ["Gaussian Mixture", "Ring", "Two Moons", "Spiral"]
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
N_CLASSES = 4


# ── Configuration dataclasses ─────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Parameters for data analysis / visualisation."""
    n_neighbors_noise: int = 30
    n_neighbors_curvature: int = 15
    n_neighbors_thickness: int = 20
    kde_bandwidth: float = 0.2
    grid_resolution: int = 80
    subsample_vis: int = 20
    n_subsample_pairwise: int = 800
    max_range: float = 4.0


@dataclass
class ModelConfig:
    """Hyper-parameters shared across all generative models."""

    # ── KDE ───────────────────────────────────────────────────────────────
    kde_cv_bandwidths: List[float] = field(
        default_factory=lambda: list(np.logspace(-2, 1, 20))
    )
    kde_mcmc_burn_in: int = 500
    kde_mcmc_thin: int = 5

    # ── GMM ───────────────────────────────────────────────────────────────
    gmm_max_components: int = 20
    gmm_covariance_type: str = "full"

    # ── VAE ───────────────────────────────────────────────────────────────
    vae_latent_dim: int = 8
    vae_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    vae_epochs: int = 800
    vae_batch_size: int = 256
    vae_lr: float = 3e-4
    vae_beta: float = 1.0

    # ── Diffusion (DDPM) ──────────────────────────────────────────────────
    diffusion_n_steps: int = 1000
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 0.02
    diffusion_hidden_dim: int = 256
    diffusion_num_layers: int = 4
    diffusion_epochs: int = 1500
    diffusion_batch_size: int = 256
    diffusion_lr: float = 2e-4


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    seed: int = 42
    n_samples_generate: int = 2000
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


class SystemConfig:
    """System utilities — seed setting, device selection."""

    @staticmethod
    def set_seed_all(seed: int = 42) -> None:
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

    @staticmethod
    def get_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
