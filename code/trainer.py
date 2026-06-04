"""Training orchestration — train all models on all classes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from code.config import CLASS_NAMES, N_CLASSES, MODEL_DIR, ModelConfig
from code.dataset import load_data, get_class_data
from code.models import KDEModel, GMMModel, VAEModel, DiffusionModel
from code.utils import get_device


# Type alias: model_type -> class_idx -> model
ModelDict = Dict[str, Dict[int, object]]


def train_all_models(cfg: ModelConfig) -> Tuple[ModelDict, np.ndarray, np.ndarray]:
    """Train KDE, GMM, VAE, and Diffusion on each of the four classes.

    Returns
    -------
    all_models : ModelDict
    train : np.ndarray
    train_label : np.ndarray
    """
    train, train_label = load_data("train")
    device = get_device()

    all_models: ModelDict = {}

    # ── KDE ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training: KDE")
    print("=" * 60)
    kde_models = {}
    for c in range(N_CLASSES):
        data = get_class_data(train, train_label, c)
        print(f"\n  Class {c}: {CLASS_NAMES[c]}  (n = {len(data)})")
        model = KDEModel().fit(data, cv_bandwidths=cfg.kde_cv_bandwidths)
        model.save(MODEL_DIR / f"kde_class{c}.pt")
        kde_models[c] = model
    all_models["KDE"] = kde_models

    # ── GMM ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training: GMM")
    print("=" * 60)
    gmm_models = {}
    for c in range(N_CLASSES):
        data = get_class_data(train, train_label, c)
        print(f"\n  Class {c}: {CLASS_NAMES[c]}  (n = {len(data)})")
        model = GMMModel(
            covariance_type=cfg.gmm_covariance_type,
        ).fit(data, max_components=cfg.gmm_max_components)
        model.save(MODEL_DIR / f"gmm_class{c}.pt")
        gmm_models[c] = model
    all_models["GMM"] = gmm_models

    # ── VAE ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training: VAE  (device = {device})")
    print("=" * 60)
    vae_models = {}
    for c in range(N_CLASSES):
        data = get_class_data(train, train_label, c)
        print(f"\n  Class {c}: {CLASS_NAMES[c]}  (n = {len(data)})")
        model = VAEModel(
            latent_dim=cfg.vae_latent_dim,
            hidden_dims=cfg.vae_hidden_dims,
            lr=cfg.vae_lr, epochs=cfg.vae_epochs,
            batch_size=cfg.vae_batch_size, beta=cfg.vae_beta,
            device=device,
        ).fit(data)
        model.save(MODEL_DIR / f"vae_class{c}.pt")
        vae_models[c] = model
    all_models["VAE"] = vae_models

    # ── Diffusion ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training: Diffusion  (device = {device})")
    print("=" * 60)
    diff_models = {}
    for c in range(N_CLASSES):
        data = get_class_data(train, train_label, c)
        print(f"\n  Class {c}: {CLASS_NAMES[c]}  (n = {len(data)})")
        model = DiffusionModel(
            n_steps=cfg.diffusion_n_steps,
            beta_start=cfg.diffusion_beta_start,
            beta_end=cfg.diffusion_beta_end,
            hidden_dim=cfg.diffusion_hidden_dim,
            num_layers=cfg.diffusion_num_layers,
            lr=cfg.diffusion_lr, epochs=cfg.diffusion_epochs,
            batch_size=cfg.diffusion_batch_size,
            device=device,
        ).fit(data)
        model.save(MODEL_DIR / f"diffusion_class{c}.pt")
        diff_models[c] = model
    all_models["Diffusion"] = diff_models

    print(f"\nAll models saved to {MODEL_DIR.resolve()}")
    return all_models, train, train_label
