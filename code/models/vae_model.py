"""Variational Autoencoder (VAE) with β-VAE support."""

from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from code.models.base import GenerativeModel


# ── Sub-networks ──────────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── VAE nn.Module ─────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """VAE network — encoder + reparameterisation + decoder."""

    def __init__(self, input_dim: int = 2, hidden_dims: Optional[List[int]] = None,
                 latent_dim: int = 8):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.latent_dim = latent_dim
        self.encoder = _Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = _Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def generate(self, n: int, device: str = "cpu") -> np.ndarray:
        """Draw *n* samples from the latent prior, decode, return numpy."""
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z).cpu().numpy().astype(np.float32)


# ── Wrapper (implements GenerativeModel) ──────────────────────────────────────

class VAEModel(GenerativeModel):
    """Trainable VAE wrapper."""

    name = "VAE"

    def __init__(self, latent_dim: int = 8,
                 hidden_dims: Optional[List[int]] = None,
                 lr: float = 3e-4, epochs: int = 800,
                 batch_size: int = 256, beta: float = 1.0,
                 device: str = "cpu"):
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        self.device = device

        self.vae: Optional[VAE] = None
        self.train_losses: List[float] = []

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray, verbose: bool = True, **kwargs) -> "VAEModel":
        self.vae = VAE(input_dim=2, hidden_dims=self.hidden_dims,
                       latent_dim=self.latent_dim).to(self.device)

        opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(data)),
            batch_size=self.batch_size, shuffle=True,
        )

        self.train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.vae.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                recon, mu, logvar = self.vae(batch)

                recon_loss = F.mse_loss(recon, batch, reduction="sum") / batch.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                loss = recon_loss + self.beta * kl_loss

                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            sched.step()
            self.train_losses.append(epoch_loss / len(loader))

            if verbose and epoch % 200 == 0:
                print(f"    VAE epoch {epoch:4d}/{self.epochs}  "
                      f"loss = {self.train_losses[-1]:.4f}")

        return self

    # ── sample ────────────────────────────────────────────────────────────

    def sample(self, n: int) -> np.ndarray:
        return self.vae.generate(n, device=self.device)

    # ── save / load ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.vae.state_dict(),
            "train_losses": self.train_losses,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "beta": self.beta,
            "device": self.device,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "VAEModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            latent_dim=checkpoint["latent_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            lr=checkpoint["lr"],
            epochs=checkpoint["epochs"],
            batch_size=checkpoint["batch_size"],
            beta=checkpoint["beta"],
            device=map_location,
        )
        model.vae = VAE(input_dim=2, hidden_dims=checkpoint["hidden_dims"],
                        latent_dim=checkpoint["latent_dim"]).to(map_location)
        model.vae.load_state_dict(checkpoint["model_state"])
        model.vae.eval()
        model.train_losses = checkpoint["train_losses"]
        return model
