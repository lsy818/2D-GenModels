"""Denoising Diffusion Probabilistic Model (DDPM) for 2D data."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from code.models.base import GenerativeModel


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-np.log(10000.0) / max(half - 1, 1))
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


# ── Denoiser MLP ──────────────────────────────────────────────────────────────

class _Denoiser(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256,
                 num_layers: int = 4, time_emb_dim: int = 64):
        super().__init__()
        self.time_emb = _TimeEmbedding(time_emb_dim)

        layers: List[nn.Module] = []
        in_dim = input_dim + time_emb_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_emb(t)], dim=-1))


# ── DDPM Model ────────────────────────────────────────────────────────────────

class DiffusionModel(GenerativeModel):
    """DDPM with linear noise schedule."""

    name = "Diffusion"

    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, hidden_dim: int = 256,
                 num_layers: int = 4, lr: float = 2e-4, epochs: int = 1500,
                 batch_size: int = 256, device: str = "cpu"):
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Build schedule (kept as numpy for CPU indexing; will be mirrored on GPU)
        self._betas = np.linspace(beta_start, beta_end, n_steps, dtype=np.float32)
        self._alphas = 1.0 - self._betas
        self._alphas_cumprod = np.cumprod(self._alphas)
        self._sqrt_alphas_cumprod = np.sqrt(self._alphas_cumprod)
        self._sqrt_one_minus = np.sqrt(1.0 - self._alphas_cumprod)

        # Pre-computed torch tensors (moved to device lazily in fit)
        self._sqrt_alphas_cumprod_t = None
        self._sqrt_one_minus_t = None

        self.denoiser: Optional[_Denoiser] = None
        self.train_losses: List[float] = []

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray, verbose: bool = True, **kwargs) -> "DiffusionModel":
        self.denoiser = _Denoiser(
            input_dim=2, hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        # Move schedule tensors to device
        self._sqrt_alphas_cumprod_t = torch.tensor(
            self._sqrt_alphas_cumprod, device=self.device)
        self._sqrt_one_minus_t = torch.tensor(
            self._sqrt_one_minus, device=self.device)

        opt = torch.optim.Adam(self.denoiser.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(data)),
            batch_size=self.batch_size, shuffle=True,
        )

        self.train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.denoiser.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                t = torch.randint(0, self.n_steps, (batch.size(0),),
                                  device=self.device)

                # Forward diffusion (index from pre-computed GPU tensors)
                sqrt_a = self._sqrt_alphas_cumprod_t[t].view(-1, 1)
                sqrt_1ma = self._sqrt_one_minus_t[t].view(-1, 1)
                noise = torch.randn_like(batch)
                xt = sqrt_a * batch + sqrt_1ma * noise

                pred = self.denoiser(xt, t)
                loss = F.mse_loss(pred, noise)

                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            self.train_losses.append(epoch_loss / len(loader))

            if verbose and epoch % 500 == 0:
                print(f"    Diffusion epoch {epoch:4d}/{self.epochs}  "
                      f"loss = {self.train_losses[-1]:.6f}")

        return self

    # ── sample (reverse process) ──────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray:
        self.denoiser.eval()
        x = torch.randn(n, 2, device=self.device)

        # Ensure schedule tensors are on the right device
        if self._sqrt_one_minus_t is None or self._sqrt_one_minus_t.device != self.device:
            self._sqrt_alphas_cumprod_t = torch.tensor(
                self._sqrt_alphas_cumprod, device=self.device)
            self._sqrt_one_minus_t = torch.tensor(
                self._sqrt_one_minus, device=self.device)

        for t_idx in range(self.n_steps - 1, -1, -1):
            t = torch.full((n,), t_idx, device=self.device, dtype=torch.long)
            pred_noise = self.denoiser(x, t)

            alpha_t = self._alphas[t_idx]
            beta_t = self._betas[t_idx]
            coef = beta_t / self._sqrt_one_minus[t_idx]
            mean = (x - coef * pred_noise) / np.sqrt(alpha_t)

            if t_idx > 0:
                x = mean + np.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

        return x.cpu().numpy().astype(np.float32)

    # ── save / load ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.denoiser.state_dict(),
            "train_losses": self.train_losses,
            "n_steps": self.n_steps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "DiffusionModel":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            n_steps=ckpt["n_steps"], beta_start=ckpt["beta_start"],
            beta_end=ckpt["beta_end"], hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"], lr=ckpt["lr"],
            epochs=ckpt["epochs"], batch_size=ckpt["batch_size"],
            device=map_location,
        )
        model.denoiser = _Denoiser(
            input_dim=2, hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
        ).to(map_location)
        model.denoiser.load_state_dict(ckpt["model_state"])
        model.denoiser.eval()
        model.train_losses = ckpt["train_losses"]
        return model
