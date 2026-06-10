"""Experiment orchestration following structure.md.

Sections mapped:
  §5  — train_main()          train VAE + DDPM on all 4 classes
  §8.1 — ablation()            β and T sensitivity on Spiral
  §8.2 — conditional()         CVAE + Conditional DDPM on joint data
  §8.3 — robustness()          uniform noise contamination on Spiral
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from code.config import (
    CLASS_NAMES, N_CLASSES, MODEL_DIR, set_seed, get_device,
    VAEConfig, DiffusionConfig,
)
from code.data import load_data, get_class
from code.metrics import compute_all_metrics

set_seed(42)
DEVICE = get_device()

# ── Load data once ────────────────────────────────────────────────────────────
train, train_label = load_data("train")
test, test_label = load_data("test")


# ═══════════════════════════════════════════════════════════════════════════════
#  §5: Main Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_main() -> Dict:
    """Train VAE and DDPM on all 4 classes independently. Returns model dict."""
    from code.models.vae_model import VAEModel
    from code.models.diffusion_model import DiffusionModel

    vae_cfg = VAEConfig()
    diff_cfg = DiffusionConfig()
    all_models: Dict[str, Dict] = {}

    print("=" * 60)
    print(f"§5  MAIN TRAINING  (device = {DEVICE})")
    print("=" * 60)

    # VAE
    print("\n--- VAE ---")
    vae_models = {}
    for c in range(N_CLASSES):
        data = get_class(train, train_label, c)
        print(f"  Class {c} ({CLASS_NAMES[c]}): n={len(data)}")
        m = VAEModel(
            latent_dim=vae_cfg.latent_dim, hidden_dims=vae_cfg.hidden_dims,
            lr=vae_cfg.lr, epochs=vae_cfg.epochs, batch_size=vae_cfg.batch_size,
            beta=vae_cfg.beta, device=DEVICE,
        ).fit(data)
        m.save(MODEL_DIR / f"vae_class{c}.pt")
        vae_models[c] = m
    all_models["VAE"] = vae_models

    # DDPM
    print("\n--- DDPM ---")
    diff_models = {}
    for c in range(N_CLASSES):
        data = get_class(train, train_label, c)
        print(f"  Class {c} ({CLASS_NAMES[c]}): n={len(data)}")
        m = DiffusionModel(
            n_steps=diff_cfg.n_steps, beta_start=diff_cfg.beta_start,
            beta_end=diff_cfg.beta_end, hidden_dim=diff_cfg.hidden_dim,
            num_layers=diff_cfg.num_layers, lr=diff_cfg.lr,
            epochs=diff_cfg.epochs, batch_size=diff_cfg.batch_size,
            device=DEVICE,
        ).fit(data)
        m.save(MODEL_DIR / f"diffusion_class{c}.pt")
        diff_models[c] = m
    all_models["DDPM"] = diff_models

    print(f"\nModels saved to {MODEL_DIR}")
    return all_models


# ═══════════════════════════════════════════════════════════════════════════════
#  §8.1: Hyperparameter Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def ablation() -> Dict:
    """β (VAE) and T (DDPM) ablation on Spiral distribution."""
    from code.models.vae_model import VAEModel
    from code.models.diffusion_model import DiffusionModel

    spiral = get_class(train, train_label, 3)
    spiral_test = get_class(test, test_label, 3)
    results = {}

    print("=" * 60)
    print("§8.1  HYPERPARAMETER ABLATION (Spiral)")
    print("=" * 60)

    # VAE: β sweep
    print("\n--- VAE: β ∈ {0.1, 1.0, 5.0} ---")
    for beta in [0.1, 1.0, 5.0]:
        cfg = VAEConfig(); cfg.beta = beta
        m = VAEModel(
            latent_dim=cfg.latent_dim, hidden_dims=cfg.hidden_dims,
            lr=cfg.lr, epochs=cfg.epochs, batch_size=cfg.batch_size,
            beta=beta, device=DEVICE,
        ).fit(spiral, verbose=False)
        gen = m.sample(2000)
        met = _quick_metrics(spiral_test, gen)
        results[f"VAE_β={beta}"] = met
        print(f"  β={beta:.1f}: MMD={met['MMD']:.4f}  Prec={met['Precision']:.3f}  Cov={met['Coverage']:.3f}")

    # DDPM: T sweep
    print("\n--- DDPM: T ∈ {100, 1000} ---")
    for T_val in [100, 1000]:
        cfg = DiffusionConfig(); cfg.n_steps = T_val; cfg.epochs = 1500
        m = DiffusionModel(
            n_steps=T_val, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
            lr=cfg.lr, epochs=cfg.epochs, batch_size=cfg.batch_size,
            device=DEVICE,
        ).fit(spiral, verbose=False)
        gen = m.sample(2000)
        met = _quick_metrics(spiral_test, gen)
        results[f"DDPM_T={T_val}"] = met
        print(f"  T={T_val}: MMD={met['MMD']:.4f}  Prec={met['Precision']:.3f}  Cov={met['Coverage']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  §8.2: Conditional Generation
# ═══════════════════════════════════════════════════════════════════════════════

class _CVAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        enc = []; prev = 6
        for h in [128, 64]:
            enc.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]); prev = h
        self.enc_net = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(64, latent_dim); self.fc_logvar = nn.Linear(64, latent_dim)
        dec = []; prev = latent_dim + 4
        for h in [64, 128]:
            dec.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]); prev = h
        dec.append(nn.Linear(128, 2))
        self.dec_net = nn.Sequential(*dec)

    def forward(self, x, label):
        h = self.enc_net(torch.cat([x, label], dim=-1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        return self.dec_net(torch.cat([z, label], dim=-1)), mu, logvar

    @torch.no_grad()
    def generate(self, n, label_idx, device="cpu"):
        self.eval()
        label = torch.zeros(n, 4, device=device); label[:, label_idx] = 1.0
        z = torch.randn(n, self.latent_dim, device=device)
        return self.dec_net(torch.cat([z, label], dim=-1)).cpu().numpy().astype(np.float32)


class _CondDenoiser(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, time_emb_dim=64):
        super().__init__()
        half = time_emb_dim // 2
        self.register_buffer("base", torch.exp(torch.arange(half).float() * (-np.log(10000.0) / max(half - 1, 1))))
        self.te_dim = time_emb_dim
        layers = []; prev = 2 + 4 + time_emb_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU()]); prev = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, label):
        half = self.te_dim // 2
        emb = t[:, None].float() * self.base[None, :]
        te = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.te_dim % 2: te = F.pad(te, (0, 1))
        return self.net(torch.cat([x, label, te], dim=-1))


class CondDDPM:
    def __init__(self, n_steps=1000, hidden_dim=256, num_layers=4,
                 lr=2e-4, epochs=1500, batch_size=256, device="cpu"):
        self.n_steps = n_steps; self.lr = lr; self.epochs = epochs
        self.batch_size = batch_size; self.device = device
        self.hidden_dim = hidden_dim; self.num_layers = num_layers
        b = np.linspace(1e-4, 0.02, n_steps, dtype=np.float32)
        self._betas = b; self._alphas = 1.0 - b
        self._acp = np.cumprod(self._alphas)
        self._sacp = np.sqrt(self._acp)
        self._som = np.sqrt(1.0 - self._acp)
        self.denoiser = None; self.train_losses = []

    def fit(self, data, labels, verbose=True):
        self.denoiser = _CondDenoiser(
            hidden_dim=self.hidden_dim, num_layers=self.num_layers).to(self.device)
        sacp_t = torch.tensor(self._sacp, device=self.device)
        som_t = torch.tensor(self._som, device=self.device)
        opt = torch.optim.Adam(self.denoiser.parameters(), lr=self.lr)
        dset = TensorDataset(torch.FloatTensor(data), torch.FloatTensor(np.eye(N_CLASSES)[labels]))
        loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        self.train_losses = []
        for ep in range(1, self.epochs + 1):
            self.denoiser.train(); el = 0.0
            for xb, lb in loader:
                xb, lb = xb.to(self.device), lb.to(self.device)
                t = torch.randint(0, self.n_steps, (xb.size(0),), device=self.device)
                noise = torch.randn_like(xb)
                xt = sacp_t[t].view(-1, 1) * xb + som_t[t].view(-1, 1) * noise
                loss = F.mse_loss(self.denoiser(xt, t, lb), noise)
                opt.zero_grad(); loss.backward(); opt.step(); el += loss.item()
            self.train_losses.append(el / len(loader))
            if verbose and ep % 500 == 0:
                print(f"    CondDDPM ep {ep}/{self.epochs} loss={self.train_losses[-1]:.6f}")
        return self

    @torch.no_grad()
    def sample(self, n, label_idx):
        self.denoiser.eval()
        label = torch.zeros(n, 4, device=self.device); label[:, label_idx] = 1.0
        x = torch.randn(n, 2, device=self.device)
        for ti in range(self.n_steps - 1, -1, -1):
            t = torch.full((n,), ti, device=self.device, dtype=torch.long)
            pred = self.denoiser(x, t, label)
            a_t = self._alphas[ti]; b_t = self._betas[ti]
            x = (x - b_t / self._som[ti] * pred) / np.sqrt(a_t)
            if ti > 0: x = x + np.sqrt(b_t) * torch.randn_like(x)
        return x.cpu().numpy().astype(np.float32)


def conditional() -> Dict:
    """Train CVAE and Conditional DDPM on joint 4-class data."""
    print("\n" + "=" * 60)
    print("§8.2  CONDITIONAL GENERATION")
    print("=" * 60)

    # CVAE
    print("\n--- Training CVAE ---")
    cvae = _CVAE(latent_dim=8).to(DEVICE)
    opt = torch.optim.Adam(cvae.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)
    dset = TensorDataset(torch.FloatTensor(train), torch.FloatTensor(np.eye(N_CLASSES)[train_label]))
    loader = DataLoader(dset, batch_size=256, shuffle=True)
    for ep in range(1, 801):
        cvae.train(); el = 0.0
        for xb, lb in loader:
            xb, lb = xb.to(DEVICE), lb.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = cvae(xb, lb)
            rloss = F.mse_loss(recon, xb, reduction="sum") / xb.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
            (rloss + kl).backward(); opt.step(); el += rloss.item() + kl.item()
        sched.step()
        if ep % 200 == 0: print(f"    CVAE ep {ep}/800 loss={el/len(loader):.4f}")

    # CondDDPM
    print("\n--- Training Conditional DDPM ---")
    cddpm = CondDDPM(epochs=1500, device=DEVICE).fit(train, train_label)

    # Evaluate
    results = {"CVAE": {}, "CondDDPM": {}}
    for c in range(N_CLASSES):
        real = get_class(test, test_label, c)
        g_cvae = cvae.generate(2000, c, DEVICE)
        g_cddpm = cddpm.sample(2000, c)
        results["CVAE"][CLASS_NAMES[c]] = _quick_metrics(real, g_cvae)
        results["CondDDPM"][CLASS_NAMES[c]] = _quick_metrics(real, g_cddpm)
        print(f"  {CLASS_NAMES[c]}: CVAE Cov={results['CVAE'][CLASS_NAMES[c]]['Coverage']:.3f}  "
              f"CondDDPM Cov={results['CondDDPM'][CLASS_NAMES[c]]['Coverage']:.3f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  §8.3: Robustness
# ═══════════════════════════════════════════════════════════════════════════════

def robustness() -> Dict:
    """Uniform noise contamination on Spiral at ρ ∈ {0%, 5%, 10%}."""
    from code.models.vae_model import VAEModel
    from code.models.diffusion_model import DiffusionModel

    spiral = get_class(train, train_label, 3)
    spiral_test = get_class(test, test_label, 3)
    n_clean = len(spiral)
    results = {}

    print("\n" + "=" * 60)
    print("§8.3  ROBUSTNESS (Spiral)")
    print("=" * 60)

    for rho in [0.0, 0.05, 0.10]:
        print(f"\n--- ρ = {rho:.0%} ---")
        n_noise = int(n_clean * rho)
        noise_pts = np.random.default_rng(42).uniform(-4, 4, size=(n_noise, 2)).astype(np.float32)
        corrupted = np.concatenate([spiral, noise_pts])
        corrupted = corrupted[np.random.default_rng(42).permutation(len(corrupted))]

        # VAE
        vae_cfg = VAEConfig()
        vae = VAEModel(
            latent_dim=vae_cfg.latent_dim, hidden_dims=vae_cfg.hidden_dims,
            lr=vae_cfg.lr, epochs=vae_cfg.epochs, batch_size=vae_cfg.batch_size,
            beta=vae_cfg.beta, device=DEVICE,
        ).fit(corrupted, verbose=False)
        mv = _quick_metrics(spiral_test, vae.sample(2000))

        # DDPM
        diff_cfg = DiffusionConfig()
        diff = DiffusionModel(
            n_steps=diff_cfg.n_steps, hidden_dim=diff_cfg.hidden_dim,
            num_layers=diff_cfg.num_layers, lr=diff_cfg.lr, epochs=diff_cfg.epochs,
            batch_size=diff_cfg.batch_size, device=DEVICE,
        ).fit(corrupted, verbose=False)
        md = _quick_metrics(spiral_test, diff.sample(2000))

        results[f"ρ={rho:.0%}"] = {"VAE": mv, "DDPM": md}
        print(f"    VAE:  MMD={mv['MMD']:.4f}  Prec={mv['Precision']:.3f}  Cov={mv['Coverage']:.3f}")
        print(f"    DDPM: MMD={md['MMD']:.4f}  Prec={md['Precision']:.3f}  Cov={md['Coverage']:.3f}")

    return results


# ── Fast metrics for extension experiments ────────────────────────────────────

def _quick_metrics(real, gen, n=500) -> Dict:
    from scipy.spatial import cKDTree, distance
    rng = np.random.default_rng(42)
    X = real[rng.choice(len(real), min(len(real), n), replace=False)]
    Y = gen[rng.choice(len(gen), min(len(gen), n), replace=False)]

    # MMD
    pooled = np.concatenate([X, Y])
    ds = distance.cdist(pooled[:200], pooled[:200])
    sigma = float(np.median(ds[ds > 0])) if ds[ds > 0].size else 1.0
    sigma = max(sigma, 1e-3); g = 0.5 / (sigma * sigma)
    def rbf(A, B): return float(np.mean(np.exp(-g * distance.cdist(A, B, "sqeuclidean"))))
    mmd = max(np.sqrt(rbf(X, X) + rbf(Y, Y) - 2 * rbf(X, Y)), 0.0)

    # Precision & Coverage
    tr = cKDTree(X); r_k, _ = tr.query(X, k=6); radii = r_k[:, 5]
    dg, _ = tr.query(Y, k=1); prec = float(np.mean(dg <= np.percentile(radii, 95)))
    ty = cKDTree(Y); dr, _ = ty.query(X, k=1); cov = float(np.mean(dr <= radii))

    return {"MMD": mmd, "Precision": prec, "Coverage": cov}
