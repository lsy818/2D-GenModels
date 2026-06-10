"""Extension experiments for the final project.

Extension 1: Hyperparameter ablation (β for VAE, T for DDPM)
Extension 2: Conditional generation (CVAE + Conditional DDPM)
Extension 3: Robustness to uniform noise corruption
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

from code.config import SystemConfig, CLASS_NAMES, N_CLASSES
SystemConfig.set_seed_all(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Data ─────────────────────────────────────────────────────────────────────
train = np.load(str(ROOT / "data/train.npy")).astype(np.float32)
train_label = np.load(str(ROOT / "data/train_label.npy")).astype(np.int64)
test = np.load(str(ROOT / "data/test.npy")).astype(np.float32)
test_label = np.load(str(ROOT / "data/test_label.npy")).astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(real: np.ndarray, gen: np.ndarray) -> Dict[str, float]:
    """Fast MMD, Precision, Coverage."""
    from scipy import spatial
    from scipy.spatial.distance import cdist
    n = min(len(real), len(gen), 500)
    rng = np.random.default_rng(42)
    X = real[rng.choice(len(real), n, replace=False)]
    Y = gen[rng.choice(len(gen), n, replace=False)]

    # MMD (RBF, median bandwidth)
    pooled = np.concatenate([X, Y])
    dists = cdist(pooled[:200], pooled[:200])
    sigma = float(np.median(dists[dists > 0])) if dists[dists > 0].size else 1.0
    sigma = max(sigma, 1e-3)
    gamma = 0.5 / (sigma * sigma)

    def rbf(A, B):
        sq = cdist(A, B, "sqeuclidean")
        return float(np.mean(np.exp(-gamma * sq)))
    mmd2 = rbf(X, X) + rbf(Y, Y) - 2 * rbf(X, Y)
    mmd = max(np.sqrt(mmd2), 0.0)

    # Precision & Coverage (k=5)
    tree_r = spatial.cKDTree(X)
    r_k, _ = tree_r.query(X, k=6)
    radii = r_k[:, 5]
    d_g2r, _ = tree_r.query(Y, k=1)
    prec = float(np.mean(d_g2r <= np.percentile(radii, 95)))
    tree_y = spatial.cKDTree(Y)
    d_r2y, _ = tree_y.query(X, k=1)
    cov = float(np.mean(d_r2y <= radii))

    return {"MMD": mmd, "Precision": prec, "Coverage": cov}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTENSION 1: HYPERPARAMETER ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_extension_1():
    """VAE β ablation and DDPM T ablation on Spiral."""
    print("\n" + "=" * 60)
    print("EXTENSION 1: Hyperparameter Ablation (Spiral)")
    print("=" * 60)

    from code.models.vae_model import VAEModel
    from code.models.diffusion_model import DiffusionModel

    spiral_data = train[train_label == 3]
    spiral_test = test[test_label == 3]

    results = {}

    # ── VAE: β ∈ {0.1, 1.0, 5.0} ───────────────────────────────────────
    print("\n--- VAE β ablation ---")
    for beta in [0.1, 1.0, 5.0]:
        print(f"  Training VAE with β={beta} ...")
        vae = VAEModel(
            latent_dim=8, hidden_dims=[128, 64],
            lr=3e-4, epochs=800, batch_size=256,
            beta=beta, device=DEVICE,
        ).fit(spiral_data, verbose=False)
        gen = vae.sample(2000)
        m = compute_metrics(spiral_test, gen)
        results[f"VAE_β={beta}"] = m
        print(f"    MMD={m['MMD']:.4f}  Prec={m['Precision']:.3f}  Cov={m['Coverage']:.3f}")

    # ── DDPM: T ∈ {100, 1000} ───────────────────────────────────────────
    print("\n--- DDPM T ablation ---")
    for T_val in [100, 1000]:
        print(f"  Training DDPM with T={T_val} ...")
        diff = DiffusionModel(
            n_steps=T_val, hidden_dim=256, num_layers=4,
            lr=2e-4, epochs=1500, batch_size=256,
            device=DEVICE,
        ).fit(spiral_data, verbose=False)
        gen = diff.sample(2000)
        m = compute_metrics(spiral_test, gen)
        results[f"DDPM_T={T_val}"] = m
        print(f"    MMD={m['MMD']:.4f}  Prec={m['Precision']:.3f}  Cov={m['Coverage']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTENSION 2: CONDITIONAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class _CondEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=None, latent_dim=8):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x_cat):
        h = self.net(x_cat)
        return self.fc_mu(h), self.fc_logvar(h)


class _CondDecoder(nn.Module):
    def __init__(self, latent_dim=8, hidden_dims=None, output_dim=2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        layers = []
        prev = latent_dim + 4  # z + one-hot label
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_cat):
        return self.net(z_cat)


class CVAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _CondEncoder(input_dim=6, hidden_dims=[128, 64], latent_dim=latent_dim)
        self.decoder = _CondDecoder(latent_dim=latent_dim, hidden_dims=[64, 128], output_dim=2)

    def forward(self, x, label_onehot):
        x_cat = torch.cat([x, label_onehot], dim=-1)
        mu, logvar = self.encoder(x_cat)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        z_cat = torch.cat([z, label_onehot], dim=-1)
        return self.decoder(z_cat), mu, logvar

    @torch.no_grad()
    def generate(self, n, label_idx, device="cpu"):
        self.eval()
        label = torch.zeros(n, 4, device=device)
        label[:, label_idx] = 1.0
        z = torch.randn(n, self.latent_dim, device=device)
        z_cat = torch.cat([z, label], dim=-1)
        return self.decoder(z_cat).cpu().numpy().astype(np.float32)


class _CondDenoiser(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=4, time_emb_dim=64):
        super().__init__()
        half = time_emb_dim // 2
        t_emb = torch.exp(torch.arange(half).float() * (-np.log(10000.0) / max(half - 1, 1)))
        self.register_buffer("t_emb_base", t_emb)
        self.time_emb_dim = time_emb_dim

        layers = []
        in_dim = input_dim + time_emb_dim  # [x_t, label_onehot] + t_emb
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, label_onehot):
        half = self.time_emb_dim // 2
        t_emb_base = self.t_emb_base
        emb = t[:, None].float() * t_emb_base[None, :]
        t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.time_emb_dim % 2:
            t_emb = F.pad(t_emb, (0, 1))
        h = torch.cat([x, label_onehot, t_emb], dim=-1)
        return self.net(h)


class CondDDPM:
    def __init__(self, n_steps=1000, hidden_dim=256, num_layers=4,
                 lr=2e-4, epochs=1500, batch_size=256, device="cpu"):
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self._betas = np.linspace(1e-4, 0.02, n_steps, dtype=np.float32)
        self._alphas = 1.0 - self._betas
        self._alphas_cumprod = np.cumprod(self._alphas)
        self._sqrt_alphas_cumprod = np.sqrt(self._alphas_cumprod)
        self._sqrt_one_minus = np.sqrt(1.0 - self._alphas_cumprod)

        self._sqrt_alpha_cp_t = None
        self._sqrt_om_t = None
        self.denoiser = None
        self.train_losses = []

    def fit(self, data, labels, verbose=True):
        self.denoiser = _CondDenoiser(
            input_dim=6, hidden_dim=self.hidden_dim, num_layers=self.num_layers
        ).to(self.device)

        self._sqrt_alpha_cp_t = torch.tensor(self._sqrt_alphas_cumprod, device=self.device)
        self._sqrt_om_t = torch.tensor(self._sqrt_one_minus, device=self.device)

        opt = torch.optim.Adam(self.denoiser.parameters(), lr=self.lr)
        dset = TensorDataset(
            torch.FloatTensor(data),
            torch.FloatTensor(np.eye(N_CLASSES)[labels])
        )
        loader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)

        self.train_losses = []
        for epoch in range(1, self.epochs + 1):
            self.denoiser.train()
            eloss = 0.0
            for x_batch, l_batch in loader:
                x_batch = x_batch.to(self.device)
                l_batch = l_batch.to(self.device)
                t = torch.randint(0, self.n_steps, (x_batch.size(0),), device=self.device)
                sqrt_a = self._sqrt_alpha_cp_t[t].view(-1, 1)
                sqrt_1ma = self._sqrt_om_t[t].view(-1, 1)
                noise = torch.randn_like(x_batch)
                xt = sqrt_a * x_batch + sqrt_1ma * noise
                pred = self.denoiser(xt, t, l_batch)
                loss = F.mse_loss(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                eloss += loss.item()
            self.train_losses.append(eloss / len(loader))
            if verbose and epoch % 500 == 0:
                print(f"    CondDDPM epoch {epoch}/{self.epochs}  loss={self.train_losses[-1]:.6f}")
        return self

    @torch.no_grad()
    def sample(self, n, label_idx):
        self.denoiser.eval()
        label = torch.zeros(n, 4, device=self.device)
        label[:, label_idx] = 1.0
        x = torch.randn(n, 2, device=self.device)
        for ti in range(self.n_steps - 1, -1, -1):
            t = torch.full((n,), ti, device=self.device, dtype=torch.long)
            pred = self.denoiser(x, t, label)
            a_t = self._alphas[ti]
            b_t = self._betas[ti]
            coef = b_t / self._sqrt_one_minus[ti]
            x = (x - coef * pred) / np.sqrt(a_t)
            if ti > 0:
                x = x + np.sqrt(b_t) * torch.randn_like(x)
        return x.cpu().numpy().astype(np.float32)


def run_extension_2():
    """Conditional generation: CVAE and CondDDPM on all 4 classes jointly."""
    print("\n" + "=" * 60)
    print("EXTENSION 2: Conditional Generation")
    print("=" * 60)

    # CVAE
    print("\n--- Training CVAE ---")
    cvae = CVAE(latent_dim=8).to(DEVICE)
    opt = torch.optim.Adam(cvae.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)
    dset = TensorDataset(
        torch.FloatTensor(train),
        torch.FloatTensor(np.eye(N_CLASSES)[train_label])
    )
    loader = DataLoader(dset, batch_size=256, shuffle=True)
    for epoch in range(1, 801):
        cvae.train()
        eloss = 0.0
        for x_batch, l_batch in loader:
            x_batch, l_batch = x_batch.to(DEVICE), l_batch.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = cvae(x_batch, l_batch)
            recon_loss = F.mse_loss(recon, x_batch, reduction="sum") / x_batch.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_batch.size(0)
            loss = recon_loss + 1.0 * kl_loss
            loss.backward()
            opt.step()
            eloss += loss.item()
        sched.step()
        if epoch % 200 == 0:
            print(f"    CVAE epoch {epoch}/800  loss={eloss/len(loader):.4f}")

    # CondDDPM
    print("\n--- Training Conditional DDPM ---")
    cddpm = CondDDPM(epochs=1500, device=DEVICE).fit(train, train_label, verbose=True)

    # Evaluate
    results_cvae = {}
    results_cddpm = {}
    print("\n--- Evaluation ---")
    for c in range(N_CLASSES):
        real = test[test_label == c]
        gen_cvae = cvae.generate(2000, c, DEVICE)
        gen_cddpm = cddpm.sample(2000, c)
        results_cvae[CLASS_NAMES[c]] = compute_metrics(real, gen_cvae)
        results_cddpm[CLASS_NAMES[c]] = compute_metrics(real, gen_cddpm)
        print(f"  {CLASS_NAMES[c]}:")
        print(f"    CVAE    MMD={results_cvae[CLASS_NAMES[c]]['MMD']:.4f}  "
              f"Prec={results_cvae[CLASS_NAMES[c]]['Precision']:.3f}  "
              f"Cov={results_cvae[CLASS_NAMES[c]]['Coverage']:.3f}")
        print(f"    CondDDPM MMD={results_cddpm[CLASS_NAMES[c]]['MMD']:.4f}  "
              f"Prec={results_cddpm[CLASS_NAMES[c]]['Precision']:.3f}  "
              f"Cov={results_cddpm[CLASS_NAMES[c]]['Coverage']:.3f}")

    return {"CVAE": results_cvae, "CondDDPM": results_cddpm}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTENSION 3: ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_extension_3():
    """Robustness: add uniform noise to Spiral training data."""
    print("\n" + "=" * 60)
    print("EXTENSION 3: Robustness to Noise (Spiral)")
    print("=" * 60)

    from code.models.vae_model import VAEModel
    from code.models.diffusion_model import DiffusionModel

    spiral_data = train[train_label == 3]
    spiral_test = test[test_label == 3]
    n_clean = len(spiral_data)

    results = {}
    for noise_ratio in [0.0, 0.05, 0.10]:
        print(f"\n--- Noise ratio = {noise_ratio:.0%} ---")
        n_noise = int(n_clean * noise_ratio)
        noise_pts = np.random.default_rng(42).uniform(-4, 4, size=(n_noise, 2)).astype(np.float32)
        corrupted = np.concatenate([spiral_data, noise_pts], axis=0)
        # Shuffle
        idx = np.random.default_rng(42).permutation(len(corrupted))
        corrupted = corrupted[idx]

        # VAE
        print("  Training VAE ...")
        vae = VAEModel(
            latent_dim=8, hidden_dims=[128, 64],
            lr=3e-4, epochs=800, batch_size=256,
            beta=1.0, device=DEVICE,
        ).fit(corrupted, verbose=False)
        gen_vae = vae.sample(2000)
        mv = compute_metrics(spiral_test, gen_vae)

        # DDPM
        print("  Training DDPM ...")
        diff = DiffusionModel(
            n_steps=1000, hidden_dim=256, num_layers=4,
            lr=2e-4, epochs=1500, batch_size=256,
            device=DEVICE,
        ).fit(corrupted, verbose=False)
        gen_diff = diff.sample(2000)
        md = compute_metrics(spiral_test, gen_diff)

        results[f"r={noise_ratio:.0%}"] = {"VAE": mv, "DDPM": md}
        print(f"    VAE:     MMD={mv['MMD']:.4f}  Prec={mv['Precision']:.3f}  Cov={mv['Coverage']:.3f}")
        print(f"    DDPM:    MMD={md['MMD']:.4f}  Prec={md['Precision']:.3f}  Cov={md['Coverage']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("EXTENSION EXPERIMENTS")
    print("=" * 60)

    ext1 = run_extension_1()
    ext2 = run_extension_2()
    ext3 = run_extension_3()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nExtension 1 — Hyperparameter Ablation (Spiral):")
    for k, v in ext1.items():
        print(f"  {k:<16} MMD={v['MMD']:.4f}  Prec={v['Precision']:.3f}  Cov={v['Coverage']:.3f}")

    print("\nExtension 2 — Conditional Generation (avg over 4 classes):")
    for model in ["CVAE", "CondDDPM"]:
        avg_mmd = np.mean([ext2[model][cn]["MMD"] for cn in CLASS_NAMES])
        avg_prec = np.mean([ext2[model][cn]["Precision"] for cn in CLASS_NAMES])
        avg_cov = np.mean([ext2[model][cn]["Coverage"] for cn in CLASS_NAMES])
        print(f"  {model:<12} MMD={avg_mmd:.4f}  Prec={avg_prec:.3f}  Cov={avg_cov:.3f}")

    print("\nExtension 3 — Robustness (Spiral):")
    for k, v in ext3.items():
        print(f"  {k:<8} VAE:   MMD={v['VAE']['MMD']:.4f}  Prec={v['VAE']['Precision']:.3f}  Cov={v['VAE']['Coverage']:.3f}")
        print(f"  {'':8} DDPM:  MMD={v['DDPM']['MMD']:.4f}  Prec={v['DDPM']['Precision']:.3f}  Cov={v['DDPM']['Coverage']:.3f}")

    print("\nDone.")
