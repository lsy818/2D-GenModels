"""Targeted VAE improvements and counterfactual diagnostics.

The experiments are designed to test the diagnosis from vae_diagnostics.py:

1. Weak KL pressure: train beta=0.1 VAE.
2. Free bits: allow each latent dimension to carry a small amount of KL before
   it is penalised.
3. Empirical latent prior: keep the baseline VAE fixed, but sample near encoded
   training latents instead of sampling from an unconstrained standard normal.

If these interventions improve geometry, the VAE failure is better explained
by latent under-utilisation and decoder interpolation than by generic
under-training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import spatial
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.config import CLASS_NAMES, DATA_DIR, MODEL_DIR, VAEConfig, set_seed
from code.metrics import coverage_precision, mmd_rbf
from code.models.vae_model import VAE, VAEModel


MODEL_OUT = MODEL_DIR / "extensions"


@dataclass
class VariantResult:
    class_id: int
    class_name: str
    variant: str
    mmd: float
    coverage: float
    precision: float
    off_support: float
    recon_mse: float
    kl_total: float
    active_dims: int


def _ensure_dirs() -> None:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)


def _encode(model: VAE, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        mu, logvar = model.encoder(x)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def _decode(model: VAE, z: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z_t = torch.tensor(z, dtype=torch.float32)
        out = model.decoder(z_t)
    return out.cpu().numpy().astype(np.float32)


def _prior_sample(model: VAE, n: int, seed: int) -> np.ndarray:
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim)
        out = model.decoder(z)
    return out.cpu().numpy().astype(np.float32)


def _empirical_latent_sample(model: VAE, train_data: np.ndarray, n: int, seed: int, temp: float = 0.25) -> np.ndarray:
    """Sample near encoded training latents.

    This is not a deployable pure-prior sampler. It is a counterfactual: if this
    improves geometry, then the decoder has learned useful local charts but
    standard-prior sampling is pushing it through bad interpolation regions.
    """
    rng = np.random.default_rng(seed)
    mu, logvar = _encode(model, train_data)
    std = np.exp(0.5 * logvar)
    idx = rng.integers(0, len(train_data), size=n)
    z = mu[idx] + temp * std[idx] * rng.normal(size=(n, model.latent_dim)).astype(np.float32)
    return _decode(model, z.astype(np.float32))


def _vae_stats(model: VAE, train_data: np.ndarray) -> tuple[float, float, int]:
    mu, logvar = _encode(model, train_data)
    recon = _decode(model, mu)
    recon_mse = float(np.mean(np.sum((recon - train_data) ** 2, axis=1)))
    kl_dims = 0.5 * (mu**2 + np.exp(logvar) - logvar - 1.0).mean(axis=0)
    active_dims = int(np.sum(mu.var(axis=0) > 1e-2))
    return recon_mse, float(kl_dims.sum()), active_dims


def _off_support(real: np.ndarray, generated: np.ndarray) -> float:
    tree = spatial.cKDTree(real)
    knn, _ = tree.query(real, k=6)
    threshold = float(np.percentile(knn[:, -1], 95))
    dist, _ = tree.query(generated, k=1)
    return float(np.mean(dist > threshold))


def _evaluate(
    class_id: int,
    variant: str,
    model: VAE,
    train_class: np.ndarray,
    test_class: np.ndarray,
    generated: np.ndarray,
) -> VariantResult:
    cp = coverage_precision(test_class, generated, k=5, max_subsample=1000)
    recon_mse, kl_total, active_dims = _vae_stats(model, train_class)
    return VariantResult(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        variant=variant,
        mmd=mmd_rbf(test_class, generated, max_subsample=1000),
        coverage=cp["coverage"],
        precision=cp["precision"],
        off_support=_off_support(test_class, generated),
        recon_mse=recon_mse,
        kl_total=kl_total,
        active_dims=active_dims,
    )


def train_custom_vae(
    train_data: np.ndarray,
    beta: float,
    free_bits: float,
    seed: int,
    epochs: int = 800,
    verbose: bool = False,
) -> VAE:
    cfg = VAEConfig()
    torch.manual_seed(seed)
    model = VAE(input_dim=2, hidden_dims=cfg.hidden_dims, latent_dim=cfg.latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_data, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            recon_loss = F.mse_loss(recon, batch, reduction="sum") / batch.size(0)
            kl_dims = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean(dim=0)
            if free_bits > 0:
                kl_loss = torch.maximum(kl_dims, torch.full_like(kl_dims, free_bits)).sum()
            else:
                kl_loss = kl_dims.sum()
            loss = recon_loss + beta * kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
        sched.step()
        if verbose and epoch % 200 == 0:
            print(f"      epoch {epoch:4d}/{epochs}: loss={running / len(loader):.4f}")
    model.eval()
    return model


def _load_custom_vae(path: Path) -> VAE:
    cfg = VAEConfig()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = VAE(input_dim=2, hidden_dims=cfg.hidden_dims, latent_dim=cfg.latent_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def train_or_load_custom_vae(
    path: Path,
    train_data: np.ndarray,
    beta: float,
    free_bits: float,
    seed: int,
    label: str,
) -> VAE:
    if path.exists():
        print(f"  loading {label}: {path.name}")
        return _load_custom_vae(path)
    print(f"  training {label}")
    model = train_custom_vae(train_data, beta=beta, free_bits=free_bits, seed=seed)
    torch.save({"model_state": model.state_dict()}, path)
    return model


def _load_baseline(class_id: int) -> VAE:
    wrapper = VAEModel.load(MODEL_DIR / f"vae_class{class_id}.pt", map_location="cpu")
    assert wrapper.vae is not None
    wrapper.vae.eval()
    return wrapper.vae


def run_experiments() -> list[VariantResult]:
    _ensure_dirs()
    set_seed(42)
    train = np.load(DATA_DIR / "train.npy").astype(np.float32)
    train_label = np.load(DATA_DIR / "train_label.npy").astype(np.int64)
    test = np.load(DATA_DIR / "test.npy").astype(np.float32)
    test_label = np.load(DATA_DIR / "test_label.npy").astype(np.int64)

    results: list[VariantResult] = []

    for class_id in range(4):
        train_class = train[train_label == class_id]
        test_class = test[test_label == class_id]

        print(f"\nClass {class_id}: {CLASS_NAMES[class_id]}")

        baseline = _load_baseline(class_id)
        baseline_gen = _prior_sample(baseline, len(test_class), 100 + class_id)
        results.append(_evaluate(class_id, "Baseline", baseline, train_class, test_class, baseline_gen))

        emp_gen = _empirical_latent_sample(baseline, train_class, len(test_class), 200 + class_id, temp=0.25)
        results.append(_evaluate(class_id, "EmpiricalPrior", baseline, train_class, test_class, emp_gen))

        beta_model = train_or_load_custom_vae(
            MODEL_OUT / f"vae_beta01_class{class_id}.pt",
            train_class,
            beta=0.1,
            free_bits=0.0,
            seed=300 + class_id,
            label="beta=0.1",
        )
        beta_gen = _prior_sample(beta_model, len(test_class), 400 + class_id)
        results.append(_evaluate(class_id, "Beta0.1", beta_model, train_class, test_class, beta_gen))

        fb_model = train_or_load_custom_vae(
            MODEL_OUT / f"vae_freebits_class{class_id}.pt",
            train_class,
            beta=1.0,
            free_bits=0.05,
            seed=500 + class_id,
            label="free-bits beta=1.0 lambda=0.05",
        )
        fb_gen = _prior_sample(fb_model, len(test_class), 600 + class_id)
        results.append(_evaluate(class_id, "FreeBits", fb_model, train_class, test_class, fb_gen))

        for row in [r for r in results if r.class_id == class_id]:
            print(
                f"    {row.variant:14s} MMD={row.mmd:.4f} "
                f"Cov={row.coverage:.3f} Prec={row.precision:.3f} "
                f"Off={row.off_support:.3f} Active={row.active_dims}/8"
            )

    return results


def main() -> None:
    results = run_experiments()
    print(f"Computed {len(results)} VAE intervention rows.")


if __name__ == "__main__":
    main()
