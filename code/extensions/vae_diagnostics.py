"""Diagnostics for explaining VAE generation failures.

The goal is to distinguish several possible failure modes:

1. posterior collapse: KL is near zero and few latent dimensions are active;
2. reconstruction failure: even encoded training samples cannot be decoded well;
3. prior/aggregate-posterior mismatch: reconstructions are good, but samples
   from N(0, I) fall outside the latent region used by encoded data.

The script writes diagnostic figures under:

    figures/extensions/
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import spatial
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.config import CLASS_COLORS, CLASS_NAMES, DATA_DIR, FIGURES_DIR, MODEL_DIR, set_seed
from code.models.vae_model import VAEModel


OUT_DIR = FIGURES_DIR / "extensions"
SHORT_NAMES = ["GMM", "Ring", "Moons", "Spiral"]


@dataclass
class ClassDiagnostics:
    class_id: int
    class_name: str
    recon_mse: float
    kl_total: float
    kl_dims: np.ndarray
    active_units: int
    mu_var_mean: float
    post_std_mean: float
    agg_mean_abs: float
    agg_var_abs_err: float
    latent_mmd: float
    prior_latent_support: float
    data_nn_p95: float
    recon_nn_mean: float
    recon_off_support: float
    prior_nn_mean: float
    prior_off_support: float


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _format_xy(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    ax.grid(color="#e7eaf0", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#ccd1dc")
        spine.set_linewidth(0.7)


def _subsample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


def _mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 1200) -> float:
    rng = np.random.default_rng(7)
    x = _subsample(x, max_n, rng)
    y = _subsample(y, max_n, rng)
    pooled = np.concatenate([x, y], axis=0)
    ref = pooled[: min(500, len(pooled))]
    d = cdist(ref, ref)
    positive = d[d > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-3)
    gamma = 0.5 / (sigma * sigma)

    def kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.exp(-gamma * cdist(a, b, "sqeuclidean")).mean())

    return max(kernel_mean(x, x) + kernel_mean(y, y) - 2.0 * kernel_mean(x, y), 0.0)


def _data_support_stats(real: np.ndarray, query: np.ndarray) -> tuple[float, float, float]:
    tree = spatial.cKDTree(real)
    knn, _ = tree.query(real, k=6)
    threshold = float(np.percentile(knn[:, -1], 95))
    dist, _ = tree.query(query, k=1)
    return threshold, float(np.mean(dist)), float(np.mean(dist > threshold))


def _latent_support_rate(mu: np.ndarray, z_prior: np.ndarray) -> float:
    tree = spatial.cKDTree(mu)
    knn, _ = tree.query(mu, k=min(6, len(mu)))
    threshold = float(np.percentile(knn[:, -1], 95))
    dist, _ = tree.query(z_prior, k=1)
    return float(np.mean(dist <= threshold))


def _save_fig(fig: plt.Figure, name: str, dpi: int = 230) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _encode(model: VAEModel, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert model.vae is not None
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32, device=model.device)
        mu, logvar = model.vae.encoder(x)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def _decode(model: VAEModel, z: np.ndarray) -> np.ndarray:
    assert model.vae is not None
    with torch.no_grad():
        z_t = torch.tensor(z, dtype=torch.float32, device=model.device)
        out = model.vae.decoder(z_t)
    return out.cpu().numpy().astype(np.float32)


def diagnose_class(class_id: int, data: np.ndarray, labels: np.ndarray, seed: int) -> tuple[ClassDiagnostics, dict[str, np.ndarray]]:
    real = data[labels == class_id]
    model = VAEModel.load(MODEL_DIR / f"vae_class{class_id}.pt", map_location="cpu")
    mu, logvar = _encode(model, real)
    std = np.exp(0.5 * logvar)
    eps = np.random.default_rng(seed + class_id).normal(size=mu.shape).astype(np.float32)
    z_post = mu + eps * std

    recon = _decode(model, mu)
    z_prior = np.random.default_rng(seed + 100 + class_id).normal(size=mu.shape).astype(np.float32)
    prior_gen = _decode(model, z_prior)

    kl_dims = 0.5 * (mu**2 + np.exp(logvar) - logvar - 1.0).mean(axis=0)
    recon_mse = float(np.mean(np.sum((recon - real) ** 2, axis=1)))
    mu_var = mu.var(axis=0)
    active_units = int(np.sum(mu_var > 1e-2))
    agg_var = (mu.var(axis=0) + np.mean(std**2, axis=0))

    data_nn_p95, recon_nn_mean, recon_off = _data_support_stats(real, recon)
    _, prior_nn_mean, prior_off = _data_support_stats(real, prior_gen)

    diag = ClassDiagnostics(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        recon_mse=recon_mse,
        kl_total=float(kl_dims.sum()),
        kl_dims=kl_dims,
        active_units=active_units,
        mu_var_mean=float(mu_var.mean()),
        post_std_mean=float(std.mean()),
        agg_mean_abs=float(np.mean(np.abs(mu.mean(axis=0)))),
        agg_var_abs_err=float(np.mean(np.abs(agg_var - 1.0))),
        latent_mmd=_mmd_rbf(z_post, z_prior),
        prior_latent_support=_latent_support_rate(mu, z_prior),
        data_nn_p95=data_nn_p95,
        recon_nn_mean=recon_nn_mean,
        recon_off_support=recon_off,
        prior_nn_mean=prior_nn_mean,
        prior_off_support=prior_off,
    )
    arrays = {
        "real": real,
        "mu": mu,
        "z_prior": z_prior,
        "recon": recon,
        "prior_gen": prior_gen,
    }
    return diag, arrays


def figure_recon_vs_prior(all_arrays: dict[int, dict[str, np.ndarray]], rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(10.2, 6.6), constrained_layout=True)
    row_titles = ["Data", "VAE reconstruction", "VAE prior samples"]
    for class_id in range(4):
        axes[0, class_id].set_title(SHORT_NAMES[class_id], fontsize=10, weight="bold")

    for class_id in range(4):
        arrays = all_arrays[class_id]
        panels = [
            (arrays["real"], "#30343b"),
            (arrays["recon"], "#2f6fb0"),
            (arrays["prior_gen"], CLASS_COLORS[class_id]),
        ]
        for row, (points, color) in enumerate(panels):
            ax = axes[row, class_id]
            p = _subsample(points, 1300, rng)
            ax.scatter(p[:, 0], p[:, 1], s=4.2, c=color, alpha=0.52, linewidths=0, rasterized=True)
            _format_xy(ax)
            if class_id == 0:
                ax.set_ylabel(row_titles[row], fontsize=9, weight="bold")

    fig.suptitle("VAE diagnostic: reconstruction is not the same as prior generation", fontsize=12, weight="bold")
    fig.savefig(OUT_DIR / "vae_reconstruction_vs_prior.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_latent_kl_by_dim(diags: list[ClassDiagnostics]) -> None:
    kl = np.vstack([d.kl_dims for d in diags])
    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    im = ax.imshow(kl, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(float(kl.max()), 1e-3))

    for i in range(kl.shape[0]):
        for j in range(kl.shape[1]):
            color = "white" if kl[i, j] > 0.45 * kl.max() else "#1f2630"
            ax.text(j, i, f"{kl[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_yticks(np.arange(len(diags)))
    ax.set_yticklabels(SHORT_NAMES)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([f"z{j + 1}" for j in range(8)])
    ax.set_xlabel("latent dimension")
    ax.set_title("Per-dimension KL: most latent dimensions are almost unused", weight="bold")
    fig.colorbar(im, ax=ax, shrink=0.82, label="mean KL")
    _save_fig(fig, "vae_latent_kl_by_dim")


def main() -> None:
    _ensure_dirs()
    set_seed(42)
    rng = np.random.default_rng(42)
    train = np.load(DATA_DIR / "train.npy").astype(np.float32)
    labels = np.load(DATA_DIR / "train_label.npy").astype(np.int64)

    diags: list[ClassDiagnostics] = []
    all_arrays: dict[int, dict[str, np.ndarray]] = {}
    for class_id in range(4):
        diag, arrays = diagnose_class(class_id, train, labels, seed=42)
        diags.append(diag)
        all_arrays[class_id] = arrays
        print(
            f"{diag.class_name}: recon_mse={diag.recon_mse:.4f}, "
            f"KL={diag.kl_total:.3f}, active={diag.active_units}/8, "
            f"prior_support={diag.prior_latent_support:.3f}, "
            f"prior_off={diag.prior_off_support:.3f}"
        )

    figure_recon_vs_prior(all_arrays, rng)
    figure_latent_kl_by_dim(diags)
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
