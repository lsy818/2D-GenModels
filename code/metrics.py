"""Generation quality metrics.

All functions take ``real`` and ``generated`` arrays of shape ``(N, 2)``
and return a scalar or dict.
"""

from __future__ import annotations

from typing import Dict, Optional, Callable
from pathlib import Path

import numpy as np
from scipy import spatial
from scipy.spatial.distance import cdist

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ensure_float32(*arrays: np.ndarray) -> list:
    return [np.asarray(a, dtype=np.float32) for a in arrays]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Maximum Mean Discrepancy  (MMD²)
# ═══════════════════════════════════════════════════════════════════════════════

def mmd_rbf(real: np.ndarray, generated: np.ndarray,
            sigma: Optional[float] = None, max_subsample: int = 1000) -> float:
    """Squared MMD with Gaussian RBF kernel.

    Uses a median-distance heuristic for the bandwidth *sigma*.
    """
    real, generated = _ensure_float32(real, generated)

    # Sub-sample for efficiency
    rng = np.random.default_rng(42)
    n_r = min(len(real), max_subsample)
    n_g = min(len(generated), max_subsample)
    X = real[rng.choice(len(real), n_r, replace=False)]
    Y = generated[rng.choice(len(generated), n_g, replace=False)]

    # Kernel bandwidth
    if sigma is None:
        pooled = np.concatenate([X, Y], axis=0)
        dists = cdist(pooled[:min(500, len(pooled))],
                      pooled[:min(500, len(pooled))])
        sigma = np.median(dists[dists > 0]) if dists[dists > 0].size else 1.0
        sigma = float(max(sigma, 1e-3))

    gamma = 0.5 / (sigma * sigma)

    # Kernel matrices
    def _rbf(A: np.ndarray, B: np.ndarray) -> float:
        sq = cdist(A, B, "sqeuclidean")
        return float(np.mean(np.exp(-gamma * sq)))

    k_xx = _rbf(X, X)
    k_yy = _rbf(Y, Y)
    k_xy = _rbf(X, Y)

    mmd2 = k_xx + k_yy - 2.0 * k_xy
    return max(mmd2, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Wasserstein Distance  (2-Wasserstein, Sinkhorn)
# ═══════════════════════════════════════════════════════════════════════════════

def wasserstein_sinkhorn(real: np.ndarray, generated: np.ndarray,
                         reg: float = 0.01, max_subsample: int = 500) -> float:
    """Entropic-regularised 2-Wasserstein via Sinkhorn algorithm.

    Uses the `POT <https://pythonot.github.io/>`_ library.
    """
    import ot
    real, generated = _ensure_float32(real, generated)

    rng = np.random.default_rng(42)
    n = min(len(real), len(generated), max_subsample)
    X = real[rng.choice(len(real), n, replace=False)]
    Y = generated[rng.choice(len(generated), n, replace=False)]

    C = cdist(X, Y, "sqeuclidean")           # cost matrix
    C /= C.max() + 1e-12                      # normalise

    a = np.ones(n) / n
    b = np.ones(n) / n
    W = ot.sinkhorn2(a, b, C, reg)
    return max(float(np.sqrt(W * C.max())), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  3 & 4.  Coverage & Precision  (k-NN based, Kynkäänniemi et al. 2019)
# ═══════════════════════════════════════════════════════════════════════════════

def _pairwise_min(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row of *A*, min Euclidean distance to any row of *B*."""
    tree = spatial.cKDTree(B)
    dists, _ = tree.query(A, k=1)
    return dists


def coverage_precision(real: np.ndarray, generated: np.ndarray,
                       k: int = 5, max_subsample: int = 1000
                       ) -> Dict[str, float]:
    """Coverage and Precision using *k*-NN manifold radius.

    - **Precision** — fraction of generated points that lie within the
      real-data manifold (i.e. within the k-NN radius of some real point).
    - **Coverage** — fraction of real points that have at least one
      generated point within the same radius.

    """
    real, generated = _ensure_float32(real, generated)

    rng = np.random.default_rng(42)
    nr = min(len(real), max_subsample)
    ng = min(len(generated), max_subsample)
    Xr = real[rng.choice(len(real), nr, replace=False)]
    Xg = generated[rng.choice(len(generated), ng, replace=False)]

    # k-NN radius in the real set
    tree_r = spatial.cKDTree(Xr)
    r_k, _ = tree_r.query(Xr, k=k + 1)
    radii = r_k[:, -1]                        # distance to k-th NN

    # Precision
    dist_g2r = _pairwise_min(Xg, Xr)
    precision = float(np.mean(dist_g2r <= np.percentile(radii, 95)))

    # Coverage
    dist_r2g = _pairwise_min(Xr, Xg)
    coverage = float(np.mean(dist_r2g <= radii))

    return {"precision": precision, "coverage": coverage}


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Negative Log-Likelihood  (NLL)
# ═══════════════════════════════════════════════════════════════════════════════

def nll_kde_gmm(model, test_data: np.ndarray) -> float:
    """NLL for KDE / GMM (models with ``score_samples``)."""
    logp = model.score_samples(test_data)
    return float(-np.mean(logp))


def nll_vae_iwae(model, test_data: np.ndarray, n_iw_samples: int = 50,
                 batch_size: int = 256) -> float:
    """Importance-weighted NLL bound for VAE."""
    import torch
    vae = model.vae
    device = model.device
    vae.eval()

    data = torch.FloatTensor(test_data).to(device)
    nll_total = 0.0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        b = batch.size(0)

        # Repeat for IW samples: (b * n_iw, 2)
        x_expanded = batch.repeat_interleave(n_iw_samples, dim=0)

        with torch.no_grad():
            mu, logvar = vae.encoder(x_expanded)
            z = vae.reparameterise(mu, logvar)
            recon = vae.decoder(z)

            # log p(x|z) — Gaussian with fixed variance
            log_px_z = -0.5 * ((recon - x_expanded) ** 2).sum(dim=1) \
                       - 0.5 * 2 * np.log(2 * np.pi)
            # log p(z) — standard normal
            log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * vae.latent_dim * np.log(2 * np.pi)
            # log q(z|x)
            log_qz_x = -0.5 * (logvar.sum(dim=1)
                               + ((z - mu) ** 2 / torch.exp(logvar)).sum(dim=1)
                               + vae.latent_dim * np.log(2 * np.pi))

            log_w = log_px_z + log_pz - log_qz_x  # (b * n_iw,)
            log_w = log_w.view(b, n_iw_samples)
            # logsumexp over IW samples
            log_px = torch.logsumexp(log_w, dim=1) - np.log(n_iw_samples)
            nll_total -= log_px.sum().item()

    return nll_total / len(test_data)


def nll_diffusion_approx(model, test_data: np.ndarray,
                         n_timesteps_sample: int = 100) -> float:
    """Approximate NLL for diffusion via discrete ELBO.

    This computes a simplified ELBO: E_t[||noise - pred_noise||^2].
    """
    import torch
    model.denoiser.eval()
    device = model.device

    data = torch.FloatTensor(test_data).to(device)
    n = len(data)
    total_loss = 0.0

    # Sample a subset of timesteps for efficiency
    ts = np.linspace(0, model.n_steps - 1, n_timesteps_sample, dtype=int)
    batch_size = 256

    for t_idx in ts:
        for i in range(0, n, batch_size):
            batch = data[i:i + batch_size].to(device)
            b = batch.size(0)
            t = torch.full((b,), t_idx, device=device, dtype=torch.long)

            sqrt_a = model._sqrt_alphas_cumprod_t[t].view(-1, 1)
            sqrt_1ma = model._sqrt_one_minus_t[t].view(-1, 1)
            noise = torch.randn_like(batch)
            xt = sqrt_a * batch + sqrt_1ma * noise

            with torch.no_grad():
                pred = model.denoiser(xt, t)
                loss = ((pred - noise) ** 2).sum(dim=1).mean()
                total_loss += loss.item() * b

    return total_loss / (n * n_timesteps_sample)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Mode Coverage
# ═══════════════════════════════════════════════════════════════════════════════

def mode_coverage(real: np.ndarray, generated: np.ndarray,
                  class_idx: int, class_name: str) -> Dict[str, float]:
    """Fraction of "modes" covered by generated samples.

    Mode definition is distribution-dependent:

    - **Gaussian Mixture**: 8 cluster centres on a circle
    - **Ring**: 12 angular sectors
    - **Two Moons**: 2 crescent arms
    - **Spiral**: 16 combined (radial × angular) sectors
    """
    real, generated = _ensure_float32(real, generated)

    if class_idx == 0:              # Gaussian Mixture — 8 modes
        num_modes = 8
        radius = 2.6
        angles = np.linspace(0, 2 * np.pi, num_modes, endpoint=False)
        centres = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
        return _mode_coverage_centres(generated, centres, num_modes)

    elif class_idx == 1:            # Ring — 12 angular sectors
        return _mode_coverage_angular(generated, n_sectors=12)

    elif class_idx == 2:            # Two Moons — 2 arms
        return _mode_coverage_two_moons(generated)

    elif class_idx == 3:            # Spiral — combined sectors
        return _mode_coverage_spiral(generated)
    else:
        return {"mode_coverage": 0.0, "n_modes_total": 0, "n_modes_covered": 0}


def _mode_coverage_centres(generated: np.ndarray, centres: np.ndarray,
                           n_modes: int) -> Dict[str, float]:
    """Check which centre each generated sample is closest to."""
    tree = spatial.cKDTree(centres)
    _, assignments = tree.query(generated, k=1)
    covered = len(np.unique(assignments))
    return {"mode_coverage": covered / n_modes,
            "n_modes_total": n_modes, "n_modes_covered": covered}


def _mode_coverage_angular(generated: np.ndarray, n_sectors: int = 12
                           ) -> Dict[str, float]:
    theta = np.arctan2(generated[:, 1], generated[:, 0])
    bins = np.linspace(-np.pi, np.pi, n_sectors + 1)
    sector_ids = np.digitize(theta, bins) - 1
    sector_ids = np.clip(sector_ids, 0, n_sectors - 1)
    covered = len(np.unique(sector_ids))
    return {"mode_coverage": covered / n_sectors,
            "n_modes_total": n_sectors, "n_modes_covered": covered}


def _mode_coverage_two_moons(generated: np.ndarray) -> Dict[str, float]:
    """Two moons — classify by y position relative to separator ~0.25."""
    # Upper moon: roughly y > 0.25, Lower moon: y < 0.25
    upper = np.sum(generated[:, 1] > 0.25)
    lower = np.sum(generated[:, 1] <= 0.25)
    covered = int(upper > 0) + int(lower > 0)
    return {"mode_coverage": covered / 2,
            "n_modes_total": 2, "n_modes_covered": covered}


def _mode_coverage_spiral(generated: np.ndarray) -> Dict[str, float]:
    """Spiral — 8 angular sectors × 2 radial bands."""
    r = np.sqrt(generated[:, 0]**2 + generated[:, 1]**2)
    theta = np.arctan2(generated[:, 1], generated[:, 0])

    r_bins = np.linspace(0, 3.2, 3)          # 2 radial bands
    t_bins = np.linspace(-np.pi, np.pi, 9)    # 8 angular sectors
    n_total = 16

    r_ids = np.clip(np.digitize(r, r_bins) - 1, 0, 1)
    t_ids = np.clip(np.digitize(theta, t_bins) - 1, 0, 7)
    cell_ids = r_ids * 8 + t_ids
    covered = len(np.unique(cell_ids))
    return {"mode_coverage": covered / n_total,
            "n_modes_total": n_total, "n_modes_covered": covered}


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(real: np.ndarray, generated: np.ndarray,
                        model, model_type: str,
                        class_idx: int, class_name: str) -> Dict[str, float]:
    """Compute the full suite of quality metrics.

    Parameters
    ----------
    real, generated : (N, 2) arrays
    model : trained model instance
    model_type : "KDE" | "GMM" | "VAE" | "Diffusion"
    class_idx : 0 … 3
    class_name : str

    Returns
    -------
    dict of metric_name → value
    """
    results: Dict[str, float] = {}

    # MMD
    results["MMD"] = mmd_rbf(real, generated)

    # Wasserstein
    results["Wasserstein"] = wasserstein_sinkhorn(real, generated)

    # Coverage & Precision
    cp = coverage_precision(real, generated)
    results["Precision"] = cp["precision"]
    results["Coverage"] = cp["coverage"]

    # NLL
    if model_type == "KDE":
        results["NLL"] = nll_kde_gmm(model, real[:500])
    elif model_type == "GMM":
        results["NLL"] = nll_kde_gmm(model, real[:500])
    elif model_type == "VAE":
        results["NLL"] = nll_vae_iwae(model, real[:500], n_iw_samples=50)
    elif model_type == "Diffusion":
        # Use training loss as proxy (lower is better)
        results["NLL"] = model.train_losses[-1] if model.train_losses else float("nan")
    else:
        results["NLL"] = float("nan")

    # Mode coverage
    mc = mode_coverage(real, generated, class_idx, class_name)
    results["ModeCoverage"] = mc["mode_coverage"]

    return results
