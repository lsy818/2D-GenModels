"""Generation quality metrics.

All functions take ``real`` and ``generated`` arrays of shape ``(N, 2)``
and return a scalar or dict.
"""

from __future__ import annotations

from typing import Dict, Optional

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


def evaluate_generation_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    max_subsample: int = 500,
) -> Dict[str, float]:
    """Compute the four report metrics used by extension experiments.

    Returns MMD, Wasserstein, Precision, and Coverage with the same sampling
    budget used throughout the report's extension tables.
    """
    real, generated = _ensure_float32(real, generated)
    rng = np.random.default_rng(42)
    n = min(len(real), len(generated), max_subsample)
    X = real[rng.choice(len(real), n, replace=False)]
    Y = generated[rng.choice(len(generated), n, replace=False)]

    pooled = np.concatenate([X, Y])
    dists = cdist(pooled[:200], pooled[:200])
    positive = dists[dists > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    gamma = 0.5 / max(sigma, 1e-3) ** 2

    def rbf(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.mean(np.exp(-gamma * cdist(A, B, "sqeuclidean"))))

    mmd = np.sqrt(max(rbf(X, X) + rbf(Y, Y) - 2.0 * rbf(X, Y), 0.0))

    cp = coverage_precision(X, Y, max_subsample=n)
    wass = wasserstein_sinkhorn(real, generated, max_subsample=n)

    return {
        "MMD": float(mmd),
        "Wasserstein": float(wass),
        "Precision": cp["precision"],
        "Coverage": cp["coverage"],
    }


