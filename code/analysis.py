"""Statistical analysis functions for 2D distribution data.

Each function takes a ``(n, 2)`` array and returns a dictionary of metrics.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from code.config import AnalysisConfig


def _get_config(cfg: AnalysisConfig = None) -> AnalysisConfig:
    return cfg if cfg is not None else AnalysisConfig()


def estimate_local_noise(points: np.ndarray, k: int = 30
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Per-point mean & std of *k*-NN distances."""
    tree = spatial.cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    dists = dists[:, 1:]           # drop self
    return dists.mean(axis=1), dists.std(axis=1)


def compute_curvature(points: np.ndarray, k: int = 15) -> np.ndarray:
    """Local curvature ≈ ``λ_min / (λ_min + λ_max)``."""
    tree = spatial.cKDTree(points)
    curves = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k + 1)
        neigh = points[idx[1:]]
        cov = np.cov(neigh.T, ddof=1)
        eig = np.linalg.eigvalsh(cov)
        s = eig.sum()
        curves[i] = eig.min() / s if s > 1e-12 else 0.0
    return curves


def compute_thickness(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Manifold thickness ≈ sqrt of smallest PCA eigenvalue in local patch."""
    tree = spatial.cKDTree(points)
    thick = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k + 1)
        neigh = points[idx[1:]]
        cov = np.cov(neigh.T, ddof=1)
        eig = np.linalg.eigvalsh(cov)
        thick[i] = np.sqrt(max(eig[0], 0.0))
    return thick


def compute_pairwise_stats(points: np.ndarray, n_subsample: int = 1000,
                           seed: int = 42) -> Dict[str, float]:
    """Mean & std of pairwise Euclidean distances (sub-sampled)."""
    rng = np.random.default_rng(seed)
    n = min(len(points), n_subsample)
    idx = rng.choice(len(points), n, replace=False)
    dists = spatial.distance.pdist(points[idx])
    return {"mean": float(dists.mean()), "std": float(dists.std())}


def analyse_distribution(points: np.ndarray,
                         cfg: AnalysisConfig = None) -> Dict[str, Any]:
    """Return a rich dictionary of metrics for *points*."""
    cfg = _get_config(cfg)
    n = len(points)
    results: Dict[str, Any] = {}

    # Basic
    results["mean"] = points.mean(axis=0)
    results["std"] = points.std(axis=0)
    results["range"] = points.max(axis=0) - points.min(axis=0)

    cov = np.cov(points.T)
    eig = np.linalg.eigvalsh(cov)
    results["cov_eigvals"] = eig
    results["global_anisotropy"] = float(
        eig.max() / max(eig.min(), 1e-12)
    )

    # Local noise
    noise_mean, noise_std = estimate_local_noise(
        points, k=cfg.n_neighbors_noise)
    results["local_noise_mean"] = float(noise_mean.mean())
    results["local_noise_cv"] = float(
        (noise_std / (noise_mean + 1e-12)).mean())

    # Thickness & curvature
    thickness = compute_thickness(points, k=cfg.n_neighbors_thickness)
    curvature = compute_curvature(points, k=cfg.n_neighbors_curvature)
    results["thickness_mean"] = float(thickness.mean())
    results["curvature_mean"] = float(curvature.mean())

    # Pairwise distances
    results["pairwise"] = compute_pairwise_stats(
        points, n_subsample=cfg.n_subsample_pairwise)

    # PCA
    pca = PCA(n_components=2).fit(points)
    results["pca_ratio"] = pca.explained_variance_ratio_

    # KDE grid
    kde = KernelDensity(bandwidth=cfg.kde_bandwidth, kernel="gaussian").fit(points)
    grid = np.linspace(-cfg.max_range, cfg.max_range, cfg.grid_resolution)
    Xg, Yg = np.meshgrid(grid, grid)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    Z = np.exp(kde.score_samples(grid_pts)).reshape(Xg.shape)
    results["density_grid"] = {"X": Xg, "Y": Yg, "Z": Z}

    # Per-point maps
    results["noise_map"] = noise_mean
    results["thickness_map"] = thickness
    results["curvature_map"] = curvature

    return results
