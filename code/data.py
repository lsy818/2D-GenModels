"""Data loading and statistical analysis utilities."""

from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity, NearestNeighbors

from code.config import DATA_DIR, CLASS_NAMES, N_CLASSES

# ── Loading ───────────────────────────────────────────────────────────────────

def load_data(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_DIR / f"{split}.npy").astype(np.float32)
    labels = np.load(DATA_DIR / f"{split}_label.npy").astype(np.int64)
    return data, labels

def get_class(data: np.ndarray, labels: np.ndarray, class_idx: int) -> np.ndarray:
    return data[labels == class_idx]


# ── Analysis ──────────────────────────────────────────────────────────────────

def local_noise(points: np.ndarray, k: int = 30) -> np.ndarray:
    """Per-point mean k-NN distance."""
    tree = spatial.cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    return dists[:, 1:].mean(axis=1)

def manifold_thickness(points: np.ndarray, k: int = 20) -> np.ndarray:
    """sqrt of smallest PCA eigenvalue in each local patch."""
    tree = spatial.cKDTree(points)
    thick = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=k + 1)
        neigh = points[idx[1:]]
        cov = np.cov(neigh.T, ddof=1)
        eig = np.linalg.eigvalsh(cov)
        thick[i] = np.sqrt(max(eig[0], 0.0))
    return thick

def local_curvature(points: np.ndarray, k: int = 15) -> np.ndarray:
    """min(eig) / sum(eig) in each local patch."""
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

def effective_dims(points: np.ndarray, k_vals=None) -> list:
    """Cross-scale local dimension via k-NN radius ratio."""
    if k_vals is None:
        k_vals = [5, 10, 20, 40, 80, 160]
    n_neighbors = min(len(points), 2 * max(k_vals) + 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    dists, _ = nbrs.kneighbors(points)
    est = []
    for k in k_vals:
        r_k = dists[:, min(k, n_neighbors - 1)]
        r_2k = dists[:, min(2 * k, n_neighbors - 1)]
        ratio = np.clip(r_2k / (r_k + 1e-12), 1.01, 100)
        est.append(float(np.log(2) / np.log(ratio).mean()))
    return est

def pairwise_stats(points: np.ndarray, n_subsample: int = 800) -> dict:
    rng = np.random.default_rng(42)
    sub = points[rng.choice(len(points), min(len(points), n_subsample), replace=False)]
    d = spatial.distance.pdist(sub)
    return {"mean": float(d.mean()), "std": float(d.std())}

def analyse(points: np.ndarray) -> Dict[str, Any]:
    """Return comprehensive statistics for a point cloud."""
    cov = np.cov(points.T)
    eig = np.linalg.eigvalsh(cov)
    pca = PCA().fit(points)
    noise = local_noise(points, 30)
    thick = manifold_thickness(points, 20)
    curv = local_curvature(points, 15)
    pw = pairwise_stats(points)
    return {
        "mean": points.mean(axis=0),
        "std": points.std(axis=0),
        "range": points.max(axis=0) - points.min(axis=0),
        "cov_eigvals": eig,
        "anisotropy": float(eig.max() / max(eig.min(), 1e-12)),
        "noise_mean": float(noise.mean()),
        "thickness_mean": float(thick.mean()),
        "curvature_mean": float(curv.mean()),
        "curvature_std": float(curv.std()),
        "pairwise_mean": pw["mean"],
        "pairwise_std": pw["std"],
        "pca_ratio": pca.explained_variance_ratio_,
        "noise_map": noise,
        "thickness_map": thick,
        "curvature_map": curv,
    }
