"""Kernel Density Estimation generative model."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from code.models.base import GenerativeModel


class KDEModel(GenerativeModel):
    """KDE with Gaussian kernel and optional cross-validated bandwidth."""

    name = "KDE"

    def __init__(self, bandwidth: Optional[float] = None):
        self.bandwidth = bandwidth
        self.kde: Optional[KernelDensity] = None
        self._data_min: Optional[np.ndarray] = None
        self._data_max: Optional[np.ndarray] = None
        self._silverman_bw: Optional[float] = None

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray,
            cv_bandwidths: Optional[List[float]] = None,
            **kwargs) -> "KDEModel":
        self._data_min = data.min(axis=0) - 0.5
        self._data_max = data.max(axis=0) + 0.5

        if cv_bandwidths is not None:
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"),
                {"bandwidth": cv_bandwidths},
                cv=5, n_jobs=-1,
            )
            grid.fit(data)
            self.bandwidth = grid.best_params_["bandwidth"]
            print(f"    KDE CV bandwidth = {self.bandwidth:.4f}  "
                  f"(score = {grid.best_score_:.2f})")
        elif self.bandwidth is None:
            # Silverman's rule of thumb
            n, d = float(data.shape[0]), float(data.shape[1])
            self._silverman_bw = 1.06 * float(np.mean(np.std(data, axis=0))) * n ** (-1.0 / (d + 4.0))
            self.bandwidth = self._silverman_bw
            print(f"    KDE Silverman bandwidth = {self.bandwidth:.4f}")
        else:
            print(f"    KDE fixed bandwidth = {self.bandwidth:.4f}")

        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
        self.kde.fit(data)
        return self

    # ── helpers ───────────────────────────────────────────────────────────

    def score_samples(self, points: np.ndarray) -> np.ndarray:
        """Log-density of each point under the fitted KDE."""
        return self.kde.score_samples(points)

    def sample(self, n: int) -> np.ndarray:
        return self._mcmc_sample(n).astype(np.float32)

    def _mcmc_sample(self, n: int, burn_in: int = 500, thin: int = 5
                     ) -> np.ndarray:
        """Metropolis–Hastings sampler."""
        rng = np.random.default_rng()

        # Initialise uniformly within data bounds
        current = rng.uniform(self._data_min, self._data_max, size=(1, 2))
        current_logp = float(self.kde.score_samples(current)[0])

        total = burn_in + n * thin
        chain = np.zeros((total, 2))
        accepted = 0
        prop_std = self.bandwidth * 1.5

        for i in range(total):
            proposal = current + rng.normal(0, prop_std, size=(1, 2))
            proposal_logp = float(self.kde.score_samples(proposal)[0])
            if np.log(rng.random()) < (proposal_logp - current_logp):
                current, current_logp = proposal, proposal_logp
                accepted += 1
            chain[i] = current

        if accepted / total < 0.1:
            print(f"    Warning: MCMC acceptance rate = {accepted / total:.3f}")
        return chain[burn_in::thin][:n]
