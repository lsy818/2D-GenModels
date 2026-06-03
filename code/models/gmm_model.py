"""Gaussian Mixture Model — BIC-based component selection."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from code.models.base import GenerativeModel


class GMMModel(GenerativeModel):
    """GMM with *full* covariance and optional BIC-driven ``n_components``."""

    name = "GMM"

    def __init__(self, n_components: Optional[int] = None,
                 covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm: Optional[GaussianMixture] = None
        self._bic_values: list = []

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, data: np.ndarray, max_components: int = 20,
            **kwargs) -> "GMMModel":
        if self.n_components is not None:
            self.gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                max_iter=500, n_init=10,
            )
            self.gmm.fit(data)
            print(f"    GMM fixed K = {self.n_components}, "
                  f"BIC = {self.gmm.bic(data):.1f}")
        else:
            self._select_k(data, max_components)
        return self

    def _select_k(self, data: np.ndarray, max_k: int) -> None:
        best_bic = np.inf
        best_gmm = None
        best_k = 1
        self._bic_values = []

        for k in range(1, max_k + 1):
            gmm = GaussianMixture(
                n_components=k, covariance_type=self.covariance_type,
                random_state=self.random_state, max_iter=500, n_init=5,
            )
            gmm.fit(data)
            bic = gmm.bic(data)
            self._bic_values.append(bic)
            if bic < best_bic:
                best_bic, best_gmm, best_k = bic, gmm, k

        self.gmm = best_gmm
        self.n_components = best_k
        print(f"    GMM BIC → K = {best_k}  (BIC = {best_bic:.1f})")

    # ── sample ────────────────────────────────────────────────────────────

    def sample(self, n: int) -> np.ndarray:
        samples, _ = self.gmm.sample(n)
        return samples.astype(np.float32)

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def bic_values(self) -> np.ndarray:
        return np.array(self._bic_values)
