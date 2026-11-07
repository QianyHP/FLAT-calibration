"""Gaussian Process surrogate for FLAT sequential LCB (sklearn GPR)."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class GPSurrogate:
    """10-D theta -> scalar J_b; fit / predict(return_std) / score for LCB loop."""

    def __init__(self, *, random_state: int = 42, min_valid: int = 10) -> None:
        self.random_state = int(random_state)
        self.min_valid = int(min_valid)
        self._x_scaler = StandardScaler()
        self._y_mean = 0.0
        self._y_std = 1.0
        self._gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(y) < self.min_valid:
            raise ValueError(f"GP fit requires >= {self.min_valid} samples, got {len(y)}")
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            raise ValueError("GP fit: non-finite X or y")

        self._x_scaler.fit(X)
        Xs = self._x_scaler.transform(X)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        ys = (y - self._y_mean) / self._y_std

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=np.ones(X.shape[1]), nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            random_state=self.random_state,
            n_restarts_optimizer=2,
        )
        self._gp.fit(Xs, ys)
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self._gp is None:
            raise RuntimeError("GPSurrogate.predict before fit")
        X = np.asarray(X, dtype=float)
        Xs = self._x_scaler.transform(X)
        if return_std:
            mu_n, std_n = self._gp.predict(Xs, return_std=True)
            mu = mu_n * self._y_std + self._y_mean
            std = np.clip(std_n * self._y_std, 1e-8, None)
            return mu, std
        mu_n = self._gp.predict(Xs)
        return mu_n * self._y_std + self._y_mean

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        return float(r2_score(y, self.predict(X)))


def gp_lcb(model: GPSurrogate, X: np.ndarray, kappa: float) -> np.ndarray:
    mu, sigma = model.predict(X, return_std=True)
    return mu - kappa * sigma


def gp_pick_lcb_point(
    model: GPSurrogate,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
    *,
    propose_candidates,
    min_rel_dist,
    n_local: int,
    n_global: int,
    min_dist_rel: float,
) -> np.ndarray:
    cand = propose_candidates(bounds, x_best, n_local, n_global, rng)
    lcb = gp_lcb(model, cand, kappa)
    md = min_rel_dist(cand, X_seen, bounds)
    lcb[md < min_dist_rel] = np.inf
    if not np.any(np.isfinite(lcb)):
        return cand[int(rng.integers(0, len(cand)))]
    return cand[int(np.argmin(lcb))]
