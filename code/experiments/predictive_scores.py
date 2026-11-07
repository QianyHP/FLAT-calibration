"""Gaussian predictive scores used by the standardized Appendix A audit."""
import numpy as np
from scipy import stats


def _nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    var = np.maximum(sigma ** 2, 1e-12)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mu) ** 2 / var))


def _crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """CRPS for Gaussian predictive distribution (closed form)."""
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return float(np.mean(sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))))
