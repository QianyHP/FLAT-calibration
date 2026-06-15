"""plot_convergence_utils.py — 收敛曲线均值 ± seed 半透明置信带绘制。"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

CI_ALPHA = 0.22
CI_Z = 1.0  # ±1×seed 内 std（与柱图误差棒口径一致）


def _clean_std(std: np.ndarray) -> np.ndarray:
    out = np.asarray(std, dtype=float)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.maximum(out, 0.0)


def draw_mean_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    color: str,
    step: bool = False,
    linewidth: float = 2.0,
    zorder: int = 3,
    label: str | None = None,
    ci_z: float = CI_Z,
    alpha: float = CI_ALPHA,
) -> None:
    """绘制均值曲线 + 半透明 fill_between 置信带。"""
    if x.size == 0:
        return
    std = _clean_std(y_std)
    y_lo = y_mean - ci_z * std
    y_hi = y_mean + ci_z * std
    has_band = bool(np.any(std > 1e-12))

    if has_band:
        if step:
            ax.fill_between(
                x, y_lo, y_hi, step="post",
                color=color, alpha=alpha, linewidth=0, zorder=zorder - 1,
            )
        else:
            ax.fill_between(
                x, y_lo, y_hi,
                color=color, alpha=alpha, linewidth=0, zorder=zorder - 1,
            )

    if step:
        ax.step(x, y_mean, where="post", color=color, linewidth=linewidth,
                zorder=zorder, label=label)
    else:
        ax.plot(x, y_mean, color=color, linewidth=linewidth,
                zorder=zorder, label=label, alpha=0.97)
