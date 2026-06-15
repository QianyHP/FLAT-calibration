"""plot_calibration_convergence.py — BF-SAC calibration convergence (RF vs MLP, 5 seeds)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys

PROJ = Path(__file__).resolve().parents[2]
CAL_ROOT = PROJ / "code" / "calibration"
EXP_ROOT = PROJ / "code" / "experiments"
for p in (CAL_ROOT, EXP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unified_calibration import BUDGET_SUMO, N_INIT  # noqa: E402
from experiment_io import REPLICATE_SEEDS, cache_path  # noqa: E402
from plot_convergence_utils import draw_mean_with_band  # noqa: E402
from plot_style import (  # noqa: E402
    SCENE_ORDER, SCENE_TITLES, METHOD_COLORS, apply_style, save_figure,
    XLABEL_SIMS, YLABEL_JB_CALIB, CI_BAND_LABEL,
)

apply_style(base_font=11.0)
CAL_DIR = PROJ / "data" / "processed_data" / "calibration"

SURROGATES = [
    ("BF-SAC-RF", "RF", METHOD_COLORS["BF-SAC-RF"], "rf"),
    ("BF-SAC-MLP", "MLP", METHOD_COLORS["BF-SAC-MLP"], "mlp"),
]


def load_surrogate_curve(scene: str, method_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves, sims_ref = [], None
    for seed in REPLICATE_SEEDS:
        p = cache_path(scene, method_key, BUDGET_SUMO, seed)
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        sims = np.asarray(d["convergence_sims"], dtype=float)
        bests = np.asarray(d["convergence_best"], dtype=float)
        mask = sims <= BUDGET_SUMO
        sims, bests = sims[mask], bests[mask]
        if sims_ref is None:
            sims_ref = sims
        curves.append(bests)
    if not curves:
        raise FileNotFoundError(f"{scene}/{method_key}: no cache")
    n = min(len(c) for c in curves)
    stacked = np.vstack([c[:n] for c in curves])
    std = stacked.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(n)
    return sims_ref[:n], stacked.mean(axis=0), std


def load_fit_metrics(scene: str, surrogate: str) -> float:
    name = f"{scene}_calibration.json" if surrogate == "rf" else f"{scene}_calibration__mlp_h64-64_m10.json"
    p = CAL_DIR / name
    if not p.exists():
        return float("nan")
    d = json.loads(p.read_text(encoding="utf-8"))
    return float(d.get("rf_r2") or float("nan"))


def plot_panel(ax: plt.Axes, scene: str) -> None:
    y_lo, y_hi = np.inf, -np.inf
    fit_lines = []
    for method_key, tag, color, surr in SURROGATES:
        sims, mean, std = load_surrogate_curve(scene, method_key)
        draw_mean_with_band(ax, sims, mean, std, color=color, step=True,
                            linewidth=2.3, zorder=4, label=tag)
        ax.scatter([sims[-1]], [mean[-1]], color=color, s=70, zorder=7,
                   marker="o", edgecolors="white", linewidths=0.8)
        y_lo = min(y_lo, float((mean - std).min()))
        y_hi = max(y_hi, float(mean[: max(1, N_INIT)].max()))
        fit_lines.append(f"{tag} $R^2$={load_fit_metrics(scene, surr):.3f}")

    ax.axvline(x=N_INIT, color="#94A3B8", linewidth=1.0, linestyle="--", alpha=0.75, zorder=2)
    ax.text(0.015, 0.04, "\n".join(fit_lines), transform=ax.transAxes, fontsize=9.5,
            va="bottom", ha="left", color="#334155",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CBD5E1", alpha=0.9))
    ax.set_title(SCENE_TITLES[scene], fontweight="bold", pad=4)
    ax.set_xlabel(XLABEL_SIMS)
    ax.set_ylabel(YLABEL_JB_CALIB)
    ax.set_xlim(0, BUDGET_SUMO)
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        ax.set_ylim(max(0.0, y_lo * 0.92), y_hi * 1.06)


def main() -> None:
    print("=" * 60)
    print("BF-SAC calibration convergence (RF vs MLP)")
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0))
    fig.patch.set_facecolor("white")
    for i, scene in enumerate(SCENE_ORDER):
        print(f"  [{scene}]")
        plot_panel(axes.flatten()[i], scene)

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-RF"], lw=2.3, label="BF-SAC-RF"),
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-MLP"], lw=2.3, label="BF-SAC-MLP"),
        mpatches.Patch(color="#94A3B8", alpha=0.22, label=CI_BAND_LABEL),
        plt.Line2D([0], [0], linestyle="--", color="#94A3B8",
                   label=f"LHS end / sequential start ($N_{{init}}$={N_INIT})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, framealpha=0.94,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"BF-SAC calibration: RF vs MLP ({N_INIT} LHS + {BUDGET_SUMO - N_INIT} sequential LCB)",
        fontweight="bold", y=1.0,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    out = save_figure(fig, "bfsac_calibration_convergence")
    plt.close(fig)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
