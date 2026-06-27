#!/usr/bin/env python3
"""Plot XAM-N6 J_b contour with LHS init and sequential LCB search path.

Outputs:
  outputs/figures/lcb_contour.{png,pdf}

Inputs:
  data/processed_data/calibration/XAM-N6_calibration_with_history.json
  outputs/results/landscape_dual_XAM-N6.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 12,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures"
PAPER_FIG = ROOT / "paper" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

SCENE = "XAM-N6"
CMAP = "RdBu_r"
C_SEQ = "#FFEB3B"
C_PATH = "#C5C5C5"
PATH_LW = 1.35
C_OPTIMUM = "#EA580C"
C_LHS = "#94A3B8"
# Match landscape_dual aspect (8.2 × 3.6)
FIG_SIZE = (8.2, 3.6)
LHS_DOT_SIZE = 14
SEQ_DOT_SIZE = 230
SEQ_SUBSAMPLE = 10
STAR_SIZE = 320

FS_LABEL = 13
FS_TICK = 12
FS_CBAR = 13
FS_LEGEND = 10.5
FS_MARKER_NUM = 10


def load_history(scene: str, max_n: int = 100) -> tuple[list[dict], dict]:
    p = ROOT / f"data/processed_data/calibration/{scene}_calibration_with_history.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["doe_history"][:max_n], data["calibrated_params"]


def load_landscape(scene: str) -> tuple[np.ndarray, list[float], list[float]]:
    p = ROOT / f"outputs/results/landscape_dual_{scene}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return np.array(d["results"]), d["accel_range"], d["tau_range"]


def _style_cbar(cbar, label: str) -> None:
    cbar.set_label(label, fontsize=FS_CBAR, color="black")
    cbar.ax.tick_params(labelsize=FS_TICK, colors="black", length=3)
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.9)
    cbar.ax.yaxis.set_ticks_position("right")


def _seq_milestones_chrono(
    hist: list[dict],
    cal: dict,
    *,
    max_k: int = SEQ_SUBSAMPLE,
) -> tuple[list[tuple[float, float, float]], tuple[float, float]]:
    seq = [e for e in hist if e.get("phase") != "init"]
    opt = (float(cal["accel"]), float(cal["tau"]))
    if not seq:
        return [], opt

    n = len(seq)
    idx = sorted({int(round(q * (n - 1))) for q in np.linspace(0, 1, min(max_k, n))})
    out: list[tuple[float, float, float]] = []
    for i in idx:
        e = seq[i]
        a = float(e["params"]["accel"])
        t = float(e["params"]["tau"])
        err = float(e.get("best_so_far", e.get("error", 9.0)))
        if np.isclose(a, opt[0], rtol=0.02) and np.isclose(t, opt[1], rtol=0.02):
            continue
        out.append((a, t, err))
    return out, opt


def _prepare_jb_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res, accel_range, tau_range = load_landscape(SCENE)
    n = 100
    ag = np.linspace(accel_range[0], accel_range[1], n)
    tg = np.linspace(tau_range[0], tau_range[1], n)
    AC, TA = np.meshgrid(ag, tg)
    pts = np.column_stack([res[:, 0], res[:, 1]])
    jb = res[:, 2]
    good = jb < 9
    Z = griddata(pts[good], jb[good], (AC, TA), method="cubic")
    nan = np.isnan(Z)
    if nan.any():
        Z[nan] = griddata(pts[good], jb[good], (AC[nan], TA[nan]), method="nearest")
    Z = gaussian_filter(Z, sigma=0.5)
    lo, hi = jb[good].min(), jb[good].max()
    Zn = (Z - lo) / max(hi - lo, 1e-9)
    return AC, TA, Zn


def _lcb_inset_legend(ax) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_LHS, markersize=4.5,
               alpha=0.8, label="Phase-A LHS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SEQ, markersize=9.5,
               markeredgecolor="black", markeredgewidth=0.6, label="Phase-B LCB"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_OPTIMUM, markersize=11,
               markeredgecolor="black", markeredgewidth=0.5, label=r"Calibrated $\theta^\star$"),
        Line2D([0], [0], color=C_PATH, lw=PATH_LW, linestyle="--", label="Search path"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        borderaxespad=0.5,
        fontsize=FS_LEGEND,
        framealpha=0.95,
        handlelength=1.0,
        borderpad=0.25,
        labelspacing=0.25,
        fancybox=True,
    )


def plot_lcb_contour(hist: list[dict], cal: dict, *, dpi: int = 320) -> tuple[Path, Path]:
    AC, TA, Zn = _prepare_jb_grid()
    opt_a, opt_t = float(cal["accel"]), float(cal["tau"])

    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="white")
    cf = ax.contourf(AC, TA, Zn, levels=16, cmap=CMAP, alpha=0.90, vmin=0, vmax=1)
    ax.contour(AC, TA, Zn, levels=7, colors="white", linewidths=0.35, alpha=0.4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.8%", pad=0.16)
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    _style_cbar(cbar, r"$\tilde{J}(\theta)$")

    init_pts = [(e["params"]["accel"], e["params"]["tau"]) for e in hist if e.get("phase") == "init"]
    if init_pts:
        ax.scatter([p[0] for p in init_pts], [p[1] for p in init_pts],
                   s=LHS_DOT_SIZE, c=C_LHS, alpha=0.45, zorder=2, linewidths=0)

    path_pts, (opt_a, opt_t) = _seq_milestones_chrono(hist, cal)
    if path_pts:
        sx, sy = zip(*[(a, t) for a, t, _ in path_pts])
        ax.plot([*sx, opt_a], [*sy, opt_t], color=C_PATH, linewidth=PATH_LW,
                linestyle="--", dashes=(5, 3), alpha=0.85, zorder=4)
        for k, (x, y, _err) in enumerate(path_pts, start=1):
            ax.scatter(x, y, s=SEQ_DOT_SIZE, c=C_SEQ, edgecolors="black", linewidths=0.45, zorder=5)
            ax.text(x, y, str(k), ha="center", va="center", fontsize=FS_MARKER_NUM,
                    fontweight="700", color="black", zorder=6)

    ax.scatter(opt_a, opt_t, s=STAR_SIZE, marker="*", c=C_OPTIMUM, edgecolors="black", linewidths=0.45, zorder=7)

    _, accel_range, tau_range = load_landscape(SCENE)
    ax.set_xlim(accel_range)
    ax.set_ylim(tau_range)
    t0 = np.ceil(tau_range[0] * 2) / 2
    t1 = np.floor(tau_range[1] * 2) / 2
    ax.set_yticks(np.arange(t0, t1 + 0.01, 0.5))
    ax.set_xlabel(r"$a_{\max}$", fontsize=FS_LABEL, color="black")
    ax.set_ylabel(r"$\tau$", fontsize=FS_LABEL, color="black")
    ax.tick_params(labelsize=FS_TICK, colors="black", direction="out")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    _lcb_inset_legend(ax)

    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.13, top=0.98)

    png = OUT / "lcb_contour.png"
    pdf = OUT / "lcb_contour.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, PAPER_FIG / pdf.name)
    return png, pdf


def main() -> None:
    hist, cal = load_history(SCENE)
    png, pdf = plot_lcb_contour(hist, cal)
    print("Wrote:", png, pdf)


if __name__ == "__main__":
    main()
