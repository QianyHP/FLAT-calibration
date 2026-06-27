"""plot_calibration_convergence.py — 六场景 FLAT 标定收敛（RF vs MLP，5 seed）

两种产物（与 n_init 消融图同款工作流）：
  1) paper panels：六个独立 16:9 子图 PDF（paper/Figures/calib_conv_*.pdf），
     在 root.tex 里用 \\subfloat 组合；legend 只放子图 (a)（右上角）。
  2) combined overview：单张 3×2 总览 PNG/PDF/SVG（outputs/figures/bfsac_*），供 docs 速览。

每子图：两代理（RF / MLP）的 best-so-far J_b 收敛（均值线 + 5 seed ±1 std 带）
+ N_INIT 处 LHS→序贯虚线 + 逐场景 R² 标注（learning 证据）。
数据：comparison_cache/{场景}_BF-SAC-{RF,MLP}_b{BUDGET}[_s{seed}].json  （键名历史前缀）
R²：data/processed_data/calibration/{场景}_calibration[__mlp_h64-64_m10].json
"""
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
    SCENE_ORDER, SCENE_TITLES, METHOD_COLORS, METHOD_LABELS,
    apply_style, apply_paper_style, save_figure, save_paper_panel,
    add_paper_panel_axes, PAPER_HALF_SIZE, XLABEL_SIMS_SHORT, YLABEL_JB, CI_BAND_LABEL,
)

apply_style(base_font=11.0)
CAL_DIR = PROJ / "data" / "processed_data" / "calibration"

SURROGATES = [
    ("BF-SAC-RF", "RF", METHOD_COLORS["BF-SAC-RF"], "rf"),
    ("BF-SAC-MLP", "MLP", METHOD_COLORS["BF-SAC-MLP"], "mlp"),
]

# SIND left column / UTE right column (same scene order as Fig. 4); legend only in (a).
PAPER_PANEL_LAYOUT = [
    ("Tianjin", True),
    ("YTDJ", False),
    ("Changchun", False),
    ("RML", False),
    ("Xian", False),
    ("XAM-N6", False),
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


def draw_scene_panel(ax: plt.Axes, scene: str, *, show_legend: bool, show_title: bool, paper: bool) -> None:
    lw = 1.0 if paper else 2.3
    band_alpha = 0.12 if paper else 0.22
    r2_fs = 5.0 if paper else 9.5
    fit_lines, y_lo, y_hi = [], np.inf, -np.inf
    for method_key, tag, color, surr in SURROGATES:
        sims, mean, std = load_surrogate_curve(scene, method_key)
        draw_mean_with_band(ax, sims, mean, std, color=color, step=True,
                            linewidth=lw, zorder=4, label=METHOD_LABELS[method_key], alpha=band_alpha)
        if not paper:
            ax.scatter([sims[-1]], [mean[-1]], color=color, s=55, zorder=7,
                       marker="o", edgecolors="white", linewidths=0.8)
        y_lo = min(y_lo, float((mean - std).min()))
        y_hi = max(y_hi, float(mean[: max(1, N_INIT)].max()))
        fit_lines.append(f"{tag} $R^2$={load_fit_metrics(scene, surr):.2f}")

    ax.axvline(x=N_INIT, color="#94A3B8", linewidth=0.6 if paper else 1.0,
               linestyle="--", alpha=0.7, zorder=2)
    if show_title:
        ax.set_title(SCENE_TITLES[scene], fontweight="bold", pad=3)
    ax.set_xlabel(XLABEL_SIMS_SHORT)
    ax.set_ylabel(YLABEL_JB)
    ax.set_xlim(0, BUDGET_SUMO)
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        ax.set_ylim(max(0.0, y_lo * 0.92), y_hi * 1.06)
    ax.text(
        0.965, 0.965, "\n".join(fit_lines), transform=ax.transAxes, fontsize=r2_fs,
        va="top", ha="right", color="#334155",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
    )
    if show_legend:
        ax.legend(
            fontsize=4.6 if paper else 8, loc="upper right",
            bbox_to_anchor=(1.0, 0.65 if paper else 0.67),
            handlelength=0.9, borderpad=0.25, labelspacing=0.22,
        )


def _scene_stem(scene: str) -> str:
    if scene == "XAM-N6":
        return "calib_conv_xam"
    if scene == "Xian":
        return "calib_conv_xian"
    return f"calib_conv_{scene.lower()}"


def export_paper_panels() -> None:
    apply_paper_style(base_font=6.0)
    for scene, show_legend in PAPER_PANEL_LAYOUT:
        fig = plt.figure(figsize=PAPER_HALF_SIZE, facecolor="#FFFFFF")
        ax = add_paper_panel_axes(fig)
        draw_scene_panel(ax, scene, show_legend=show_legend, show_title=False, paper=True)
        save_paper_panel(fig, _scene_stem(scene))
        plt.close(fig)
        print(f"  paper panel → {_scene_stem(scene)}.pdf")
    apply_style(base_font=11.0)


def export_combined() -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 11.4))
    fig.patch.set_facecolor("white")
    for ax, scene in zip(axes.flatten(), SCENE_ORDER):
        draw_scene_panel(ax, scene, show_legend=False, show_title=True, paper=False)
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-RF"], lw=2.3, label="FLAT-RF"),
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-MLP"], lw=2.3, label="FLAT-MLP"),
        mpatches.Patch(color="#94A3B8", alpha=0.22, label=CI_BAND_LABEL),
        plt.Line2D([0], [0], linestyle="--", color="#94A3B8",
                   label=f"LHS end / sequential start ($N_{{init}}$={N_INIT})"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, framealpha=0.94,
               bbox_to_anchor=(0.995, 1.005))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = save_figure(fig, "bfsac_calibration_convergence")
    plt.close(fig)
    return out


def main() -> None:
    print("=" * 60)
    print("FLAT calibration convergence (RF vs MLP)")
    export_paper_panels()
    out = export_combined()
    print(f"  Saved combined → {out}")
    print("  Saved paper panels to paper/Figures/")


if __name__ == "__main__":
    main()
