"""plot_calibration_convergence.py — 六场景 FLAT 标定收敛（RF / MLP / GP，5 seed）

两种产物（与 n_init 消融图同款工作流）：
  1) paper panels：六个独立 16:9 子图 PDF（paper/figure/calib_conv_*.pdf），
     在 root.tex 里用 \\subfloat 组合；legend 只放子图 (a)（右上角）。
  2) combined overview：单张 3×2 总览 PNG/PDF/SVG（outputs/figures/bfsac_*），供 docs 速览。

每子图：三代理（RF / MLP / GP）的 best-so-far J_b 收敛（均值线 + 5 seed ±1 std 带）
+ N_INIT 处 LHS→序贯虚线 + GP/RF/MLP 逐场景 R² 标注。
数据：outputs/results/comparison_convergence_main.csv（冻结主实验曲线）
R²：outputs/results/surrogate_fit_metrics.csv（已保存的最终训练集拟合指标）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sys

PROJ = Path(__file__).resolve().parents[2]
CAL_ROOT = PROJ / "code" / "calibration"
EXP_ROOT = PROJ / "code" / "experiments"
for p in (CAL_ROOT, EXP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unified_calibration import BUDGET_SUMO, N_INIT  # noqa: E402
from experiment_io import REPLICATE_SEEDS  # noqa: E402
from plot_convergence_utils import draw_mean_with_band  # noqa: E402
from plot_style import (  # noqa: E402
    SCENE_ORDER, SCENE_TITLES, METHOD_COLORS, METHOD_LABELS,
    apply_style, apply_paper_style, save_figure, save_paper_panel,
    add_paper_panel_axes, paper_figure_size, PAPER_3X2_LAYOUT, apply_panel_axis_labels,
    XLABEL_SIMS_SHORT, YLABEL_JB, CI_BAND_LABEL,
)

apply_style(base_font=11.0)

SURROGATES = [
    ("BF-SAC-GP", "GP", METHOD_COLORS["BF-SAC-GP"], "gp"),
    ("BF-SAC-RF", "RF", METHOD_COLORS["BF-SAC-RF"], "rf"),
    ("BF-SAC-MLP", "MLP", METHOD_COLORS["BF-SAC-MLP"], "mlp"),
]

# (scene, col, row, show_legend) — y-label left column; x-label bottom row only.
PAPER_PANEL_LAYOUT = PAPER_3X2_LAYOUT


def load_surrogate_curve(scene: str, method_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = PROJ / "outputs/results/comparison_convergence_main.csv"
    frame = pd.read_csv(source)
    frame = frame[(frame.scene == scene) & (frame.method_key == method_key)
                  & (frame.sim_count <= BUDGET_SUMO)]
    matrix = frame.pivot(index="sim_count", columns="run_seed", values="best_error")
    if set(matrix.columns) != set(REPLICATE_SEEDS) or matrix.isna().any().any():
        raise ValueError(f"Incomplete frozen main curves: {scene}/{method_key}")
    return (matrix.index.to_numpy(dtype=float), matrix.mean(axis=1).to_numpy(),
            matrix.std(axis=1, ddof=1).to_numpy())


def load_fit_metrics(scene: str, surrogate: str) -> float:
    frame = pd.read_csv(PROJ / "outputs/results/surrogate_fit_metrics.csv")
    row = frame[(frame.scene == scene) & (frame.surrogate == surrogate)]
    if len(row) != 1 or not np.isfinite(row.iloc[0].r2):
        raise ValueError(f"Missing saved fit metric: {scene}/{surrogate}")
    return float(row.iloc[0].r2)


def draw_scene_panel(
    ax: plt.Axes,
    scene: str,
    *,
    show_legend: bool,
    show_title: bool,
    paper: bool,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    lw = 1.0 if paper else 2.3
    band_alpha = 0.12 if paper else 0.22
    legend_fs = 4.2 if paper else 8
    r2_fs = legend_fs
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
        r2 = load_fit_metrics(scene, surr)
        if np.isfinite(r2):
            fit_lines.append(f"{tag} $R^2$={r2:.2f}")

    ax.axvline(x=N_INIT, color="#94A3B8", linewidth=0.6 if paper else 1.0,
               linestyle="--", alpha=0.7, zorder=2)
    if show_title:
        ax.set_title(SCENE_TITLES[scene], fontweight="bold", pad=3)
    if paper:
        apply_panel_axis_labels(
            ax,
            show_ylabel=show_ylabel,
            show_xlabel=show_xlabel,
            ylabel=r"$J_b$",
            xlabel="SUMO sims",
        )
    else:
        ax.set_xlabel(XLABEL_SIMS_SHORT)
        ax.set_ylabel(YLABEL_JB)
    ax.set_xlim(0, BUDGET_SUMO)
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        ax.set_ylim(max(0.0, y_lo * 0.92), y_hi * 1.06)
    if paper and show_legend:
        r2_xy, legend_anchor = (0.965, 0.51), (1.0, 0.965)
    else:
        r2_xy, legend_anchor = (0.965, 0.965), (1.0, 0.58 if paper else 0.67)
    r2_kw = dict(
        transform=ax.transAxes, fontsize=r2_fs,
        va="top", ha="right", color="#000000",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
    )
    if paper:
        r2_kw["linespacing"] = 0.95
    ax.text(r2_xy[0], r2_xy[1], "\n".join(fit_lines), **r2_kw)
    if show_legend:
        ax.legend(
            fontsize=legend_fs, loc="upper right",
            bbox_to_anchor=legend_anchor,
            handlelength=0.9, borderpad=0.25, labelspacing=0.18,
        )


def _scene_stem(scene: str) -> str:
    if scene == "XAM-N6":
        return "calib_conv_xam"
    if scene == "Xian":
        return "calib_conv_xian"
    return f"calib_conv_{scene.lower()}"


def export_paper_panels() -> None:
    apply_paper_style(base_font=6.0)
    for scene, col, row, show_legend in PAPER_PANEL_LAYOUT:
        fig = plt.figure(figsize=paper_figure_size(row), facecolor="#FFFFFF")
        ax = add_paper_panel_axes(fig, col=col, row=row)
        draw_scene_panel(
            ax, scene,
            show_legend=show_legend,
            show_title=False,
            paper=True,
            show_ylabel=(col == "left"),
            show_xlabel=(row == "bottom"),
        )
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
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-GP"], lw=2.3, label="FLAT-GP"),
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-RF"], lw=2.3, label="FLAT-RF"),
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-MLP"], lw=2.3, label="FLAT-MLP"),
        mpatches.Patch(color="#94A3B8", alpha=0.22, label=CI_BAND_LABEL),
        plt.Line2D([0], [0], linestyle="--", color="#94A3B8",
                   label=f"LHS end / sequential start ($N_{{init}}$={N_INIT})"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, framealpha=0.94,
               bbox_to_anchor=(0.995, 1.005))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = save_figure(fig, "bfsac_calibration_convergence")
    plt.close(fig)
    return out


def main() -> None:
    print("=" * 60)
    print("FLAT calibration convergence (RF / MLP / GP)")
    export_paper_panels()
    out = export_combined()
    print(f"  Saved combined → {out}")
    print("  Saved paper panels to paper/figure/")


if __name__ == "__main__":
    main()
