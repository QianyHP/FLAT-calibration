"""plot_multiscene_convergence.py — 六场景 × 多方法收敛对比图（2×3 子图）

读取 comparison_convergence.csv；多 seed 时绘制均值曲线 + 半透明 ±1 std 带。
输出: outputs/figures/multiscene_method_convergence.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import sys

PROJ = Path(__file__).resolve().parents[2]
CAL_ROOT = PROJ / "code" / "calibration"
EXP_ROOT = PROJ / "code" / "experiments"
for p in (CAL_ROOT, EXP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unified_calibration import BUDGET_SUMO  # noqa: E402
from experiment_io import aggregate_convergence_per_scene  # noqa: E402
from plot_convergence_utils import draw_mean_with_band  # noqa: E402
from plot_style import (  # noqa: E402
    SCENE_ORDER, SCENE_TITLES, METHOD_ORDER, METHOD_COLORS, METHOD_LABELS,
    apply_style, save_figure, XLABEL_SIMS, YLABEL_JB, CI_BAND_LABEL, budget_vline_label,
)

apply_style()
RES = PROJ / "outputs" / "results"


def load_convergence() -> pd.DataFrame:
    conv = pd.read_csv(RES / "comparison_convergence.csv")
    return conv[conv["sim_count"] <= BUDGET_SUMO]


def _plot_method(ax: plt.Axes, mdf: pd.DataFrame, mkey: str, x_max: int) -> None:
    mdf = mdf[mdf["sim_count"] <= x_max].sort_values("sim_count")
    if mdf.empty:
        return
    x = mdf["sim_count"].to_numpy(dtype=float)
    y_mean = mdf["best_error_mean"].to_numpy(dtype=float)
    y_std = mdf["best_error_std"].fillna(0.0).to_numpy(dtype=float)
    lw = 2.4 if mkey == "BF-SAC-RF" else 1.85
    zorder = 5 if mkey == "BF-SAC-RF" else 3
    draw_mean_with_band(
        ax, x, y_mean, y_std,
        color=METHOD_COLORS[mkey], step=mkey in ("BF-SAC-RF", "BF-SAC-MLP"),
        linewidth=lw, zorder=zorder,
    )


def plot_all_scenes(out_name: str = "multiscene_method_convergence") -> Path:
    conv = load_convergence()
    agg = aggregate_convergence_per_scene(conv)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), facecolor="#FFFFFF")
    axes = axes.flatten()

    for ax, scene in zip(axes, SCENE_ORDER):
        scene_df = agg[agg["scene"] == scene]
        for mkey in METHOD_ORDER:
            _plot_method(ax, scene_df[scene_df["method_key"] == mkey], mkey, BUDGET_SUMO)
        ax.axvline(x=BUDGET_SUMO, color="#94A3B8", linewidth=1.1, linestyle=":", alpha=0.75)
        ax.set_title(SCENE_TITLES[scene], fontweight="bold", pad=6)
        ax.set_xlim(0, BUDGET_SUMO)
        ax.set_xlabel(XLABEL_SIMS, fontsize=10)
        ax.set_ylabel(YLABEL_JB, fontsize=10)

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[k], lw=2.4, label=METHOD_LABELS[k])
        for k in METHOD_ORDER
    ]
    handles.append(mpatches.Patch(facecolor="#94A3B8", alpha=0.22, edgecolor="none", label=CI_BAND_LABEL))
    fig.legend(handles=handles, loc="upper center", ncol=3, framealpha=0.97, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Multi-scene convergence ({budget_vline_label(BUDGET_SUMO)})",
                 fontsize=15, fontweight="bold", y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = save_figure(fig, out_name)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main() -> None:
    print("=" * 60)
    print("Multi-scene method convergence")
    plot_all_scenes()


if __name__ == "__main__":
    main()
