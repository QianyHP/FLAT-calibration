"""plot_multiscene_convergence.py — 六场景 × 多方法收敛对比图（2×3 子图）

读取 comparison_convergence.csv。
输出: outputs/figures/multiscene_method_convergence.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "outputs" / "results"
FIG = PROJ / "outputs" / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

SCENE_ORDER = ["Tianjin", "Changchun", "Xian", "YTDJ", "RML", "XAM-N6"]
SCENE_TITLES = {
    "Tianjin": "Tianjin",
    "Changchun": "Changchun",
    "Xian": "Xian",
    "YTDJ": "YTDJ",
    "RML": "RML",
    "XAM-N6": "XAM-N6",
}

METHOD_ORDER = ["BF-SAC", "No-LHS", "No-RF", "SPSA", "GA"]
METHOD_COLORS = {
    "BF-SAC": "#1E40AF",
    "No-LHS": "#60A5FA",
    "No-RF": "#93C5FD",
    "SPSA": "#F97316",
    "GA": "#DC2626",
}
METHOD_LABELS = {
    "BF-SAC": "BF-SAC（序贯LCB）",
    "No-LHS": "无LHS（随机 101）",
    "No-RF": "无RF（LHS 101）",
    "SPSA": "SPSA",
    "GA": "遗传算法（GA）",
}


def load_convergence() -> pd.DataFrame:
    return pd.read_csv(RES / "comparison_convergence.csv")


def _plot_method(ax: plt.Axes, mdf: pd.DataFrame, mkey: str, x_max: int) -> None:
    mdf = mdf[mdf["sim_count"] <= x_max].sort_values("sim_count")
    if mdf.empty:
        return
    x = mdf["sim_count"].to_numpy(dtype=float)
    y = mdf["best_error"].to_numpy(dtype=float)
    color = METHOD_COLORS[mkey]
    lw = 2.4 if mkey == "BF-SAC" else 1.85
    zorder = 5 if mkey == "BF-SAC" else 3
    if mkey in ("SPSA", "GA"):
        ls = "-"
        ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, zorder=zorder, alpha=0.95)
    else:
        ls = (0, (4, 3)) if mkey != "BF-SAC" else "-"
        ax.step(x, y, where="post", color=color, linewidth=lw, linestyle=ls, zorder=zorder, alpha=0.95)


def plot_all_scenes(
    budget: int = 101,
    x_max: int = 200,
    out_name: str = "multiscene_method_convergence.png",
) -> Path:
    conv = load_convergence()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), dpi=300, facecolor="#FFFFFF")
    axes = axes.flatten()

    for ax, scene in zip(axes, SCENE_ORDER):
        scene_df = conv[conv["scene"] == scene]
        for mkey in METHOD_ORDER:
            mdf = scene_df[scene_df["method_key"] == mkey]
            _plot_method(ax, mdf, mkey, x_max)

        ax.axvline(x=budget, color="#94A3B8", linewidth=1.1, linestyle=":", alpha=0.75)
        ax.set_title(SCENE_TITLES[scene], fontsize=13, fontweight="bold", pad=6)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("累计 SUMO 仿真次数", fontsize=10)
        ax.set_ylabel("运行最优 $J_b$", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[k], lw=2.4, ls="-" if k in ("BF-SAC", "SPSA", "GA") else (0, (4, 3)),
                   label=METHOD_LABELS[k])
        for k in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10.5,
               framealpha=0.97, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"六场景收敛对比（虚线：{budget} 次预算）", fontsize=15, fontweight="bold", y=1.06)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main() -> None:
    print("=" * 60)
    print("六场景多方法收敛对比图")
    plot_all_scenes()


if __name__ == "__main__":
    main()
