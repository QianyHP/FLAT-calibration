"""plot_calibration_convergence.py — 六场景 BF-SAC 标定收敛曲线

展示完整标定流程：60 次 LHS + 序贯 LCB 加点至总预算 101 次 SUMO。
数据来源：calibration/*_with_history.json；不足 101 点时从 comparison_cache BF-SAC 补全。
输出: outputs/figures/bfsac_calibration_convergence.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
CAL_DIR = PROJ / "data" / "processed_data" / "calibration"
CACHE_DIR = PROJ / "outputs" / "results" / "comparison_cache"
FIG_DIR = PROJ / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13.5

N_INIT = 60
BUDGET = 101

SCENE_ORDER = ["Tianjin", "Changchun", "Xian", "YTDJ", "RML", "XAM-N6"]
SCENE_LABELS = {
    "Tianjin":   "Tianjin（信号交叉口）",
    "Changchun": "Changchun（信号交叉口）",
    "Xian":      "Xian（信号交叉口）",
    "YTDJ":      "YTDJ（快速路合流区）",
    "RML":       "RML（快速路交织区）",
    "XAM-N6":    "XAM-N6（高速合流区）",
}

COLORS = ["#2563EB", "#7C3AED", "#DB2777", "#D97706", "#059669", "#DC2626"]


def load_history(scene: str) -> dict | None:
    p = CAL_DIR / f"{scene}_calibration_with_history.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_metrics(scene: str, data: dict) -> tuple[float, float, float]:
    j = float(data["feature_error"])
    r2 = float(data.get("rf_r2") or 0.0)
    rmse = float(data.get("rf_rmse") or 0.0)
    return j, r2, rmse


def _from_calibration_history(data: dict) -> tuple[np.ndarray, np.ndarray] | None:
    hist = data.get("doe_history", [])
    if not hist:
        return None
    has_seq = any(d.get("phase") == "sequential" for d in hist)
    max_idx = max(int(d["sim_index"]) for d in hist)
    if not has_seq and max_idx <= N_INIT:
        return None
    rows = sorted(
        (int(d["sim_index"]), float(d["best_so_far"]))
        for d in hist
        if int(d["sim_index"]) <= BUDGET
    )
    if len(rows) < BUDGET:
        return None
    sims = np.array([r[0] for r in rows], dtype=float)
    bests = np.array([r[1] for r in rows], dtype=float)
    return sims, bests


def _from_comparison_cache(scene: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = CACHE_DIR / f"{scene}_BF-SAC_b{BUDGET}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    sims = np.array(raw.get("convergence_sims", []), dtype=float)
    bests = np.array(raw.get("convergence_best", []), dtype=float)
    if sims.size < BUDGET:
        return None
    mask = sims <= BUDGET
    return sims[mask], bests[mask]


def build_convergence(scene: str, data: dict | None) -> tuple[np.ndarray, np.ndarray]:
    if data is not None:
        curve = _from_calibration_history(data)
        if curve is not None:
            return curve
    curve = _from_comparison_cache(scene)
    if curve is not None:
        return curve
    raise RuntimeError(
        f"{scene}: 缺少 {BUDGET} 点 BF-SAC 收敛数据，"
        f"请运行 unified_calibration.py {scene} 或 run_comparison.py --scenes {scene}"
    )


def plot_panel(ax, scene: str, data: dict | None, color: str) -> None:
    sims, bests = build_convergence(scene, data)
    if data is not None:
        _, r2, rmse = load_metrics(scene, data)
    else:
        cache = json.loads((CACHE_DIR / f"{scene}_BF-SAC_b{BUDGET}.json").read_text(encoding="utf-8"))
        _, r2, rmse = load_metrics(scene, cache)

    mask_init = sims <= N_INIT
    ax.step(
        sims[mask_init], bests[mask_init], where="post",
        color=color, linewidth=2.3, zorder=4, label="_nolegend_",
    )
    mask_seq = sims > N_INIT
    if mask_seq.any():
        ax.step(
            sims[mask_seq], bests[mask_seq], where="post",
            color=color, linewidth=2.3, linestyle="-", zorder=4,
        )
        if mask_init.any():
            ax.plot(
                [sims[mask_init][-1], sims[mask_seq][0]],
                [bests[mask_init][-1], bests[mask_seq][0]],
                color=color, linewidth=2.3, zorder=4,
            )

    ax.axvline(x=N_INIT, color="#94A3B8", linewidth=1.0, linestyle="--", alpha=0.75, zorder=2)
    ax.axvline(x=BUDGET, color="#CBD5E1", linewidth=0.9, linestyle=":", alpha=0.65, zorder=2)

    n_end = int(sims[-1])
    y_end = float(bests[-1])
    ax.scatter([n_end], [y_end], color="#DC2626", s=165, zorder=7, marker="*",
               edgecolors="#FEE2E2", linewidths=0.6)
    doe_end = float(bests[mask_init][-1]) if mask_init.any() else y_end
    if y_end < doe_end - 1e-6:
        ax.annotate(
            "", xy=(n_end, y_end), xytext=(N_INIT, doe_end),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.1),
        )

    metric_handles = [
        plt.Line2D([], [], linestyle="", label=f"RF $R^2$={r2:.3f}"),
        plt.Line2D([], [], linestyle="", label=f"RMSE={rmse:.3f}"),
    ]
    ax.legend(
        handles=metric_handles, loc="upper right", bbox_to_anchor=(0.90, 0.98),
        fontsize=10, framealpha=0.92, facecolor="white", edgecolor="#CBD5E1",
        handlelength=0, handletextpad=0.3, borderpad=0.35, labelcolor="#1E40AF",
    )

    ax.set_title(SCENE_LABELS[scene], fontsize=13, fontweight="bold", pad=4)
    ax.set_xlabel("累计 SUMO 仿真次数", fontsize=11.5)
    ax.set_ylabel("运行最优行为特征误差 $J_b$", fontsize=11.5)
    ax.set_xlim(0, BUDGET + 8)
    y_min = min(y_end, bests.min()) * 0.92
    y_max = max(bests[0], bests[: max(1, int(np.sum(mask_init)))].max()) * 1.06
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.28, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    print(f"    终点 J_b={y_end:.4f}")


def main() -> None:
    print("=" * 60)
    print("六场景 BF-SAC 标定收敛曲线")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 6.6), dpi=300)
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    for i, scene in enumerate(SCENE_ORDER):
        print(f"  [{scene}]")
        data = load_history(scene)
        plot_panel(axes_flat[i], scene, data, COLORS[i])

    init_patch = mpatches.Patch(color="#2563EB", label=f"初始设计阶段（LHS，{N_INIT} 次）")
    seq_patch = mpatches.Patch(color="#2563EB", alpha=0.55,
                               label=f"序贯 LCB 加点（{BUDGET - N_INIT} 次）")
    star_patch = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#DC2626",
                            markersize=11, label=f"标定终点（{BUDGET} 次预算）")
    v60_patch = plt.Line2D([0], [0], linestyle="--", color="#94A3B8",
                           label="初始设计结束 / 序贯加点开始")
    v101_patch = plt.Line2D([0], [0], linestyle=":", color="#CBD5E1",
                            label=f"总仿真预算 {BUDGET} 次")
    fig.legend(
        handles=[init_patch, seq_patch, star_patch, v60_patch, v101_patch],
        loc="lower center", ncol=3, fontsize=10.2, framealpha=0.94,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.98])
    out = FIG_DIR / "bfsac_calibration_convergence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
