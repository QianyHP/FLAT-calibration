"""plot_method_comparison.py — 对比/消融结果组合图

读取 run_comparison.py 输出的 comparison_summary.csv、comparison_convergence.csv。
子图为收敛对比、预算快照误差、评分 proxy（max(0, 100−45·J_b)）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "outputs" / "results"
FIG = PROJ / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 300

SPSA_LABEL = "SPSA"

METHOD_KEY_ORDER = ["BF-SAC", "No-LHS", "No-RF", "SPSA", "GA"]
METHOD_COLORS = {
    "BF-SAC": "#1E40AF",
    "No-LHS": "#60A5FA",
    "No-RF": "#93C5FD",
    "SPSA": "#F97316",
    "GA": "#DC2626",
}

SCORE_PROXY_LABEL = "评分 proxy (100−45·J_b)"
SCENE_ORDER = ["Tianjin", "Changchun", "Xian", "YTDJ", "RML", "XAM-N6"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RES / "comparison_summary.csv")
    conv = pd.read_csv(RES / "comparison_convergence.csv")
    summary["method"] = summary["method"].replace({"SPSA": SPSA_LABEL})
    if "backend" not in summary.columns:
        raise ValueError(
            "comparison_summary.csv 格式过旧，请先运行: "
            "python code/experiments/run_comparison.py"
        )
    bad = summary[summary["backend"] != "sumo"]
    if not bad.empty:
        raise ValueError("comparison_summary.csv 存在 backend!=sumo 的记录")
    return summary, conv


def _score_proxy_col(summary_df: pd.DataFrame) -> str:
    if "score_proxy_at_budget" not in summary_df.columns:
        raise ValueError("comparison_summary.csv 缺少 score_proxy_at_budget 列")
    return "score_proxy_at_budget"


def _aggregate_convergence(conv_df: pd.DataFrame) -> pd.DataFrame:
    """六场景在相同 sim_count 上对 best_error 取算术平均。"""
    scoped = conv_df[conv_df["scene"].isin(SCENE_ORDER)].copy()
    return (
        scoped.groupby(["method_key", "sim_count"], as_index=False)
        .agg(best_error=("best_error", "mean"), method=("method", "first"))
    )


def _draw_convergence(ax: plt.Axes, conv_df: pd.DataFrame, budget: int) -> None:
    avg_df = _aggregate_convergence(conv_df)
    x_max = max(150, budget + 20)
    label_map = avg_df.groupby("method_key")["method"].first().to_dict()

    for mkey in METHOD_KEY_ORDER:
        mdf = avg_df[avg_df["method_key"] == mkey].copy()
        if mdf.empty:
            continue
        mdf = mdf[mdf["sim_count"] <= x_max].sort_values("sim_count")
        color = METHOD_COLORS.get(mkey, "#666")
        label = label_map.get(mkey, mkey)
        x = mdf["sim_count"].to_numpy(dtype=float)
        y = mdf["best_error"].to_numpy(dtype=float)
        lw = 3.0 if mkey == "BF-SAC" else 2.15
        zorder = 6 if mkey == "BF-SAC" else 3
        ls = "-" if mkey in ("SPSA", "GA", "BF-SAC") else (0, (4, 3))
        if mkey in ("SPSA", "GA"):
            ax.plot(x, y, color=color, linewidth=lw, linestyle=ls,
                    zorder=zorder, label=label, alpha=0.97)
        else:
            ax.step(x, y, where="post", color=color, linewidth=lw, linestyle=ls,
                    zorder=zorder, label=label, alpha=0.97)

    bfsac_at = avg_df[
        (avg_df["method_key"] == "BF-SAC") & (avg_df["sim_count"] == budget)
    ]
    if not bfsac_at.empty:
        final_val = float(bfsac_at.iloc[0]["best_error"])
        ax.scatter([budget], [final_val], color="#1E40AF", s=168, zorder=9, marker="*",
                   edgecolors="#E8EEF9", linewidths=0.85)
        ax.annotate(
            f"BF-SAC @{budget}\n平均 $J_b$={final_val:.3f}",
            xy=(budget, final_val), xytext=(budget + 22, final_val * 1.28),
            fontsize=12, color="#1E40AF", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=1.25),
        )

    ax.axvline(x=budget, color="#94A3B8", linewidth=1.45, linestyle=":", alpha=0.78)
    ymax_val = ax.get_ylim()[1]
    ax.text(budget + 1.8, ymax_val * 0.97, f"{budget}次\n预算线",
            fontsize=12, color="#64748B", va="top")

    ax.set_xlabel("累计 SUMO 仿真次数", fontsize=14)
    ax.set_ylabel("六场景平均运行最优 $J_b$", fontsize=14)
    ax.set_xlim(0, x_max)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.97)


def _draw_error_bar(ax: plt.Axes, avg_df: pd.DataFrame, budget: int) -> None:
    colors = [METHOD_COLORS.get(m, "#666") for m in avg_df["method_key"]]
    x = list(range(len(avg_df)))
    bars = ax.bar(x, avg_df["avg_error"], color=colors, width=0.64, zorder=3)
    for bar, val in zip(bars, avg_df["avg_error"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(avg_df["method"].tolist()), rotation=32, ha="right", fontsize=11)
    ax.set_ylabel(f"六场景平均 $J_b$ @{budget}次", fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_score_proxy_bar(ax: plt.Axes, avg_df: pd.DataFrame) -> None:
    colors = [METHOD_COLORS.get(m, "#666") for m in avg_df["method_key"]]
    x = list(range(len(avg_df)))
    bars = ax.bar(x, avg_df["avg_score_proxy"], color=colors, width=0.64, zorder=3)
    for bar, val in zip(bars, avg_df["avg_score_proxy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(_short_labels(avg_df["method"].tolist()), rotation=32, ha="right", fontsize=11)
    ax.set_ylabel(SCORE_PROXY_LABEL, fontsize=13)
    ax.set_ylim(50, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _short_labels(methods: list[str]) -> list[str]:
    out = []
    for m in methods:
        if m == SPSA_LABEL:
            out.append("SPSA")
        else:
            out.append(m.replace("（", "\n（", 1) if "（" in m else m)
    return out


def plot_combined(
    summary_df: pd.DataFrame,
    conv_df: pd.DataFrame,
    out_name: str = "method_comparison_panel.png",
) -> tuple[Path, pd.DataFrame]:
    budget = int(summary_df["budget_fair"].iloc[0]) if "budget_fair" in summary_df.columns else 101
    err_col = "error_at_budget"
    proxy_col = _score_proxy_col(summary_df)

    avg = (
        summary_df.groupby(["method_key", "method"])[[err_col, proxy_col, "n_sims"]]
        .mean()
        .reset_index()
    )
    avg.rename(columns={err_col: "avg_error", proxy_col: "avg_score_proxy"}, inplace=True)
    order_map = {k: i for i, k in enumerate(METHOD_KEY_ORDER)}
    avg["order"] = avg["method_key"].map(order_map)
    avg = avg[avg["method_key"].isin(METHOD_KEY_ORDER)].sort_values("order").reset_index(drop=True)

    bfsac_row = avg[avg["method_key"] == "BF-SAC"]
    if bfsac_row.empty:
        bfsac_err = float(avg["avg_error"].min())
        bfsac_proxy = float(avg["avg_score_proxy"].max())
    else:
        bfsac_err = float(bfsac_row["avg_error"].iloc[0])
        bfsac_proxy = float(bfsac_row["avg_score_proxy"].iloc[0])
    avg["efficiency_gain_pct"] = (bfsac_proxy - avg["avg_score_proxy"]) / max(bfsac_proxy, 1e-9) * 100.0
    avg["error_increase_pct"] = (avg["avg_error"] - bfsac_err) / max(bfsac_err, 1e-9) * 100.0

    fig = plt.figure(figsize=(14.6, 10.0), dpi=300, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.9], hspace=0.28, wspace=0.32)

    ax_conv = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_proxy = fig.add_subplot(gs[1, 1])

    _draw_convergence(ax_conv, conv_df, budget)
    _draw_error_bar(ax_err, avg, budget)
    _draw_score_proxy_bar(ax_proxy, avg)

    ax_conv.set_title("六场景平均收敛对比", fontsize=14, fontweight="bold", pad=8)
    ax_err.set_title(f"{budget} 次预算平均 $J_b$", fontsize=14, fontweight="bold", pad=8)
    ax_proxy.set_title(SCORE_PROXY_LABEL, fontsize=14, fontweight="bold", pad=8)

    out = FIG / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out, avg


def print_markdown_table(avg_df: pd.DataFrame) -> None:
    bfsac_err = float(avg_df.loc[avg_df["method_key"] == "BF-SAC", "avg_error"].iloc[0])
    print("\n| 方法 | 仿真次数 | 预算快照 $J_b$ | 评分proxy | 误差增幅 |")
    print("|:----:|:--------:|:-------------:|:---------:|:--------:|")
    for _, row in avg_df.iterrows():
        m = row["method"]
        err = float(row["avg_error"])
        is_ref = row["method_key"] == "BF-SAC"
        rel = "—" if is_ref else f"+{(err/bfsac_err-1)*100:.1f}%"
        b = "**" if is_ref else ""
        print(
            f"| {b}{m}{b} | {int(row['n_sims'])} | {b}{err:.3f}{b} | "
            f"{b}{row['avg_score_proxy']:.1f}{b} | {rel} |"
        )


def main() -> None:
    print("=" * 60)
    print("绘制对比/消融组合图")
    summary_df, conv_df = load_data()
    _, avg_df = plot_combined(summary_df, conv_df)
    print_markdown_table(avg_df)
    tbl = RES / "ablation_summary.csv"
    avg_df.to_csv(tbl, index=False, encoding="utf-8-sig")
    print(f"  Wrote: {tbl}")


if __name__ == "__main__":
    main()
