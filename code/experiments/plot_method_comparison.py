"""plot_method_comparison.py — Method comparison panel (convergence + budget bars)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import sys

PROJ = Path(__file__).resolve().parents[2]
CAL_ROOT = PROJ / "code" / "calibration"
EXP_ROOT = PROJ / "code" / "experiments"
for p in (CAL_ROOT, EXP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unified_calibration import BUDGET_SUMO  # noqa: E402
from experiment_io import aggregate_convergence_global  # noqa: E402
from plot_convergence_utils import draw_mean_with_band  # noqa: E402
from plot_style import (  # noqa: E402
    SCENE_ORDER, METHOD_ORDER, METHOD_COLORS, METHOD_LABELS, REF_METHOD,
    SCORE_PROXY_LABEL, apply_style, save_figure, XLABEL_SIMS, YLABEL_JB,
    budget_vline_label,
)

apply_style(base_font=12.0)
RES = PROJ / "outputs" / "results"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RES / "comparison_summary.csv")
    conv = pd.read_csv(RES / "comparison_convergence.csv")
    conv = conv[conv["sim_count"] <= BUDGET_SUMO]
    if "backend" not in summary.columns:
        raise ValueError("Stale comparison_summary.csv — rebuild with run_comparison.py")
    return summary, conv


def _score_proxy_col(summary_df: pd.DataFrame) -> str:
    if "score_proxy_at_budget" not in summary_df.columns:
        raise ValueError("comparison_summary.csv missing score_proxy_at_budget")
    return "score_proxy_at_budget"


def _draw_convergence(ax: plt.Axes, conv_df: pd.DataFrame, budget: int) -> None:
    avg_df = aggregate_convergence_global(conv_df, SCENE_ORDER)
    for mkey in METHOD_ORDER:
        mdf = avg_df[avg_df["method_key"] == mkey].sort_values("sim_count")
        mdf = mdf[mdf["sim_count"] <= budget]
        if mdf.empty:
            continue
        x = mdf["sim_count"].to_numpy(dtype=float)
        y_mean = mdf["best_error_mean"].to_numpy(dtype=float)
        y_std = mdf["best_error_std"].fillna(0.0).to_numpy(dtype=float)
        lw = 3.0 if mkey == REF_METHOD else 2.15
        draw_mean_with_band(
            ax, x, y_mean, y_std,
            color=METHOD_COLORS[mkey], step=mkey in ("BF-SAC-RF", "BF-SAC-MLP"),
            linewidth=lw, zorder=6 if mkey == REF_METHOD else 3,
            label=METHOD_LABELS[mkey],
        )

    bfsac_at = avg_df[(avg_df["method_key"] == REF_METHOD) & (avg_df["sim_count"] == budget)]
    if not bfsac_at.empty:
        final_val = float(bfsac_at.iloc[0]["best_error_mean"])
        ax.scatter([budget], [final_val], color=METHOD_COLORS[REF_METHOD], s=168, zorder=9,
                   marker="*", edgecolors="#E8EEF9", linewidths=0.85)
        ax.annotate(
            f"BF-SAC-RF @{budget}\n$J_b$={final_val:.3f}",
            xy=(budget, final_val), xytext=(budget * 0.72, final_val * 1.12),
            fontsize=10, color=METHOD_COLORS[REF_METHOD], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=METHOD_COLORS[REF_METHOD], lw=1.1),
        )

    ax.axvline(x=budget, color="#94A3B8", linewidth=1.45, linestyle=":", alpha=0.78)
    ymax_val = ax.get_ylim()[1]
    ax.text(budget - 2, ymax_val * 0.97, budget_vline_label(budget),
            fontsize=10, color="#64748B", va="top", ha="right")
    ax.set_xlabel(XLABEL_SIMS)
    ax.set_ylabel(YLABEL_JB)
    ax.set_xlim(0, budget)
    ax.legend(loc="upper right", framealpha=0.97)


def _load_method_stderr() -> dict[str, float]:
    p = RES / "comparison_multiseed_stats.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty or df["n_seeds"].max() <= 1:
        return {}
    return {str(k): float(v) for k, v in df.groupby("method_key")["error_std"].mean().items()}


def _draw_error_bar(ax: plt.Axes, avg_df: pd.DataFrame, budget: int) -> None:
    colors = [METHOD_COLORS[m] for m in avg_df["method_key"]]
    x = list(range(len(avg_df)))
    stderr_map = _load_method_stderr()
    yerr = np.array([stderr_map.get(m, 0.0) for m in avg_df["method_key"]])
    bars = ax.bar(x, avg_df["avg_error"], yerr=yerr, capsize=4, color=colors, width=0.64, zorder=3,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})
    for bar, val, se in zip(bars, avg_df["avg_error"], yerr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in avg_df["method_key"]],
                       rotation=28, ha="right")
    ax.set_ylabel(rf"$J_b$ @ {budget} sims")


def _draw_score_proxy_bar(ax: plt.Axes, avg_df: pd.DataFrame) -> None:
    colors = [METHOD_COLORS[m] for m in avg_df["method_key"]]
    x = list(range(len(avg_df)))
    bars = ax.bar(x, avg_df["avg_score_proxy"], color=colors, width=0.64, zorder=3)
    for bar, val in zip(bars, avg_df["avg_score_proxy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in avg_df["method_key"]],
                       rotation=28, ha="right")
    ax.set_ylabel(SCORE_PROXY_LABEL)
    ax.set_ylim(50, 100)


def plot_combined(summary_df: pd.DataFrame, conv_df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    budget = BUDGET_SUMO
    err_col, proxy_col = "error_at_budget", _score_proxy_col(summary_df)
    avg = (
        summary_df.groupby(["method_key", "method"])[[err_col, proxy_col, "n_sims"]]
        .mean().reset_index()
        .rename(columns={err_col: "avg_error", proxy_col: "avg_score_proxy"})
    )
    order_map = {k: i for i, k in enumerate(METHOD_ORDER)}
    avg = avg[avg["method_key"].isin(METHOD_ORDER)].assign(
        order=lambda d: d["method_key"].map(order_map),
    ).sort_values("order").reset_index(drop=True)

    fig = plt.figure(figsize=(14.6, 10.0), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.9], hspace=0.28, wspace=0.32)
    ax_conv, ax_err, ax_proxy = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    _draw_convergence(ax_conv, conv_df, budget)
    _draw_error_bar(ax_err, avg, budget)
    _draw_score_proxy_bar(ax_proxy, avg)

    n_scenes = summary_df["scene"].nunique()
    scope = f"{n_scenes}-scene mean" if n_scenes > 1 else summary_df["scene"].iloc[0]
    ax_conv.set_title(f"Convergence ({scope})", fontweight="bold", pad=8)
    ax_err.set_title(rf"$J_b$ @ {budget} sims ({scope})", fontweight="bold", pad=8)
    ax_proxy.set_title(SCORE_PROXY_LABEL, fontweight="bold", pad=8)

    out = save_figure(fig, "method_comparison_panel")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out, avg


def main() -> None:
    print("=" * 60)
    print("Method comparison panel")
    summary_df, conv_df = load_data()
    _, avg_df = plot_combined(summary_df, conv_df)
    tbl = RES / "ablation_summary.csv"
    avg_df.to_csv(tbl, index=False, encoding="utf-8-sig")
    print(f"  Wrote: {tbl}")


if __name__ == "__main__":
    main()
