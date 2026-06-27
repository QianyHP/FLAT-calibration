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
    SCORE_PROXY_LABEL, apply_style, apply_paper_style, save_figure, save_paper_panel,
    PAPER_FULL_SIZE, XLABEL_SIMS, YLABEL_JB, budget_vline_label,
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


def _draw_convergence(ax: plt.Axes, conv_df: pd.DataFrame, budget: int, *, paper: bool = False) -> None:
    avg_df = aggregate_convergence_global(conv_df, SCENE_ORDER)
    if paper:
        _draw_convergence_paper(ax, avg_df, budget)
        return

    ref_lw, base_lw = 3.0, 2.15
    for mkey in METHOD_ORDER:
        mdf = avg_df[avg_df["method_key"] == mkey].sort_values("sim_count")
        mdf = mdf[mdf["sim_count"] <= budget]
        if mdf.empty:
            continue
        x = mdf["sim_count"].to_numpy(dtype=float)
        y_mean = mdf["best_error_mean"].to_numpy(dtype=float)
        y_std = mdf["best_error_std"].fillna(0.0).to_numpy(dtype=float)
        lw = ref_lw if mkey == REF_METHOD else base_lw
        draw_mean_with_band(
            ax, x, y_mean, y_std,
            color=METHOD_COLORS[mkey], step=mkey in ("BF-SAC-RF", "BF-SAC-MLP"),
            linewidth=lw, zorder=6 if mkey == REF_METHOD else 3,
            label=METHOD_LABELS[mkey],
        )

    ax.axvline(x=budget, color="#94A3B8", linewidth=1.45, linestyle=":", alpha=0.78)
    ymax_val = ax.get_ylim()[1]
    ax.text(budget - 2, ymax_val * 0.97, budget_vline_label(budget),
            fontsize=10, color="#64748B", va="top", ha="right")
    ax.set_xlabel(XLABEL_SIMS)
    ax.set_ylabel(YLABEL_JB)
    ax.set_xlim(0, budget)
    ax.legend(loc="upper right", framealpha=0.97)


def _draw_convergence_paper(ax: plt.Axes, avg_df: pd.DataFrame, budget: int) -> None:
    """Paper panel: all six methods with ±1 std bands; FLAT lines thicker + step."""
    for mkey in METHOD_ORDER:
        mdf = avg_df[avg_df["method_key"] == mkey].sort_values("sim_count")
        mdf = mdf[mdf["sim_count"] <= budget]
        if mdf.empty:
            continue
        x      = mdf["sim_count"].to_numpy(dtype=float)
        y_mean = mdf["best_error_mean"].to_numpy(dtype=float)
        y_std  = mdf["best_error_std"].fillna(0.0).to_numpy(dtype=float)

        is_rf  = mkey == "BF-SAC-RF"
        is_mlp = mkey == "BF-SAC-MLP"
        is_flat = is_rf or is_mlp
        lw = 1.6 if is_rf else (1.3 if is_mlp else 0.75)
        z  = 7   if is_rf else (5   if is_mlp else 3)

        draw_mean_with_band(
            ax, x, y_mean, y_std,
            color=METHOD_COLORS[mkey],
            step=is_flat,
            linewidth=lw, zorder=z,
            label=METHOD_LABELS[mkey], alpha=0.12,
        )

    ax.axvline(x=budget, color="#B0BAC4", linewidth=0.65, linestyle=":", alpha=0.8, zorder=1)
    ax.set_xlabel(XLABEL_SIMS)
    ax.set_ylabel(YLABEL_JB)
    ax.set_xlim(0, budget)
    ax.legend(
        loc="upper right", fontsize=4.5, ncol=2,
        handlelength=1.0, columnspacing=0.5, borderpad=0.3, labelspacing=0.25,
    )


def _load_method_stderr() -> dict[str, float]:
    p = RES / "comparison_multiseed_stats.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty or df["n_seeds"].max() <= 1:
        return {}
    return {str(k): float(v) for k, v in df.groupby("method_key")["error_std"].mean().items()}


def _load_significance() -> dict[str, float]:
    """baseline -> max p-value over both BF-SAC references (dagger = both variants sig.)."""
    p = RES / "comparison_significance.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    df = df[df["reference"].isin(("BF-SAC-RF", "BF-SAC-MLP"))]
    return {str(k): float(v) for k, v in df.groupby("baseline")["p_value"].max().items()}


def _draw_error_bar(ax: plt.Axes, avg_df: pd.DataFrame, budget: int, *, paper: bool = False) -> None:
    colors = [METHOD_COLORS[m] for m in avg_df["method_key"]]
    x = list(range(len(avg_df)))
    stderr_map = _load_method_stderr()
    sig_map = _load_significance()
    yerr = np.array([stderr_map.get(m, 0.0) for m in avg_df["method_key"]])
    bars = ax.bar(
        x, avg_df["avg_error"], yerr=yerr, capsize=2 if paper else 4,
        color=colors, width=0.62, zorder=3,
        error_kw={"elinewidth": 0.9 if paper else 1.2, "capthick": 0.9 if paper else 1.2},
    )
    label_fs = 4.5 if paper else 10
    for bar, mkey, val, se in zip(bars, avg_df["method_key"], avg_df["avg_error"], yerr):
        is_ref = str(mkey).startswith("BF-SAC")
        p_val = sig_map.get(str(mkey))
        label = f"{val:.3f}"
        if p_val is not None and p_val < 0.05:
            label += "$^\\dagger$"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.004,
            label, ha="center", va="bottom", fontsize=label_fs,
            fontweight="bold" if is_ref else "normal",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in avg_df["method_key"]],
        rotation=0 if paper else 28,
        ha="center" if paper else "right",
    )
    ax.set_ylabel(r"$J_b$")
    ax.set_ylim(0, float(np.max(avg_df["avg_error"] + yerr)) * 1.18)
    ax.text(
        0.015, 0.98,
        r"$^\dagger$ both FLAT variants sig. better ($p<0.05$)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=4.5 if paper else 9.0, color="#334155",
    )


def _export_paper_panels(conv_df: pd.DataFrame, avg: pd.DataFrame, budget: int) -> None:
    apply_paper_style(base_font=5.5)

    fig_c = plt.figure(figsize=PAPER_FULL_SIZE, facecolor="#FFFFFF")
    ax_c = fig_c.add_axes([0.13, 0.26, 0.86, 0.68])
    _draw_convergence(ax_c, conv_df, budget, paper=True)
    save_paper_panel(fig_c, "method_comparison_convergence")
    plt.close(fig_c)

    fig_b = plt.figure(figsize=PAPER_FULL_SIZE, facecolor="#FFFFFF")
    ax_b = fig_b.add_axes([0.13, 0.30, 0.86, 0.64])
    _draw_error_bar(ax_b, avg, budget, paper=True)
    save_paper_panel(fig_b, "method_comparison_budget_bars")
    plt.close(fig_b)

    apply_style(base_font=12.0)


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

    _export_paper_panels(conv_df, avg, budget)

    fig = plt.figure(figsize=(8.4, 9.6), facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.05, 0.95], hspace=0.34)
    ax_conv, ax_err = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    _draw_convergence(ax_conv, conv_df, budget)
    _draw_error_bar(ax_err, avg, budget)
    out = save_figure(fig, "method_comparison_panel")
    plt.close(fig)
    print("  Saved paper panels to paper/Figures/")
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
