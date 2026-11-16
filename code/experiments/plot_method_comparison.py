"""plot_method_comparison.py — Method comparison panel (convergence + budget bars)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
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
from plot_convergence_utils import draw_mean_with_band, CI_ALPHA  # noqa: E402
from plot_style import (  # noqa: E402
    SCENE_ORDER, METHOD_ORDER, METHOD_COLORS, METHOD_LABELS, REF_METHOD, FLAT_SURROGATE_ORDER,
    SCORE_PROXY_LABEL, apply_style, apply_paper_style, save_figure, save_paper_panel, save_archive_panel,
    PAPER_FULL_SIZE, XLABEL_SIMS, YLABEL_JB, budget_vline_label,
)

apply_style(base_font=12.0)
RES = PROJ / "outputs" / "results"

# Paper Fig. 5 inset layout (axes fraction; top edge flush with main axes top)
PAPER_INSET_HEIGHT = 0.30
PAPER_INSET_LW_SCALE = 0.60
PAPER_INSET_ZOOM_LEFT = [0.10, 1.0 - PAPER_INSET_HEIGHT, 0.24, PAPER_INSET_HEIGHT]
PAPER_INSET_ZOOM_RIGHT = [0.42, 1.0 - PAPER_INSET_HEIGHT, 0.24, PAPER_INSET_HEIGHT]
PAPER_INSET_ZOOM_RANGES = ((0, 40), (40, 80))
PAPER_INSET_SIMS_ABBR = "sims"
PAPER_LEGEND_FS = 4.5
PAPER_INSET_LABEL_FS = 4.0
PAPER_BUDGET_TICK_SHIFT = 2.5


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RES / "comparison_summary_main.csv")
    conv = pd.read_csv(RES / "comparison_convergence_main.csv")
    conv = conv[conv["sim_count"] <= BUDGET_SUMO]
    if "backend" not in summary.columns:
        raise ValueError("Stale comparison_summary_main.csv — rebuild with run_comparison.py")
    return summary, conv


def _score_proxy_col(summary_df: pd.DataFrame) -> str:
    if "score_proxy_at_budget" not in summary_df.columns:
        raise ValueError("comparison_summary_main.csv missing score_proxy_at_budget")
    return "score_proxy_at_budget"


def _final_error_at_budget(avg_df: pd.DataFrame, budget: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for mkey in METHOD_ORDER:
        mdf = avg_df[avg_df["method_key"] == mkey].sort_values("sim_count")
        if mdf.empty:
            out[mkey] = float("inf")
            continue
        at_budget = mdf[mdf["sim_count"] == budget]
        out[mkey] = float(at_budget["best_error_mean"].iloc[0]) if not at_budget.empty else float(mdf["best_error_mean"].iloc[-1])
    return out


def _curve_draw_order(avg_df: pd.DataFrame, budget: int) -> list[str]:
    """Worst final error first (bottom layer); best last (top layer)."""
    finals = _final_error_at_budget(avg_df, budget)
    return sorted(METHOD_ORDER, key=lambda k: finals.get(k, float("inf")), reverse=True)


def _paper_legend_keys() -> list[str | None]:
    """Matplotlib ncol=2 fills column-major: left col FLAT (GP→RF→MLP), then baselines."""
    flat = list(FLAT_SURROGATE_ORDER)
    base = [k for k in METHOD_ORDER if k not in FLAT_SURROGATE_ORDER]
    n_rows = max(len(flat), len(base))
    left: list[str | None] = [flat[i] if i < len(flat) else None for i in range(n_rows)]
    right: list[str | None] = [base[i] if i < len(base) else None for i in range(n_rows)]
    return left + right


def _draw_paper_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_handles: list = []
    ordered_labels: list[str] = []
    for mkey in _paper_legend_keys():
        if mkey is None:
            ordered_handles.append(
                Line2D([], [], color="none", marker="none", linestyle="none", linewidth=0, label=" "),
            )
            ordered_labels.append(" ")
            continue
        label = METHOD_LABELS[mkey]
        ordered_handles.append(by_label[label])
        ordered_labels.append(label)
    ax.legend(
        ordered_handles, ordered_labels,
        loc="upper right", fontsize=PAPER_LEGEND_FS, ncol=2,
        framealpha=0.92, handlelength=1.0, columnspacing=0.5,
        borderpad=0.3, labelspacing=0.25,
    )


def _method_curve_style(mkey: str, *, paper: bool, zorder: int, inset: bool = False) -> dict:
    is_flat = mkey in FLAT_SURROGATE_ORDER
    if paper:
        lw = 1.0 if is_flat else 0.75
        if inset:
            lw *= PAPER_INSET_LW_SCALE
        band_alpha = 0.12 if is_flat else 0.10
    else:
        lw = 3.0 if mkey == REF_METHOD else 2.15
        band_alpha = CI_ALPHA
    return {
        "step": is_flat,
        "linewidth": lw,
        "zorder": zorder,
        "linestyle": "-" if is_flat else "--",
        "line_alpha": 1.0 if is_flat else 0.8,
        "alpha": band_alpha,
        "sync_band_alpha": not paper,
    }


def _plot_convergence_curves(
    ax: plt.Axes,
    avg_df: pd.DataFrame,
    budget: int,
    *,
    paper: bool,
    x_range: tuple[int, int] | None = None,
    show_legend: bool = True,
    inset: bool = False,
) -> None:
    x_lo, x_hi = x_range if x_range else (0, budget)
    draw_order = _curve_draw_order(avg_df, budget)
    for rank, mkey in enumerate(draw_order):
        mdf = avg_df[avg_df["method_key"] == mkey].sort_values("sim_count")
        mdf = mdf[(mdf["sim_count"] >= x_lo) & (mdf["sim_count"] <= x_hi)]
        if mdf.empty:
            continue
        x = mdf["sim_count"].to_numpy(dtype=float)
        y_mean = mdf["best_error_mean"].to_numpy(dtype=float)
        y_std = mdf["best_error_std"].fillna(0.0).to_numpy(dtype=float)
        style = _method_curve_style(mkey, paper=paper, zorder=3 + rank, inset=inset)
        draw_mean_with_band(
            ax, x, y_mean, y_std,
            color=METHOD_COLORS[mkey],
            label=METHOD_LABELS[mkey] if show_legend else None,
            **style,
        )

    if x_range is None:
        ax.axvline(x=budget, color="#B0BAC4" if paper else "#94A3B8",
                   linewidth=0.65 if paper else 1.45, linestyle=":", alpha=0.8, zorder=1)
        if not paper:
            ymax_val = ax.get_ylim()[1]
            ax.text(budget - 2, ymax_val * 0.97, budget_vline_label(budget),
                    fontsize=10, color="#64748B", va="top", ha="right")

    ax.set_xlim(x_lo, x_hi)
    if show_legend and x_range is None:
        if paper:
            _draw_paper_legend(ax)
        else:
            ax.legend(
                loc="upper right", fontsize=10, ncol=1,
                framealpha=0.97,
                handlelength=1.0, columnspacing=0.5, borderpad=0.3, labelspacing=0.25,
            )


def _ylim_for_range(avg_df: pd.DataFrame, x_lo: int, x_hi: int, *, pad: float = 0.04) -> tuple[float, float]:
    sub = avg_df[(avg_df["sim_count"] >= x_lo) & (avg_df["sim_count"] <= x_hi)]
    if sub.empty:
        return 0.0, 1.0
    y_lo = float((sub["best_error_mean"] - sub["best_error_std"].fillna(0.0)).min())
    y_hi = float((sub["best_error_mean"] + sub["best_error_std"].fillna(0.0)).max())
    span = max(y_hi - y_lo, 0.05)
    return max(0.0, y_lo - pad * span), y_hi + pad * span


def _style_inset(ax: plt.Axes) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(NullLocator())
        axis.set_minor_locator(NullLocator())
    ax.tick_params(
        axis="both", which="both",
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False, labeltop=False, labelright=False,
        length=0, width=0, pad=0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#64748B")
    ax.set_facecolor("#FFFFFF")
    ax.patch.set_alpha(0.94)


def _inset_title(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes,
        fontsize=PAPER_INSET_LABEL_FS, color="#0F172A",
        va="top", ha="left",
    )


def _inset_range_label(x_lo: int, x_hi: int) -> str:
    return f"{x_lo}-{x_hi} {PAPER_INSET_SIMS_ABBR}"


def _add_paper_insets(ax: plt.Axes, avg_df: pd.DataFrame, budget: int) -> None:
    """Segmented zoom insets (0–40 sims, 40–80 sims)."""
    zoom_specs = [
        (PAPER_INSET_ZOOM_LEFT, PAPER_INSET_ZOOM_RANGES[0]),
        (PAPER_INSET_ZOOM_RIGHT, PAPER_INSET_ZOOM_RANGES[1]),
    ]
    for bbox, x_range in zoom_specs:
        iax = ax.inset_axes(bbox)
        _plot_convergence_curves(
            iax, avg_df, budget, paper=True, x_range=x_range, show_legend=False, inset=True,
        )
        iax.set_ylim(_ylim_for_range(avg_df, *x_range))
        iax.set_xlabel("")
        iax.set_ylabel("")
        iax.set_xlim(*x_range)
        _style_inset(iax)
        _inset_title(iax, _inset_range_label(*x_range))
        _style_inset(iax)


def _shift_budget_xtick_left(ax: plt.Axes, budget: int, *, shift: float = PAPER_BUDGET_TICK_SHIFT) -> None:
    """Nudge the budget tick label inward so it is not clipped at the right edge."""
    target = str(budget)
    for label in ax.get_xticklabels():
        if label.get_text().strip() == target:
            x, y = label.get_position()
            label.set_position((x - shift, y))
            label.set_horizontalalignment("right")
            break


def _draw_convergence(ax: plt.Axes, conv_df: pd.DataFrame, budget: int, *, paper: bool = False) -> None:
    avg_df = aggregate_convergence_global(conv_df, SCENE_ORDER)
    if paper:
        _plot_convergence_curves(ax, avg_df, budget, paper=True, show_legend=True)
        _add_paper_insets(ax, avg_df, budget)
        ax.set_xlabel(XLABEL_SIMS)
        ax.set_ylabel(YLABEL_JB)
        ax.set_xlim(0, budget)
        _shift_budget_xtick_left(ax, budget)
        return

    _plot_convergence_curves(ax, avg_df, budget, paper=False, show_legend=True)
    ax.set_xlabel(XLABEL_SIMS)
    ax.set_ylabel(YLABEL_JB)


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
    df = df[df["reference"].isin(("BF-SAC-GP", "BF-SAC-RF", "BF-SAC-MLP"))]
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
    save_archive_panel(fig_b, "method_comparison_budget_bars")
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
    print("  Saved paper panels to paper/figure/")
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
