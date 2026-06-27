"""plot_n_init_sweep.py — N_init 样本效率消融图

读取 comparison_summary_sweep.csv 或 sweep 缓存重建结果。
输出: outputs/figures/n_init_sample_efficiency.png
"""
from __future__ import annotations

import json
import re
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

from unified_calibration import BUDGET_SUMO, N_INIT_MAIN, SWEEP_N_INIT  # noqa: E402
from plot_convergence_utils import draw_mean_with_band  # noqa: E402
from plot_style import (  # noqa: E402
    SCENE_ORDER, METHOD_COLORS, METHOD_LABELS, apply_style, apply_paper_style, save_figure,
    save_paper_panel, add_paper_panel_axes, PAPER_HALF_SIZE, XLABEL_NINIT, YLABEL_JB_SHORT,
    CI_BAND_LABEL,
)

apply_style()
RES = PROJ / "outputs" / "results"
CACHE = RES / "comparison_cache"
_CACHE_TAIL_RE = re.compile(r"_b(\d+)(?:_s(\d+))?\.json$")

# SIND left column / UTE right column (same order as Fig. 1).
# (scene, show_ylabel, show_xlabel, show_legend) — every panel shows the x-axis label.
PAPER_SCENE_LAYOUT = [
    ("Tianjin", True, True, True),
    ("YTDJ", True, True, False),
    ("Changchun", True, True, False),
    ("RML", True, True, False),
    ("Xian", True, True, False),
    ("XAM-N6", True, True, False),
]


def _surrogate_key(method_key: str) -> str | None:
    if method_key.startswith("BF-SAC-RF"):
        return "BF-SAC-RF"
    if method_key.startswith("BF-SAC-MLP"):
        return "BF-SAC-MLP"
    return None


def _prepare_sweep_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        surr = _surrogate_key(str(r["method_key"]))
        if surr is None:
            continue
        n_init = int(r["n_init"])
        if n_init not in SWEEP_N_INIT:
            continue
        rows.append({
            "scene": r["scene"], "surr": surr, "n_init": n_init,
            "error_at_budget": float(r["error_at_budget"]),
        })
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["scene", "surr", "n_init"], as_index=False)
        .agg(error_mean=("error_at_budget", "mean"),
             error_std=("error_at_budget", "std"),
             n_runs=("error_at_budget", "count"))
        .sort_values(["scene", "surr", "n_init"])
    )


def _load_sweep_df() -> pd.DataFrame:
    p = RES / "comparison_summary_sweep.csv"
    if p.exists():
        return pd.read_csv(p)
    rows = []
    for fp in CACHE.glob(f"*_b{BUDGET_SUMO}*.json"):
        d = json.loads(fp.read_text(encoding="utf-8"))
        mkey = d.get("method_key", "")
        if "-n" not in mkey and mkey not in ("BF-SAC-RF", "BF-SAC-MLP"):
            continue
        n_init = N_INIT_MAIN if mkey in ("BF-SAC-RF", "BF-SAC-MLP") else int(mkey.split("-n")[-1])
        if n_init not in SWEEP_N_INIT:
            continue
        m = _CACHE_TAIL_RE.search(fp.name)
        seed = int(m.group(2)) if m and m.group(2) else 42
        base = fp.name[: m.start()] if m else fp.stem
        scene = base[: -(len(mkey) + 1)]
        rows.append({
            "scene": scene, "method_key": mkey, "n_init": n_init,
            "error_at_budget": d["error_at_budget"], "run_seed": seed,
        })
    return pd.DataFrame(rows)


def _draw_scene_sweep(
    ax: plt.Axes,
    agg: pd.DataFrame,
    scene: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    show_legend: bool,
    paper: bool = False,
) -> None:
    sub = agg[agg["scene"] == scene]
    ms = 4.0 if paper else 7
    lw = 1.0 if paper else 2.2
    band_alpha = 0.10 if paper else 0.22
    for surr, marker in (("BF-SAC-RF", "o"), ("BF-SAC-MLP", "s")):
        ssub = sub[sub["surr"] == surr].sort_values("n_init")
        if ssub.empty:
            continue
        x = ssub["n_init"].to_numpy(dtype=float)
        y = ssub["error_mean"].to_numpy(dtype=float)
        y_std = ssub["error_std"].fillna(0.0).to_numpy(dtype=float)
        color = METHOD_COLORS[surr]
        draw_mean_with_band(ax, x, y, y_std, color=color, step=False,
                            linewidth=lw, zorder=4, label=METHOD_LABELS[surr], alpha=band_alpha)
        ax.plot(x, y, linestyle="None", marker=marker, color=color,
                markersize=ms, zorder=5, markeredgewidth=0.75, markeredgecolor="white")
    ax.axvline(N_INIT_MAIN, color="#94A3B8", linestyle="--", alpha=0.7, linewidth=0.6)
    ax.set_ylabel(YLABEL_JB_SHORT)
    if show_xlabel:
        ax.set_xlabel(XLABEL_NINIT)
    else:
        ax.set_xlabel("")
        if paper:
            ax.tick_params(labelbottom=False)
    ax.set_xticks(list(SWEEP_N_INIT))
    ax.set_xlim(15, 105)
    if show_legend:
        ax.legend(
            fontsize=4.6 if paper else 8, loc="upper left",
            handlelength=0.9, columnspacing=0.4, borderpad=0.25, labelspacing=0.22,
        )


def _scene_stem(scene: str) -> str:
    if scene == "XAM-N6":
        return "n_init_xam"
    return f"n_init_{scene.lower().replace('-', '_')}"


def _export_paper_panels(agg: pd.DataFrame) -> None:
    apply_paper_style(base_font=6.0)
    for scene, show_ylabel, show_xlabel, show_legend in PAPER_SCENE_LAYOUT:
        if scene not in agg["scene"].unique():
            continue
        fig = plt.figure(figsize=PAPER_HALF_SIZE, facecolor="#FFFFFF")
        ax = add_paper_panel_axes(fig)
        _draw_scene_sweep(
            ax, agg, scene,
            show_ylabel=show_ylabel, show_xlabel=show_xlabel,
            show_legend=show_legend, paper=True,
        )
        save_paper_panel(fig, _scene_stem(scene))
        plt.close(fig)
    apply_style()


def plot_sweep(df: pd.DataFrame) -> Path:
    agg = _prepare_sweep_df(df)
    scenes = [s for s in SCENE_ORDER if s in agg["scene"].unique()]
    if not scenes:
        raise FileNotFoundError("No sweep data — run: run_comparison.py --mode sweep")

    _export_paper_panels(agg)

    ncols, nrows = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, scene in zip(axes, scenes):
        _draw_scene_sweep(
            ax, agg, scene,
            show_ylabel=True, show_xlabel=True, show_legend=False,
        )

    for ax in axes[len(scenes):]:
        ax.set_visible(False)

    band_patch = mpatches.Patch(facecolor="#94A3B8", alpha=0.22, edgecolor="none", label=CI_BAND_LABEL)
    handles, _ = axes[0].get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-RF"], marker="o", linestyle="-", label="FLAT-RF"),
        plt.Line2D([0], [0], color=METHOD_COLORS["BF-SAC-MLP"], marker="s", linestyle="-", label="FLAT-MLP"),
        band_patch,
    ])
    fig.legend(handles=handles, loc="upper center", ncol=3, framealpha=0.97, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = save_figure(fig, "n_init_sample_efficiency")
    plt.close(fig)
    print("  Saved paper panels to paper/Figures/")
    print(f"  Saved: {out}")
    return out


def main() -> None:
    print("=" * 60)
    print("N_init sample-efficiency ablation")
    plot_sweep(_load_sweep_df())


if __name__ == "__main__":
    main()
