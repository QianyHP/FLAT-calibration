#!/usr/bin/env python3
"""k-sensitivity of response-standardized p95 local Lipschitz proxies on 10-D LHS landscapes (Appendix A1b).

Each panel: two curves — p95 L̂_Jb and p95 L̂_Js vs k (same palette as A2).
Ratio reported in text only (@ k=5); not plotted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

PROJ = Path(__file__).resolve().parents[2]
EXP = PROJ / "code" / "experiments"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(PROJ / "code" / "calibration"))
from plot_style import (  # noqa: E402
    APPENDIX_A1_MARKER_K,
    APPENDIX_A1_XLIM,
    APPENDIX_A1_XTICKS,
    APPENDIX_K_REPORT,
    APPENDIX_OBJ_MARKERS,
    APPENDIX_PAPER_LINEWIDTH,
    APPENDIX_PAPER_MARKER_SIZE,
    APPENDIX_REF_LINE,
    APPENDIX_SCENES,
    paper_figure_size,
    PAPER_HALF_SIZE,
    add_paper_panel_axes,
    apply_panel_axis_labels,
    apply_paper_style,
    apply_style,
    save_figure,
    save_paper_panel,
    PAPER_FIG_DIR,
)
from unified_calibration import PARAM_BOUNDS, PARAM_NAMES  # noqa: E402
from analyze_landscape_lipschitz import standardize_response  # noqa: E402

FAIL_VALUE = 10.0
K_GRID = list(range(1, 11))
PERCENTILE = 95
RES = PROJ / "outputs" / "results"
_OBJ_COLORS = {"js": "#DC2626", "jb": "#6366F1"}
_OBJ_ORDER = (("js", r"$\hat L_{J_s}$"), ("jb", r"$\hat L_{J_b}$"))


def _scene_stem(scene: str) -> str:
    return "landscape_lipschitz_xam" if scene == "XAM-N6" else "landscape_lipschitz_xian"


def build_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lo, hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    X, jb, js = [], [], []
    for item in data["results"]:
        if float(item["J_b"]) >= FAIL_VALUE or float(item["J_s"]) >= FAIL_VALUE:
            continue
        vec = np.array([item["params"][n] for n in PARAM_NAMES], dtype=float)
        X.append((vec - lo) / (hi - lo))
        jb.append(float(item["J_b"]))
        js.append(float(item["J_s"]))
    return np.vstack(X), np.array(jb), np.array(js)


def local_slopes(theta_n: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    n = len(values)
    tree = cKDTree(theta_n)
    dists, idx = tree.query(theta_n, k=min(k + 1, n))
    point_slopes = []
    for i in range(n):
        slopes_i = []
        for d, j in zip(dists[i], idx[i]):
            if j == i or d <= 1e-12:
                continue
            slopes_i.append(abs(values[i] - values[j]) / d)
            if len(slopes_i) >= k:
                break
        if slopes_i:
            point_slopes.append(max(slopes_i))
    return np.array(point_slopes)


def p95_at_k(theta_n: np.ndarray, values: np.ndarray, k: int) -> float:
    slopes = local_slopes(theta_n, values, k)
    if len(slopes) == 0:
        return float("nan")
    return float(np.percentile(slopes, PERCENTILE))


def compute_all() -> dict:
    out: dict = {
        "k_grid": K_GRID,
        "percentile": PERCENTILE,
        "report_k": APPENDIX_K_REPORT,
        "response_scale": "per-objective population standard deviation on the valid LHS design",
        "scenes": {},
    }
    for scene in APPENDIX_SCENES:
        path = RES / f"landscape_full10d_{scene}.json"
        theta_n, jb, js = build_dataset(path)
        jb_scaled, sd_b = standardize_response(jb)
        js_scaled, sd_s = standardize_response(js)
        ljb = [p95_at_k(theta_n, jb_scaled, k) for k in K_GRID]
        ljs = [p95_at_k(theta_n, js_scaled, k) for k in K_GRID]
        out["scenes"][scene] = {
            "response_std": {"J_b": sd_b, "J_s": sd_s},
            "p95_L_Jb": ljb,
            "p95_L_Js": ljs,
            "ratio": [a / b if b > 0 else float("inf") for a, b in zip(ljs, ljb)],
            "n_valid": int(len(theta_n)),
        }
    return out


def _draw_obj_curve(ax, ks: np.ndarray, vals: np.ndarray, obj: str, *, z_mark: int) -> None:
    color = _OBJ_COLORS[obj]
    marker = APPENDIX_OBJ_MARKERS[obj]
    mark_mask = np.isin(ks, APPENDIX_A1_MARKER_K)
    ax.plot(ks, vals, "-", color=color, lw=APPENDIX_PAPER_LINEWIDTH, zorder=3, alpha=0.95,
            label=_OBJ_ORDER[0][1] if obj == "js" else _OBJ_ORDER[1][1])
    ax.plot(
        ks[mark_mask], vals[mark_mask],
        linestyle="None", marker=marker, color=color,
        markersize=APPENDIX_PAPER_MARKER_SIZE, zorder=z_mark,
        markeredgewidth=0.75, markeredgecolor="white",
    )


def draw_scene_panel(
    ax: plt.Axes,
    data: dict,
    scene: str,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    show_legend: bool,
) -> None:
    ks = np.array(K_GRID)
    block = data["scenes"][scene]
    ljs = np.array(block["p95_L_Js"], dtype=float)
    ljb = np.array(block["p95_L_Jb"], dtype=float)

    _draw_obj_curve(ax, ks, ljs, "js", z_mark=5)
    _draw_obj_curve(ax, ks, ljb, "jb", z_mark=6)

    ax.axvline(APPENDIX_K_REPORT, color=APPENDIX_REF_LINE, ls=":", lw=0.6, alpha=0.7, zorder=2)
    ax.set_xticks(APPENDIX_A1_XTICKS)
    ax.set_xlim(*APPENDIX_A1_XLIM)
    yhi = float(np.nanmax(np.concatenate([ljs, ljb]))) * 1.12 + 0.15
    ax.set_ylim(0.0, yhi)
    apply_panel_axis_labels(
        ax,
        show_ylabel=show_ylabel,
        show_xlabel=show_xlabel,
        ylabel=r"$\hat L$",
        xlabel=r"Neighborhood size $k$",
    )
    if show_legend:
        ax.legend(loc="lower right", fontsize=4.2, handlelength=0.9,
                  columnspacing=0.4, borderpad=0.25, labelspacing=0.22)


def _export_paper_panels(data: dict) -> None:
    apply_paper_style(base_font=6.0)
    for scene, col, show_legend in zip(APPENDIX_SCENES, ("left", "right"), (True, False)):
        fig = plt.figure(figsize=paper_figure_size("bottom"), facecolor="#FFFFFF")
        ax = add_paper_panel_axes(fig, col=col, row="bottom")
        draw_scene_panel(
            ax, data, scene,
            show_ylabel=(col == "left"),
            show_xlabel=True,
            show_legend=show_legend,
        )
        save_paper_panel(fig, _scene_stem(scene))
        plt.close(fig)
    apply_style()


def plot_combined(data: dict) -> plt.Figure:
    apply_paper_style(base_font=6.0)
    w, h = PAPER_HALF_SIZE
    fig, axes = plt.subplots(1, 2, figsize=(w * 2.05, h), facecolor="#FFFFFF")
    for ax, scene, show_legend in zip(axes, APPENDIX_SCENES, (True, False)):
        draw_scene_panel(
            ax, data, scene,
            show_ylabel=True,
            show_xlabel=True,
            show_legend=show_legend,
        )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.20, top=0.88, wspace=0.28)
    return fig


def main() -> None:
    data = compute_all()
    json_path = RES / "landscape_lipschitz_k_sensitivity.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    _export_paper_panels(data)
    fig = plot_combined(data)
    out = PROJ / "outputs" / "figures" / "landscape_lipschitz_k_sensitivity"
    save_figure(fig, out)
    plt.close(fig)
    print(f"Saved: {json_path}")
    print(f"Saved: {out}.png / .pdf / .svg")


if __name__ == "__main__":
    main()
