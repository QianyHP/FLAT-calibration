"""export_fig1_teaser_assets.py — data-grounded assets for Fig. 1 teaser.

Fig. 1 itself is a teaser / visual abstract. Panels A/B should be drawn as
conceptual vector elements in Illustrator/Figma/PowerPoint/LaTeX, while Panel C
should use real experiment data. This script exports only the data-grounded
elements needed for Panel C:

  outputs/figures/fig1_teaser_assets/
    data_panels/part_c_convergence.{png,pdf,svg}
    data_panels/part_c_budget_bars.{png,pdf,svg}
    data_panels/part_c_summary.csv

PNG exports are 4K-friendly (3840 px wide for the default 8 x 4.5 in canvas at
480 dpi). PDF/SVG are vector outputs for final editing.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
EXP = PROJ / "code" / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_style import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    REF_METHOD,
    SCENE_ORDER,
    apply_style,
)

RES = PROJ / "outputs" / "results"
OUT = PROJ / "outputs" / "figures" / "fig1_teaser_assets" / "data_panels"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str) -> None:
    kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(OUT / f"{stem}.png", dpi=480, **kwargs)
    fig.savefig(OUT / f"{stem}.pdf", **kwargs)
    fig.savefig(OUT / f"{stem}.svg", **kwargs)


def _six_scene_convergence() -> pd.DataFrame:
    df = pd.read_csv(RES / "comparison_convergence.csv")
    df = df[(df["method_key"].isin(METHOD_ORDER)) & (df["sim_count"] <= 100)]
    # Mean over seeds within each scene, then mean/std over scenes.
    per_scene = (
        df.groupby(["scene", "method_key", "sim_count"], as_index=False)["best_error"]
        .mean()
        .rename(columns={"best_error": "scene_mean"})
    )
    return (
        per_scene.groupby(["method_key", "sim_count"], as_index=False)["scene_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_jb", "std": "std_jb"})
    )


def _budget_summary() -> pd.DataFrame:
    df = pd.read_csv(RES / "comparison_summary.csv")
    sig = pd.read_csv(RES / "comparison_significance.csv")
    out = (
        df.groupby("method_key", as_index=False)["error_at_budget"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_jb", "std": "std_jb"})
    )
    rf_sig = sig[sig["reference"] == "BF-SAC-RF"][["baseline", "p_value", "pairs_won"]]
    rf_sig = rf_sig.rename(columns={"baseline": "method_key", "p_value": "p_vs_bfsac_rf", "pairs_won": "rf_pairs_won"})
    out = out.merge(rf_sig, on="method_key", how="left")
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["order"] = out["method_key"].map(order)
    out = out[out["method_key"].isin(METHOD_ORDER)].sort_values("order")
    out.drop(columns=["order"], inplace=True)
    out.to_csv(OUT / "part_c_summary.csv", index=False, encoding="utf-8-sig")
    return out


def plot_convergence() -> None:
    apply_style(base_font=13)
    data = _six_scene_convergence()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    for method in METHOD_ORDER:
        sub = data[data["method_key"] == method].sort_values("sim_count")
        if sub.empty:
            continue
        x = sub["sim_count"].to_numpy()
        y = sub["mean_jb"].to_numpy()
        color = METHOD_COLORS[method]
        is_bfsac = method.startswith("BF-SAC")
        ax.plot(
            x, y,
            color=color,
            linewidth=3.1 if method == REF_METHOD else (2.7 if is_bfsac else 1.7),
            alpha=1.0 if is_bfsac else 0.68,
            label=METHOD_LABELS.get(method, method),
            zorder=5 if is_bfsac else 3,
        )

    ax.axvline(100, color="#64748B", linestyle=":", linewidth=1.4)
    ax.text(98, 0.64, "budget = 100", ha="right", va="top", fontsize=11, color="#475569")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.23, 0.66)
    ax.set_xlabel("Cumulative SUMO simulations")
    ax.set_ylabel(r"Best-so-far $J_b$ (lower is better)")
    ax.set_title("Six-scene convergence", fontweight="bold", pad=8)
    ax.legend(loc="upper right", framealpha=0.96, fontsize=10)
    _save(fig, "part_c_convergence")
    plt.close(fig)


def plot_budget_bars() -> None:
    apply_style(base_font=13)
    df = _budget_summary()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(df))
    colors = [METHOD_COLORS[m] for m in df["method_key"]]
    bars = ax.bar(
        x, df["mean_jb"],
        width=0.64,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    for bar, method, val, p in zip(bars, df["method_key"], df["mean_jb"], df["p_vs_bfsac_rf"]):
        label = f"{val:.3f}"
        if pd.notna(p):
            label += "†"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold" if method.startswith("BF-SAC") else "normal",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in df["method_key"]], rotation=22, ha="right")
    ax.set_ylabel(r"$J_b$ @ 100 simulations (lower is better)")
    ax.set_title("Budget-100 calibration error", fontweight="bold", pad=8)
    ax.set_ylim(0, max(df["mean_jb"]) + 0.12)
    ax.text(
        0.01, 0.96,
        r"$^\dagger$ BF-SAC-RF significantly better ($p<0.05$, paired Wilcoxon)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#334155",
    )
    _save(fig, "part_c_budget_bars")
    plt.close(fig)


def main() -> None:
    plot_convergence()
    plot_budget_bars()
    print(f"Saved Fig.1 data panels to: {OUT}")


if __name__ == "__main__":
    main()
