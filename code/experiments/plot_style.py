"""plot_style.py — 统一 release 图表样式（英文轴标签、配色、矢量导出）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
FIG_DIR = PROJ / "outputs" / "figures"

# Method palette (color-blind friendly, consistent across figures)
METHOD_ORDER = ["BF-SAC-RF", "BF-SAC-MLP", "SPSA", "GA", "CMA-ES", "TPE"]
METHOD_COLORS = {
    "BF-SAC-RF": "#1E40AF",
    "BF-SAC-MLP": "#0D9488",
    "SPSA": "#F97316",
    "GA": "#DC2626",
    "CMA-ES": "#7C3AED",
    "TPE": "#CA8A04",
}
METHOD_LABELS = {
    "BF-SAC-RF": "BF-SAC-RF",
    "BF-SAC-MLP": "BF-SAC-MLP",
    "SPSA": "SPSA",
    "GA": "GA",
    "CMA-ES": "CMA-ES",
    "TPE": "TPE",
}

SCENE_ORDER = ["Tianjin", "Changchun", "Xian", "YTDJ", "RML", "XAM-N6"]
SCENE_TITLES = {
    "Tianjin": "Tianjin",
    "Changchun": "Changchun",
    "Xian": "Xian",
    "YTDJ": "YTDJ",
    "RML": "RML",
    "XAM-N6": "XAM-N6",
}

REF_METHOD = "BF-SAC-RF"
DPI = 300
CI_BAND_LABEL = "±1 std (5 seeds)"

# Axis / annotation strings (English for ICTAI figures)
XLABEL_SIMS = "Cumulative SUMO simulations"
YLABEL_JB = r"Best-so-far $J_b$"
YLABEL_JB_CALIB = r"Best-so-far behavioral error $J_b$"
XLABEL_NINIT = r"$N_{\mathrm{init}}$ (LHS design points)"
SCORE_PROXY_LABEL = r"Score proxy ($100 - 45 \cdot J_b$)"


def apply_style(*, base_font: float = 11.0) -> None:
    """Matplotlib rcParams for publication figures (Latin labels, no CJK dependency)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "axes.unicode_minus": False,
        "font.size": base_font,
        "axes.labelsize": base_font + 1,
        "axes.titlesize": base_font + 2,
        "legend.fontsize": base_font,
        "xtick.labelsize": base_font - 0.5,
        "ytick.labelsize": base_font - 0.5,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def budget_vline_label(budget: int) -> str:
    return f"Budget ({budget} sims)"


def save_figure(fig: plt.Figure, stem: str, *, fig_dir: Path | None = None) -> Path:
    """Save PNG + PDF + SVG (vector formats for submission)."""
    out_dir = fig_dir or FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    png = out_dir / f"{stem}.png"
    fig.savefig(png, **kwargs)
    fig.savefig(out_dir / f"{stem}.pdf", **kwargs)
    fig.savefig(out_dir / f"{stem}.svg", **kwargs)
    return png
