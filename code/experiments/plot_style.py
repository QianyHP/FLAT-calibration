"""plot_style.py — 统一 release / paper 图表样式（英文轴标签、配色、矢量导出）。

对外显示 FLAT-RF / FLAT-MLP；METHOD_ORDER / cache 键仍用历史名 BF-SAC-* 以兼容
已提交的 comparison_cache 与标定 JSON，经 METHOD_LABELS 映射到论文口径。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
FIG_DIR = PROJ / "outputs" / "figures"

# Method palette — keys match comparison_cache filenames (BF-SAC-*); labels = FLAT-*
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
    "BF-SAC-RF": "FLAT-RF",
    "BF-SAC-MLP": "FLAT-MLP",
    "SPSA": "SPSA",
    "GA": "GA",
    "CMA-ES": "CMA-ES",
    "TPE": "TPE",
}

SCENE_ORDER = ["Tianjin", "Changchun", "Xian", "YTDJ", "RML", "XAM-N6"]
# Display titles (data keys stay "Xian"/"XAM-N6"; only the rendered label changes).
SCENE_TITLES = {
    "Tianjin": "Tianjin",
    "Changchun": "Changchun",
    "Xian": "Xi'an",
    "YTDJ": "YTDJ",
    "RML": "RML",
    "XAM-N6": "XAM",
}

REF_METHOD = "BF-SAC-RF"
DPI = 300
PAPER_DPI = 600
CI_BAND_LABEL = "±1 std (5 seeds)"

# IEEE single-column panel sizes (inches), 16:9
PAPER_COL_IN = 3.35
PAPER_HALF_SIZE = (PAPER_COL_IN * 0.48, PAPER_COL_IN * 0.48 * 9 / 16)
PAPER_FULL_SIZE = (PAPER_COL_IN, PAPER_COL_IN * 9 / 16)
PAPER_FIG_DIR = PROJ / "paper" / "Figures"
# Shared axes box for 3×2 subfloat panels (calib + n_init); fixed canvas, no tight crop.
PAPER_PANEL_RECT = [0.26, 0.30, 0.69, 0.63]

# Axis / annotation strings (English for ICTAI figures)
XLABEL_SIMS = "Cumulative SUMO simulations"
YLABEL_JB = r"Best-so-far $J_b$"
YLABEL_JB_SHORT = r"$J_b$"
YLABEL_JB_CALIB = r"Best-so-far behavioral error $J_b$"
XLABEL_NINIT = r"$N_{\mathrm{init}}$"
XLABEL_SIMS_SHORT = "SUMO simulations"
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
        "axes.grid.axis": "y",
        "grid.alpha": 0.2,
        "grid.linestyle": "-",
        "grid.linewidth": 0.45,
    })


def budget_vline_label(budget: int) -> str:
    return f"Budget ({budget} sims)"


def apply_paper_style(*, base_font: float = 7.5) -> None:
    """Compact rcParams for paper/Figures sub-panels."""
    apply_style(base_font=base_font)
    plt.rcParams.update({
        "figure.dpi": PAPER_DPI,
        "savefig.dpi": PAPER_DPI,
        "axes.labelsize": base_font,
        "legend.fontsize": base_font - 0.5,
        "xtick.labelsize": base_font - 1.0,
        "ytick.labelsize": base_font - 1.0,
        # Vector-editable text
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        # Refined axes
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        # Frameless legend, compact handles
        "legend.frameon": False,
        "legend.handlelength": 0.9,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.22,
    })


def add_paper_panel_axes(fig: plt.Figure) -> plt.Axes:
    """Single half-column panel with unified margins across figure families."""
    return fig.add_axes(PAPER_PANEL_RECT)


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


def save_paper_panel(fig: plt.Figure, stem: str) -> Path:
    """Save vector PDF (+ high-DPI PNG preview) for LaTeX inclusion."""
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = dict(bbox_inches=None, pad_inches=0, facecolor="white", edgecolor="none")
    pdf = PAPER_FIG_DIR / f"{stem}.pdf"
    fig.savefig(pdf, **kwargs)
    png = PAPER_FIG_DIR / f"{stem}.png"
    fig.savefig(png, dpi=PAPER_DPI, **kwargs)
    return pdf
