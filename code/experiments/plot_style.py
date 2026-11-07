"""plot_style.py — 统一 release / paper 图表样式（英文轴标签、配色、矢量导出）。

对外显示 FLAT-RF / FLAT-MLP；METHOD_ORDER / cache 键仍用历史名 BF-SAC-* 以兼容
已提交的 comparison_cache 与标定 JSON，经 METHOD_LABELS 映射到论文口径。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[2]
FIG_DIR = PROJ / "outputs" / "figures"

# Method palette — keys match comparison_cache filenames (BF-SAC-*); labels = FLAT-*
# FLAT surrogates first (GP → RF → MLP), then baselines.
FLAT_SURROGATE_ORDER = ["BF-SAC-GP", "BF-SAC-RF", "BF-SAC-MLP"]
METHOD_ORDER = FLAT_SURROGATE_ORDER + ["SPSA", "GA", "CMA-ES", "TPE"]
METHOD_COLORS = {
    "BF-SAC-RF": "#1E40AF",
    "BF-SAC-MLP": "#0D9488",
    "BF-SAC-GP": "#6366F1",
    "SPSA": "#DC2626",      # red
    "GA": "#7C3AED",        # violet
    "CMA-ES": "#EA580C",    # orange
    "TPE": "#EAB308",       # yellow
}
METHOD_LABELS = {
    "BF-SAC-RF": "FLAT-RF",
    "BF-SAC-MLP": "FLAT-MLP",
    "BF-SAC-GP": "FLAT-GP",
    "SPSA": "SPSA",
    "GA": "GA",
    "CMA-ES": "CMA-ES",
    "TPE": "TPE",
}

SCENE_ORDER = ["Tianjin", "Changchun", "Xian", "YTDJ", "RML", "XAM-N6"]
# Appendix A/B anchor pair (easy expressway + hard intersection; matches main-text bounds).
APPENDIX_SCENES = ["XAM-N6", "Xian"]
# Appendix A1: figure annotates these k; tables/text use APPENDIX_K_REPORT.
APPENDIX_K_ANNOTATE = [3, 5, 7, 9]
APPENDIX_K_REPORT = 5
# Appendix figure styling (aligned with Fig. 7/8 paper panels).
APPENDIX_SCENE_COLORS = {
    "XAM-N6": "#1E40AF",   # blue
    "Xian": "#F97316",     # amber-orange
}
APPENDIX_SCENE_MARKERS = {
    "XAM-N6": "o",
    "Xian": "s",
}
APPENDIX_OBJ_COLORS = {
    "jb": METHOD_COLORS["BF-SAC-GP"],
    "js": METHOD_COLORS["GA"],
}
APPENDIX_OBJ_MARKERS = {
    "jb": "^",
    "js": "o",
}
# Fig. 9 top row + bottom |∇J| accents: RdBu_r hue, slightly lifted from endpoints.
CMAP_LANDSCAPE = "RdBu_r"
_LANDSCAPE_CMAP = plt.colormaps[CMAP_LANDSCAPE]
APPENDIX_GRAD_ACCENT_LEVEL = {"XAM-N6": 0.15, "Xian": 0.85}
APPENDIX_SCENE_GRAD_COLORS = {
    scene: mcolors.to_hex(_LANDSCAPE_CMAP(level))
    for scene, level in APPENDIX_GRAD_ACCENT_LEVEL.items()
}


def make_appendix_grad_cmap(scene: str) -> mcolors.LinearSegmentedColormap:
    accent = APPENDIX_SCENE_GRAD_COLORS.get(scene, "#334155")
    light = _LANDSCAPE_CMAP(0.5)  # same near-white as landscape mid-tone
    safe = scene.replace("-", "_")
    return mcolors.LinearSegmentedColormap.from_list(
        f"appendix_grad_{safe}", [light, accent], N=256,
    )

APPENDIX_A1_XTICKS = [1, 3, 5, 7, 9, 11]
APPENDIX_A1_XLIM = (0.0, 11.0)
APPENDIX_A1_MARKER_K = [1, 3, 5, 7, 9]
APPENDIX_A2_MARKER_N = [40, 80, 100, 160, 200, 280, 300]
APPENDIX_A2_XLIM = (0, 320)
APPENDIX_A2_XTICKS = [0, 40, 100, 200, 300]
APPENDIX_A2_YLIM = (-0.3, 1.2)
APPENDIX_A2_YTICKS = [0.0, 0.5, 1.0]
APPENDIX_PAPER_MARKER_SIZE = 4.0
APPENDIX_PAPER_LINEWIDTH = 1.0
APPENDIX_PAPER_BAND_ALPHA = 0.12
APPENDIX_REF_LINE = "#94A3B8"
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

# IEEE single-column panel sizes (inches) — half width matches LaTeX \plotpanelpair.
PAPER_COL_IN = 3.35
PAPER_HALF_FRAC = 0.495
PAPER_HALF_W_IN = PAPER_COL_IN * PAPER_HALF_FRAC
PAPER_FIG_DIR = PROJ / "paper" / "figure"
PAPER_ARCHIVE_DIR = PROJ / "paper" / "achieve"
PAPER_XLABEL_PAD = 2.0
PAPER_FULL_SIZE = (PAPER_COL_IN, PAPER_COL_IN * 9 / 16)

# Canvas margins (inches); axes patch below is always width:height = 16:9.
PAPER_MARGIN_LEFT_IN = {"left": 0.44, "right": 0.30}
PAPER_MARGIN_RIGHT_IN = {"left": 0.10, "right": 0.24}
PAPER_TOP_PAD_IN = 0.025
PAPER_TICK_BOTTOM_IN = 0.055
PAPER_XLABEL_BOTTOM_IN = 0.27

# 16:9 axes patch — width drives height; identical on every panel/row/column.
PAPER_PLOT_W_IN = (
    PAPER_HALF_W_IN
    - PAPER_MARGIN_LEFT_IN["left"]
    - PAPER_MARGIN_RIGHT_IN["left"]
)
PAPER_PLOT_H_IN = PAPER_PLOT_W_IN * 9 / 16


def _paper_bottom_in(row: str) -> float:
    return PAPER_XLABEL_BOTTOM_IN if row == "bottom" else PAPER_TICK_BOTTOM_IN


def paper_figure_size(row: str = "bottom") -> tuple[float, float]:
    """Row-aware canvas height; 16:9 plot patch size is row-independent."""
    return (PAPER_HALF_W_IN, PAPER_TOP_PAD_IN + PAPER_PLOT_H_IN + _paper_bottom_in(row))


def paper_panel_rect_inches(*, col: str = "left", row: str = "bottom") -> tuple[float, float, float, float]:
    """Return axes (left, bottom, width, height) in inches — width:height = 16:9."""
    return (
        PAPER_MARGIN_LEFT_IN["left" if col == "left" else "right"],
        _paper_bottom_in(row),
        PAPER_PLOT_W_IN,
        PAPER_PLOT_H_IN,
    )


# Default half-panel size (bottom row, with x-label band).
PAPER_HALF_SIZE = paper_figure_size("bottom")

# 3×2 mosaic roles for Fig. 7/8 (left col y-label; bottom row x-label only).
PAPER_3X2_LAYOUT: list[tuple[str, str, str, bool]] = [
    ("Tianjin", "left", "top", True),
    ("YTDJ", "right", "top", False),
    ("Changchun", "left", "mid", False),
    ("RML", "right", "mid", False),
    ("Xian", "left", "bottom", False),
    ("XAM-N6", "right", "bottom", False),
]

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
    """Compact rcParams for paper/figure sub-panels."""
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


def paper_panel_rect(*, col: str = "left", row: str = "bottom") -> list[float]:
    """Normalized [left, bottom, width, height] for fig.add_axes; axes patch is 16:9."""
    w_fig, h_fig = paper_figure_size(row)
    left_in, bot_in, plot_w_in, plot_h_in = paper_panel_rect_inches(col=col, row=row)
    return [left_in / w_fig, bot_in / h_fig, plot_w_in / w_fig, plot_h_in / h_fig]


# Legacy alias; prefer paper_panel_rect().
PAPER_PANEL_RECT = paper_panel_rect(col="left", row="bottom")


def add_paper_panel_axes(fig: plt.Figure, *, col: str = "left", row: str = "bottom") -> plt.Axes:
    """Half-column panel with column/row-aware margins for 3×2 mosaics."""
    return fig.add_axes(paper_panel_rect(col=col, row=row))


def apply_panel_axis_labels(
    ax: plt.Axes,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    ylabel: str = "",
    xlabel: str = "",
) -> None:
    """Shared mosaic labelling: y-name on left column only; x-name on bottom row only."""
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    if show_xlabel and xlabel:
        ax.set_xlabel(xlabel, labelpad=PAPER_XLABEL_PAD)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)


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
    return _save_panel(fig, stem, PAPER_FIG_DIR)


def save_archive_panel(fig: plt.Figure, stem: str) -> Path:
    """Save panel exports not included in root.tex (paper/achieve/)."""
    return _save_panel(fig, stem, PAPER_ARCHIVE_DIR)


def _save_panel(fig: plt.Figure, stem: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(bbox_inches=None, pad_inches=0, facecolor="white", edgecolor="none")
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(pdf, **kwargs)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=PAPER_DPI, **kwargs)
    return pdf
