"""plot_speed_distribution_violin.py — Real vs FLAT-RF calibrated speed distributions.

Compares observed speeds with SUMO speeds under FLAT-RF calibrated parameters.
Violin fill uses RdBu_r (same as Fig. 1/2): blue = low speed, red = high speed.

Outputs:
  outputs/figures/speed_distribution_violin.{png,pdf,svg}
  paper/Figures/speed_distribution_violin.{pdf,png}

Usage:
  python code/experiments/plot_speed_distribution_violin.py
  python code/experiments/plot_speed_distribution_violin.py --refresh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, ks_2samp

PROJ = Path(__file__).resolve().parents[2]
CAL_ROOT = PROJ / "code" / "calibration"
EXP_ROOT = PROJ / "code" / "experiments"
for p in (CAL_ROOT, EXP_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unified_calibration import SCENARIOS, feature_error, get_feature_weights  # noqa: E402
from plot_style import (  # noqa: E402
    SCENE_ORDER, SCENE_TITLES, METHOD_COLORS, apply_paper_style,
    save_figure, PAPER_FIG_DIR, PAPER_FULL_SIZE,
)

CACHE = PROJ / "outputs" / "results" / "speed_distribution_violin.json"
CAL_DIR = PROJ / "data" / "processed_data" / "calibration"
SIM_SEED = 42
REAL_SUBSAMPLE_N = 8000
REAL_SUBSAMPLE_SEED = 42

# Match Fig. 1 / Fig. 2 landscape & LCB contour (fill only; no colorbar)
CMAP_NAME = "RdBu_r"
EDGE_OBS = "#334155"          # slate-700, neutral
EDGE_SIM = METHOD_COLORS["BF-SAC-RF"]  # FLAT-RF blue
STAR_COLOR = "#EA580C"        # same accent as calibrated θ* in Fig. 2
Y_MAX = 18.0                  # shared y-axis cap (m/s)
LINE_LW = 0.7                 # outline + v15/v85, unified weight
EDGE_ALPHA = 0.9              # Observed / FLAT-RF outline
QUANTILE_COLOR = "#64748B"    # slate-500, v15/v85 ticks inside violin
QUANTILE_ALPHA = 0.85
# Per-scene similarity label position (x in gap right of violin, y in m/s)
SCENE_SIM_LABEL_Y: dict[str, float] = {
    "Tianjin": 9.5,
    "Changchun": 13.5,
    "Xian": 14.0,
    "YTDJ": 14.0,
    "RML": 11.5,
    "XAM-N6": 15.5,
}
SIM_LABEL_FS = 4.4
SIM_LABEL_COLOR = "#475569"
SIM_LABEL_X_OFFSET = -0.12   # shift ρ labels slightly left of inter-scene gap


def _sumo_binary() -> str:
    home = os.environ.get("SUMO_HOME", r"D:\RZ\SUMO")
    for name in ("sumo.exe", "sumo"):
        p = Path(home) / "bin" / name
        if p.exists():
            return str(p)
    return "sumo"


def load_calibrated_params(scene: str) -> dict:
    p = CAL_DIR / f"{scene}_calibration.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing calibration: {p}")
    return json.loads(p.read_text(encoding="utf-8"))["calibrated_params"]


def collect_real_speeds(scene: str) -> np.ndarray:
    sc = SCENARIOS[scene]
    if sc["type"] == "SIND":
        df = pd.read_csv(sc["data"] / "veh_tracks.csv")
        speed = np.sqrt(df["vx"].values ** 2 + df["vy"].values ** 2)
    else:
        df = pd.read_csv(sc["data"] / "frenet.csv")
        cols = {c.lower(): c for c in df.columns}
        speed_col = cols.get("speed(km/h)", cols.get("velocity(km/h)"))
        speed = df[speed_col].values / 3.6
    speed = speed[np.isfinite(speed)]
    return speed[(speed >= 0.0) & (speed <= 40.0)]


def collect_sim_speeds(scene: str, params: dict, *, seed: int = SIM_SEED) -> np.ndarray:
    import traci

    sc = SCENARIOS[scene]
    cfg = str(sc["data"] / "sumo" / f"{sc['cfg']}.sumocfg")
    sumo = _sumo_binary()

    traci.start([sumo, "-c", cfg,
                 "--no-step-log", "true",
                 "--no-warnings", "true",
                 "--seed", str(seed)])
    speeds: list[float] = []
    try:
        for vtype in traci.vehicletype.getIDList():
            traci.vehicletype.setAccel(vtype, float(params["accel"]))
            traci.vehicletype.setDecel(vtype, float(params["decel"]))
            traci.vehicletype.setImperfection(vtype, float(params["sigma"]))
            traci.vehicletype.setTau(vtype, float(params["tau"]))
            traci.vehicletype.setMinGap(vtype, float(params["minGap"]))
            traci.vehicletype.setSpeedFactor(vtype, float(params["speedFactor"]))
            for lc_key in ("lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain"):
                traci.vehicletype.setParameter(vtype, lc_key, str(float(params[lc_key])))

        warmup, sim_end = sc["warmup"], sc["sim_end"]
        step = 0
        while traci.simulation.getTime() < sim_end:
            traci.simulationStep()
            step += 1
            if traci.simulation.getTime() < warmup or step % 3 != 0:
                continue
            for vid in traci.vehicle.getIDList():
                speeds.append(traci.vehicle.getSpeed(vid))
    finally:
        traci.close()

    arr = np.asarray(speeds, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= 0.0) & (arr <= 40.0)]


def _subsample_real(speeds: np.ndarray, scene: str, *, n: int = REAL_SUBSAMPLE_N) -> np.ndarray:
    """Fixed-seed subsample for KDE plotting (full data kept for stats)."""
    if speeds.size <= n:
        return speeds
    seed = REAL_SUBSAMPLE_SEED + sum(ord(c) for c in scene)
    rng = np.random.default_rng(seed)
    idx = rng.choice(speeds.size, size=n, replace=False)
    return speeds[idx]


def _speed_stats(speeds: np.ndarray) -> dict[str, float]:
    return {
        "v_mean": float(np.mean(speeds)),
        "v_p15": float(np.percentile(speeds, 15)),
        "v_p85": float(np.percentile(speeds, 85)),
    }


def distribution_similarity(real: np.ndarray, sim: np.ndarray) -> float:
    """1 − KS distance on full speed samples (higher = more similar)."""
    if real.size < 20 or sim.size < 20:
        return float("nan")
    ks = ks_2samp(real, sim, method="auto").statistic
    return float(max(0.0, 1.0 - ks))


def speed_fingerprint_similarity(real: np.ndarray, sim: np.ndarray, scene: str) -> float:
    """1 − speed-only weighted relative fingerprint error (full data)."""
    real_feat = np.array([np.mean(real), np.percentile(real, 15), np.percentile(real, 85)])
    sim_feat = np.array([np.mean(sim), np.percentile(sim, 15), np.percentile(sim, 85)])
    w = get_feature_weights(scene)[:3]
    w = w / w.sum()
    return float(max(0.0, 1.0 - feature_error(real_feat, sim_feat, w)))


def load_or_build_cache(*, refresh: bool) -> dict:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    data: dict = {"sim_seed": SIM_SEED, "scenes": {}}
    for scene in SCENE_ORDER:
        print(f"  [{scene}] real speeds …")
        real = collect_real_speeds(scene)
        params = load_calibrated_params(scene)
        print(f"  [{scene}] SUMO (FLAT-RF calibrated) …")
        sim = collect_sim_speeds(scene, params, seed=SIM_SEED)
        data["scenes"][scene] = {
            "real": real.tolist(),
            "sim": sim.tolist(),
            "real_stats": _speed_stats(real),
            "sim_stats": _speed_stats(sim),
            "distribution_similarity": distribution_similarity(real, sim),
            "speed_fingerprint_similarity": speed_fingerprint_similarity(real, sim, scene),
            "n_real": int(len(real)),
            "n_sim": int(len(sim)),
        }
        sim_s = data["scenes"][scene]["distribution_similarity"]
        print(f"       n_real={len(real):,}  n_sim={len(sim):,}  "
              f"sim={sim_s:.2f}  "
              f"v_mean {data['scenes'][scene]['real_stats']['v_mean']:.2f} → "
              f"{data['scenes'][scene]['sim_stats']['v_mean']:.2f} m/s")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Cached → {CACHE}")
    return data


def _kde_half_width(speeds: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Normalised KDE half-width in [0, 1] over y_grid."""
    if speeds.size < 20:
        return np.zeros_like(y_grid)
    spd = speeds[(speeds >= 0) & (speeds <= Y_MAX)]
    if spd.size < 20:
        return np.zeros_like(y_grid)
    kde = gaussian_kde(spd, bw_method=0.22)
    dens = kde(y_grid)
    mx = float(dens.max()) if dens.size else 1.0
    return dens / max(mx, 1e-9)


def _draw_gradient_violin(
    ax: plt.Axes,
    speeds: np.ndarray,
    xc: float,
    *,
    norm: Normalize,
    cmap,
    edge_color: str,
    width: float,
    side: str,
    scene: str,
    subsample_real: bool,
) -> None:
    """Half-violin with RdBu_r fill keyed to speed (y-axis)."""
    if speeds.size < 20:
        return

    # Stats (★, v15, v85) always from full data — matches calibration fingerprint.
    # KDE only subsamples real trajectories for stable density curves.
    stats_speeds = speeds
    kde_speeds = _subsample_real(speeds, scene) if subsample_real else speeds

    y_grid = np.linspace(0.0, Y_MAX, 120)
    hw = _kde_half_width(kde_speeds, y_grid) * width

    sign = -1.0 if side == "left" else 1.0
    for i in range(len(y_grid) - 1):
        y0, y1 = y_grid[i], y_grid[i + 1]
        w0, w1 = hw[i], hw[i + 1]
        if w0 < 1e-4 and w1 < 1e-4:
            continue
        y_mid = 0.5 * (y0 + y1)
        face = cmap(norm(y_mid))
        xs = [xc, xc + sign * w0, xc + sign * w1, xc]
        ys = [y0, y0, y1, y1]
        ax.fill(xs, ys, facecolor=face, edgecolor="none", alpha=0.92, zorder=2)

    # Outline — same weight & alpha as percentile ticks
    mask = hw > 0.02
    if mask.any():
        ax.plot(xc + sign * hw[mask], y_grid[mask],
                color=edge_color, lw=LINE_LW, ls="-", alpha=EDGE_ALPHA,
                solid_capstyle="round", zorder=4)

    # Fingerprint speed markers: p15, p85 (gray ticks); mean star
    stats = _speed_stats(stats_speeds)
    p15, mean, p85 = stats["v_p15"], stats["v_mean"], stats["v_p85"]
    for val in (p15, p85):
        if 0 <= val <= Y_MAX:
            idx = int(np.argmin(np.abs(y_grid - val)))
            tick_w = hw[idx] * sign
            ax.hlines(val, xc, xc + tick_w, colors=QUANTILE_COLOR,
                      linewidth=LINE_LW, alpha=QUANTILE_ALPHA, zorder=5)
    if 0 <= mean <= Y_MAX:
        ax.scatter([xc + sign * hw[int(np.argmin(np.abs(y_grid - mean)))] * 0.55],
                   [mean], marker="*", s=18, color=STAR_COLOR,
                   edgecolors="white", linewidths=0.45, zorder=6)


def _similarity_label_xy(i: int, scene: str, *, gap: float, width: float, n_scenes: int) -> tuple[float, float]:
    xc = i * gap
    if i < n_scenes - 1:
        x = xc + gap / 2 + SIM_LABEL_X_OFFSET
    else:
        x = xc + width + 0.22 + SIM_LABEL_X_OFFSET
    return x, SCENE_SIM_LABEL_Y[scene]


def plot_violin(data: dict) -> plt.Figure:
    apply_paper_style(base_font=5.8)
    fig, ax = plt.subplots(figsize=PAPER_FULL_SIZE, facecolor="white")

    cmap = plt.colormaps[CMAP_NAME]
    norm = Normalize(vmin=0.0, vmax=Y_MAX)

    gap = 1.28
    width = 0.36
    positions: list[float] = []

    for i, scene in enumerate(SCENE_ORDER):
        sc = data["scenes"][scene]
        real = np.asarray(sc["real"], dtype=float)
        sim = np.asarray(sc["sim"], dtype=float)
        xc = i * gap
        positions.append(xc)

        _draw_gradient_violin(
            ax, real, xc, norm=norm, cmap=cmap,
            edge_color=EDGE_OBS, width=width, side="left",
            scene=scene, subsample_real=True,
        )
        _draw_gradient_violin(
            ax, sim, xc, norm=norm, cmap=cmap,
            edge_color=EDGE_SIM, width=width, side="right",
            scene=scene, subsample_real=False,
        )

        sim_score = sc.get("distribution_similarity")
        if sim_score is None:
            sim_score = distribution_similarity(real, sim)
        lx, ly = _similarity_label_xy(
            i, scene, gap=gap, width=width, n_scenes=len(SCENE_ORDER),
        )
        ax.text(
            lx, ly, rf"$\rho = {sim_score:.2f}$",
            ha="center", va="center", fontsize=SIM_LABEL_FS, color=SIM_LABEL_COLOR,
            zorder=7,
        )

        # Subtle pair bracket under scene label
        ax.plot([xc - width * 0.55, xc + width * 0.55], [-0.55, -0.55],
                color="#CBD5E1", lw=0.5, clip_on=False, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels([SCENE_TITLES[s] for s in SCENE_ORDER], fontsize=5.0)
    ax.set_xlim(-0.62, (len(SCENE_ORDER) - 1) * gap + 0.62)
    ax.set_ylim(0, Y_MAX)
    ax.set_ylabel("Speed (m/s)")
    ax.set_xlabel("Scene")
    ax.set_yticks(np.arange(0, Y_MAX + 0.1, 2.5))

    handles = [
        Line2D([0], [0], color=EDGE_OBS, lw=LINE_LW, alpha=EDGE_ALPHA, label="Observed"),
        Line2D([0], [0], color=EDGE_SIM, lw=LINE_LW, alpha=EDGE_ALPHA, label="Simulated"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=STAR_COLOR,
               markeredgecolor="white", markeredgewidth=0.4, markersize=5,
               label=r"$v_{\mathrm{mean}}$"),
        Line2D([0], [0], color=QUANTILE_COLOR, lw=LINE_LW, alpha=QUANTILE_ALPHA,
               label=r"$v_{15},\,v_{85}$"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=4.8,
              handlelength=1.0, borderpad=0.25, labelspacing=0.22)

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.97)
    return fig


def save_paper(fig: plt.Figure, stem: str) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="none")
    fig.savefig(PAPER_FIG_DIR / f"{stem}.pdf", **kw)
    fig.savefig(PAPER_FIG_DIR / f"{stem}.png", dpi=600, **kw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-run SUMO and rebuild cache")
    args = parser.parse_args()

    print("=" * 60)
    print("Speed distribution violin (RdBu_r, paper style)")
    data = load_or_build_cache(refresh=args.refresh)

    fig = plot_violin(data)
    out = save_figure(fig, "speed_distribution_violin")
    save_paper(fig, "speed_distribution_violin")
    plt.close(fig)
    print(f"  Saved → {out}")
    print(f"  Saved → {PAPER_FIG_DIR / 'speed_distribution_violin.pdf'}")


if __name__ == "__main__":
    main()
