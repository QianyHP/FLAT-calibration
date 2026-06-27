#!/usr/bin/env python3
"""Plot YTDJ observed vs simulated space–time trajectories (lane 7 zoom).

Outputs:
  outputs/figures/trajectory_compare.{png,pdf}

Inputs:
  data/processed_data/calibration/YTDJ_calibration_with_history.json
  data/.../YTDJ/frenet.csv
  SUMO simulation (requires SUMO_HOME)
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "calibration"))

from unified_calibration import SCENARIOS  # noqa: E402

OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CMAP = "RdBu_r"
NORM = Normalize(vmin=0.0, vmax=1.0, clip=True)
FIG_16_9 = (8.0, 4.5)

ZOOM = {"t_min": 250.0, "t_max": 320.0, "s_min": 169.0, "s_max": 224.0, "lane_id": 7}

Traj = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class TrajSceneConfig:
    scene: str
    lane_id: int
    lane_maps: tuple[tuple[str, dict[int, int]], ...]
    edge_phys: dict[str, tuple[float, float]]
    speed_col: str


YTDJ = TrajSceneConfig(
    scene="YTDJ",
    lane_id=7,
    lane_maps=(
        ("we_main_in", {0: 6, 1: 7, 2: 8}),
        ("we_parallel", {2: 6, 3: 7, 4: 8}),
        ("we_main_out", {0: 6, 1: 7, 2: 8}),
    ),
    edge_phys={
        "we_main_in": (0.0, 80.0),
        "we_ramp_in": (0.0, 80.0),
        "we_parallel": (80.0, 202.0),
        "we_main_out": (202.0, 362.0),
    },
    speed_col="velocity(km/h)",
)


def _sumo_binary() -> str:
    home = os.environ.get("SUMO_HOME", r"D:\RZ\SUMO")
    for name in ("sumo.exe", "sumo"):
        p = Path(home) / "bin" / name
        if p.exists():
            return str(p)
    return "sumo"


def load_calibrated_params(scene: str) -> dict:
    p = ROOT / f"data/processed_data/calibration/{scene}_calibration_with_history.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["calibrated_params"]


def _sumo_edge_lengths(scene: str) -> dict[str, float]:
    sc = SCENARIOS[scene]
    net = sc["data"] / "sumo" / f"{sc['cfg']}.net.xml"
    lengths: dict[str, list[float]] = {}
    for lane in ET.parse(net).getroot().findall(".//lane"):
        lid = lane.get("id", "")
        if lid.startswith(":"):
            continue
        edge = lid.rsplit("_", 1)[0] if "_" in lid else lid
        lengths.setdefault(edge, []).append(float(lane.get("length", 0)))
    return {e: float(np.mean(v)) for e, v in lengths.items()}


def _sumo_lane_to_frenet(lane_id: str, cfg: TrajSceneConfig) -> int | None:
    if lane_id.startswith(":"):
        return None
    for prefix, mapping in cfg.lane_maps:
        if lane_id.startswith(prefix + "_"):
            return mapping.get(int(lane_id.rsplit("_", 1)[-1]))
    return None


def _lane_to_s(lane_id: str, pos: float, cfg: TrajSceneConfig, edge_len: dict[str, float]) -> float:
    for edge, (s0, s1) in cfg.edge_phys.items():
        if lane_id.startswith(edge + "_"):
            L = edge_len.get(edge, max(s1 - s0, 1.0))
            return s0 + np.clip(pos / L, 0.0, 1.0) * (s1 - s0)
    return float("nan")


def _apply_sumo_params(traci, params: dict) -> None:
    for vtype in traci.vehicletype.getIDList():
        traci.vehicletype.setAccel(vtype, float(params["accel"]))
        traci.vehicletype.setDecel(vtype, float(params["decel"]))
        traci.vehicletype.setImperfection(vtype, float(params["sigma"]))
        traci.vehicletype.setTau(vtype, float(params["tau"]))
        traci.vehicletype.setMinGap(vtype, float(params["minGap"]))
        traci.vehicletype.setSpeedFactor(vtype, float(params["speedFactor"]))
        for k in ("lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain"):
            traci.vehicletype.setParameter(vtype, k, str(float(params[k])))


def collect_observed(cfg: TrajSceneConfig) -> dict[str, Traj]:
    df = pd.read_csv(SCENARIOS[cfg.scene]["data"] / "frenet.csv")
    sub = df[df["laneID"] == cfg.lane_id]
    out: dict[str, Traj] = {}
    for vid, grp in sub.groupby("vehicleID"):
        v = grp.sort_values("time(s)")
        out[str(vid)] = (
            v["time(s)"].values.astype(float),
            v["longitudinalDistance(m)"].values.astype(float),
            v[cfg.speed_col].values.astype(float) / 3.6,
        )
    return out


def collect_simulated(params: dict, cfg: TrajSceneConfig) -> dict[str, Traj]:
    import traci

    sc = SCENARIOS[cfg.scene]
    edge_len = _sumo_edge_lengths(cfg.scene)
    store: dict[str, list[tuple[float, float, float]]] = {}

    traci.start(
        [_sumo_binary(), "-c", str(sc["data"] / "sumo" / f"{sc['cfg']}.sumocfg"),
         "--no-step-log", "true", "--no-warnings", "true", "--seed", "42"],
        label="trajectory-compare",
    )
    try:
        _apply_sumo_params(traci, params)
        while traci.simulation.getTime() < sc["sim_end"]:
            traci.simulationStep()
            t = traci.simulation.getTime()
            for vid in traci.vehicle.getIDList():
                lid = traci.vehicle.getLaneID(vid)
                if _sumo_lane_to_frenet(lid, cfg) != cfg.lane_id:
                    continue
                s = _lane_to_s(lid, traci.vehicle.getLanePosition(vid), cfg, edge_len)
                if np.isfinite(s):
                    store.setdefault(vid, []).append((t, s, traci.vehicle.getSpeed(vid)))
    finally:
        traci.close()

    return {
        vid: (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), np.array([p[2] for p in pts]))
        for vid, pts in store.items()
        if len(pts) >= 2
    }


def _speeds_in_window(trajs: dict[str, Traj], t_lim: tuple[float, float], s_lim: tuple[float, float]) -> np.ndarray:
    t0, t1 = t_lim
    s0, s1 = s_lim
    chunks = []
    for t, s, v in trajs.values():
        m = (t >= t0) & (t <= t1) & (s >= s0) & (s <= s1)
        if m.any():
            chunks.append(v[m])
    return np.concatenate(chunks) if chunks else np.array([])


def _style_ax(
    ax,
    *,
    xlabel: str = "",
    ylabel: str = "",
    labelsize: float = 8,
    show_ylabel: bool = True,
) -> None:
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=labelsize, color="black")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_xlabel(xlabel, fontsize=labelsize, color="black")
    ax.tick_params(labelsize=labelsize - 0.5, colors="black", direction="out")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.set_facecolor("#ffffff")


def _style_cbar(cbar, label: str) -> None:
    cbar.set_label(label, fontsize=9, color="black")
    cbar.ax.tick_params(labelsize=8, colors="black", length=3)
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.9)
    cbar.ax.yaxis.set_ticks_position("right")


def _draw_spacetime(
    ax,
    trajs: dict[str, Traj],
    *,
    t_lim: tuple[float, float],
    s_lim: tuple[float, float],
    speed_lo: float,
    speed_hi: float,
    lw: float = 1.15,
    labelsize: float = 8,
    show_ylabel: bool = True,
) -> LineCollection | None:
    cmap = plt.get_cmap(CMAP)
    lc_ref = None
    span = max(speed_hi - speed_lo, 1e-6)
    t0, t1 = t_lim
    s0, s1 = s_lim

    for t, s, v in trajs.values():
        m = (t >= t0) & (t <= t1) & (s >= s0) & (s <= s1)
        if m.sum() < 2:
            continue
        t, s, v = t[m], s[m], v[m]
        vn = np.clip((v - speed_lo) / span, 0.0, 1.0)
        pts = np.column_stack([t, s]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=NORM, linewidths=lw, alpha=0.9)
        lc.set_array(0.5 * (vn[:-1] + vn[1:]))
        ax.add_collection(lc)
        lc_ref = lc

    ax.set_xlim(t_lim)
    ax.set_ylim(s_lim)
    _style_ax(ax, xlabel="Time (s)", ylabel="Distance (m)", labelsize=labelsize, show_ylabel=show_ylabel)
    return lc_ref


def plot_trajectory_compare(obs: dict[str, Traj], sim: dict[str, Traj], *, dpi: int = 320) -> tuple[Path, Path]:
    t_lim = (ZOOM["t_min"], ZOOM["t_max"])
    s_lim = (ZOOM["s_min"], ZOOM["s_max"])
    speeds = np.concatenate([_speeds_in_window(obs, t_lim, s_lim), _speeds_in_window(sim, t_lim, s_lim)])
    speed_lo = float(np.percentile(speeds, 2))
    speed_hi = float(np.percentile(speeds, 98))

    fig = plt.figure(figsize=FIG_16_9, facecolor="white")
    gs = GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 1.0],
        left=0.09, right=0.90, bottom=0.16, top=0.82, wspace=0.18,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)

    lc = _draw_spacetime(
        ax0, obs, t_lim=t_lim, s_lim=s_lim, speed_lo=speed_lo, speed_hi=speed_hi,
        labelsize=9, show_ylabel=True,
    )
    _draw_spacetime(
        ax1, sim, t_lim=t_lim, s_lim=s_lim, speed_lo=speed_lo, speed_hi=speed_hi,
        labelsize=9, show_ylabel=False,
    )

    for ax, title, cx in ((ax0, "Observed", gs[0, 0].get_position(fig)), (ax1, "Simulated", gs[0, 1].get_position(fig))):
        fig.text(cx.x0 + 0.5 * cx.width, cx.y1 + 0.018, title, ha="center", va="bottom",
                 fontsize=10, fontweight="600", color="black")

    if lc is not None:
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="4.2%", pad=0.11)
        cbar = fig.colorbar(lc, cax=cax)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(["0", "0.5", "1"])
        _style_cbar(cbar, "Speed (norm.)")

    png = OUT / "trajectory_compare.png"
    pdf = OUT / "trajectory_compare.pdf"
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    cal = load_calibrated_params(YTDJ.scene)
    obs = collect_observed(YTDJ)
    sim = collect_simulated(cal, YTDJ)
    png, pdf = plot_trajectory_compare(obs, sim)
    print("Wrote:", png, pdf)


if __name__ == "__main__":
    main()
