"""build_xam_sumo.py  ──  XAM-N6 SUMO 仿真场景构建（几何精确版）

XAM-N6 路段几何（南京龙蟠中路西安门段，全长 140 m）
─────────────────────────────────────────────────────────────────────
上半幅（车道0~4，E→W，图中从右向左行驶，靠右规则）
  外侧交织车道（0,1）：入口汇入 / 出口分流
  内侧主线车道（2,3,4）：主线直行

下半幅（车道5~9，W→E，图中从左向右行驶，靠右规则）
  内侧主线车道（5,6,7）：主线直行
  外侧交织车道（8,9）：入口汇入 / 出口分流

节点布局（X: 0m=西端, 140m=东端；Y: 正=上半幅, 负=下半幅）
  bot_merge  (15m)  ：下半幅匝道与主线进入交织区
  bot_diverge(125m) ：下半幅交织区结束，匝道分流
  top_merge  (125m) ：上半幅匝道与主线进入交织区
  top_diverge(15m)  ：上半幅交织区结束，匝道分流
  核心交织段有效长度 = 125 - 15 = 110 m
─────────────────────────────────────────────────────────────────────
限速分级（与 YTDJ 一致）：
  主线进出段：80 km/h (22.22 m/s)
  交织区：    60 km/h (16.67 m/s)
  匝道入出段：40 km/h (11.11 m/s)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
UTE  = PROJ / "data" / "raw_data" / "UTE"
PROC = PROJ / "data" / "processed_data"
PROC.mkdir(parents=True, exist_ok=True)

# 车道分组（基于 frenet.csv 实际 laneID 0-9）
TOP_MAIN  = {2, 3, 4}   # 上半幅内侧主线
TOP_WEAVE = {0, 1}      # 上半幅外侧交织/匝道
BOT_MAIN  = {5, 6, 7}   # 下半幅内侧主线
BOT_WEAVE = {8, 9}      # 下半幅外侧交织/匝道


def _write_xml(path: Path, root: ET.Element) -> None:
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def _conn(parent: ET.Element, frm: str, to: str, fl: int, tl: int) -> None:
    ET.SubElement(parent, "connection", **{
        "from": frm, "to": to, "fromLane": str(fl), "toLane": str(tl),
    })


# ─────────────────────────────────────────────────────────────────────
# Network builder
# ─────────────────────────────────────────────────────────────────────

def build_xam_network(sumo_dir: Path) -> None:
    """
    Write xam.nod.xml / xam.edg.xml / xam.con.xml based on roadmap geometry.

    Y-coordinate (lane width ≈ 3.2 m):
      下半幅主线中心  y ≈ -4.8  (lanes 5,6,7 × 3.2 m / 2 = 4.8 m from center)
      下半幅匝道中心  y ≈ -12.8 (4.8 + 2 × 3.2 m中心)
      上半幅主线中心  y ≈ +4.8
      上半幅匝道中心  y ≈ +12.8
    """
    sumo_dir.mkdir(parents=True, exist_ok=True)

    # ── Nodes ────────────────────────────────────────────────────────
    nodes = ET.Element("nodes")

    # 下半幅 W→E（lanes 5-9）
    ET.SubElement(nodes, "node", id="bot_start_main", x="0",   y="-4.8")
    ET.SubElement(nodes, "node", id="bot_start_ramp", x="0",   y="-12.8")
    ET.SubElement(nodes, "node", id="bot_merge",      x="15",  y="-8.0")
    ET.SubElement(nodes, "node", id="bot_diverge",    x="125", y="-8.0")
    ET.SubElement(nodes, "node", id="bot_end_main",   x="140", y="-4.8")
    ET.SubElement(nodes, "node", id="bot_end_ramp",   x="140", y="-12.8")

    # 上半幅 E→W（lanes 0-4）
    ET.SubElement(nodes, "node", id="top_start_main", x="140", y="4.8")
    ET.SubElement(nodes, "node", id="top_start_ramp", x="140", y="12.8")
    ET.SubElement(nodes, "node", id="top_merge",      x="125", y="8.0")
    ET.SubElement(nodes, "node", id="top_diverge",    x="15",  y="8.0")
    ET.SubElement(nodes, "node", id="top_end_main",   x="0",   y="4.8")
    ET.SubElement(nodes, "node", id="top_end_ramp",   x="0",   y="12.8")

    _write_xml(sumo_dir / "xam.nod.xml", nodes)

    # ── Edges ────────────────────────────────────────────────────────
    edges = ET.Element("edges")
    s_main  = "22.22"   # 80 km/h 主线
    s_weave = "16.67"   # 60 km/h 交织区
    s_ramp  = "11.11"   # 40 km/h 匝道

    # 下半幅（W→E）
    ET.SubElement(edges, "edge", id="bot_main_in",
                  shape="0,-4.8 15,-4.8",
                  **{"from": "bot_start_main", "to": "bot_merge",
                     "numLanes": "3", "speed": s_main})
    ET.SubElement(edges, "edge", id="bot_ramp_in",
                  shape="0,-12.8 15,-12.8",
                  **{"from": "bot_start_ramp", "to": "bot_merge",
                     "numLanes": "2", "speed": s_ramp})
    ET.SubElement(edges, "edge", id="bot_weave",
                  shape="15,-8.0 125,-8.0",
                  **{"from": "bot_merge", "to": "bot_diverge",
                     "numLanes": "5", "speed": s_weave})
    ET.SubElement(edges, "edge", id="bot_main_out",
                  shape="125,-4.8 140,-4.8",
                  **{"from": "bot_diverge", "to": "bot_end_main",
                     "numLanes": "3", "speed": s_main})
    ET.SubElement(edges, "edge", id="bot_ramp_out",
                  shape="125,-12.8 140,-12.8",
                  **{"from": "bot_diverge", "to": "bot_end_ramp",
                     "numLanes": "2", "speed": s_ramp})

    # 上半幅（E→W）
    ET.SubElement(edges, "edge", id="top_main_in",
                  shape="140,4.8 125,4.8",
                  **{"from": "top_start_main", "to": "top_merge",
                     "numLanes": "3", "speed": s_main})
    ET.SubElement(edges, "edge", id="top_ramp_in",
                  shape="140,12.8 125,12.8",
                  **{"from": "top_start_ramp", "to": "top_merge",
                     "numLanes": "2", "speed": s_ramp})
    ET.SubElement(edges, "edge", id="top_weave",
                  shape="125,8.0 15,8.0",
                  **{"from": "top_merge", "to": "top_diverge",
                     "numLanes": "5", "speed": s_weave})
    ET.SubElement(edges, "edge", id="top_main_out",
                  shape="15,4.8 0,4.8",
                  **{"from": "top_diverge", "to": "top_end_main",
                     "numLanes": "3", "speed": s_main})
    ET.SubElement(edges, "edge", id="top_ramp_out",
                  shape="15,12.8 0,12.8",
                  **{"from": "top_diverge", "to": "top_end_ramp",
                     "numLanes": "2", "speed": s_ramp})

    _write_xml(sumo_dir / "xam.edg.xml", edges)

    # ── Connections ───────────────────────────────────────────────────
    # SUMO lane 0 = rightmost（靠驾驶员右手侧）。
    # 下半幅（W→E）：rightmost = 外侧 = 交织/匝道车道 → SUMO lane 0,1 对应匝道
    # 上半幅（E→W）：rightmost（from driver's right）= 也是外侧 = 匝道 → SUMO lane 0,1 对应匝道
    conns = ET.Element("connections")

    # 下半幅：合流(15m) → 交织区
    _conn(conns, "bot_ramp_in",  "bot_weave", 0, 0)
    _conn(conns, "bot_ramp_in",  "bot_weave", 1, 1)
    _conn(conns, "bot_main_in",  "bot_weave", 0, 2)
    _conn(conns, "bot_main_in",  "bot_weave", 1, 3)
    _conn(conns, "bot_main_in",  "bot_weave", 2, 4)

    # 下半幅：交织区 → 分流(125m)
    _conn(conns, "bot_weave", "bot_ramp_out", 0, 0)
    _conn(conns, "bot_weave", "bot_ramp_out", 1, 1)
    _conn(conns, "bot_weave", "bot_main_out", 2, 0)
    _conn(conns, "bot_weave", "bot_main_out", 3, 1)
    _conn(conns, "bot_weave", "bot_main_out", 4, 2)

    # 上半幅：合流(125m) → 交织区
    _conn(conns, "top_ramp_in",  "top_weave", 0, 0)
    _conn(conns, "top_ramp_in",  "top_weave", 1, 1)
    _conn(conns, "top_main_in",  "top_weave", 0, 2)
    _conn(conns, "top_main_in",  "top_weave", 1, 3)
    _conn(conns, "top_main_in",  "top_weave", 2, 4)

    # 上半幅：交织区 → 分流(15m)
    _conn(conns, "top_weave", "top_ramp_out", 0, 0)
    _conn(conns, "top_weave", "top_ramp_out", 1, 1)
    _conn(conns, "top_weave", "top_main_out", 2, 0)
    _conn(conns, "top_weave", "top_main_out", 3, 1)
    _conn(conns, "top_weave", "top_main_out", 4, 2)

    _write_xml(sumo_dir / "xam.con.xml", conns)


# ─────────────────────────────────────────────────────────────────────
# netconvert
# ─────────────────────────────────────────────────────────────────────

def run_netconvert(sumo_dir: Path, prefix: str) -> None:
    cmd = [
        "netconvert",
        "--node-files",       str(sumo_dir / f"{prefix}.nod.xml"),
        "--edge-files",       str(sumo_dir / f"{prefix}.edg.xml"),
        "--connection-files", str(sumo_dir / f"{prefix}.con.xml"),
        "--output-file",      str(sumo_dir / f"{prefix}.net.xml"),
        "--geometry.min-radius", "5",
        "--no-turnarounds",   "true",
        "--junctions.join",   "false",
        "--offset.disable-normalization", "true",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("netconvert STDERR:\n", r.stderr[-3000:])
        raise RuntimeError(f"netconvert failed for {prefix}")
    print(f"  netconvert OK → {sumo_dir / (prefix + '.net.xml')}")


# ─────────────────────────────────────────────────────────────────────
# Flow estimation from frenet.csv
# ─────────────────────────────────────────────────────────────────────

def classify_od(start_lane: int, end_lane: int) -> str | None:
    """将起终车道映射为8类OD。"""
    in_top  = start_lane in TOP_MAIN | TOP_WEAVE
    in_bot  = start_lane in BOT_MAIN | BOT_WEAVE
    if not (in_top or in_bot):
        return None

    if in_top:
        prefix = "N"
        sm, sw = TOP_MAIN, TOP_WEAVE
    else:
        prefix = "S"
        sm, sw = BOT_MAIN, BOT_WEAVE

    src = "main" if start_lane in sm else "weave"
    dst_m = end_lane in sm
    dst_w = end_lane in sw

    if src == "main"  and dst_m:  return f"{prefix}_through"
    if src == "main"  and dst_w:  return f"{prefix}_weave_exit"   # 主→匝
    if src == "weave" and dst_m:  return f"{prefix}_weave_enter"  # 匝→主
    if src == "weave" and dst_w:  return f"{prefix}_ramp_through"
    return None


ODS = [
    "N_through", "N_weave_enter", "N_weave_exit", "N_ramp_through",
    "S_through", "S_weave_enter", "S_weave_exit", "S_ramp_through",
]

# 每个 OD 对应的 SUMO 路由（edge 序列）
OD_ROUTES = {
    "N_through":      "top_main_in top_weave top_main_out",
    "N_weave_exit":   "top_main_in top_weave top_ramp_out",
    "N_weave_enter":  "top_ramp_in top_weave top_main_out",
    "N_ramp_through": "top_ramp_in top_weave top_ramp_out",
    "S_through":      "bot_main_in bot_weave bot_main_out",
    "S_weave_exit":   "bot_main_in bot_weave bot_ramp_out",
    "S_weave_enter":  "bot_ramp_in bot_weave bot_main_out",
    "S_ramp_through": "bot_ramp_in bot_weave bot_ramp_out",
}


def estimate_flows(scene_dir: Path) -> dict:
    df   = pd.read_csv(scene_dir / "frenet.csv")
    df   = df.sort_values(["vehicleID", "time(s)"]).copy()
    g    = df.groupby("vehicleID", sort=False)
    veh  = g.agg(
        start_time=("time(s)", "first"),
        end_time=("time(s)", "last"),
        start_lane=("laneID", "first"),
        end_lane=("laneID", "last"),
    ).reset_index()
    veh["start_lane"] = veh["start_lane"].astype(int)
    veh["end_lane"]   = veh["end_lane"].astype(int)
    veh["od"]         = veh.apply(lambda r: classify_od(r["start_lane"], r["end_lane"]), axis=1)
    veh = veh[veh["od"].notna()].copy()

    tmin = float(df["time(s)"].min())
    tmax = float(df["time(s)"].max())
    dur_h = (tmax - tmin) / 3600.0

    vph = lambda v: max(50, int(round(v / dur_h)))
    counts = veh["od"].value_counts().to_dict()
    return {
        "duration_s": tmax - tmin,
        **{f"{od}_vph": vph(counts.get(od, 0)) for od in ODS},
    }


# ─────────────────────────────────────────────────────────────────────
# Route file and sumocfg
# ─────────────────────────────────────────────────────────────────────

def build_routes(sumo_dir: Path, flows: dict, sim_end: int) -> None:
    e = str(sim_end)
    lines = ["<routes>"]
    lines.append('    <vType id="car"      accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="22.22"/>')
    lines.append('    <vType id="car_ramp" accel="2.0" decel="4.5" sigma="0.5" length="5.0" maxSpeed="11.11"/>')
    lines.append("")

    for od in ODS:
        vt = "car_ramp" if "ramp" in od or "weave" in od else "car"
        vph_val = flows[f"{od}_vph"]
        depart  = "best" if "through" in od else "free"
        lines.append(f'    <!-- {od} -->')
        lines.append(f'    <route id="r_{od}" edges="{OD_ROUTES[od]}"/>')
        lines.append(
            f'    <flow  id="f_{od}" type="{vt}" route="r_{od}"'
            f' begin="0" end="{e}" vehsPerHour="{vph_val}" departLane="{depart}"/>'
        )
        lines.append("")

    lines.append("</routes>")
    (sumo_dir / "xam.rou.xml").write_text("\n".join(lines), encoding="utf-8")


def build_sumocfg(sumo_dir: Path, sim_end: int) -> Path:
    cfg = f"""<configuration>
    <input>
        <net-file    value="xam.net.xml"/>
        <route-files value="xam.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end   value="{sim_end}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
    <report>
        <no-warnings value="false"/>
        <verbose      value="false"/>
    </report>
</configuration>
"""
    p = sumo_dir / "xam.sumocfg"
    p.write_text(cfg, encoding="utf-8")
    return p


def run_quick_sumo(cfg_path: Path) -> dict:
    trip = cfg_path.with_suffix(".tripinfo.xml")
    cmd  = ["sumo", "-c", str(cfg_path),
            "--tripinfo-output", str(trip),
            "--no-step-log", "true"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("SUMO STDERR:", r.stderr[-2000:])
        return {"departed": -1, "arrived": -1}
    trips = ET.parse(trip).getroot().findall("tripinfo")
    return {"departed": len(trips), "arrived": len(trips)}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    scene_dir = UTE / "XAM-N6"
    sumo_dir  = scene_dir / "sumo"

    print("=" * 60)
    print("  XAM-N6 SUMO scenario build  (geometry-accurate)")
    print("=" * 60)

    print("\n[1/5]  Writing network XML files …")
    build_xam_network(sumo_dir)

    print("[2/5]  Running netconvert …")
    run_netconvert(sumo_dir, "xam")

    print("[3/5]  Estimating 8-class OD flows from frenet.csv …")
    flows = estimate_flows(scene_dir)
    for od in ODS:
        print(f"       {od:<22}: {flows[f'{od}_vph']:>5} vph")

    sim_end = int(flows["duration_s"])
    print(f"[4/5]  Writing routes & sumocfg  (sim_end = {sim_end} s) …")
    build_routes(sumo_dir, flows, sim_end)
    cfg_path = build_sumocfg(sumo_dir, sim_end)

    print("[5/5]  SUMO verification run …")
    result = run_quick_sumo(cfg_path)
    print(f"       departed={result['departed']}  arrived={result['arrived']}")

    summary = {
        "scene": "XAM-N6",
        "description": "双向对称交织区（南京西安门段），上半幅E→W，下半幅W→E，110m核心交织段",
        "total_length_m": 140,
        "weave_length_m": 110,
        "lane_config": {
            "top_half":  {"direction": "E→W", "main_lanes": [2,3,4], "weave_lanes": [0,1]},
            "bot_half":  {"direction": "W→E", "main_lanes": [5,6,7], "weave_lanes": [8,9]},
        },
        "flows_vph": {od: flows[f"{od}_vph"] for od in ODS},
        "sim_end_s": sim_end,
        "verification": result,
        "sumo_files": {
            "net":     str(sumo_dir / "xam.net.xml"),
            "rou":     str(sumo_dir / "xam.rou.xml"),
            "sumocfg": str(cfg_path),
        },
    }
    out = PROC / "xam_build_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary → {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
