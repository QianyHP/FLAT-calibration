"""build_rml_sumo.py  ──  RML SUMO 仿真场景构建（几何精确版）

RML 路段几何（长沙东二环人民路，全长 220 m，自北向南 N→S）
─────────────────────────────────────────────────────────────────────
车道编号（frenet.csv 真实 laneID）：
  Lane 1  ①：主线最左侧（内侧）
  Lane 2  ②：主线中间
  Lane 3  ③：主线最右侧（外侧主线）
  Lane 4  ④：入口匝道 / 加速车道（最外侧，下游 Lane Drop）

几何分段：
  [  0m ➝  25m]  物理隔离汇入段（25m）：主线3车道 + 匝道1车道平行，白实线隔离
  [ 25m ➝ 165m]  并行加速合流区（140m）：单向4车道，匝道车辆须在此完成强制换道
  [165m ➝ 220m]  车道收窄 & 主线保持段（55m）：Lane Drop，④车道消失，恢复3车道

限速分级（与 YTDJ/XAM 保持一致）：
  主线进出段：80 km/h (22.22 m/s)
  并行合流区：60 km/h (16.67 m/s)
  入口匝道：  40 km/h (11.11 m/s)
─────────────────────────────────────────────────────────────────────
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

# 车道分组（frenet.csv laneID 1-4；lane 5 极少忽略）
MAIN_LANES = {1, 2, 3}   # 主线直行车道
RAMP_LANES = {4}          # 入口匝道/加速车道（下游 Lane Drop）


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

def build_rml_network(sumo_dir: Path) -> None:
    """
    Write rml.nod.xml / rml.edg.xml / rml.con.xml.

    Y-coordinate（车道宽 3.5 m，与 YTDJ/XAM 对齐）：
      主线3车道中心     Y =  0.0
      4车道段中心       Y = -1.75  （主路上边界保持 y=+5.25，底部向外扩一条 3.5m 车道）
      匝道单车道中心    Y = -7.0   （3.5×2 lanes below road center）

    修复说明（v2）：
      - 删除所有 shape 覆盖：shape 端点≠节点会导致 SUMO 插补扭曲几何，
        使左侧 merge 渲染成和右侧 Lane Drop 对称的"漏斗"外观。
      - 匝道起点后移至 x=-60（场景外 60m），使匝道在全图中清晰可见，
        避免仅有 25m 匝道看起来像"不存在"。
      - 让 SUMO 根据节点坐标自动计算各段直线几何，parallel 段仍保留
        shape 确保绝对水平。
    """
    sumo_dir.mkdir(parents=True, exist_ok=True)

    # ── Nodes ────────────────────────────────────────────────────────
    nodes = ET.Element("nodes")
    # 主线入口：场景左端 (0m)，Y=0
    ET.SubElement(nodes, "node", id="start_main", x="0",   y="0")
    # 匝道起点：场景外 60m 处（x=-60），Y=-7.0（单车道④中心）
    # 这 60m 让匝道在 SUMO 全局视图中清晰可见
    ET.SubElement(nodes, "node", id="ramp_start", x="-60", y="-7.0")
    # 25m 处：物理隔离带结束，merge junction，4车道中心 Y=-1.75
    ET.SubElement(nodes, "node", id="merge_pt",   x="25",  y="-1.75")
    # 165m 处：Lane Drop 起点
    ET.SubElement(nodes, "node", id="drop_pt",    x="165", y="-1.75")
    # 场景右端 (220m)，主线恢复 Y=0
    ET.SubElement(nodes, "node", id="end_main",   x="220", y="0")
    _write_xml(sumo_dir / "rml.nod.xml", nodes)

    # ── Edges ────────────────────────────────────────────────────────
    edges = ET.Element("edges")
    s_main  = "22.22"   # 80 km/h
    s_merge = "16.67"   # 60 km/h（合流交织区降速）
    s_ramp  = "11.11"   # 40 km/h

    # 主线入口（0→25m）：SUMO 按节点直线计算，无 shape 覆盖
    ET.SubElement(edges, "edge", id="main_in",
                  **{"from": "start_main", "to": "merge_pt",
                     "numLanes": "3", "speed": s_main})

    # 入口匝道（-60→25m，共 85m）：从场景外斜入，SUMO 按节点直线计算
    # 视觉上清晰呈现为从下方斜向汇入主路的独立匝道
    ET.SubElement(edges, "edge", id="ramp_in",
                  **{"from": "ramp_start", "to": "merge_pt",
                     "numLanes": "1", "speed": s_ramp})

    # 并行加速合流区（25→165m，4车道，水平直线）
    ET.SubElement(edges, "edge", id="parallel",
                  shape="25,-1.75 165,-1.75",
                  **{"from": "merge_pt", "to": "drop_pt",
                     "numLanes": "4", "speed": s_merge})

    # Lane Drop + 主线保持段（165→220m，3车道）：SUMO 按节点直线计算
    ET.SubElement(edges, "edge", id="main_out",
                  **{"from": "drop_pt", "to": "end_main",
                     "numLanes": "3", "speed": s_main})
    _write_xml(sumo_dir / "rml.edg.xml", edges)

    # ── Connections ───────────────────────────────────────────────────
    # SUMO lane 0 = rightmost（最外侧 = ④ 匝道车道）
    conns = ET.Element("connections")

    # 25m 汇流：ramp[0]→parallel[0]；main[0,1,2]→parallel[1,2,3]
    # SUMO lane 0 = rightmost = ④（最外侧匝道）
    _conn(conns, "ramp_in",  "parallel", 0, 0)   # ④ 匝道 → parallel 最外侧
    _conn(conns, "main_in",  "parallel", 0, 1)   # ③ 主线外侧
    _conn(conns, "main_in",  "parallel", 1, 2)   # ②
    _conn(conns, "main_in",  "parallel", 2, 3)   # ① 主线内侧

    # 165m Lane Drop：parallel[0]（④车道）故意不接出口 → 强制换道
    # parallel[1,2,3] → main_out[0,1,2]（③②① 平滑通过）
    _conn(conns, "parallel", "main_out", 1, 0)
    _conn(conns, "parallel", "main_out", 2, 1)
    _conn(conns, "parallel", "main_out", 3, 2)

    _write_xml(sumo_dir / "rml.con.xml", conns)


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
        "--geometry.min-radius", "10",
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

ODS = ["main_through", "ramp_merge"]

OD_ROUTES = {
    "main_through": "main_in parallel main_out",
    "ramp_merge":   "ramp_in parallel main_out",
}


def classify_od(start_lane: int, end_lane: int) -> str | None:
    if start_lane in MAIN_LANES and end_lane in MAIN_LANES:
        return "main_through"
    if start_lane in RAMP_LANES and end_lane in MAIN_LANES:
        return "ramp_merge"
    # 匝道车未完成合流（end_lane=4）：仍按 ramp_merge 路由，SUMO 会强制换道
    if start_lane in RAMP_LANES and end_lane in RAMP_LANES:
        return "ramp_merge"
    return None


def estimate_flows(scene_dir: Path) -> dict:
    df  = pd.read_csv(scene_dir / "frenet.csv")
    df  = df.sort_values(["vehicleID", "time(s)"]).copy()
    g   = df.groupby("vehicleID", sort=False)
    veh = g.agg(
        start_lane=("laneID", "first"),
        end_lane=("laneID", "last"),
    ).reset_index()
    veh["start_lane"] = veh["start_lane"].astype(int)
    veh["end_lane"]   = veh["end_lane"].astype(int)
    veh["od"]         = veh.apply(lambda r: classify_od(r["start_lane"], r["end_lane"]), axis=1)
    veh = veh[veh["od"].notna()].copy()

    tmin  = float(df["time(s)"].min())
    tmax  = float(df["time(s)"].max())
    dur_h = (tmax - tmin) / 3600.0
    vph   = lambda v: max(50, int(round(v / dur_h)))

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
    lines = [
        "<routes>",
        '    <vType id="car"      accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="22.22"/>',
        '    <vType id="car_ramp" accel="2.0" decel="4.5" sigma="0.5" length="5.0" maxSpeed="11.11"/>',
        "",
        "    <!-- 主线直行 (lanes 1~3) -->",
        f'    <route id="r_main_through" edges="{OD_ROUTES["main_through"]}"/>',
        f'    <flow  id="f_main_through" type="car" route="r_main_through"'
        f' begin="0" end="{e}" vehsPerHour="{flows["main_through_vph"]}" departLane="best"/>',
        "",
        "    <!-- 入口匝道合流 (lane 4 → lanes 1~3) -->",
        f'    <route id="r_ramp_merge" edges="{OD_ROUTES["ramp_merge"]}"/>',
        f'    <flow  id="f_ramp_merge" type="car_ramp" route="r_ramp_merge"'
        f' begin="0" end="{e}" vehsPerHour="{flows["ramp_merge_vph"]}" departLane="free"/>',
        "",
        "</routes>",
    ]
    (sumo_dir / "rml.rou.xml").write_text("\n".join(lines), encoding="utf-8")


def build_sumocfg(sumo_dir: Path, sim_end: int) -> Path:
    cfg = f"""<configuration>
    <input>
        <net-file    value="rml.net.xml"/>
        <route-files value="rml.rou.xml"/>
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
    p = sumo_dir / "rml.sumocfg"
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
    scene_dir = UTE / "RML"
    sumo_dir  = scene_dir / "sumo"

    print("=" * 60)
    print("  RML SUMO scenario build  (geometry-accurate)")
    print("=" * 60)

    print("\n[1/5]  Writing network XML files …")
    build_rml_network(sumo_dir)

    print("[2/5]  Running netconvert …")
    run_netconvert(sumo_dir, "rml")

    print("[3/5]  Estimating OD flows from frenet.csv …")
    flows = estimate_flows(scene_dir)
    for od in ODS:
        print(f"       {od:<20}: {flows[f'{od}_vph']:>5} vph")

    sim_end = int(flows["duration_s"])
    print(f"[4/5]  Writing routes & sumocfg  (sim_end = {sim_end} s) …")
    build_routes(sumo_dir, flows, sim_end)
    cfg_path = build_sumocfg(sumo_dir, sim_end)

    print("[5/5]  SUMO verification run …")
    result = run_quick_sumo(cfg_path)
    print(f"       departed={result['departed']}  arrived={result['arrived']}")

    summary = {
        "scene": "RML",
        "description": "城市快速路入口匝道汇入与车道收缩（长沙东二环，Lane Drop 场景）",
        "total_length_m": 220,
        "key_points_m": {
            "isolation_end": 25,
            "lane_drop_start": 165,
            "merge_window_m": 140,
        },
        "lane_config": {
            "main_lanes": [1, 2, 3],
            "ramp_lane": [4],
        },
        "flows_vph": {od: flows[f"{od}_vph"] for od in ODS},
        "sim_end_s": sim_end,
        "verification": result,
        "sumo_files": {
            "net":     str(sumo_dir / "rml.net.xml"),
            "rou":     str(sumo_dir / "rml.rou.xml"),
            "sumocfg": str(cfg_path),
        },
    }
    out = PROC / "rml_build_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary → {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
