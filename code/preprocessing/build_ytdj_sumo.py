"""build_ytdj_sumo.py  ──  YTDJ SUMO 仿真场景构建（几何精确版）

YTDJ 路段几何（全长 362 m）
─────────────────────────────────────────────────────────────────────
北侧（车道1~5，E→W，图中从右向左行驶）
  362 → 180 m  : 3 车道主线保持段              (ew_main_in)
  180 → 104 m  : 5 车道（180m展宽，142m完成，104m前平行准备）(ew_parallel)
  104 →   0 m  : 分流 → 主线3车道(ew_main_out) + 出口匝道2车道(ew_ramp_out)

南侧（车道6~10，W→E，图中从左向右行驶）
    0 →  80 m  : 物理隔离段：主线3车道(we_main_in) + 匝道2车道(we_ramp_in)
   80 → 202 m  : 5车道并行加速/交织区           (we_parallel)
  202 → 362 m  : 收窄恢复3车道主线（202m起收，240m完成）(we_main_out)
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
UTE  = PROJ / "data" / "raw_data" / "UTE"
PROC = PROJ / "data" / "processed_data"
PROC.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _write_xml(path: Path, root: ET.Element) -> None:
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass  # Python < 3.9 has no ET.indent
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def _conn(parent: ET.Element, frm: str, to: str, fl: int, tl: int) -> None:
    ET.SubElement(parent, "connection", **{
        "from": frm, "to": to, "fromLane": str(fl), "toLane": str(tl),
    })


# ─────────────────────────────────────────────────────────────────────
# Network builder
# ─────────────────────────────────────────────────────────────────────

def build_ytdj_network(sumo_dir: Path) -> None:
    """
    Write ytdj.nod.xml / ytdj.edg.xml / ytdj.con.xml based on roadmap geometry.

    Y-coordinate layout (lane width ≈ 3.5 m):
      North 3-lane road centre  ≈  +10 m
      South 3-lane road centre  ≈  -10 m
      North ramp exits northward (y ≈ +17)
      South ramp enters from south (y ≈ -17)
      ~20 m total separation between directions (median + shoulder).
    """
    sumo_dir.mkdir(parents=True, exist_ok=True)

    # ── Nodes ────────────────────────────────────────────────────────
    nodes = ET.Element("nodes")

    # South side  W→E  (real lanes 6~10)
    ET.SubElement(nodes, "node", id="we_start_main", x="0",   y="-5")   # main-line west entry
    ET.SubElement(nodes, "node", id="we_start_ramp", x="0",   y="-17")  # ramp west entry (south of main, physically isolated)
    ET.SubElement(nodes, "node", id="we_merge",      x="80",  y="-10")  # 80m: isolation ends → 5-lane parallel starts
    ET.SubElement(nodes, "node", id="we_narrow",     x="202", y="-10")  # 202m: 5-lane starts narrowing to 3-lane
    ET.SubElement(nodes, "node", id="we_end",        x="367", y="-10")  # east exit (slightly beyond 362m for clean geometry)

    # North side  E→W  (real lanes 1~5)
    ET.SubElement(nodes, "node", id="ew_start",      x="362", y="10")   # east entry (3 lanes)
    ET.SubElement(nodes, "node", id="ew_widen",      x="180", y="10")   # 180m: 3-lane → 5-lane widening starts
    ET.SubElement(nodes, "node", id="ew_split",      x="104", y="10")   # 104m: physical diverge starts → ramp separates
    ET.SubElement(nodes, "node", id="ew_end_main",   x="0",   y="5")    # main-line west exit
    ET.SubElement(nodes, "node", id="ew_end_ramp",   x="0",   y="17")   # ramp west exit (north of main, physically isolated)

    _write_xml(sumo_dir / "ytdj.nod.xml", nodes)

    # ── Edges ────────────────────────────────────────────────────────
    edges = ET.Element("edges")
    # 中国城市快速路常见控制：
    # - 主线限速通常不高于 80 km/h
    # - 交织/分流过渡段一般下调至约 60 km/h
    # - 匝道常见约 40 km/h（以现场标志为准）
    s_main = "22.22"      # 80 km/h
    s_transition = "16.67"  # 60 km/h
    s_ramp = "11.11"      # 40 km/h

    # South side (W→E)
    ET.SubElement(edges, "edge", id="we_main_in",  **{"from": "we_start_main", "to": "we_merge",      "numLanes": "3", "speed": s_main})
    ET.SubElement(edges, "edge", id="we_ramp_in",  **{"from": "we_start_ramp", "to": "we_merge",      "numLanes": "2", "speed": s_ramp})
    ET.SubElement(edges, "edge", id="we_parallel", **{"from": "we_merge",      "to": "we_narrow",     "numLanes": "5", "speed": s_transition})
    ET.SubElement(edges, "edge", id="we_main_out", **{"from": "we_narrow",     "to": "we_end",        "numLanes": "3", "speed": s_transition})

    # North side (E→W)
    ET.SubElement(edges, "edge", id="ew_main_in",  **{"from": "ew_start",      "to": "ew_widen",      "numLanes": "3", "speed": s_main})
    ET.SubElement(edges, "edge", id="ew_parallel", **{"from": "ew_widen",      "to": "ew_split",      "numLanes": "5", "speed": s_transition})
    ET.SubElement(edges, "edge", id="ew_main_out", **{"from": "ew_split",      "to": "ew_end_main",   "numLanes": "3", "speed": s_transition})
    ET.SubElement(edges, "edge", id="ew_ramp_out", **{"from": "ew_split",      "to": "ew_end_ramp",   "numLanes": "2", "speed": s_ramp})

    _write_xml(sumo_dir / "ytdj.edg.xml", edges)

    # ── Connections ───────────────────────────────────────────────────
    # SUMO convention: lane 0 = rightmost (driver's right).
    #
    # North E→W view: rightmost of the 5-lane section = real lanes 4,5 (exit ramp).
    # South W→E view: rightmost of the 5-lane section = real lanes 9,10 (entry ramp).
    conns = ET.Element("connections")

    # ── South: ramp + main merge at 80 m ─────────────────────────────
    # Ramp lanes (9,10) enter as SUMO lanes 0,1 (rightmost = physically south / outer).
    # Main lanes (6,7,8) enter as SUMO lanes 2,3,4 (left / inner).
    _conn(conns, "we_ramp_in",  "we_parallel", 0, 0)
    _conn(conns, "we_ramp_in",  "we_parallel", 1, 1)
    _conn(conns, "we_main_in",  "we_parallel", 0, 2)
    _conn(conns, "we_main_in",  "we_parallel", 1, 3)
    _conn(conns, "we_main_in",  "we_parallel", 2, 4)

    # ── South: narrowing at 202 m ─────────────────────────────────────
    # Ramp lanes (0,1) physically disappear; vehicles must have already merged
    # into lanes 2-4 during the 122 m parallel section.
    # Only lanes 2,3,4 carry physical connections forward.
    _conn(conns, "we_parallel", "we_main_out", 2, 0)
    _conn(conns, "we_parallel", "we_main_out", 3, 1)
    _conn(conns, "we_parallel", "we_main_out", 4, 2)

    # ── North: widening at 180 m ──────────────────────────────────────
    # Original main lanes (1,2,3) → SUMO lanes 2,3,4 of ew_parallel (left/inner).
    # New outer lanes (4,5 = ramp lanes, SUMO lanes 0,1) appear from shoulder.
    # Add direct connections from lane 0 so exit-ramp vehicles can immediately
    # position themselves in the rightmost lanes without needing 2 lc in 76 m.
    _conn(conns, "ew_main_in",  "ew_parallel", 0, 0)   # direct to ramp lane
    _conn(conns, "ew_main_in",  "ew_parallel", 0, 1)   # direct to ramp lane
    _conn(conns, "ew_main_in",  "ew_parallel", 0, 2)   # main→ inner
    _conn(conns, "ew_main_in",  "ew_parallel", 1, 3)
    _conn(conns, "ew_main_in",  "ew_parallel", 2, 4)

    # ── North: diverge at 104 m ───────────────────────────────────────
    # Lanes 0,1 (rightmost = real lanes 4,5) → exit ramp.
    # Lanes 2,3,4 (left = real lanes 1,2,3) → main line continues.
    _conn(conns, "ew_parallel", "ew_ramp_out", 0, 0)
    _conn(conns, "ew_parallel", "ew_ramp_out", 1, 1)
    _conn(conns, "ew_parallel", "ew_main_out", 2, 0)
    _conn(conns, "ew_parallel", "ew_main_out", 3, 1)
    _conn(conns, "ew_parallel", "ew_main_out", 4, 2)

    _write_xml(sumo_dir / "ytdj.con.xml", conns)


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
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("netconvert STDERR:\n", r.stderr[-3000:])
        raise RuntimeError(f"netconvert failed for {prefix}")
    print(f"  netconvert OK → {sumo_dir / (prefix + '.net.xml')}")


# ─────────────────────────────────────────────────────────────────────
# Flow estimation from frenet.csv
# ─────────────────────────────────────────────────────────────────────

def estimate_ytdj_flows(scene_dir: Path) -> dict:
    df   = pd.read_csv(scene_dir / "frenet.csv")
    cols = {c.lower(): c for c in df.columns}

    time_col = cols["time(s)"]
    vid_col  = cols["vehicleid"]
    lane_col = cols["laneid"]

    tmin = float(df[time_col].min())
    tmax = float(df[time_col].max())
    dur_h = (tmax - tmin) / 3600.0

    lanes = df[lane_col].dropna().astype(int)
    n_total = df[lanes.between(1, 5)][vid_col].nunique()
    s_total = df[lanes.between(6, 10)][vid_col].nunique()
    n_ramp  = df[lanes.isin([4, 5])][vid_col].nunique()
    s_ramp  = df[lanes.isin([9, 10])][vid_col].nunique()

    vph = lambda v: max(50, int(round(v / dur_h)))
    return {
        "duration_s":  tmax - tmin,
        "n_main_vph":  vph(n_total - n_ramp),
        "n_ramp_vph":  vph(n_ramp),
        "s_main_vph":  vph(s_total - s_ramp),
        "s_ramp_vph":  vph(s_ramp),
    }


# ─────────────────────────────────────────────────────────────────────
# Route file and sumocfg
# ─────────────────────────────────────────────────────────────────────

def build_routes(sumo_dir: Path, flows: dict, sim_end: int) -> None:
    e = str(sim_end)
    rou = f"""<routes>
    <vType id="car"      accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="22.22"/>
    <vType id="car_ramp" accel="2.0" decel="4.5" sigma="0.5" length="5.0" maxSpeed="11.11"/>

    <!-- 北侧 E→W 主线直行（车道1~3） -->
    <route id="r_N_main" edges="ew_main_in ew_parallel ew_main_out"/>
    <flow  id="f_N_main" type="car"      route="r_N_main"
           begin="0" end="{e}" vehsPerHour="{flows['n_main_vph']}" departLane="best"/>

    <!-- 北侧 E→W 驶出匝道（车道4~5） -->
    <route id="r_N_ramp" edges="ew_main_in ew_parallel ew_ramp_out"/>
    <flow  id="f_N_ramp" type="car_ramp" route="r_N_ramp"
           begin="0" end="{e}" vehsPerHour="{flows['n_ramp_vph']}" departLane="free"/>

    <!-- 南侧 W→E 主线直行（车道6~8） -->
    <route id="r_S_main" edges="we_main_in we_parallel we_main_out"/>
    <flow  id="f_S_main" type="car"      route="r_S_main"
           begin="0" end="{e}" vehsPerHour="{flows['s_main_vph']}" departLane="best"/>

    <!-- 南侧 W→E 匝道汇入主线（车道9~10） -->
    <route id="r_S_ramp" edges="we_ramp_in we_parallel we_main_out"/>
    <flow  id="f_S_ramp" type="car_ramp" route="r_S_ramp"
           begin="0" end="{e}" vehsPerHour="{flows['s_ramp_vph']}" departLane="best"/>
</routes>
"""
    (sumo_dir / "ytdj.rou.xml").write_text(rou, encoding="utf-8")


def build_sumocfg(sumo_dir: Path, sim_end: int) -> Path:
    cfg = f"""<configuration>
    <input>
        <net-file    value="ytdj.net.xml"/>
        <route-files value="ytdj.rou.xml"/>
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
    p = sumo_dir / "ytdj.sumocfg"
    p.write_text(cfg, encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────
# Quick verification
# ─────────────────────────────────────────────────────────────────────

def run_quick_sumo(cfg_path: Path) -> dict:
    trip = cfg_path.with_suffix(".tripinfo.xml")
    cmd  = ["sumo", "-c", str(cfg_path),
            "--tripinfo-output", str(trip),
            "--no-step-log", "true"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("SUMO STDERR:", r.stderr[-2000:])
        return {"departed": -1, "arrived": -1, "error": r.stderr[-500:]}
    trips = ET.parse(trip).getroot().findall("tripinfo")
    return {"departed": len(trips), "arrived": len(trips)}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    scene_dir = UTE / "YTDJ"
    sumo_dir  = scene_dir / "sumo"

    print("=" * 60)
    print("  YTDJ SUMO scenario build  (geometry-accurate)")
    print("=" * 60)

    print("\n[1/5]  Writing network XML files …")
    build_ytdj_network(sumo_dir)

    print("[2/5]  Running netconvert …")
    run_netconvert(sumo_dir, "ytdj")

    print("[3/5]  Estimating directional flows from frenet.csv …")
    flows = estimate_ytdj_flows(scene_dir)
    print(f"       North: main={flows['n_main_vph']} vph   ramp={flows['n_ramp_vph']} vph")
    print(f"       South: main={flows['s_main_vph']} vph   ramp={flows['s_ramp_vph']} vph")

    sim_end = int(flows["duration_s"])
    print(f"[4/5]  Writing routes & sumocfg  (sim_end = {sim_end} s) …")
    build_routes(sumo_dir, flows, sim_end)
    cfg_path = build_sumocfg(sumo_dir, sim_end)

    print("[5/5]  SUMO verification run …")
    result = run_quick_sumo(cfg_path)
    print(f"       departed={result['departed']}  arrived={result['arrived']}")

    summary = {
        "scene": "YTDJ",
        "description": "双向非对称城市快速路出入口段（北侧E→W分流出口，南侧W→E匝道汇入）",
        "total_length_m": 362,
        "north_side": {
            "direction":   "E→W",
            "lanes_max":   5,
            "main_lanes":  3,
            "ramp_lanes":  2,
            "key_points":  {"widen_start_m": 180, "diverge_start_m": 104},
        },
        "south_side": {
            "direction":   "W→E",
            "lanes_max":   5,
            "main_lanes":  3,
            "ramp_lanes":  2,
            "key_points":  {"merge_at_m": 80, "narrow_start_m": 202, "narrow_end_m": 240},
        },
        "flows_vph": flows,
        "sim_end_s": sim_end,
        "verification": result,
        "sumo_files": {
            "net":     str(sumo_dir / "ytdj.net.xml"),
            "rou":     str(sumo_dir / "ytdj.rou.xml"),
            "sumocfg": str(cfg_path),
        },
    }
    out = PROC / "ytdj_build_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary → {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
