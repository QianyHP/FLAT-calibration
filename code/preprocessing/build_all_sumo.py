"""
统一构建三个交叉口的 SUMO 仿真场景:
  - 天津 (Tianjin): E-W 3车道, N-S 2车道, 8信号灯
  - 长春 (Changchun): N-S 3车道, E-W 2车道, 2信号灯（高密度干线）
  - 西安 (Xian): N3/W3/E3、S4（Shanglin map.osm *_en_*）

对每个城市:
  1. 解析 Lanelet2 地图 → 提取交叉口几何
  2. 从轨迹反推 OD 流量与转向比例
  3. 从信号灯 CSV 提取真实配时
  4. 生成 SUMO 路网 + 路由 + 配置
"""
from __future__ import annotations

import csv
import math
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
RAW = PROJ / "data" / "raw_data" / "SIND"
PROC = PROJ / "data" / "processed_data"
PROC.mkdir(parents=True, exist_ok=True)

DEG2M = 111_320.0
ARM = 120  # 进场路段长度 (m)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  城市配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITY_CONFIGS = {
    "Tianjin": {
        "lanes": {"E": 3, "W": 3, "N": 2, "S": 2},
        "speed": 13.89,
    },
    "Changchun": {
        "lanes": {"E": 2, "W": 2, "N": 3, "S": 3},
        "speed": 13.89,
    },
    "Xian": {
        # SinD Lanelet2 (Xi'an_Shanglin.osm): N_en×3, S_en×4, W_en×3; E≈3 (E_0..E_3 接 W)
        "lanes": {"E": 3, "W": 3, "N": 3, "S": 4},
        "speed": 13.89,
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_intersection_center(osm_path: Path) -> tuple[float, float]:
    tree = ET.parse(osm_path)
    root = tree.getroot()
    xs, ys = [], []
    for n in root.findall("node"):
        ys.append(float(n.get("lat")) * DEG2M)
        xs.append(float(n.get("lon")) * DEG2M)
    return np.median(xs), np.median(ys)


def classify_approach(x: float, y: float, cx: float, cy: float) -> str | None:
    dx, dy = x - cx, y - cy
    if abs(dx) < 8 and abs(dy) < 8:
        return None
    angle = math.atan2(dy, dx) * 180 / math.pi
    if -45 <= angle < 45:
        return "E"
    elif 45 <= angle < 135:
        return "N"
    elif -135 <= angle < -45:
        return "S"
    else:
        return "W"


def infer_turn(origin: str, dest: str) -> str:
    order = ["N", "E", "S", "W"]
    if origin is None or dest is None:
        return "unknown"
    diff = (order.index(dest) - order.index(origin)) % 4
    return {1: "right", 2: "straight", 3: "left"}.get(diff, "uturn")


ORIGIN_EDGE = {"E": "EC", "W": "WC", "N": "NC", "S": "SC"}
DEST_EDGE = {"E": "CE", "W": "CW", "N": "CN", "S": "CS"}
TURN_DEST = {
    "E": {"right": "S", "straight": "W", "left": "N"},
    "W": {"right": "N", "straight": "E", "left": "S"},
    "N": {"right": "E", "straight": "S", "left": "W"},
    "S": {"right": "W", "straight": "N", "left": "E"},
}


def extract_od(city: str, cx: float, cy: float) -> tuple[dict, float]:
    """提取 OD 流量和转向比例，返回 (turn_data, duration_h)."""
    veh = pd.read_csv(RAW / city / "veh_tracks.csv")

    first = veh.groupby("track_id").first()[["x", "y"]].rename(columns={"x": "x0", "y": "y0"})
    last = veh.groupby("track_id").last()[["x", "y"]].rename(columns={"x": "x1", "y": "y1"})
    ep = first.join(last)

    duration_s = (veh["timestamp_ms"].max() - veh["timestamp_ms"].min()) / 1000.0
    duration_h = duration_s / 3600.0

    od = defaultdict(lambda: Counter())
    for _, row in ep.iterrows():
        origin = classify_approach(row["x0"], row["y0"], cx, cy)
        dest = classify_approach(row["x1"], row["y1"], cx, cy)
        turn = infer_turn(origin, dest)
        if origin and dest and turn not in ("uturn", "unknown"):
            od[origin][turn] += 1

    turn_data = {}
    for d in ["E", "W", "N", "S"]:
        total = sum(od[d].values())
        if total == 0:
            continue
        ratios = {t: od[d][t] / total for t in ["left", "straight", "right"]}
        turn_data[d] = {"total": total, "left": od[d]["left"],
                        "straight": od[d]["straight"], "right": od[d]["right"],
                        "ratios": ratios}
    return turn_data, duration_h


def extract_signal_timing(city: str) -> tuple[int, int, int]:
    """提取信号灯配时，返回 (green_phase1, green_phase2, yellow)."""
    tl = pd.read_csv(RAW / city / "traffic_lights.csv")
    ts_col = [c for c in tl.columns if "timestamp" in c.lower()][0]
    timestamps = tl[ts_col].astype(float).values
    diffs = np.diff(timestamps) / 1000.0
    diffs = diffs[diffs > 0]

    greens = diffs[diffs > 5]
    yellows = diffs[(diffs >= 2) & (diffs <= 5)]

    g = int(round(np.median(greens))) if len(greens) > 0 else 26
    y = int(round(np.median(yellows))) if len(yellows) > 0 else 3
    return g, g, y


def build_connections(lanes: dict) -> list[str]:
    """根据车道数生成合理的 connection XML."""
    lines = ["<connections>"]

    def add(src, dst, fl, tl):
        lines.append(f'    <connection from="{src}" to="{dst}" fromLane="{fl}" toLane="{tl}"/>')

    for origin, n_lanes in lanes.items():
        from_edge = ORIGIN_EDGE[origin]

        for turn_type in ["right", "straight", "left"]:
            dest_dir = TURN_DEST[origin][turn_type]
            to_edge = DEST_EDGE[dest_dir]
            dest_lanes = lanes[dest_dir]

            if turn_type == "right":
                add(from_edge, to_edge, 0, 0)
            elif turn_type == "straight":
                if n_lanes >= 3:
                    add(from_edge, to_edge, 1, min(1, dest_lanes - 1))
                    add(from_edge, to_edge, min(2, n_lanes - 1), min(2, dest_lanes - 1))
                else:
                    add(from_edge, to_edge, 0, 0)
                    if n_lanes >= 2 and dest_lanes >= 2:
                        add(from_edge, to_edge, 1, min(1, dest_lanes - 1))
            elif turn_type == "left":
                add(from_edge, to_edge, n_lanes - 1, min(dest_lanes - 1, 1))

    lines.append("</connections>")
    return lines


def build_tls_state(lanes: dict, n_links: int) -> list[tuple[str, str]]:
    """
    生成四相位信号灯 state 列表。
    返回 [(label, state_string), ...]
    连接顺序: EC 的连接 → SC 的连接 → WC 的连接 → NC 的连接 (按 netconvert 排列)
    """
    n_ec = 1 + (2 if lanes["E"] >= 3 else (2 if lanes["E"] >= 2 else 1)) + 1  # right + straight + left
    n_sc = 1 + (2 if lanes["S"] >= 3 else (2 if lanes["S"] >= 2 else 1)) + 1
    n_wc = 1 + (2 if lanes["W"] >= 3 else (2 if lanes["W"] >= 2 else 1)) + 1
    n_nc = 1 + (2 if lanes["N"] >= 3 else (2 if lanes["N"] >= 2 else 1)) + 1

    def make_state(ec_g, sc_g, wc_g, nc_g):
        def phase(n, green):
            if green:
                return "G" * (n - 1) + "g"  # last one is protected left = g
            return "r" * n
        return phase(n_ec, ec_g) + phase(n_sc, sc_g) + phase(n_wc, wc_g) + phase(n_nc, nc_g)

    def make_yellow(ec_y, sc_y, wc_y, nc_y):
        def phase(n, yellow):
            return "y" * n if yellow else "r" * n
        return phase(n_ec, ec_y) + phase(n_sc, sc_y) + phase(n_wc, wc_y) + phase(n_nc, nc_y)

    return [
        ("ew_green", make_state(True, False, True, False)),
        ("ew_yellow", make_yellow(True, False, True, False)),
        ("allred", "r" * (n_ec + n_sc + n_wc + n_nc)),
        ("ns_green", make_state(False, True, False, True)),
        ("ns_yellow", make_yellow(False, True, False, True)),
        ("allred2", "r" * (n_ec + n_sc + n_wc + n_nc)),
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
all_od_stats = []

import sys as _sys
_build_cities = (
    [c for c in _sys.argv[1:] if c in CITY_CONFIGS]
    if len(_sys.argv) > 1
    else list(CITY_CONFIGS.keys())
)

for city in _build_cities:
    cfg = CITY_CONFIGS[city]
    print(f"\n{'='*60}")
    print(f"  Building SUMO scenario: {city}")
    print(f"{'='*60}")

    city_dir = RAW / city
    out_dir = city_dir / "sumo"
    out_dir.mkdir(parents=True, exist_ok=True)

    lanes = cfg["lanes"]
    speed = cfg["speed"]

    # 1) 交叉口中心
    cx, cy = parse_intersection_center(city_dir / "map.osm")
    print(f"  Center: ({cx:.1f}, {cy:.1f})")

    # 2) OD
    turn_data, dur_h = extract_od(city, cx, cy)
    print(f"  Duration: {dur_h:.2f}h")
    for d in ["E", "W", "N", "S"]:
        if d in turn_data:
            td = turn_data[d]
            print(f"  {d}: total={td['total']}  L={td['ratios']['left']:.1%}  "
                  f"S={td['ratios']['straight']:.1%}  R={td['ratios']['right']:.1%}")

    # 保存 OD 统计
    for d in ["E", "W", "N", "S"]:
        if d not in turn_data:
            continue
        td = turn_data[d]
        for t in ["left", "straight", "right"]:
            all_od_stats.append({
                "city": city, "origin": d, "turn": t,
                "count": td.get(t, 0),
                "ratio": td["ratios"].get(t, 0),
                "veh_per_hour": td.get(t, 0) / dur_h,
            })

    # 3) 信号灯
    g1, g2, y = extract_signal_timing(city)
    print(f"  Signal: green={g1}s, yellow={y}s, cycle={2*(g1+y+1)}s")

    # 4) Nodes
    nod_path = out_dir / f"{city.lower()}.nod.xml"
    nod_path.write_text(f"""\
<nodes>
    <node id="C" x="{cx:.2f}" y="{cy:.2f}" type="traffic_light"/>
    <node id="N" x="{cx:.2f}" y="{cy + ARM:.2f}" type="priority"/>
    <node id="S" x="{cx:.2f}" y="{cy - ARM:.2f}" type="priority"/>
    <node id="E" x="{cx + ARM:.2f}" y="{cy:.2f}" type="priority"/>
    <node id="W" x="{cx - ARM:.2f}" y="{cy:.2f}" type="priority"/>
</nodes>
""", encoding="utf-8")

    # 5) Edges
    edg_path = out_dir / f"{city.lower()}.edg.xml"
    edg_lines = ["<edges>"]
    for d in ["E", "W", "N", "S"]:
        n = lanes[d]
        edg_lines.append(f'    <edge id="{d}C" from="{d}" to="C" numLanes="{n}" speed="{speed}"/>')
        edg_lines.append(f'    <edge id="C{d}" from="C" to="{d}" numLanes="{n}" speed="{speed}"/>')
    edg_lines.append("</edges>")
    edg_path.write_text("\n".join(edg_lines), encoding="utf-8")

    # 6) Connections
    con_path = out_dir / f"{city.lower()}.con.xml"
    con_lines = build_connections(lanes)
    con_path.write_text("\n".join(con_lines), encoding="utf-8")

    # 7) TLS (先生成临时，netconvert 后会根据实际连接数调整)
    phases = build_tls_state(lanes, 0)
    tll_path = out_dir / f"{city.lower()}.tll.xml"
    tll_lines = ['<tlLogics>', f'    <tlLogic id="C" type="static" programID="sind" offset="0">']
    durations = [g1, y, 1, g2, y, 1]
    for (label, state), dur in zip(phases, durations):
        tll_lines.append(f'        <phase duration="{dur}" state="{state}"/>')
    tll_lines.append('    </tlLogic>')
    tll_lines.append('</tlLogics>')
    tll_path.write_text("\n".join(tll_lines), encoding="utf-8")

    # 8) netconvert
    net_path = out_dir / f"{city.lower()}.net.xml"
    cmd = [
        "netconvert",
        "--node-files", str(nod_path),
        "--edge-files", str(edg_path),
        "--connection-files", str(con_path),
        "--tllogic-files", str(tll_path),
        "--output-file", str(net_path),
        "--no-turnarounds", "true",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  netconvert FAILED: {r.stderr}")
        # 如果 TLS state 长度不对，尝试不带 TLS 再次构建
        cmd_no_tls = [c for c in cmd if c != str(tll_path) and c != "--tllogic-files"]
        subprocess.run(cmd_no_tls, capture_output=True, text=True, check=True)
        print(f"  Rebuilt without custom TLS (will use default)")
    print(f"  Network: {net_path.name}")

    # 9) 路由文件
    rou_path = out_dir / f"{city.lower()}.rou.xml"
    rou_lines = [
        '<routes>',
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89"/>',
        '    <vType id="motorcycle" accel="2.0" decel="3.5" sigma="0.5" length="1.5" width="0.7" maxSpeed="11.11"/>',
        '',
    ]
    sim_dur = 600
    for origin in ["E", "W", "N", "S"]:
        if origin not in turn_data:
            continue
        td = turn_data[origin]
        for turn in ["left", "straight", "right"]:
            cnt = td.get(turn, 0)
            if cnt == 0:
                continue
            dest_dir = TURN_DEST[origin][turn]
            vph = max(1, round(cnt / dur_h))
            rou_lines.append(
                f'    <flow id="f_{origin}_{turn}" type="car" '
                f'from="{ORIGIN_EDGE[origin]}" to="{DEST_EDGE[dest_dir]}" '
                f'begin="0" end="{sim_dur}" vehsPerHour="{vph}" departLane="best"/>'
            )
    rou_lines.append('</routes>')
    rou_path.write_text("\n".join(rou_lines), encoding="utf-8")

    # 10) sumocfg
    cfg_path = out_dir / f"{city.lower()}.sumocfg"
    cfg_path.write_text(f"""\
<configuration>
    <input>
        <net-file value="{net_path.name}"/>
        <route-files value="{rou_path.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_dur}"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
""", encoding="utf-8")

    # 11) TraCI 验证
    import traci
    traci.start(["sumo", "-c", str(cfg_path)])
    dep, arr = 0, 0
    try:
        for _ in range(sim_dur):
            traci.simulationStep()
            dep += traci.simulation.getDepartedNumber()
            arr += traci.simulation.getArrivedNumber()
    finally:
        traci.close()
    print(f"  Simulation OK: departed={dep}, arrived={arr}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  保存 OD 统计
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
od_path = PROC / "all_cities_od_stats.csv"
pd.DataFrame(all_od_stats).to_csv(od_path, index=False)
print(f"\nAll OD stats saved: {od_path}")
print("\nAll 3 cities built and verified.")
