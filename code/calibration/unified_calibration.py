"""unified_calibration.py  ──  BF-SAC（序贯代理标定）

Behavioral Fingerprint Surrogate-Assisted Calibration (BF-SAC)
==============================================================
8 维行为指纹 + 固定 SUMO 预算下的序贯 RF 代理标定（LCB 采集）。

  - 阶段 A：``N_INIT`` 次 LHS 初始设计
  - 阶段 B：``BUDGET_SUMO - N_INIT`` 次序贯加点：重训 RF → 信赖域 + 全局候选
    → 下置信界 (LCB) 选点 → 真实 SUMO 评估并入库

用法:
  python unified_calibration.py              # 六场景
  python unified_calibration.py XAM-N6       # 单场景
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

PROJ = Path(__file__).resolve().parents[2]

# ═════════════════════════════════════════════════════════════════════
# 场景配置
# ═════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "Tianjin":   {"type": "SIND", "data": PROJ / "data/raw_data/SIND/Tianjin",
                  "cfg": "tianjin",   "sim_end": 600, "warmup": 60},
    "Changchun": {"type": "SIND", "data": PROJ / "data/raw_data/SIND/Changchun",
                  "cfg": "changchun", "sim_end": 600, "warmup": 60},
    "Xian":      {"type": "SIND", "data": PROJ / "data/raw_data/SIND/Xian",
                  "cfg": "xian",      "sim_end": 600, "warmup": 60},
    "YTDJ":      {"type": "UTE",  "data": PROJ / "data/raw_data/UTE/YTDJ",
                  "cfg": "ytdj",      "sim_end": 545, "warmup": 30},
    "RML":       {"type": "UTE",  "data": PROJ / "data/raw_data/UTE/RML",
                  "cfg": "rml",       "sim_end": 300, "warmup": 30},
    "XAM-N6":    {"type": "UTE",  "data": PROJ / "data/raw_data/UTE/XAM-N6",
                  "cfg": "xam",       "sim_end": 329, "warmup": 30},
}

# ═════════════════════════════════════════════════════════════════════
# 标定参数（6 维 Krauss 跟驰 + 4 维 LC2013 换道，TraCI 写入）
# ═════════════════════════════════════════════════════════════════════

PARAM_NAMES = [
    "accel", "decel", "sigma", "tau", "minGap", "speedFactor",
    "lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain",
]
PARAM_BOUNDS = np.array([
    [1.0, 3.5],   # accel:       最大加速度 (m/s²)
    [2.0, 6.0],   # decel:       舒适减速度 (m/s²)
    [0.1, 0.9],   # sigma:       驾驶不确定性 [0,1]
    [0.5, 2.5],   # tau:         期望车头时距 (s)
    [1.0, 4.0],   # minGap:      最小车间距 (m)
    [0.5, 1.2],   # speedFactor: 限速遵从系数
    [0.35, 2.80],  # lcStrategic:  战略/强制变道积极性（SUMO 参数）
    [0.35, 2.00],  # lcCooperative: 协作变道意愿
    [0.35, 2.80],  # lcAssertive:   变道“果断度”/间隙接受
    [0.35, 2.80],  # lcSpeedGain:   主动变道对速度增益的权重
])

# ═════════════════════════════════════════════════════════════════════
# 行为特征指纹（8 维，去冗余）
# ═════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "v_mean", "v_p15", "v_p85",
    "a_pos_mean", "a_neg_mean", "a_p05", "a_p95",
    "stop_frac",
]

# 误差加权（第一性原理：速度水平/区间 + 驱动强度 + 拥堵占有）
FEATURE_WEIGHTS = np.array([
    0.18, 0.12, 0.15,  # 速度层
    0.12, 0.12, 0.11, 0.10,  # 加减速层
    0.10,  # 拥堵层
])

# 长春：综合分瓶颈在加速度强度失配，略提高加减速维权重
FEATURE_WEIGHTS_CHANGCHUN = np.array([
    0.15, 0.10, 0.12,
    0.14, 0.14, 0.12, 0.11,
    0.12,
])
# 西安：低速交叉口，强调停车占比与 v 低分位
FEATURE_WEIGHTS_XIAN = np.array([
    0.14, 0.16, 0.08,
    0.10, 0.10, 0.09, 0.09,
    0.24,
])
# RML：交织区速度谱与跟驰间距并重
FEATURE_WEIGHTS_RML = np.array([
    0.22, 0.14, 0.17,
    0.11, 0.11, 0.08, 0.07,
    0.10,
])


def get_feature_weights(name: str) -> np.ndarray:
    if name == "Changchun":
        return FEATURE_WEIGHTS_CHANGCHUN
    if name == "Xian":
        return FEATURE_WEIGHTS_XIAN
    if name == "RML":
        return FEATURE_WEIGHTS_RML
    return FEATURE_WEIGHTS


def get_param_bounds(name: str) -> np.ndarray:
    """与 3.3.4 思路一致：对瓶颈场景收紧可写边界以引导搜索。"""
    b = PARAM_BOUNDS.copy()
    if name == "Xian":
        b[5, 1] = min(b[5, 1], 0.62)
        b[5, 0] = max(b[5, 0], 0.30)
        b[0, 1] = min(b[0, 1], 2.15)
        b[3, 0] = max(b[3, 0], 1.10)
        b[4, 0] = max(b[4, 0], 2.25)
    elif name == "Changchun":
        b[0, 1] = min(b[0, 1], 2.35)   # 抑制过冲加速度
        b[0, 0] = max(b[0, 0], 0.95)
        b[2, 0] = max(b[2, 0], 0.38)  # sigma 略抬升，轨迹更平滑
    elif name == "RML":
        b[4, 0] = max(b[4, 0], 0.85)   # 匝道交织：更小车间距下界
        b[4, 1] = min(b[4, 1], 3.35)
        b[3, 0] = max(b[3, 0], 0.42)
        b[0, 1] = min(b[0, 1], 2.90)
    return b


def lhs_samples_for_scene(n: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n=n)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return unit * (hi - lo) + lo


BUDGET_SUMO = 101   # 每场景总 SUMO 次数
N_INIT = 60         # 阶段 A：LHS 初始设计
N_DOE = N_INIT      # 对比实验 No-RF 等消融的 DoE 样本数

# 序贯 LCB 采集
LC_KAPPA = 1.96
N_CAND_LOCAL = 2500
N_CAND_GLOBAL = 2500
TRUST_FRAC = 0.18   # 信赖域：最优点附近各维 ± 该比例 × 参数跨度
MIN_DIST_REL = 0.02 # 与已评估点最小相对距离（避免重复仿真）
MIN_VALID_RF = 10


def _lhs_seed_for_scene(name: str) -> int:
    return {
        "Changchun": 203, "Tianjin": 211, "RML": 207, "Xian": 204,
    }.get(name, 42)


def _rf_n_jobs(name: str) -> int:
    return 1 if name in ("RML", "Tianjin") else -1


def _fit_rf(X: np.ndarray, Y: np.ndarray, name: str) -> RandomForestRegressor | None:
    valid = ~np.isnan(Y) & (Y < 10.0)
    if valid.sum() < MIN_VALID_RF:
        return None
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=20, random_state=42, n_jobs=_rf_n_jobs(name),
    )
    rf.fit(X[valid], Y[valid])
    return rf


def _rf_lcb(rf: RandomForestRegressor, X: np.ndarray, kappa: float) -> np.ndarray:
    """下置信界：预测均值 − κ×树间标准差（越小越优）。"""
    preds = np.stack([t.predict(X) for t in rf.estimators_], axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu - kappa * sigma


def _min_rel_dist(cand: np.ndarray, X_seen: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-6)
    d = np.linalg.norm((cand[:, None, :] - X_seen[None, :, :]) / span, axis=2)
    return d.min(axis=1)


def _propose_candidates(
    bounds: np.ndarray,
    x_best: np.ndarray,
    n_local: int,
    n_global: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    d = len(lo)
    global_c = rng.random((n_global, d)) * span + lo
    delta = TRUST_FRAC * span
    local_c = x_best + rng.uniform(-1.0, 1.0, (n_local, d)) * delta
    local_c = np.clip(local_c, lo, hi)
    return np.vstack([local_c, global_c])


def _pick_lcb_point(
    rf: RandomForestRegressor,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    cand = _propose_candidates(bounds, x_best, N_CAND_LOCAL, N_CAND_GLOBAL, rng)
    lcb = _rf_lcb(rf, cand, kappa)
    md = _min_rel_dist(cand, X_seen, bounds)
    lcb[md < MIN_DIST_REL] = np.inf
    idx = int(np.argmin(lcb))
    return cand[idx]


def _eval_params(
    name: str,
    params: dict,
    real_feat: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    try:
        sf = _run_sumo_traci(name, params)
        err = feature_error(real_feat, sf, weights)
    except Exception as exc:
        print(f"       [!] SUMO failed: {exc}", flush=True)
        sf = np.full(len(FEATURE_NAMES), np.nan)
        err = 10.0
    return float(err), sf


def compute_features(speeds: np.ndarray, accels: np.ndarray) -> np.ndarray:
    """从速度(m/s)和纵向加速度(m/s²)序列计算 8 维行为指纹。"""
    if len(speeds) < 20 or len(accels) < 20:
        return np.full(len(FEATURE_NAMES), np.nan)

    v = speeds
    a = accels

    v_mean = float(np.mean(v))
    v_p15  = float(np.percentile(v, 15))
    v_p85  = float(np.percentile(v, 85))

    a_pos = a[a > 0.05]
    a_neg = a[a < -0.05]
    a_pos_mean = float(np.mean(a_pos))  if len(a_pos) > 5 else 0.0
    a_neg_mean = float(np.mean(np.abs(a_neg))) if len(a_neg) > 5 else 0.0
    a_p05  = float(np.percentile(a, 5))
    a_p95  = float(np.percentile(a, 95))

    stop_frac = float(np.mean(v < 1.0))

    return np.array([
        v_mean, v_p15, v_p85,
        a_pos_mean, a_neg_mean, a_p05, a_p95,
        stop_frac,
    ])


# ═════════════════════════════════════════════════════════════════════
# 真实特征提取
# ═════════════════════════════════════════════════════════════════════

def extract_real_features(name: str) -> np.ndarray:
    sc = SCENARIOS[name]
    if sc["type"] == "SIND":
        df = pd.read_csv(sc["data"] / "veh_tracks.csv")
        vx, vy = df["vx"].values, df["vy"].values
        ax_arr, ay_arr = df["ax"].values, df["ay"].values
        speed = np.sqrt(vx**2 + vy**2)
        # 纵向加速度 = 速度方向上的分量
        sp_safe = np.maximum(speed, 0.01)
        a_lon = (vx * ax_arr + vy * ay_arr) / sp_safe
        a_lon[speed < 0.3] = 0.0
        return compute_features(speed, a_lon)
    else:
        df = pd.read_csv(sc["data"] / "frenet.csv")
        cols = {c.lower(): c for c in df.columns}
        speed_col = cols.get("speed(km/h)", cols.get("velocity(km/h)"))
        acc_col = cols["acceleration(m/s^2)"]
        speed = df[speed_col].values / 3.6
        accel = df[acc_col].values
        return compute_features(speed, accel)


# ═════════════════════════════════════════════════════════════════════
# SUMO 仿真（TraCI）
# ═════════════════════════════════════════════════════════════════════

def _run_sumo_traci(name: str, params: dict) -> np.ndarray:
    """通过 TraCI 运行 SUMO，动态设置参数，收集速度/加速度，返回特征指纹。"""
    import traci

    sc = SCENARIOS[name]
    cfg = str(sc["data"] / "sumo" / f"{sc['cfg']}.sumocfg")

    traci.start(["sumo", "-c", cfg,
                 "--no-step-log", "true",
                 "--no-warnings", "true",
                 "--seed", str(np.random.randint(1, 99999))])
    try:
        for vtype in traci.vehicletype.getIDList():
            traci.vehicletype.setAccel(vtype, float(params["accel"]))
            traci.vehicletype.setDecel(vtype, float(params["decel"]))
            traci.vehicletype.setImperfection(vtype, float(params["sigma"]))
            traci.vehicletype.setTau(vtype, float(params["tau"]))
            traci.vehicletype.setMinGap(vtype, float(params["minGap"]))
            traci.vehicletype.setSpeedFactor(vtype, float(params["speedFactor"]))
            for lc_key in ("lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain"):
                traci.vehicletype.setParameter(
                    vtype, lc_key, str(float(params[lc_key]))
                )

        speeds, accels = [], []
        warmup = sc["warmup"]
        sim_end = sc["sim_end"]
        step = 0

        while traci.simulation.getTime() < sim_end:
            traci.simulationStep()
            step += 1
            if traci.simulation.getTime() < warmup or step % 3 != 0:
                continue
            for vid in traci.vehicle.getIDList():
                speeds.append(traci.vehicle.getSpeed(vid))
                accels.append(traci.vehicle.getAcceleration(vid))
    finally:
        traci.close()

    return compute_features(np.array(speeds, dtype=float),
                            np.array(accels, dtype=float))


# ═════════════════════════════════════════════════════════════════════
# 误差计算
# ═════════════════════════════════════════════════════════════════════

def feature_error(
    real_feat: np.ndarray,
    sim_feat: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """加权平均相对误差。"""
    w = FEATURE_WEIGHTS if weights is None else weights
    if np.any(np.isnan(sim_feat)):
        return 10.0
    eps = 0.01
    rel_err = np.abs(sim_feat - real_feat) / np.maximum(np.abs(real_feat), eps)
    return float(np.dot(w, np.minimum(rel_err, 3.0)))


# ═════════════════════════════════════════════════════════════════════
# 单场景标定
# ═════════════════════════════════════════════════════════════════════

def calibrate_one(name: str) -> dict:
    """N_INIT 次 LHS + 序贯 LCB 加点，总预算 BUDGET_SUMO。"""
    print(f"\n{'=' * 64}")
    print(f"  BF-SAC (sequential LCB): {name}")
    print(f"  Budget={BUDGET_SUMO}  (init LHS={N_INIT}, sequential={BUDGET_SUMO - N_INIT})")
    print(f"{'=' * 64}")

    weights = get_feature_weights(name)
    bounds_arr = get_param_bounds(name)
    lhs_seed = _lhs_seed_for_scene(name)
    n_seq = BUDGET_SUMO - N_INIT
    rng = np.random.default_rng(lhs_seed + 17)

    print("[1/3] Extracting real behavioral fingerprint …")
    real_feat = extract_real_features(name)
    for fn, val in zip(FEATURE_NAMES, real_feat):
        print(f"       {fn:<14s} = {val:.4f}")

    print(f"[2/3] Phase A: {N_INIT} LHS initial runs …")
    X = np.zeros((BUDGET_SUMO, len(PARAM_NAMES)))
    X[:N_INIT] = lhs_samples_for_scene(N_INIT, bounds_arr, lhs_seed)
    Y = np.full(BUDGET_SUMO, np.nan)
    sim_feats = np.zeros((BUDGET_SUMO, len(FEATURE_NAMES)))
    history: list[dict] = []
    t0 = time.time()
    running_best = float("inf")

    for i in range(N_INIT):
        params = dict(zip(PARAM_NAMES, X[i]))
        err, sf = _eval_params(name, params, real_feat, weights)
        Y[i] = err
        sim_feats[i] = sf
        running_best = min(running_best, err)
        history.append({
            "sim_index": i + 1,
            "phase": "init",
            "error": round(err, 6),
            "best_so_far": round(running_best, 6),
            "params": {k: round(float(v), 4) for k, v in params.items()},
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_INIT - i - 1)
            print(f"       {i+1:>3d}/{N_INIT}  best={running_best:.4f}"
                  f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    doe_best_err = float(np.nanmin(Y[:N_INIT]))
    n_seen = N_INIT

    print(f"[3/3] Phase B: {n_seq} sequential LCB acquisitions …", flush=True)
    rf_rmse: float | None = None
    rf_r2: float | None = None
    seq_errors: list[float] = []

    for j in range(n_seq):
        rf = _fit_rf(X[:n_seen], Y[:n_seen], name)
        if rf is None:
            print("       [!] Too few valid runs for RF; stopping sequential phase", flush=True)
            break

        pred_tr = rf.predict(X[:n_seen])
        valid = ~np.isnan(Y[:n_seen]) & (Y[:n_seen] < 10.0)
        rf_rmse = float(np.sqrt(mean_squared_error(Y[:n_seen][valid], pred_tr[valid])))
        rf_r2 = float(rf.score(X[:n_seen][valid], Y[:n_seen][valid]))

        best_idx = int(np.nanargmin(Y[:n_seen]))
        x_best = X[best_idx].copy()
        x_new = _pick_lcb_point(rf, bounds_arr, X[:n_seen], x_best, LC_KAPPA, rng)
        params = dict(zip(PARAM_NAMES, x_new))

        err, sf = _eval_params(name, params, real_feat, weights)
        X[n_seen] = x_new
        Y[n_seen] = err
        sim_feats[n_seen] = sf
        n_seen += 1
        running_best = min(running_best, err)
        seq_errors.append(err)

        history.append({
            "sim_index": n_seen,
            "phase": "sequential",
            "lcb_round": j + 1,
            "error": round(err, 6),
            "best_so_far": round(running_best, 6),
            "rf_r2": round(rf_r2, 4),
            "params": {k: round(float(v), 4) for k, v in params.items()},
        })
        if (j + 1) % 5 == 0 or j == n_seq - 1:
            print(f"       seq {j+1:>2d}/{n_seq}  err={err:.4f}  best={running_best:.4f}"
                  f"  RF R2={rf_r2:.3f}", flush=True)

    best_idx = int(np.nanargmin(Y[:n_seen]))
    best_params = dict(zip(PARAM_NAMES, X[best_idx]))
    best_feat = sim_feats[best_idx]
    best_err = float(Y[best_idx])
    init_best_idx = int(np.nanargmin(Y[:N_INIT]))
    doe_best_err = float(Y[init_best_idx])

    seq_best = float(np.min(seq_errors)) if seq_errors else None
    proxy_beats_doe = bool(best_idx >= N_INIT and best_err <= doe_best_err - 1e-6)

    print(f"\n  Calibrated params:")
    for k, v in best_params.items():
        print(f"       {k:<14s} = {v:.4f}")
    print(f"  Feature error: {best_err:.4f}  (init best={doe_best_err:.4f})")
    print(f"  Total SUMO runs: {n_seen}  |  sequential improved: {proxy_beats_doe}")

    comparison = {}
    for i, fn in enumerate(FEATURE_NAMES):
        rv = float(real_feat[i])
        sv = float(best_feat[i]) if not np.isnan(best_feat[i]) else 0.0
        comparison[fn] = {
            "real": round(rv, 4),
            "sim_calibrated": round(sv, 4),
            "rel_error": round(abs(sv - rv) / max(abs(rv), 0.01), 4),
        }

    return {
        "scenario": name,
        "calibration_mode": "sequential_lcb",
        "calibrated_params": {k: round(v, 4) for k, v in best_params.items()},
        "feature_error": round(best_err, 4),
        "doe_best_error": round(doe_best_err, 4),
        "sequential_best_error": round(seq_best, 4) if seq_best is not None else None,
        "surrogate_verified_error": round(seq_best, 4) if seq_best is not None else None,
        "proxy_beats_doe": proxy_beats_doe,
        "n_sumo_runs": n_seen,
        "rf_r2": round(rf_r2, 4) if rf_r2 is not None else None,
        "rf_rmse": round(rf_rmse, 4) if rf_rmse is not None else None,
        "feature_comparison": comparison,
        "doe_history": history,
    }

# ═════════════════════════════════════════════════════════════════════
# 应用标定结果到 .rou.xml
# ═════════════════════════════════════════════════════════════════════

def apply_calibration(name: str, params: dict) -> None:
    """将标定参数写回场景的 .rou.xml 文件中的所有 vType。"""
    import xml.etree.ElementTree as ET
    sc = SCENARIOS[name]
    sumo_dir = sc["data"] / "sumo"
    rou_files = list(sumo_dir.glob("*.rou.xml"))
    for rou in rou_files:
        tree = ET.parse(rou)
        for vtype in tree.findall(".//vType"):
            vtype.set("accel", f"{params['accel']:.2f}")
            vtype.set("decel", f"{params['decel']:.2f}")
            vtype.set("sigma", f"{params['sigma']:.2f}")
            vtype.set("tau",   f"{params['tau']:.2f}")
            vtype.set("minGap", f"{params['minGap']:.2f}")
            vtype.set("speedFactor", f"{params['speedFactor']:.2f}")
            for lc_key in ("lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain"):
                vtype.set(lc_key, f"{params[lc_key]:.2f}")
        tree.write(str(rou), encoding="utf-8", xml_declaration=True)
        print(f"  Updated: {rou.name}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    out_dir = PROJ / "data" / "processed_data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = sys.argv[1:] if len(sys.argv) > 1 else list(SCENARIOS.keys())

    all_results = {}
    for name in targets:
        if name not in SCENARIOS:
            print(f"[!] Unknown scenario '{name}', skipping.")
            continue
        result = calibrate_one(name)
        all_results[name] = result

        per_file = out_dir / f"{name}_calibration.json"
        slim = {k: v for k, v in result.items() if k != "doe_history"}
        per_file.write_text(json.dumps(slim, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  Saved: {per_file}")

        # Full version with convergence history
        hist_file = out_dir / f"{name}_calibration_with_history.json"
        hist_file.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"  Saved (with history): {hist_file}")

        apply_calibration(name, result["calibrated_params"])

    if not all_results:
        return

    all_path = out_dir / "calibration_all.json"
    merged: dict = {}
    if all_path.exists():
        try:
            merged = json.loads(all_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(all_results)
    all_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Full results → {all_path}")

    summary_rows = []
    for name in SCENARIOS:
        pf = out_dir / f"{name}_calibration.json"
        if not pf.exists():
            continue
        r = json.loads(pf.read_text(encoding="utf-8"))
        row = {"scenario": name, "feature_error": r["feature_error"]}
        row.update(r["calibrated_params"])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nCalibration summary → {summary_path}")


if __name__ == "__main__":
    main()
